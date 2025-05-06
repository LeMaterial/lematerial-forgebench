from concurrent.futures import ProcessPoolExecutor
from functools import partial

from material_hasher.dataset_store import DatasetStore
from material_hasher.hasher import HASHERS
from material_hasher.hasher.base import HasherBase
from material_hasher.similarity import SIMILARITY_MATCHERS
from pymatgen.core import Structure
from tqdm import tqdm

from lematerial_forgebench.data.references.huggingface import HFDataset
from lematerial_forgebench.utils.logging import logger


# Hasher that extracts the BAWL fingerprint from the HF dataset
# since they are already computed
class ExtractFingerprintHasher(HasherBase):
    def __init__(self):
        pass

    def get_material_hash(self, structure: Structure) -> str:
        return structure.fingerprint


HASHERS["extract_fingerprint"] = ExtractFingerprintHasher


def process_batch(
    batch_num, total_batches, indices, dataset_class, dataset_store_class, hasher_class
):
    """Process a batch of indices in a separate process.

    This function reinitializes necessary objects in each process to avoid
    pickling issues.
    """
    dataset = dataset_class()
    dataset_store = dataset_store_class(hasher_class)

    # Select only the needed indices
    dataset = dataset.select(indices)

    embeddings = []
    desc = f"Batch {batch_num}/{total_batches}"
    for structure in tqdm(
        dataset, desc=desc, leave=False, dynamic_ncols=True, position=batch_num
    ):
        try:
            embedding = dataset_store._get_structure_embedding(
                structure, dataset_store.equivalence_checker
            )
            embeddings.append(embedding)
        except Exception as e:
            logger.warning(f"Error processing structure: {e}")
            continue

    return embeddings


def fit_store(
    dataset: HFDataset, dataset_store: DatasetStore, store_path: str, n_jobs: int = 1
):
    """Fit a DatasetStore on a HuggingFace dataset.

    Parameters
    ----------
    dataset : HFDataset
        The HuggingFace dataset to fit the store on.
    dataset_store : DatasetStore
        The dataset store to fit.
    store_path : str
        The path to save the store to.
    n_jobs : int, optional
        The number of jobs to use for fitting the store.
    """
    # Calculate batch indices
    total_size = len(dataset)
    batch_size = total_size // n_jobs
    indices = [list(range(i * batch_size, (i + 1) * batch_size)) for i in range(n_jobs)]

    # Handle remainder
    if total_size % n_jobs != 0:
        indices[-1].extend(range(n_jobs * batch_size, total_size))

    print(f"Processing {total_size} structures in {n_jobs} batches")

    if n_jobs <= 1:
        # Process directly in main process if n_jobs <= 1
        embeddings = process_batch(
            1,
            1,  # batch_num, total_batches
            list(range(total_size)),
            dataset.__class__,
            dataset_store.__class__,
            dataset_store.equivalence_checker.__class__,
        )
        dataset_store.store_embeddings(embeddings)
    else:
        process_fn = partial(
            process_batch,
            total_batches=n_jobs,
            dataset_class=dataset.__class__,
            dataset_store_class=dataset_store.__class__,
            hasher_class=dataset_store.equivalence_checker.__class__,
        )

        with ProcessPoolExecutor(
            max_workers=n_jobs,
        ) as executor:
            futures = []
            for batch_num, idx_batch in enumerate(indices, 1):
                future = executor.submit(process_fn, batch_num, indices=idx_batch)
                futures.append(future)

            # Process results as they complete
            total_processed = 0
            with tqdm(total=total_size, desc="Total progress") as pbar:
                for future in futures:
                    try:
                        result = future.result()
                        dataset_store.store_embeddings(result)
                        total_processed += len(result)
                        pbar.update(len(result))
                    except Exception as e:
                        logger.error(f"Error processing batch: {e}")
                        raise

    logger.info(f"\nSaving results to {store_path}")
    if dataset_store.equivalence_checker_class == ExtractFingerprintHasher:
        dataset_store.equivalence_checker_class = HASHERS["BAWL-Legacy"]
    dataset_store.save(store_path)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["lemat_bulk"], default="lemat_bulk")
    parser.add_argument(
        "--algorithm",
        choices=list(HASHERS.keys()) + list(SIMILARITY_MATCHERS.keys()),
        required=True,
    )
    parser.add_argument("--n_jobs", type=int, default=10)
    parser.add_argument("--subsample", type=int, default=None)
    args = parser.parse_args()

    hasher_class = (
        HASHERS[args.algorithm]
        if args.algorithm in HASHERS
        else SIMILARITY_MATCHERS[args.algorithm]
    )
    if args.dataset == "lemat_bulk":
        dataset = HFDataset(subsample=args.subsample)
    else:
        raise ValueError(f"Dataset {args.dataset} not supported")
    store = DatasetStore(hasher_class)

    fit_store(
        dataset, store, f"store_{args.dataset}_{args.algorithm}.npy", n_jobs=args.n_jobs
    )


if __name__ == "__main__":
    main()
