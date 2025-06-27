from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from tqdm import tqdm


def scale_individual_row(row):
    # print(row)
    a = row["a"]
    b = row["b"]
    c = row["c"]

    lattice_params = [a, b, c]
    if row.CrystalSystem == "orthorhombic":
        lattice_params.sort()
        lattice_params.append(row["LeMatID"])
        return lattice_params
    else:
        lattice_params.append(row["LeMatID"])
        return lattice_params


def scale_df_parallel(dataset, chunk_size=1000, num_workers=None):
    if num_workers is None:
        num_workers = cpu_count()

    def chunks():
        for i in np.arange(0, len(dataset), chunk_size):
            chunk = dataset.iloc[i : i + chunk_size]
            for j in range(0, len(chunk)):
                row = chunk.iloc[j]
                yield row

    total_items = len(dataset)
    with Pool(processes=num_workers) as pool:
        # Using imap instead of map for memory efficiency
        with tqdm(total=total_items, desc="Processing structures") as pbar:
            for result in pool.imap(scale_individual_row, chunks(), chunksize=100):
                pbar.update(1)
                yield result


def test_row_fidelity(ids):
    try:
        assert ids[0] == ids[1]
    except AssertionError:
        print(ids[0], ids[1], ids[2])
        if np.isnan(ids[0]) and np.isnan(ids[1]):
            pass
        else:
            raise AssertionError
    return True


def test_dataset_fidelity_parallel(
    dataset, dataset_scaled, chunk_size=1000, num_workers=None
):
    if num_workers is None:
        num_workers = cpu_count()

    def chunks():
        for i in np.arange(0, len(dataset), chunk_size):
            chunk = dataset.iloc[i : i + chunk_size]
            chunk_scaled = dataset_scaled.iloc[i : i + chunk_size]
            for j in range(0, len(chunk)):
                row = chunk.iloc[j]
                row_scaled = chunk_scaled.iloc[j]
                yield [row.LeMatID, row_scaled.LeMatID_reference, i]

    total_items = len(dataset)
    with Pool(processes=num_workers) as pool:
        # Using imap instead of map for memory efficiency
        with tqdm(total=total_items, desc="Processing structures") as pbar:
            for result in pool.imap(test_row_fidelity, chunks(), chunksize=100):
                pbar.update(1)
                yield result


if __name__ == "__main__":
    # Process and handle results as they come

    run_tests = False

    df = pd.read_csv("lematbulk.csv")
    print(f"Processing {len(df)} structures using {cpu_count()} workers...")

    try:
        results_df = pd.read_csv("scaled_lattice.csv")
    except FileNotFoundError:
        results = []
        for result in scale_df_parallel(df):
            # print(result)
            results.append(result)

        results_df = pd.DataFrame(
            results, columns=["a_scaled", "b_scaled", "c_scaled", "LeMatID_reference"]
        )
        results_df.to_csv("scaled_lattice.csv", index=False)

    if run_tests:
        results = []
        for result in test_dataset_fidelity_parallel(df, results_df):
            # print(result)
            results.append(result)
        if sum(results) == len(results):
            print("All tests passed")

    df["a_scaled"] = results_df.a_scaled
    df["b_scaled"] = results_df.b_scaled
    df["c_scaled"] = results_df.c_scaled
    df["LeMatID_reference"] = results_df.LeMatID_reference

    df.to_csv("lematbulk_scaled.csv", index=False)
