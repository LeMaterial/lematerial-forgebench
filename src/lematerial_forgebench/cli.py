"""Command line interface for running benchmarks.

This module provides a CLI for running material generation benchmarks
using configuration files.
"""

import os
from pathlib import Path

import click
import yaml

from lematerial_forgebench.benchmarks.example import ExampleBenchmark
from lematerial_forgebench.data.structure import format_structures
from lematerial_forgebench.utils.logging import logger

CONFIGS_DIR = Path(__file__).parent.parent / "config"


def load_benchmark_config(config_name: str) -> dict:
    """Load benchmark configuration from YAML file.

    Parameters
    ----------
    config_name : str
        Name of the config file (with or without .yaml extension)
        Will look for the config in the standard configs directory

    Returns
    -------
    dict
        Benchmark configuration
    """
    # Ensure configs directory exists
    if not CONFIGS_DIR.exists():
        CONFIGS_DIR.mkdir(parents=True)

    # Add .yaml extension if not present
    if not config_name.endswith(".yaml"):
        config_name = f"{config_name}.yaml"

    config_path = CONFIGS_DIR / config_name

    # If config doesn't exist but it's the example config, create it
    if not config_path.exists() and config_name == "example.yaml":
        example_config = {
            "type": "example",
            "quality_weight": 0.4,
            "diversity_weight": 0.4,
            "novelty_weight": 0.2,
        }
        with open(config_path, "w") as f:
            yaml.dump(example_config, f, default_flow_style=False)

    if not config_path.exists():
        raise click.ClickException(
            f"Config '{config_name}' not found in {CONFIGS_DIR}. "
            "Available configs: "
            + ", ".join(f.stem for f in CONFIGS_DIR.glob("*.yaml"))
        )

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def save_results(results: dict, output_path: str):
    """Save benchmark results to file.

    Parameters
    ----------
    results : dict
        Benchmark results
    output_path : str
        Path to save results
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save results in YAML format
    with open(output_path, "w") as f:
        yaml.dump(results, f, default_flow_style=False)


@click.command()
@click.argument("input", type=click.Path(exists=True))
@click.argument("config_name", type=str)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Path to save results",
    default="benchmark_results.yaml",
)
def main(input: str, config_name: str, output: str):
    """Run a benchmark on structures using the specified configuration.

    STRUCTURES_CSV: Path to CSV file containing structures to evaluate
    CONFIG_NAME: Name of the benchmark configuration (e.g. 'example' for example.yaml) or path to a config file
    """
    try:
        # Load structures
        logger.info(f"Loading structures from {input}")
        structures = format_structures(input)
        if not structures:
            logger.error("No valid structures loaded")
            return

        # Benchmark configuration
        logger.info(f"Loading benchmark configuration '{config_name}'")
        config = load_benchmark_config(config_name)

        # Initialization
        benchmark_type = config.get("type", "example")
        if benchmark_type == "example":
            benchmark = ExampleBenchmark(
                quality_weight=config.get("quality_weight", 0.4),
                diversity_weight=config.get("diversity_weight", 0.4),
                novelty_weight=config.get("novelty_weight", 0.2),
            )
        else:
            raise ValueError(f"Unknown benchmark type: {benchmark_type}")

        # Run benchmark
        logger.info("Running benchmark evaluation")
        results = benchmark.evaluate(structures=structures)

        # Save results
        logger.info(f"Saving results to {output}")
        save_results(results.__dict__, output)

        click.echo("\nBenchmark Results Summary:")
        for score_name, score in results.final_scores.items():
            click.echo(f"{score_name}: {score:.3f}")

    except Exception as e:
        logger.error("Benchmark execution failed", exc_info=True)
        raise click.ClickException(str(e))


if __name__ == "__main__":
    main()
