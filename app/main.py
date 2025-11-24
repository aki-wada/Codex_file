import argparse
import logging
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def read_csv(input_path: Path) -> pd.DataFrame:
    logger.info("Reading data from %s", input_path)
    df = pd.read_csv(input_path)
    if df.empty:
        raise ValueError("Loaded DataFrame is empty")
    return df


def summarize(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return numeric and categorical summaries."""
    numeric_cols = df.select_dtypes(include="number").columns
    categorical_cols = df.select_dtypes(exclude="number").columns

    num_summary = df[numeric_cols].describe().T if len(numeric_cols) else pd.DataFrame()
    cat_summary = (
        df[categorical_cols]
        .describe(include=["object", "category"])
        .T
        if len(categorical_cols)
        else pd.DataFrame()
    )

    return num_summary, cat_summary


def plot_numeric(df: pd.DataFrame, output_dir: Path, max_plots: int = 6) -> List[Path]:
    """Create histograms and boxplots for numeric columns (capped)."""
    numeric_cols = list(df.select_dtypes(include="number").columns)
    numeric_cols = numeric_cols[:max_plots]  # cap to avoid too many plots
    output_paths: List[Path] = []

    for col in numeric_cols:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        sns.histplot(df[col].dropna(), kde=True, ax=axes[0])
        axes[0].set_title(f"{col} - Histogram")

        sns.boxplot(x=df[col], ax=axes[1])
        axes[1].set_title(f"{col} - Boxplot")

        fig.tight_layout()
        plot_path = output_dir / f"{col}_plot.png"
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        output_paths.append(plot_path)
        logger.info("Saved plot: %s", plot_path)

    if not numeric_cols:
        logger.info("No numeric columns found; skipping plots.")

    return output_paths


def preview(df: pd.DataFrame, rows: int = 5) -> pd.DataFrame:
    """Return head preview."""
    return df.head(rows)


def ensure_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def main():
    parser = argparse.ArgumentParser(
        description="Data import + descriptive statistics + simple visualization."
    )
    parser.add_argument("csv_path", type=Path, help="Path to input CSV file.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory to write summaries and plots.",
    )
    parser.add_argument(
        "--max-plots",
        type=int,
        default=6,
        help="Maximum number of numeric columns to plot.",
    )
    args = parser.parse_args()

    out_dir = ensure_output_dir(args.out_dir)
    df = read_csv(args.csv_path)

    logger.info("Preview:\n%s", preview(df))

    num_summary, cat_summary = summarize(df)

    num_path = out_dir / "summary_numeric.csv"
    cat_path = out_dir / "summary_categorical.csv"
    if not num_summary.empty:
        num_summary.to_csv(num_path)
        logger.info("Saved numeric summary: %s", num_path)
    else:
        logger.info("No numeric columns to summarize.")

    if not cat_summary.empty:
        cat_summary.to_csv(cat_path)
        logger.info("Saved categorical summary: %s", cat_path)
    else:
        logger.info("No categorical columns to summarize.")

    plot_numeric(df, out_dir, max_plots=args.max_plots)

    logger.info("Done. Outputs written to %s", out_dir.resolve())


if __name__ == "__main__":
    main()
