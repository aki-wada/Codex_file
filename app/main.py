import argparse
import logging
from pathlib import Path
from typing import List, Optional, Tuple

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
    if not input_path.exists():
        raise FileNotFoundError(f"CSV not found: {input_path}")
    if input_path.is_dir():
        raise IsADirectoryError(f"Path is a directory, expected CSV file: {input_path}")
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


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    total = len(df)
    counts = df.isna().sum()
    pct = counts / total * 100 if total else 0
    summary = pd.DataFrame({"missing_count": counts, "missing_pct": pct})
    return summary


def outlier_summary(
    df: pd.DataFrame, max_rows: int = 100
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Detect outliers via IQR fences. Returns per-column summary and sample rows."""
    numeric_cols = df.select_dtypes(include="number").columns
    summary_records = []
    row_records = []

    for col in numeric_cols:
        series = df[col].dropna()
        if series.empty:
            continue
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        mask = (df[col] < lower) | (df[col] > upper)
        count = mask.sum()
        pct = count / len(df) * 100 if len(df) else 0
        summary_records.append(
            {
                "column": col,
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "lower_fence": lower,
                "upper_fence": upper,
                "outlier_count": count,
                "outlier_pct": pct,
            }
        )
        if count:
            limited = mask[mask].head(max_rows)
            for idx in limited.index:
                row_records.append(
                    {
                        "index": idx,
                        "column": col,
                        "value": df.at[idx, col],
                        "lower_fence": lower,
                        "upper_fence": upper,
                    }
                )

    summary_df = pd.DataFrame(summary_records)
    rows_df = pd.DataFrame(row_records)
    return summary_df, rows_df


def cohen_d(series_a: pd.Series, series_b: pd.Series) -> Optional[float]:
    """Compute Cohen's d for two samples. Returns None if insufficient data."""
    a = series_a.dropna()
    b = series_b.dropna()
    if len(a) < 2 or len(b) < 2:
        return None
    diff = a.mean() - b.mean()
    pooled_sd = (((len(a) - 1) * a.std(ddof=1) ** 2 + (len(b) - 1) * b.std(ddof=1) ** 2) / (len(a) + len(b) - 2)) ** 0.5
    if pooled_sd == 0:
        return None
    return diff / pooled_sd


def odds_ratio_from_table(table: pd.Series) -> Optional[float]:
    """Compute simple odds ratio for binary outcome cross-tab (2x2)."""
    if table.shape != (2, 2):
        return None
    (a, b), (c, d) = table.values
    # Avoid division by zero
    if b == 0 or c == 0 or d == 0:
        return None
    return (a * d) / (b * c)


def group_summaries(df: pd.DataFrame, group_col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return numeric and categorical summaries by group."""
    if group_col not in df.columns:
        raise KeyError(f"Group column not found: {group_col}")
    grouped = df.groupby(group_col)
    num_cols = df.select_dtypes(include="number").columns
    cat_cols = df.select_dtypes(exclude="number").columns

    num_summary = grouped[num_cols].describe().unstack(level=0).T if len(num_cols) else pd.DataFrame()
    cat_summary = (
        grouped[cat_cols].agg(["count", "nunique"]).unstack(level=0).T if len(cat_cols) else pd.DataFrame()
    )
    return num_summary, cat_summary


def effect_sizes(df: pd.DataFrame, group_col: str, outcome_cols: List[str]) -> pd.DataFrame:
    """Compute simple effect sizes for two-group comparisons."""
    groups = df[group_col].dropna().unique()
    if len(groups) != 2:
        logger.info("Effect sizes limited to 2 groups; found %s groups, skipping.", len(groups))
        return pd.DataFrame()
    g1, g2 = groups
    df1 = df[df[group_col] == g1]
    df2 = df[df[group_col] == g2]

    records = []
    for col in outcome_cols:
        if col not in df.columns:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            d = cohen_d(df1[col], df2[col])
            records.append({"variable": col, "type": "numeric", "cohens_d": d, "group_a": g1, "group_b": g2})
        else:
            tab = pd.crosstab(df[group_col], df[col])
            if tab.shape == (2, 2):
                or_val = odds_ratio_from_table(tab)
                records.append(
                    {
                        "variable": col,
                        "type": "categorical",
                        "odds_ratio": or_val,
                        "group_a": g1,
                        "group_b": g2,
                    }
                )
    return pd.DataFrame(records)


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
    parser.add_argument(
        "--max-outlier-rows",
        type=int,
        default=100,
        help="Maximum number of outlier rows to export across columns.",
    )
    parser.add_argument(
        "--group-col",
        type=str,
        help="Column name for group comparison (expects two groups for effect sizes).",
    )
    parser.add_argument(
        "--effect-cols",
        type=str,
        nargs="*",
        help="Outcome columns to compute effect sizes (numeric: Cohen's d, categorical binary: odds ratio).",
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

    miss_df = missing_summary(df)
    miss_path = out_dir / "missing_summary.csv"
    miss_df.to_csv(miss_path)
    logger.info("Saved missing summary: %s", miss_path)

    outlier_df, outlier_rows = outlier_summary(df, max_rows=args.max_outlier_rows)
    outlier_sum_path = out_dir / "outlier_summary.csv"
    outlier_rows_path = out_dir / "outlier_rows.csv"
    if not outlier_df.empty:
        outlier_df.to_csv(outlier_sum_path, index=False)
        logger.info("Saved outlier summary: %s", outlier_sum_path)
    else:
        logger.info("No numeric columns for outlier detection.")
    if not outlier_rows.empty:
        outlier_rows.to_csv(outlier_rows_path, index=False)
        logger.info("Saved outlier rows (sample): %s", outlier_rows_path)
    else:
        logger.info("No outlier rows detected (or none within sample limit).")

    if args.group_col:
        try:
            grp_num, grp_cat = group_summaries(df, args.group_col)
            grp_num_path = out_dir / "group_numeric_summary.csv"
            grp_cat_path = out_dir / "group_categorical_summary.csv"
            if not grp_num.empty:
                grp_num.to_csv(grp_num_path)
                logger.info("Saved group numeric summary: %s", grp_num_path)
            if not grp_cat.empty:
                grp_cat.to_csv(grp_cat_path)
                logger.info("Saved group categorical summary: %s", grp_cat_path)
            if args.effect_cols:
                eff_df = effect_sizes(df, args.group_col, args.effect_cols)
                eff_path = out_dir / "effect_sizes.csv"
                if not eff_df.empty:
                    eff_df.to_csv(eff_path, index=False)
                    logger.info("Saved effect sizes: %s", eff_path)
                else:
                    logger.info("No effect sizes computed (check group count or columns).")
        except KeyError as e:
            logger.error(str(e))

    logger.info("Done. Outputs written to %s", out_dir.resolve())


if __name__ == "__main__":
    main()
