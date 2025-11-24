import argparse
import base64
import logging
import os
from itertools import combinations
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd


logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def ensure_matplotlib_config(base_dir: Path) -> None:
    """Ensure Matplotlib uses a writable config dir to avoid cache warnings."""
    target = base_dir / ".mplconfig"
    target.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(target))


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


def effect_sizes(
    df: pd.DataFrame, group_col: str, outcome_cols: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute effect sizes for multi-arm trials.

    - Pairwise effect sizes for all group pairs:
        - numeric: Cohen's d
        - categorical binary: odds ratio (2x2)
    - One-way ANOVA for numeric outcomes (>=2 groups)
    - Tukey HSD post-hoc for numeric outcomes (>=3 groups)
    """
    groups = [g for g in df[group_col].dropna().unique()]
    if len(groups) < 2:
        logger.info("Need at least 2 groups for effect sizes; found %s.", len(groups))
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    pairwise_records = []
    anova_records = []
    tukey_records = []

    for col in outcome_cols:
        if col not in df.columns:
            continue
        col_data = df[[group_col, col]].dropna()
        # Numeric outcomes
        if pd.api.types.is_numeric_dtype(col_data[col]):
            # Pairwise Cohen's d
            for g1, g2 in combinations(groups, 2):
                d = cohen_d(
                    col_data[col_data[group_col] == g1][col],
                    col_data[col_data[group_col] == g2][col],
                )
                pairwise_records.append(
                    {"variable": col, "type": "numeric", "group_a": g1, "group_b": g2, "cohens_d": d}
                )
            # ANOVA
            samples = [col_data[col_data[group_col] == g][col] for g in groups if not col_data[col_data[group_col] == g].empty]
            if len(samples) >= 2:
                f_stat, p_val = stats.f_oneway(*samples)
                anova_records.append(
                    {"variable": col, "type": "numeric", "groups": len(groups), "f_stat": f_stat, "p_value": p_val}
                )
            # Tukey (requires >=3 groups)
            if len(groups) >= 3 and len(col_data) > 0:
                tukey = pairwise_tukeyhsd(endog=col_data[col], groups=col_data[group_col], alpha=0.05)
                tukey_df = pd.DataFrame(data=tukey.summary().data[1:], columns=tukey.summary().data[0])
                tukey_df.insert(0, "variable", col)
                tukey_records.extend(tukey_df.to_dict(orient="records"))
        else:
            # Categorical: compute odds ratio only for 2x2 in each pair
            for g1, g2 in combinations(groups, 2):
                pair_df = col_data[col_data[group_col].isin([g1, g2])]
                tab = pd.crosstab(pair_df[group_col], pair_df[col])
                if tab.shape == (2, 2):
                    or_val = odds_ratio_from_table(tab)
                    pairwise_records.append(
                        {
                            "variable": col,
                            "type": "categorical",
                            "group_a": g1,
                            "group_b": g2,
                            "odds_ratio": or_val,
                        }
                    )

    return pd.DataFrame(pairwise_records), pd.DataFrame(anova_records), pd.DataFrame(tukey_records)


def hypothesis_tests_numeric(
    df: pd.DataFrame, group_col: str, outcome_cols: List[str], alpha: float = 0.05
) -> pd.DataFrame:
    """Welch's t-test for two groups with CI of mean difference."""
    groups = [g for g in df[group_col].dropna().unique()]
    if len(groups) != 2:
        logger.info("Numeric tests require exactly 2 groups; found %s.", len(groups))
        return pd.DataFrame()
    g1, g2 = groups
    res = []
    for col in outcome_cols:
        if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
            continue
        s1 = df[df[group_col] == g1][col].dropna()
        s2 = df[df[group_col] == g2][col].dropna()
        if len(s1) < 2 or len(s2) < 2:
            continue
        t_stat, p_val = stats.ttest_ind(s1, s2, equal_var=False)
        diff = s1.mean() - s2.mean()
        # Welch-Satterthwaite DF
        v1, v2 = s1.var(ddof=1), s2.var(ddof=1)
        se = (v1 / len(s1) + v2 / len(s2)) ** 0.5
        df_w = (v1 / len(s1) + v2 / len(s2)) ** 2 / ((v1**2) / ((len(s1) - 1) * (len(s1) ** 2)) + (v2**2) / ((len(s2) - 1) * (len(s2) ** 2)))
        t_crit = stats.t.ppf(1 - alpha / 2, df_w) if se > 0 else float("nan")
        ci_low, ci_high = diff - t_crit * se, diff + t_crit * se if se > 0 else (float("nan"), float("nan"))
        res.append(
            {
                "variable": col,
                "group_a": g1,
                "group_b": g2,
                "n_a": len(s1),
                "n_b": len(s2),
                "mean_diff": diff,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "p_value": p_val,
                "test": "Welch t-test",
            }
        )
    return pd.DataFrame(res)


def hypothesis_tests_categorical(df: pd.DataFrame, group_col: str, outcome_cols: List[str]) -> pd.DataFrame:
    """Chi-square test for categorical outcomes vs group (all groups)."""
    if len(df[group_col].dropna().unique()) < 2:
        logger.info("Categorical tests need at least 2 groups.")
        return pd.DataFrame()
    res = []
    for col in outcome_cols:
        if col not in df.columns or pd.api.types.is_numeric_dtype(df[col]):
            continue
        tab = pd.crosstab(df[group_col], df[col])
        if tab.shape[0] < 2 or tab.shape[1] < 2:
            continue
        chi2, p_val, dof, _ = stats.chi2_contingency(tab)
        res.append(
            {
                "variable": col,
                "groups": tab.shape[0],
                "categories": tab.shape[1],
                "chi2": chi2,
                "dof": dof,
                "p_value": p_val,
                "test": "Chi-square",
            }
        )
    return pd.DataFrame(res)


def render_table(
    df: Optional[pd.DataFrame],
    title: str,
    precision: int = 3,
    highlight_cols: Optional[List[str]] = None,
) -> str:
    """Render HTML section with optional highlight."""
    if df is None or df.empty:
        return f"<h2>{title}</h2><p class='muted'>データなし</p>"
    if isinstance(df, pd.Series):
        df = df.to_frame()
    styler = df.style.format(precision=precision).set_table_attributes('class="table"')
    if highlight_cols:
        highlight = [c for c in highlight_cols if c in df.columns]
        if highlight:
            styler = styler.background_gradient(cmap="YlOrRd", subset=highlight)
    return f"<h2>{title}</h2>" + styler.to_html()


def encode_image(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    data = path.read_bytes()
    return base64.b64encode(data).decode("utf-8")


def generate_html_report(
    out_path: Path,
    preview_df: pd.DataFrame,
    num_summary: pd.DataFrame,
    cat_summary: pd.DataFrame,
    miss_df: pd.DataFrame,
    outlier_df: pd.DataFrame,
    outlier_rows: pd.DataFrame,
    group_num: Optional[pd.DataFrame] = None,
    group_cat: Optional[pd.DataFrame] = None,
    effect_df: Optional[pd.DataFrame] = None,
    anova_df: Optional[pd.DataFrame] = None,
    tukey_df: Optional[pd.DataFrame] = None,
    plot_paths: Optional[List[Path]] = None,
    ttest_df: Optional[pd.DataFrame] = None,
    chi2_df: Optional[pd.DataFrame] = None,
    alpha: float = 0.05,
) -> None:
    """Write a simple HTML report with tables."""
    plot_html = ""
    if plot_paths:
        imgs = []
        for p in plot_paths:
            enc = encode_image(p)
            if enc:
                imgs.append(f'<div class="plot"><img src="data:image/png;base64,{enc}" alt="{p.name}"/></div>')
        if imgs:
            plot_html = "<h2>プロット</h2><div class='plot-grid'>" + "".join(imgs) + "</div>"

    outlier_tabs = f"""
    <h2>外れ値</h2>
    <div class="tabs card">
      <div class="tab-buttons">
        <button data-tab="outlier-summary" class="active">サマリー</button>
        <button data-tab="outlier-rows">サンプル行</button>
      </div>
      <div id="outlier-summary" class="tab-content active">
        {render_table(outlier_df, "", highlight_cols=["outlier_count", "outlier_pct"])}
      </div>
      <div id="outlier-rows" class="tab-content">
        {render_table(outlier_rows, "")}
      </div>
    </div>
    """

    # Simple textual insights
    def insights_block() -> str:
        insights = []
        insights.append(f"行数: {len(preview_df)} / 数値列: {len(num_summary)} / カテゴリ列: {len(cat_summary)}")
        if not miss_df.empty:
            total_miss = miss_df["missing_count"].sum()
            insights.append(f"総欠測: {total_miss}")
        if not outlier_df.empty and "outlier_count" in outlier_df.columns:
            total_out = outlier_df["outlier_count"].sum()
            insights.append(f"外れ値検知: {total_out} 件 (IQR)")
        if ttest_df is not None and not ttest_df.empty:
            sig = ttest_df[ttest_df["p_value"] < alpha]
            if not sig.empty:
                vars_sig = ", ".join(sig["variable"].astype(str).tolist())
                insights.append(f"t検定で有意 (p<{alpha}): {vars_sig}")
        if chi2_df is not None and not chi2_df.empty:
            sig = chi2_df[chi2_df["p_value"] < alpha]
            if not sig.empty:
                vars_sig = ", ".join(sig["variable"].astype(str).tolist())
                insights.append(f"カイ二乗で有意 (p<{alpha}): {vars_sig}")
        if not insights:
            insights.append("特記事項なし")
        items = "".join(f"<li>{i}</li>" for i in insights)
        return f"<ul>{items}</ul>"

    html = f"""
<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="UTF-8" />
  <title>記述統計レポート</title>
  <style>
    :root {{
      --bg: #0f1729;
      --panel: #111b2e;
      --card: #1b2a44;
      --accent: #4ade80;
      --accent-2: #22d3ee;
      --muted: #9fb3c8;
      --text: #e7ecf5;
      --border: #1f2f4c;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      min-height: 100vh;
      background: radial-gradient(circle at 20% 20%, #14213d 0, #0f1729 45%), radial-gradient(circle at 80% 0, #0e7490 0, #0f1729 40%);
      color: var(--text);
      font-family: "IBM Plex Sans", "Helvetica Neue", Arial, sans-serif;
    }}
    .shell {{
      max-width: 1200px;
      margin: 0 auto;
      padding: 32px 20px 48px;
      display: flex;
      flex-direction: column;
      gap: 18px;
    }}
    h1 {{ margin: 0; font-size: 26px; letter-spacing: 0.01em; }}
    h2 {{ margin-top: 18px; }}
    .pill {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 8px 12px;
      border-radius: 999px;
      background: linear-gradient(135deg, rgba(74, 222, 128, 0.18), rgba(34, 211, 238, 0.18));
      color: var(--muted);
      font-size: 13px;
      border: 1px solid rgba(34, 211, 238, 0.35);
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      gap: 12px;
    }}
    .card {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 14px;
      position: relative;
      overflow: hidden;
    }}
    .card small {{ color: var(--muted); }}
    .spark {{
      position: absolute;
      inset: 0;
      background: radial-gradient(circle at 80% 0%, rgba(34, 211, 238, 0.18), transparent 45%);
      pointer-events: none;
    }}
    .section {{ background: var(--panel); border: 1px solid var(--border); border-radius: 14px; padding: 16px; }}
    .muted {{ color: var(--muted); }}
    .table {{ width: 100%; border-collapse: collapse; font-size: 14px; background: #111827; color: #e5e7eb; }}
    .table th, .table td {{ padding: 8px 10px; border: 1px solid #1f2937; }}
    .table th {{ background: #1f2937; text-align: left; }}
    .table tr:nth-child(even) {{ background: #0b1220; }}
    .plot-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 12px; }}
    .plot img {{ width: 100%; border: 1px solid var(--border); border-radius: 10px; background: #0b1220; }}
    .tabs {{ margin-top: 8px; }}
    .tab-buttons {{ display: flex; gap: 8px; margin-bottom: 8px; flex-wrap: wrap; }}
    .tab-buttons button {{ padding: 8px 12px; border-radius: 8px; border: 1px solid var(--border); background: #111827; color: var(--text); cursor: pointer; }}
    .tab-buttons button.active {{ background: #10b981; color: #0b1220; border-color: #10b981; }}
    .tab-content {{ display: none; }}
    .tab-content.active {{ display: block; }}
  </style>
</head>
<body>
  <div class="shell">
    <div class="card" style="display:flex;justify-content:space-between;align-items:center;gap:12px;">
      <div>
        <h1>記述統計レポート</h1>
        <div class="pill">データプレビューとサマリー / multi-arm 対応</div>
      </div>
    </div>

    <div class="grid">
      <div class="card"><div class="spark"></div><small>レコード数</small><div style="font-size:24px;font-weight:600;">{len(preview_df)} 行</div><small>head() を表示</small></div>
      <div class="card"><div class="spark"></div><small>数値列</small><div style="font-size:24px;font-weight:600;">{len(num_summary)}</div><small>describe()</small></div>
      <div class="card"><div class="spark"></div><small>カテゴリ列</small><div style="font-size:24px;font-weight:600;">{len(cat_summary)}</div><small>頻度・ユニーク数</small></div>
      <div class="card"><div class="spark"></div><small>総欠測</small><div style="font-size:24px;font-weight:600;">{miss_df['missing_count'].sum() if not miss_df.empty else 0}</div><small>missing_count 合計</small></div>
    </div>

    <div class="section">
      <h2>簡易サマリー</h2>
      {insights_block()}
    </div>

    <div class="section">{render_table(preview_df.head(10), "データプレビュー (先頭10行)")}</div>
    <div class="section">{render_table(num_summary, "数値列サマリー")}</div>
    <div class="section">{render_table(cat_summary, "カテゴリ列サマリー")}</div>
    <div class="section">{render_table(miss_df, "欠測サマリー", highlight_cols=["missing_count", "missing_pct"])}</div>
    <div class="section">{outlier_tabs}</div>
    {f'<div class="section">{plot_html}</div>' if plot_html else ''}
    {f'<div class="section">{render_table(group_num, "グループ別 数値サマリー")}</div>' if group_num is not None else ""}
    {f'<div class="section">{render_table(group_cat, "グループ別 カテゴリサマリー")}</div>' if group_cat is not None else ""}
    {f'<div class="section">{render_table(effect_df, "効果量 (ペアワイズ)")}</div>' if effect_df is not None else ""}
    {f'<div class="section">{render_table(anova_df, "ANOVA")}</div>' if anova_df is not None else ""}
    {f'<div class="section">{render_table(tukey_df, "Tukey HSD")}</div>' if tukey_df is not None else ""}
    {f'<div class="section">{render_table(ttest_df, "t検定 (2群)", highlight_cols=["p_value"] )}</div>' if ttest_df is not None else ""}
    {f'<div class="section">{render_table(chi2_df, "カイ二乗検定", highlight_cols=["p_value"] )}</div>' if chi2_df is not None else ""}
  </div>
  <script>
    const tabs = document.querySelectorAll('.tab-buttons button');
    tabs.forEach(btn => {{
      btn.addEventListener('click', () => {{
        const target = btn.dataset.tab;
        document.querySelectorAll('.tab-buttons button').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
        btn.classList.add('active');
        document.getElementById(target).classList.add('active');
      }});
    }});
  </script>
</body>
</html>
"""
    out_path.write_text(html, encoding="utf-8")
    logger.info("Saved HTML report: %s", out_path)


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
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level for confidence intervals and tests.",
    )
    parser.add_argument(
        "--html-report",
        type=Path,
        help="Path to save HTML report (defaults to <out-dir>/report.html).",
    )
    args = parser.parse_args()

    out_dir = ensure_output_dir(args.out_dir)
    ensure_matplotlib_config(out_dir)
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

    plot_paths = plot_numeric(df, out_dir, max_plots=args.max_plots)

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

    grp_num_df: Optional[pd.DataFrame] = None
    grp_cat_df: Optional[pd.DataFrame] = None
    eff_df: Optional[pd.DataFrame] = None
    anova_df: Optional[pd.DataFrame] = None
    tukey_df: Optional[pd.DataFrame] = None
    ttest_df: Optional[pd.DataFrame] = None
    chi2_df: Optional[pd.DataFrame] = None

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
            grp_num_df, grp_cat_df = grp_num, grp_cat
            if args.effect_cols:
                eff_df, anova_df, tukey_df = effect_sizes(df, args.group_col, args.effect_cols)
                eff_path = out_dir / "effect_sizes.csv"
                anova_path = out_dir / "effect_anova.csv"
                tukey_path = out_dir / "effect_tukey.csv"
                if eff_df is not None and not eff_df.empty:
                    eff_df.to_csv(eff_path, index=False)
                    logger.info("Saved effect sizes: %s", eff_path)
                else:
                    logger.info("No effect sizes computed (check group count or columns).")
                if anova_df is not None and not anova_df.empty:
                    anova_df.to_csv(anova_path, index=False)
                    logger.info("Saved ANOVA results: %s", anova_path)
                if tukey_df is not None and not tukey_df.empty:
                    tukey_df.to_csv(tukey_path, index=False)
                    logger.info("Saved Tukey HSD: %s", tukey_path)
                # Hypothesis tests (2-group Welch t, chi-square)
                ttest_df = hypothesis_tests_numeric(df, args.group_col, args.effect_cols, alpha=args.alpha)
                chi2_df = hypothesis_tests_categorical(df, args.group_col, args.effect_cols)
                if ttest_df is not None and not ttest_df.empty:
                    ttest_path = out_dir / "tests_ttest.csv"
                    ttest_df.to_csv(ttest_path, index=False)
                    logger.info("Saved t-tests: %s", ttest_path)
                if chi2_df is not None and not chi2_df.empty:
                    chi2_path = out_dir / "tests_chi2.csv"
                    chi2_df.to_csv(chi2_path, index=False)
                    logger.info("Saved chi-square tests: %s", chi2_path)
        except KeyError as e:
            logger.error(str(e))

    report_path = args.html_report or (out_dir / "report.html")
    generate_html_report(
        report_path,
        preview(df),
        num_summary,
        cat_summary,
        miss_df,
        outlier_df,
        outlier_rows,
        grp_num_df,
        grp_cat_df,
        eff_df,
        anova_df,
        tukey_df,
        plot_paths,
        ttest_df,
        chi2_df,
        alpha=args.alpha,
    )

    logger.info("Done. Outputs written to %s", out_dir.resolve())


if __name__ == "__main__":
    main()
