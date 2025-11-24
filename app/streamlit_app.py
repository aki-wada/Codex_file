import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from main import (
    effect_sizes,
    ensure_matplotlib_config,
    ensure_output_dir,
    generate_html_report,
    group_summaries,
    missing_summary,
    outlier_summary,
    plot_numeric,
    preview,
    summarize,
)


st.set_page_config(page_title="MedStats Assist", layout="wide")
st.title("医療統計アシスト - 記述統計デモ")

uploaded = st.file_uploader("CSV/TSV をアップロード", type=["csv", "tsv"])
max_plots = st.slider("最大プロット数", min_value=1, max_value=12, value=6)
max_outlier_rows = st.slider("外れ値サンプル行の上限", min_value=10, max_value=200, value=100, step=10)

if uploaded:
    sep = "\t" if uploaded.name.endswith(".tsv") else ","
    df = pd.read_csv(uploaded, sep=sep)
    st.success(f"読み込み完了: {df.shape[0]} 行 x {df.shape[1]} 列")

    group_col = st.selectbox("グループ列（効果量/グループ別集計に使用）", options=["(なし)"] + list(df.columns))
    effect_cols = st.multiselect("効果量を計算する列（数値: Cohen's d, 2x2カテゴリ: OR）", options=list(df.columns))

    with st.expander("データプレビュー"):
        st.dataframe(preview(df), use_container_width=True)

    # Prepare outputs
    outputs_dir = ensure_output_dir(Path("outputs/streamlit"))
    ensure_matplotlib_config(outputs_dir)

    num_summary, cat_summary = summarize(df)
    miss_df = missing_summary(df)
    out_sum, out_rows = outlier_summary(df, max_rows=max_outlier_rows)

    grp_num_df = grp_cat_df = eff_df = anova_df = tukey_df = None
    if group_col and group_col != "(なし)":
        grp_num_df, grp_cat_df = group_summaries(df, group_col)
        if effect_cols:
            eff_df, anova_df, tukey_df = effect_sizes(df, group_col, effect_cols)

    # Plots
    plot_paths = plot_numeric(df, outputs_dir, max_plots=max_plots)

    col1, col2, col3 = st.columns(3)
    col1.metric("数値列", len(num_summary))
    col2.metric("カテゴリ列", len(cat_summary))
    col3.metric("総欠測", miss_df["missing_count"].sum())

    st.subheader("数値列サマリー")
    st.dataframe(num_summary, use_container_width=True)

    st.subheader("カテゴリ列サマリー")
    st.dataframe(cat_summary, use_container_width=True)

    st.subheader("欠測サマリー")
    st.dataframe(miss_df, use_container_width=True)

    st.subheader("外れ値サマリー")
    st.dataframe(out_sum, use_container_width=True)
    st.subheader("外れ値サンプル行")
    st.dataframe(out_rows.head(max_outlier_rows), use_container_width=True)

    if grp_num_df is not None:
        st.subheader("グループ別 数値サマリー")
        st.dataframe(grp_num_df, use_container_width=True)
    if grp_cat_df is not None:
        st.subheader("グループ別 カテゴリサマリー")
        st.dataframe(grp_cat_df, use_container_width=True)
    if eff_df is not None and not eff_df.empty:
        st.subheader("効果量 (ペアワイズ)")
        st.dataframe(eff_df, use_container_width=True)
    if anova_df is not None and not anova_df.empty:
        st.subheader("ANOVA")
        st.dataframe(anova_df, use_container_width=True)
    if tukey_df is not None and not tukey_df.empty:
        st.subheader("Tukey HSD")
        st.dataframe(tukey_df, use_container_width=True)

    st.subheader("プロット")
    for p in plot_paths:
        st.image(str(p))

    # HTML report generation for download
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp:
        generate_html_report(
            Path(tmp.name),
            preview(df),
            num_summary,
            cat_summary,
            miss_df,
            out_sum,
            out_rows,
            grp_num_df,
            grp_cat_df,
            eff_df,
            anova_df,
            tukey_df,
            plot_paths,
        )
        html_bytes = Path(tmp.name).read_bytes()
        st.download_button("HTMLレポートをダウンロード", data=html_bytes, file_name="report.html", mime="text/html")

    st.info("multi-arm trial では全ペアの効果量と ANOVA/Tukey を計算します。2群のみの場合は従来の計算です。")
else:
    st.write("CSV/TSV をアップロードしてください。")
