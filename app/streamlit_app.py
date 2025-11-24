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

# Mockup-like styling
CSS = """
<style>
:root {
  --bg: #0f1729;
  --panel: #111b2e;
  --card: #1b2a44;
  --accent: #4ade80;
  --accent-2: #22d3ee;
  --muted: #9fb3c8;
  --text: #e7ecf5;
  --border: #1f2f4c;
}
body, .main, .block-container {
  background: radial-gradient(circle at 20% 20%, #14213d 0, #0f1729 45%), radial-gradient(circle at 80% 0, #0e7490 0, #0f1729 40%) !important;
  color: var(--text) !important;
}
.css-1d391kg, .css-18e3th9, .css-1lcbmhc {  /* sidebar containers (Streamlit class names may change by version) */
  background: linear-gradient(180deg, #0c1222 0%, #0f1729 40%, #0c1222 100%) !important;
  border-right: 1px solid var(--border) !important;
}
div.stButton > button {
  background: linear-gradient(135deg, rgba(74, 222, 128, 0.22), rgba(34, 211, 238, 0.18));
  border: 1px solid rgba(74, 222, 128, 0.4);
  color: #0a1525;
  border-radius: 10px;
  font-weight: 700;
}
.pill {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 8px 12px;
  border-radius: 999px;
  background: rgba(34, 211, 238, 0.12);
  color: var(--muted);
  font-size: 13px;
  border: 1px solid rgba(34, 211, 238, 0.3);
}
.card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 14px;
  position: relative;
  overflow: hidden;
}
.spark {
  position: absolute;
  inset: 0;
  background: radial-gradient(circle at 80% 0%, rgba(34, 211, 238, 0.18), transparent 45%);
  pointer-events: none;
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

st.sidebar.markdown(
    """
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:12px;">
      <div style="width:12px;height:12px;border-radius:50%;background:linear-gradient(135deg,#4ade80,#22d3ee);box-shadow:0 0 12px rgba(34,211,238,0.4);"></div>
      <span style="font-weight:700;color:#e7ecf5;">MedStats Assist</span>
    </div>
    <div style="display:flex;flex-direction:column;gap:8px;">
      <a href="../frontend/index.html" target="_blank" style="color:#e7ecf5;text-decoration:none;padding:10px 12px;border-radius:10px;border:1px solid rgba(34,211,238,0.35);background:rgba(34,211,238,0.08);">ダッシュボード</a>
      <a href="../outputs/report.html" target="_blank" style="color:#e7ecf5;text-decoration:none;padding:10px 12px;border-radius:10px;border:1px solid var(--border);">最新レポート</a>
      <a href="http://localhost:8504" target="_blank" style="color:#e7ecf5;text-decoration:none;padding:10px 12px;border-radius:10px;border:1px solid var(--border);">Streamlitトップ</a>
      <a href="../work/manual.md" target="_blank" style="color:#e7ecf5;text-decoration:none;padding:10px 12px;border-radius:10px;border:1px solid var(--border);">操作マニュアル</a>
      <a href="../work/spec.md" target="_blank" style="color:#e7ecf5;text-decoration:none;padding:10px 12px;border-radius:10px;border:1px solid var(--border);">仕様書</a>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="card" style="display:flex;justify-content:space-between;align-items:center;gap:12px;margin-bottom:12px;">
      <div>
        <h2 style="margin:0;">医療統計アシスト - 記述統計デモ</h2>
        <div class="pill">モックアップ風UI / CSVアップロードで集計</div>
      </div>
      <div style="display:flex;gap:8px;flex-wrap:wrap;">
        <a href="http://localhost:8504" target="_blank" style="text-decoration:none;" class="pill">Streamlitトップ</a>
        <a href="../outputs/report.html" target="_blank" style="text-decoration:none;" class="pill">最新レポート</a>
        <a href="../frontend/index.html" target="_blank" style="text-decoration:none;" class="pill">フロントページ</a>
        <a href="../work/manual.md" target="_blank" style="text-decoration:none;" class="pill">操作マニュアル</a>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

uploaded = st.file_uploader("CSV/TSV をアップロード", type=["csv", "tsv"])
delimiter = st.sidebar.selectbox("区切り文字", ["自動判定", "カンマ(,)", "タブ(\\t)", "セミコロン(;)"], index=0)
encoding = st.sidebar.selectbox("文字コード", ["utf-8", "shift_jis"], index=0)
preview_rows = st.sidebar.slider("プレビュー行数", min_value=5, max_value=20, value=10)
max_plots = st.slider("最大プロット数", min_value=1, max_value=12, value=6)
max_outlier_rows = st.slider("外れ値サンプル行の上限", min_value=10, max_value=200, value=100, step=10)
impute_numeric = st.sidebar.selectbox("数値の欠測処理", ["none", "mean", "median"], index=0)
impute_categorical = st.sidebar.selectbox("カテゴリの欠測処理", ["none", "mode"], index=0)
drop_thresh = st.sidebar.slider("行を残すための非欠測割合", min_value=0.5, max_value=1.0, value=1.0, step=0.05)

if uploaded:
    if delimiter == "自動判定":
        sep = None
    elif delimiter == "カンマ(,)":
        sep = ","
    elif delimiter == "タブ(\\t)":
        sep = "\t"
    else:
        sep = ";"
    try:
        df = pd.read_csv(uploaded, sep=sep, engine="python" if sep is None else "c", encoding=encoding)
    except Exception as e:
        st.error(f"読み込みに失敗しました: {e}")
        st.stop()

    # Preprocess: drop then impute
    df, preproc_info = preprocess_df(
        df,
        impute_numeric=impute_numeric,
        impute_categorical=impute_categorical,
        drop_missing_thresh=drop_thresh,
    )
    st.info(
        f"前処理: dropped_rows={preproc_info.get('dropped_rows',0)}, "
        f"impute_numeric={preproc_info.get('imputed_numeric')}, "
        f"impute_categorical={preproc_info.get('imputed_categorical')}"
    )
    st.success(f"読み込み完了: {df.shape[0]} 行 x {df.shape[1]} 列")

    group_col = st.selectbox("グループ列（効果量/グループ別集計に使用）", options=["(なし)"] + list(df.columns))
    effect_cols = st.multiselect("効果量を計算する列（数値: Cohen's d, 2x2カテゴリ: OR）", options=list(df.columns))

    with st.expander("データプレビュー"):
        st.dataframe(preview(df, rows=preview_rows), use_container_width=True)

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

    # Plots (allow selection of numeric columns)
    numeric_cols = list(df.select_dtypes(include="number").columns)
    plot_select = st.multiselect("プロットする数値列を選択", options=numeric_cols, default=numeric_cols[:max_plots])
    plot_paths = plot_numeric(df[numeric_cols], outputs_dir, max_plots=max_plots, selected_cols=plot_select)

    col1, col2, col3 = st.columns(3)
    col1.markdown(
        f"""
        <div class="card">
          <div class="spark"></div>
          <small>数値列</small>
          <div style="font-size:24px;font-weight:700;">{len(num_summary)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    col2.markdown(
        f"""
        <div class="card">
          <div class="spark"></div>
          <small>カテゴリ列</small>
          <div style="font-size:24px;font-weight:700;">{len(cat_summary)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    col3.markdown(
        f"""
        <div class="card">
          <div class="spark"></div>
          <small>総欠測</small>
          <div style="font-size:24px;font-weight:700;">{miss_df["missing_count"].sum()}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

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
