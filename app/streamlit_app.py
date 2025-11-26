import tempfile
import datetime
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
    normality_tests,
    outlier_summary,
    preprocess_df,
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
.step-badge {
  display: inline-block;
  padding: 6px 10px;
  border-radius: 10px;
  font-weight: 700;
  color: #0a1525;
  margin-bottom: 6px;
}
.step-pending { background: #cbd5e1; border: 1px solid #94a3b8; }
.step-current { background: #22d3ee; border: 1px solid #0ea5e9; }
.step-done { background: #4ade80; border: 1px solid #22c55e; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

def step_badge(label: str, state: str) -> str:
    cls = {"pending": "step-pending", "current": "step-current", "done": "step-done"}.get(state, "step-pending")
    return f'<span class="step-badge {cls}">{label}</span>'

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

view = st.sidebar.radio("画面", ["解析", "設定"], index=0)

if "config" not in st.session_state:
    st.session_state["config"] = {
        "delimiter": "自動判定",
        "encoding": "utf-8",
        "preview_rows": 10,
        "max_plots": 6,
        "max_outlier_rows": 100,
        "impute_numeric": "none",
        "impute_categorical": "none",
        "drop_thresh": 1.0,
    }
if "last_upload" not in st.session_state:
    st.session_state["last_upload"] = None

st.markdown(
    """
    <div class="card" style="display:flex;justify-content:space-between;align-items:center;gap:12px;margin-bottom:12px;">
      <div>
        <h2 style="margin:0;">医療統計アシスト - 記述統計デモ</h2>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if view == "設定":
    st.markdown("### 設定")
    cfg = st.session_state["config"]
    with st.form("settings_form"):
        delimiter = st.selectbox("区切り文字", ["自動判定", "カンマ(,)", "タブ(\\t)", "セミコロン(;)"], index=["自動判定", "カンマ(,)", "タブ(\\t)", "セミコロン(;)"].index(cfg["delimiter"]))
        encoding = st.selectbox("文字コード", ["utf-8", "shift_jis"], index=["utf-8", "shift_jis"].index(cfg["encoding"]))
        preview_rows = st.slider("プレビュー行数", min_value=5, max_value=20, value=int(cfg["preview_rows"]))
        max_plots = st.slider("最大プロット数", min_value=1, max_value=12, value=int(cfg["max_plots"]))
        max_outlier_rows = st.slider("外れ値サンプル行の上限", min_value=10, max_value=200, value=int(cfg["max_outlier_rows"]), step=10)
        impute_numeric = st.selectbox("数値の欠測処理", ["none", "mean", "median"], index=["none", "mean", "median"].index(cfg["impute_numeric"]))
        impute_categorical = st.selectbox("カテゴリの欠測処理", ["none", "mode"], index=["none", "mode"].index(cfg["impute_categorical"]))
        drop_thresh = st.slider("行を残すための非欠測割合", min_value=0.5, max_value=1.0, value=float(cfg["drop_thresh"]), step=0.05)
        submitted = st.form_submit_button("保存")
        if submitted:
            st.session_state["config"] = {
                "delimiter": delimiter,
                "encoding": encoding,
                "preview_rows": preview_rows,
                "max_plots": max_plots,
                "max_outlier_rows": max_outlier_rows,
                "impute_numeric": impute_numeric,
                "impute_categorical": impute_categorical,
                "drop_thresh": drop_thresh,
            }
            st.success("設定を保存しました。左のラジオで解析に戻ってください。")

if view == "解析":
    cfg = st.session_state["config"]
    delimiter = cfg["delimiter"]
    encoding = cfg["encoding"]
    preview_rows = cfg["preview_rows"]
    max_plots = cfg["max_plots"]
    max_outlier_rows = cfg["max_outlier_rows"]
    impute_numeric = cfg["impute_numeric"]
    impute_categorical = cfg["impute_categorical"]
    drop_thresh = cfg["drop_thresh"]

    uploaded = st.file_uploader("CSV/TSV をアップロード", type=["csv", "tsv"])
    cached = st.session_state.get("last_upload")
    use_cached = False
    if cached:
        st.sidebar.markdown(
            f"""
            <div style="margin-top:8px;padding:8px;border:1px solid var(--border);border-radius:8px;">
              <div style="color:#e7ecf5;font-weight:700;">前回のファイル</div>
              <div style="color:#9fb3c8;">{cached.get('name','')}</div>
              <div style="color:#9fb3c8;">sep={cached.get('sep','auto')}, enc={cached.get('encoding','')}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.sidebar.button("前回のアップロードを再利用"):
            use_cached = True
    step_state = {"load": "pending", "preprocess": "pending", "describe": "pending", "test": "pending"}
    num_summary = cat_summary = miss_df = out_sum = out_rows = None
    grp_num_df = grp_cat_df = eff_df = anova_df = tukey_df = None
    ttest_df = mwu_df = kw_df = chi2_df = fisher_df = normality_df = None
    plot_paths = []

    if uploaded or use_cached:
        if use_cached and cached:
            cache_path = Path(cached["path"])
            sep = cached.get("sep", None)
            encoding = cached.get("encoding", "utf-8")
            try:
                df = pd.read_csv(cache_path, sep=sep, engine="python" if sep is None else "c", encoding=encoding)
            except Exception as e:
                st.error(f"前回ファイルの読み込みに失敗しました: {e}")
                st.stop()
            step_state["load"] = "done"
        elif uploaded:
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
            step_state["load"] = "done"

            # 保存: 前回ファイル情報
            cache_dir = Path("outputs/upload_cache")
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = cache_dir / uploaded.name
            cache_path.write_bytes(uploaded.getbuffer())
            st.session_state["last_upload"] = {
                "path": str(cache_path),
                "name": uploaded.name,
                "sep": sep,
                "encoding": encoding,
            }

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
        step_state["preprocess"] = "done"
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
        step_state["describe"] = "done"

        numeric_cols = list(df.select_dtypes(include="number").columns)
        grp_num_df = grp_cat_df = eff_df = anova_df = tukey_df = None
        # Normality test
        normality_df = normality_tests(df, numeric_cols)

        if group_col and group_col != "(なし)":
            grp_num_df, grp_cat_df = group_summaries(df, group_col)
            if effect_cols:
                eff_df, anova_df, tukey_df = effect_sizes(df, group_col, effect_cols)

        # Plots (allow selection of numeric columns)
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

        tab_overview, tab_tests, tab_effect, tab_outlier, tab_plots = st.tabs(
            ["概要", "検定", "効果量", "欠測/外れ値", "グラフ"]
        )

        with tab_overview:
            st.subheader("数値列サマリー")
            st.dataframe(num_summary, use_container_width=True)
            st.subheader("カテゴリ列サマリー")
            st.dataframe(cat_summary, use_container_width=True)
            st.subheader("欠測サマリー")
            st.dataframe(miss_df, use_container_width=True)

        with tab_tests:
            st.subheader("t検定 / Mann-Whitney / Kruskal / ANOVA / カイ二乗")
            if ttest_df is not None:
                st.markdown("t検定 (2群, Welch)")
                st.dataframe(ttest_df, use_container_width=True)
            if anova_df is not None and not anova_df.empty:
                st.markdown("ANOVA")
                st.dataframe(anova_df, use_container_width=True)
            if tukey_df is not None and not tukey_df.empty:
                st.markdown("Tukey HSD")
                st.dataframe(tukey_df, use_container_width=True)
            if mwu_df is not None and not mwu_df.empty:
                st.markdown("Mann-Whitney U")
                st.dataframe(mwu_df, use_container_width=True)
            if kw_df is not None and not kw_df.empty:
                st.markdown("Kruskal-Wallis")
                st.dataframe(kw_df, use_container_width=True)
            if normality_df is not None and not normality_df.empty:
                st.markdown("正規性検定 (Shapiro-Wilk)")
                st.dataframe(normality_df, use_container_width=True)
            if chi2_df is not None and not chi2_df.empty:
                st.markdown("カイ二乗検定")
                st.dataframe(chi2_df, use_container_width=True)
            if fisher_df is not None and not fisher_df.empty:
                st.markdown("Fisher exact (2x2)")
                st.dataframe(fisher_df, use_container_width=True)

        with tab_effect:
            if grp_num_df is not None:
                st.subheader("グループ別 数値サマリー")
                st.dataframe(grp_num_df, use_container_width=True)
            if grp_cat_df is not None:
                st.subheader("グループ別 カテゴリサマリー")
                st.dataframe(grp_cat_df, use_container_width=True)
            if eff_df is not None and not eff_df.empty:
                st.subheader("効果量 (ペアワイズ)")
                st.dataframe(eff_df, use_container_width=True)
            if (eff_df is not None and not eff_df.empty) or (anova_df is not None and not anova_df.empty):
                step_state["test"] = "done"

        with tab_outlier:
            st.subheader("欠測サマリー")
            st.dataframe(miss_df, use_container_width=True)
            st.subheader("外れ値サマリー")
            st.dataframe(out_sum, use_container_width=True)
            st.subheader("外れ値サンプル行")
            st.dataframe(out_rows.head(max_outlier_rows), use_container_width=True)

        with tab_plots:
            st.subheader("プロット")
            for p in plot_paths:
                st.image(str(p))

        st.subheader("解釈メモ")
        memo = []
        if eff_df is not None and not eff_df.empty:
            memo.append("効果量: |d|≈0.2小, 0.5中, 0.8大 / OR>1でグループA優位, <1でB優位。")
        if (anova_df is not None and not anova_df.empty) or (tukey_df is not None and not tukey_df.empty):
            memo.append("ANOVA/Tukey: 多群の差を検定し、有意ならTukeyでどの組が異なるか確認。")
        if (eff_df is not None and not eff_df.empty) or (anova_df is not None and not anova_df.empty):
            memo.append("p値と効果量を併せて解釈し、実質的な大きさを判断してください。")
        if normality_df is not None and not normality_df.empty:
            memo.append("正規性: Shapiro p<α なら非パラ検定も参考に。")
        if not memo:
            memo.append("特記事項なし。")
        for line in memo:
            st.markdown(f"- {line}")

        st.subheader("手法選択のアドバイス")
        adv = [
            "2群の数値: Welch t検定 + Cohen's d。非正規/外れ値が強い場合は Mann-Whitney を併用。",
            "多群の数値: ANOVA + Tukey。非正規/外れ値が強い場合は Kruskal-Wallis を併用。",
            "カテゴリ2x2: カイ二乗（期待度数が小さいときは Fisher）。",
            "カテゴリ多水準: カイ二乗（期待度数を確認）。",
            "正規性: Shapiro p<α なら非パラ検定の結果も参考に。",
        ]
        for line in adv:
            st.markdown(f"- {line}")

        # Metadata for report
        meta = {
            "input_file": uploaded.name,
            "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "impute_numeric": impute_numeric,
            "impute_categorical": impute_categorical,
            "drop_missing_thresh": drop_thresh,
            "group_col": group_col or "-",
            "effect_cols": effect_cols,
        }

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
                meta=meta,
            )
            html_bytes = Path(tmp.name).read_bytes()
            st.download_button("HTMLレポートをダウンロード", data=html_bytes, file_name="report.html", mime="text/html")

        st.sidebar.markdown(
            f"""
            <div style="margin-top:16px; color:#e7ecf5; display:flex; flex-direction:column; gap:6px;">
              <div style="font-weight:700; margin-bottom:2px;">進行状況</div>
              <div>{step_badge("データ読み込み", step_state.get("load", "pending"))}</div>
              <div>{step_badge("前処理", step_state.get("preprocess", "pending"))}</div>
              <div>{step_badge("記述統計/可視化", step_state.get("describe", "pending"))}</div>
              <div>{step_badge("統計解析/効果量", step_state.get("test", "pending"))}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.info("multi-arm trial では全ペアの効果量と ANOVA/Tukey を計算します。2群のみの場合は従来の計算です。")
    else:
        st.write("CSV/TSV をアップロードしてください。")
