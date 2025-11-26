# 操作マニュアル（初心者向け）

## 1. 画面で使う（最も簡単）
```bash
cd /Users/wadaakihiko/Desktop/wada_work/Codex_file
./.venv/bin/streamlit run app/streamlit_app.py
```
ブラウザで http://localhost:8504 を開き、CSV/TSV をアップロードするだけで表・グラフ・効果量を確認し、HTMLレポートもダウンロードできます。

- サイドバーで「設定」「解析」を切替
  - 設定: 区切り/文字コード、欠測処理、プレビュー行数、プロット数などを保存
  - 解析: 保存した設定で読み込み→前処理→記述統計→検定を実行
  - 上部にバージョン、進行状況バッジ（読み込み→前処理→記述/可視化→統計解析）
- 解析タブの見方
  - 「検定」: t検定 / ANOVA / Tukey / Mann-Whitney / Kruskal / カイ二乗 / Fisher / 正規性 / 相関 (Pearson/Spearman)
  - 「グラフ」: 相関ヒートマップ（Altair）とヒストグラム・箱ひげ
  - 「効果量」: 全ペアのCohen's d / OR、多群のANOVA/Tukey
  - 「概要」: 記述統計と欠測サマリー
- レポート: 入力ファイル名・生成日時・設定のメタ情報付き。HTMLをダウンロード可能。

## 2. 事前準備（初回のみ）
```bash
cd /Users/wadaakihiko/Desktop/wada_work/Codex_file
/Users/wadaakihiko/homebrew/bin/python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
※ 警告は基本無視でOK。消したい場合は `MPLCONFIGDIR=./outputs/.mplconfig` を設定。

## 3. コマンドで使う（バッチ処理向け）
サンプルデータで試す例:
```bash
./.venv/bin/python app/main.py data/clinical_trial_data.csv \
  --out-dir outputs \
  --group-col arm \
  --effect-cols baseline_score week4_score week8_score ae_grade \
  --html-report outputs/report.html
```
ポイント:
- `--group-col` に群の列名を入れると群別の表を出力。
- `--effect-cols` に効果量を出したい列を並べると、全ペアの効果量・ANOVA・Tukeyを計算（3群以上もOK）。
- `report.html` をブラウザで開けば表とグラフをまとめて確認。

主な出力（`outputs` 内）:
- `summary_numeric.csv` / `summary_categorical.csv` … 記述統計
- `missing_summary.csv` … 欠測数・率
- `outlier_summary.csv` / `outlier_rows.csv` … 外れ値サマリー/サンプル
- `group_numeric_summary.csv` / `group_categorical_summary.csv` … 群別サマリー
- `effect_sizes.csv` … 効果量（数値: Cohen's d、2x2カテゴリ: OR）
- `effect_anova.csv` / `effect_tukey.csv` … ANOVAとTukey
- `tests_ttest.csv` / `tests_chi2.csv` … Welch t検定とカイ二乗
- `tests_mwu.csv` / `tests_kruskal.csv` / `tests_fisher.csv` … 非正規・多群・小標本の検定
- `corr_pearson.csv` / `corr_spearman.csv` … 相関係数
- `<列名>_plot.png` … ヒストグラム＋箱ひげ
- `corr_heatmap.png` … 相関ヒートマップ
- `report.html` … すべてをまとめたレポート

前処理オプション（CLI）:
- `--impute-numeric {none,mean,median}` / `--impute-categorical {none,mode}`
- `--drop-missing-thresh` 非欠測割合の閾値（例: 0.8 で 80%以上埋まっている行のみ残す）
- 前処理後データは `cleaned.csv`（`--cleaned-csv` で変更可）

## 4. よくあるつまずき
- `pip install` がSSLエラーになる: Homebrew版の Python 3.12 を使い、もう一度 `pip install -r requirements.txt` を実行。
- Matplotlibのキャッシュ警告: `MPLCONFIGDIR=./outputs/.mplconfig` を設定。
- 効果量が空になる: グループが1つしかない、または2x2以外のカテゴリ列を指定している可能性。列名とグループ数を確認してください。
