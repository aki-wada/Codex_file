# 操作マニュアル（初心者向け）

## 1. 事前準備（最初だけ）
```bash
cd /Users/wadaakihiko/Desktop/wada_work/Codex_file
/Users/wadaakihiko/homebrew/bin/python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
※ 警告が出ても基本はそのままでOK。消したいときは `MPLCONFIGDIR=./outputs/.mplconfig` を環境変数に設定。

## 2. コマンドで使う方法（CSVを解析）
サンプルデータで試す例:
```bash
./.venv/bin/python app/main.py data/clinical_trial_data.csv \
  --out-dir outputs \
  --group-col arm \
  --effect-cols baseline_score week4_score week8_score ae_grade \
  --html-report outputs/report.html
```
ポイント:
- `--group-col` に群を示す列名を入れると群別の表を出力。
- `--effect-cols` に効果量を出したい列名を並べると、全ペアの効果量・ANOVA・Tukeyを計算（3群以上もOK）。
- `report.html` が作られるのでブラウザで開けば表とグラフをまとめて確認できます。

主な出力（`outputs` 内）:
- `summary_numeric.csv` / `summary_categorical.csv` … 全体の記述統計
- `missing_summary.csv` … 欠測数・率
- `outlier_summary.csv` / `outlier_rows.csv` … 外れ値サマリー/サンプル
- `group_numeric_summary.csv` / `group_categorical_summary.csv` … 群別サマリー
- `effect_sizes.csv` … 全ペアの効果量（数値: Cohen's d、2x2カテゴリ: OR）
- `effect_anova.csv` / `effect_tukey.csv` … 数値列のANOVAとTukey
- `<列名>_plot.png` … 数値列のヒストグラム＋箱ひげ
- `report.html` … 上記をまとめたレポート（プロット埋め込み、欠測/外れ値タブ表示）

## 3. 画面で使う方法（Streamlit）
```bash
cd /Users/wadaakihiko/Desktop/wada_work/Codex_file
./.venv/bin/streamlit run app/streamlit_app.py
```
ブラウザで http://localhost:8504 を開き、CSV/TSV をアップロードするだけで表・グラフ・効果量を確認し、HTMLレポートもダウンロードできます。

- サイドバーのラジオで「設定」「解析」を切替
  - 設定: 区切り/文字コード、欠測処理、プレビュー行数、プロット数などを保存
  - 解析: 保存した設定でファイル読み込みと解析を実行
  - 左下に工程進行状況バッジ（読み込み→前処理→記述/可視化→統計解析）

## 4. よくあるつまずき
- `pip install` がSSLエラーになる: Homebrew版の Python 3.12 を使い、もう一度 `pip install -r requirements.txt` を実行。
- Matplotlibのキャッシュ警告: `MPLCONFIGDIR=./outputs/.mplconfig` を設定。
- 効果量が空になる: グループが1つしかない、または2x2以外のカテゴリ列を指定している可能性。列名とグループ数を確認してください。
