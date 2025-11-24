# 操作マニュアル（医療統計アシスト・プロトタイプ）

## 環境準備
1. 依存インストールと仮想環境
   ```bash
   cd /Users/wadaakihiko/Desktop/wada_work/Codex_file
   /Users/wadaakihiko/homebrew/bin/python3.12 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
   ※ Matplotlibのキャッシュ警告が出る場合は `MPLCONFIGDIR=./outputs/.mplconfig` を環境変数に指定。

## CLIでの実行（記述統計＋可視化＋効果量）
基本形:
```bash
./.venv/bin/python app/main.py <path/to/data.csv> --out-dir outputs \
  --max-plots 6 --max-outlier-rows 100 \
  --group-col <group_column> \
  --effect-cols col1 col2 ...
```
- `--html-report <path>` を付けるとHTMLレポートを生成（省略時は `<out-dir>/report.html`）。
- グループ列を指定するとグループ別サマリーを出力。効果量列を併用すると全ペアの効果量・ANOVA・Tukeyを計算（multi-arm対応）。

## 主な出力ファイル（`--out-dir` 配下）
- `summary_numeric.csv` / `summary_categorical.csv` … 全体の記述統計
- `missing_summary.csv` … 欠測数・率
- `outlier_summary.csv` / `outlier_rows.csv` … IQRによる外れ値サマリー/サンプル
- `group_numeric_summary.csv` / `group_categorical_summary.csv` … グループ別記述統計
- `effect_sizes.csv` … 全ペアの効果量（数値: Cohen's d、2x2カテゴリ: OR）
- `effect_anova.csv` … 数値列の一元配置分散分析
- `effect_tukey.csv` … 数値列のTukey HSD（3群以上）
- `<col>_plot.png` … 数値列のヒストグラム＋箱ひげ
- `report.html` … 上記をまとめたHTMLレポート（プロット埋め込み、欠測/外れ値タブ表示）

## サンプルデータでの例
```bash
./.venv/bin/python app/main.py data/clinical_trial_data.csv \
  --out-dir outputs \
  --group-col arm \
  --effect-cols baseline_score week4_score week8_score ae_grade \
  --html-report outputs/report.html
```
生成された `outputs/report.html` をブラウザで開く。

## Streamlitフロントエンド
```bash
./.venv/bin/streamlit run app/streamlit_app.py
```
- デフォルトポート: 8504（`http://localhost:8504`）
- CSV/TSVアップロード → 記述統計/欠測/外れ値/グループ別/効果量を確認
- PNGプロット表示、HTMLレポートをその場でダウンロード可能

## トラブルシューティング
- SSL関連で `pip install` が失敗する場合: Homebrew版の Python 3.12 を使用し、再度 `pip install -r requirements.txt` を実行。
- Matplotlibのキャッシュ警告: `MPLCONFIGDIR=./outputs/.mplconfig` を設定して再実行。
- 効果量が空になる: グループが1つしかない、または2x2でないカテゴリ列を指定している可能性があります。グループ列と対象列を確認してください。
