# 医療統計アシスト・プロトタイプ

CSVを読み込み、記述統計と簡単な可視化を出力する最小プロトタイプです。

## セットアップ

```bash
cd Codex_file
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 使い方

```bash
python app/main.py <path/to/data.csv> --out-dir outputs
```

- `summary_numeric.csv` と `summary_categorical.csv` を `--out-dir` に保存します。
- 数値列のヒストグラム+ボックスプロットを最大6列までPNGで保存します（`--max-plots` で変更）。
- `outputs` ディレクトリは自動作成されます。

### サンプルデータ

`data/sample_medical.csv` を同梱しています。試す場合:

```bash
python app/main.py data/sample_medical.csv --out-dir outputs
```

欠測を含む例が必要な場合は `data/sample_medical_incomplete.csv` を利用できます。

```bash
python app/main.py data/sample_medical_incomplete.csv --out-dir outputs
```

追加サンプルデータ:
- `data/survival_analysis_data.csv` … 生存解析用 (event_time_months, event_observed など)
- `data/patient_outcomes.csv` … バイタル/血糖・再入院・死亡アウトカム
- `data/clinical_trial_data.csv` … 臨床試験の多腕データ (placebo/low/high)

## 次の拡張候補

- 欠測・外れ値検知のレポート化
- グループ別記述統計（例: 介入/対照）
- サンプルサイズ計算とレポートHTML化
