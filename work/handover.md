# 引き継ぎメモ（明日用）

## 状態
- `main` ブランチ最新。Streamlitの設定/解析切替や進行バッジ、欠測前処理、検定/効果量など動作。
- 開発ログ: `work/dev_log.md`、マニュアル: `work/manual.md`。
- サンプルデータ: 2群/多群/欠測/生存解析など各種あり。

## 未着手/検討中の次ステップ案
- 検定・効果量の解釈コメント（HTML/Streamlitへの短文追加）。
- Streamlitのタブ整理（概要/検定/効果量/外れ値/グラフ）。
- 追加解析: U検定、Fisher、簡易KM/Coxサマリー。
- 生成物のメタ情報表示（入力ファイル名、生成日時、設定の表示）。

## 注意点
- Streamlit起動はプロジェクト直下で: `cd /Users/wadaakihiko/Desktop/wada_work/Codex_file && ./.venv/bin/streamlit run app/streamlit_app.py`
- Matplotlibキャッシュ警告が出る場合は `MPLCONFIGDIR=./outputs/.mplconfig` を設定。
- 解析結果は `outputs/` に出力、HTMLレポートは `report.html`。
