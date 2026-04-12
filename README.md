# citypop-fadeout-analyzer

日本のポップス (シティポップ等) の「終わり方」を音声信号から自動分類するCLIツールです。

楽曲の末尾区間のRMS減衰パターンとオンセット自己相関を解析し、以下の3クラスに分類します。

| 分類 | 判定の概要 |
|------|-----------|
| **Fade-out** | テール区間の音量が緩やかに減衰してゼロに向かう |
| **明確終止 (Cold ending)** | 最後の瞬間まで本体と同等の音量で鳴り、短い残響の後に終わる |
| **リフ反復終止** | コード進行やリフを繰り返しながら明確に終わる |

## 必要なもの

- Python 3.12
- [uv](https://docs.astral.sh/uv/) (パッケージマネージャ)
- 分析対象の音源ファイル (MP3 / FLAC / WAV / M4A / OGG)

## セットアップ

```bash
git clone https://github.com/okamyuji/citypop-fadeout-analyzer.git
cd citypop-fadeout-analyzer
uv sync
```

## 使い方

```bash
# 1. audioディレクトリに自分の音源を配置
mkdir -p audio
cp /path/to/your/music/*.mp3 audio/

# 2. 全音源を解析してCSVに出力
uv run citypop-analyze analyze --audio-dir ./audio --out ./out/results.csv

# 3. 統計サマリを表示
uv run citypop-analyze report --csv ./out/results.csv

# 4. 可視化グラフを生成
uv run citypop-analyze visualize --csv ./out/results.csv --out-dir ./out/figures
```

ID3タグが含まれている音源はアーティスト名・タイトル・リリース年を自動取得します。タグがない場合はファイル名から推測を試みます。

## 出力例

129曲を分析した結果:

```
total: 129
      Fade-out:  106 ( 82.2%)
      明確終止:   17 ( 13.2%)
    リフ反復終止:    6 (  4.7%)
```

## 分析の仕組み

1. librosaで音源を22,050 Hzモノラルとして読み込む
2. フレーム単位のRMS (dB) を算出し、楽曲の「実効的な終端」を推定する
3. 終端から遡って12秒間のテール窓を取り出す
4. テール窓のRMS傾き (dB/秒) と、本体音量からの減衰量でFade-outを判定する
5. オンセット強度の自己相関でリフ反復性を検出する

## 品質ゲート

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy src tests
uv run pytest
uv build
```

## ライセンス

MIT License (c) 2026 okamyuji
