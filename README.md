# Smart Signal — XAUUSD GoldNet

Deep-learning **buy / sell / hold** engine for gold (XAUUSD). It is not a single LSTM or a tree model. GoldNet is a multi-timeframe network:

1. **Variable Selection Network** (Temporal Fusion Transformer style) — learns which technical features matter on each bar.
2. **Dilated causal TCN** — local / multi-scale candle patterns without peeking into the future inside the window.
3. **Transformer encoder** per timeframe — longer-range structure on 15m, 1h, 4h, and 1d.
4. **Hierarchical cross-attention** — 15m queries 1h / 4h / 1d so higher-timeframe bias conditions the entry.
5. **Regime GRU** on the fused 15m sequence.
6. **Multi-task heads** — direction (sell/hold/buy), expected log-return, and move size.

Labels use **triple-barrier** labeling (Lopez de Prado): ATR take-profit vs stop vs time barrier.

## Data

Live and historical FOREXCOM XAUUSD candles already sit on the gold VPS (`cp_fetcher` → MongoDB `historical_data.xauusd_1m` / `_1h` / daily). Public endpoints used by this repo:

| Source | URL |
| --- | --- |
| Last price | `http://185.222.163.116/crypto-api/prices/xauusd/?timeframe=1m` |
| History | `http://185.222.163.116/crypto-api/prices/xauusd/history/?timeframe=1m&limit=2000` |
| Fast 1m bars | `http://185.222.163.116/trh-api/bars?limit=2000` |

FOREXCOM 1-minute history on the VPS is roughly two weeks. Training therefore **pretrains on COMEX gold futures (`GC=F`)** (multi-year hourly + 60d 15m) and applies the same model to live FOREXCOM XAUUSD. Directional structure transfers; the live path always reads the VPS.

The VPS is 1 vCPU / 2 GB RAM. **Train on a larger machine** (this repo’s default). Inference of the ~1.0M-parameter GoldNet checkpoint is light enough to serve.

Do **not** commit SSH passwords. Put secrets only in a local `.env` (see `.env.example`).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

```bash
python -m smart_signal fetch
python -m smart_signal train
python -m smart_signal backtest
python -m smart_signal signal
python -m smart_signal serve --port 8080
```

Open `http://127.0.0.1:8080/` for the dashboard. `GET /signal` returns the live decision:

```json
{
  "symbol": "XAUUSD",
  "timeframe": "15m",
  "signal": "BUY",
  "confidence": 0.61,
  "price": 4430.29,
  "take_profit": 4448.1,
  "stop_loss": 4419.4
}
```

## Layout

```
configs/goldnet.yaml     model + label + train hyperparameters
src/smart_signal/models  GoldNet (TCN / Transformer / fusion)
src/smart_signal/data    FOREXCOM + Yahoo ingest, MTF windows
src/smart_signal/labels  triple-barrier
src/smart_signal/api.py  FastAPI
web/index.html           live terminal
```

## Tests

```bash
python -m pytest -q
```
