## Disclaimer

Codes generated using LLM.

## Custom Data Download

Download US and Chinese A-share daily OHLCV + fundamentals.

### Dependencies

```bash
pip install yfinance baostock pandas
```

### Usage

**1. Provide a ticker list** — pick one method:

| Method | Example |
|--------|---------|
| Edit `config.py` | `US_TICKERS = ["AAPL", "MSFT", "GOOGL"]` |
| CLI args | `--tickers AAPL MSFT GOOGL` |
| CSV file | `--tickers-csv nasdaq100.csv` |

The CSV is column-flexible — it auto-detects `ticker`, `code`, or `symbol` columns, falling back to the first column.

**2. Download:**

```bash
python download_us.py                          # uses config.US_TICKERS
python download_us.py --tickers-csv nasdaq.csv
python download_us.py --tickers AAPL MSFT GOOGL

python download_cn.py                          # uses config.CN_TICKERS
python download_cn.py --tickers-csv hs300.csv
python download_cn.py --tickers sh.600519 sz.000858
```

Flags: `--start 2015-01-01 --end 2023-12-31 --output combined.csv`

**3. Merge:**

```bash
python merge.py                                # -> raw/all_stocks.csv
```

### Output

```
raw/
├── us/              # one CSV per ticker: date,ticker,open,high,low,close,volume,market_cap,sector,industry
├── cn/              # one CSV per ticker: date,ticker,open,high,low,close,volume,name,industry
└── all_stocks.csv   # unified dataset (after merge)
```

### Ticker Formats

- **US**: standard Yahoo symbols — `AAPL`, `BRK-B`, `^GSPC`
- **CN (baostock)**: `sh.600519` (Shanghai), `sz.000858` (Shenzhen)
