"""
Fetch index component tickers and save as CSV files in components/.

Usage:  python fetch_components.py            # all indices
        python fetch_components.py --index sp500,nasdaq100
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import baostock as bs

OUT_DIR = Path(__file__).resolve().parent / "components"

# ---- Static fallback for NASDAQ-100 (as of 2025) ----
NASDAQ_100 = [
    ("AAPL", "Apple Inc."), ("MSFT", "Microsoft Corp."), ("AMZN", "Amazon.com Inc."),
    ("NVDA", "NVIDIA Corp."), ("GOOGL", "Alphabet Inc. (Class A)"),
    ("GOOG", "Alphabet Inc. (Class C)"), ("META", "Meta Platforms Inc."),
    ("AVGO", "Broadcom Inc."), ("COST", "Costco Wholesale Corp."),
    ("TSLA", "Tesla Inc."), ("NFLX", "Netflix Inc."), ("ADBE", "Adobe Inc."),
    ("PEP", "PepsiCo Inc."), ("TMUS", "T-Mobile US Inc."), ("CSCO", "Cisco Systems Inc."),
    ("AMD", "Advanced Micro Devices Inc."), ("LIN", "Linde plc"),
    ("INTU", "Intuit Inc."), ("QCOM", "QUALCOMM Inc."), ("TXN", "Texas Instruments Inc."),
    ("ISRG", "Intuitive Surgical Inc."), ("AMGN", "Amgen Inc."),
    ("CMCSA", "Comcast Corp."), ("HON", "Honeywell International Inc."),
    ("AMAT", "Applied Materials Inc."), ("BKNG", "Booking Holdings Inc."),
    ("VRTX", "Vertex Pharmaceuticals Inc."), ("ADP", "Automatic Data Processing Inc."),
    ("PANW", "Palo Alto Networks Inc."), ("SBUX", "Starbucks Corp."),
    ("ADI", "Analog Devices Inc."), ("REGN", "Regeneron Pharmaceuticals Inc."),
    ("MU", "Micron Technology Inc."), ("LRCX", "Lam Research Corp."),
    ("KLAC", "KLA Corp."), ("MDLZ", "Mondelez International Inc."),
    ("MELI", "MercadoLibre Inc."), ("GILD", "Gilead Sciences Inc."),
    ("INTC", "Intel Corp."), ("SNPS", "Synopsys Inc."), ("CDNS", "Cadence Design Systems Inc."),
    ("CTAS", "Cintas Corp."), ("ASML", "ASML Holding N.V."),
    ("PYPL", "PayPal Holdings Inc."), ("MAR", "Marriott International Inc."),
    ("ORLY", "O'Reilly Automotive Inc."), ("PDD", "PDD Holdings Inc."),
    ("ABNB", "Airbnb Inc."), ("CRWD", "CrowdStrike Holdings Inc."),
    ("CSX", "CSX Corp."), ("WDAY", "Workday Inc."), ("FTNT", "Fortinet Inc."),
    ("CEG", "Constellation Energy Corp."), ("NXPI", "NXP Semiconductors N.V."),
    ("MRVL", "Marvell Technology Inc."), ("ROP", "Roper Technologies Inc."),
    ("AEP", "American Electric Power Co. Inc."), ("ADSK", "Autodesk Inc."),
    ("ROST", "Ross Stores Inc."), ("MNST", "Monster Beverage Corp."),
    ("DASH", "DoorDash Inc."), ("CPRT", "Copart Inc."), ("KDP", "Keurig Dr Pepper Inc."),
    ("WBD", "Warner Bros. Discovery Inc."), ("AZN", "AstraZeneca PLC"),
    ("PCAR", "PACCAR Inc."), ("PAYX", "Paychex Inc."),
    ("ODFL", "Old Dominion Freight Line Inc."), ("CHTR", "Charter Communications Inc."),
    ("KHC", "The Kraft Heinz Co."), ("DDOG", "Datadog Inc."),
    ("FAST", "Fastenal Co."), ("EA", "Electronic Arts Inc."),
    ("EXC", "Exelon Corp."), ("GEHC", "GE HealthCare Technologies Inc."),
    ("VRSK", "Verisk Analytics Inc."), ("CTSH", "Cognizant Technology Solutions Corp."),
    ("BKR", "Baker Hughes Co."), ("FANG", "Diamondback Energy Inc."),
    ("LULU", "Lululemon Athletica Inc."), ("IDXX", "IDEXX Laboratories Inc."),
    ("CCEP", "Coca-Cola Europacific Partners PLC"), ("TTD", "The Trade Desk Inc."),
    ("CDW", "CDW Corp."), ("XEL", "Xcel Energy Inc."),
    ("ON", "ON Semiconductor Corp."), ("TEAM", "Atlassian Corp."),
    ("CSGP", "CoStar Group Inc."), ("MCHP", "Microchip Technology Inc."),
    ("ZS", "Zscaler Inc."), ("ANSS", "ANSYS Inc."),
    ("DXCM", "DexCom Inc."), ("TTWO", "Take-Two Interactive Software Inc."),
    ("GFS", "GLOBALFOUNDRIES Inc."), ("ARM", "Arm Holdings plc"),
    ("MDB", "MongoDB Inc."), ("ILMN", "Illumina Inc."),
    ("WBA", "Walgreens Boots Alliance Inc."), ("BIIB", "Biogen Inc."),
    ("DLTR", "Dollar Tree Inc."), ("SMCI", "Super Micro Computer Inc."),
]

INDICES = {
    "sp500": {
        "url": "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv",
        "ticker_col": "Symbol",
        "name_col": "Security",
        "output": "sp500.csv",
    },
    "nasdaq100": {"output": "nasdaq100.csv"},
    "hs300": {"output": "hs300.csv"},
    "zz500": {"output": "zz500.csv"},
}


def fetch_csv_source(index_key: str, spec: dict) -> pd.DataFrame:
    print(f"  [{index_key}] Fetching from {spec['url']} ...")
    raw = pd.read_csv(spec["url"])
    result = pd.DataFrame()
    result["ticker"] = raw[spec["ticker_col"]].astype(str).str.strip()
    result["name"] = raw[spec["name_col"]].astype(str).str.strip()
    for col in spec.get("extra_cols", []):
        if col in raw.columns:
            result[col.lower().replace(" ", "_")] = raw[col]
    result = result.dropna(subset=["ticker"])
    result["ticker"] = result["ticker"].str.replace(".", "-", regex=False)
    return result


def fetch_baostock_components(index_key: str, _spec: dict) -> pd.DataFrame:
    bs.login()

    query_funcs = {
        "hs300": bs.query_hs300_stocks,
        "zz500": bs.query_zz500_stocks,
    }
    func = query_funcs.get(index_key)
    if func is None:
        raise ValueError(f"Unknown baostock index: {index_key}")

    rs = func()
    records = []
    while rs.next():
        records.append(rs.get_row_data())

    bs.logout()

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records, columns=["date", "code", "name"])
    df = df.drop(columns=["date"])
    df["code"] = df["code"].apply(_baostock_code_to_ticker)
    return df.rename(columns={"code": "ticker"})


def _baostock_code_to_ticker(code: str) -> str:
    if code.startswith("6"):
        return f"sh.{code}"
    elif code.startswith(("0", "3")):
        return f"sz.{code}"
    return code


def fetch_nasdaq100() -> pd.DataFrame:
    print("  [nasdaq100] Using embedded list ...")
    return pd.DataFrame(NASDAQ_100, columns=["ticker", "name"])


def main():
    parser = argparse.ArgumentParser(description="Fetch index component ticker lists")
    parser.add_argument(
        "--index", default=None,
        help="Comma-separated indices to fetch (sp500,nasdaq100,hs300,zz500). Default: all."
    )
    args = parser.parse_args()

    if args.index:
        selected = [k.strip() for k in args.index.split(",")]
        unknown = set(selected) - set(INDICES)
        if unknown:
            print(f"Unknown indices: {unknown}. Available: {list(INDICES)}")
            sys.exit(1)
    else:
        selected = list(INDICES)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for key in selected:
        spec = INDICES[key]
        out_path = OUT_DIR / spec["output"]
        try:
            if key == "nasdaq100":
                df = fetch_nasdaq100()
            elif "url" in spec:
                df = fetch_csv_source(key, spec)
            else:
                df = fetch_baostock_components(key, spec)

            if df.empty:
                print(f"  [{key}] No data. Skipping.")
                continue

            df.to_csv(out_path, index=False)
            print(f"  [{key}] {len(df)} tickers -> {out_path}")
        except Exception as e:
            print(f"  [{key}] Error: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
