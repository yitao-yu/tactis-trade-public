"""Xueqiu cookie: retrieve into env for the runner (never printed)."""
from __future__ import annotations

import argparse
import os
import sys


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--print-name", action="store_true",
                    help="print the SSM param name only (never the value)")
    args = ap.parse_args()
    if args.print_name:
        print("/tactis/xueqiu_cookie")
        return 0
    from creds import get_secret
    cookie = get_secret("xueqiu_cookie")
    if not cookie:
        print("xueqiu_cookie not resolvable", file=sys.stderr)
        return 1
    # give it to the process env for the shell that sourced us; typical use:
    #   export XUEQIU_COOKIE=$(python3 app/xueqiu/fetch_cookie.py)
    print(cookie)  # shell substitution only; do not copy into logs
    return 0


if __name__ == "__main__":
    sys.exit(main())
