"""Run one EOD live cycle for a venue (IBKR or Xueqiu). Dry-run by default."""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--venue", choices=["ibkr", "xueqiu"], required=True)
    ap.add_argument("--weights", required=True, help="JSON file of target weights")
    ap.add_argument("--dry-run", action="store_true",
                    help="force dry-run (LiveGuard still enforces its two-factor gate)")
    ap.add_argument("--log-dir", default="~/.tactis/app-logs")
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from pipeline import load_target_weights, run_venue

    print(f"[live {datetime.now(timezone.utc).strftime('%H:%M:%S')}] "
          f"run_live start  venue={args.venue}  weights_file={args.weights}  "
          f"dry_run={args.dry_run}", flush=True)
    weights = load_target_weights(args.weights)
    log_dir = Path(args.log_dir).expanduser()
    # pipeline.run_venue handles guard logic internally via LiveGuard
    return run_venue(args.venue, args.dry_run, weights, log_dir,
                     weights_source=Path(args.weights).name)


if __name__ == "__main__":
    sys.exit(main())
