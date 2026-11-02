"""Bind frozen diagnostics to a native historical D-only checkpoint.

This command never generates a training prepared dataset or relabels anatomy.
"""
from __future__ import annotations
import argparse
from consistent_cdm.runtime import prepare_historical_diagnostics, verify_prepared


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--calibration', required=True, help='Existing train-only calibration JSON')
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    result = prepare_historical_diagnostics(args.checkpoint, args.calibration, args.output)
    verify_prepared(result)
    print(f"Bound {len(result['split']['val'])} held-out patients; historical stored order remains unverified anatomy")


if __name__ == '__main__':
    main()
