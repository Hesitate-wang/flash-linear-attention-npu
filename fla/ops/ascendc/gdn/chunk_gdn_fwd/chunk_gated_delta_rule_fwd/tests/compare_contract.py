#!/usr/bin/env python3
"""Host-only contract tests for the lightweight mixed comparison rule."""

from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lightweight_gdn_fwd import error_metrics

CONFIG = {"diff_thd": 1e-3, "pct_thd": 1e-4, "max_diff_hd": 0.1}


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    identical = error_metrics(torch.zeros(4), torch.zeros(4), CONFIG)
    check(identical["passed"] and identical["fulfill_percent"] == 100.0, "identical tensors")

    small = error_metrics(torch.tensor([5e-5]), torch.tensor([0.0]), CONFIG)
    check(small["passed"], "small absolute error must pass")

    relative = error_metrics(torch.tensor([1.01]), torch.tensor([1.0]), CONFIG)
    check(not relative["passed"] and relative["max_error"] > 1e-4,
          "large relative error must fail")

    one_bad = error_metrics(torch.ones(10000), torch.ones(10000), CONFIG)
    check(one_bad["passed"], "baseline must pass")
    actual = torch.ones(10000)
    actual[0] = 1.01
    one_bad = error_metrics(actual, torch.ones(10000), CONFIG)
    check(one_bad["passed"], "one failure in 10000 is allowed at 99.99%")
    actual[1] = 1.01
    two_bad = error_metrics(actual, torch.ones(10000), CONFIG)
    check(not two_bad["passed"], "two failures in 10000 exceed allowed fraction")

    nan_case = error_metrics(torch.tensor([float("nan")]), torch.tensor([0.0]), CONFIG)
    check(not nan_case["passed"] and not nan_case["finite"], "NaN must fail")
    inf_case = error_metrics(torch.tensor([0.0]), torch.tensor([float("inf")]), CONFIG)
    check(not inf_case["passed"] and not inf_case["finite"], "Inf must fail")

    print("mixed comparison contract: passed")


if __name__ == "__main__":
    main()
