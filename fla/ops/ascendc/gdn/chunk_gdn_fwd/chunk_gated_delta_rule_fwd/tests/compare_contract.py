#!/usr/bin/env python3
"""Online element-by-element comparison used by the lightweight NPU tests."""

from __future__ import annotations

import math
from pathlib import Path
import sys

import torch


def _log(logger, message):
    if logger is None:
        print(message)
    else:
        logger.info("%s", message)


def _nonfinite_metrics(count):
    return {"max_abs": math.inf, "mean_abs": math.inf, "rmse": math.inf,
            "max_rel": math.inf, "mean_rel": math.inf,
            "mismatch_count": count, "mismatch_ratio": 1.0 if count else 0.0,
            "fulfill_percent": 0.0, "max_error": math.inf,
            "passed": False, "finite": False}


def data_compare(actual: torch.Tensor, expected: torch.Tensor, config: dict,
                 logger=None, label: str = "output") -> dict:
    """Compare online tensors and print data_compare_h-style details.

    No files are read. Both tensors are converted to CPU FP32 for comparison.
    """
    if actual.shape != expected.shape:
        raise ValueError(f"{label}: shape mismatch actual={tuple(actual.shape)}, expected={tuple(expected.shape)}")
    actual = actual.detach().cpu().float().flatten()
    expected = expected.detach().cpu().float().flatten()
    count = actual.numel()
    if not bool(torch.isfinite(actual).all() and torch.isfinite(expected).all()):
        _log(logger, f"{label}: NaN/Inf detected; result=Failed")
        return _nonfinite_metrics(count)

    diff_thd = config["diff_thd"]
    pct_thd = config["pct_thd"]
    max_diff_hd = config["max_diff_hd"]
    difference = (actual - expected).abs()
    floor = (1.0 / (1 << 14)) / diff_thd
    mixed_denominator = torch.maximum(actual.abs(), expected.abs()).clamp_min(floor) + 1e-9
    mixed = torch.where(difference < diff_thd, difference, difference / mixed_denominator)
    failures = mixed > diff_thd
    error_count = int(failures.sum())
    fulfill_percent = 100.0 * (count - error_count) / count if count else 100.0
    max_error = float(mixed[failures].max()) if error_count else 0.0
    passed = fulfill_percent >= (1.0 - pct_thd) * 100.0 and max_error < max_diff_hd
    relative = difference / expected.abs().clamp_min(1e-6)
    worst = int(difference.argmax()) if count else 0

    _log(logger, "---------------------------------------------------------------------------------------")
    _log(logger, f"{label}: Loop                 ExpectOut          RealOut            FpDiff             RateDiff")
    _log(logger, "---------------------------------------------------------------------------------------")
    if count <= 20:
        indices = list(range(count))
    else:
        indices = list(range(10)) + list(range(count - 10, count))
    for index in indices:
        _log(logger, "%08d         %.7f         %.7f         %.7f         %.7f" % (
            index + 1, float(expected[index]), float(actual[index]),
            float(difference[index]), float(mixed[index])))
    if error_count:
        _log(logger, "Error Line-----------------------------------------------------------------------------")
        error_indices = torch.nonzero(failures, as_tuple=False).flatten()[:100].tolist()
        for index in error_indices:
            _log(logger, "%08d         %.7f         %.7f         %.7f         %.7f" % (
                index + 1, float(expected[index]), float(actual[index]),
                float(difference[index]), float(mixed[index])))
    result = "Pass" if passed else "Failed"
    _log(logger, "---------------------------------------------------------------------------------------")
    _log(logger, "DiffThd          PctThd           PctRlt           Result")
    _log(logger, "---------------------------------------------------------------------------------------")
    _log(logger, "%.6f         %.4f%%         %.6f%%         %s" % (
        diff_thd, (1.0 - pct_thd) * 100.0, fulfill_percent, result))
    if error_count:
        _log(logger, "Maximum error is: %s. Tolerance threshold is: %s." % (max_error, max_diff_hd))

    return {
        "max_abs": float(difference.max()) if count else 0.0,
        "mean_abs": float(difference.mean()) if count else 0.0,
        "rmse": float(difference.square().mean().sqrt()) if count else 0.0,
        "max_rel": float(relative.max()) if count else 0.0,
        "mean_rel": float(relative.mean()) if count else 0.0,
        "mismatch_count": error_count,
        "mismatch_ratio": error_count / count if count else 0.0,
        "fulfill_percent": fulfill_percent,
        "max_error": max_error,
        "worst_flat_index": worst,
        "worst_actual": float(actual[worst]) if count else 0.0,
        "worst_expected": float(expected[worst]) if count else 0.0,
        "passed": passed,
        "finite": True,
    }


class _QuietLogger:
    @staticmethod
    def info(*args, **kwargs):
        return None


def main():
    config = {"diff_thd": 1e-3, "pct_thd": 1e-4, "max_diff_hd": 0.1}
    logger = _QuietLogger()
    check(data_compare(torch.zeros(4), torch.zeros(4), config, logger)["passed"], "identical")
    check(data_compare(torch.tensor([5e-5]), torch.tensor([0.0]), config, logger)["passed"], "absolute")
    check(not data_compare(torch.tensor([1.01]), torch.tensor([1.0]), config, logger)["passed"], "relative")
    actual = torch.ones(10000)
    actual[0] = 1.01
    check(data_compare(actual, torch.ones(10000), config, logger)["passed"], "one bad in 10000")
    actual[1] = 1.01
    check(not data_compare(actual, torch.ones(10000), config, logger)["passed"], "two bad in 10000")
    check(not data_compare(torch.tensor([float("nan")]), torch.tensor([0.0]), config, logger)["passed"], "NaN")
    check(not data_compare(torch.tensor([0.0]), torch.tensor([float("inf")]), config, logger)["passed"], "Inf")
    print("mixed comparison contract: passed")


def check(condition, message):
    if not condition:
        raise AssertionError(message)


if __name__ == "__main__":
    main()
