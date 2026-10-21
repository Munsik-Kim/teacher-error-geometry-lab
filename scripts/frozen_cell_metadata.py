"""Use recorded teacher parameters after checking their mathematical design.

Raw retains exact equality to the frozen cell plan. Recomputed derived delta
values use the original FP64 tolerance across CPU/BLAS arithmetic differences.
"""
import copy
import math
import numpy as np


def validated_frozen_plan(runtime_cells, runtime_groups, recorded_cells, tolerance):
    if len(runtime_cells) != len(recorded_cells):
        raise ValueError("Frozen/runtime cell count differs")
    fixed = copy.deepcopy(recorded_cells)
    differences = []
    for current, saved in zip(runtime_cells, fixed):
        if current.keys() != saved.keys():
            raise ValueError("Frozen/runtime cell schema differs")
        for key in saved:
            if key in ("delta1", "delta2"):
                a, b = float(current[key]), float(saved[key])
                if not math.isfinite(a) or not math.isfinite(b) or not np.isclose(a, b, **tolerance):
                    raise ValueError("Frozen delta disagrees with configured design: " + saved["config_id"] + "/" + key)
                if a != b:
                    differences.append({"config_id": saved["config_id"], "field": key,
                                        "recomputed": a, "recorded": b, "difference": a-b})
            elif current[key] != saved[key]:
                raise ValueError("Frozen literal metadata differs: " + str(key))
    by_id = {cell["config_id"]: cell for cell in fixed}
    if len(by_id) != len(fixed):
        raise ValueError("Duplicate frozen cell ID")
    groups = copy.deepcopy(runtime_groups)
    for group in groups:
        group["cells"] = [by_id[cell["config_id"]] for cell in group["cells"]]
        if any(cell["group_id"] != group["group_id"] for cell in group["cells"]):
            raise ValueError("Frozen group pairing differs")
    receipt = {"mode": "exact raw-to-recorded metadata; recomputed design checked at original tolerance",
               "tolerance": dict(tolerance), "cells_checked": len(fixed), "runtime_differences": differences,
               "max_absolute_recomputed_delta_difference": max((abs(d["difference"]) for d in differences), default=0.),
               "historical_raw_equality_check": "UNCHANGED_EXACT_EQUALITY"}
    return fixed, groups, receipt
