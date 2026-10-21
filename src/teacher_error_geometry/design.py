"""Explicit cell IDs and shared dataset groups, in config list order."""
import copy
import numpy as np
from .model import teacher_delta


def validate_config(config):
    if config["repetitions"] < 2 or int(config["repetitions"]) != config["repetitions"]:
        raise ValueError("At least two integer repetitions required")
    if config["batch_size"] < 1 or config["wall_limit_seconds"] <= 0:
        raise ValueError("Invalid execution budget")
    if config["dtype"] != "float64":
        raise ValueError("Only float64 supported")
    for block in "ABC":
        if len(set(config[block]["rho"])) != len(config[block]["rho"]):
            raise ValueError("Duplicate rho")
        if len(set(config[block]["n"])) != len(config[block]["n"]):
            raise ValueError("Duplicate n")
        for n in config[block]["n"]:
            if int(n) != n or n < 5:
                raise ValueError("Sweep requires integer n>=5 for finite risk variance")
    if not set(config["B"]["rho"]).issubset(config["A"]["rho"]) or not set(config["B"]["n"]).issubset(config["A"]["n"]):
        raise ValueError("B must reference existing A dataset groups")
    if config["A"]["teachers"] != ["axis_visible", "axis_hidden"] or config["B"]["teachers"] != ["projection_zero"]:
        raise ValueError("Unexpected teacher mapping")
    if len(set(config["C"]["theta"])) != len(config["C"]["theta"]):
        raise ValueError("Duplicate theta")
    cells, groups = design(config)
    if len({c["config_id"] for c in cells}) != len(cells):
        raise ValueError("Nonunique config IDs")
    return cells, groups


def design(config):
    cells, groups = [], []
    for data_block, name in enumerate(("A", "C")):
        for ri, rho in enumerate(config[name]["rho"]):
            for ni, n in enumerate(config[name]["n"]):
                group = {"data_block": data_block, "rho_index": ri, "n_index": ni,
                         "rho": rho, "n": n, "group_id": f"d{data_block}_r{ri}_n{ni}", "cells": []}
                candidates = []
                if name == "A":
                    candidates.extend(("A", t, None, f"A_r{ri}_n{ni}_{t}") for t in config["A"]["teachers"])
                    if rho in config["B"]["rho"] and n in config["B"]["n"]:
                        candidates.append(("B", "projection_zero", None, f"B_r{ri}_n{ni}_projection_zero"))
                else:
                    candidates.extend(("C", "normalized_angle", theta, f"C_r{ri}_n{ni}_t{ti}")
                                      for ti, theta in enumerate(config["C"]["theta"]))
                for block, kind, theta, cid in candidates:
                    delta = teacher_delta(kind, rho, config["c"], theta)
                    cell = {"config_id": cid, "block": block, "teacher": kind,
                            "rho": rho, "n": n, "c": config["c"], "theta": theta,
                            "delta1": float(delta[0]), "delta2": float(delta[1]),
                            "group_id": group["group_id"]}
                    cells.append(cell)
                    group["cells"].append(cell)
                groups.append(group)
    return cells, groups


def small_config(config, repetitions=20):
    result = copy.deepcopy(config)
    result["experiment"] = "P0-pilot-NOT-FULL"
    result["repetitions"] = repetitions
    result["planned"] = {"cells": 84, "fit_objects": 84 * repetitions,
                         "unique_training_datasets": 31 * repetitions}
    return result
