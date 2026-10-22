"""Frozen-feature post-hoc analysis and independent CPU verification.

This module never imports legacy experiment code, a model, or a scenario
generator. Only arrays in the explicit frozen input NPZ are consumed.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .math import (ATOL, RTOL, design, finite_array, fit_student, mse,
                   predict, qr_reference, quadratic_mse, risk_decomposition,
                   transfer_ratio)

ARM_NAMES = ("high_plus", "high_minus", "low_plus", "low_minus")
REQUIRED = ("z_train", "z_eval", "y_train", "y_eval", "e_train", "e_eval",
            "gold_coefficient", "gold_train_prediction", "gold_eval_prediction")


def array_hash(value):
    a = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode("ascii"))
    h.update(str(a.shape).encode("ascii"))
    h.update(a.tobytes())
    return h.hexdigest()


def dump_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                     allow_nan=False) + "\n", encoding="utf-8")


def load_frozen(path):
    if isinstance(path, dict):
        a = {key: np.asarray(value).copy() for key, value in path.items()}
    else:
        with np.load(path, allow_pickle=False) as archive:
            a = {key: archive[key].copy() for key in archive.files}
    missing = sorted(set(REQUIRED) - set(a))
    if missing:
        raise ValueError("missing frozen input fields: " + ", ".join(missing))
    # Deterministic schema aliases only; frozen input bytes are not rewritten.
    for split in ("train", "eval"):
        for singular, plural in (("scenario_id", "scenario_ids"),
                                 ("group_id", "group_ids"),
                                 ("canonical_signature", "canonical_signatures")):
            old, new = split + "_" + singular, split + "_" + plural
            if old in a:
                if new in a and not np.array_equal(a[old], a[new]):
                    raise ValueError("conflicting frozen row metadata aliases: " + old)
                a[new] = a[old].copy()
    for key in REQUIRED:
        a[key] = finite_array(a[key], name=key)
    nt, ne = len(a["z_train"]), len(a["z_eval"])
    if a["z_train"].ndim != 2 or a["z_eval"].ndim != 2:
        raise ValueError("frozen features must be matrices")
    if a["z_train"].shape[1] != a["z_eval"].shape[1]:
        raise ValueError("train/evaluation feature count mismatch")
    if a["e_train"].shape != (nt, 2) or a["e_eval"].shape != (ne, 2):
        raise ValueError("error arrays must contain high/low columns")
    for key, n in (("y_train", nt), ("gold_train_prediction", nt),
                   ("y_eval", ne), ("gold_eval_prediction", ne)):
        if a[key].shape != (n,):
            raise ValueError(f"{key}: frozen row mismatch")
    if a["gold_coefficient"].shape != (a["z_train"].shape[1] + 1,):
        raise ValueError("gold coefficient/intercept mismatch")
    for split, n in (("train", nt), ("eval", ne)):
        for suffix in ("scenario_ids", "group_ids", "canonical_signatures"):
            key = split + "_" + suffix
            if key in a and a[key].shape != (n,):
                raise ValueError(f"{key}: ID row mismatch")
        key = split + "_scenario_ids"
        if key in a and len(set(a[key].tolist())) != n:
            raise ValueError(f"{key}: duplicate scenario ID")
    for suffix in ("group_ids", "canonical_signatures"):
        train_key, eval_key = "train_" + suffix, "eval_" + suffix
        if train_key in a and eval_key in a:
            if set(a[train_key].tolist()) & set(a[eval_key].tolist()):
                raise ValueError("train/evaluation overlap: " + suffix)
    return a


def _assert_close(actual, expected, label):
    if not np.allclose(actual, expected, atol=ATOL, rtol=RTOL):
        raise AssertionError(label)


def summarize_arrays(raw):
    y = raw["y_eval"]
    y_train = raw["y_train"]
    r0 = mse(raw["gold_eval_prediction"], y)
    rows = []
    for j, arm in enumerate(ARM_NAMES):
        col, sign = j // 2, 1.0 if j % 2 == 0 else -1.0
        error = sign * raw["e_eval"][:, col]
        target = y + error
        parts = risk_decomposition(raw["gold_eval_prediction"],
                                   raw["eval_predictions"][:, j], y)
        rows.append({
            "arm": arm,
            "truth_risk": mse(raw["eval_predictions"][:, j], y),
            "teacher_agreement_mse": mse(raw["eval_predictions"][:, j], target),
            "teacher_error_mse": float(np.mean(error * error)),
            "train_mse": mse(raw["train_predictions"][:, j],
                             y_train + sign * raw["e_train"][:, col]),
            **parts,
        })
    transfers = []
    for col, name in enumerate(("high", "low")):
        delta = (rows[2 * col]["truth_risk"] + rows[2 * col + 1]["truth_risk"]) / 2 - r0
        energy = float(np.mean(raw["e_eval"][:, col] ** 2))
        ratio, status = transfer_ratio(delta, energy)
        transfers.append({"error": name, "eval_error_mse": energy,
                          "sign_averaged_risk_increase": float(delta),
                          "transfer_ratio_G": ratio, "ratio_status": status})
    gh, gl = (x["transfer_ratio_G"] for x in transfers)
    return {
        "analysis_kind": "CLOSURE_POSTHOC_SINGLE_TRAIN_DATASET",
        "gold": {"truth_risk": r0, "train_mse": mse(raw["gold_train_prediction"], y_train),
                 "teacher_agreement_mse": None},
        "arms": rows, "transfer": transfers,
        "G_high_minus_G_low": None if gh is None or gl is None else gh - gl,
        "eval_rows": int(len(y)), "train_rows": int(len(y_train)),
        "independent_train_datasets": 1,
        "registered_H1": "NOT_RUN", "registered_H1_effect": None,
        "registered_H1_interval": None, "registered_H2": "NOT_RUN",
        "registered_H3": "NOT_RUN",
        "planned": 4, "attempted": 4, "valid": 4, "failed": 0,
        "missing": 0, "gold_reused": 1, "gold_replay_primary": 0,
        "scope_note": "Existing qualification reused post hoc; ratio of means; no hypothesis decision or confidence interval.",
    }


def analyze_frozen(input_path, output):
    """Fit exactly four target heads. Call only after parent freezes protocol."""
    started = time.monotonic()
    a = load_frozen(input_path)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for split in ("train", "eval"):
        _assert_close(predict(a["z_" + split], a["gold_coefficient"]),
                      a["gold_" + split + "_prediction"],
                      "NUMERICAL_BASELINE_MISMATCH: stored gold " + split)
    coefficients, train_predictions, eval_predictions, meta = [], [], [], []
    ledger={"planned":4,"attempted":0,"valid":0,"failed":0,"missing":4,
            "reused_primary":0,"gold_reused":1,"gold_replay_primary":0}
    dump_json(output / "fit_ledger.json",ledger)
    for j, name in enumerate(ARM_NAMES):
        col, sign = j // 2, 1.0 if j % 2 == 0 else -1.0
        target = a["y_train"] + sign * a["e_train"][:, col]
        ledger.update(attempted=j+1,missing=3-j)
        try:
            fitted = fit_student(a["z_train"], target)
        except Exception as error:
            ledger.update(failed=1);dump_json(output / "fit_ledger.json",ledger)
            dump_json(output / "failure.json",{"arm":name,"type":type(error).__name__,"error":str(error)})
            raise
        c = fitted.pop("coefficient")
        singular = fitted.pop("singular_values")
        if fitted["rank_status"] != "FULL_COLUMN_RANK":
            ledger.update(failed=1);dump_json(output / "fit_ledger.json",ledger)
            dump_json(output / "failure.json", {"status": "NUMERICAL_RANK_FAILURE", "arm": name, **fitted})
            raise ValueError("closure head design is not full column rank")
        coefficients.append(c)
        train_predictions.append(predict(a["z_train"], c))
        eval_predictions.append(predict(a["z_eval"], c))
        meta.append({"arm": name, **fitted, "target_array_sha256": array_hash(target),
                     "train_design_sha256":array_hash(a['z_train']),
                     "singular_values": singular.tolist(), "actual_lstsq_calls": 1})
        ledger['valid']=j+1;dump_json(output / "fit_ledger.json",ledger)
    raw = {key: a[key] for key in ("y_train", "y_eval", "e_train", "e_eval",
                                  "gold_train_prediction", "gold_eval_prediction", "gold_coefficient")}
    for key in a:
        if key.endswith("_ids") or key.endswith("_signatures"):
            raw[key] = a[key]
    raw.update(coefficients=np.column_stack(coefficients),
               train_predictions=np.column_stack(train_predictions),
               eval_predictions=np.column_stack(eval_predictions))
    np.savez_compressed(output / "raw.npz", **raw)
    dump_json(output / "fit_metadata.json", {"arm_order": list(ARM_NAMES), "heads": meta,
                                             "primary_lstsq_calls": 4, "primary_target_fit_objects": 4,
                                             "gold_reused": 1, "gold_replay_primary": 0})
    result = summarize_arrays(raw)
    dump_json(output / "summary.json", result)
    dump_json(output / "analysis_execution.json", {
        "status": "COMPLETE", "wall_seconds": time.monotonic() - started,
        "input_npz_sha256": None if isinstance(input_path, dict) else hashlib.sha256(Path(input_path).read_bytes()).hexdigest(),
        "GPU_forward_calls": 0, "new_scenario_generation": 0,
        "original_H1_main_fit_calls": 0, "primary_lstsq_calls": 4,
    })
    return result


def summarize_saved(run):
    """Reaggregate only saved raw arrays; no solver or original cache access."""
    with np.load(Path(run) / "raw.npz", allow_pickle=False) as f:
        raw = {key: f[key] for key in f.files}
    return summarize_arrays(raw)


def verify_saved(input_path, run):
    """Fresh-process entry: direct predictions + independent QR references."""
    a = load_frozen(input_path)
    run = Path(run)
    with np.load(run / "raw.npz", allow_pickle=False) as f:
        raw = {key: f[key] for key in f.files}
    checks = []
    def check(actual, expected, label):
        _assert_close(actual, expected, label)
        checks.append({"name": label, "status": "PASS",
                       "max_abs_difference": float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))})
    for key in ("y_train", "y_eval", "e_train", "e_eval", "gold_coefficient",
                "gold_train_prediction", "gold_eval_prediction"):
        check(raw[key], a[key], "saved raw/input identity: " + key)
    for key in a:
        if key.endswith("_ids") or key.endswith("_signatures"):
            if key not in raw or not np.array_equal(raw[key], a[key]):
                raise AssertionError("saved row ID mismatch: " + key)
            checks.append({"name": "saved row ID exact identity: " + key, "status": "PASS",
                           "max_abs_difference": 0.0})
    # One independent gold solve, one multi-target error-only solve.
    cg = qr_reference(a["z_train"], a["y_train"])
    ce = qr_reference(a["z_train"], a["e_train"])
    check(cg, a["gold_coefficient"], "QR gold coefficient vs original pilot")
    check(predict(a["z_eval"], cg), a["gold_eval_prediction"], "QR gold qualification prediction")
    ke_train, ke_eval = predict(a["z_train"], ce), predict(a["z_eval"], ce)
    for j, name in enumerate(ARM_NAMES):
        sign, col = (1.0 if j % 2 == 0 else -1.0), j // 2
        c = raw["coefficients"][:, j]
        check(predict(a["z_train"], c), raw["train_predictions"][:, j], name + ": saved train coefficient prediction")
        check(predict(a["z_eval"], c), raw["eval_predictions"][:, j], name + ": saved eval coefficient prediction")
        check(cg + sign * ce[:, col], c, name + ": independent QR coefficient linearity")
        check(a["gold_train_prediction"] + sign * ke_train[:, col], raw["train_predictions"][:, j], name + ": train K-error identity")
        check(a["gold_eval_prediction"] + sign * ke_eval[:, col], raw["eval_predictions"][:, j], name + ": qualification K-error identity")
        target = a["y_eval"] + sign * a["e_eval"][:, col]
        check(mse(raw["eval_predictions"][:, j], a["y_eval"]),
              quadratic_mse(a["z_eval"], c, a["y_eval"]), name + ": risk residual vs quadratic")
        check(mse(raw["eval_predictions"][:, j], target),
              quadratic_mse(a["z_eval"], c, target), name + ": agreement residual vs quadratic")
        parts = risk_decomposition(a["gold_eval_prediction"], raw["eval_predictions"][:, j], a["y_eval"])
        check(parts["risk_difference"], parts["prediction_change_energy"] + parts["signed_cross_term"], name + ": signed risk decomposition")
    summary = summarize_arrays(raw)
    for col, name in enumerate(("high", "low")):
        check(summary["transfer"][col]["sign_averaged_risk_increase"],
              np.mean(ke_eval[:, col] ** 2), name + ": sign-average independent K-error energy")
    check(ke_train[:, 0], a["e_train"][:, 0], "high error train full-span representation")
    check(ke_eval[:, 0], a["e_eval"][:, 0], "high error heldout full-span extension")
    check(summary["transfer"][0]["transfer_ratio_G"], 1.0, "high error G=1 known linear identity")
    stored_summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    if summary != stored_summary:
        raise AssertionError("raw-only summary differs from stored summary")
    checks.append({"name": "raw-only summary exact Python values", "status": "PASS", "max_abs_difference": 0.0})
    return {"status": "PASS", "checks": checks, "check_count": len(checks),
            "atol": ATOL, "rtol": RTOL,
            "reference_QR_factorizations": 2, "reference_solve_calls": 2,
            "reference_target_fit_objects": 3,
            "primary_refits": 0, "new_scenarios": 0, "GPU_forward_calls": 0,
            "claim": "Numerical implementation verification, not registered H1 evidence."}
