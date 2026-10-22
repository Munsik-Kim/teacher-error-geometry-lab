"""Small synthetic arrays only; no project generator or official split access."""
import unittest
from pathlib import Path
import tempfile

import numpy as np

from teg_closeout.math import (design, fit_student, mse, predict, qr_reference,
                              quadratic_mse, risk_decomposition, transfer_ratio)
from teg_closeout.analysis import analyze_frozen, load_frozen, summarize_saved, verify_saved


class CloseoutMathTests(unittest.TestCase):
    def setUp(self):
        # Pure arrays in a namespace independent of every scientific data stream.
        rng = np.random.default_rng(715013)
        self.zt = rng.uniform(-2, 2, (80, 4))
        self.ze = rng.uniform(-2, 2, (31, 4))
        self.yt = np.sin(self.zt[:, 0]) + self.zt[:, 1] ** 2
        self.ye = np.sin(self.ze[:, 0]) + self.ze[:, 1] ** 2
        self.et = np.column_stack((0.3 + self.zt[:, 2], np.cos(self.zt[:, 0] * self.zt[:, 3])))
        self.ee = np.column_stack((0.3 + self.ze[:, 2], np.cos(self.ze[:, 0] * self.ze[:, 3])))

    def test_lstsq_and_independent_qr(self):
        ls = fit_student(self.zt, self.yt)
        self.assertEqual(ls["rank_status"], "FULL_COLUMN_RANK")
        np.testing.assert_allclose(ls["coefficient"], qr_reference(self.zt, self.yt), atol=1e-10, rtol=1e-8)

    def test_signed_risk_and_quadratic(self):
        c0 = fit_student(self.zt, self.yt)["coefficient"]
        c1 = fit_student(self.zt, self.yt + self.et[:, 1])["coefficient"]
        f0, f1 = predict(self.ze, c0), predict(self.ze, c1)
        p = risk_decomposition(f0, f1, self.ye)
        self.assertAlmostEqual(p["risk_difference"], p["prediction_change_energy"] + p["signed_cross_term"], places=12)
        self.assertNotEqual(p["signed_cross_term"], 0.0)
        self.assertAlmostEqual(mse(f1, self.ye), quadratic_mse(self.ze, c1, self.ye), places=12)

    def test_sign_pair_and_amplitude_invariance_arrays_only(self):
        c0 = fit_student(self.zt, self.yt)["coefficient"]
        f0 = predict(self.ze, c0)
        ratios = []
        for scale in (1.0, -2.7):
            ps = []
            for sign in (1.0, -1.0):
                c = fit_student(self.zt, self.yt + sign * scale * self.et[:, 1])["coefficient"]
                ps.append(predict(self.ze, c))
            delta = (mse(ps[0], self.ye) + mse(ps[1], self.ye)) / 2 - mse(f0, self.ye)
            ratios.append(transfer_ratio(delta, np.mean((scale * self.ee[:, 1]) ** 2))[0])
            ref = predict(self.ze, qr_reference(self.zt, scale * self.et[:, 1]))
            np.testing.assert_allclose(ps[0], f0 + ref, atol=1e-10, rtol=1e-8)
            self.assertAlmostEqual(delta, np.mean(ref * ref), places=12)
        self.assertAlmostEqual(ratios[0], ratios[1], places=12)

    def test_high_span_G_one_even_with_weak_gold(self):
        # The gold task gate is not a premise of the linear sign-pair identity.
        y = 8 * self.yt
        gold = predict(self.ze, fit_student(self.zt, y)["coefficient"])
        self.assertGreater(mse(gold, 8 * self.ye), .25)
        risks = []
        for s in (1., -1.):
            p = predict(self.ze, fit_student(self.zt, y + s * self.et[:, 0])["coefficient"])
            risks.append(mse(p, 8 * self.ye))
        ratio, status = transfer_ratio(np.mean(risks) - mse(gold, 8 * self.ye), np.mean(self.ee[:, 0] ** 2))
        self.assertEqual(status, "DEFINED")
        self.assertAlmostEqual(ratio, 1.0, places=11)

    def test_zero_error_and_undefined_ratio_not_epsilon(self):
        c0 = fit_student(self.zt, self.yt)["coefficient"]
        cz = fit_student(self.zt, self.yt + np.zeros(len(self.yt)))["coefficient"]
        np.testing.assert_array_equal(c0, cz)
        self.assertEqual(transfer_ratio(0., 0.), (None, "UNDEFINED"))
        self.assertEqual(transfer_ratio(1., np.inf), (None, "UNDEFINED"))
        self.assertEqual(transfer_ratio(-1e-15, 1.)[0], -1e-15)
        self.assertEqual(transfer_ratio(2., 1.)[0], 2.)

    def test_rank_deficiency_explicit_and_qr_refusal(self):
        z = np.column_stack((self.zt, self.zt[:, 0]))
        self.assertEqual(fit_student(z, self.yt)["rank_status"], "RANK_DEFICIENT")
        with self.assertRaises(ValueError):
            qr_reference(z, self.yt)

    def test_nonfinite_and_missing_row_rejected(self):
        with self.assertRaises(ValueError):
            fit_student(self.zt, self.yt[:-1])
        z = self.zt.copy()
        z[0, 0] = np.nan
        with self.assertRaises(ValueError):
            fit_student(z, self.yt)

    def _inputs(self):
        cg = fit_student(self.zt, self.yt)["coefficient"]
        return dict(z_train=self.zt, z_eval=self.ze, y_train=self.yt, y_eval=self.ye,
                    e_train=self.et, e_eval=self.ee, gold_coefficient=cg,
                    gold_train_prediction=predict(self.zt, cg), gold_eval_prediction=predict(self.ze, cg),
                    train_scenario_ids=np.array(["t"+str(j) for j in range(len(self.zt))]),
                    eval_scenario_ids=np.array(["e"+str(j) for j in range(len(self.ze))]),
                    train_group_ids=np.array(["tg"+str(j) for j in range(len(self.zt))]),
                    eval_group_ids=np.array(["eg"+str(j) for j in range(len(self.ze))]))

    def test_frozen_io_reaggregate_and_independent_verification(self):
        with tempfile.TemporaryDirectory(prefix="teg_closeout_synthetic_") as temp:
            inp, out = Path(temp) / "arrays.npz", Path(temp) / "run"
            np.savez(inp, **self._inputs())
            summary = analyze_frozen(inp, out)
            self.assertEqual(summary, summarize_saved(out))
            v = verify_saved(inp, out)
            self.assertEqual(v["status"], "PASS")
            self.assertEqual(v["reference_target_fit_objects"], 3)
            self.assertEqual(summary["registered_H1"], "NOT_RUN")
            with self.assertRaises(FileExistsError):
                analyze_frozen(inp, out)

    def test_input_groups_duplicate_and_corrupt_prediction_guard(self):
        with tempfile.TemporaryDirectory(prefix="teg_closeout_synthetic_") as temp:
            path = Path(temp) / "arrays.npz"
            inp = self._inputs()
            inp["eval_group_ids"][0] = inp["train_group_ids"][0]
            np.savez(path, **inp)
            with self.assertRaises(ValueError):
                load_frozen(path)
            inp = self._inputs()
            inp["train_scenario_id"] = inp.pop("train_scenario_ids")
            inp["eval_scenario_id"] = inp.pop("eval_scenario_ids")
            inp["train_group_id"] = inp.pop("train_group_ids")
            inp["eval_group_id"] = inp.pop("eval_group_ids")
            np.savez(path, **inp)
            loaded = load_frozen(path)
            np.testing.assert_array_equal(loaded["train_scenario_ids"], inp["train_scenario_id"])
            inp["gold_eval_prediction"][0] += 1.
            np.savez(path, **inp)
            with self.assertRaises(AssertionError):
                analyze_frozen(path, Path(temp) / "bad_run")


if __name__ == "__main__":
    unittest.main()
