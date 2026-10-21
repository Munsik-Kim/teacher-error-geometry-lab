import inspect
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from teacher_error_geometry.artifacts import ROOT, read_csv, read_json
from teacher_error_geometry.design import small_config, validate_config
from teacher_error_geometry.experiment import execute
from teacher_error_geometry.model import covariance, data_hash, data_seed, generate, teacher, teacher_delta
from teacher_error_geometry.risk import conditional_teacher_risk, fitted_risks
from teacher_error_geometry.solver import fit_student
from teacher_error_geometry.summary import paired_deterministic, sample_stats, summarize
from teacher_error_geometry.theory import eta_curve, expectation, geometry, risk_variance
from teacher_error_geometry.verification import verify

TOL = {"atol": 1e-10, "rtol": 1e-9}
CONFIG = read_json(ROOT / "configs/p1.json")


class ModelContract(unittest.TestCase):
    def assertClose(self, a, b):
        np.testing.assert_allclose(a, b, **TOL)

    def test_zero_c_and_visible_exact_controls(self):
        for rho in CONFIG["A"]["rho"]:
            x1, _, x2 = generate(30, rho, data_seed(CONFIG["root_seed"], 2, 0, 0, 8))
            for c in (0.0, .5):
                for kind in ("axis_visible", "axis_hidden", "projection_zero"):
                    d = teacher_delta(kind, rho, c)
                    f = fit_student(x1, teacher(x1, x2, d))
                    if c == 0:
                        self.assertClose(f.w, 1)
                        self.assertClose(list(fitted_risks(f.w, rho, d).values()), [0,0])
                    if kind == "axis_visible":
                        self.assertClose(f.w, 1+c)
                        self.assertClose(fitted_risks(f.w, rho, d)["teacher_risk"], 0)

    def test_hidden_rho_zero_and_sign_symmetries(self):
        for n in CONFIG["A"]["n"]:
            self.assertClose(expectation(n, 0, [0,.5])["truth_risk"], .25/(n-2))
            for rho in (.5,.9):
                for d in ([.5,0], [0,.5]):
                    self.assertClose(list(expectation(n, rho, d).values()), list(expectation(n, -rho, d).values()))
                self.assertClose(list(expectation(n, rho, [.1,.3]).values()), list(expectation(n, -rho, [.1,-.3]).values()))

    def test_all_population_teacher_errors_are_c_squared(self):
        cells, _ = validate_config(CONFIG)
        for cell in cells:
            d = np.array([cell["delta1"],cell["delta2"]])
            g = geometry(cell["rho"], d)
            self.assertClose(d @ covariance(cell["rho"]) @ d, .25)
            self.assertClose(g["teacher_mse"], g["a"]**2+g["b"]**2)

    def test_projection_zero_and_eta_curve(self):
        for rho in CONFIG["B"]["rho"]:
            for n in CONFIG["B"]["n"]:
                d = teacher_delta("projection_zero",rho,.5)
                self.assertClose(geometry(rho,d)["a"],0)
                self.assertClose(expectation(n,rho,d)["truth_risk"],.25/(n-2))
        cells, _ = validate_config(CONFIG)
        for cell in cells:
            d = [cell["delta1"],cell["delta2"]]
            g = geometry(cell["rho"], d)
            self.assertClose(expectation(cell["n"],cell["rho"],d)["truth_risk"], eta_curve(cell["n"],.5,g["eta"]))
        self.assertIsNone(geometry(.5,[0,0])["eta"])
        self.assertEqual(eta_curve(30,0,None),0)

    def test_analytic_domains_and_variance_requirement(self):
        for n in (-1,0,1,2,3.5,float("nan")):
            with self.assertRaises(ValueError):
                expectation(n,0,[0,.5])
        with self.assertRaises(ValueError):
            risk_variance(4,0,[0,.5],"truth_risk")
        self.assertEqual(risk_variance(4,0,[.5,0],"truth_risk"),0)
        with self.assertRaises(ValueError):
            generate(10,1,[1,2])

    def test_lstsq_scalar_and_derived_coefficient(self):
        for index,(rho,n) in enumerate(((-.9,10),(0,30),(.5,300),(.9,1000))):
            x1,u,x2 = generate(n,rho,data_seed(CONFIG["root_seed"],2,index,0,3))
            d = teacher_delta("normalized_angle",rho,.5,np.pi/4)
            t = teacher(x1,x2,d)
            f = fit_student(x1,t)
            self.assertEqual(f.status,"ok")
            self.assertClose(f.w,np.sum(x1*t)/np.sum(x1*x1))
            g = geometry(rho,d)
            self.assertClose(f.w,1+g["a"]+g["b"]*np.sum(x1*u)/np.sum(x1*x1))

    def test_quadratic_and_conditional_risk_paths(self):
        for rho in (-.9,0,.75):
            for w in (-2,0,1,1.7):
                for d in ([.5,0],[0,.5],[-.3,.7]):
                    self.assertClose(fitted_risks(w,rho,d)["teacher_risk"],conditional_teacher_risk(w,rho,d))

    def test_generator_definition_and_reorder_reproducibility(self):
        seeds = [data_seed(CONFIG["root_seed"],0,1,2,r) for r in range(7)]
        hashes = {}
        for r in range(7):
            x1,u,x2 = generate(100,.5,seeds[r])
            self.assertTrue(np.array_equal(x2,.5*x1+np.sqrt(.75)*u))
            hashes[r] = data_hash(x1,u,x2)
        for r in (6,0,3,1,5,2,4):
            self.assertEqual(hashes[r],data_hash(*generate(100,.5,seeds[r])))
        self.assertEqual(len(set(hashes.values())),7)
        self.assertNotEqual(hashes[0],data_hash(*generate(100,.5,data_seed(CONFIG["root_seed"],1,1,2,0))))

    def test_learner_information_boundary_and_gold_control(self):
        self.assertEqual(list(inspect.signature(fit_student).parameters),["z","target"])
        x1,_,_ = generate(30,.5,[2026102101,0,2,0,0,99])
        gold = fit_student(x1,x1.copy())  # separate gold-target control, never teacher training
        self.assertClose(gold.w,1)
        self.assertClose(fitted_risks(gold.w,.5,[0,0])["truth_risk"],0)

    def test_abnormal_inputs_and_solver_exception_are_explicit(self):
        self.assertEqual(fit_student([0,0],[1,2]).status,"rank_deficient")
        self.assertEqual(fit_student([1,float("nan")],[1,2]).status,"nonfinite_input")
        self.assertEqual(fit_student([1,2],[1]).status,"invalid_shape")
        self.assertEqual(fit_student(["x"],[1]).status,"invalid_input")
        with patch("teacher_error_geometry.solver.np.linalg.lstsq",side_effect=np.linalg.LinAlgError("injected")):
            fit = fit_student([1,2],[1,2])
        self.assertEqual(fit.status,"solver_exception")
        self.assertTrue(fit.solver_called)

    def test_deterministic_and_zero_se_are_not_divided(self):
        deterministic = sample_stats([.25]*20,.25,CONFIG,deterministic=True)
        self.assertEqual(deterministic["comparison"],"DETERMINISTIC_PASS")
        self.assertIsNone(deterministic["standardized_difference"])
        zero = sample_stats([0]*20,1,CONFIG)
        self.assertEqual(zero["comparison"],"ZERO_SE_INVESTIGATE")
        self.assertIsNone(zero["standardized_difference"])

    def test_fixed_full_plan_and_pair_groups(self):
        cells, groups = validate_config(CONFIG)
        self.assertEqual((len(cells),len(groups)),(84,31))
        self.assertEqual({b:sum(c["block"]==b for c in cells) for b in "ABC"},{"A":50,"B":10,"C":24})
        self.assertEqual(len(cells)*CONFIG["repetitions"],168000)
        self.assertEqual(len(groups)*CONFIG["repetitions"],62000)
        for group in groups:
            if any(c["block"]=="B" for c in group["cells"]):
                self.assertEqual({c["block"] for c in group["cells"]},{"A","B"})

    def test_paired_angle_teacher_risk_is_deterministic(self):
        cells, _ = validate_config(CONFIG)
        left = next(c for c in cells if c["config_id"] == "C_r1_n0_t1")
        right = next(c for c in cells if c["config_id"] == "C_r1_n0_t3")
        self.assertTrue(paired_deterministic(left,right,"teacher_risk"))
        self.assertFalse(paired_deterministic(left,right,"truth_risk"))
        x1,u,x2 = generate(30,0,data_seed(CONFIG["root_seed"],1,1,0,17))
        actual = []
        for c in (left,right):
            d = [c["delta1"],c["delta2"]]
            w = fit_student(x1,teacher(x1,x2,d)).w
            actual.append(fitted_risks(w,0,d)["teacher_risk"])
        self.assertClose(actual[0],actual[1])


class RawPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="teg-p0-")
        cls.root = Path(cls.temp.name)
        cls.config = small_config(CONFIG,3)
        cls.run_path = cls.root/"run"
        cls.execution = execute(cls.config,cls.run_path,progress=False)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_raw_keys_pairing_hashes_and_replay(self):
        result = verify(self.run_path)
        self.assertEqual(result["implementation_verification"],"PASS",result["issues_first_100"])
        self.assertEqual(result["raw_rows_checked"],252)
        self.assertEqual(result["unique_dataset_metadata_checked"],93)
        self.assertEqual(self.execution["actual"]["solver_calls"],252)

    def test_summary_regeneration_and_paired_zero_duplicate(self):
        first = self.root/"summary1"
        second = self.root/"summary2"
        summarize(self.run_path,first)
        summarize(self.run_path,second)
        for filename in ("cells.csv","paired_differences.csv","failures.csv","overview.json"):
            self.assertEqual((first/filename).read_bytes(),(second/filename).read_bytes())
        pairs = list(read_csv(first/"paired_differences.csv"))
        duplicate = [r for r in pairs if r["identical_teacher_reuse"]=="True"]
        self.assertEqual(len(duplicate),4)
        self.assertTrue(all(float(r["mean"])==0 and r["independent_evidence"]=="False" for r in duplicate))
        symmetric = [r for r in pairs if r["left_config_id"] in ("C_r1_n0_t1","C_r1_n1_t1")
                     and r["right_config_id"] in ("C_r1_n0_t3","C_r1_n1_t3") and r["metric"]=="teacher_risk"]
        self.assertEqual(len(symmetric),2)
        self.assertTrue(all(r["comparison"]=="DETERMINISTIC_PASS" and r["standardized_difference"]=="" for r in symmetric))

    def test_batch_size_change_keeps_raw_exact(self):
        cfg = json.loads(json.dumps(self.config))
        cfg["batch_size"] = 1
        second = self.root/"batch1"
        execute(cfg,second,progress=False)
        for path in (self.run_path/"raw").glob("*.csv"):
            self.assertEqual(path.read_bytes(),(second/"raw"/path.name).read_bytes())

    def test_nonfinite_rows_retained_in_denominators(self):
        cfg = small_config(CONFIG,2)
        failed = self.root/"failed-fixture"
        with patch("teacher_error_geometry.experiment.teacher",side_effect=lambda z,x,d: np.full_like(z,np.nan)):
            execution = execute(cfg,failed,progress=False)
        result = summarize(failed,self.root/"failed-summary")
        self.assertEqual(execution["actual"]["raw_rows"],168)
        self.assertEqual(execution["actual"]["solver_calls"],0)
        self.assertEqual(result["failed_rows"],168)
        self.assertEqual(result["missing_rows"],0)
        for row in read_csv(self.root/"failed-summary/cells.csv"):
            self.assertEqual((row["N_valid"],row["N_failed"],row["R_rows"]),("0","2","2"))


if __name__ == "__main__":
    unittest.main()
