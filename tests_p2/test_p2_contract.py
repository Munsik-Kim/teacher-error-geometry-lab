import copy
import inspect
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from teacher_error_geometry.artifacts import ROOT,read_csv,read_json
from teacher_error_geometry.p2.analysis import pair_deterministic,stats,summarize
from teacher_error_geometry.p2.checks import qr_solve
from teacher_error_geometry.p2.experiment import execute
from teacher_error_geometry.p2.model import METHODS,assigned,generate,hashes,noise_factor,seed,validate
from teacher_error_geometry.p2.risk import coordinate_risk,decomposition,fitted_risk
from teacher_error_geometry.p2.solver import fit_ols
from teacher_error_geometry.p2.theory import exact_anchor,expectation,projection
from teacher_error_geometry.p2.verification import verify

CONFIG=read_json(ROOT/"configs/p2_paper_anchor.json")
TOL={"atol":1e-10,"rtol":1e-9}


def fixture_config(repetitions=6):
    cfg=copy.deepcopy(CONFIG); cfg["experiment"]="P2_unit_fixture_NOT_FULL"; cfg["repetitions"]=repetitions
    for key in cfg["rng"]["streams"]:cfg["rng"]["streams"][key]+=cfg["rng"]["unit_fixture_stream_offset"]
    return cfg


class P2Contract(unittest.TestCase):
    def assertClose(self,left,right):np.testing.assert_allclose(left,right,**TOL)

    def test_matrix_projection_and_independent_fraction_anchor(self):
        validate(CONFIG); g=projection(CONFIG); exact=exact_anchor(); exp=expectation(CONFIG)
        for key,value in {"B":.2378,"B_A":.08,"D_A":.0753,"G_A":.4625,"S_A":.19}.items():
            self.assertClose(g[key],value); self.assertClose(g[key],exact["geometry"][key])
        self.assertEqual(exact["fractions"],{"FO":"13/42","FT":"6067/15000","RO":"13/50","RM":"9516257/46000000"})
        for method in METHODS:self.assertClose(exp["risks"][method],exact["risks"][method])
        self.assertClose(exp["rm_bracket"],.85580625)
        self.assertClose(exp["components"]["RM"]["teacher_bias"],.01524825)
        self.assertClose(exp["components"]["RM"]["estimation_quadratic"],.11162690217391304)

    def test_noise_factor_and_invalid_covariance(self):
        L=noise_factor(1.3,.7,.22)
        self.assertClose(L@L.T,[[1.3,.22],[.22,.7]])
        self.assertClose(np.linalg.det(L@L.T),.8616)
        for args in ((-1,.7,.22),(1,.1,1),(0,1,.1),(1,float("nan"),0)):
            with self.assertRaises(ValueError):noise_factor(*args)
        self.assertClose(noise_factor(0,0,0),np.zeros((2,2)))
        self.assertClose(noise_factor(1,1,1),[[1,0],[1,0]])

    def test_identical_outcome_teacher_and_lambda_zero_controls(self):
        cfg=fixture_config(); cfg["delta"]=[0]*5; cfg["sigma2"]=cfg["tau2"]=cfg["omega"]=1.
        data=generate(cfg,5)
        self.assertTrue(np.array_equal(data["Y"],data["T"]))
        fitted={m:fit_ols(*assigned(data,m)).coefficient for m in METHODS}
        self.assertClose(fitted["FO"],fitted["FT"]); self.assertClose(fitted["RO"],fitted["RM"])
        exp=expectation(cfg)["risks"]
        self.assertClose(exp["FO"],exp["FT"]); self.assertClose(exp["RO"],exp["RM"])
        self.assertTrue(pair_deterministic("FO","FT",cfg)); self.assertTrue(pair_deterministic("RO","RM",cfg))
        cfg=fixture_config(); cfg["lambda"]=0
        exp=expectation(cfg)["risks"]; self.assertClose(exp["RO"],exp["RM"])

    def test_lstsq_independent_qr_and_mixed_target_linearity(self):
        cfg=fixture_config(); data=generate(cfg,13)
        fitted={}
        for m in METHODS:
            design,target=assigned(data,m); fit=fit_ols(design,target); self.assertEqual(fit.status,"ok")
            self.assertClose(fit.coefficient,qr_solve(design,target)); fitted[m]=fit.coefficient
        teacher_head=fit_ols(data["Z"],data["T"]).coefficient
        self.assertClose(fitted["RM"],(1-cfg["lambda"])*fitted["RO"]+cfg["lambda"]*teacher_head)

    def test_fitted_quadratic_coordinate_and_signed_cross_identity(self):
        cfg=fixture_config(); g=projection(cfg); data=generate(cfg,11)
        for m in METHODS:
            fit=fit_ols(*assigned(data,m)); risk=fitted_risk(m,fit.coefficient,cfg); parts=decomposition(m,fit.coefficient,cfg,g)
            self.assertClose(risk,coordinate_risk(m,fit.coefficient,cfg))
            self.assertClose(risk,parts["decomposition_total"])
        w=np.array([1.2,-.8,.4]); l=cfg["lambda"]; beta=np.array(cfg["beta"]); delta=np.array(cfg["delta"])
        h=w-beta[:3]-l*delta[:3]; cross=sum(2*l*delta[i]*h[i] for i in range(3))
        parts=decomposition("RM",w,cfg,g)
        self.assertClose(parts["cross_term"],cross)
        self.assertGreater(abs(cross),1e-4)
        self.assertClose(fitted_risk("RM",w,cfg),.08+l*l*.0753+cross+sum(h*h))

    def test_generator_seed_order_hash_and_pairing(self):
        cfg=fixture_config(); expected={rep:hashes(generate(cfg,rep)) for rep in range(6)}
        self.assertEqual(len({v["x_sha256"] for v in expected.values()}),6)
        for rep in (5,2,0,4,1,3):self.assertEqual(expected[rep],hashes(generate(cfg,rep)))
        self.assertEqual(seed(CONFIG,"X",7),[2026102102,0,7])
        self.assertNotEqual(seed(CONFIG,"X",7),seed(CONFIG,"E",7))
        data=generate(cfg,2)
        self.assertIs(assigned(data,"FO")[0],assigned(data,"FT")[0])
        self.assertIs(assigned(data,"FO")[1],assigned(data,"RO")[1])
        self.assertIs(assigned(data,"RO")[0],assigned(data,"RM")[0])

    def test_solver_boundary_and_abnormal_statuses(self):
        self.assertEqual(list(inspect.signature(fit_ols).parameters),["design","target"])
        self.assertEqual(fit_ols(np.zeros((6,3)),np.ones(6)).status,"rank_deficient")
        self.assertEqual(fit_ols([[1],[float("nan")]],[1,2]).status,"nonfinite_input")
        self.assertEqual(fit_ols([[1],[2]],[1]).status,"invalid_shape")
        with patch("teacher_error_geometry.p2.solver.np.linalg.lstsq",side_effect=np.linalg.LinAlgError("injected")):
            self.assertEqual(fit_ols([[1],[2]],[1,2]).status,"solver_exception")

    def test_structural_determinism_and_zero_se_rule(self):
        self.assertTrue(all(not pair_deterministic(l,r,CONFIG) for l,r in CONFIG["paired_contrasts"]))
        cfg=copy.deepcopy(CONFIG); cfg["lambda"]=0
        self.assertTrue(pair_deterministic("RO","RM",cfg))
        result=stats([0]*10,0,cfg,deterministic=True)
        self.assertEqual(result["comparison"],"DETERMINISTIC_PASS"); self.assertIsNone(result["z"])
        result=stats([0]*10,1,cfg)
        self.assertEqual(result["comparison"],"ZERO_SE_INVESTIGATE"); self.assertIsNone(result["z"])


class P2RawPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory(prefix="teg-p2-unit-")
        cls.root=Path(cls.temp.name); cls.cfg=fixture_config(); cls.run_path=cls.root/"run"
        cls.execution=execute(cls.cfg,cls.run_path,progress=False)

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def test_raw_denominators_and_fixed_key_replay(self):
        result=verify(self.run_path,self.root/"verification.json")
        self.assertEqual(result["implementation_verification"],"PASS",result["issues_first_100"])
        self.assertEqual(result["raw_rows_checked"],24); self.assertEqual(result["unique_dataset_metadata_checked"],6)
        self.assertEqual(self.execution["actual"]["lstsq_calls"],24)
        self.assertEqual(result["verification_extra_calls"]["lstsq_calls"],15)

    def test_raw_reproducible_across_batch_change(self):
        cfg=copy.deepcopy(self.cfg); cfg["batch_size"]=1
        second=self.root/"batch1"; execute(cfg,second,progress=False)
        for file in (self.run_path/"raw").glob("*.csv"):
            self.assertEqual(file.read_bytes(),(second/"raw"/file.name).read_bytes())

    def test_summary_rebuild_and_paired_se(self):
        first=self.root/"summary1"; second=self.root/"summary2"
        a=summarize(self.run_path,first); summarize(self.run_path,second)
        for name in ("methods.csv","paired.csv","decomposition.csv","failures.csv","overview.json"):
            self.assertEqual((first/name).read_bytes(),(second/name).read_bytes())
        raw=list(read_csv(self.run_path/"raw/fits.csv")); indexed={(int(v["replicate"]),v["method"]):float(v["truth_risk"]) for v in raw}
        diffs=np.array([indexed[(i,"FO")]-indexed[(i,"FT")] for i in range(6)])
        pair=next(v for v in a["paired"] if v["left"]=="FO" and v["right"]=="FT")
        np.testing.assert_allclose(pair["mean"],diffs.mean(),**TOL)
        np.testing.assert_allclose(pair["mc_se"],diffs.std(ddof=1)/np.sqrt(6),**TOL)

    def test_rank_failures_kept_in_raw_and_summary_denominators(self):
        cfg=fixture_config(2); bad=self.root/"rank-failed"
        original=generate
        def zero_design(config,rep):
            data=original(config,rep); data["X"][:]=0; data["Z"][:]=0; return data
        with patch("teacher_error_geometry.p2.experiment.generate",side_effect=zero_design):
            result=execute(cfg,bad,progress=False)
        summary=summarize(bad,self.root/"rank-failed-summary")
        self.assertEqual(result["actual"]["raw_rows"],8); self.assertEqual(summary["failed_rows"],8)
        self.assertTrue(all(v["N_valid"]==0 and v["N_failed"]==2 and v["N_missing"]==0 for v in summary["methods"]))


if __name__=="__main__":unittest.main()
