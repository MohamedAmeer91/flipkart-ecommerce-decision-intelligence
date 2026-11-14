"""Run from the repo root: python -m unittest discover -s web/tests -t ."""
import os
import unittest

from fastapi.testclient import TestClient

from web import outputs_io as io
from web.app import app

STATIC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static")
GOOD = dict(tenure_months=12, complaints_90d=5, discount_share=0.7, avg_order_value=15,
            loyalty_score=40, cod_flag=1, plus_member_flag=0)


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c = TestClient(app)
        cls.r2 = io.read_results("part2_results.txt")
        cls.r4 = io.read_results("part4_results.txt")

    def get(self, url, status=200):
        r = self.c.get(url)
        self.assertEqual(r.status_code, status, r.text)
        return r.json()

    # ---- basic endpoints
    def test_health(self):
        self.assertEqual(self.get("/api/health"), {"status": "ok"})

    def test_overview_schema_and_numbers(self):
        d = self.get("/api/overview")
        for k in ("title", "synthetic_notice", "questions", "headline", "findings"):
            self.assertIn(k, d)
        heads = {x["label"]: x["value"] for x in d["headline"]}
        self.assertEqual(heads["Test cost at chosen threshold (INR)"], self.r2["cost_test_at_tau"])
        self.assertEqual(heads["Records needed for margin INR 50"], io.read_results("part1_results.txt")["n_required"])
        self.assertEqual(len(d["findings"]), 4)

    def test_part_endpoints_match_files(self):
        self.assertEqual(self.get("/api/part1")["results"], io.read_results("part1_results.txt"))
        d2 = self.get("/api/part2")
        self.assertEqual(d2["results"], self.r2)
        self.assertEqual(len(d2["thresholds"]), 9)
        self.assertEqual(len(d2["knn_cv"]), 11)
        d3 = self.get("/api/part3")
        self.assertEqual(len(d3["segments"]), 3)
        self.assertEqual(len(d3["hierarchical"]), 15)
        d4 = self.get("/api/part4")
        self.assertEqual(d4["results"], self.r4)
        self.assertEqual(len(d4["voting"]), 5)
        self.assertEqual(len(d4["adaboost"]), 16)
        self.assertEqual(len(d4["architecture"]), 7)

    def test_content_from_readme(self):
        d = self.get("/api/content")
        self.assertIn("logistic-regression", d["memo"])
        self.assertIn("synthetic", d["limitations"])

    # ---- threshold: cross-check with outputs
    def test_threshold_matches_part2_results(self):
        for tau in ("0.5", "0.3"):
            d = self.get("/api/threshold?tau=" + tau)
            for k in ("tp", "fp", "tn", "fn"):
                self.assertEqual(d[k], self.r2["tau_%s_%s" % (tau, k)])
            for k in ("accuracy", "precision", "recall", "f1", "specificity"):
                self.assertAlmostEqual(d[k], self.r2["tau_%s_%s" % (tau, k)], places=4)

    def test_threshold_cost_rule(self):
        d = self.get("/api/threshold?tau=0.15")
        self.assertEqual(d["cost"], self.r2["cost_test_at_tau"])
        self.assertEqual(self.get("/api/threshold?tau=0.5")["cost"], self.r2["cost_test_at_0.5"])
        self.assertEqual(d["cost_flag_nobody"], self.r2["cost_test_flag_nobody"])
        tr = self.get("/api/threshold?tau=0.15&dataset=train")
        self.assertEqual(tr["cost"], self.r2["cost_train_at_tau"])
        self.assertEqual(tr["cost_best_tau"], self.r2["cost_tau"])
        custom = self.get("/api/threshold?tau=0.5&fp_cost=10&fn_cost=20")
        self.assertEqual(custom["cost"], 10 * custom["fp"] + 20 * custom["fn"])

    def test_curve_matches_threshold_csv(self):
        pts = {round(p["tau"], 2): p for p in self.get("/api/curve")["points"]}
        self.assertEqual(len(pts), 101)
        for row in io.read_csv_table("part2_thresholds.csv"):
            p = pts[round(row["tau"], 2)]
            self.assertAlmostEqual(p["tpr"], row["tpr"], places=4)
            self.assertAlmostEqual(p["fpr"], row["fpr"], places=4)

    def test_knn_matches_part2(self):
        d = self.get("/api/knn?k=5")
        self.assertEqual(d["best_k"], self.r2["knn_best_k"])
        for k in ("tp", "fp", "tn", "fn"):
            self.assertEqual(d[k], self.r2["knn_" + k])
        self.assertAlmostEqual(d["f1"], self.r2["knn_f1"], places=4)
        cv = {x["K"]: x["mean_cv_f1"] for x in io.read_csv_table("part2_knn_cv.csv")}
        for row in d["cv"]:
            self.assertAlmostEqual(row["f1"], cv[row["k"]], places=4)
        self.assertEqual(len(d["cv"]), 11)

    def test_segments(self):
        d = self.get("/api/segments")
        self.assertEqual([s["segment"] for s in d["segments"]], ["A", "B", "C"])
        self.assertEqual(d["overall"]["size"], 800)
        self.assertAlmostEqual(d["overall"]["outcome_rate"], 170 / 800, places=3)

    def test_ensemble_matches_voting_csv(self):
        for rule in ("hard", "soft"):
            d = self.get("/api/ensemble?tau=0.3&rule=" + rule)
            ref = {r["model"]: r for r in io.read_csv_table("part4_voting.csv")}
            got = {r["model"]: r for r in d["rows"]}
            for name in ("logistic", "knn", "stump", rule + "_vote"):
                for k in ("tp", "fp", "tn", "fn"):
                    self.assertEqual(got[name][k], ref[name][k], (name, k))
                self.assertAlmostEqual(got[name]["f1"], ref[name]["f1"], places=4)

    def test_score(self):
        r = self.c.post("/api/score", json=GOOD)
        self.assertEqual(r.status_code, 200)
        d = r.json()
        self.assertTrue(0 < d["probability"] < 1)
        self.assertEqual(len(d["contributions"]), 7)
        self.assertAlmostEqual(sum(c["effect"] for c in d["contributions"]) + d["bias"],
                               __import__("math").log(d["probability"] / (1 - d["probability"])), places=6)
        self.assertEqual(d["nearest_segment"], "C")
        risky = dict(GOOD, tenure_months=3, complaints_90d=10, discount_share=0.95, loyalty_score=10)
        safe = dict(GOOD, tenure_months=70, complaints_90d=0, discount_share=0.1, loyalty_score=90)
        self.assertGreater(self.c.post("/api/score", json=risky).json()["probability"],
                           self.c.post("/api/score", json=safe).json()["probability"])

    # ---- validation and errors
    def test_validation_errors(self):
        for url in ("/api/threshold?tau=1.5", "/api/threshold?tau=abc", "/api/threshold?dataset=val",
                    "/api/threshold?fp_cost=-1", "/api/knn?k=4", "/api/knn?k=99", "/api/knn?k=0",
                    "/api/ensemble?rule=maybe", "/api/ensemble?models=svm", "/api/ensemble?tau=0"):
            r = self.c.get(url)
            self.assertEqual(r.status_code, 422, url)
            self.assertIn("detail", r.json())
        for bad in (dict(GOOD, tenure_months=-1), dict(GOOD, discount_share=1.5), dict(GOOD, cod_flag=2),
                    dict(GOOD, loyalty_score="high"), {k: v for k, v in GOOD.items() if k != "cod_flag"}):
            r = self.c.post("/api/score", json=bad)
            self.assertEqual(r.status_code, 422)
            self.assertIn("errors", r.json())

    def test_unknown_api_404(self):
        r = self.c.get("/api/nope")
        self.assertEqual(r.status_code, 404)
        self.assertEqual(r.json()["detail"], "Unknown endpoint")

    # ---- static
    def test_static_served(self):
        r = self.c.get("/")
        self.assertEqual(r.status_code, 200)
        self.assertIn("text/html", r.headers["content-type"])
        self.assertIn("main.js", r.text)
        for p in ("css/style.css", "js/main.js", "js/pages.js", "js/charts.js", "js/api.js", "js/md.js"):
            r = self.c.get("/static/" + p)
            self.assertEqual(r.status_code, 200, p)
            self.assertTrue(len(r.content) > 100)
        self.assertEqual(self.c.get("/static/missing.js").status_code, 404)

    def test_no_external_resources(self):
        for root, _, files in os.walk(STATIC):
            for f in files:
                with open(os.path.join(root, f), encoding="utf-8") as fh:
                    self.assertNotIn("http://", fh.read(), f)
                with open(os.path.join(root, f), encoding="utf-8") as fh:
                    self.assertNotIn("https://", fh.read(), f)


if __name__ == "__main__":
    unittest.main()
