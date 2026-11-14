"""FastAPI portfolio backend. Run from the repo root: uvicorn web.app:app"""
import os
from contextlib import asynccontextmanager
from typing import List, Literal

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from web import outputs_io as io
from web.engine import KS, SEGMENT_LABELS, get_engine
from web.schemas import ScoreRequest

STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


@asynccontextmanager
async def lifespan(app):
    get_engine()  # warm the cache once at startup
    yield


app = FastAPI(title="Flipkart E-commerce capstone portfolio", version="1.0.0", lifespan=lifespan)

Dataset = Literal["test", "train"]


@app.exception_handler(RequestValidationError)
async def on_validation(request: Request, exc: RequestValidationError):
    errs = [{"field": ".".join(str(p) for p in e["loc"][1:]), "message": e["msg"]} for e in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": "Invalid input", "errors": errs})


@app.exception_handler(Exception)
async def on_error(request: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"detail": "Internal error"})


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/overview")
def overview():
    r1, r2 = io.read_results("part1_results.txt"), io.read_results("part2_results.txt")
    r3 = io.read_results("part3_results.txt")
    secs = io.readme_sections()
    return {
        "title": "Flipkart E-commerce Decision Intelligence",
        "synthetic_notice": "All data is synthetic; nothing reflects real Flipkart records or performance.",
        "questions": [
            "Is the INR 800 average-order-value planning assumption still right, and how far can a simple model be trusted?",
            "Which customers will leave (no purchase in the next 90 days), and where should the alert threshold sit when money decides?",
            "What shopper segments exist, and how big would later image or text models need to be?"],
        "headline": [
            {"label": "Mean order value (INR hundreds)", "value": r1["mean_value"],
             "note": "vs planned 8.0, z = %.4f" % r1["z_stat"]},
            {"label": "Records needed for margin INR 50", "value": r1["n_required"], "note": "800 available"},
            {"label": "Test cost at chosen threshold (INR)", "value": r2["cost_test_at_tau"],
             "note": "tau = %.2f" % r2["cost_tau"]},
            {"label": "Test cost at tau 0.5 (INR)", "value": r2["cost_test_at_0.5"], "note": "default threshold"},
            {"label": "Cost if nobody flagged (INR)", "value": r2["cost_test_flag_nobody"],
             "note": "all churners missed"},
            {"label": "Segments found", "value": r3["elbow_k"], "note": "K-means elbow"},
            {"label": "Recommended model", "value": "Logistic regression",
             "note": "F1 at tau 0.3 = %.4f" % r2["tau_0.3_f1"]}],
        "findings": {k: secs[k] for k in ("Part 1 findings", "Part 2 findings", "Part 3 findings",
                                          "Part 4 findings") if k in secs},
    }


@app.get("/api/part1")
def part1():
    return {"results": io.read_results("part1_results.txt")}


@app.get("/api/part2")
def part2():
    return {"results": io.read_results("part2_results.txt"),
            "thresholds": io.read_csv_table("part2_thresholds.csv"),
            "knn_cv": io.read_csv_table("part2_knn_cv.csv"),
            "train_means": [float(v) for v in get_engine().S["mu"]]}


@app.get("/api/part3")
def part3():
    return {"results": io.read_results("part3_results.txt"),
            "segments": io.read_csv_table("part3_segments.csv"),
            "hierarchical": io.read_csv_table("part3_hierarchical.csv")}


@app.get("/api/part4")
def part4():
    return {"results": io.read_results("part4_results.txt"),
            "splits": io.read_csv_table("part4_splits.csv"),
            "voting": io.read_csv_table("part4_voting.csv"),
            "adaboost": io.read_csv_table("part4_adaboost.csv"),
            "architecture": io.read_csv_table("part4_architecture.csv")}


@app.get("/api/content")
def content():
    s = io.readme_sections()
    return {"memo": s.get("Model-choice memo", ""), "limitations": s.get("Limitations", "")}


@app.get("/api/threshold")
def threshold(tau: float = Query(0.5, ge=0, le=1), fp_cost: float = Query(50, ge=0, le=100000),
              fn_cost: float = Query(400, ge=0, le=100000), dataset: Dataset = "test"):
    return get_engine().threshold(tau, dataset, fp_cost, fn_cost)


@app.get("/api/curve")
def curve(dataset: Dataset = "test"):
    return {"dataset": dataset, "points": get_engine().curve(dataset)}


@app.get("/api/knn")
def knn(k: int = Query(5, ge=1, le=21)):
    if k not in KS:
        raise HTTPException(status_code=422, detail="k must be odd (1, 3, ..., 21)")
    e = get_engine()
    res = e.knn_result(k)
    res["cv"] = [{"k": K, "f1": e.cv_scores[K]} for K in KS]
    return res


@app.get("/api/segments")
def segments():
    e = get_engine()
    segs = [dict(s, label=SEGMENT_LABELS[s["segment"]]) for s in e.segments]
    return {"segments": segs, "overall": e.overall}


@app.get("/api/ensemble")
def ensemble(tau: float = Query(0.3, ge=0.01, le=0.99), rule: Literal["hard", "soft"] = "hard",
             models: List[Literal["logistic", "knn", "stump"]] = Query(["logistic", "knn", "stump"])):
    if not models:
        raise HTTPException(status_code=422, detail="choose at least one model")
    return get_engine().ensemble(tau, rule, set(models))


@app.post("/api/score")
def score(rec: ScoreRequest, tau: float = Query(0.15, ge=0, le=1)):
    return get_engine().score(rec, tau)


@app.get("/api/{rest:path}")
def api_404(rest: str):
    raise HTTPException(status_code=404, detail="Unknown endpoint")


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC, "index.html"))


app.mount("/static", StaticFiles(directory=STATIC), name="static")
