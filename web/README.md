# Portfolio web layer

An interactive view of the capstone: FastAPI backend plus a plain HTML/CSS/ES-module frontend (no build step, no external resources, charts are inline SVG). It only reads `outputs/` and the root README text, and imports `common.py` to compute model probabilities on the project's 560/240 split. It never writes to the graded files.

## Run (from the repository root)

Windows (PowerShell):

    python -m venv .venv
    .venv\Scripts\activate
    pip install -r web/requirements.txt
    uvicorn web.app:app --port 8000

macOS / Linux:

    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r web/requirements.txt
    uvicorn web.app:app --port 8000

Open `localhost:8000` in a browser. Interactive API docs are at `/docs`.

## Test

    python -m unittest discover -s web/tests -t .

## Docker

The build context is the repository root:

    docker build -f web/Dockerfile -t flipkart-portfolio .
    docker run -p 8000:8000 flipkart-portfolio

## Pages

Overview, Part 1 (order value), Part 2 (threshold lab, KNN K selector, score a record), Part 3 (segment explorer), Part 4 (ensemble voting lab, AdaBoost, sizing), Memo, Limitations.

## API

`/api/health`, `/api/overview`, `/api/part1` to `/api/part4`, `/api/content`, `/api/threshold`, `/api/curve`, `/api/knn`, `/api/segments`, `/api/ensemble`, and `POST /api/score`. Schemas are described in `web/docs/DESIGN.md`; design choices in `web/docs/DECISIONS.md`.
