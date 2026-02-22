from __future__ import annotations

import argparse
import json
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from analyze import compute_insights, run_analysis
from SeaWorldSimulation import SCENARIOS, run_single_scenario
from simulate import parse_scenarios, write_csv

ROOT = Path(__file__).resolve().parent
WEB_DIR = ROOT / "web"
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"

DEFAULT_QUICK_SCENARIOS = ["BASE", "ALT1", "ALT2"]
DEFAULT_QUICK_RUNS = 5
DEFAULT_FULL_RUNS = 30

FEATURE_OPTIONS = [
    "avg_rating",
    "avg_food_income",
    "avg_food_income_per_visitor",
    "avg_reception_income",
    "avg_photo_income",
    "avg_abandonments_per_visitor",
    "total_customers",
    "customers_arriving",
    "customers_leaving",
    "total_people_ate",
    "food_income",
    "reception_income",
    "photo_income",
    "total_revenue",
    "abandonments",
    "drop_count",
]

_RUN_LOCK = threading.Lock()
_RUN_STATE = {
    "running": False,
    "total": 0,
    "current": 0,
    "scenario": "",
    "run_id": 0,
    "message": "idle",
    "started_at": 0.0,
    "elapsed_seconds": 0.0,
    "eta_seconds": 0.0,
}


def _update_state(**kwargs: Any) -> None:
    with _RUN_LOCK:
        _RUN_STATE.update(kwargs)


def _get_state() -> dict[str, Any]:
    with _RUN_LOCK:
        return dict(_RUN_STATE)


def run_simulation_job(scenarios: list[str], runs: int, seed: int, output_path: Path) -> None:
    start = time.time()
    total = len(scenarios) * runs
    _update_state(
        running=True,
        total=total,
        current=0,
        scenario="",
        run_id=0,
        message="starting",
        started_at=start,
        elapsed_seconds=0.0,
        eta_seconds=0.0,
    )

    rows: list[dict] = []
    try:
        count = 0
        for scn_key in scenarios:
            scenario_name = SCENARIOS[scn_key]["label"]
            for run_id in range(runs):
                count += 1
                elapsed = max(0.001, time.time() - start)
                rate = count / elapsed
                remaining = max(0, total - count)
                eta = remaining / rate if rate > 0 else 0.0
                _update_state(
                    current=count,
                    scenario=scn_key,
                    run_id=run_id + 1,
                    message=f"{scn_key} run {run_id + 1} of {runs}",
                    elapsed_seconds=elapsed,
                    eta_seconds=eta,
                )

                seed_used = seed + run_id
                result = run_single_scenario(scn_key, seed=seed_used)
                row = {
                    "scenario_key": scn_key,
                    "scenario_name": scenario_name,
                    "run_id": run_id,
                    "seed_used": seed_used,
                }
                row.update(result)
                rows.append(row)

        DATA_DIR.mkdir(parents=True, exist_ok=True)
        write_csv(rows, output_path)
        _update_state(
            running=False,
            message="complete",
            elapsed_seconds=time.time() - start,
            eta_seconds=0.0,
        )
    except Exception as exc:
        _update_state(
            running=False,
            message=f"error: {exc}",
            elapsed_seconds=time.time() - start,
            eta_seconds=0.0,
        )


class AppHandler(BaseHTTPRequestHandler):
    def _send_json(self, payload: dict, status: int = 200) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_file(self, path: Path, content_type: str) -> None:
        if not path.exists() or not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "File not found")
            return
        data = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _parse_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        body = self.rfile.read(length)
        try:
            return json.loads(body.decode("utf-8"))
        except json.JSONDecodeError:
            return {}

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0].rstrip("/")
        if path == "":
            path = "/"
        if path == "/":
            self._send_file(WEB_DIR / "index.html", "text/html; charset=utf-8")
            return
        if path == "/app.js":
            self._send_file(WEB_DIR / "app.js", "text/javascript; charset=utf-8")
            return
        if path == "/styles.css":
            self._send_file(WEB_DIR / "styles.css", "text/css; charset=utf-8")
            return
        if path == "/status":
            self._send_json(_get_state())
            return
        if path == "/scenarios":
            self._send_json(
                {
                    "scenarios": [
                        {"key": key, "label": SCENARIOS[key]["label"]}
                        for key in SCENARIOS
                    ]
                }
            )
            return
        if path == "/features":
            self._send_json({"features": FEATURE_OPTIONS})
            return
        if path.startswith("/data/"):
            file_path = DATA_DIR / path.replace("/data/", "", 1)
            self._send_file(file_path, "text/csv; charset=utf-8")
            return
        if path.startswith("/outputs/"):
            file_path = OUT_DIR / path.replace("/outputs/", "", 1)
            self._send_file(file_path, "image/png")
            return

        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self) -> None:
        if self.path == "/run":
            payload = self._parse_json()
            mode = payload.get("mode", "quick")
            raw_scenarios = payload.get("scenarios")
            runs = int(payload.get("runs", 0) or 0)
            seed = int(payload.get("seed", 102))

            if _get_state().get("running"):
                self._send_json({"status": "busy", "message": "Simulation already running"}, status=409)
                return

            if mode == "full":
                scenarios = parse_scenarios("ALL")
                runs = DEFAULT_FULL_RUNS
            elif mode == "custom" and raw_scenarios:
                scenarios = parse_scenarios(raw_scenarios)
                if runs <= 0:
                    runs = DEFAULT_QUICK_RUNS
            else:
                scenarios = DEFAULT_QUICK_SCENARIOS
                runs = DEFAULT_QUICK_RUNS

            output_path = DATA_DIR / "simulation_results.csv"

            thread = threading.Thread(
                target=run_simulation_job,
                args=(scenarios, runs, seed, output_path),
                daemon=True,
            )
            thread.start()

            self._send_json(
                {
                    "status": "started",
                    "scenarios": scenarios,
                    "runs": runs,
                    "seed": seed,
                    "output": str(output_path),
                }
            )
            return

        if self.path == "/insights":
            payload = self._parse_json()
            input_path = payload.get("input_path", "data/simulation_results.csv")
            target = payload.get("target", "total_revenue")
            features = payload.get("features", "avg_rating,avg_food_income,total_customers")
            scenario_dummies = bool(payload.get("scenario_dummies", True))
            robust_se = payload.get("robust_se")
            ridge_alpha = payload.get("ridge_alpha")
            ridge_auto = bool(payload.get("ridge_auto", False))
            boxcox_check = bool(payload.get("boxcox_check", False))

            insights = compute_insights(
                input_path=Path(input_path),
                target=target,
                features=features,
                scenario_as_dummies=scenario_dummies,
                robust_se=robust_se,
                ridge_alpha=float(ridge_alpha) if ridge_alpha else None,
                ridge_auto=ridge_auto,
                boxcox_check=boxcox_check,
            )

            self._send_json({"status": "ok", "insights": insights})
            return

        if self.path == "/analyze":
            payload = self._parse_json()
            input_path = payload.get("input_path", "data/simulation_results.csv")
            outdir = payload.get("outdir", "outputs")
            target = payload.get("target", "total_revenue")
            features = payload.get("features", "avg_rating,avg_food_income,total_customers")
            scenario_dummies = bool(payload.get("scenario_dummies", True))
            robust_se = payload.get("robust_se")
            no_plots = bool(payload.get("no_plots", False))

            summary, outputs = run_analysis(
                input_path=Path(input_path),
                outdir=Path(outdir),
                target=target,
                features=features,
                scenario_as_dummies=scenario_dummies,
                robust_se=robust_se,
                no_plots=no_plots,
            )

            self._send_json(
                {
                    "status": "ok",
                    "summary": summary,
                    "outputs": outputs,
                }
            )
            return

        self.send_error(HTTPStatus.NOT_FOUND, "Not found")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the SeaWorld web dashboard.")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind")
    parser.add_argument("--port", type=int, default=8001, help="Port to bind")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    server = ThreadingHTTPServer((args.host, args.port), AppHandler)
    print(f"Web UI running at http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop")
    server.serve_forever()


if __name__ == "__main__":
    main()
