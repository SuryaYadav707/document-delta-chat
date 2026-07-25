.PHONY: setup up down run chat markup eval eval-baseline eval-compare fmt lint

# Single-command entrypoints required by the assignment acceptance criteria.
# Each target is a thin wrapper over `python -m ...` — no logic lives here.

setup:            ## install deps into a venv
	python -m venv .venv && . .venv/bin/activate && pip install -e ".[dev]"

up:               ## start Langfuse (+ postgres) via docker-compose
	docker compose up -d

down:
	docker compose down

# core pipeline: ingest two PIDs -> canonical -> delta -> report -> index
run:              ## make run A=<pid_or_path> B=<pid_or_path>
	python -m src.app.cli compare --a "$(A)" --b "$(B)"

chat:             ## launch FastAPI + minimal web UI (PORT overridable; 8000 is taken by nginx here)
	uvicorn src.app.api:app --reload --port $(or $(PORT),8001)

markup:           ## bonus: overlay delta bboxes -> annotated PDF
	python -m src.app.cli markup --comparison "$(CMP)"

eval:             ## run eval harness, print scorecard, write results/run-<timestamp>.json
	python -m eval.run_eval

eval-baseline:    ## freeze the current run as results/baseline.json (regression reference)
	python -m eval.run_eval baseline

eval-compare:     ## diff baseline vs newest run -> regression table (exit 1 on F1 drop)
	python -m eval.compare

fmt:
	ruff format .

lint:
	ruff check .
