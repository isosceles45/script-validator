.PHONY: install test ingest serve eval score docker-build deploy

install:
	python3 -m venv .venv && .venv/bin/pip install -q -r requirements.txt

test:
	.venv/bin/python -m pytest -q

ingest:
	.venv/bin/python -m app.cli ingest

serve:
	.venv/bin/uvicorn app.api:app --reload --port 8080

eval:
	.venv/bin/python -m app.cli eval

score:
	.venv/bin/python -m app.cli score --brief samples/brief.md --script samples/script_mixed.md

docker-build:
	docker build -t tfs-script-validator .

deploy:
	./deploy/cloudrun.sh
