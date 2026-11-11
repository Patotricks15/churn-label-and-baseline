.PHONY: install test reproduce

install:
	pip install -e ".[dev]"

test:
	python -m pytest -q

reproduce:
	@test -n "$(INPUT)" || (echo "Set INPUT=/path/to/online-retail-ii file" && exit 1)
	python scripts/run_baseline.py --input "$(INPUT)" --out-dir reports
