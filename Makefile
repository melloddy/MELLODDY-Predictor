.PHONY: inputs test doc lint clean

inputs:
	@if [ -e "inputs" ]; \
		then \
		echo 'inputs folder already exists, skipping download' >&2; \
	else \
		curl -s -o inputs.zip https://zenodo.org/record/6807845/files/inputs.zip?download=1; \
		unzip inputs.zip; \
		rm inputs.zip; \
	fi

test: inputs
	pytest ./tests/
	pytest --nbmake "./examples/"
	python ./examples/*.py
doc:
	pdoc --http localhost:8080 --config show_source_code=False melloddy_predictor

lint: ## Run pre-commit checks on all files
	pre-commit run --hook-stage manual --all-files

clean: clean-build clean-pyc clean-test ## remove all build, test, coverage and Python artifacts

clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -path ./.venv -prune -false -o -name '*.egg-info' -exec rm -fr {} +
	find . -path ./.venv -prune -false -o -name '*.egg' -exec rm -f {} +

clean-pyc: ## remove Python file artifacts
	find . -path ./.venv -prune -false -o -name '*.pyc' -exec rm -f {} +
	find . -path ./.venv -prune -false -o -name '*.pyo' -exec rm -f {} +
	find . -path ./.venv -prune -false -o -name '*~' -exec rm -f {} +
	find . -path ./.venv -prune -false -o -name '__pycache__' -exec rm -fr {} +

clean-test: ## remove test and coverage artifacts
	rm -f .coverage
	rm -f coverage.xml
	rm -fr htmlcov/
	rm -fr .pytest_cache
	rm -fr .mypy_cache
	rm -fr prof/
