.PHONY: inputs test doc pre-commit-checks

inputs:
	curl -s -o inputs.zip https://zenodo.org/record/6807845/files/inputs.zip?download=1
	unzip inputs.zip
	rm inputs.zip

test: inputs
	pytest ./tests/
	pytest --nbmake "./examples/"
	python ./examples/example.py

doc:
	pdoc --http localhost:8080 --config show_source_code=False melloddy_predictor

lint: ## Run pre-commit checks on all files
	pre-commit run --hook-stage manual --all-files
