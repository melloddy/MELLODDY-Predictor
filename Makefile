inputs:
	curl -s -o inputs.zip https://zenodo.org/record/6807845/files/inputs.zip?download=1
	unzip inputs.zip
	rm inputs.zip

.PHONY: test
test: inputs
	pytest ./tests/
	pytest --nbmake "./examples/"
	python ./examples/example.py

.PHONY: doc
doc:
	pdoc --http localhost:8080 --config show_source_code=False melloddy_predictor
