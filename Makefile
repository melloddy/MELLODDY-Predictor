inputs:
	curl -s -o inputs.zip https://zenodo.org/record/6579398/files/mms.zip?download=1
	unzip inputs.zip
	rm inputs.zip

.PHONY: test
test: inputs
	pytest ./tests/
	pytest --nbmake "./tests/examples/"
	python ./examples/example.py

.PHONY: doc
doc:
	pdoc --http localhost:8080 --config show_source_code=False model_manipulation_software
