inputs:
	curl -s -o inputs.zip https://zenodo.org/record/6560873/files/mms.zip?download=1
	unzip inputs.zip
	rm inputs.zip

.PHONY: test
test: inputs
	pytest .
	pytest --nbmake "./examples/"
	python ./examples/example.py
