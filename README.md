# Model Manipulation Software

## Developement

To install the package for developement run:

1. Install [git lfs](https://git-lfs.github.com/)

	```sh
	git lfs install
	```
2. Clone the repository

	```sh
	git clone git@github.com:owkin/melloddy_mms.git
	```
3. Install all the requirements

	```sh
	pip install -e ".[gitlab]"
	pip install -r requirements-dev.txt
	```

### Testing

We use pytest for testing, you can just run the following command to run the full test suite

```sh
pytest .
```

