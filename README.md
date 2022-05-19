# Model Manipulation Software

## Installation

To install the package, run in a new python 3.8+ environment:

1. Clone the repository

   ```sh
   git clone git@github.com:melloddy/MMS.git
   ```

2. Install the requirements

   ```sh
   pip install -e ".[gitlab]"
   ```

3. (Optional) To be able to run the examples and the tests, download the [dummy files](https://zenodo.org/record/6560873). You can download it using `make inputs`. Otherwise download the archive, extract it an place the `inputs` folder at the root of the project.

## Usage

To build the doc:

```sh
pdoc --http localhost:8080 model_manipulation_software
```

Then in your browser, go to [http://localhost:8080/model_manipulation_software/](http://localhost:8080/model_manipulation_software/)

You can see an example in `example.py` and run it with:

```sh
python example.py
```

### Development & Testing

Install all the requirements

```sh
pip install -r requirements-dev.txt
```

We use pytest for testing, you can just run the following command to run the full test suite:

```sh
make test
```

Note that input data will be downloaded from Zenodo when running the tests for the first time.
