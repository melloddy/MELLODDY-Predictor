# Melloddy Predictor

This open-source software is a Python package made for external data scientists without high knowledge of the MELLODDY stack to perform predictions on new data easily, from the models produced during the yearly runs. It is built on top of `Melloddy-Tuner` and `Sparsechem` to manage both data pre-processing and model inference steps. It is flexible enough to handle multiple models and data size, and predict on subset on tasks.

> :warning: The model should be compatible with `sparsechem` `0.9.6+`. If it is not, you can convert it with
[the `convert.py` script](https://git.infra.melloddy.eu/wp2/sparsechem/-/blob/convert_v0.9.5_to_v0.9.6/examples/chembl/convert.py) from `sparsechem`.

## Installation

To install the package, run in a new python 3.8+ environment:

1. Clone the repository

   ```sh
   git clone git@github.com:melloddy/MELLODDY-Predictor.git
   ```

2. Install the requirements

   ```sh
   pip install -e ".[gitlab]"
   ```

3. (Optional) To be able to run the examples and the tests, download the [dummy files](https://zenodo.org/record/6579398/). You can download it using `make inputs`. Otherwise download the archive, extract it an place the `inputs` folder at the root of the project.

## Usage

To build the doc, run in a new terminal with your python environment:

```sh
make doc
```

Then in your browser, go to [http://localhost:8080/melloddy_predictor/](http://localhost:8080/melloddy_predictor/)

You can see an example in `example.py` and run it with:

```sh
python examples/example.py
```

## Development & Testing

Install all the requirements

```sh
pip install -r requirements-dev.txt
```

We use pytest for testing, you can just run the following command to run the full test suite:

```sh
make test
```

Note that input data will be downloaded from Zenodo when running the tests for the first time.

## Troubleshooting

If you want to remove the following warning:

```
[W ParallelNative.cpp:206] Warning: Cannot set number of intraop threads after parallel work has started or after set_num_threads call when using native parallel backend (function set_num_threads)
```

run:

```sh
export OMP_NUM_THREADS=1
```
