# Model Manipulation Software

## Installation

To install the package, run in a new python 3.9+ environment:

1. Install [git lfs](https://git-lfs.github.com/)

   ```sh
   git lfs install
   ```

1. Clone the repository

   ```sh
   git clone git@github.com:owkin/melloddy_mms.git
   ```

1. Install the requirements

   ```sh
   pip install -e ".[gitlab]"
   ```

# Usage

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

We use pytest for testing, you can just run the following command to run the full test suite

```sh
python -m pytest .
```
