dgfreg
==============================

Regression models estimating standard condition Gibbs free energy changes of biological compounds

# How to run the analysis

Install python >= 3.9. If this is not your default python3 then you should manually create the 
virtual environment with e.g. `python3.10 -m venv .venv`

To run the analysis, run the command `make analysis` from the project root. This
will install a fresh virtual environment if one doesn't exist already, activate
it and install python dependencies and cmdstan, then run the analysis with the
following commands:

- `python dgfreg/prepare_data.py`
- `python dgfreg/sample.py`
- `jupyter execute dgfreg/investigate.ipynb`

# How to create a pdf report

First make sure you have installed [quarto](https://https://quarto.org/).

Now run this command from the project root:

```
make docs
```




# How to run tests

Run this command from the project root:

```
python -m pytest
```

