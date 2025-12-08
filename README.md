# StaMBO: Statistical model comparison with bootstrap 
[![DOI](https://zenodo.org/badge/747404839.svg)](https://zenodo.org/doi/10.5281/zenodo.10669416)
[![Tests](https://github.com/imedslab/stambo/actions/workflows/tests.yml/badge.svg?branch=dev)](https://github.com/imedslab/stambo/actions/workflows/tests.yml)
[![PyPI version](https://badge.fury.io/py/stambo.svg?branch=master)](https://badge.fury.io/py/stambo)
[![docs](https://github.com/imedslab/stambo/workflows/documentation/badge.svg)](https://imedslab.github.io/stambo/)
[![License](http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat)](LICENSE.md)
------------------------
This package is aimed to be a one-stop-shop for statistical testing in machine learning when it comes to evaluating models on a test set and comparing whether our *improved* model is really beating the baseline. That is, we cover the following very typical use-case in machine learning:
![usecase](docs/source/_static/usecase.png)

Currently, we support the cases of classification, regresson, and semantic segmentation. We do not yet support the significance of ranking, as well as grouped data. It is coming in the future releases.

## In practice
Install from PyPI:
```
pip install stambo
```

The use of the library is then straightforward:
```
import stambo
...
seed = 42
testing_result = stambo.compare_models(y_test, preds_1, preds_2, metrics=("ROCAUC", "AP", "QKappa", "BACC", "MCC"), seed=seed)
print(stambo.to_latex(testing_result))
```

The above will print a LaTeX table, which one can easily copy-paste. As an example, below is the rendered table, which was returned in [`notebooks/Classification.ipynb`](https://github.com/Oulu-IMEDS/stambo/blob/main/notebooks/Classification.ipynb) ([![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Oulu-IMEDS/stambo/main?labpath=notebooks%2FClassification.ipynb)):
![Table](docs/source/_static/example_table.png)

**Note:** From version `0.1.5` we support block-diagonal structure of the data. That is, if you have data from the same patient in the test set, it can easily be adjusted for by specifying the `groups` argument. 

The regression example can be found at [`notebooks/Regression.ipynb`](https://github.com/Oulu-IMEDS/stambo/blob/main/notebooks/Regression.ipynb) ([![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Oulu-IMEDS/stambo/main?labpath=notebooks%2FRegression.ipynb)
)

For more advanced explanation, see the [documentation](https://oulu-imeds.github.io/stambo/). By default, binary, multi-class, and multi-label classification, as well as regression are supported.

One can also use the library to perform a simple two-sample test. For example, to compare the means of two distributions:
```
import stambo
...
seed = 42
res = stambo.two_sample_test(sample_1, sample_2, statistics={"Mean": lambda x: x.mean()})
```

A more detailed and full example of the above is shown here: [`notebooks/Two_sample_test.ipynb`](notebooks/Two_sample_test.ipynb) ([![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Oulu-IMEDS/stambo/main?labpath=notebooks%2FTwo_sample_test.ipynb))

## Contributing

To setup a dev environment, you should use uv and install the project as follows:
```
uv venv
uv pip install -e ".[dev]"
```

## Author
Dr. Aleksei Tiulpin, PhD
