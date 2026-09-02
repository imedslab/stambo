.. stambo documentation master file, created by
   sphinx-quickstart on Sat Feb 10 00:04:32 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to stambo!
==================================

About
------------


Statistical Model Comparison with Bootstrap (STAMBO) focuses on statistically sound comparisons between models and samples by implementing
the two-tailed bootstrap hypothesis tests:

.. figure:: /_static/banner.png
   :alt: stambo banner
   :align: left
   :class: banner
   :width: 70%

We have abstracted the bootstrap two-sample test into a single function: :func:`stambo.two_sample_test`.
To start using the library, one can simply compare just two means (the default assumpes paired design).

.. code-block:: python

   import stambo
   ...
   seed = 42
   res = stambo.two_sample_test(sample_1, sample_2, statistics={"Mean": lambda x: x.mean()})

If you would like to avoid the paired design, you can simply set the `non_paired` argument to `True`.

What makes this libarry different, is that we support implementation of bootsyrap across many metrics at the same time and clustered bootstrap. 
The latter is particularly useful when the data is from the same patient. 
Here is how we run it for the case when predictions come a dataset with repeated measurements from the same patient:

.. code-block:: python

   import stambo
   ...
   seed = 42
   results = stambo.compare_models(y_test, preds_1, preds_2, ("ROCAUC", "AP", "QKappa", "BACC", "MCC"), seed=seed, n_bootstrap=1000)
   print(stambo.to_latex(results))

The above will print a LaTeX table, which one can easily copy-paste:

.. figure:: /_static/example_table.png
   :alt: example table
   :align: left
   :width: 90%

If you have more than two models (or samples) to compare, :func:`stambo.compare_models_pairwise`
(and its lower-level building block, :func:`stambo.pairwise_bootstrap_test`) run the bootstrap
test on every pair, with a Holm-Bonferroni correction for the multiple comparisons applied by
default:

.. code-block:: python

   import stambo
   ...
   seed = 42
   results = stambo.compare_models_pairwise(y_test, (preds_1, preds_2, preds_3), ("ROCAUC", "AP"), seed=seed, n_bootstrap=1000)
   print(stambo.pairwise_to_latex(results))

See the :doc:`Pairwise_comparison` example for a full walkthrough, including why the correction
matters and how it interacts with clustered/grouped data.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

.. toctree::
   :hidden:

   self


.. toctree::
   :maxdepth: 3
   :caption: Documentation:

   stambo
   metrics

.. toctree::
   :maxdepth: 1
   :caption: Examples:

   Classification
   Classification_non_iid
   Regression
   Two_sample_test
   Pairwise_comparison