Main functionality
=====================

We perform a one-tailed (right-tailed) bootstrap test, comparing two samples using pre-defined statistics.
The library works so that the user provides a function that computes the statistic of interest,
and predictions of two models on a test set. 

.. automodule:: stambo
    :members:
    :member-order: bysource
    :special-members: __call__, __getitem__
    :show-inheritance:
   