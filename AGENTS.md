# AGENTS.md — using `stambo`

`stambo` is a bootstrap-based library for statistically comparing samples or ML models,
with correct handling of nested/clustered (non-i.i.d.) data and of multiple comparisons.
This file is a cheat sheet so you don't have to re-derive the API from source. `pytest
tests/` is the way to validate any change you make to the library itself.

## Which function do I call?

```
Comparing an arbitrary statistic on two raw samples (not ML predictions)
    -> two_sample_test(sample_1, sample_2, statistics={...})

Comparing exactly two models' predictions on ML metrics (stambo.metrics or a custom Metric)
    -> compare_models(y_test, preds_1, preds_2, metrics=(...))

Comparing 3+ models / samples
    -> compare_models_pairwise(y_test, (preds_1, preds_2, preds_3, ...), metrics=(...))
    -> pairwise_bootstrap_test(samples=(s1, s2, s3, ...), statistics={...})   # non-ML version
```

**Do not** call `compare_models` in a manual loop over pairs of models — that skips the
multiple-comparison correction and will report false positives. Always use
`compare_models_pairwise`/`pairwise_bootstrap_test` for 3+ things being compared; they
apply a Holm-Bonferroni correction per statistic by default (`correction="holm"`).

## Paired vs. `non_paired` vs. `groups`

- **Paired (default)** — use when the two/N samples are measured on the *same* underlying
  rows (e.g. two models' predictions on the same test set). This is what
  `compare_models`/`compare_models_pairwise` always use.
- **`non_paired=True`** (`two_sample_test` only) — only for genuinely independent samples
  (e.g. two separate cohorts). Do not use this for two models evaluated on the same test set.
- **`groups=`** — pass whenever some rows are repeated/correlated measurements from the
  same subject (e.g. multiple scans per patient in the test set). This resamples whole
  subjects instead of individual rows; skipping it when it applies will make you
  overconfident (falsely significant results). See `notebooks/Classification_non_iid.ipynb`
  and `notebooks/Pairwise_comparison.ipynb` for worked examples of exactly this failure mode.

## The test is two-tailed

The p-value does not depend on which sample/model is passed first — `H0: f(x1) = f(x2)`.
The **sign** of the returned effect size (`diff = f(x2) - f(x1)`, i.e. "second minus first")
tells you which one scored higher on whatever was passed in.

## Return formats

`two_sample_test` / `compare_models` return `Dict[str, numpy.ndarray]`, one 10-element array
per statistic/metric, in this fixed order:

```
[p_value, diff, ci_es_lo, ci_es_hi, emp_s1, ci_s1_lo, ci_s1_hi, emp_s2, ci_s2_lo, ci_s2_hi]
```

That array is **not** JSON-serializable directly. Call `stambo.to_dict(report)` to get the
same data as `{statistic: {"p_value": ..., "diff": ..., "ci_es": (lo, hi), "ci_s1": (lo, hi),
"ci_s2": (lo, hi), "emp_s1": ..., "emp_s2": ...}}` with plain Python floats — this round-trips
through `json.dumps` and uses the same field names as the pairwise functions below.

`pairwise_bootstrap_test` / `compare_models_pairwise` already return this named-dict shape
directly, one level deeper (keyed by comparison label `"{label_i} / {label_j}"`), plus a
`p_value_adjusted` field (the Holm-corrected p-value, or `None` if `correction=None`):

```python
{"ROCAUC": {"kNN / LogReg": {"p_value": ..., "p_value_adjusted": ..., "diff": ..., "ci_es": (lo, hi), ...}}}
```

`stambo.to_latex(report, ...)` / `stambo.pairwise_to_latex(report, ...)` render either shape
as a copy-paste LaTeX table.

## Builtin metrics (`stambo.metrics`)

`ROCAUC`, `AP`, `F1Score`, `QKappa`, `BACC`, `MCC` (classification, pass thresholded/argmax
or probability scores per metric), `MSE`, `MAE` (regression). Pass by name as a string
(`"ROCAUC"`) or an instance for custom ones — subclass `stambo.metrics.Metric`.

## Minimal snippets

```python
import stambo

# Two raw samples, arbitrary statistic
res = stambo.two_sample_test(sample_1, sample_2, statistics={"Mean": lambda x: x.mean()})

# Two models
res = stambo.compare_models(y_test, preds_1, preds_2, metrics=("ROCAUC", "AP"))

# Two models, repeated measurements per subject in the test set
res = stambo.compare_models(y_test, preds_1, preds_2, metrics=("ROCAUC",), groups=subject_ids)

# N models, corrected for multiple comparisons by default
res = stambo.compare_models_pairwise(y_test, (preds_1, preds_2, preds_3), metrics=("ROCAUC",), labels=("A", "B", "C"))
```

## Full examples

`notebooks/Classification.ipynb`, `notebooks/Classification_non_iid.ipynb`,
`notebooks/Regression.ipynb`, `notebooks/Two_sample_test.ipynb`,
`notebooks/Pairwise_comparison.ipynb`.
