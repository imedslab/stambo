import numpy as np

import stambo

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold


def _make_classification_non_iid_fixture(seed: int = 2025):
    # Mirrors notebooks/Classification_non_iid.ipynb
    data, y, subject_ids = stambo.synthetic.generate_non_iid_measurements(
        n_data=300,
        n_subjects=100,
        rho=0.9,
        subj_sigma=1.0,
        noise_sigma=0.5,
        gamma=0.8,
        mu_cls_1=[2, 2],
        mu_cls_2=[2.1, 2.4],
        overlap=0.2,
        feat_corr=0.5,
        seed=42,
    )

    gss = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed)
    train_idx, test_idx = next(gss.split(data, y, groups=subject_ids))

    X_train, X_test = data[train_idx], data[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    groups_test = subject_ids[test_idx]

    knn = KNeighborsClassifier(n_neighbors=5)
    logreg = LogisticRegression()
    knn.fit(X_train, y_train)
    logreg.fit(X_train, y_train)

    return y_test, groups_test, logreg.predict(X_test), knn.predict(X_test)


def test_compare_models_p_values_never_zero_and_respect_bootstrap_grid():
    seed = 2025
    n_bootstrap = 400
    y_test, groups_test, logreg_predictions, knn_predictions = _make_classification_non_iid_fixture(seed=seed)

    naive = stambo.compare_models(
        y_test,
        logreg_predictions,
        knn_predictions,
        metrics=("ROCAUC", "AP"),
        seed=seed,
        n_bootstrap=n_bootstrap,
        silent=True,
    )

    # With the +1 smoothing and two-tailed doubling used in stambo,
    # the theoretical minimum non-zero p-value is 2/(B+1).
    p_min = 2.0 / (n_bootstrap + 1)
    eps = 1e-12

    for metric in naive:
        p = float(naive[metric]["p_value"])
        assert 0.0 < p <= 1.0
        assert p + eps >= p_min


def test_compare_models_clustered_p_values_not_smaller_than_naive_for_fixture():
    # Regression guard for the specific synthetic non-iid notebook scenario:
    # clustered bootstrap should not make p-values smaller than naive bootstrap.
    seed = 2025
    n_bootstrap = 400
    y_test, groups_test, logreg_predictions, knn_predictions = _make_classification_non_iid_fixture(seed=seed)

    naive = stambo.compare_models(
        y_test,
        logreg_predictions,
        knn_predictions,
        metrics=("ROCAUC", "AP"),
        seed=seed,
        n_bootstrap=n_bootstrap,
        silent=True,
    )
    clustered = stambo.compare_models(
        y_test,
        logreg_predictions,
        knn_predictions,
        metrics=("ROCAUC", "AP"),
        groups=groups_test,
        seed=seed,
        n_bootstrap=n_bootstrap,
        silent=True,
    )

    for metric in naive:
        assert float(clustered[metric]["p_value"]) >= float(naive[metric]["p_value"]) - 1e-12


