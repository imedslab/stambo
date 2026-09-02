from typing import Iterable, Dict, Tuple, Optional, Any
from tqdm import tqdm as base_tqdm

try:  # tqdm notebook widget is nicer inside Jupyter
    from tqdm.notebook import tqdm as notebook_tqdm  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    notebook_tqdm = base_tqdm


def _in_notebook() -> bool:
    """Detects whether code runs inside a Jupyter notebook."""
    try:
        from IPython import get_ipython  # type: ignore

        shell = get_ipython().__class__.__name__
        return shell == "ZMQInteractiveShell"
    except Exception:
        return False


def pbar(iterable: Iterable, total: int, desc: str, silent: bool=False) -> base_tqdm:
    r"""Progress bar wrapper.
    
    Args:
        iterable: The iterable to wrap.
        total: The total number of iterations.
        desc: The description of the progress bar.
        silent: Whether to suppress the progress bar. Defaults to False.

    Returns:
        The progress bar if ``silent`` is False, otherwise the iterable.
    """
    if silent:
        return iterable

    tqdm_impl = notebook_tqdm if _in_notebook() else base_tqdm
    return tqdm_impl(iterable, total=total, desc=desc)

def to_latex(report: Dict[str, Tuple[float]], m1_name: Optional[str]="M1", m2_name: Optional[str]="M2", n_digits: int=2) -> str:
    r"""Converts a report returned by StamBO into a LaTeX table for convenient viewing.

    Note: The alternative hypothesis is that the second model (M2) is different from the first model (M1).
    The p-value is the two-tailed p-value.

    Args:
        report: Dictionary with metrics in the StamBO-generated format.
        m1_name: Name to assign to the first model row. Defaults to M1.
        m2_name: Name to assign to the second model row. Defaults to M2.
        n_digits: Number of digits to round to. Defaults to 2.

    Returns:
        A cut-and-paste LaTeX table in the tabular environment.
    """
    # Format: three rows: one per metric, another per model
    tbl = "% \\usepackage{booktabs} <-- do not forget to have this imported. \n"
    tbl += "\\begin{tabular}{" + "l"*(1 + len(report)) + "} \\\\ \n"
    tbl += "\\toprule \n"
    tbl += "\\textbf{Model}"
    # Building up the header
    for metric in report:
        tbl += " & \\textbf{" + metric + "}"
    tbl += " \\\\ \n\\midrule \n"
    tbl += m1_name
    # Filling the first row
    for metric in report:
        tbl += " & " + f"${report[metric][4]:.{n_digits}f}$ [${report[metric][5]:.{n_digits}f}$-${report[metric][6]:.{n_digits}f}$]"
    tbl += " \\\\ \n"
    tbl += m2_name
    # Filling the second row
    for metric in report:
        tbl += " & " + f"${report[metric][7]:.{n_digits}f}$ [${report[metric][8]:.{n_digits}f}$-${report[metric][9]:.{n_digits}f}$]"
    tbl += " \\\\ \n\\midrule\n"
    # Filling the final row with p-value per metric
    tbl += "Effect size"
    for metric in report:
        tbl += " & " + f"${report[metric][1]:.{n_digits}f}$ [${report[metric][2]:.{n_digits}f}$-${report[metric][3]:.{n_digits}f}]$"
    tbl += " \\\\ \n\\midrule\n"
    
    tbl += "$p$-value"
    for metric in report:
        tbl += " & " + f"${report[metric][0]:.{n_digits}f}$"
    tbl += " \\\\ \n\\bottomrule\n"
    # Final row
    tbl += "\\end{tabular}"

    return tbl


def pairwise_to_latex(report: Dict[str, Dict[str, Dict[str, Any]]], n_digits: int=2) -> str:
    r"""Converts a report returned by :func:`stambo.pairwise_bootstrap_test` /
    :func:`stambo.compare_models_pairwise` into a LaTeX table, one row per pairwise comparison.

    For each statistic, the table shows the observed effect size with its confidence interval,
    and the p-value. If the report was produced with a multiple-comparison correction (i.e. at
    least one comparison has a non-None ``p_value_adjusted``), the adjusted p-value is shown in
    parentheses next to the raw one; otherwise only the raw p-value is shown.

    Args:
        report: Dictionary in the format returned by ``pairwise_bootstrap_test``/``compare_models_pairwise``:
            ``{statistic: {"label_i / label_j": {"p_value": ..., "p_value_adjusted": ..., "diff": ..., "ci_es": (lo, hi), ...}}}``.
        n_digits: Number of digits to round to. Defaults to 2.

    Returns:
        A cut-and-paste LaTeX table in the tabular environment.
    """
    statistics = list(report.keys())
    comparisons = list(next(iter(report.values())).keys()) if statistics else []
    has_adjusted = any(
        report[s_tag][comparison]["p_value_adjusted"] is not None
        for s_tag in statistics for comparison in comparisons
    )

    tbl = "% \\usepackage{booktabs} <-- do not forget to have this imported. \n"
    tbl += "\\begin{tabular}{l" + "ll" * len(statistics) + "} \\\\ \n"
    tbl += "\\toprule \n"
    tbl += "\\textbf{Comparison}"
    for s_tag in statistics:
        tbl += " & \\multicolumn{2}{c}{\\textbf{" + s_tag + "}}"
    tbl += " \\\\ \n"
    tbl += " " + " & \\textbf{Diff [CI]} & \\textbf{$p$-value}" * len(statistics)
    tbl += " \\\\ \n\\midrule \n"

    for comparison in comparisons:
        tbl += comparison
        for s_tag in statistics:
            entry = report[s_tag][comparison]
            diff, ci_lo, ci_hi = entry["diff"], entry["ci_es"][0], entry["ci_es"][1]
            tbl += " & " + f"${diff:.{n_digits}f}$ [${ci_lo:.{n_digits}f}$-${ci_hi:.{n_digits}f}$]"
            p_val = entry["p_value"]
            if has_adjusted and entry["p_value_adjusted"] is not None:
                tbl += " & " + f"${p_val:.{n_digits}f}$ (${entry['p_value_adjusted']:.{n_digits}f}$)"
            else:
                tbl += " & " + f"${p_val:.{n_digits}f}$"
        tbl += " \\\\ \n"
    tbl += "\\bottomrule\n"
    tbl += "\\end{tabular}"

    return tbl

