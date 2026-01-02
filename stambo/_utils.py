from typing import Iterable, Dict, Tuple, Optional
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
    
    Note: The alternative hypothesis is that the second model is different from the first model. 
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
        emp_s1 = report[metric]["emp_s1"]
        ci_low_s1 = report[metric]["ci_s1"][0]
        ci_high_s1 = report[metric]["ci_s1"][1]
        tbl += " & " + f"${emp_s1:.{n_digits}f}$ [${ci_low_s1:.{n_digits}f}$-${ci_high_s1:.{n_digits}f}$]"
    tbl += " \\\\ \n"
    tbl += m2_name
    # Filling the second row
    for metric in report:
        emp_s2 = report[metric]["emp_s2"]
        ci_low_s2 = report[metric]["ci_s2"][0]
        ci_high_s2 = report[metric]["ci_s2"][1]
        tbl += " & " + f"${emp_s2:.{n_digits}f}$ [${ci_low_s2:.{n_digits}f}$-${ci_high_s2:.{n_digits}f}$]"
    tbl += " \\\\ \n\\midrule\n"
    # Filling the final row with p-value per metric
    tbl += "Effect size"
    for metric in report:
        diff = report[metric]["diff"]
        ci_low_diff = report[metric]["ci_es"][0]
        ci_high_diff = report[metric]["ci_es"][1]
        tbl += " & " + f"${diff:.{n_digits}f}$ [${ci_low_diff:.{n_digits}f}$-${ci_high_diff:.{n_digits}f}]$"
    tbl += " \\\\ \n\\midrule\n"
    
    tbl += "$p$-value"
    for metric in report:
        p_value = report[metric]["p_value"]
        tbl += " & " + f"${p_value:.{n_digits}f}$"
    tbl += " \\\\ \n\\bottomrule\n"
    # Final row
    tbl += "\\end{tabular}"
    
    return tbl

