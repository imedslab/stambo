from tqdm import tqdm
from typing import Iterable, Dict, Tuple, Optional

def pbar(iterable: Iterable, total: int, desc: str, silent: bool=False) -> tqdm:
    """Progress bar wrapper.
    
    Args:
        iterable (Iterable): The iterable to wrap.
        total (int): The total number of iterations.
        desc (str): The description of the progress bar.
        silent (bool, optional): Whether to suppress the progress bar. Defaults to False.
    Returns:
        [tqdm, Iterable]: The progress bar if silent is False, otherwise the iterable.
    """
    if silent:
        return iterable
    return tqdm(iterable, total=total, desc=desc)

def to_latex(report: Dict[str, Tuple[float]], m1_name: Optional[str]="M1", m2_name: Optional[str]="M2", n_digits: int=2) -> str:
    """Converts a report returned by StamBO into a LaTeX table for convenient viewing.
    
        Note: the p-value is the left-sided p-value. The alternative hypothesis is that the second model
        
    Args:
        report (Dict[str, Tuple[float]]): A dictionary with metrics. Use the stambo-generated format.
        m1 (str, optional): Name to assign to the table row. Defaults to M1.
        m2 (str, optional): Name to assign to the table row. Defaults to M2.
        n_digits (int, optional): Number of digits to round to. Defaults to 2.
    Returns:
        str: A cut-and-paste LaTeX table in tabular environment.
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

