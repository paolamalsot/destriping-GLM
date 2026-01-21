import pandas as pd
from IPython.display import display
import logging, textwrap
import pandas as pd
import re

def print_full(df: pd.DataFrame):
    """Display the full DataFrame in Jupyter, without truncation."""
    with pd.option_context('display.max_rows', None, 
                           'display.max_columns', None,
                           'display.expand_frame_repr', False,
                           'display.max_colwidth', None):
        display(df)


def log_df(df, logger):
    df_str = df.to_string()
    indented = textwrap.indent(df_str, "    ")
    logger("DataFrame:\n%s", indented)

def save_latex(table, path, tabular_star = True, width=r"\columnwidth", **to_latex_kwargs):
    if tabular_star:
        latex_table = to_tabular_star(table, width, **to_latex_kwargs)
    else:
        latex_table = table.to_latex(**to_latex_kwargs)
    with open(path, "w", encoding="utf-8") as f:
        f.write(latex_table)

def to_tabular_star(df: pd.DataFrame, width=r"\columnwidth", **to_latex_kwargs) -> str:
    s = df.to_latex(**to_latex_kwargs)

    # Capture the column spec from \begin{tabular}{...}
    m = re.search(r"\\begin\{tabular\}\{([^}]*)\}", s)
    if not m:
        raise ValueError("Could not find tabular column specification in to_latex output.")

    colspec = m.group(1)
    new_colspec = colspec.replace("r", "c")

    # Put \extracolsep{\fill} on both sides
    new_colspec = r"@{\extracolsep{\fill}}" + new_colspec + r"@{\extracolsep{\fill}}"

    # s = s.replace(
    #     r"\begin{tabular}{"+colspec+"}",
    #     r"\begin{tabular*}{"+width+"}{"+new_colspec+"}"
    # ).replace(r"\end{tabular}", r"\end{tabular*}", 1)

    s = s.replace(
    rf"\begin{{tabular}}{{{colspec}}}",
    rf"\begin{{tabular*}}{{{width}}}{{{new_colspec}}}",
    1,
    ).replace(r"\end{tabular}", r"\end{tabular*}", 1)

    # Replace booktabs bottom rule macro
    s = s.replace(r"\bottomrule", r"\botrule")

    return s
