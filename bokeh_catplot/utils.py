"""Utility functions for parsing categorical plots."""

import numpy as np
import pandas as pd


def _fig_dimensions(kwargs):
    if "width" not in kwargs and "plot_width" not in kwargs:
        kwargs["width"] = 400
    if "height" not in kwargs and "plot_height" not in kwargs:
        kwargs["height"] = 300

    return kwargs


def _data_cats(data, cats, show_legend):
    data = pd.DataFrame(data)

    if cats is None:
        data["__dummy_cat"] = " "
        cats = "__dummy_cat"
        show_legend = False

    return data, cats, show_legend


def _fill_between(p, x1=None, y1=None, x2=None, y2=None, **kwargs):
    """
    Create a filled region between two curves.
    Parameters
    ----------
    p : bokeh.plotting.Figure instance
        Figure to be populated.
    x1 : array_like
        Array of x-values for first curve
    y1 : array_like
        Array of y-values for first curve
    x2 : array_like
        Array of x-values for second curve
    y2 : array_like
        Array of y-values for second curve
    kwargs
        Any kwargs passed to p.patch.

    Returns
    -------
    output : bokeh.plotting.Figure instance
        Plot populated with fill-between.

    """
    p.patch(
        x=np.concatenate((x1, x2[::-1])), y=np.concatenate((y1, y2[::-1])), **kwargs
    )

    return p


def _get_marker(p, marker):
    if marker.lower() == "asterisk":
        return p.asterisk
    if marker.lower() == "circle":
        return p.circle
    if marker.lower() == "circle_cross":
        return p.circle_cross
    if marker.lower() == "circle_x":
        return p.circle_x
    if marker.lower() == "cross":
        return p.cross
    if marker.lower() == "dash":
        return p.dash
    if marker.lower() == "diamond":
        return p.diamond
    if marker.lower() == "diamond_cross":
        return p.diamond_cross
    if marker.lower() == "hex":
        return p.hex
    if marker.lower() == "inverted_triangle":
        return p.inverted_triangle
    if marker.lower() == "square":
        return p.square
    if marker.lower() == "square_cross":
        return p.square_cross
    if marker.lower() == "square_x":
        return p.square_x
    if marker.lower() == "triangle":
        return p.triangle
    if marker.lower() == "x":
        return p.x

    raise RuntimeError(
        f"{marker} is an invalid marker specification. Acceptable values are ['asterisk', 'circle', 'circlecross', 'circle_x', 'cross', 'dash', 'diamond', 'diamondcross', 'hex', 'invertedtriangle', 'square', 'squarecross', 'squarex', 'triangle', 'x']"
    )


def _source_and_labels_from_cats(df, cats):
    if type(cats) in [list, tuple]:
        cat_source = list(zip(*tuple([df[cat].astype(str) for cat in cats])))
        return cat_source, [", ".join(cat) for cat in cat_source]
    else:
        cat_source = list(df[cats].astype(str).values)
        return cat_source, cat_source


def _tooltip_cols(tooltips):
    if tooltips is None:
        return []
    if type(tooltips) not in [list, tuple]:
        raise RuntimeError("`tooltips` must be a list or tuple of two-tuples.")

    cols = []
    for tip in tooltips:
        if type(tip) not in [list, tuple] or len(tip) != 2:
            raise RuntimeError("Invalid tooltip.")
        if tip[1][0] == "@":
            if tip[1][1] == "{":
                cols.append(tip[1][2 : tip[1].find("}")])
            elif "{" in tip[1]:
                cols.append(tip[1][1 : tip[1].find("{")])
            else:
                cols.append(tip[1][1:])

    return cols


def _cols_to_keep(cats, val, color_column, tooltips):
    cols = _tooltip_cols(tooltips)
    cols += [val]

    if type(cats) in [list, tuple]:
        cols += list(cats)
    else:
        cols += [cats]

    if color_column is not None:
        cols += [color_column]

    return list(set(cols))


def _check_cat_input(df, cats, val, color_column, tooltips, palette, order, kwargs):
    if df is None:
        raise RuntimeError("`df` argument must be provided.")
    if cats is None:
        raise RuntimeError("`cats` argument must be provided.")
    if val is None:
        raise RuntimeError("`val` argument must be provided.")

    if type(palette) not in [list, tuple]:
        raise RuntimeError("`palette` must be a list or tuple.")

    if val not in df.columns:
        raise RuntimeError(f"{val} is not a column in the inputted data frame")

    cats_array = type(cats) in [list, tuple]
    if cats_array and len(cats) == 1:
        cats = cats[0]
        cats_array = False

    if cats_array:
        for cat in cats:
            if cat not in df.columns:
                raise RuntimeError(f"{cat} is not a column in the inputted data frame")
    else:
        if cats not in df.columns:
            raise RuntimeError(f"{cats} is not a column in the inputted data frame")

    if color_column is not None and color_column not in df.columns:
        raise RuntimeError(f"{color_column} is not a column in the inputted data frame")

    cols = _cols_to_keep(cats, val, color_column, tooltips)

    for col in cols:
        if col not in df.columns:
            raise RuntimeError(f"{col} is not a column in the inputted data frame")

    bad_kwargs = ["x", "y", "source", "cat", "legend"]
    if kwargs is not None and any([key in kwargs for key in bad_kwargs]):
        raise RuntimeError(", ".join(bad_kwargs) + " are not allowed kwargs.")

    if val == "cat":
        raise RuntimeError("`'cat'` cannot be used as `val`.")

    if val == "__label" or (cats == "__label" or (cats_array and "__label" in cats)):
        raise RuntimeError("'__label' cannot be used for `val` or `cats`.")

    if order is not None:
        grouped = df.groupby(cats)
        if grouped.ngroups > len(order):
            raise RuntimeError(
                "`order` must have at least as many elements as the number of unique groups in `cats`."
            )
        for entry in order:
            if entry not in grouped.groups.keys():
                raise RuntimeError(
                    f"Entry {entry} in grouping of input data but not present in as a group in the inputted data."
                )

    return cats, cols


def _convert_data(data, inf_ok=False, min_len=1):
    """
    Convert inputted 1D data set into NumPy array of floats.
    All nan's are dropped.

    Parameters
    ----------
    data : int, float, or array_like
        Input data, to be converted.
    inf_ok : bool, default False
        If True, np.inf values are allowed in the arrays.
    min_len : int, default 1
        Minimum length of array.

    Returns
    -------
    output : ndarray
        `data` as a one-dimensional NumPy array, dtype float.
    """
    # If it's scalar, convert to array
    if np.isscalar(data):
        data = np.array([data], dtype=np.float)

    # Convert data to NumPy array
    data = np.array(data, dtype=np.float)

    # Make sure it is 1D
    if len(data.shape) != 1:
        raise RuntimeError("Input must be a 1D array or Pandas series.")

    # Remove NaNs
    data = data[~np.isnan(data)]

    # Check for infinite entries
    if not inf_ok and np.isinf(data).any():
        raise RuntimeError("All entries must be finite.")

    # Check to minimal length
    if len(data) < min_len:
        raise RuntimeError(
            "Array must have at least {0:d} non-NaN entries.".format(min_len)
        )

    return data
