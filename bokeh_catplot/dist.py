"""Visualization of how data are distributed, split or colored by a
categorical variable."""

import warnings

import numpy as np
import pandas as pd
import numba

import bokeh.models
import bokeh.palettes
import bokeh.plotting

from . import utils


def ecdf(
    data=None,
    cats=None,
    val=None,
    palette=[
        "#4e79a7",
        "#f28e2b",
        "#e15759",
        "#76b7b2",
        "#59a14f",
        "#edc948",
        "#b07aa1",
        "#ff9da7",
        "#9c755f",
        "#bab0ac",
    ],
    order=None,
    p=None,
    show_legend=True,
    tooltips=None,
    complementary=False,
    kind="collection",
    formal=False,
    conf_int=False,
    ptiles=[2.5, 97.5],
    n_bs_reps=1000,
    click_policy="hide",
    marker="circle",
    marker_kwargs=None,
    conf_int_kwargs=None,
    **kwargs,
):
    """
    Parameters
    ----------
    data : Pandas DataFrame
        DataFrame containing tidy data for plotting.
    cats : hashable or list of hastables
        Name of column(s) to use as categorical variable(s).
    val : hashable
        Name of column to use as value variable.
    palette : list of strings of hex colors, or single hex string
        If a list, color palette to use. If a single string representing
        a hex color, all glyphs are colored with that color. Default is
        the default color cycle employed by Vega-Lite.
    order : list or None
        If not None, must be a list of unique group names when the input
        data frame is grouped by `cats`. The order of the list specifies
        the ordering of the categorical variables in the legend. If
        None, the categories appear in the order in which they appeared
        in the inputted data frame.
    p : bokeh.plotting.Figure instance, or None (default)
        If None, create a new figure. Otherwise, populate the existing
        figure `p`.
    show_legend : bool, default False
        If True, display legend.
    tooltips : list of 2-tuples
        Specification for tooltips as per Bokeh specifications. For
        example, if we want `col1` and `col2` tooltips, we can use
        `tooltips=[('label 1': '@col1'), ('label 2': '@col2')]`. Ignored
        if `formal` is True.
    complementary : bool, default False
        If True, plot the empirical complementary cumulative
        distribution function.
    kind : str, default 'collection'
        If 'collection', the figure is populated with a collection of
        ECDFs coded with colors based on the categorical variables. If
        'colored', the figure is populated with a single ECDF with
        circles colored based on the categorical variables.
    formal : bool, default False
        If True, make a plot of a formal ECDF (staircase). If False,
        plot the ECDF as dots located at the concave corners of the
        formal staircase ECDF.
    conf_int : bool, default False
        If True, display a confidence interval on the ECDF.
    ptiles : list, default [2.5, 97.5]
        The percentiles to use for the confidence interval. Ignored if
        `conf_int` is False.
    n_bs_reps : int, default 1000
        Number of bootstrap replicates to do to compute confidence
        interval. Ignored if `conf_int` is False.
    click_policy : str, default 'hide'
        Either 'hide', 'mute', or None; how the glyphs respond when the
        corresponding category is clicked in the legend.
    marker : str, default 'circle'
        Name of marker to be used in the plot (ignored if `formal` is
        False). Must be one of['asterisk', 'circle', 'circle_cross',
        'circlex', 'cross', 'dash', 'diamond', 'diamond_cross', 'hex',
        'inverted_triangle', 'square', 'square_cross', 'square_x',
        'triangle', 'x']
    marker_kwargs : dict
        Keyword arguments to be passed to `p.line()` if `formal`, or
        `p.circle()`.
    kwargs
        Any kwargs to be passed to `bokeh.plotting.figure()` when making
        the plot.

    Returns
    -------
    output : bokeh.plotting.Figure instance
        Plot populated with jitter plot or box plot.
    """
    data, cats, show_legend = utils._data_cats(data, cats, show_legend)

    cats, cols = utils._check_cat_input(
        data, cats, val, None, tooltips, palette, order, marker_kwargs
    )

    kwargs = utils._fig_dimensions(kwargs)

    if conf_int and "y_axis_type" in kwargs and kwargs["y_axis_type"] == "log":
        warnings.warn(
            "Cannot reliably draw confidence intervals with a y-axis on a log scale because zero cannot be represented. Omitting confidence interval."
        )
        conf_int = False
    if (
        conf_int
        and "x_axis_type" in kwargs
        and kwargs["x_axis_type"] == "log"
        and (df[val] <= 0).any()
    ):
        warnings.warn(
            "Cannot draw confidence intervals with a x-axis on a log scale because some values are negative. Any negative values will be omitted from the ECDF."
        )
        conf_int = False

    if marker_kwargs is None:
        marker_kwargs = {}

    y = "__ECCDF" if complementary else "__ECDF"

    if "y_axis_label" not in kwargs:
        if complementary:
            kwargs["y_axis_label"] = "ECCDF"
        else:
            kwargs["y_axis_label"] = "ECDF"

    if "x_axis_label" not in kwargs:
        kwargs["x_axis_label"] = val

    if marker_kwargs is None:
        marker_kwargs = {}
    if formal and "line_width" not in marker_kwargs:
        marker_kwargs["line_width"] = 2

    if conf_int_kwargs is None:
        conf_int_kwargs = {}
    if marker_kwargs is None:
        marker_kwargs = {}
    if "fill_alpha" not in conf_int_kwargs:
        conf_int_kwargs["fill_alpha"] = 0.5
    if "line_alpha" not in conf_int_kwargs and "line_color" not in conf_int_kwargs:
        conf_int_kwargs["line_alpha"] = 0

    df = data.copy()
    if kind == "collection":
        if not formal:
            df[y] = df.groupby(cats)[val].transform(
                _ecdf_y, complementary=complementary
            )
    elif kind == "colored":
        df[y] = df[val].transform(_ecdf_y, complementary=complementary)
        cols += [y]
    else:
        raise RuntimeError("`kind` must be in `['collection', 'colored']")

    _, df["__label"] = utils._source_and_labels_from_cats(df, cats)
    cols += ["__label"]

    if order is not None:
        if type(cats) in [list, tuple]:
            df["__sort"] = df.apply(lambda r: order.index(tuple(r[cats])), axis=1)
        else:
            df["__sort"] = df.apply(lambda r: order.index(r[cats]), axis=1)
        df = df.sort_values(by="__sort")

    if p is None:
        p = bokeh.plotting.figure(**kwargs)

    if not formal:
        marker_fun = utils._get_marker(p, marker)

    if tooltips is not None:
        if formal:
            warnings.warn(
                "Cannot have tooltips for formal ECDFs because there are not point to hover over. Omitting tooltips"
            )
        else:
            p.add_tools(bokeh.models.HoverTool(tooltips=tooltips))

    if kind == "collection":
        # Explicitly loop to enable click policies on the legend (not possible with factors)
        for i, (name, g) in enumerate(df.groupby(cats, sort=False)):
            if conf_int:
                conf_int_kwargs["fill_color"] = palette[i % len(palette)]
                conf_int_kwargs["legend"] = g["__label"].iloc[0]
                p = _ecdf_conf_int(
                    p,
                    g[val],
                    complementary=complementary,
                    n_bs_reps=n_bs_reps,
                    ptiles=ptiles,
                    **conf_int_kwargs,
                )

            marker_kwargs["color"] = palette[i % len(palette)]
            marker_kwargs["legend"] = g["__label"].iloc[0]
            if formal:
                p = _formal_ecdf(
                    p,
                    data=g[val],
                    complementary=False,
                    conf_int_kwargs=conf_int_kwargs,
                    marker_kwargs=marker_kwargs,
                )
            else:
                marker_fun(source=g, x=val, y=y, **marker_kwargs)
    elif kind == "colored":
        if formal:
            raise RuntimeError("Cannot have a formal ECDF with `kind='colored'.")

        if conf_int:
            if "fill_color" not in conf_int_kwargs:
                conf_int_kwargs["fill_color"] = "gray"

            p = _ecdf_conf_int(
                p,
                df[val],
                complementary=complementary,
                n_bs_reps=n_bs_reps,
                ptiles=ptiles,
                **conf_int_kwargs,
            )

        y = "__ECCDF" if complementary else "__ECDF"

        # Explicitly loop to enable click policies on the legend (not possible with factors)
        for i, (name, g) in enumerate(df.groupby(cats, sort=False)):
            source = bokeh.models.ColumnDataSource(g[cols])
            mkwargs = marker_kwargs
            mkwargs["legend"] = g["__label"].iloc[0]
            mkwargs["color"] = palette[i % len(palette)]
            marker_fun(source=source, x=val, y=y, **mkwargs)

    return _ecdf_legend(p, complementary, click_policy, show_legend)


def histogram(
    data=None,
    cats=None,
    val=None,
    palette=[
        "#4e79a7",
        "#f28e2b",
        "#e15759",
        "#76b7b2",
        "#59a14f",
        "#edc948",
        "#b07aa1",
        "#ff9da7",
        "#9c755f",
        "#bab0ac",
    ],
    order=None,
    show_legend=None,
    p=None,
    bins="freedman-diaconis",
    density=False,
    kind="step_filled",
    click_policy="hide",
    line_kwargs=None,
    fill_kwargs=None,
    **kwargs,
):
    """
    Make a plot of histograms of a data set.

    Parameters
    ----------
    data : Pandas DataFrame
        DataFrame containing tidy data for plotting.
    cats : hashable or list of hastables
        Name of column(s) to use as categorical variable(s).
    val : hashable
        Name of column to use as value variable.
    palette : list of strings of hex colors, or single hex string
        If a list, color palette to use. If a single string representing
        a hex color, all glyphs are colored with that color. Default is
        the default color cycle employed by Vega-Lite.
    order : list or None
        If not None, must be a list of unique group names when the input
        data frame is grouped by `cats`. The order of the list specifies
        the ordering of the categorical variables in the legend. If
        None, the categories appear in the order in which they appeared
        in the inputted data frame.
    p : bokeh.plotting.Figure instance, or None (default)
        If None, create a new figure. Otherwise, populate the existing
        figure `p`.
    show_legend : bool, default False
        If True, display legend.
    bins : int, array_like, or str, default 'freedman-diaconis'
        If int or array_like, setting for `bins` kwarg to be passed to
        `np.histogram()`. If 'exact', then each unique value in the
        data gets its own bin. If 'integer', then integer data is
        assumed and each integer gets its own bin. If 'sqrt', uses the
        square root rule to determine number of bins. If
        `freedman-diaconis`, uses the Freedman-Diaconis rule for number
        of bins.
    density : bool, default False
        If True, normalize the histograms. Otherwise, base the
        histograms on counts.
    kind : str, default 'step_filled'
        The kind of histogram to display. Allowed values are 'step' and
        'step_filled'.
    click_policy : str, default 'hide'
        Either 'hide', 'mute', or None; how the glyphs respond when the
        corresponding category is clicked in the legend.
    line_kwargs : dict
        Keyword arguments to pass to `p.line()` in constructing the
        histograms. By default, {"line_width": 2}.
    fill_kwargs : dict
        Keyword arguments to pass to `p.patch()` when making the fill
        for the step-filled histogram. Ignored if `kind = 'step'`. By
        default {"fill_alpha": 0.3, "line_alpha": 0}.
    kwargs
        Any kwargs to be passed to `bokeh.plotting.figure()` when making
        the plot.

    Returns
    -------
    output : Bokeh figure
        Figure populated with histograms.
    """
    df, cats, show_legend = utils._data_cats(data, cats, show_legend)

    if show_legend is None:
        if cats is None:
            show_legend = False
        else:
            show_legend = True

    if type(bins) == str and bins not in [
        "integer",
        "exact",
        "sqrt",
        "freedman-diaconis",
    ]:
        raise RuntimeError("Invalid bin specification.")

    if cats is None:
        df["__cat"] = "__dummy_cat"
        if show_legend:
            raise RuntimeError("No legend to show if `cats` is None.")
        if order is not None:
            raise RuntimeError("No `order` is allowed if `cats` is None.")
        cats = "__cat"

    cats, cols = utils._check_cat_input(
        df, cats, val, None, None, palette, order, kwargs
    )

    kwargs = utils._fig_dimensions(kwargs)

    if line_kwargs is None:
        line_kwargs = {"line_width": 2}
    if fill_kwargs is None:
        fill_kwargs = {}
    if "fill_alpha" not in fill_kwargs:
        fill_kwargs["fill_alpha"] = 0.3
    if "line_alpha" not in fill_kwargs:
        fill_kwargs["line_alpha"] = 0

    _, df["__label"] = utils._source_and_labels_from_cats(df, cats)
    cols += ["__label"]

    if order is not None:
        if type(cats) in [list, tuple]:
            df["__sort"] = df.apply(lambda r: order.index(tuple(r[cats])), axis=1)
        else:
            df["__sort"] = df.apply(lambda r: order.index(r[cats]), axis=1)
        df = df.sort_values(by="__sort")

    if bins == "exact":
        a = np.unique(df[val])
        if len(a) == 1:
            bins = np.array([a[0] - 0.5, a[0] + 0.5])
        else:
            bins = np.concatenate(
                (
                    (a[0] - (a[1] - a[0]) / 2,),
                    (a[1:] + a[:-1]) / 2,
                    (a[-1] + (a[-1] - a[-2]) / 2,),
                )
            )
    elif bins == "integer":
        if np.any(df[val] != np.round(df[val])):
            raise RuntimeError("'integer' bins chosen, but data are not integer.")
        bins = np.arange(df[val].min() - 1, df[val].max() + 1) + 0.5

    if p is None:
        kwargs = utils._fig_dimensions(kwargs)

        if "x_axis_label" not in kwargs:
            kwargs["x_axis_label"] = val

        if "y_axis_label" not in kwargs:
            if density:
                kwargs["y_axis_label"] = "density"
            else:
                kwargs["y_axis_label"] = "count"
        if "y_range" not in kwargs:
            kwargs["y_range"] = bokeh.models.DataRange1d(start=0)

        p = bokeh.plotting.figure(**kwargs)

    # Explicitly loop to enable click policies on the legend (not possible with factors)
    for i, (name, g) in enumerate(df.groupby(cats, sort=False)):
        e0, f0 = _compute_histogram(g[val], bins, density)
        line_kwargs["color"] = palette[i % len(palette)]
        p.line(e0, f0, **line_kwargs, legend=g["__label"].iloc[0])

        if kind == "step_filled":
            x2 = [e0.min(), e0.max()]
            y2 = [0, 0]
            fill_kwargs["color"] = palette[i % len(palette)]
            p = utils._fill_between(
                p, e0, f0, x2, y2, legend=g["__label"].iloc[0], **fill_kwargs
            )

    if show_legend:
        p.legend.location = "top_right"
        p.legend.click_policy = click_policy
    else:
        p.legend.visible = False

    return p


def _formal_ecdf(p, data, complementary=False, conf_int_kwargs={}, marker_kwargs={}):
    """
    Create a plot of an ECDF.

    Parameters
    ----------
    data : array_like
        One-dimensional array of data. Nan's are ignored.
    fill_color : str, default 'lightgray'
        Color of the confidence interbal. Ignored if `conf_int` is
        False.
    fill_alpha : float, default 1
        Opacity of confidence interval. Ignored if `conf_int` is False.
    p : bokeh.plotting.Figure instance, or None (default)
        If None, create a new figure. Otherwise, populate the existing
        figure `p`.
    x_axis_label : str, default None
        Label for the x-axis. Ignored if `p` is not None.
    y_axis_label : str, default 'ECDF' or 'ECCDF'
        Label for the y-axis. Ignored if `p` is not None.
    title : str, default None
        Title of the plot. Ignored if `p` is not None.
    plot_height : int, default 300
        Height of plot, in pixels. Ignored if `p` is not None.
    plot_width : int, default 450
        Width of plot, in pixels. Ignored if `p` is not None.
    complementary : bool, default False
        If True, plot the empirical complementary cumulative
        distribution functon.
    x_axis_type : str, default 'linear'
        Either 'linear' or 'log'.
    y_axis_type : str, default 'linear'
        Either 'linear' or 'log'.
    kwargs
        Any kwargs to be passed to either p.circle or p.line, for
        `formal` being False or True, respectively.

    Returns
    -------
    output : bokeh.plotting.Figure instance
        Plot populated with ECDF.
    """
    # Extract data
    data = utils._convert_data(data)

    # Data points on ECDF
    x, y = _ecdf_vals(data, True, complementary)

    # Line of steps
    p.line(x, y, **marker_kwargs)

    # Rays for ends
    if complementary:
        p.ray(x[0], 1, None, np.pi, **maker_kwargs)
        p.ray(x[-1], 0, None, 0, **marker_kwargs)
    else:
        p.ray(x[0], 0, None, np.pi, **marker_kwargs)
        p.ray(x[-1], 1, None, 0, **marker_kwargs)

    return p


def _ecdf_vals(data, formal=False, complementary=False):
    """Get x, y, values of an ECDF for plotting.
    Parameters
    ----------
    data : ndarray
        One dimensional Numpy array with data.
    formal : bool, default False
        If True, generate x and y values for formal ECDF (staircase). If
        False, generate x and y values for ECDF as dots.
    complementary : bool
        If True, return values for ECCDF.

    Returns
    -------
    x : ndarray
        x-values for plot
    y : ndarray
        y-values for plot
    """
    x = np.sort(data)
    y = np.arange(1, len(data) + 1) / len(data)

    if formal:
        x, y = _to_formal(x, y)
        if complementary:
            y = 1 - y
    elif complementary:
        y = 1 - y + 1 / len(y)

    return x, y


def _to_formal(x, y):
    """Convert to formal ECDF."""
    # Set up output arrays
    x_formal = np.empty(2 * len(x))
    y_formal = np.empty(2 * len(x))

    # y-values for steps
    y_formal[0] = 0
    y_formal[1::2] = y
    y_formal[2::2] = y[:-1]

    # x- values for steps
    x_formal[::2] = x
    x_formal[1::2] = x

    return x_formal, y_formal


def _ecdf_conf_int(
    p, data, complementary=False, n_bs_reps=1000, ptiles=[2.5, 97.5], **kwargs
):
    """Add an ECDF confidence interval to a plot."""
    data = utils._convert_data(data)
    x_plot = np.sort(np.unique(data))
    bs_reps = np.array(
        [
            _ecdf_arbitrary_points(np.random.choice(data, size=len(data)), x_plot)
            for _ in range(n_bs_reps)
        ]
    )

    # Compute the confidence intervals
    ecdf_low, ecdf_high = np.percentile(np.array(bs_reps), ptiles, axis=0)

    # Make them formal
    _, ecdf_low = _to_formal(x=x_plot, y=ecdf_low)
    x_plot, ecdf_high = _to_formal(x=x_plot, y=ecdf_high)

    if complementary:
        p = utils._fill_between(
            p, x1=x_plot, y1=1 - ecdf_low, x2=x_plot, y2=1 - ecdf_high, **kwargs
        )
    else:
        p = utils._fill_between(
            p, x1=x_plot, y1=ecdf_low, x2=x_plot, y2=ecdf_high, **kwargs
        )

    return p


def _ecdf_y(data, complementary=False):
    """Give y-values of an ECDF for an unsorted column in a data frame.

    Parameters
    ----------
    data : Pandas Series
        Series (or column of a DataFrame) from which to generate ECDF
        values
    complementary : bool, default False
        If True, give the ECCDF values.

    Returns
    -------
    output : Pandas Series
        Corresponding y-values for an ECDF when plotted with dots.

    Notes
    -----
    .. This only works for plotting an ECDF with points, not for formal
       ECDFs
    """
    if complementary:
        return 1 - data.rank(method="first") / len(data) + 1 / len(data)
    else:
        return data.rank(method="first") / len(data)


@numba.njit
def _ecdf_arbitrary_points(data, x):
    """Give the value of an ECDF at arbitrary points x."""
    y = np.arange(len(data) + 1) / len(data)
    return y[np.searchsorted(np.sort(data), x, side="right")]


@numba.njit
def _y_ecdf(data, x):
    y = np.arange(len(data) + 1) / len(data)
    return y[np.searchsorted(np.sort(data), x, side="right")]


@numba.njit
def _draw_ecdf_bootstrap(L, n, n_bs_reps=100000):
    x = np.arange(L + 1)
    ys = np.empty((n_bs_reps, len(x)))
    for i in range(n_bs_reps):
        draws = np.random.randint(0, L + 1, size=n)
        ys[i, :] = _y_ecdf(draws, x)
    return ys


def _ecdf_legend(p, complementary, click_policy, show_legend):
    if show_legend:
        if complementary:
            p.legend.location = "top_right"
        else:
            p.legend.location = "bottom_right"
        p.legend.click_policy = click_policy
    else:
        p.legend.visible = False

    return p


def _compute_histogram(data, bins, density):
    if bins == "sqrt":
        bins = int(np.ceil(np.sqrt(len(data))))
    elif bins == "freedman-diaconis":
        h = 2 * (np.percentile(data, 75) - np.percentile(data, 25)) / np.cbrt(len(data))
        bins = int(np.ceil((data.max() - data.min()) / h))

    f, e = np.histogram(data, bins=bins, density=density)
    e0 = np.empty(2 * len(e))
    f0 = np.empty(2 * len(e))
    e0[::2] = e
    e0[1::2] = e
    f0[0] = 0
    f0[-1] = 0
    f0[1:-1:2] = f
    f0[2:-1:2] = f

    return e0, f0
