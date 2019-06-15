import warnings

import numpy as np
import pandas as pd
import numba

import bokeh.models
import bokeh.palettes
import bokeh.plotting

from . import utils


def strip(
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
    show_legend=False,
    horizontal=False,
    color_column=None,
    val_axis_type=None,
    tooltips=None,
    marker="circle",
    jitter=False,
    marker_kwargs=None,
    jitter_kwargs=None,
    **kwargs,
):
    """
    Make a strip plot from a tidy DataFrame.

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
        the ordering of the categorical variables on the categorical
        axis and legend. If None, the categories appear in the order in
        which they appeared in the inputted data frame.
    p : bokeh.plotting.Figure instance, or None (default)
        If None, create a new figure. Otherwise, populate the existing
        figure `p`.
    show_legend : bool, default False
        If True, display legend.
    horizontal : bool, default False
        If true, the categorical axis is the vertical axis.
    color_column : str, default None
        Column of `data` to use in determining color of glyphs. If None,
        then `cats` is used.
    val_axis_type : str, default 'linear'
        Type of scaling for the quantitative axis, either 'linear' or
        'log'.
    tooltips : list of 2-tuples
        Specification for tooltips as per Bokeh specifications. For
        example, if we want `col1` and `col2` tooltips, we can use
        `tooltips=[('label 1': '@col1'), ('label 2': '@col2')]`.
    marker : str, default 'circle'
        Name of marker to be used in the plot (ignored if `formal` is
        False). Must be one of['asterisk', 'circle', 'circle_cross',
        'circlex', 'cross', 'dash', 'diamond', 'diamond_cross', 'hex',
        'inverted_triangle', 'square', 'square_cross', 'square_x',
        'triangle', 'x']
    jitter : bool, default False
        If True, apply a jitter transform to the glyphs.
    marker_kwargs : dict
        Keyword arguments to pass when adding markers to the plot.
        ["x", "y", "source", "cat", "legend"] are note allowed because
        they are determined by other inputs.
    jitter_kwargs : dict
        Keyword arguments to be passed to `bokeh.transform.jitter()`. If
        not specified, default is
        `{'distribution': 'normal', 'width': 0.1}`. If the user
        specifies `{'distribution': 'uniform'}`, the `'width'` entry is
        adjusted to 0.4.
    kwargs
        Any kwargs to be passed to `bokeh.plotting.figure()` when
        instantiating the figure.

    Returns
    -------
    output : bokeh.plotting.Figure instance
        Plot populated with jitter plot.
    """
    data, cats, show_legend = utils._data_cats(data, cats, show_legend)

    cats, cols = utils._check_cat_input(
        data, cats, val, color_column, tooltips, palette, order, kwargs
    )

    grouped = data.groupby(cats, sort=False)

    if p is None:
        p, factors, color_factors = _cat_figure(
            data, grouped, val, order, color_column, horizontal, val_axis_type, kwargs
        )
    else:
        if type(p.x_range) == bokeh.models.ranges.FactorRange and horizontal:
            raise RuntimeError(
                "Attempting to add glyphs to vertical plot at `horizontal` is True."
            )
        elif type(p.y_range) == bokeh.models.ranges.FactorRange and not horizontal:
            raise RuntimeError(
                "Attempting to add glyphs to horizontal plot at `horizontal` is False."
            )

        _, factors, color_factors = _get_cat_range(
            data, grouped, order, color_column, horizontal
        )

    if tooltips is not None:
        p.add_tools(bokeh.models.HoverTool(tooltips=tooltips))

    if jitter_kwargs is None:
        jitter_kwargs = dict(width=0.1, mean=0, distribution="normal")
    elif type(jitter_kwargs) != dict:
        raise RuntimeError("`jitter_kwargs` must be a dict.")
    elif "width" not in jitter_kwargs:
        if (
            "distribution" not in jitter_kwargs
            or jitter_kwargs["distribution"] == "uniform"
        ):
            jitter_kwargs["width"] = 0.4
        else:
            jitter_kwargs["width"] = 0.1

    if marker_kwargs is None:
        marker_kwargs = {}
    elif type(marker_kwargs) != dict:
        raise RuntimeError("`marker_kwargs` must be a dict.")

    if "color" not in marker_kwargs:
        if color_column is None:
            color_column = "cat"
        marker_kwargs["color"] = bokeh.transform.factor_cmap(
            color_column, palette=palette, factors=color_factors
        )

    if marker == "tick":
        marker = "dash"
    marker_fun = utils._get_marker(p, marker)

    if marker == "dash":
        if "angle" not in marker_kwargs and horizontal:
            marker_kwargs["angle"] = np.pi / 2
        if "size" not in marker_kwargs:
            if horizontal:
                marker_kwargs["size"] = p.plot_height * 0.25 / len(grouped)
            else:
                marker_kwargs["size"] = p.plot_width * 0.25 / len(grouped)

    source = _cat_source(data, cats, cols, color_column)

    if show_legend and "legend" not in marker_kwargs:
        marker_kwargs["legend"] = "__label"

    if horizontal:
        x = val
        if jitter:
            jitter_kwargs["range"] = p.y_range
            y = bokeh.transform.jitter("cat", **jitter_kwargs)
        else:
            y = "cat"
        p.ygrid.grid_line_color = None
    else:
        y = val
        if jitter:
            jitter_kwargs["range"] = p.x_range
            x = bokeh.transform.jitter("cat", **jitter_kwargs)
        else:
            x = "cat"
        p.xgrid.grid_line_color = None

    marker_fun(source=source, x=x, y=y, **marker_kwargs)

    return p


def box(
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
    horizontal=False,
    val_axis_type=None,
    box_width=0.4,
    whisker_caps=False,
    display_outliers=True,
    outlier_marker="circle",
    box_kwargs=None,
    median_kwargs=None,
    whisker_kwargs=None,
    outlier_kwargs=None,
    **kwargs,
):
    """
    Make a box-and-whisker plot from a tidy DataFrame.

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
        the ordering of the categorical variables on the categorical
        axis and legend. If None, the categories appear in the order in
        which they appeared in the inputted data frame.
    p : bokeh.plotting.Figure instance, or None (default)
        If None, create a new figure. Otherwise, populate the existing
        figure `p`.
    horizontal : bool, default False
        If true, the categorical axis is the vertical axis.
    val_axis_type : str, default 'linear'
        Type of scaling for the quantitative axis, either 'linear' or
        'log'.
    box_width : float, default 0.4
        Width of the boxes. A value of 1 means that the boxes take the
        entire space allotted.
    whisker_caps : bool, default False
        If True, put caps on whiskers. If False, omit caps.
    display_outliers : bool, default True
        If True, display outliers, otherwise suppress them. This should
        only be False when making an overlay with a strip plot.
    outlier_marker : str, default 'circle'
        Name of marker to be used in the plot (ignored if `formal` is
        False). Must be one of['asterisk', 'circle', 'circle_cross',
        'circlex', 'cross', 'dash', 'diamond', 'diamond_cross', 'hex',
        'inverted_triangle', 'square', 'square_cross', 'square_x',
        'triangle', 'x']
    box_kwargs : dict, default None
        A dictionary of kwargs to be passed into `p.hbar()` or
        `p.vbar()` when constructing the boxes for the box plot.
    median_kwargs : dict, default None
        A dictionary of kwargs to be passed into `p.hbar()` or
        `p.vbar()` when constructing the median line for the box plot.
    whisker_kwargs : dict, default None
        A dictionary of kwargs to be passed into `p.segment()`
        when constructing the whiskers for the box plot.
    outlier_kwargs : dict, default None
        A dictionary of kwargs to be passed into `p.circle()`
        when constructing the outliers for the box plot.
    kwargs
        Kwargs that are passed to bokeh.plotting.figure() in contructing
        the figure.

    Returns
    -------
    output : bokeh.plotting.Figure instance
        Plot populated with box-and-whisker plot.

    Notes
    -----
    .. Uses the Tukey convention for box plots. The top and bottom of
       the box are respectively the 75th and 25th percentiles of the
       data. The line in the middle of the box is the median. The top
       whisker extends to the maximum of the set of data points that are
       less than 1.5 times the IQR beyond the top of the box, with an
       analogous definition for the lower whisker. Data points not
       between the ends of the whiskers are considered outliers and are
       plotted as individual points.
    """
    data, cats, _ = utils._data_cats(data, cats, False)

    cats, cols = utils._check_cat_input(
        data, cats, val, None, None, palette, order, box_kwargs
    )

    if outlier_kwargs is None:
        outlier_kwargs = dict()
    elif type(outlier_kwargs) != dict:
        raise RuntimeError("`outlier_kwargs` must be a dict.")

    if box_kwargs is None:
        box_kwargs = {"line_color": None}
    elif type(box_kwargs) != dict:
        raise RuntimeError("`box_kwargs` must be a dict.")
    elif "line_color" not in box_kwargs:
        box_kwargs["line_color"] = None

    if whisker_kwargs is None:
        if "fill_color" in box_kwargs:
            whisker_kwargs = {"line_color": box_kwargs["fill_color"]}
        else:
            whisker_kwargs = {"line_color": "black"}
    elif type(whisker_kwargs) != dict:
        raise RuntimeError("`whisker_kwargs` must be a dict.")

    if median_kwargs is None:
        median_kwargs = {"line_color": "white"}
    elif type(median_kwargs) != dict:
        raise RuntimeError("`median_kwargs` must be a dict.")
    elif "line_color" not in median_kwargs:
        median_kwargs["line_color"] = white

    if horizontal:
        if "height" in box_kwargs:
            warnings.warn("'height' entry in `box_kwargs` ignored; using `box_width`.")
            del box_kwargs["height"]
    else:
        if "width" in box_kwargs:
            warnings.warn("'width' entry in `box_kwargs` ignored; using `box_width`.")
            del box_kwargs["width"]

    grouped = data.groupby(cats, sort=False)

    if p is None:
        p, factors, color_factors = _cat_figure(
            data, grouped, val, order, None, horizontal, val_axis_type, kwargs
        )
    else:
        _, factors, color_factors = _get_cat_range(
            data, grouped, order, None, horizontal
        )

    marker_fun = utils._get_marker(p, outlier_marker)

    source_box, source_outliers = _box_source(data, cats, val, cols)

    if "color" in outlier_kwargs:
        if "line_color" in outlier_kwargs or "fill_color" in outlier_kwargs:
            raise RuntimeError(
                "If `color` is in `outlier_kwargs`, `line_color` and `fill_color` cannot be."
            )
    else:
        if "fill_color" in box_kwargs:
            if "fill_color" not in outlier_kwargs:
                outlier_kwargs["fill_color"] = box_kwargs["fill_color"]
            if "line_color" not in outlier_kwargs:
                outlier_kwargs["line_color"] = box_kwargs["fill_color"]
        else:
            if "fill_color" not in outlier_kwargs:
                outlier_kwargs["fill_color"] = bokeh.transform.factor_cmap(
                    "cat", palette=palette, factors=factors
                )
            if "line_color" not in outlier_kwargs:
                outlier_kwargs["line_color"] = bokeh.transform.factor_cmap(
                    "cat", palette=palette, factors=factors
                )

    if "fill_color" not in box_kwargs:
        box_kwargs["fill_color"] = bokeh.transform.factor_cmap(
            "cat", palette=palette, factors=factors
        )

    if horizontal:
        p.segment(
            source=source_box,
            y0="cat",
            y1="cat",
            x0="top",
            x1="top_whisker",
            **whisker_kwargs,
        )
        p.segment(
            source=source_box,
            y0="cat",
            y1="cat",
            x0="bottom",
            x1="bottom_whisker",
            **whisker_kwargs,
        )
        if whisker_caps:
            p.hbar(
                source=source_box,
                y="cat",
                left="top_whisker",
                right="top_whisker",
                height=box_width / 4,
                **whisker_kwargs,
            )
            p.hbar(
                source=source_box,
                y="cat",
                left="bottom_whisker",
                right="bottom_whisker",
                height=box_width / 4,
                **whisker_kwargs,
            )
        p.hbar(
            source=source_box,
            y="cat",
            left="bottom",
            right="top",
            height=box_width,
            **box_kwargs,
        )
        p.hbar(
            source=source_box,
            y="cat",
            left="middle",
            right="middle",
            height=box_width,
            **median_kwargs,
        )
        if display_outliers:
            marker_fun(source=source_outliers, y="cat", x=val, **outlier_kwargs)
        p.ygrid.grid_line_color = None
    else:
        p.segment(
            source=source_box,
            x0="cat",
            x1="cat",
            y0="top",
            y1="top_whisker",
            **whisker_kwargs,
        )
        p.segment(
            source=source_box,
            x0="cat",
            x1="cat",
            y0="bottom",
            y1="bottom_whisker",
            **whisker_kwargs,
        )
        if whisker_caps:
            p.vbar(
                source=source_box,
                x="cat",
                bottom="top_whisker",
                top="top_whisker",
                width=box_width / 4,
                **whisker_kwargs,
            )
            p.vbar(
                source=source_box,
                x="cat",
                bottom="bottom_whisker",
                top="bottom_whisker",
                width=box_width / 4,
                **whisker_kwargs,
            )
        p.vbar(
            source=source_box,
            x="cat",
            bottom="bottom",
            top="top",
            width=box_width,
            **box_kwargs,
        )
        p.vbar(
            source=source_box,
            x="cat",
            bottom="middle",
            top="middle",
            width=box_width,
            **median_kwargs,
        )
        if display_outliers:
            marker_fun(source=source_outliers, x="cat", y=val, **outlier_kwargs)
        p.xgrid.grid_line_color = None

    return p


def _get_cat_range(df, grouped, order, color_column, horizontal):
    if order is None:
        if isinstance(list(grouped.groups.keys())[0], tuple):
            factors = tuple(
                [tuple([str(k) for k in key]) for key in grouped.groups.keys()]
            )
        else:
            factors = tuple([str(key) for key in grouped.groups.keys()])
    else:
        if type(order[0]) in [list, tuple]:
            factors = tuple([tuple([str(k) for k in key]) for key in order])
        else:
            factors = tuple([str(entry) for entry in order])

    if horizontal:
        cat_range = bokeh.models.FactorRange(*(factors[::-1]))
    else:
        cat_range = bokeh.models.FactorRange(*factors)

    if color_column is None:
        color_factors = factors
    else:
        color_factors = tuple(sorted(list(df[color_column].unique().astype(str))))

    return cat_range, factors, color_factors


def _cat_figure(
    df, grouped, val, order, color_column, horizontal, val_axis_type, kwargs
):
    cat_range, factors, color_factors = _get_cat_range(
        df, grouped, order, color_column, horizontal
    )

    kwargs = utils._fig_dimensions(kwargs)

    if horizontal:
        if "x_axis_label" not in kwargs:
            kwargs["x_axis_label"] = val

        if "y_axis_type" in kwargs:
            warnings.warn("`y_axis_type` specified for categorical axis. Ignoring.")
            del kwargs["y_axis_type"]

        kwargs["y_range"] = cat_range

        if "x_axis_type" in kwargs:
            if val_axis_type is not None and val_axis_type != kwargs["x_axis_type"]:
                raise RuntimeError("Mismatch in `val_axis_type` and `x_axis_type`")
        elif val_axis_type is not None:
            kwargs["x_axis_type"] = val_axis_type
    else:
        if "y_axis_label" not in kwargs:
            kwargs["y_axis_label"] = val

        if "x_axis_type" in kwargs:
            warnings.warn("`x_axis_type` specified for categorical axis. Ignoring.")
            del kwargs["x_axis_type"]

        kwargs["x_range"] = cat_range

        if "y_axis_type" in kwargs:
            if val_axis_type is not None and val_axis_type != kwargs["y_axis_type"]:
                raise RuntimeError("Mismatch in `val_axis_type` and `y_axis_type`")
        elif val_axis_type is not None:
            kwargs["y_axis_type"] = val_axis_type

    return bokeh.plotting.figure(**kwargs), factors, color_factors


def _cat_source(df, cats, cols, color_column):
    cat_source, labels = utils._source_and_labels_from_cats(df, cats)

    if type(cols) in [list, tuple, pd.core.indexes.base.Index]:
        source_dict = {col: list(df[col].values) for col in cols}
    else:
        source_dict = {cols: list(df[cols].values)}

    source_dict["cat"] = cat_source
    if color_column in [None, "cat"]:
        source_dict["__label"] = labels
    else:
        source_dict["__label"] = list(df[color_column].astype(str).values)
        source_dict[color_column] = list(df[color_column].astype(str).values)

    return bokeh.models.ColumnDataSource(source_dict)


def _outliers(data):
    bottom, middle, top = np.percentile(data, [25, 50, 75])
    iqr = top - bottom
    outliers = data[(data > top + 1.5 * iqr) | (data < bottom - 1.5 * iqr)]
    return outliers


def _box_and_whisker(data):
    middle = data.median()
    bottom = data.quantile(0.25)
    top = data.quantile(0.75)
    iqr = top - bottom
    top_whisker = data[data <= top + 1.5 * iqr].max()
    bottom_whisker = data[data >= bottom - 1.5 * iqr].min()
    return pd.Series(
        {
            "middle": middle,
            "bottom": bottom,
            "top": top,
            "top_whisker": top_whisker,
            "bottom_whisker": bottom_whisker,
        }
    )


def _box_source(df, cats, val, cols):
    """Construct a data frame for making box plot."""
    # Need to reset index for use in slicing outliers
    df_source = df.reset_index(drop=True)

    if type(cats) in [list, tuple]:
        level = list(range(len(cats)))
    else:
        level = 0

    if cats is None:
        grouped = df_source
    else:
        grouped = df_source.groupby(cats, sort=False)

    # Data frame for boxes and whiskers
    df_box = grouped[val].apply(_box_and_whisker).unstack().reset_index()
    source_box = _cat_source(
        df_box, cats, ["middle", "bottom", "top", "top_whisker", "bottom_whisker"], None
    )

    # Data frame for outliers
    df_outliers = grouped[val].apply(_outliers).reset_index(level=level)
    df_outliers[cols] = df_source.loc[df_outliers.index, cols]
    source_outliers = _cat_source(df_outliers, cats, cols, None)

    return source_box, source_outliers
