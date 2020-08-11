.. _purpose:


Purpose and contents
====================

This package generates plots of data with the following properties:

- One variable is quantitative
- All other variables of interest, if any, are categorical.

We call this subclass of data sets "one quantitative/*n* categorical," or "1QNC." These 1QNC data sets are ubiquitous in the biological sciences, which was the primary motivation for making this package in the first place.

There are four types of plots that Bokeh-catplot can generate.

- **Plots with a categorical axis**

    + Box plots
    + Strip plots    
    
- **Plots without a categorical axis**

    + Histograms
    + `ECDFs <https://en.wikipedia.org/wiki/Empirical_distribution_function)>`_

This package was originally developed to enable rapid generation of these plots, particularly ECDFs using `Bokeh <https://bokeh.pydata.org/>`_, a powerful plotting library. Since its initial development, `HoloViews <https://holoviews.org/>`_ has emerged as an excellent high-level plotting package that can use Bokeh to render plots. Much of what this module provides is available in HoloViews, and you can see comparisons :ref:`holoviews.ipynb`. Nonetheless, I have still found this package useful to quickly generate useful plots. Importantly, generating ECDFs with bootstrapped confidence intervals is available in this package, but is nontrivial to do using HoloViews.

