
from numbers import Number
import numpy as np
import pandas as pd

from scipy.stats import gaussian_kde


from Contrastive_uncertainty.toy_replica.toy_general.callbacks.statistics_utils import _check_argument

class Histogram:
    """Univariate and bivariate histogram estimator."""
    def __init__(
        self,
        stat="count",
        bins="auto",
        binwidth=None,
        binrange=None,
        discrete=False,
        cumulative=False,
    ):
        """Initialize the estimator with its parameters.
        Parameters
        ----------
        stat : {"count", "frequency", "density", "probability", "percent"}
            Aggregate statistic to compute in each bin.
            - ``count`` shows the number of observations
            - ``frequency`` shows the number of observations divided by the bin width
            - ``density`` normalizes counts so that the area of the histogram is 1
            - ``probability`` normalizes counts so that the sum of the bar heights is 1
        bins : str, number, vector, or a pair of such values
            Generic bin parameter that can be the name of a reference rule,
            the number of bins, or the breaks of the bins.
            Passed to :func:`numpy.histogram_bin_edges`.
        binwidth : number or pair of numbers
            Width of each bin, overrides ``bins`` but can be used with
            ``binrange``.
        binrange : pair of numbers or a pair of pairs
            Lowest and highest value for bin edges; can be used either
            with ``bins`` or ``binwidth``. Defaults to data extremes.
        discrete : bool or pair of bools
            If True, set ``binwidth`` and ``binrange`` such that bin
            edges cover integer values in the dataset.
        cumulative : bool
            If True, return the cumulative statistic.
        """
        stat_choices = ["count", "frequency", "density", "probability", "percent"]
        _check_argument("stat", stat_choices, stat)

        self.stat = stat
        self.bins = bins
        self.binwidth = binwidth
        self.binrange = binrange
        self.discrete = discrete
        self.cumulative = cumulative

        self.bin_kws = None

    def _define_bin_edges(self, x, weights, bins, binwidth, binrange, discrete):
        """Inner function that takes bin parameters as arguments."""
        if binrange is None:
            start, stop = x.min(), x.max()
        else:
            start, stop = binrange

        if discrete:
            bin_edges = np.arange(start - .5, stop + 1.5)
        elif binwidth is not None:
            step = binwidth
            bin_edges = np.arange(start, stop + step, step)
        else:
            bin_edges = np.histogram_bin_edges(
                x, bins, binrange, weights,
            )
        return bin_edges

    def define_bin_params(self, x1, x2=None, weights=None, cache=True):
        """Given data, return numpy.histogram parameters to define bins."""
        if x2 is None:

            bin_edges = self._define_bin_edges(
                x1, weights, self.bins, self.binwidth, self.binrange, self.discrete,
            )

            if isinstance(self.bins, (str, Number)):
                n_bins = len(bin_edges) - 1
                bin_range = bin_edges.min(), bin_edges.max()
                bin_kws = dict(bins=n_bins, range=bin_range)
            else:
                bin_kws = dict(bins=bin_edges)

        else:

            bin_edges = []
            for i, x in enumerate([x1, x2]):

                # Resolve out whether bin parameters are shared
                # or specific to each variable

                bins = self.bins
                if not bins or isinstance(bins, (str, Number)):
                    pass
                elif isinstance(bins[i], str):
                    bins = bins[i]
                elif len(bins) == 2:
                    bins = bins[i]

                binwidth = self.binwidth
                if binwidth is None:
                    pass
                elif not isinstance(binwidth, Number):
                    binwidth = binwidth[i]

                binrange = self.binrange
                if binrange is None:
                    pass
                elif not isinstance(binrange[0], Number):
                    binrange = binrange[i]

                discrete = self.discrete
                if not isinstance(discrete, bool):
                    discrete = discrete[i]

                # Define the bins for this variable

                bin_edges.append(self._define_bin_edges(
                    x, weights, bins, binwidth, binrange, discrete,
                ))

            bin_kws = dict(bins=tuple(bin_edges))

        if cache:
            self.bin_kws = bin_kws

        return bin_kws

    def _eval_bivariate(self, x1, x2, weights):
        """Inner function for histogram of two variables."""
        bin_kws = self.bin_kws
        if bin_kws is None:
            bin_kws = self.define_bin_params(x1, x2, cache=False)

        density = self.stat == "density"

        hist, *bin_edges = np.histogram2d(
            x1, x2, **bin_kws, weights=weights, density=density
        )

        area = np.outer(
            np.diff(bin_edges[0]),
            np.diff(bin_edges[1]),
        )

        if self.stat == "probability":
            hist = hist.astype(float) / hist.sum()
        elif self.stat == "percent":
            hist = hist.astype(float) / hist.sum() * 100
        elif self.stat == "frequency":
            hist = hist.astype(float) / area

        if self.cumulative:
            if self.stat in ["density", "frequency"]:
                hist = (hist * area).cumsum(axis=0).cumsum(axis=1)
            else:
                hist = hist.cumsum(axis=0).cumsum(axis=1)

        return hist, bin_edges

    def _eval_univariate(self, x, weights):
        """Inner function for histogram of one variable."""
        bin_kws = self.bin_kws
        if bin_kws is None:
            bin_kws = self.define_bin_params(x, weights=weights, cache=False)

        density = self.stat == "density"
        hist, bin_edges = np.histogram(
            x, **bin_kws, weights=weights, density=density,
        )

        if self.stat == "probability":
            hist = hist.astype(float) / hist.sum()
        elif self.stat == "percent":
            hist = hist.astype(float) / hist.sum() * 100
        elif self.stat == "frequency":
            hist = hist.astype(float) / np.diff(bin_edges)

        if self.cumulative:
            if self.stat in ["density", "frequency"]:
                hist = (hist * np.diff(bin_edges)).cumsum()
            else:
                hist = hist.cumsum()

        return hist, bin_edges

    def __call__(self, x1, x2=None, weights=None):
        """Count the occurrences in each bin, maybe normalize."""
        if x2 is None:
            return self._eval_univariate(x1, weights)
        else:
            return self._eval_bivariate(x1, x2, weights)
