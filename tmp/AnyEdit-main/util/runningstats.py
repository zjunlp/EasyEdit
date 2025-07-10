"""
To use a runningstats object,

    1. Create the the desired stat object, e.g., `m = Mean()`
    2. Feed it batches via the add method, e.g., `m.add(batch)`
    3. Repeat step 2 any number of times.
    4. Read out the statistic of interest, e.g., `m.mean()`

Built-in runningstats objects include:

    Mean - produces mean().
    Variance - mean() and variance() and stdev().
    Covariance - mean(), covariance(), correlation(), variance(), stdev().
    SecondMoment - moment() is the non-mean-centered covariance, E[x x^T].
    Quantile - quantile(), min(), max(), median(), mean(), variance(), stdev().
    TopK - topk() returns (values, indexes).
    Bincount - bincount() histograms nonnegative integer data.
    IoU - intersection(), union(), iou() tally binary co-occurrences.
    History - history() returns concatenation of data.
    CrossCovariance - covariance between two signals, without self-covariance.
    CrossIoU - iou between two signals, without self-IoU.
    CombinedStat - aggregates any set of stats.

Add more running stats by subclassing the Stat class.

These statistics are vectorized along dim>=1, so stat.add()
should supply a two-dimensional input where the zeroth
dimension is the batch/sampling dimension and the first
dimension is the feature dimension.

The data type and device used matches the data passed to add();
for example, for higher-precision covariances, convert to double
before calling add().

It is common to want to compute and remember a statistic sampled
over a Dataset, computed in batches, possibly caching the computed
statistic in a file. The tally(stat, dataset, cache) handles
this pattern.  It takes a statistic, a dataset, and a cache filename
and sets up a data loader that can be run (or not, if cached) to
compute the statistic, adopting the convention that cached stats are
saved to and loaded from numpy npz files.
"""

import math
import os
import random
import struct

import numpy
import torch
from torch.utils.data.sampler import Sampler


def tally(stat, dataset, cache=None, quiet=False, **kwargs):
    """
    To use tally, write code like the following.

        stat = Mean()
        ds = MyDataset()
        for batch in tally(stat, ds, cache='mymean.npz', batch_size=50):
           stat.add(batch)
        mean = stat.mean()

    The first argument should be the Stat being computed. After the
    loader is exhausted, tally will bring this stat to the cpu and
    cache it (if a cache is specified).

    The dataset can be a torch Dataset or a plain Tensor, or it can
    be a callable that returns one of those.

    Details on caching via the cache= argument:

        If the given filename cannot be loaded, tally will leave the
        statistic object empty and set up a DataLoader object so that
        the loop can be run.  After the last iteration of the loop, the
        completed statistic will be moved to the cpu device and also
        saved in the cache file.

        If the cached statistic can be loaded from the given file, tally
        will not set up the data loader and instead will return a fully
        loaded statistic object (on the cpu device) and an empty list as
        the loader.

        The `with cache_load_enabled(False):` context manager can
        be used to disable loading from the cache.

    If needed, a DataLoader will be created to wrap the dataset:

        Keyword arguments of tally are passed to the DataLoader,
        so batch_size, num_workers, pin_memory, etc. can be specified.

    Subsampling is supported via sample_size= and random_sample=:

        If sample_size=N is specified, rather than loading the whole
        dataset, only the first N items are sampled.  If additionally
        random_sample=S is specified, the pseudorandom seed S will be
        used to select a fixed psedorandom sample of size N to sample.
    """
    assert isinstance(stat, Stat)
    args = {}
    for k in ["sample_size"]:
        if k in kwargs:
            args[k] = kwargs[k]
    cached_state = load_cached_state(cache, args, quiet=quiet)
    if cached_state is not None:
        stat.load_state_dict(cached_state)

        def empty_loader():
            return
            yield

        return empty_loader()
    loader = make_loader(dataset, **kwargs)

    def wrapped_loader():
        yield from loader
        stat.to_(device="cpu")
        if cache is not None:
            save_cached_state(cache, stat, args)

    return wrapped_loader()


class cache_load_enabled:
    """
    When used as a context manager, cache_load_enabled(False) will prevent
    tally from loading cached statsitics, forcing them to be recomputed.
    """

    def __init__(self, enabled=True):
        self.prev = False
        self.enabled = enabled

    def __enter__(self):
        global global_load_cache_enabled
        self.prev = global_load_cache_enabled
        global_load_cache_enabled = self.enabled

    def __exit__(self, exc_type, exc_value, traceback):
        global global_load_cache_enabled
        global_load_cache_enabled = self.prev


class Stat:
    """
    Abstract base class for a running pytorch statistic.
    """

    def __init__(self, state):
        """
        By convention, all Stat subclasses can be initialized by passing
        state=; and then they will initialize by calling load_state_dict.
        """
        self.load_state_dict(resolve_state_dict(state))

    def add(self, x, *args, **kwargs):
        """
        Observes a batch of samples to be incorporated into the statistic.
        Dimension 0 should be the batch dimension, and dimension 1 should
        be the feature dimension of the pytorch tensor x.
        """
        pass

    def load_state_dict(self, d):
        """
        Loads this Stat from a dictionary of numpy arrays as saved
        by state_dict.
        """
        pass

    def state_dict(self):
        """
        Saves this Stat as a dictionary of numpy arrays that can be
        stored in an npz or reloaded later using load_state_dict.
        """
        return {}

    def save(self, filename):
        """
        Saves this stat as an npz file containing the state_dict.
        """
        save_cached_state(filename, self, {})

    def load(self, filename):
        """
        Loads this stat from an npz file containing a saved state_dict.
        """
        self.load_state_dict(load_cached_state(filename, {}, quiet=True, throw=True))

    def to_(self, device):
        """
        Moves this Stat to the given device.
        """
        pass

    def cpu_(self):
        """
        Moves this Stat to the cpu device.
        """
        self.to_("cpu")

    def cuda_(self):
        """
        Moves this Stat to the default cuda device.
        """
        self.to_("cuda")

    def _normalize_add_shape(self, x, attr="data_shape"):
        """
        Flattens input data to 2d.
        """
        if not torch.is_tensor(x):
            x = torch.tensor(x)
        if len(x.shape) < 1:
            x = x.view(-1)
        data_shape = getattr(self, attr, None)
        if data_shape is None:
            data_shape = x.shape[1:]
            setattr(self, attr, data_shape)
        else:
            assert x.shape[1:] == data_shape
        return x.view(x.shape[0], int(numpy.prod(data_shape)))

    def _restore_result_shape(self, x, attr="data_shape"):
        """
        Restores output data to input data shape.
        """
        data_shape = getattr(self, attr, None)
        if data_shape is None:
            return x
        return x.view(data_shape * len(x.shape))


class Mean(Stat):
    """
    Running mean.
    """

    def __init__(self, state=None):
        if state is not None:
            return super().__init__(state)
        self.count = 0
        self.batchcount = 0
        self._mean = None
        self.data_shape = None

    def add(self, a):
        a = self._normalize_add_shape(a)
        if len(a) == 0:
            return
        batch_count = a.shape[0]
        batch_mean = a.sum(0) / batch_count
        self.batchcount += 1
        # Initial batch.
        if self._mean is None:
            self.count = batch_count
            self._mean = batch_mean
            return
        # Update a batch using Chan-style update for numerical stability.
        self.count += batch_count
        new_frac = float(batch_count) / self.count
        # Update the mean according to the batch deviation from the old mean.
        delta = batch_mean.sub_(self._mean).mul_(new_frac)
        self._mean.add_(delta)

    def size(self):
        return self.count

    def mean(self):
        return self._restore_result_shape(self._mean)

    def to_(self, device):
        if self._mean is not None:
            self._mean = self._mean.to(device)

    def load_state_dict(self, state):
        self.count = state["count"]
        self.batchcount = state["batchcount"]
        self._mean = torch.from_numpy(state["mean"])
        self.data_shape = (
            None if state["data_shape"] is None else tuple(state["data_shape"])
        )

    def state_dict(self):
        return dict(
            constructor=self.__module__ + "." + self.__class__.__name__ + "()",
            count=self.count,
            data_shape=self.data_shape and tuple(self.data_shape),
            batchcount=self.batchcount,
            mean=self._mean.cpu().numpy(),
        )


class NormMean(Mean):
    """
    Running average of the norm of input vectors
    """

    def __init__(self, state=None):
        super().__init__(state)

    def add(self, a):
        super().add(a.norm(dim=-1))


class Variance(Stat):
    """
    Running computation of mean and variance. Use this when you just need
    basic stats without covariance.
    """

    def __init__(self, state=None):
        if state is not None:
            return super().__init__(state)
        self.count = 0
        self.batchcount = 0
        self._mean = None
        self.v_cmom2 = None
        self.data_shape = None

    def add(self, a):
        a = self._normalize_add_shape(a)
        if len(a) == 0:
            return
        batch_count = a.shape[0]
        batch_mean = a.sum(0) / batch_count
        centered = a - batch_mean
        self.batchcount += 1
        # Initial batch.
        if self._mean is None:
            self.count = batch_count
            self._mean = batch_mean
            self.v_cmom2 = centered.pow(2).sum(0)
            return
        # Update a batch using Chan-style update for numerical stability.
        oldcount = self.count
        self.count += batch_count
        new_frac = float(batch_count) / self.count
        # Update the mean according to the batch deviation from the old mean.
        delta = batch_mean.sub_(self._mean).mul_(new_frac)
        self._mean.add_(delta)
        # Update the variance using the batch deviation
        self.v_cmom2.add_(centered.pow(2).sum(0))
        self.v_cmom2.add_(delta.pow_(2).mul_(new_frac * oldcount))

    def size(self):
        return self.count

    def mean(self):
        return self._restore_result_shape(self._mean)

    def variance(self, unbiased=True):
        return self._restore_result_shape(
            self.v_cmom2 / (self.count - (1 if unbiased else 0))
        )

    def stdev(self, unbiased=True):
        return self.variance(unbiased=unbiased).sqrt()

    def to_(self, device):
        if self._mean is not None:
            self._mean = self._mean.to(device)
        if self.v_cmom2 is not None:
            self.v_cmom2 = self.v_cmom2.to(device)

    def load_state_dict(self, state):
        self.count = state["count"]
        self.batchcount = state["batchcount"]
        self._mean = torch.from_numpy(state["mean"])
        self.v_cmom2 = torch.from_numpy(state["cmom2"])
        self.data_shape = (
            None if state["data_shape"] is None else tuple(state["data_shape"])
        )

    def state_dict(self):
        return dict(
            constructor=self.__module__ + "." + self.__class__.__name__ + "()",
            count=self.count,
            data_shape=self.data_shape and tuple(self.data_shape),
            batchcount=self.batchcount,
            mean=self._mean.cpu().numpy(),
            cmom2=self.v_cmom2.cpu().numpy(),
        )


class Covariance(Stat):
    """
    Running computation. Use this when the entire covariance matrix is needed,
    and when the whole covariance matrix fits in the GPU.

    Chan-style numerically stable update of mean and full covariance matrix.
    Chan, Golub. LeVeque. 1983. http://www.jstor.org/stable/2683386
    """

    def __init__(self, state=None):
        if state is not None:
            return super().__init__(state)
        self.count = 0
        self._mean = None
        self.cmom2 = None
        self.data_shape = None

    def add(self, a):
        a = self._normalize_add_shape(a)
        if len(a) == 0:
            return
        batch_count = a.shape[0]
        # Initial batch.
        if self._mean is None:
            self.count = batch_count
            self._mean = a.sum(0) / batch_count
            centered = a - self._mean
            self.cmom2 = centered.t().mm(centered)
            return
        # Update a batch using Chan-style update for numerical stability.
        self.count += batch_count
        # Update the mean according to the batch deviation from the old mean.
        delta = a - self._mean
        self._mean.add_(delta.sum(0) / self.count)
        delta2 = a - self._mean
        # Update the variance using the batch deviation
        self.cmom2.addmm_(mat1=delta.t(), mat2=delta2)

    def to_(self, device):
        if self._mean is not None:
            self._mean = self._mean.to(device)
        if self.cmom2 is not None:
            self.cmom2 = self.cmom2.to(device)

    def mean(self):
        return self._restore_result_shape(self._mean)

    def covariance(self, unbiased=True):
        return self._restore_result_shape(
            self.cmom2 / (self.count - (1 if unbiased else 0))
        )

    def correlation(self, unbiased=True):
        cov = self.cmom2 / (self.count - (1 if unbiased else 0))
        rstdev = cov.diag().sqrt().reciprocal()
        return self._restore_result_shape(rstdev[:, None] * cov * rstdev[None, :])

    def variance(self, unbiased=True):
        return self._restore_result_shape(
            self.cmom2.diag() / (self.count - (1 if unbiased else 0))
        )

    def stdev(self, unbiased=True):
        return self.variance(unbiased=unbiased).sqrt()

    def state_dict(self):
        return dict(
            constructor=self.__module__ + "." + self.__class__.__name__ + "()",
            count=self.count,
            data_shape=self.data_shape and tuple(self.data_shape),
            mean=self._mean.cpu().numpy(),
            cmom2=self.cmom2.cpu().numpy(),
        )

    def load_state_dict(self, state):
        self.count = state["count"]
        self._mean = torch.from_numpy(state["mean"])
        self.cmom2 = torch.from_numpy(state["cmom2"])
        self.data_shape = (
            None if state["data_shape"] is None else tuple(state["data_shape"])
        )


class SecondMoment(Stat):
    """
    Running computation. Use this when the entire non-centered 2nd-moment
    'covariance-like' matrix is needed, and when the whole matrix fits
    in the GPU.
    """

    def __init__(self, split_batch=True, state=None):
        if state is not None:
            return super().__init__(state)
        self.count = 0
        self.mom2 = None
        self.split_batch = split_batch

    def add(self, a):
        a = self._normalize_add_shape(a)
        if len(a) == 0:
            return
        # Initial batch reveals the shape of the data.
        if self.count == 0:
            self.mom2 = a.new(a.shape[1], a.shape[1]).zero_()
        batch_count = a.shape[0]
        # Update the covariance using the batch deviation
        self.count += batch_count
        self.mom2 += a.t().mm(a)

    def to_(self, device):
        if self.mom2 is not None:
            self.mom2 = self.mom2.to(device)

    def moment(self):
        return self.mom2 / self.count

    def state_dict(self):
        return dict(
            constructor=self.__module__ + "." + self.__class__.__name__ + "()",
            count=self.count,
            mom2=self.mom2.cpu().numpy(),
        )

    def load_state_dict(self, state):
        self.count = int(state["count"])
        self.mom2 = torch.from_numpy(state["mom2"])


class Bincount(Stat):
    """
    Running bincount.  The counted array should be an integer type with
    non-negative integers.
    """

    def __init__(self, state=None):
        if state is not None:
            return super().__init__(state)
        self.count = 0
        self._bincount = None

    def add(self, a, size=None):
        a = a.view(-1)
        bincount = a.bincount()
        if self._bincount is None:
            self._bincount = bincount
        elif len(self._bincount) < len(bincount):
            bincount[: len(self._bincount)] += self._bincount
            self._bincount = bincount
        else:
            self._bincount[: len(bincount)] += bincount
        if size is None:
            self.count += len(a)
        else:
            self.count += size

    def to_(self, device):
        self._bincount = self._bincount.to(device)

    def size(self):
        return self.count

    def bincount(self):
        return self._bincount

    def state_dict(self):
        return dict(
            constructor=self.__module__ + "." + self.__class__.__name__ + "()",
            count=self.count,
            bincount=self._bincount.cpu().numpy(),
        )

    def load_state_dict(self, dic):
        self.count = int(dic["count"])
        self._bincount = torch.from_numpy(dic["bincount"])


class CrossCovariance(Stat):
    """
    Covariance. Use this when an off-diagonal block of the covariance
    matrix is needed (e.g., when the whole covariance matrix does
    not fit in the GPU, this could use a quarter of the memory).

    Chan-style numerically stable update of mean and full covariance matrix.
    Chan, Golub. LeVeque. 1983. http://www.jstor.org/stable/2683386
    """

    def __init__(self, split_batch=True, state=None):
        if state is not None:
            return super().__init__(state)
        self.count = 0
        self._mean = None
        self.cmom2 = None
        self.v_cmom2 = None
        self.split_batch = split_batch

    def add(self, a, b):
        if len(a.shape) == 1:
            a = a[None, :]
            b = b[None, :]
        assert a.shape[0] == b.shape[0]
        if len(a.shape) > 2:
            a, b = [
                d.view(d.shape[0], d.shape[1], -1)
                .permute(0, 2, 1)
                .reshape(-1, d.shape[1])
                for d in [a, b]
            ]
        batch_count = a.shape[0]
        # Initial batch.
        if self._mean is None:
            self.count = batch_count
            self._mean = [d.sum(0) / batch_count for d in [a, b]]
            centered = [d - bm for d, bm in zip([a, b], self._mean)]
            self.v_cmom2 = [c.pow(2).sum(0) for c in centered]
            self.cmom2 = centered[0].t().mm(centered[1])
            return
        # Update a batch using Chan-style update for numerical stability.
        self.count += batch_count
        # Update the mean according to the batch deviation from the old mean.
        delta = [(d - bm) for d, bm in zip([a, b], self._mean)]
        for m, d in zip(self._mean, delta):
            m.add_(d.sum(0) / self.count)
        delta2 = [(d - bm) for d, bm in zip([a, b], self._mean)]
        # Update the cross-covariance using the batch deviation
        self.cmom2.addmm_(mat1=delta[0].t(), mat2=delta2[1])
        # Update the variance using the batch deviation
        for vc2, d, d2 in zip(self.v_cmom2, delta, delta2):
            vc2.add_((d * d2).sum(0))

    def mean(self):
        return self._mean

    def variance(self, unbiased=True):
        return [vc2 / (self.count - (1 if unbiased else 0)) for vc2 in self.v_cmom2]

    def stdev(self, unbiased=True):
        return [v.sqrt() for v in self.variance(unbiased=unbiased)]

    def covariance(self, unbiased=True):
        return self.cmom2 / (self.count - (1 if unbiased else 0))

    def correlation(self):
        covariance = self.covariance(unbiased=False)
        rstdev = [s.reciprocal() for s in self.stdev(unbiased=False)]
        cor = rstdev[0][:, None] * covariance * rstdev[1][None, :]
        # Remove NaNs
        cor[torch.isnan(cor)] = 0
        return cor

    def to_(self, device):
        self._mean = [m.to(device) for m in self._mean]
        self.v_cmom2 = [vcs.to(device) for vcs in self.v_cmom2]
        self.cmom2 = self.cmom2.to(device)

    def state_dict(self):
        return dict(
            constructor=self.__module__ + "." + self.__class__.__name__ + "()",
            count=self.count,
            mean_a=self._mean[0].cpu().numpy(),
            mean_b=self._mean[1].cpu().numpy(),
            cmom2_a=self.v_cmom2[0].cpu().numpy(),
            cmom2_b=self.v_cmom2[1].cpu().numpy(),
            cmom2=self.cmom2.cpu().numpy(),
        )

    def load_state_dict(self, state):
        self.count = int(state["count"])
        self._mean = [torch.from_numpy(state[f"mean_{k}"]) for k in "ab"]
        self.v_cmom2 = [torch.from_numpy(state[f"cmom2_{k}"]) for k in "ab"]
        self.cmom2 = torch.from_numpy(state["cmom2"])


def _float_from_bool(a):
    """
    Since pytorch only supports matrix multiplication on float,
    IoU computations are done using floating point types.

    This function binarizes the input (positive to True and
    nonpositive to False), and converts from bool to float.
    If the data is already a floating-point type, it leaves
    it keeps the same type; otherwise it uses float.
    """
    if a.dtype == torch.bool:
        return a.float()
    if a.dtype.is_floating_point:
        return a.sign().clamp_(0)
    return (a > 0).float()


class IoU(Stat):
    """
    Running computation of intersections and unions of all features.
    """

    def __init__(self, state=None):
        if state is not None:
            return super().__init__(state)
        self.count = 0
        self._intersection = None

    def add(self, a):
        assert len(a.shape) == 2
        a = _float_from_bool(a)
        if self._intersection is None:
            self._intersection = torch.mm(a.t(), a)
        else:
            self._intersection.addmm_(a.t(), a)
        self.count += len(a)

    def size(self):
        return self.count

    def intersection(self):
        return self._intersection

    def union(self):
        total = self._intersection.diagonal(0)
        return total[:, None] + total[None, :] - self._intersection

    def iou(self):
        return self.intersection() / (self.union() + 1e-20)

    def to_(self, _device):
        self._intersection = self._intersection.to(_device)

    def state_dict(self):
        return dict(
            constructor=self.__module__ + "." + self.__class__.__name__ + "()",
            count=self.count,
            intersection=self._intersection.cpu().numpy(),
        )

    def load_state_dict(self, state):
        self.count = int(state["count"])
        self._intersection = torch.tensor(state["intersection"])


class CrossIoU(Stat):
    """
    Running computation of intersections and unions of two binary vectors.
    """

    def __init__(self, state=None):
        if state is not None:
            return super().__init__(state)
        self.count = 0
        self._intersection = None
        self.total_a = None
        self.total_b = None

    def add(self, a, b):
        assert len(a.shape) == 2 and len(b.shape) == 2
        assert len(a) == len(b), f"{len(a)} vs {len(b)}"
        a = _float_from_bool(a)  # CUDA only supports mm on float...
        b = _float_from_bool(b)  # otherwise we would use integers.
        intersection = torch.mm(a.t(), b)
        asum = a.sum(0)
        bsum = b.sum(0)
        if self._intersection is None:
            self._intersection = intersection
            self.total_a = asum
            self.total_b = bsum
        else:
            self._intersection += intersection
            self.total_a += asum
            self.total_b += bsum
        self.count += len(a)

    def size(self):
        return self.count

    def intersection(self):
        return self._intersection

    def union(self):
        return self.total_a[:, None] + self.total_b[None, :] - self._intersection

    def iou(self):
        return self.intersection() / (self.union() + 1e-20)

    def to_(self, _device):
        self.total_a = self.total_a.to(_device)
        self.total_b = self.total_b.to(_device)
        self._intersection = self._intersection.to(_device)

    def state_dict(self):
        return dict(
            constructor=self.__module__ + "." + self.__class__.__name__ + "()",
            count=self.count,
            total_a=self.total_a.cpu().numpy(),
            total_b=self.total_b.cpu().numpy(),
            intersection=self._intersection.cpu().numpy(),
        )

    def load_state_dict(self, state):
        self.count = int(state["count"])
        self.total_a = torch.tensor(state["total_a"])
        self.total_b = torch.tensor(state["total_b"])
        self._intersection = torch.tensor(state["intersection"])


class Quantile(Stat):
    """
    Streaming randomized quantile computation for torch.

    Add any amount of data repeatedly via add(data).  At any time,
    quantile estimates be read out using quantile(q).

    Implemented as a sorted sample that retains at least r samples
    (by default r = 3072); the number of retained samples will grow to
    a finite ceiling as the data is accumulated.  Accuracy scales according
    to r: the default is to set resolution to be accurate to better than about
    0.1%, while limiting storage to about 50,000 samples.

    Good for computing quantiles of huge data without using much memory.
    Works well on arbitrary data with probability near 1.

    Based on the optimal KLL quantile algorithm by Karnin, Lang, and Liberty
    from FOCS 2016.  http://ieee-focs.org/FOCS-2016-Papers/3933a071.pdf
    """

    def __init__(self, r=3 * 1024, buffersize=None, seed=None, state=None):
        if state is not None:
            return super().__init__(state)
        self.depth = None
        self.dtype = None
        self.device = None
        resolution = r * 2  # sample array is at least half full before discard
        self.resolution = resolution
        # Default buffersize: 128 samples (and smaller than resolution).
        if buffersize is None:
            buffersize = min(128, (resolution + 7) // 8)
        self.buffersize = buffersize
        self.samplerate = 1.0
        self.data = None
        self.firstfree = [0]
        self.randbits = torch.ByteTensor(resolution)
        self.currentbit = len(self.randbits) - 1
        self.extremes = None
        self.count = 0
        self.batchcount = 0

    def size(self):
        return self.count

    def _lazy_init(self, incoming):
        self.depth = incoming.shape[1]
        self.dtype = incoming.dtype
        self.device = incoming.device
        self.data = [
            torch.zeros(
                self.depth, self.resolution, dtype=self.dtype, device=self.device
            )
        ]
        self.extremes = torch.zeros(self.depth, 2, dtype=self.dtype, device=self.device)
        self.extremes[:, 0] = float("inf")
        self.extremes[:, -1] = -float("inf")

    def to_(self, device):
        """Switches internal storage to specified device."""
        if device != self.device:
            old_data = self.data
            old_extremes = self.extremes
            self.data = [d.to(device) for d in self.data]
            self.extremes = self.extremes.to(device)
            self.device = self.extremes.device
            del old_data
            del old_extremes

    def add(self, incoming):
        if self.depth is None:
            self._lazy_init(incoming)
        assert len(incoming.shape) == 2
        assert incoming.shape[1] == self.depth, (incoming.shape[1], self.depth)
        self.count += incoming.shape[0]
        self.batchcount += 1
        # Convert to a flat torch array.
        if self.samplerate >= 1.0:
            self._add_every(incoming)
            return
        # If we are sampling, then subsample a large chunk at a time.
        self._scan_extremes(incoming)
        chunksize = int(math.ceil(self.buffersize / self.samplerate))
        for index in range(0, len(incoming), chunksize):
            batch = incoming[index : index + chunksize]
            sample = sample_portion(batch, self.samplerate)
            if len(sample):
                self._add_every(sample)

    def _add_every(self, incoming):
        supplied = len(incoming)
        index = 0
        while index < supplied:
            ff = self.firstfree[0]
            available = self.data[0].shape[1] - ff
            if available == 0:
                if not self._shift():
                    # If we shifted by subsampling, then subsample.
                    incoming = incoming[index:]
                    if self.samplerate >= 0.5:
                        # First time sampling - the data source is very large.
                        self._scan_extremes(incoming)
                    incoming = sample_portion(incoming, self.samplerate)
                    index = 0
                    supplied = len(incoming)
                ff = self.firstfree[0]
                available = self.data[0].shape[1] - ff
            copycount = min(available, supplied - index)
            self.data[0][:, ff : ff + copycount] = torch.t(
                incoming[index : index + copycount, :]
            )
            self.firstfree[0] += copycount
            index += copycount

    def _shift(self):
        index = 0
        # If remaining space at the current layer is less than half prev
        # buffer size (rounding up), then we need to shift it up to ensure
        # enough space for future shifting.
        while self.data[index].shape[1] - self.firstfree[index] < (
            -(-self.data[index - 1].shape[1] // 2) if index else 1
        ):
            if index + 1 >= len(self.data):
                return self._expand()
            data = self.data[index][:, 0 : self.firstfree[index]]
            data = data.sort()[0]
            if index == 0 and self.samplerate >= 1.0:
                self._update_extremes(data[:, 0], data[:, -1])
            offset = self._randbit()
            position = self.firstfree[index + 1]
            subset = data[:, offset::2]
            self.data[index + 1][:, position : position + subset.shape[1]] = subset
            self.firstfree[index] = 0
            self.firstfree[index + 1] += subset.shape[1]
            index += 1
        return True

    def _scan_extremes(self, incoming):
        # When sampling, we need to scan every item still to get extremes
        self._update_extremes(
            torch.min(incoming, dim=0)[0], torch.max(incoming, dim=0)[0]
        )

    def _update_extremes(self, minr, maxr):
        self.extremes[:, 0] = torch.min(
            torch.stack([self.extremes[:, 0], minr]), dim=0
        )[0]
        self.extremes[:, -1] = torch.max(
            torch.stack([self.extremes[:, -1], maxr]), dim=0
        )[0]

    def _randbit(self):
        self.currentbit += 1
        if self.currentbit >= len(self.randbits):
            self.randbits.random_(to=2)
            self.currentbit = 0
        return self.randbits[self.currentbit]

    def state_dict(self):
        state = dict(
            constructor=self.__module__ + "." + self.__class__.__name__ + "()",
            resolution=self.resolution,
            depth=self.depth,
            buffersize=self.buffersize,
            samplerate=self.samplerate,
            sizes=numpy.array([d.shape[1] for d in self.data]),
            extremes=self.extremes.cpu().detach().numpy(),
            size=self.count,
            batchcount=self.batchcount,
        )
        for i, (d, f) in enumerate(zip(self.data, self.firstfree)):
            state[f"data.{i}"] = d.cpu().detach().numpy()[:, :f].T
        return state

    def load_state_dict(self, state):
        self.resolution = int(state["resolution"])
        self.randbits = torch.ByteTensor(self.resolution)
        self.currentbit = len(self.randbits) - 1
        self.depth = int(state["depth"])
        self.buffersize = int(state["buffersize"])
        self.samplerate = float(state["samplerate"])
        firstfree = []
        buffers = []
        for i, s in enumerate(state["sizes"]):
            d = state[f"data.{i}"]
            firstfree.append(d.shape[0])
            buf = numpy.zeros((d.shape[1], s), dtype=d.dtype)
            buf[:, : d.shape[0]] = d.T
            buffers.append(torch.from_numpy(buf))
        self.firstfree = firstfree
        self.data = buffers
        self.extremes = torch.from_numpy((state["extremes"]))
        self.count = int(state["size"])
        self.batchcount = int(state.get("batchcount", 0))
        self.dtype = self.extremes.dtype
        self.device = self.extremes.device

    def min(self):
        return self.minmax()[0]

    def max(self):
        return self.minmax()[-1]

    def minmax(self):
        if self.firstfree[0]:
            self._scan_extremes(self.data[0][:, : self.firstfree[0]].t())
        return self.extremes.clone()

    def median(self):
        return self.quantiles(0.5)

    def mean(self):
        return self.integrate(lambda x: x) / self.count

    def variance(self, unbiased=True):
        mean = self.mean()[:, None]
        return self.integrate(lambda x: (x - mean).pow(2)) / (
            self.count - (1 if unbiased else 0)
        )

    def stdev(self, unbiased=True):
        return self.variance(unbiased=unbiased).sqrt()

    def _expand(self):
        cap = self._next_capacity()
        if cap > 0:
            # First, make a new layer of the proper capacity.
            self.data.insert(
                0, torch.zeros(self.depth, cap, dtype=self.dtype, device=self.device)
            )
            self.firstfree.insert(0, 0)
        else:
            # Unless we're so big we are just subsampling.
            assert self.firstfree[0] == 0
            self.samplerate *= 0.5
        for index in range(1, len(self.data)):
            # Scan for existing data that needs to be moved down a level.
            amount = self.firstfree[index]
            if amount == 0:
                continue
            position = self.firstfree[index - 1]
            # Move data down if it would leave enough empty space there
            # This is the key invariant: enough empty space to fit half
            # of the previous level's buffer size (rounding up)
            if self.data[index - 1].shape[1] - (amount + position) >= (
                -(-self.data[index - 2].shape[1] // 2) if (index - 1) else 1
            ):
                self.data[index - 1][:, position : position + amount] = self.data[
                    index
                ][:, :amount]
                self.firstfree[index - 1] += amount
                self.firstfree[index] = 0
            else:
                # Scrunch the data if it would not.
                data = self.data[index][:, :amount]
                data = data.sort()[0]
                if index == 1:
                    self._update_extremes(data[:, 0], data[:, -1])
                offset = self._randbit()
                scrunched = data[:, offset::2]
                self.data[index][:, : scrunched.shape[1]] = scrunched
                self.firstfree[index] = scrunched.shape[1]
        return cap > 0

    def _next_capacity(self):
        cap = int(math.ceil(self.resolution * (0.67 ** len(self.data))))
        if cap < 2:
            return 0
        # Round up to the nearest multiple of 8 for better GPU alignment.
        cap = -8 * (-cap // 8)
        return max(self.buffersize, cap)

    def _weighted_summary(self, sort=True):
        if self.firstfree[0]:
            self._scan_extremes(self.data[0][:, : self.firstfree[0]].t())
        size = sum(self.firstfree)
        weights = torch.FloatTensor(size)  # Floating point
        summary = torch.zeros(self.depth, size, dtype=self.dtype, device=self.device)
        index = 0
        for level, ff in enumerate(self.firstfree):
            if ff == 0:
                continue
            summary[:, index : index + ff] = self.data[level][:, :ff]
            weights[index : index + ff] = 2.0**level
            index += ff
        assert index == summary.shape[1]
        if sort:
            summary, order = torch.sort(summary, dim=-1)
            weights = weights[order.view(-1).cpu()].view(order.shape)
            summary = torch.cat(
                [self.extremes[:, :1], summary, self.extremes[:, 1:]], dim=-1
            )
            weights = torch.cat(
                [
                    torch.zeros(weights.shape[0], 1),
                    weights,
                    torch.zeros(weights.shape[0], 1),
                ],
                dim=-1,
            )
        return (summary, weights)

    def quantiles(self, quantiles):
        if not hasattr(quantiles, "cpu"):
            quantiles = torch.tensor(quantiles)
        qshape = quantiles.shape
        if self.count == 0:
            return torch.full((self.depth,) + qshape, torch.nan)
        summary, weights = self._weighted_summary()
        cumweights = torch.cumsum(weights, dim=-1) - weights / 2
        cumweights /= torch.sum(weights, dim=-1, keepdim=True)
        result = torch.zeros(
            self.depth, quantiles.numel(), dtype=self.dtype, device=self.device
        )
        # numpy is needed for interpolation
        nq = quantiles.view(-1).cpu().detach().numpy()
        ncw = cumweights.cpu().detach().numpy()
        nsm = summary.cpu().detach().numpy()
        for d in range(self.depth):
            result[d] = torch.tensor(
                numpy.interp(nq, ncw[d], nsm[d]), dtype=self.dtype, device=self.device
            )
        return result.view((self.depth,) + qshape)

    def integrate(self, fun):
        result = []
        for level, ff in enumerate(self.firstfree):
            if ff == 0:
                continue
            result.append(
                torch.sum(fun(self.data[level][:, :ff]) * (2.0**level), dim=-1)
            )
        if len(result) == 0:
            return None
        return torch.stack(result).sum(dim=0) / self.samplerate

    def readout(self, count=1001):
        return self.quantiles(torch.linspace(0.0, 1.0, count))

    def normalize(self, data):
        """
        Given input data as taken from the training distirbution,
        normalizes every channel to reflect quantile values,
        uniformly distributed, within [0, 1].
        """
        assert self.count > 0
        assert data.shape[0] == self.depth
        summary, weights = self._weighted_summary()
        cumweights = torch.cumsum(weights, dim=-1) - weights / 2
        cumweights /= torch.sum(weights, dim=-1, keepdim=True)
        result = torch.zeros_like(data).float()
        # numpy is needed for interpolation
        ndata = data.cpu().numpy().reshape((data.shape[0], -1))
        ncw = cumweights.cpu().numpy()
        nsm = summary.cpu().numpy()
        for d in range(self.depth):
            normed = torch.tensor(
                numpy.interp(ndata[d], nsm[d], ncw[d]),
                dtype=torch.float,
                device=data.device,
            ).clamp_(0.0, 1.0)
            if len(data.shape) > 1:
                normed = normed.view(*(data.shape[1:]))
            result[d] = normed
        return result


def sample_portion(vec, p=0.5):
    """
    Subsamples a fraction (given by p) of the given batch.  Used by
    Quantile when the data gets very very large.
    """
    bits = torch.bernoulli(
        torch.zeros(vec.shape[0], dtype=torch.uint8, device=vec.device), p
    )
    return vec[bits]


class TopK:
    """
    A class to keep a running tally of the the top k values (and indexes)
    of any number of torch feature components.  Will work on the GPU if
    the data is on the GPU.  Tracks largest by default, but tracks smallest
    if largest=False is passed.

    This version flattens all arrays to avoid crashes.
    """

    def __init__(self, k=100, largest=True, state=None):
        if state is not None:
            return super().__init__(state)
        self.k = k
        self.count = 0
        # This version flattens all data internally to 2-d tensors,
        # to avoid crashes with the current pytorch topk implementation.
        # The data is puffed back out to arbitrary tensor shapes on ouput.
        self.data_shape = None
        self.top_data = None
        self.top_index = None
        self.next = 0
        self.linear_index = 0
        self.perm = None
        self.largest = largest

    def add(self, data, index=None):
        """
        Adds a batch of data to be considered for the running top k.
        The zeroth dimension enumerates the observations.  All other
        dimensions enumerate different features.
        """
        if self.top_data is None:
            # Allocation: allocate a buffer of size 5*k, at least 10, for each.
            self.data_shape = data.shape[1:]
            feature_size = int(numpy.prod(self.data_shape))
            self.top_data = torch.zeros(
                feature_size, max(10, self.k * 5), out=data.new()
            )
            self.top_index = self.top_data.clone().long()
            self.linear_index = (
                0
                if len(data.shape) == 1
                else torch.arange(feature_size, out=self.top_index.new()).mul_(
                    self.top_data.shape[-1]
                )[:, None]
            )
        size = data.shape[0]
        sk = min(size, self.k)
        if self.top_data.shape[-1] < self.next + sk:
            # Compression: if full, keep topk only.
            self.top_data[:, : self.k], self.top_index[:, : self.k] = self.topk(
                sorted=False, flat=True
            )
            self.next = self.k
        # Pick: copy the top sk of the next batch into the buffer.
        # Currently strided topk is slow.  So we clone after transpose.
        # TODO: remove the clone() if it becomes faster.
        cdata = data.reshape(size, numpy.prod(data.shape[1:])).t().clone()
        td, ti = cdata.topk(sk, sorted=False, largest=self.largest)
        self.top_data[:, self.next : self.next + sk] = td
        if index is not None:
            ti = index[ti]
        else:
            ti = ti + self.count
        self.top_index[:, self.next : self.next + sk] = ti
        self.next += sk
        self.count += size

    def size(self):
        return self.count

    def topk(self, sorted=True, flat=False):
        """
        Returns top k data items and indexes in each dimension,
        with channels in the first dimension and k in the last dimension.
        """
        k = min(self.k, self.next)
        # bti are top indexes relative to buffer array.
        td, bti = self.top_data[:, : self.next].topk(
            k, sorted=sorted, largest=self.largest
        )
        # we want to report top indexes globally, which is ti.
        ti = self.top_index.view(-1)[(bti + self.linear_index).view(-1)].view(
            *bti.shape
        )
        if flat:
            return td, ti
        else:
            return (
                td.view(*(self.data_shape + (-1,))),
                ti.view(*(self.data_shape + (-1,))),
            )

    def to_(self, device):
        if self.top_data is not None:
            self.top_data = self.top_data.to(device)
        if self.top_index is not None:
            self.top_index = self.top_index.to(device)
        if isinstance(self.linear_index, torch.Tensor):
            self.linear_index = self.linear_index.to(device)

    def state_dict(self):
        return dict(
            constructor=self.__module__ + "." + self.__class__.__name__ + "()",
            k=self.k,
            count=self.count,
            largest=self.largest,
            data_shape=self.data_shape and tuple(self.data_shape),
            top_data=self.top_data.cpu().detach().numpy(),
            top_index=self.top_index.cpu().detach().numpy(),
            next=self.next,
            linear_index=(
                self.linear_index.cpu().numpy()
                if isinstance(self.linear_index, torch.Tensor)
                else self.linear_index
            ),
            perm=self.perm,
        )

    def load_state_dict(self, state):
        self.k = int(state["k"])
        self.count = int(state["count"])
        self.largest = bool(state.get("largest", True))
        self.data_shape = (
            None if state["data_shape"] is None else tuple(state["data_shape"])
        )
        self.top_data = torch.from_numpy(state["top_data"])
        self.top_index = torch.from_numpy(state["top_index"])
        self.next = int(state["next"])
        self.linear_index = (
            torch.from_numpy(state["linear_index"])
            if len(state["linear_index"].shape) > 0
            else int(state["linear_index"])
        )


class History(Stat):
    """
    Accumulates the concatenation of all the added data.
    """

    def __init__(self, data=None, state=None):
        if state is not None:
            return super().__init__(state)
        self._data = data
        self._added = []

    def _cat_added(self):
        if len(self._added):
            self._data = torch.cat(
                ([self._data] if self._data is not None else []) + self._added
            )
            self._added = []

    def add(self, d):
        self._added.append(d)
        if len(self._added) > 100:
            self._cat_added()

    def history(self):
        self._cat_added()
        return self._data

    def load_state_dict(self, state):
        data = state["data"]
        self._data = None if data is None else torch.from_numpy(data)
        self._added = []

    def state_dict(self):
        self._cat_added()
        return dict(
            constructor=self.__module__ + "." + self.__class__.__name__ + "()",
            data=None if self._data is None else self._data.cpu().numpy(),
        )

    def to_(self, device):
        """Switches internal storage to specified device."""
        self._cat_added()
        if self._data is not None:
            self._data = self._data.to(device)


class CombinedStat(Stat):
    """
    A Stat that bundles together multiple Stat objects.
    Convenient for loading and saving a state_dict made up of a
    hierarchy of stats, and for use with the tally() function.
    Example:

        cs = CombinedStat(m=Mean(), q=Quantile())
        for [b] in tally(cs, MyDataSet(), cache=fn, batch_size=100):
            cs.add(b)
        print(cs.m.mean())
        print(cs.q.median())
    """

    def __init__(self, state=None, **kwargs):
        self._objs = kwargs
        if state is not None:
            return super().__init__(state)

    def __getattr__(self, k):
        if k in self._objs:
            return self._objs[k]
        raise AttributeError()

    def add(self, d, *args, **kwargs):
        for obj in self._objs.values():
            obj.add(d, *args, **kwargs)

    def load_state_dict(self, state):
        for prefix, obj in self._objs.items():
            obj.load_state_dict(pull_key_prefix(prefix, state))

    def state_dict(self):
        result = {}
        for prefix, obj in self._objs.items():
            result.update(push_key_prefix(prefix, obj.state_dict()))
        return result

    def to_(self, device):
        """Switches internal storage to specified device."""
        for v in self._objs.values():
            v.to_(device)


def push_key_prefix(prefix, d):
    """
    Returns a dict with the same values as d, but where each key
    adds the prefix, followed by a dot.
    """
    return {prefix + "." + k: v for k, v in d.items()}


def pull_key_prefix(prefix, d):
    """
    Returns a filtered dict of all the items of d that start with
    the given key prefix, plus a dot, with that prefix removed.
    """
    pd = prefix + "."
    lpd = len(pd)
    return {k[lpd:]: v for k, v in d.items() if k.startswith(pd)}


# We wish to be able to save None (null) values in numpy npz files,
# yet do so without setting the unsecure 'allow_pickle' flag.  To do
# that, we will encode null as a special kind of IEEE 754 NaN value.
# Inspired by https://github.com/zuiderkwast/nanbox/blob/master/nanbox.h
# we follow the same Nanboxing scheme used in JavaScriptCore
# (search for JSCJSValue.h#L435), which encodes null values in NaN
# as the NaN value with hex pattern 0xfff8000000000002.

null_numpy_value = numpy.array(
    struct.unpack(">d", struct.pack(">Q", 0xFFF8000000000002))[0], dtype=numpy.float64
)


def is_null_numpy_value(v):
    """
    True if v is a 64-bit float numpy scalar NaN matching null_numpy_value.
    """
    return (
        isinstance(v, numpy.ndarray)
        and numpy.ndim(v) == 0
        and v.dtype == numpy.float64
        and numpy.isnan(v)
        and 0xFFF8000000000002 == struct.unpack(">Q", struct.pack(">d", v))[0]
    )


def box_numpy_null(d):
    """
    Replaces None with null_numpy_value, leaving non-None values unchanged.
    Recursively descends into a dictionary replacing None values.
    """
    try:
        return {k: box_numpy_null(v) for k, v in d.items()}
    except Exception:
        return null_numpy_value if d is None else d


def unbox_numpy_null(d):
    """
    Reverses box_numpy_null, replacing null_numpy_value with None.
    Recursively descends into a dictionary replacing None values.
    """
    try:
        return {k: unbox_numpy_null(v) for k, v in d.items()}
    except Exception:
        return None if is_null_numpy_value(d) else d


def resolve_state_dict(s):
    """
    Resolves a state, which can be a filename or a dict-like object.
    """
    if isinstance(s, str):
        return unbox_numpy_null(numpy.load(s))
    return s


global_load_cache_enabled = True


def load_cached_state(cachefile, args, quiet=False, throw=False):
    """
    Resolves a state, which can be a filename or a dict-like object.
    """
    if not global_load_cache_enabled or cachefile is None:
        return None
    try:
        if isinstance(cachefile, dict):
            dat = cachefile
            cachefile = "state"  # for printed messages
        else:
            dat = unbox_numpy_null(numpy.load(cachefile))
        for a, v in args.items():
            if a not in dat or dat[a] != v:
                if not quiet:
                    print("%s %s changed from %s to %s" % (cachefile, a, dat[a], v))
                return None
    except (FileNotFoundError, ValueError) as e:
        if throw:
            raise e
        return None
    else:
        if not quiet:
            print("Loading cached %s" % cachefile)
        return dat


def save_cached_state(cachefile, obj, args):
    """
    Saves the state_dict of the given object in a dict or npz file.
    """
    if cachefile is None:
        return
    dat = obj.state_dict()
    for a, v in args.items():
        if a in dat:
            assert dat[a] == v
        dat[a] = v
    if isinstance(cachefile, dict):
        cachefile.clear()
        cachefile.update(dat)
    else:
        os.makedirs(os.path.dirname(cachefile), exist_ok=True)
        numpy.savez(cachefile, **box_numpy_null(dat))


class FixedSubsetSampler(Sampler):
    """Represents a fixed sequence of data set indices.
    Subsets can be created by specifying a subset of output indexes.
    """

    def __init__(self, samples):
        self.samples = samples

    def __iter__(self):
        return iter(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, key):
        return self.samples[key]

    def subset(self, new_subset):
        return FixedSubsetSampler(self.dereference(new_subset))

    def dereference(self, indices):
        """
        Translate output sample indices (small numbers indexing the sample)
        to input sample indices (larger number indexing the original full set)
        """
        return [self.samples[i] for i in indices]


class FixedRandomSubsetSampler(FixedSubsetSampler):
    """Samples a fixed number of samples from the dataset, deterministically.
    Arguments:
        data_source,
        sample_size,
        seed (optional)
    """

    def __init__(self, data_source, start=None, end=None, seed=1):
        rng = random.Random(seed)
        shuffled = list(range(len(data_source)))
        rng.shuffle(shuffled)
        self.data_source = data_source
        super(FixedRandomSubsetSampler, self).__init__(shuffled[start:end])

    def class_subset(self, class_filter):
        """
        Returns only the subset matching the given rule.
        """
        if isinstance(class_filter, int):

            def rule(d):
                return d[1] == class_filter

        else:
            rule = class_filter
        return self.subset(
            [i for i, j in enumerate(self.samples) if rule(self.data_source[j])]
        )


def make_loader(
    dataset, sample_size=None, batch_size=1, sampler=None, random_sample=None, **kwargs
):
    """Utility for creating a dataloader on fixed sample subset."""
    import typing

    if isinstance(dataset, typing.Callable):
        # To support deferred dataset loading, support passing a factory
        # that creates the dataset when called.
        dataset = dataset()
    if isinstance(dataset, torch.Tensor):
        # The dataset can be a simple tensor.
        dataset = torch.utils.data.TensorDataset(dataset)
    if sample_size is not None:
        assert sampler is None, "sampler cannot be specified with sample_size"
        if sample_size > len(dataset):
            print(
                "Warning: sample size %d > dataset size %d"
                % (sample_size, len(dataset))
            )
            sample_size = len(dataset)
        if random_sample is None:
            sampler = FixedSubsetSampler(list(range(sample_size)))
        else:
            sampler = FixedRandomSubsetSampler(
                dataset, seed=random_sample, end=sample_size
            )
    return torch.utils.data.DataLoader(
        dataset, sampler=sampler, batch_size=batch_size, **kwargs
    )


# Unit Tests
def _unit_test():
    import warnings

    warnings.filterwarnings("error")
    import argparse
    import random
    import shutil
    import tempfile
    import time

    parser = argparse.ArgumentParser(description="Test things out")
    parser.add_argument("--mode", default="cpu", help="cpu or cuda")
    parser.add_argument("--test_size", type=int, default=1000000)
    args = parser.parse_args()
    testdir = tempfile.mkdtemp()
    batch_size = random.randint(500, 1500)

    # Test NaNboxing.
    assert numpy.isnan(null_numpy_value)
    assert is_null_numpy_value(null_numpy_value)
    assert not is_null_numpy_value(numpy.nan)

    # Test Covariance
    goal = torch.tensor(numpy.random.RandomState(1).standard_normal(10 * 10)).view(
        10, 10
    )
    data = (
        torch.tensor(numpy.random.RandomState(2).standard_normal(args.test_size * 10))
        .view(args.test_size, 10)
        .mm(goal)
    )
    data += torch.randn(1, 10) * 999
    dcov = data.t().cov()
    dcorr = data.t().corrcoef()
    rcov = Covariance()
    rcov.add(data)  # All one batch
    assert (rcov.covariance() - dcov).abs().max() < 1e-16
    cs = CombinedStat(cov=Covariance(), xcov=CrossCovariance())
    ds = torch.utils.data.TensorDataset(data)
    for [a] in tally(cs, ds, batch_size=9876):
        cs.cov.add(a)
        cs.xcov.add(a[:, :3], a[:, 3:])
    assert (data.mean(0) - cs.cov.mean()).abs().max() < 1e-12
    assert (dcov - cs.cov.covariance()).abs().max() < 2e-12
    assert (dcov[:3, 3:] - cs.xcov.covariance()).abs().max() < 1e-12
    assert (dcov.diagonal() - torch.cat(cs.xcov.variance())).abs().max() < 1e-12
    assert (dcorr - cs.cov.correlation()).abs().max() < 2e-12

    # Test CrossCovariance and CrossIoU
    fn = f"{testdir}/cross_cache.npz"
    ds = torch.utils.data.TensorDataset(
        (
            torch.arange(args.test_size)[:, None] % torch.arange(1, 6)[None, :] == 0
        ).double(),
        (
            torch.arange(args.test_size)[:, None] % torch.arange(5, 8)[None, :] == 0
        ).double(),
    )
    c = CombinedStat(c=CrossCovariance(), iou=CrossIoU())
    riou = IoU()
    count = 0
    for [a, b] in tally(c, ds, cache=fn, batch_size=100):
        count += 1
        c.add(a, b)
        riou.add(torch.cat([a, b], dim=1))
    assert count == -(-args.test_size // 100)
    cor = c.c.correlation()
    iou = c.iou.iou()
    assert cor.shape == iou.shape == (5, 3)
    assert iou[4, 0] == 1.0
    assert abs(iou[0, 2] + (-args.test_size // 7 / float(args.test_size))) < 1e-6
    assert abs(cor[4, 0] - 1.0) < 1e-2
    assert abs(cor[0, 2] - 0.0) < 1e-6
    assert all((riou.iou()[:5, -3:] == iou).view(-1))
    assert all(riou.iou().diagonal(0) == 1)
    c = CombinedStat(c=CrossCovariance(), iou=CrossIoU())
    count = 0
    for [a, b] in tally(c, ds, cache=fn, batch_size=10):
        count += 1
        c.add(a, b)
    assert count == 0
    assert all((c.c.correlation() == cor).view(-1))
    assert all((c.iou.iou() == iou).view(-1))

    # Test Concatantaion, Mean, Bincount and tally.
    fn = f"{testdir}/series_cache.npz"
    count = 0
    ds = torch.utils.data.TensorDataset(torch.arange(args.test_size))
    c = CombinedStat(s=History(), m=Mean(), b=Bincount())
    for [b] in tally(c, ds, cache=fn, batch_size=batch_size):
        count += 1
        c.add(b)
    assert count == -(-args.test_size // batch_size)
    assert len(c.s.history()) == args.test_size
    assert c.s.history()[-1] == args.test_size - 1
    assert all(c.s.history() == ds.tensors[0])
    assert all(c.b.bincount() == torch.ones(args.test_size))
    assert c.m.mean() == float(args.test_size - 1) / 2.0
    c2 = CombinedStat(s=History(), m=Mean(), b=Bincount())
    batches = tally(c2, ds, cache=fn)
    assert len(c2.s.history()) == args.test_size
    assert all(c2.s.history() == c.s.history())
    assert all(c2.b.bincount() == torch.ones(args.test_size))
    assert c2.m.mean() == c.m.mean()
    count = 0
    for b in batches:
        count += 1
    assert count == 0  # Shouldn't do anything when it's cached

    # An adverarial case: we keep finding more numbers in the middle
    # as the stream goes on.
    amount = args.test_size
    quantiles = 1000
    data = numpy.arange(float(amount))
    data[1::2] = data[-1::-2] + (len(data) - 1)
    data /= 2
    depth = 50
    alldata = data[:, None] + (numpy.arange(depth) * amount)[None, :]
    actual_sum = torch.FloatTensor(numpy.sum(alldata * alldata, axis=0))
    amt = amount // depth
    for r in range(depth):
        numpy.random.shuffle(alldata[r * amt : r * amt + amt, r])
    if args.mode == "cuda":
        alldata = torch.cuda.FloatTensor(alldata)
        device = torch.device("cuda")
    else:
        alldata = torch.FloatTensor(alldata)
        device = None
    starttime = time.time()
    cs = CombinedStat(
        qc=Quantile(),
        m=Mean(),
        v=Variance(),
        c=Covariance(),
        s=SecondMoment(),
        t=TopK(),
        i=IoU(),
    )
    # Feed data in little batches
    i = 0
    while i < len(alldata):
        batch_size = numpy.random.randint(1000)
        cs.add(alldata[i : i + batch_size])
        i += batch_size
    # Test state dict
    saved = cs.state_dict()
    # numpy.savez(f'{testdir}/saved.npz', **box_numpy_null(saved))
    # saved = unbox_numpy_null(numpy.load(f'{testdir}/saved.npz'))
    cs.save(f"{testdir}/saved.npz")
    loaded = unbox_numpy_null(numpy.load(f"{testdir}/saved.npz"))
    assert set(loaded.keys()) == set(saved.keys())

    # Restore using state=saved in constructor.
    cs2 = CombinedStat(
        qc=Quantile(),
        m=Mean(),
        v=Variance(),
        c=Covariance(),
        s=SecondMoment(),
        t=TopK(),
        i=IoU(),
        state=saved,
    )
    # saved = unbox_numpy_null(numpy.load(f'{testdir}/saved.npz'))
    assert not cs2.qc.device.type == "cuda"
    cs2.to_(device)
    # alldata = alldata.cpu()
    cs2.add(alldata)
    actual_sum *= 2
    # print(abs(alldata.mean(0) - cs2.m.mean()) / alldata.mean())
    assert all(abs(alldata.mean(0) - cs2.m.mean()) / alldata.mean() < 1e-5)
    assert all(abs(alldata.mean(0) - cs2.v.mean()) / alldata.mean() < 1e-5)
    assert all(abs(alldata.mean(0) - cs2.c.mean()) / alldata.mean() < 1e-5)
    # print(abs(alldata.var(0) - cs2.v.variance()) / alldata.var(0))
    assert all(abs(alldata.var(0) - cs2.v.variance()) / alldata.var(0) < 1e-3)
    assert all(abs(alldata.var(0) - cs2.c.variance()) / alldata.var(0) < 1e-2)
    # print(abs(alldata.std(0) - cs2.v.stdev()) / alldata.std(0))
    assert all(abs(alldata.std(0) - cs2.v.stdev()) / alldata.std(0) < 1e-4)
    # print(abs(alldata.std(0) - cs2.c.stdev()) / alldata.std(0))
    assert all(abs(alldata.std(0) - cs2.c.stdev()) / alldata.std(0) < 2e-3)
    moment = (alldata.t() @ alldata) / len(alldata)
    # print(abs(moment - cs2.s.moment()) / moment.abs())
    assert all((abs(moment - cs2.s.moment()) / moment.abs()).view(-1) < 1e-2)
    assert all(alldata.max(dim=0)[0] == cs2.t.topk()[0][:, 0])
    assert cs2.i.iou()[0, 0] == 1
    assert all((cs2.i.iou()[1:, 1:] == 1).view(-1))
    assert all(cs2.i.iou()[1:, 0] < 1)
    assert all(cs2.i.iou()[1:, 0] == cs2.i.iou()[0, 1:])

    # Restore using cs.load() method.
    cs = CombinedStat(
        qc=Quantile(),
        m=Mean(),
        v=Variance(),
        c=Covariance(),
        s=SecondMoment(),
        t=TopK(),
        i=IoU(),
    )
    cs.load(f"{testdir}/saved.npz")
    assert not cs.qc.device.type == "cuda"
    cs.to_(device)
    cs.add(alldata)
    # actual_sum *= 2
    # print(abs(alldata.mean(0) - cs.m.mean()) / alldata.mean())
    assert all(abs(alldata.mean(0) - cs.m.mean()) / alldata.mean() < 1e-5)
    assert all(abs(alldata.mean(0) - cs.v.mean()) / alldata.mean() < 1e-5)
    assert all(abs(alldata.mean(0) - cs.c.mean()) / alldata.mean() < 1e-5)
    # print(abs(alldata.var(0) - cs.v.variance()) / alldata.var(0))
    assert all(abs(alldata.var(0) - cs.v.variance()) / alldata.var(0) < 1e-3)
    assert all(abs(alldata.var(0) - cs.c.variance()) / alldata.var(0) < 1e-2)
    # print(abs(alldata.std(0) - cs.v.stdev()) / alldata.std(0))
    assert all(abs(alldata.std(0) - cs.v.stdev()) / alldata.std(0) < 1e-4)
    # print(abs(alldata.std(0) - cs.c.stdev()) / alldata.std(0))
    assert all(abs(alldata.std(0) - cs.c.stdev()) / alldata.std(0) < 2e-3)
    moment = (alldata.t() @ alldata) / len(alldata)
    # print(abs(moment - cs.s.moment()) / moment.abs())
    assert all((abs(moment - cs.s.moment()) / moment.abs()).view(-1) < 1e-2)
    assert all(alldata.max(dim=0)[0] == cs.t.topk()[0][:, 0])
    assert cs.i.iou()[0, 0] == 1
    assert all((cs.i.iou()[1:, 1:] == 1).view(-1))
    assert all(cs.i.iou()[1:, 0] < 1)
    assert all(cs.i.iou()[1:, 0] == cs.i.iou()[0, 1:])

    # Randomized quantile test
    qc = cs.qc
    ro = qc.readout(1001).cpu()
    endtime = time.time()
    gt = (
        torch.linspace(0, amount, quantiles + 1)[None, :]
        + (torch.arange(qc.depth, dtype=torch.float) * amount)[:, None]
    )
    maxreldev = torch.max(torch.abs(ro - gt) / amount) * quantiles
    print("Randomized quantile test results:")
    print("Maximum relative deviation among %d perentiles: %f" % (quantiles, maxreldev))
    minerr = torch.max(
        torch.abs(
            qc.minmax().cpu()[:, 0] - torch.arange(qc.depth, dtype=torch.float) * amount
        )
    )
    maxerr = torch.max(
        torch.abs(
            (qc.minmax().cpu()[:, -1] + 1)
            - (torch.arange(qc.depth, dtype=torch.float) + 1) * amount
        )
    )
    print("Minmax error %f, %f" % (minerr, maxerr))
    interr = torch.max(
        torch.abs(qc.integrate(lambda x: x * x).cpu() - actual_sum) / actual_sum
    )
    print("Integral error: %f" % interr)
    medianerr = torch.max(
        torch.abs(qc.median() - alldata.median(0)[0]) / alldata.median(0)[0]
    ).cpu()
    print("Median error: %f" % medianerr)
    meanerr = torch.max(torch.abs(qc.mean() - alldata.mean(0)) / alldata.mean(0)).cpu()
    print("Mean error: %f" % meanerr)
    varerr = torch.max(torch.abs(qc.variance() - alldata.var(0)) / alldata.var(0)).cpu()
    print("Variance error: %f" % varerr)
    counterr = (
        (qc.integrate(lambda x: torch.ones(x.shape[-1]).cpu()) - qc.size())
        / (0.0 + qc.size())
    ).item()
    print("Count error: %f" % counterr)
    print("Time %f" % (endtime - starttime))
    # Algorithm is randomized, so some of these will fail with low probability.
    assert maxreldev < 1.0
    assert minerr == 0.0
    assert maxerr == 0.0
    assert interr < 0.01
    assert abs(counterr) < 0.001
    shutil.rmtree(testdir, ignore_errors=True)
    print("OK")


if __name__ == "__main__":
    _unit_test()
