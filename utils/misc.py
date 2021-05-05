from typing import Callable, Iterable, Iterator, Optional, TypeVar, Tuple, List
from itertools import chain
from more_itertools import pairwise
import numpy as np
ichain = chain.from_iterable


def find_roots(data: Iterable[float],
               at: Optional[float]=None
        ) -> Iterator[int]:
    """
    Simple algorithm that checks if the product of two consecutive values
    in the series is less than zero, for this one must positive and the other
    negative, then yields the index for the value that is closer to zero

    TODO: For small peaks this can return the same index twice
    """
    if at is not None:
        data = (data - at for data in data)
    for i, (x0, x1) in enumerate(pairwise(data)):
        if x0*x1 <= 0:
            yield i if abs(x0) < abs(x1) else i+1

def resample(*arrays: Tuple[np.ndarray, np.ndarray],
             max_samples = 2**16
        ) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Resamples a dataset with a common time variable to a set of
    equally spaced time series using linear interpolation.
    """
    min_time = min(ichain(time for time, values in arrays))
    max_time = max(ichain(time for time, values in arrays))
    interval = max_time - min_time
    min_res = min(
        filter(lambda dt: abs(dt) > interval/max_samples,
               ichain(np.ediff1d(time) for time, values in arrays)))
    steps = int((max_time - min_time)//min_res) + 1
    new_time = np.linspace(min_time, max_time, steps)
    interpolated = [np.interp(new_time, time, values) for time, values in arrays]
    return new_time, interpolated
