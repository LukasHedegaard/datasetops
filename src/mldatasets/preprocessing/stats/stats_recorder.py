
import numpy as np


class StatsRecorder:
    """Object which implements statatistical measures using online algorithms.

    Raises
    ------
    ValueError
        Thrown when update is called with data which does not match the previously seen data.
    """

    def __init__(self, data=None):
        """
        data: ndarray, shape (nobservations, ndimensions)
        """
        if data is not None:
            data = np.atleast_2d(data)
            self.mean = data.mean(axis=0)
            self.std = data.std(axis=0)
            self.nobservations = data.shape[0]
            self.ndimensions = data.shape[1]
        else:
            self.nobservations = 0

    def update(self, data):
        """
        data: ndarray, shape (nobservations, ndimensions)
        """
        if self.nobservations == 0:
            self.__init__(data)
        else:
            data = np.atleast_2d(data)
            if data.shape[1] != self.ndimensions:
                raise ValueError("Data dims don't match prev observations.")

            newmean = data.mean(axis=0)
            newstd = data.std(axis=0)

            m = self.nobservations * 1.0
            n = data.shape[0]

            tmp = self.mean

            self.mean = m/(m+n)*tmp + n/(m+n)*newmean
            self.std = m/(m+n)*self.std**2 + n/(m+n)*newstd**2 +\
                m*n/(m+n)**2 * (tmp - newmean)**2
            self.std = np.sqrt(self.std)

            self.nobservations += n
