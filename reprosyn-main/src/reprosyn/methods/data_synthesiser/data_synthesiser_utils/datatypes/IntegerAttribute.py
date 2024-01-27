from numpy import linspace, histogram

from .AbstractAttribute import AbstractAttribute
from .utils.DataType import (
    DataType,
)
from ..utils import (
    normalize_given_distribution,
)


class IntegerAttribute(AbstractAttribute):
    def __init__(self, name, data, histogram_size):
        super().__init__(name, data, histogram_size)
        self.is_categorical = False
        self.is_numerical = True
        self.data_type = DataType.INTEGER
        self.data = self.data.astype(int)
        self.data_dropna = self.data_dropna.astype(int)

    def set_domain(self, domain=None):
        if domain is not None:
            self.min, self.max = domain
        else:
            self.min = self.data_dropna.min()
            self.max = self.data_dropna.max()

        self.min = int(self.min)
        self.max = int(self.max)
        self.distribution_bins = linspace(
            self.min, self.max, self.histogram_size + 1
        ).astype(int)
        self.domain_size = self.histogram_size

    def infer_distribution(self):
        frequency_counts, _ = histogram(
            self.data_dropna, bins=self.distribution_bins
        )
        self.distribution_probabilities = normalize_given_distribution(
            frequency_counts
        )

    def generate_values_as_candidate_key(self, n):
        return super().generate_values_as_candidate_key(n)

    def sample_values_from_binning_indices(self, binning_indices):
        column = super().sample_values_from_binning_indices(binning_indices)
        column = column.round()
        column = column.astype(int)
        # column[~column.isnull()] = column[~column.isnull()].astype(int)
        return column
