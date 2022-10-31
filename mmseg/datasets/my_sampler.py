import torch
import numpy as np
from torch.utils.data import DistributedSampler
import math


class My_sampler(DistributedSampler):

    def __init__(self,
                 dataset,
                 num_replicas=None,
                 rank=None,
                 shuffle=True,
                 seed=0):
        super(My_sampler, self).__init__(
            dataset=dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            seed=seed)

    def __iter__(self):
        if self.shuffle:
            cum = self.dataset.cumulative_sizes
            len_dataset = np.array([cum[0], cum[1] - cum[0], cum[-1] - cum[1]])
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = []
            indices.append(torch.randperm(len_dataset[0], generator=g).tolist())
            for i in range(1, 3):
                indices.append((torch.randperm(int(len_dataset[i]), generator=g) + cum[i - 1]).tolist())
            indice = []
            while True:
                mmin = np.argmin(len_dataset)
                mmax = np.argmax(len_dataset)
                if mmin != mmax:
                    for k in range(len_dataset[mmin] // self.num_replicas):
                        for i in range(len(len_dataset)):
                            indice += indices[i][k * self.num_replicas:(k + 1) * self.num_replicas]
                    mol = len_dataset[mmin] % self.num_replicas
                    if mol != 0:
                        indice += indices[mmin][-mol:]
                    m_min = len_dataset[mmin]
                    len_dataset = np.delete(len_dataset, mmin)
                    indices.pop(mmin)
                    len_dataset -= (m_min - mol)
                    for i in range(len(indices)):
                        indices[i] = indices[i][(m_min - mol):]

                else:
                    if len_dataset[0] != 0:
                        indice += indices[0]
                    break
        else:
            indice = list(range(len(self.dataset)))

        padding_size = self.total_size - len(indice)
        if padding_size <= len(indice):
            indice += indice[:padding_size]
        else:
            indice += (indice * math.ceil(padding_size / len(indice)))[:padding_size]
        assert len(indice) == self.total_size
        indices = indice[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)
