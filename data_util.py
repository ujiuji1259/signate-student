import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

class BertDataaset(Dataset):
    def __init__(self, data, label, transform=None):
        self.transform = transform

        self.data = data
        self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        data = self.data[item]
        label = self.label[item]

        if self.transform:
            data = self.transform(data)

        data = [1] + data
        label = int(label) - 1

        return data, label


def my_collate_fn(batch):
    data, label = list(zip(*batch))
    data = [torch.tensor(d) for d in data]
    label = torch.tensor(label)
    data = pad_sequence(data, batch_first=True)
    mask = torch.tensor([[int(i>0) for i in ii] for ii in data])

    return data, label, mask