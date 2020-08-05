# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

import csv

from torch.utils.data import DataLoader, random_split
from transformers import BertForSequenceClassification, BertTokenizer

from data_util import BertDataaset, my_collate_fn
from trainer import train, predict
# Press the green button in the gutter to run the script.

if __name__ == '__main__':
    train_x, train_y = [], []
    with open("data/train.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            train_x.append(row["description"])
            train_y.append(int(row["jobflag"]))

    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    train_dataset = BertDataaset(train_x, train_y, transform=lambda s: tokenizer.convert_tokens_to_ids(tokenizer.tokenize(s)))
    train_size = int(len(train_dataset) * 0.8)
    valid_size = len(train_dataset) - train_size
    train_subset, valid_subset = random_split(train_dataset, [train_size, valid_size])
    dataloader = DataLoader(valid_subset, batch_size=16, collate_fn=my_collate_fn)

    model = BertForSequenceClassification.from_pretrained("bert-base-cased", num_labels=4)
    #train(dataloader, model, 0.01, 10)
    predict(dataloader, model)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
