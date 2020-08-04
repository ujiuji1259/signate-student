
import torch
import torch.optim as optim
import torch.nn as nn

def train(dataloader, model, lr, epoch):
    classifier_params = ["classifier.weight", "classifier.bias"]
    params = model.state_dict()
    optimizer_params = [{"params":[], "lr":3e-5}, {"params":[], "lr":0.01}]
    for key, value in params.items():
        if key in classifier_params:
            optimizer_params[1]["params"].append(value)
        else:
            optimizer_params[0]["params"].append(value)

    optimizer = optim.Adam(optimizer_params)
    loss = nn.NLLLoss()

    for e in range(1, epoch+1):
        for x, y, mask in dataloader:
            optimizer.zero_grad()
            output = model(x, labels=y, attention_mask=mask)
            loss = output[0]
            loss.backward()
            optimizer.step()

def predict(dataloader, model):
    with torch.no_grad():
        for x, y, mask in dataloader:
            output = model(x, attention_mask=mask)[0]
            preds = torch.max(output, 1)[1].detach().numpy()



