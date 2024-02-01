import numpy as np
import torch
import torchvision
from model import CNN
from torch.utils.data import DataLoader
from utils import *
from tqdm import tqdm
# argparse
import argparse

def train(epochs, batch_size):
    

    train_dataset, val_dataset = loaddata()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    input_channels = 3
    num_classes = 13
    print(f"Input channels: {input_channels}, Num classes: {num_classes}")


    #model
    device= torch.device('cpu')
    model = CNN(input_channels, num_classes).to(device)

    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    #train
    for epoch in tqdm(range(epochs), desc='Training'):
        model.train()
        tot_loss =0
        for i, (x,y) in tqdm(enumerate(train_loader), desc='iteration'):
            # print("input image", x, "ground truth class label", y)
            x, y = x.to(device), y.to(device)
            #zero out gradients
            optimizer.zero_grad()

            #Forward pass
            pred = model(x)

            #Compute loss
            loss = criterion(pred, y)

            #Backprop
            loss.backward()

            #Update weights
            optimizer.step()

            tot_loss += loss.item()

            # print training loss every 5 epochs

        avg_loss= tot_loss /len(train_loader)
        print(f'Epoch {epoch}, Average Loss: {avg_loss}')

        #Validation

        model.eval()
        with torch.no_grad(): # no need to update weights
            correct = 0
            total = 0
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred=model(x)
                _, pred_indices =torch.max(pred,1)
                total += y.size(0)
                correct += (pred_indices == y).sum().item()
            print(f'Epoch: {epoch}, Acuuracy: {100*correct/total}%')



if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epochs', type=int, default=15,required = True)
    argparser.add_argument('--batch', type=int, default=8, required =True)

    args = argparser.parse_args()

    epochs = args.epochs
    batch_size = args.batch

    train(args.epochs, args.batch)
