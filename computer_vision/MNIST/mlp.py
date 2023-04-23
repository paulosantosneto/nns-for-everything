import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from typing import Tuple
from tqdm import tqdm 
import os

class NN4MNIST(object):

    def __init__(self, device: str='cpu', epochs: int=5):
        self.device = device
        self.train_dl, self.test_dl = self.load_data()
        self.model, self.optimizer, self.criterion = self.get_model()        
        self.epochs = epochs
                    
    def save_model(self, path: any=None):
        
        torch.save(self.model.state_dict(), 'model_weights.pth')
        

    def load_weights(self):

        self.model = self.model.load_state_dict(torch.load('model_weights.pth')) 

    def inference(self):

        pass
    
    def train(self):
        
        global_loss = []

        for epoch in range(self.epochs):
            local_loss = []
            
            for idx, batch in enumerate(iter(tqdm(self.train_dl))):
                X, y = batch
                local_loss.append(self.train_batch(X.to(self.device), y.to(self.device)))
            mean_epoch = sum(local_loss) / len(local_loss)
            global_loss.append(mean_epoch)

        print(global_loss)

    def train_batch(self, X, y) -> float:

        self.model.train()
        preds = self.model(X)
        self.loss = self.criterion(preds, y)
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

        return self.loss.item()

    def get_model(self) -> Tuple[nn.Sequential, torch.optim.Adam, nn.CrossEntropyLoss]:
        '''
        Here the structure of the neural network is defined, such as the number of layers,
        activation functions and so on. Soon after, the type of optimizer and the cost function
        are defined. Remember that the optimizer has a hyperparameter (learning rate), which can
        be modified depending on the distribution and complexity of the dataset.
        '''

        model = nn.Sequential(nn.Flatten(),
                              nn.Linear(in_features=28*28, out_features=50, bias=True),
                              nn.ReLU(),
                              nn.Linear(in_features=50, out_features=50, bias=True),
                              nn.ReLU(),
                              nn.Linear(in_features=50, out_features=10),
        ).to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss() # loss function
        
        return model, optimizer, criterion

    def load_data(self) -> Tuple[DataLoader, DataLoader]:

        train_ds = datasets.FashionMNIST(root='data',
                                         train=True,
                                         download=True,
                                         transform=ToTensor(),
        )

        test_ds = datasets.FashionMNIST(root='data',
                                        train=False,
                                        download=True,
                                        transform=ToTensor(),
        )

        train_dl = DataLoader(train_ds, batch_size=64, shuffle=True, drop_last=True)
        test_dl = DataLoader(test_ds, batch_size=64, shuffle=True, drop_last=True)

        return train_dl, test_dl
        
nn = NN4MNIST()
nn.train()
nn.save_model()
nn.load_weights()
