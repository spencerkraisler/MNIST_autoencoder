#
#
# mnist_autoencoder.py
#
#
# Using Pytorch, this program trains an autoencoder with 28x28 images of handwritten
# digits from the famous MNIST database. The autoencoder is uploaded to a .pth file
#
# Autoencoders are extremely powerful. They are able to create vector spaces that
# are "conceptually regionalized". In other words, the closer two points are in the
# image space, the more conceptually similar the two corresponding images are.
#
#
# Spencer Kraisler 2018
#
#

import torch 
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import pandas as pd
import numpy as np
import csv
from random import randint

batch_size = 100
num_epochs = 5
learning_rate = .001

class MNISTDataset(Dataset):

    # Args:
        # csv_file: the csv file containing the training set of MNIST digits
    def __init__(self, csv_file):
        self.frame = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        if len(self) == 42000:
    		label = self.frame.iloc[idx, 0]

            # reshape to a 28x28 image
    		image = np.array(self.frame.iloc[idx, 1:]).reshape(28, 28)

            # reformat to Torch tensor, normalize, and reshape to 1x28x28 for model
    		image = torch.from_numpy(image).float().div(255.0).unsqueeze(0)
    		return (image, label)
        else:
            # reshape to a 28x28 image
            image = np.array(self.frame.iloc[idx, 0:]).reshape(28, 28)

            # reformat to Torch tensor, normalize, and reshape to 1x28x28 for model
            image = torch.from_numpy(image).float().div(255.0).unsqueeze(0)
            return image

dataset = MNISTDataset("MNIST/train.csv")
dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 256),
            nn.ReLU(True),
            nn.Linear(256, 5),
            nn.ReLU(True))

        self.decoder = nn.Sequential(
            nn.Linear(5, 256),
            nn.ReLU(True),
            nn.Linear(256, 28 * 28),
            nn.Sigmoid())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

model = Autoencoder()

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
total_step = len(dataloader)

for epoch in range(num_epochs):
    for data in dataloader:
        img, _ = data
        img = img.view(img.size(0), -1)
        img = Variable(img)

        # forward
        output = model(img)
        loss = criterion(output, img)
        MSE_loss = nn.MSELoss()(output, img)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('epoch [{}/{}], loss:{:.4f}, MSE_loss:{:.4f}'
        .format(epoch + 1, num_epochs, loss.data[0], MSE_loss.data[0]))

torch.save(model.state_dict(), './autoencoder.pth')






