#
#
# mnist_cnn.py
#
#
# This file trains a simple convolutional neural network to 
# classify 28x28 images of handwritten digits from the 
# famous MNIST database.
#
#
# Spencer Kraisler 2018

import torch 
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torchvision.transforms as transforms
import pandas as pd
import numpy as np

n_epochs = 5
n_classes = 10
batch_size = 100
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

    # uses matplotlib to show a picture of any image in dataset
	def showImage(self, idx):
		sample = self[idx]
		image = sample[0]
		label = sample[1]
		image = image.mul(255).long()
		image = image.numpy()
		print(label)
		imgplot=plt.imshow(image)
		plt.show()

train_set = MNISTDataset("MNIST/train.csv")
test_set = MNISTDataset("MNIST/test.csv")

train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=False)

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(1568, 10)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

model = ConvNet()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

total_step = len(train_loader)

for epoch in range(n_epochs):
    for i, (images, labels) in enumerate(train_loader):

        # Forward pass
        outputs = model.forward(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

     	if (i+1) % 10 == 0:
      		print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                .format(epoch+1, n_epochs, i+1, total_step, loss.item()))

model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for i, (images, labels) in enumerate(train_loader):
        outputs = model.forward(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        if i % 10 == 0: print(str(100 * float(correct) / total) + "%") 

print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))
