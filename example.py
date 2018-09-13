#
#
# example.py
#
#
# This program loads an autoencoder from a autoencoder.pth. It uses the 
# model to construct a 5-dimensional vector space where every vector coordinate
# corresponds to a unique 28x28 image. 
# The vector space "conceptually regionalized" into complete regions of "one-ish" 
# digits, regions of "two-ish" digits and so forth, with inter-regions being 
# hybrid digits. 
# Simply run this file and it will create a gif called "anitest.gif".
#
# You can use this autoencoder to generate new handwritten digits,
# or rather 'computerwritten' digits.
#
#
# Spencer Kraisler 2018
#
#

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

# model structures are not saved in .pth files, only weight matrices 
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        # encoder and decoder are functional inverses
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

# loads in autoencoder weight matrices
model.load_state_dict(torch.load('pre_made_autoencoder.pth'))

# example
images = []

# traverses through the 5-dim. image space through a single line,
# collected 60 photos on its voyage and compacting them into a .gif
for i in range(30):
	vec = torch.Tensor([[13.7612, i, 15.8152,  0, 17.9234]])
	img = model.decoder(vec)
	img = img.view(1,28,28)
	img = transforms.ToPILImage()(img)
	images.append(img)

for i in range(30):
    vec = torch.Tensor([[13.7612, 30, 15.8152,  i, 17.9234]])
    img = model.decoder(vec)
    img = img.view(1,28,28)
    img = transforms.ToPILImage()(img)
    images.append(img)

images[0].save('./anitest.gif',
               save_all=True,
               append_images=images[1:],
               duration=100,
               loop=0)




