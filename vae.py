
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.optim import Adam
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Model Hyperparameters
batch_size = 100
x_dim  = 784
hidden_dim = 512
latent_dim = 200
lr = 0.001

epochs = 100
mnist_transform = transforms.Compose([
        transforms.ToTensor(),
])


train_dataset = MNIST('.', transform=mnist_transform, train=True, download=True)
test_dataset  = MNIST('.', transform=mnist_transform, train=False, download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader  = DataLoader(dataset=test_dataset,  batch_size=batch_size, shuffle=False, num_workers=0)

class Encoder(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()

        self.FC_input = nn.Linear(input_dim, hidden_dim)
        self.FC_input2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_mean  = nn.Linear(hidden_dim, latent_dim)
        self.FC_var   = nn.Linear (hidden_dim, latent_dim)
        
        self.ReLU = nn.ReLU(0.2)
        
        self.training = True
        
    def forward(self, x):
        h_1      = self.ReLU(self.FC_input(x))
        h_1      = self.ReLU(self.FC_input2(h_1))
        mean     = self.FC_mean(h_1)
        log_var  = self.FC_var(h_1)                     
        
        return mean, log_var
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.FC_hidden = nn.Linear(latent_dim, hidden_dim)
        self.FC_hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_output = nn.Linear(hidden_dim, output_dim)
        
        self.ReLU = nn.ReLU(0.2)
        
    def forward(self, x):
        h     = self.ReLU(self.FC_hidden(x))
        h     = self.ReLU(self.FC_hidden2(h))
        
        x_hat = torch.sigmoid(self.FC_output(h))
        return x_hat
        
class Model(nn.Module):
    def __init__(self, Encoder, Decoder):
        super(Model, self).__init__()
        self.Encoder = Encoder
        self.Decoder = Decoder
        
    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var)           
        z = mean + var*epsilon                      
        return z
        
                
    def forward(self, x):
        mean, log_var = self.Encoder(x)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var)) 
        x_hat = self.Decoder(z)
        
        return x_hat, mean, log_var
encoder = Encoder(input_dim=x_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
decoder = Decoder(latent_dim=latent_dim, hidden_dim = hidden_dim, output_dim = x_dim)

model = Model(Encoder=encoder, Decoder=decoder)


BCE_loss = nn.BCELoss()

def loss_function(x, x_hat, mean, log_var):
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
    KLD   = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())

    return reproduction_loss + KLD


optimizer = Adam(model.parameters(), lr=lr)
model.train()

for epoch in range(epochs):
    overall_loss = 0
    for batch_idx, (x, _) in enumerate(train_loader):
        x = x.view(batch_size, x_dim)
        x = x

        optimizer.zero_grad()

        x_hat, mean, log_var = model(x)
        loss = loss_function(x, x_hat, mean, log_var)
        
        overall_loss += loss.item()
        
        loss.backward()
        optimizer.step()
        
    print("\tEpoch", epoch + 1, "\tAverage Loss: ", overall_loss / (batch_idx*batch_size))
    

model.eval()

with torch.no_grad():
    for batch_idx, (x, _) in enumerate(test_loader):
        x = x.view(batch_size, x_dim)
        x = x
        
        x_hat, _, _ = model(x)
        break

def show_image(x, idx):
    x = x.view(batch_size, 28, 28)

    plt.figure()
    plt.imshow(x[idx].cpu().numpy())
    plt.show()
show_image(x, idx=0)
show_image(x_hat, idx=0)
