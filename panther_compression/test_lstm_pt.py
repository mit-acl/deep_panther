import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

class Net(nn.Module):
    def __init__(self, lstm_input_size):
        super(Net, self).__init__()
        self.lstm = nn.LSTM(lstm_input_size, 10)
        self.fc1 = nn.Linear(11, 11)  # +1 because we are adding one more input to the second layer
        self.fc2 = nn.Linear(11, 5)  
    
    def forward(self, sa, so):
        # so is input to the LSTM layer, y is the input to the second fully connected layer
        lstm_out, _ = self.lstm(so)
        print(lstm_out)
        print(lstm_out[-1])
        print(sa)
        last_h = lstm_out[-1]
        fc1_input = torch.cat([last_h, sa]) # concatenate the output from fc1 and the extra input
        print(fc1_input)
        exit()
        fc2_input = self.fc1(fc1_input)
        output = self.fc2(fc2_input)
        return output

##
## initilization
##

##
## define the net
##

net = Net(lstm_input_size=10)

##
## data set
##

sa = torch.randn(300, 1)  # m = 300, input_size = 1
so = torch.randn(300, 20, 10)  # m = 300, n = 20, input_size = 10
target_tensor = torch.randn(300, 5)  # m = 3, output_size = 1 

dataset = TensorDataset(sa, so, target_tensor)
# dataloader = DataLoader(dataset, batch_size=100, shuffle=True)

##
## define a loss function and optimier
##

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

##
## training
##

num_epochs = 10
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(dataset):
        sa, so, y = data  # Assuming your training data is in the format (input sequence, additional input, target output)
        optimizer.zero_grad()
        output = net(sa, so)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        # Print statistics
        running_loss += loss.item()
        if i % 100 == 99:    # Print every 100 batches
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0
