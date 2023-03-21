import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

class Net(nn.Module):
    def __init__(self, lstm_input_size, agent_state_size, lstm_hidden_state_size, fc1_output_size, fc2_output_size):
        super(Net, self).__init__()
        
        self.lstm = nn.LSTM(input_size=lstm_input_size, hidden_size=lstm_hidden_state_size)
        self.fc1 = nn.Linear(lstm_hidden_state_size+agent_state_size, fc1_output_size)  # +1 because we are adding one more input to the second layer
        self.fc2 = nn.Linear(fc1_output_size, fc2_output_size)  
    
    def forward(self, sa, so):
        # so is input to the LSTM layer, y is the input to the second fully connected layer
        lstm_out, (h_n, c_n) = self.lstm(so)
        last_h = lstm_out[-1] # get the last output from the LSTM layer
        fc1_input = torch.cat([last_h, sa]) # concatenate the output from fc1 and the extra input
        fc2_input = self.fc1(fc1_input)
        output = self.fc2(fc2_input)
        return output

# ref: https://discuss.pytorch.org/t/extract-features-from-layer-of-submodule-of-a-model/20181/12
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output[0].detach().numpy()
        # activation[name] = output.detach()
    return hook

##
## initilization
##

##
## define the net
##

m = 300 # total number of samples
lstm_input_size = 10
num_of_recurrent_input = 3 # this will the total number of obstacles in the env (LSTM should be able to handle it when it changes) 
agent_state_size = 1 # each sample has n_agent inputs
lstm_hidden_state_size = 5 # hidden state size (= the size of h vector (output of LSTM))
fc1_output_size = 10
fc2_output_size = 5

model = Net(lstm_input_size, agent_state_size, lstm_hidden_state_size, fc1_output_size, fc2_output_size)

##
## data set
##

state_agent = torch.randn(m, agent_state_size) 
state_obst = torch.randn(m, num_of_recurrent_input, lstm_input_size)
target_tensor = torch.randn(m, fc2_output_size)

dataset = TensorDataset(state_agent, state_obst, target_tensor)
# dataloader = DataLoader(dataset, batch_size=100, shuffle=True)

##
## define a loss function and optimier
##

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

##
## training
##

num_epochs = 10
model.lstm.register_forward_hook(get_activation('lstm'))
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(dataset):
        sa, so, y = data  # Assuming your training data is in the format (input sequence, additional input, target output)
        optimizer.zero_grad()
        output = model(sa, so)
        print(activation['lstm'])
        
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        # Print statistics
        running_loss += loss.item()
        if i % 100 == 99:    # Print every 100 batches
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0
