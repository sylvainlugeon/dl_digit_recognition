#!/usr/bin/env python
# coding: utf-8

# For training 
import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
import dlc_practical_prologue as prologue
import time


##### Global parameters #####

#For reproductibility
SEED = 123 
torch.manual_seed(SEED)

if torch.cuda.is_available(): 
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')

# Training parameters
N = 1000 #Dataset size (train and test)
BATCH_SIZE = 25 #Batch size for stochastic optimization
EPOCHS = 200 # Number of epochs for one round of training

#Learing rate evolution (multiply LEARNING_RATE by GAMMA every LR_STEP epochs)
LEARNING_RATE = 1e-3 
LR_STEP = int(0.5 * EPOCHS)
GAMMA = 0.1

# Auxiliary and main losses ponderation 
AUX_LOSS = 0.5

##### Helper functions ####

def accuracy(model_output, test_target):
    """Return the accuracy of the model output."""
    nb_samples = model_output.shape[0]
    output_int = torch.zeros(nb_samples)
    
    # Convert probability to decision
    output_int = torch.argmax(model_output, 1)
    nb_errors = (output_int - test_target).type(torch.BoolTensor).sum().item()
    
    return (nb_samples - nb_errors) / nb_samples

def accuracyMnist(model_output, test_target):
    """Return the accuracy of the predicted digits of a Digit Net."""
    nb_samples = model_output.shape[0]
    model_class = model_output.argmax(dim=1)
    nb_errors = (model_class - test_target).type(torch.BoolTensor).sum().item()
    
    return (nb_samples - nb_errors) / nb_samples


def nb_param(model):
    """Return the number of trained parameters of the input model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

##### Neural Nets Definition ####

class FCNet(nn.Module):
    """Naive fully connected net."""
    def __init__(self):
        super(FCNet, self).__init__()
        self.fc1 = nn.Linear(392,200)
        self.fc2 = nn.Linear(200,20)
        self.fc3 = nn.Linear(20,2)
        
        self.drop = nn.Dropout(0.25)
        self.activ = F.relu

    def forward(self, x):
        x = self.fc1(x.view(x.size(0),-1))
        x = self.activ(x)
        x = self.drop(x)
        x = self.fc2(x)
        x1, x2 = x[:, 0:10], x[:, 10:20]
        x = self.activ(x)
        x = self.fc3(x)
        
        return x, x1, x2

class ConvNet(nn.Module):
    """Naive convolutional net."""
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(2, 12, kernel_size=3) #(1,14,14) to (12,12,12)
        self.conv2 = nn.Conv2d(12, 32, kernel_size=3) #(12,12,12) to (32,10,10)
        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2) #(32,10,10) to (32,5,5)
        self.fc1 = nn.Linear(800, 100)
        self.fc2 = nn.Linear(100, 20)
        self.fc3 = nn.Linear(20, 2)
        self.drop = nn.Dropout(0.5)
        
    def forward(self, x):   
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.max_pool1(x)
        x = self.drop(x.view(x.size(0), -1))
        x = self.fc1(x)
        x = F.relu(x)
        x = self.drop(x)
        x = self.fc2(x)
        x1, x2 = x[:, 0:10], x[:, 10:20]
        x = F.relu(x)
        x = self.fc3(x)
        
        return x, x1, x2

class DigitNet(nn.Module):
    """Inspired by LeNet5, dropout 0.5 and 2 fc layers."""
    def __init__(self):
        super(DigitNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 12, kernel_size=3) #(1,14,14) to (12,12,12)
        self.conv2 = nn.Conv2d(12, 32, kernel_size=3) #(12,12,12) to (32,10,10)
        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2) #(32,10,10) to (32,5,5)
        self.fc1 = nn.Linear(800, 100)
        self.fc2 = nn.Linear(100, 10)
        self.drop = nn.Dropout(0.5)
        
    def forward(self, x):   
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.max_pool1(x)
        x = self.drop(x.view(x.size(0), -1))
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        
        return x
    

class ConvSepNet(nn.Module):
    """Run DigitNet in parrallel on each chanel and combine at the
    end with two fully connected layers (20->10->2). No Dropout in the f.c. layers.
    """
    def __init__(self):
        super(ConvSepNet, self).__init__()
        self.mnistNet = DigitNet()
        self.fc1 = nn.Linear(20,10)
        self.fc2 = nn.Linear(10,2)


    def forward(self, x):
        x1, x2 = x[:,0:1,:,:], x[:,1:2,:,:]
        x1 = self.mnistNet(x1)
        x2 = self.mnistNet(x2)       
        x = torch.cat((x1, x2), 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        
        return x, x1, x2
    

class FinalDigitNet(nn.Module):
    """Inspired by LeNet5, dropout 0.5 and 3 fc layers"""
    def __init__(self):
        super(FinalDigitNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 12, kernel_size=3) #(1,14,14) to (12,12,12)
        self.conv2 = nn.Conv2d(12, 32, kernel_size=3) #(12,12,12) to (32,10,10)
        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2) #(32,10,10) to (32,5,5)
        self.fc1 = nn.Linear(800, 400)
        self.fc2 = nn.Linear(400, 100)
        self.fc3 = nn.Linear(100, 10)
        self.drop = nn.Dropout(0.5)
        
    def forward(self, x):   
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.max_pool1(x)
        x = self.drop(x.view(x.size(0), -1))
        x = self.fc1(x)
        x = F.relu(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.drop(x)
        x = self.fc3(x)
        
        return x
    
class FinalNet(nn.Module):
    """DigitNet with two fully connected layers (20->10->2). No Dropout"""
    def __init__(self):
        super(FinalNet, self).__init__()
        self.mnistNet = FinalDigitNet()
        self.fc1 = nn.Linear(20,10)
        self.fc2 = nn.Linear(10,2)


    def forward(self, x):
        x1, x2 = x[:,0:1,:,:], x[:,1:2,:,:]
        x1 = self.mnistNet(x1)
        x2 = self.mnistNet(x2)       
        x = torch.cat((x1, x2), 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        
        return x, x1, x2

##### Training routine ####
    
def train_routine(model, train_input, train_target, train_classes, test_input, test_target, test_classes):
    """Train a model and compute its performance on train and test data."""
    
    # Loss
    criterion = nn.CrossEntropyLoss().to(DEVICE)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), LEARNING_RATE)
    
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP, gamma = GAMMA)
    
    # Start timer
    t0 = time.time() 
    
    # Training the model
    model.train(True)
    
    for e in range(EPOCHS):
        
        print('\rTraining {}... (Epoch {}/{})'.format(model.__class__.__name__, e+1, EPOCHS), end="")
        
        # Ponderation of the main loss => (1-f): ponderation of the auxiliray loss. 
        f = AUX_LOSS

        for inputs, targets, classes in zip(train_input.split(BATCH_SIZE), \
                                            train_target.split(BATCH_SIZE), \
                                            train_classes.split(BATCH_SIZE)):
            
            output, aux1, aux2 = model(inputs)

            loss = (1-f) * criterion(output, targets) + f * (criterion(aux1, classes[:,0]) + criterion(aux2, classes[:,1]))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Updtate learning rate
        scheduler.step()
            

    # End timer
    t1 = time.time() 
    dt = t1-t0
    
    # Evaluating model performance on train and test data
    model.train(False)
    tr_output, tr_aux1, tr_aux2 = model(train_input)
    te_output, te_aux1, te_aux2 = model(test_input)
    
    tr_acc = accuracy(tr_output, train_target)
    te_acc = accuracy(te_output, test_target)
    
    tr_acc_mnist = 0.5*(accuracyMnist(tr_aux1, train_classes[:,0]) + \
                        accuracyMnist(tr_aux2, train_classes[:,1]))
    te_acc_mnist = 0.5*(accuracyMnist(te_aux1, test_classes[:,0]) +  \
                        accuracyMnist(te_aux2, test_classes[:,1]))
    
    # Showing results
    print("\nTraining time : {:.2f}s\n".format(dt) + \
          "Main performance:\n" + \
          "    -Train accuracy : {:.2f}%\n".format(100 * tr_acc) + \
          "    -Test accuracy : {:.2f}%\n".format(100 * te_acc) + \
          "Auxiliary performance:\n" + \
          "    -Train digit accuracy : {:.2f}%\n".format(100 * tr_acc_mnist) + \
          "    -Test digit accuracy : {:.2f}%\n".format(100 * te_acc_mnist) + \
          "-----------------------------------")


if __name__ == '__main__':
    
    # Display information about training procedure
    
    print('Train and test dataset size: {}\n'.format(N) + \
          'Number of epochs: {}\n'.format(EPOCHS) + \
          'Batch size for stochastic optimization: {}\n'.format(BATCH_SIZE) + \
          'Learning rate: {} (multiplied by {} after {} epochs)\n'.format(LEARNING_RATE, GAMMA, LR_STEP) + \
          'Device used for training: {}\n'.format(DEVICE) + \
          'Weight of auxiliary loss: f={}'.format(AUX_LOSS))

    
    # Load data and move it to DEVICE
    print('Loading the data...')
    train_input, train_target, train_classes, test_input, test_target, test_classes = prologue.generate_pair_sets(N)
    train_input, train_target, train_classes = train_input.to(DEVICE), train_target.to(DEVICE), train_classes.to(DEVICE)
    test_input, test_target, test_classes = test_input.to(DEVICE), test_target.to(DEVICE), test_classes.to(DEVICE)
    print('Data loaded.') 
    
    
    # Model constructions
    print('Constructing the models:')
    myFCNet = FCNet().to(DEVICE)
    myConvNet = ConvNet().to(DEVICE)
    myConvSepNet = ConvSepNet().to(DEVICE)
    myFinalNet = FinalNet().to(DEVICE)
    print('  -FCNet: {} parameters\n'.format(nb_param(myFCNet)) + \
          '  -ConvNet: {} parameters\n'.format(nb_param(myConvNet)) + \
          '  -ConvSepNet: {} parameters\n'.format(nb_param(myConvSepNet)) + \
          '  -FinalNet: {} parameters\n'.format(nb_param(myFinalNet)))

    # Training 

    train_routine(myFCNet, train_input, train_target, train_classes, test_input, test_target, test_classes)
    train_routine(myConvNet, train_input, train_target, train_classes, test_input, test_target, test_classes)
    train_routine(myConvSepNet, train_input, train_target, train_classes, test_input, test_target, test_classes)
    train_routine(myFinalNet, train_input, train_target, train_classes, test_input, test_target, test_classes)



