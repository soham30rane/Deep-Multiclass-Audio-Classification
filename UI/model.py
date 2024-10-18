import torch
import torch.nn as nn
import torch.nn.functional as F

class NNModel(nn.Module):
    def __init__(self,input_size=2048,num_classes=50,dropout_prob = 0.):
        super().__init__()
        self.layer1 = nn.Linear(input_size,50)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(50,num_classes)
        self.dropout = nn.Dropout(dropout_prob)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self,input_data):
        x = self.layer1(input_data)
        x = self.relu(x)
        x = self.dropout(x)
        logits = self.fc(x)
        predictions = self.softmax(logits)
        return predictions
    
class NNModel2(nn.Module):
    def __init__(self,input_size,num_classes,dropout_prob = 0.):
        super().__init__()
        self.fc = nn.Linear(input_size,num_classes)
        self.dropout = nn.Dropout(dropout_prob)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self,input_data):
        x = self.dropout(input_data)
        logits = self.fc(x)
        predictions = self.softmax(logits)
        return predictions

def predict(model, input):
    model.eval()
    with torch.no_grad():
        A = model(input)
        predictions = A.argmax(dim=1)  # Get the index of the max value (class)
    return predictions
