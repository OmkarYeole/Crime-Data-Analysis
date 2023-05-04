import torch
import torch.nn as nn

class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=3, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, num_classes)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out)
        out = self.fc(out[:, -1, :])
        return out