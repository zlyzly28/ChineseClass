import torch.nn.functional as F
import torch.nn as nn
import torch
class text_rnn(nn.Module):
    def __init__(self, config):
        super(text_rnn, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained)
        self.lstm = nn.LSTM(config.embed, config.hidden_size, config.num_layers,
                           bidirectional=True, batch_first = True, dropout = config.dropout)
        self.fc = nn.Linear(config.hidden_size * 2, config.num_classes)
        
    def forward(self, x):
        x,_ = x
        out = self.embedding(x)
        out, _ = self.lstm(out)
        out = self.fc(out[:, -1, :])
        return out
    
class text_cnn(nn.Module):
    def __init__(self, config):
        super(text_cnn, self).__init__()

        self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)

        self.convs = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, config.embed)) for k in config.filter_sizes])
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.num_filters * len(config.filter_sizes), config.num_classes)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        out = self.embedding(x[0])
        out = out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out