from torch import nn

class MyModel(nn.Module):

    def __init__(self, vocabulary_size, number_of_categories):
        super().__init__()
        # One trainable 64-number vector for each vocabulary entry.
        self.embedding = nn.Embedding(vocabulary_size, 64, padding_idx=0)
        self.lstm = nn.LSTM(64, 64, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(64, number_of_categories)

    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.lstm(embedded)
        # Keep real-word positions when averaging the LSTM outputs.
        mask = (x != 0).unsqueeze(-1)
        pooled = (output * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        # Dropout operates during training and is disabled by model.eval().
        pooled = self.dropout(pooled)
        return self.fc1(pooled)

