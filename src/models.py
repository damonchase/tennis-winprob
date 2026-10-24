import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import torch
import torch.nn as nn

class TennisSequenceDataset(Dataset):
    def __init__(self, feature_df: pd.DataFrame, labels: pd.Series, match_ids: pd.Series):
        """
        Groups point-level features and labels by match so each item is a full match sequence.
        """
        # 1. Store the unique match IDs as a list or array so we can index them by position
        # Hint: use match_ids.unique()
        self.match_ids = match_ids.unique().tolist()
        
        # 2. Store feature_df and labels inside self for easy access
        self.feature_df = feature_df
        self.labels = labels
        
        # 3. Pre-group the indices or rows by match_id to make __getitem__ fast
        # Hint: use a dictionary or groupby object to map each match_id to its row positions
        self.match_groups = dict(feature_df.groupby(match_ids).groups)

    def __len__(self) -> int:
        # Return the total number of unique matches
        return len(self.match_groups)

    def __getitem__(self, idx):
        # 1. Get the match_id corresponding to 'idx'
        m_id = self.match_ids[idx]
        
        # 2. Extract all feature rows for this specific match
        row_indices = self.match_groups[m_id]
        match_features = self.feature_df.loc[row_indices].values
        
        # 3. Extract the target label for this match (it's the same for every point in the match)
        match_label = self.labels.loc[row_indices].iloc[0]

        # 4. Convert features and label to PyTorch tensors (torch.FloatTensor)
        # Hint: torch.tensor(..., dtype=torch.float32)
        return torch.tensor(match_features, dtype=torch.float32), torch.tensor(match_label, dtype=torch.float32)

def tennis_collate_fn(batch):
    """
    Takes a batch of (features, label) tuples of varying lengths and pads them 
    so they can be stacked into a single tensor batch.
    
    batch: List of tuples [(match_features_tensor, match_label_tensor), ...]
    """
    # 1. Unpack the batch into separate lists of features and labels
    features_tuple, labels_list = zip(*batch)
    
    # 2. Pad the sequence features so every match in the batch has the same time-dimension length
    # Hint: Use pad_sequence(features_list, batch_first=True, padding_value=0.0)
    padded_features = pad_sequence(list(features_tuple), batch_first=True, padding_value=0.0)
    
    # 3. Create a binary mask of shape (batch_size, max_seq_len) where 1 means a real point 
    # and 0 means padding. 
    # Hint: We can build this by checking where features are non-zero or tracking lengths.
    mask = (padded_features[:, :, 0] != 0.0).float()

    # 4. Stack the labels into a single tensor
    # Hint: torch.stack(labels_list)
    stacked_labels = torch.stack(labels_list)
    
    return padded_features, stacked_labels, mask


class TennisGRUModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 1, dropout: float = 0.2):
        super().__init__()
        
        # 1. Define the GRU layer
        # Hint: Use nn.GRU with input_dim, hidden_dim, num_layers, batch_first=True, and dropout
        gru_dropout = dropout if num_layers > 1 else 0.0
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=gru_dropout)

        self.dropout = nn.Dropout(dropout)
        
        # 3. Define a final fully connected (Linear) layer to map hidden states to a single prediction logit
        # Hint: Use nn.Linear(hidden_dim, 1)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x, mask=None):
        """
        Forward pass for the GRU model.
        x shape: (batch_size, seq_len, input_dim)
        mask shape: (batch_size, seq_len)
        """
        # 1. Pass the input features through the GRU layer
        # Hint: gru_out, _ = self.gru(x)
        gru_out, _ = self.gru(x)
        
        # 2. Apply dropout to the GRU outputs
        gru_out = self.dropout(gru_out)
        
        # 3. Pass through the linear layer to get raw prediction scores (logits)
        out = self.fc(gru_out)
        
        # 4. Squeeze the last dimension so shape matches batch labels: (batch_size, seq_len) or just the final step
        # Hint: out.squeeze(-1)
        return out.squeeze(-1)

class TennisMLPModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, dropout=0.2):
        super().__init__()
        # TODO: Define a sequential or linear layers for the MLP
        # Hint: Linear -> ReLU -> Dropout -> Linear -> outputlogit (size 1)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, mask=None):
        # x shape: (batch, time, features) or flattened
        # Note: If your dataloader feeds sequences, you can apply the MLP 
        # independently to each time step: (batch * time, features) -> reshape back!
        return self.net(x).squeeze(-1)