import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from torch.nn.utils.rnn import pad_sequence

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