import os
import torch
from transformers import BertTokenizer, BertModel
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import pandas as pd

# Define GCNClassifier (same as in training)
class GCNClassifier(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

# Load everything from saved folder
model_dir = './saved_models'
tokenizer = BertTokenizer.from_pretrained(model_dir)
bert_base = BertModel.from_pretrained(model_dir)
gcn_model = GCNClassifier(in_channels=768, hidden_channels=8, num_classes=2)
gcn_model.load_state_dict(torch.load(os.path.join(model_dir, 'gcn_model.pth')))
class_weights = torch.load(os.path.join(model_dir, 'class_weights.pt'))
gcn_model.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gcn_model.to(device)
bert_base.to(device)

# Function to get BERT embeddings
def get_bert_embeddings(texts, tokenizer, model, device):
    model.eval()
    embeddings = []
    batch_size = 32
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        cls_embeddings = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        embeddings.append(cls_embeddings.cpu())
    return torch.cat(embeddings, dim=0)

# Example inference
df = "./MP_IN_adm_val.csv"

test_df = pd.read_csv(df)

texts = test_df["text"].tolist()
X = get_bert_embeddings(texts, tokenizer, bert_base, device).to(device)

from sklearn.neighbors import NearestNeighbors
from torch_geometric.utils import from_scipy_sparse_matrix
from torch_geometric.data import Data

knn = NearestNeighbors(n_neighbors=5, metric='cosine')
knn.fit(X.cpu())
adj_matrix = knn.kneighbors_graph(X.cpu(), mode='connectivity')
edge_index, _ = from_scipy_sparse_matrix(adj_matrix)
data = Data(x=X, edge_index=edge_index).to(device)

with torch.no_grad():
    logits = gcn_model(data)
    probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
    preds = torch.argmax(logits, dim=1).cpu().numpy()

import pandas as pd

# Assume test_df is your DataFrame with 'id' and 'text', and preds is your predictions array
results_df = test_df[['id']].copy()
results_df['prediction'] = preds

# Save to predictions.txt as tab-separated
results_df.to_csv('predictions.csv', index=False, header=True)
print('Predictions saved to predictions.csv')
