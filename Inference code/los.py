import os
import torch
from transformers import BertTokenizer, BertModel
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from torch_geometric.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix
from sklearn.neighbors import NearestNeighbors
import os

# --- Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_dir = 'saved_los_models'

# --- Load Saved Models ---
tokenizer = BertTokenizer.from_pretrained(model_dir)
bert_model = BertModel.from_pretrained(model_dir).to(device)

# bert_base = bert_model.bert  # Get base BERT for embeddings

class GCNClassifier(torch.nn.Module):
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

# --- Load GCN Model ---
gcn_model = GCNClassifier(in_channels=768, hidden_channels=16, num_classes=4).to(device)
gcn_model.load_state_dict(torch.load(os.path.join(model_dir, 'gcn_model.pth')))
gcn_model.eval()

# --- Prepare Text Data ---
def get_bert_embeddings(texts, tokenizer, model):
    model.eval()
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :]  # CLS token

# --- Inference Texts ---
# texts = ["patient has high fever and cough", "low oxygen saturation requiring ventilation"]

df = "LOS_WEEKS_adm_val.csv"

test_df = pd.read_csv(df,nrows=500)

texts = test_df["text"].tolist()

# --- Generate BERT Embeddings ---
X = get_bert_embeddings(texts, tokenizer, bert_model).to(device)

# --- Build Graph from Embeddings using KNN ---
knn = NearestNeighbors(n_neighbors=2, metric='cosine')
knn.fit(X.cpu())
adj_matrix = knn.kneighbors_graph(X.cpu(), mode='connectivity')
edge_index, _ = from_scipy_sparse_matrix(adj_matrix)

data = Data(x=X.to(device), edge_index=edge_index).to(device)

# --- Run Inference ---
with torch.no_grad():
    logits = gcn_model(data)
    probs = F.softmax(logits, dim=1).cpu().numpy()
    preds = torch.argmax(logits, dim=1).cpu().numpy()

# Assume test_df is your DataFrame with 'id' and 'text', and preds is your predictions array
results_df = test_df[['id']].copy()
results_df['prediction'] = preds

# Save to predictions.txt as tab-separated
results_df.to_csv('LOSpredictions.csv', index=False, header=True)
print('Predictions saved to LOSpredictions.csv')


