import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.utils import from_scipy_sparse_matrix
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv
from sklearn.model_selection import train_test_split

train_path = "LOS_WEEKS_adm_train.csv"#../aa_nic/text.csv"
val_path = "LOS_WEEKS_adm_val.csv"
test_path = "LOS_WEEKS_adm_test.csv"

train_df = pd.read_csv(train_path)
val_df= pd.read_csv(val_path)
test_df = pd.read_csv(test_path)

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print(device)

# --- Data Cleaning and Preparation ---
train_df = train_df.rename(columns={'los_label': 'label'})
val_df = val_df.rename(columns={'los_label': 'label'})
test_df = test_df.rename(columns={'los_label': 'label'})

train_df = train_df.dropna(subset=['label'])
val_df = val_df.dropna(subset=['label'])
test_df = test_df.dropna(subset=['label'])

train_df = train_df[train_df['label'].astype(str).isin(['0', '1','2','3'])]
val_df = val_df[val_df['label'].astype(str).isin(['0', '1','2','3'])]
test_df = test_df[test_df['label'].astype(str).isin(['0', '1','2','3'])]

train_df['label'] = train_df['label'].astype(int)
val_df['label'] = val_df['label'].astype(int)
test_df['label'] = test_df['label'].astype(int)




# --- Hugging Face Dataset and Tokenization ---
train_dataset = Dataset.from_pandas(train_df[['text', 'label']])
val_dataset = Dataset.from_pandas(val_df[['text', 'label']])
test_dataset = Dataset.from_pandas(test_df[['text', 'label']])

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(example):
    return tokenizer(example["text"], padding="max_length", truncation=True, max_length=128)

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# --- BERT Model Training and Evaluation ---
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=4).to(device)

# def compute_metrics(eval_pred):
#     logits, labels = eval_pred
#     preds = np.argmax(logits, axis=1)
#     return {
#         "accuracy": accuracy_score(labels, preds),
#         "f1": f1_score(labels, preds),
#         "auroc": roc_auc_score(labels, logits[:, 1])
#     }


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = F.softmax(torch.tensor(logits), dim=1).numpy()
    preds = np.argmax(probs, axis=1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average='macro'),
#         "f1_micro": f1_score(labels, preds, average='micro'),
#         "f1_weighted": f1_score(labels, preds, average='weighted'),
        "auroc": roc_auc_score(labels, probs, multi_class='ovr', average='macro'),
#         "auroc_micro": roc_auc_score(labels, probs, multi_class='ovr', average='micro'),
    }




training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="f1"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

trainer.train()
results = trainer.evaluate(eval_dataset=test_dataset)


# --- GNN Model using BERT Embeddings as Node Features ---
# Get BERT embeddings for each text (use [CLS] token)
def get_bert_embeddings(texts, tokenizer, model):
    model.eval()
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.bert(**inputs)
    embeddings = outputs.last_hidden_state[:, 0, :]  # [CLS] token
    return embeddings

train_texts = train_df['text'].tolist()
val_texts = val_df['text'].tolist()
test_texts = test_df['text'].tolist()

# Use the BERT model's base (not classifier head) for embeddings
bert_base = model.bert

X_train = get_bert_embeddings(train_texts, tokenizer, model)
X_val = get_bert_embeddings(val_texts, tokenizer, model)
X_test = get_bert_embeddings(test_texts, tokenizer, model)


# Ensure data and labels on correct device
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Use already computed BERT embeddings
X_train = X_train.to(device)
X_val = X_val.to(device)
X_test = X_test.to(device)

y_train = torch.tensor(train_df['label'].values, dtype=torch.long).to(device)
y_val = torch.tensor(val_df['label'].values, dtype=torch.long).to(device)
y_test = torch.tensor(test_df['label'].values, dtype=torch.long).to(device)


# Build graph for training data
knn = NearestNeighbors(n_neighbors=5, metric='cosine')
knn.fit(X_train.cpu())  # Fit on CPU for NearestNeighbors
adj_matrix = knn.kneighbors_graph(X_train.cpu(), mode='connectivity')
edge_index, _ = from_scipy_sparse_matrix(adj_matrix)

data = Data(x=X_train, edge_index=edge_index, y=y_train).to(device)

# --- Define GCN ---
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

# --- Prepare for training ---
model = GCNClassifier(in_channels=X_train.shape[1], hidden_channels=64, num_classes=4).to(device)

class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train.cpu()), y=y_train.cpu().numpy())
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss(weight=class_weights)

# --- Training Loop with Early Stopping ---
best_f1 = 0
patience = 15
counter = 0

for epoch in range(50):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out, data.y)
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        val_logits = model(data)
        val_preds = torch.argmax(val_logits, dim=1)
        val_f1 = f1_score(data.y.cpu(), val_preds.cpu(), average='macro')

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}, Val Macro F1: {val_f1:.4f}")

    if val_f1 > best_f1:
        best_f1 = val_f1
        counter = 0
        best_model_state = model.state_dict()
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping triggered.")
            break

model.load_state_dict(best_model_state)

# --- Final Evaluation ---
model.eval()
with torch.no_grad():
    logits = model(data)
    probs = F.softmax(logits, dim=1).cpu().numpy()
    preds = torch.argmax(logits, dim=1).cpu().numpy()
    labels = data.y.cpu().numpy()

accuracy = accuracy_score(labels, preds)
f1 = f1_score(labels, preds, average='micro')
auroc = roc_auc_score(labels, probs, multi_class='ovr', average='macro')

print("\n--- GCN Evaluation Results (Train Graph) ---")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1: {f1:.4f}")
print(f"AUROC: {auroc:.4f}")
print("\nClassification Report:\n", classification_report(labels, preds))


