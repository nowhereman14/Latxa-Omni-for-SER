import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score, classification_report, confusion_matrix
torch.set_num_threads(4)

TAG_MAP = {
            'neutroa': 0,
            'poza': 1,
            'tristura': 2,
            'harridura': 3,
            'haserrea': 4
}

def load_split(split_name):
    embeddings = np.load(f"emotion2vec_embeddings/base/emo2vec_{split_name}_embeddings.npy")
    labels = np.load(f"emotion2vec_embeddings/base/emo2vec_{split_name}_labels.npy")
    labels = np.array([TAG_MAP[label] for label in labels])
    return embeddings, labels

def to_tensor(embeddings, labels):
    X = torch.tensor(embeddings, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)
    return X, y

class DownstreamHead(nn.Module):
    def __init__(self, dim, l):
        super().__init__()

        self.learning_rate = 2e-5
        self.training_epochs = 100
        self.display_epoch_freq = 1
        self.dim = dim
        self.batch_size = 256
        self.relu = nn.ReLU()
        self.l = l
        self.hidden_layer_sizes = [256, 5]

        self.fc1 = nn.Linear(dim, 256)
        self.fc2 =nn.Linear(256, 5)

        self.optimizer = torch.optim.Adam(self.parameters(), lr = self.learning_rate, weight_decay = self.l)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        h0 = self.relu(self.fc1(x))
        logits = self.fc2(h0)
        return logits

    def validate(self, val_dataloader):
            super().eval()
            total_val_loss = 0
            with torch.no_grad():
                for embeddings, labels in val_dataloader:
                    outputs = self.forward(embeddings)
                    loss = self.criterion(outputs, labels)
                    total_val_loss += loss.item() * labels.size(0)
    
            avg_val_loss = (total_val_loss / len(val_dataloader.dataset))
            super().train()
            return avg_val_loss

    def fit(self, train_dataloader, val_dataloader):
            super().train()
            best_val_loss = float('inf')
            patience = 3
            patience_counter = 0
    
            for epoch in range(self.training_epochs):
                print(f"Starting epoch {epoch+1}", flush=True)
                total_train_loss = 0
                for embeddings, labels in train_dataloader:
                    self.optimizer.zero_grad()
                    outputs = self.forward(embeddings)
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    self.optimizer.step()
                    total_train_loss += (loss.item() * labels.size(0))
                avg_train_loss = (total_train_loss / len(train_dataloader.dataset))
                val_loss = self.validate(val_dataloader)
    
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(self.state_dict(), 'best_model.pth')
                    patience_counter = 0
                else:
                    patience_counter += 1
            
                if epoch % self.display_epoch_freq == 0:
                    print(f"Epoch {epoch+1}/{self.training_epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {val_loss:.4f}", flush=True)
                
                if patience_counter >= patience:
                    print(f"Early stopping in epoch {epoch + 1}")
                    break

            self.load_state_dict(torch.load('best_model.pth'))

if __name__ == "__main__":
    print("Loading train split...", flush=True)
    train_embeddings, train_labels = load_split("train")
 
    print("Loading validation split...", flush=True)
    val_embeddings, val_labels = load_split("val")
 
    print("Loading test split...", flush=True)
    test_embeddings, test_labels = load_split("test")
 
    print("Creating train tensors...", flush=True)
    train_X, train_y = to_tensor(train_embeddings, train_labels)
    train_dataset = TensorDataset(train_X, train_y)
    train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=0, pin_memory=False)
 
    print("Creating validation tensors...", flush=True)
    val_X, val_y = to_tensor(val_embeddings, val_labels)
    val_dataset = TensorDataset(val_X, val_y)
    val_dataloader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=0, pin_memory=False)
 
    print("Creating test tensors...", flush=True)
    test_X, test_y = to_tensor(test_embeddings, test_labels)
    test_dataset = TensorDataset(test_X, test_y)
    test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=0, pin_memory=False)
 
    print("Creating model...", flush=True)
    model = DownstreamHead(dim=train_X.shape[1], l=1e-4)
 
    print("Initializing training...", flush=True)
    model.fit(train_dataloader, val_dataloader)
 
    test_loss = model.validate(test_dataloader)
 
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for embeddings, labels in test_dataloader:
            outputs = model.forward(embeddings)
            all_preds.append(outputs.argmax(dim=1))
            all_labels.append(labels)
    model.train()
    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_labels).numpy()
    test_f1 = f1_score(y_true, y_pred, average="macro")
 
    print(f"\n=== TEST === loss: {test_loss:.4f}  f1_macro: {test_f1:.4f}", flush=True)
    print(classification_report(y_true, y_pred, target_names=list(TAG_MAP.keys()), digits=3))
    print("Confusion matrix (rows=real, columns=predicted):")
    print(confusion_matrix(y_true, y_pred))   
