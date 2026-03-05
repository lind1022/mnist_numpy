import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import time
import os

# ==========================================
# 1. Dummy Data Generation
# ==========================================
def generate_dummy_csv(filename="transactions.csv", n_rows=10000):
    print(f"Generating dummy data: {filename}...")
    
    # Create somewhat realistic patterns
    n_accounts = 500
    account_ids = [f"ACC_{i:04d}" for i in range(n_accounts)]
    types = ["PAYMENT", "TRANSFER", "DEBIT", "FEE", "REFUND"]
    
    data = {
        'originator_account': np.random.choice(account_ids, n_rows),
        'beneficiary_account': np.random.choice(account_ids, n_rows),
        'timestamp': [time.time() - np.random.randint(0, 1000000) for _ in range(n_rows)],
        'amount': np.random.exponential(scale=100, size=n_rows).round(2),
        'type': np.random.choice(types, n_rows)
    }
    
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print("Data generation complete.\n")

# ==========================================
# 2. Data Preprocessing & Vocabulary
# ==========================================
class DataPreprocessor:
    def __init__(self):
        self.acct_encoder = LabelEncoder()
        self.type_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        
    def fit_transform(self, df):
        print("Preprocessing data...")
        df = df.copy()
        
        # 1. Categorical Encoding (Accounts)
        # We combine both columns to ensure shared vocabulary
        all_accts = pd.concat([df['originator_account'], df['beneficiary_account']]).unique()
        self.acct_encoder.fit(all_accts)
        
        # Add 1 to index because 0 will be reserved for masking/padding in augmentation
        df['originator_idx'] = self.acct_encoder.transform(df['originator_account']) + 1
        df['beneficiary_idx'] = self.acct_encoder.transform(df['beneficiary_account']) + 1
        
        # 2. Categorical Encoding (Type)
        df['type_idx'] = self.type_encoder.fit_transform(df['type']) + 1
        
        # 3. Time Feature Engineering (Cyclical)
        # Convert epoch to hour of day for cyclical embedding
        dates = pd.to_datetime(df['timestamp'], unit='s')
        hours = dates.dt.hour
        # Sin/Cos transform to capture "23:00 is close to 00:00"
        df['time_sin'] = np.sin(2 * np.pi * hours / 24)
        df['time_cos'] = np.cos(2 * np.pi * hours / 24)
        
        # 4. Numerical Normalization (Amount)
        df[['amount']] = self.scaler.fit_transform(df[['amount']])
        
        self.vocab_size_acct = len(self.acct_encoder.classes_) + 2 # +1 for 0-index, +1 for potential unknown
        self.vocab_size_type = len(self.type_encoder.classes_) + 2
        
        return df

# ==========================================
# 3. Contrastive Dataset
# ==========================================
class ContrastiveTransactionDataset(Dataset):
    def __init__(self, processed_df):
        self.data = processed_df
        # Convert columns to tensors for speed
        self.orig = torch.tensor(self.data['originator_idx'].values, dtype=torch.long)
        self.bene = torch.tensor(self.data['beneficiary_idx'].values, dtype=torch.long)
        self.txn_type = torch.tensor(self.data['type_idx'].values, dtype=torch.long)
        
        # Numerical features: Amount, Time_Sin, Time_Cos
        self.numerical = torch.tensor(
            self.data[['amount', 'time_sin', 'time_cos']].values, 
            dtype=torch.float32
        )

    def __len__(self):
        return len(self.data)
    
    def augment(self, orig, bene, txn_type, numerical):
        """
        Create a 'view' of the data by randomly corrupting features.
        This forces the model to learn robust representations.
        """
        orig_aug = orig.clone()
        bene_aug = bene.clone()
        type_aug = txn_type.clone()
        num_aug = numerical.clone()
        
        # Probability of masking a categorical feature (setting to 0)
        mask_prob = 0.2
        
        if torch.rand(1) < mask_prob: orig_aug = 0
        if torch.rand(1) < mask_prob: bene_aug = 0
        if torch.rand(1) < mask_prob: type_aug = 0
        
        # Add Gaussian noise to numerical features
        noise = torch.randn_like(num_aug) * 0.1
        num_aug += noise
        
        return orig_aug, bene_aug, type_aug, num_aug

    def __getitem__(self, idx):
        # Retrieve raw row
        o, b, t, n = self.orig[idx], self.bene[idx], self.txn_type[idx], self.numerical[idx]
        
        # Generate two different augmented views of the SAME transaction
        view1 = self.augment(o, b, t, n)
        view2 = self.augment(o, b, t, n)
        
        return view1, view2

# ==========================================
# 4. The Embedding Model
# ==========================================
class TransactionEncoder(nn.Module):
    def __init__(self, vocab_acct, vocab_type, embedding_dim=64):
        super(TransactionEncoder, self).__init__()
        
        # 1. Embedding Layers
        # Padding_idx=0 ensures the masked value returns a zero vector
        self.acct_embedding = nn.Embedding(vocab_acct, embedding_dim, padding_idx=0)
        self.type_embedding = nn.Embedding(vocab_type, embedding_dim // 2, padding_idx=0)
        
        # 2. Numerical Processor (MLP)
        self.num_mlp = nn.Sequential(
            nn.Linear(3, 32), # 3 inputs: Amount, Sin, Cos
            nn.ReLU(),
            nn.Linear(32, 32)
        )
        
        # Calculate concatenation size:
        # Orig(64) + Bene(64) + Type(32) + Numerical(32) = 192
        concat_dim = embedding_dim * 2 + (embedding_dim // 2) + 32
        
        # 3. Main Encoder (Projection to latent space z)
        self.encoder = nn.Sequential(
            nn.Linear(concat_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128) # This is the final embedding size
        )
        
        # 4. Projection Head (for Contrastive Loss only - discarded after training)
        # Maps embedding to space where loss is calculated
        self.projector = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )

    def forward_one_view(self, orig, bene, txn_type, num):
        # Lookups
        emb_o = self.acct_embedding(orig)
        emb_b = self.acct_embedding(bene)
        emb_t = self.type_embedding(txn_type)
        
        # Process numericals
        emb_n = self.num_mlp(num)
        
        # Concatenate all features
        x = torch.cat([emb_o, emb_b, emb_t, emb_n], dim=1)
        
        # Encode
        representation = self.encoder(x)
        
        # Project
        projection = self.projector(representation)
        return representation, projection

    def forward(self, view1, view2):
        # Unpack views
        o1, b1, t1, n1 = view1
        o2, b2, t2, n2 = view2
        
        # Process both views
        h1, z1 = self.forward_one_view(o1, b1, t1, n1)
        h2, z2 = self.forward_one_view(o2, b2, t2, n2)
        
        return z1, z2

# ==========================================
# 5. Contrastive Loss (NT-Xent)
# ==========================================
class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, z_i, z_j):
        batch_size = z_i.shape[0]
        
        # Normalize vectors
        z_i = torch.nn.functional.normalize(z_i, dim=1)
        z_j = torch.nn.functional.normalize(z_j, dim=1)
        
        # Concatenate outputs
        representations = torch.cat([z_i, z_j], dim=0)
        
        # Similarity matrix (Cosine similarity)
        similarity_matrix = torch.matmul(representations, representations.T)
        
        # Create labels: For index i, the positive is i + batch_size
        # (and vice versa)
        labels = torch.cat([
            torch.arange(batch_size, 2 * batch_size), 
            torch.arange(batch_size)
        ], dim=0).to(z_i.device)
        
        # Mask out self-similarity (diagonal)
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(z_i.device)
        similarity_matrix = similarity_matrix / self.temperature
        
        # We need to remove the diagonal (self-similarity) from calculations
        # But CrossEntropyLoss expects standard logits. 
        # A simpler trick for SimCLR implementation:
        
        # Select positives
        sim_ij = torch.diag(similarity_matrix, batch_size)
        sim_ji = torch.diag(similarity_matrix, -batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0).view(2 * batch_size, 1)
        
        # Select negatives (all other similarities)
        # This part requires careful masking, simplified here for readability:
        # In a production SimCLR, we compute full CrossEntropy against all pairs
        
        logits = similarity_matrix # (2N, 2N)
        # To strictly use CrossEntropy, we set diagonal to -infinity
        logits.fill_diagonal_(-float('inf'))
        
        return self.criterion(logits, labels)

# ==========================================
# 6. Training Pipeline
# ==========================================
def train_embeddings():
    # Settings
    BATCH_SIZE = 256
    EPOCHS = 5
    LR = 3e-4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {DEVICE}")

    # 1. Load and Prep
    if not os.path.exists("transactions.csv"):
        generate_dummy_csv()
        
    df = pd.read_csv("transactions.csv")
    preprocessor = DataPreprocessor()
    df_processed = preprocessor.fit_transform(df)
    
    dataset = ContrastiveTransactionDataset(df_processed)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    
    # 2. Init Model
    model = TransactionEncoder(
        vocab_acct=preprocessor.vocab_size_acct,
        vocab_type=preprocessor.vocab_size_type
    ).to(DEVICE)
    
    loss_fn = NTXentLoss(temperature=0.1)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    # 3. Train Loop
    print("\nStarting training...")
    model.train()
    
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch in dataloader:
            # Unpack batch (each item in list is a tuple of features)
            # Dataloader stacks them: ([orig_batch], [bene_batch]...)
            view1_raw, view2_raw = batch
            
            # Move to device
            view1 = [t.to(DEVICE) for t in view1_raw]
            view2 = [t.to(DEVICE) for t in view2_raw]
            
            optimizer.zero_grad()
            
            # Forward
            z1, z2 = model(view1, view2)
            
            # Loss
            loss = loss_fn(z1, z2)
            
            # Backward
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss / len(dataloader):.4f}")

    # ==========================================
    # 7. Inference / Extraction
    # ==========================================
    print("\nExtracting embeddings for first 5 transactions...")
    model.eval()
    
    # Get clean data (no augmentation for inference)
    sample_data = df_processed.iloc[:5]
    
    # Manually prepare tensors
    orig = torch.tensor(sample_data['originator_idx'].values, dtype=torch.long).to(DEVICE)
    bene = torch.tensor(sample_data['beneficiary_idx'].values, dtype=torch.long).to(DEVICE)
    typ = torch.tensor(sample_data['type_idx'].values, dtype=torch.long).to(DEVICE)
    num = torch.tensor(sample_data[['amount', 'time_sin', 'time_cos']].values, dtype=torch.float32).to(DEVICE)
    
    with torch.no_grad():
        # We only need the representation h, not the projection z
        embeddings, _ = model.forward_one_view(orig, bene, typ, num)
        
    print("Embedding Shape:", embeddings.shape)
    print("First Embedding Vector sample:", embeddings[0][:5])

if __name__ == "__main__":
    train_embeddings()
