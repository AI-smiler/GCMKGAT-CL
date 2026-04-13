
import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import math

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EMB_DIM = 32
BATCH_SIZE = 2048
LR = 0.001
EPOCHS = 30
K = 20
L2_REG = 1e-5
N_LAYERS = 2

def set_seed(seed=2024):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class MultimodalFeatureExtractor(nn.Module):
    def __init__(self, num_users, num_items, movies_df, emb_dim=EMB_DIM):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.emb_dim = emb_dim

        textual_raw = self._extract_textual_raw(movies_df)

        self.text_proj = nn.Linear(textual_raw.shape[1], emb_dim, bias=False)
        nn.init.xavier_normal_(self.text_proj.weight, gain=1.0)

        self.register_buffer('textual_features', self.text_proj(textual_raw).detach())
        
        # 视觉和音频特征（随机初始化，可学习）
        self.visual_features = nn.Parameter(torch.randn(num_items, emb_dim) * 0.1)
        self.acoustic_features = nn.Parameter(torch.randn(num_items, emb_dim) * 0.1)
        
        print(f"视觉特征维度: {self.visual_features.shape}")
        print(f"音频特征维度: {self.acoustic_features.shape}")
        print(f"文本特征维度: {self.textual_features.shape}")

    def _extract_textual_raw(self, movies_df):
        genres_list = movies_df['genres'].apply(lambda x: x.split('|') if pd.notna(x) else [])
        
        all_genres = set()
        for genres in genres_list:
            all_genres.update(genres)
        all_genres = sorted(list(all_genres))
        
        genre_to_idx = {genre: idx for idx, genre in enumerate(all_genres)}
        num_genres = len(all_genres)
        
        text_features = np.zeros((self.num_items, num_genres), dtype=np.float32)
        
        for idx, genres in enumerate(genres_list):
            if idx < self.num_items:
                for genre in genres:
                    if genre in genre_to_idx:
                        text_features[idx, genre_to_idx[genre]] = 1.0
        
        print(f"文本原始特征维度: {text_features.shape}, 类型: {all_genres}")
        return torch.FloatTensor(text_features)

    def get_modal_features(self):
        return {
            'visual': self.visual_features,
            'acoustic': self.acoustic_features,
            'textual': self.textual_features
        }

class ML1MDataset(Dataset):
    def __init__(self, ratings_df, num_items, negative_samples=1):
        self.u = ratings_df['uid'].values.astype(np.int32)
        self.i = ratings_df['iid'].values.astype(np.int32)
        self.r = ratings_df['rating'].values.astype(np.float32)
        self.num_items = num_items
        self.negative_samples = negative_samples
        
        self.user_items = {}
        for u, i in zip(self.u, self.i):
            if u not in self.user_items:
                self.user_items[u] = set()
            self.user_items[u].add(i)

    def __len__(self):
        return len(self.u)

    def __getitem__(self, idx):
        u_pos = torch.tensor(self.u[idx], dtype=torch.long)
        i_pos = torch.tensor(self.i[idx], dtype=torch.long)
        
        user = self.u[idx]
        interacted = self.user_items.get(user, set())
        i_neg_list = []
        
        while len(i_neg_list) < self.negative_samples:
            i_neg = np.random.randint(0, self.num_items)
            if i_neg not in interacted:
                i_neg_list.append(i_neg)
        
        i_neg = torch.tensor(i_neg_list, dtype=torch.long)
        return (u_pos, i_pos, i_neg), (torch.tensor(1.0), torch.tensor(0.0))


class MMGCNDataset:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        
        ratings = self._load_ratings()
        movies_df = self._load_movies()
        users_df = self._load_users()
        
        ratings = ratings[ratings['rating'] >= 4]
        ratings['rating'] = 1

        self.user_map = {u: i for i, u in enumerate(ratings['user'].unique())}
        self.item_map = {i: j for j, i in enumerate(ratings['item'].unique())}

        ratings['uid'] = ratings['user'].map(self.user_map)
        ratings['iid'] = ratings['item'].map(self.item_map)
        ratings = ratings.dropna(subset=['uid', 'iid'])
        ratings['uid'] = ratings['uid'].astype(np.int32)
        ratings['iid'] = ratings['iid'].astype(np.int32)

        self.num_users = ratings['uid'].max() + 1
        self.num_items = ratings['iid'].max() + 1
        
        print(f"用户数: {self.num_users}, 电影数: {self.num_items}")

        ratings = ratings.sort_values('ts')
        train_end = int(0.7 * len(ratings))
        valid_end = int(0.8 * len(ratings))
        self.train_df = ratings.iloc[:train_end]
        self.valid_df = ratings.iloc[train_end:valid_end]
        self.test_df = ratings.iloc[valid_end:]
        
        print(f"训练集: {len(self.train_df)}, 验证集: {len(self.valid_df)}, 测试集: {len(self.test_df)}")

        self.mm_extractor = MultimodalFeatureExtractor(
            self.num_users, self.num_items, movies_df, EMB_DIM
        ).to(DEVICE)
        
        self._build_adj()

    def _load_ratings(self):
        ratings = pd.read_csv(
            os.path.join(self.data_dir, "ratings.txt"),
            sep='::', engine='python', header=None, encoding='latin-1',
            names=['user', 'item', 'rating', 'ts']
        )
        return ratings
    
    def _load_movies(self):
        movies = pd.read_csv(
            os.path.join(self.data_dir, "movies.txt"),
            sep='::', engine='python', header=None, encoding='latin-1',
            names=['movie_id', 'title', 'genres']
        )
        return movies
    
    def _load_users(self):
        users = pd.read_csv(
            os.path.join(self.data_dir, "users.txt"),
            sep='::', engine='python', header=None, encoding='latin-1',
            names=['user_id', 'gender', 'age', 'occupation', 'zipcode']
        )
        return users

    def _build_adj(self):
        n_nodes = self.num_users + self.num_items
        edges = []
        
        for _, row in self.train_df.iterrows():
            u = row['uid']
            i = row['iid'] + self.num_users
            edges.append([u, i])
            edges.append([i, u])
        
        edges = np.array(edges).T
        values = np.ones(edges.shape[1])
        
        adj_mat = sp.coo_matrix((values, edges), shape=(n_nodes, n_nodes))
        
        rowsum = np.array(adj_mat.sum(1)).flatten()
        rowsum[rowsum == 0] = 1e-8
        d_inv_sqrt = np.power(rowsum, -0.5)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        
        self.norm_adj = d_mat_inv_sqrt.dot(adj_mat).dot(d_mat_inv_sqrt).tocsr()
        
        print(f"邻接矩阵维度: {self.norm_adj.shape}")

    def get_loaders(self):
        train_set = ML1MDataset(self.train_df, self.num_items, negative_samples=1)
        valid_set = ML1MDataset(self.valid_df, self.num_items, negative_samples=1)
        test_set = ML1MDataset(self.test_df, self.num_items, negative_samples=1)

        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
        valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)
        return train_loader, valid_loader, test_loader

class MMGCN(nn.Module):
    def __init__(self, n_users, n_items, emb_dim, norm_adj, mm_extractor, n_layers=2):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_nodes = n_users + n_items
        self.emb_dim = emb_dim
        self.n_layers = n_layers
        self.mm_extractor = mm_extractor

        self.user_id_emb = nn.Embedding(n_users, emb_dim)
        self.item_id_emb = nn.Embedding(n_items, emb_dim)
        nn.init.xavier_normal_(self.user_id_emb.weight, gain=1.0)
        nn.init.xavier_normal_(self.item_id_emb.weight, gain=1.0)

        self.user_visual_emb = nn.Embedding(n_users, emb_dim)
        self.user_acoustic_emb = nn.Embedding(n_users, emb_dim)
        self.user_textual_emb = nn.Embedding(n_users, emb_dim)
        nn.init.xavier_normal_(self.user_visual_emb.weight, gain=1.0)
        nn.init.xavier_normal_(self.user_acoustic_emb.weight, gain=1.0)
        nn.init.xavier_normal_(self.user_textual_emb.weight, gain=1.0)

        self.visual_gcn = nn.ModuleList([nn.Linear(emb_dim, emb_dim, bias=False) for _ in range(n_layers)])
        self.acoustic_gcn = nn.ModuleList([nn.Linear(emb_dim, emb_dim, bias=False) for _ in range(n_layers)])
        self.textual_gcn = nn.ModuleList([nn.Linear(emb_dim, emb_dim, bias=False) for _ in range(n_layers)])
        
        for gcn in [self.visual_gcn, self.acoustic_gcn, self.textual_gcn]:
            for layer in gcn:
                nn.init.xavier_normal_(layer.weight, gain=1.0)
        
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.norm_adj = norm_adj
        self._build_sparse_tensor()
        self.to(DEVICE)

    def _build_sparse_tensor(self):
        adj_coo = self.norm_adj.tocoo()
        values = adj_coo.data
        indices = np.vstack((adj_coo.row, adj_coo.col))
        i = torch.LongTensor(indices).to(DEVICE)
        v = torch.FloatTensor(values).to(DEVICE)
        shape = torch.Size(adj_coo.shape)
        self.adj_norm = torch.sparse_coo_tensor(i, v, shape, dtype=torch.float32).coalesce()

    def forward(self):
        modal_features = self.mm_extractor.get_modal_features()
        
        uid = torch.cat([self.user_id_emb.weight, self.item_id_emb.weight], dim=0)

        visual_emb = torch.cat([
            self.user_visual_emb.weight,
            modal_features['visual']
        ], dim=0)
        
        acoustic_emb = torch.cat([
            self.user_acoustic_emb.weight,
            modal_features['acoustic']
        ], dim=0)
        
        textual_emb = torch.cat([
            self.user_textual_emb.weight,
            modal_features['textual']
        ], dim=0)

        for layer_idx in range(self.n_layers):
            side_visual = torch.sparse.mm(self.adj_norm, visual_emb)
            visual_emb = self.leaky_relu(self.visual_gcn[layer_idx](side_visual))
            
            side_acoustic = torch.sparse.mm(self.adj_norm, acoustic_emb)
            acoustic_emb = self.leaky_relu(self.acoustic_gcn[layer_idx](side_acoustic))
            
            side_textual = torch.sparse.mm(self.adj_norm, textual_emb)
            textual_emb = self.leaky_relu(self.textual_gcn[layer_idx](side_textual))

        fused_emb = visual_emb + acoustic_emb + textual_emb + uid

        user_emb = fused_emb[:self.n_users]
        item_emb = fused_emb[self.n_users:]
        
        return user_emb, item_emb

    def predict(self, users, items):
        user_emb, item_emb = self.forward()
        u_emb = user_emb[users]
        i_emb = item_emb[items]
        logits = (u_emb * i_emb).sum(dim=-1)
        return logits

class Trainer:
    def __init__(self, model, train_loader, valid_loader, test_loader):
        self.model = model.to(DEVICE)
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=LR, weight_decay=L2_REG
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )

    def run_epoch(self, loader, train=True):
        if train:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0
        for (u, i_pos, i_neg), _ in loader:
            u = u.to(DEVICE)
            i_pos = i_pos.to(DEVICE)
            i_neg = i_neg.squeeze(1).to(DEVICE)

            if train:
                self.optimizer.zero_grad()

            pos_scores = self.model.predict(u, i_pos)
            neg_scores = self.model.predict(u, i_neg)

            loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10).mean()

            if torch.isinf(loss) or torch.isnan(loss):
                continue

            if train:
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item() * len(u)

        return total_loss / len(loader.dataset)

    def train(self, epochs=EPOCHS):
        best_val = float('inf')
        patience = 5
        patience_counter = 0
        
        for epoch in range(epochs):
            tr_loss = self.run_epoch(self.train_loader, True)
            
            if (epoch + 1) % 3 == 0 or epoch == epochs - 1:
                val_loss = self.run_epoch(self.valid_loader, False)
                self.scheduler.step(val_loss)
                print(f"Epoch {epoch+1:02d} | train_loss={tr_loss:.4f} | val_loss={val_loss:.4f}")

                if not math.isinf(val_loss) and val_loss < best_val:
                    best_val = val_loss
                    torch.save(self.model.state_dict(), 'mmgcn_best.pt')
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            else:
                print(f"Epoch {epoch+1:02d} | train_loss={tr_loss:.4f}")

class Evaluator:
    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset
        self.n_users = dataset.num_users
        self.n_items = dataset.num_items
        self.K = K
        self.all_items = torch.arange(self.n_items, device=DEVICE)

    def _recall_ndcg(self, df):
        user_pos = df.groupby('uid')['iid'].apply(list).to_dict()
        recalls, ndcgs, mrrs = [], [], []
        
        for u in tqdm(range(self.n_users), desc="Evaluating"):
            seen = set(self.dataset.train_df[self.dataset.train_df['uid'] == u]['iid'].tolist())
            pos = user_pos.get(u, [])
            if not pos:
                continue
            
            scores = self.model.predict(torch.tensor([u] * self.n_items, device=DEVICE), self.all_items)
            scores[list(seen)] = -float('inf')
            
            _, topk = torch.topk(scores, k=self.K)
            topk = topk.cpu().numpy()
            hit = np.isin(topk, pos)
            recall = hit.sum() / len(pos)

            dcg = (hit / np.log2(np.arange(2, self.K + 2))).sum()
            idcg = (1.0 / np.log2(np.arange(2, min(len(pos), self.K) + 2))).sum()
            ndcg = dcg / (idcg + 1e-9)

            rr = 1.0 / (np.where(hit)[0][0] + 1) if hit.any() else 0.0
            
            recalls.append(recall)
            ndcgs.append(ndcg)
            mrrs.append(rr)
            
        return np.mean(recalls), np.mean(ndcgs), np.mean(mrrs)

    def _coverage(self):
        scores = []
        for u in range(self.n_users):
            s = self.model.predict(torch.tensor([u]*self.n_items, device=DEVICE), self.all_items)
            _, topk = torch.topk(s, k=self.K)
            scores.extend(topk.cpu().numpy())
        return len(set(scores)) / self.n_items

    def evaluate(self, df):
        recall, ndcg, mrr = self._recall_ndcg(df)
        coverage = self._coverage()
        return dict(recall=recall, ndcg=ndcg, mrr=mrr, coverage=coverage)

