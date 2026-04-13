# ==============================================
# 版权所有 © [2026] [Jing Wang]
# 禁止盗用、修改、商用，未经授权禁止一切使用
# Copyright (c) [2026] [Jing Wang]. All Rights Reserved.
# ==============================================
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
from collections import defaultdict

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EMB_DIM = 32
BATCH_SIZE = 2048
LR = 0.001
EPOCHS = 30  
K = 10
DATA_DIR = "data"
L2_REG = 1e-5
N_LAYERS = 2

TEMPERATURE = 0.2  
CONTRAST_WEIGHT = 0.05 
KG_EDGE_DROPOUT = 0.3  
UI_EDGE_DROPOUT = 0.2  

def set_seed(seed=2024):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class KnowledgeGraph:
    def __init__(self, kg_file, entity_map, relation_map):
        self.kg_file = kg_file
        self.entity_map = entity_map  
        self.relation_map = relation_map  
        self.triplets = []
        self.entity2items = defaultdict(set)  
        self.item2entities = defaultdict(set) 
        self._load_kg()
        
    def _load_kg(self):
        with open(self.kg_file, 'r', encoding='latin-1') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    h, r, t = int(parts[0]), int(parts[1]), int(parts[2])
                    self.triplets.append((h, r, t))
                    # 记录实体-物品关联
                    if h in self.entity_map:
                        self.entity2items[h].add(h)
                    if t in self.entity_map:
                        self.entity2items[t].add(t)
                        
    def get_kg_embeddings(self, n_entities, n_relations, emb_dim):
        entity_emb = nn.Embedding(n_entities, emb_dim)
        relation_emb = nn.Embedding(n_relations, emb_dim)
        nn.init.xavier_normal_(entity_emb.weight, gain=1.0)
        nn.init.xavier_normal_(relation_emb.weight, gain=1.0)
        return entity_emb, relation_emb

    def augment_kg(self, dropout_rate=KG_EDGE_DROPOUT):
        n_keep = int(len(self.triplets) * (1 - dropout_rate))
        keep_idx = np.random.choice(len(self.triplets), size=n_keep, replace=False)
        augmented_triplets = [self.triplets[i] for i in keep_idx]
        return augmented_triplets

class MultimodalFeatureExtractor:
    def __init__(self, num_users, num_items, movies_df, users_df):
        self.num_users = num_users
        self.num_items = num_items
        self.visual_features, self.text_features = self._extract_item_features(movies_df)
        self.user_features = self._extract_user_features(users_df)

    def _extract_item_features(self, movies_df):
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
        visual_dim = 128
        np.random.seed(42)
        visual_features = np.random.randn(self.num_items, visual_dim).astype(np.float32) * 0.1
        for i in range(min(self.num_items, len(text_features))):
            genre_indices = np.where(text_features[i] > 0)[0]
            if len(genre_indices) > 0:
                for g_idx in genre_indices:
                    visual_features[i, g_idx % visual_dim] += 0.3
        text_features = text_features / (np.linalg.norm(text_features, axis=1, keepdims=True) + 1e-8)
        visual_features = visual_features / (np.linalg.norm(visual_features, axis=1, keepdims=True) + 1e-8)
        
        return (torch.FloatTensor(visual_features),
                torch.FloatTensor(text_features))

    def _extract_user_features(self, users_df):
        gender_map = {'M': 0, 'F': 1}
        genders = users_df['gender'].map(gender_map).fillna(0).values

        age_bins = [0, 18, 25, 35, 45, 50, 56, 100]
        age_labels = [0, 1, 2, 3, 4, 5, 6]
        ages = pd.cut(users_df['age'], bins=age_bins, labels=age_labels, right=False).fillna(0).values
        occupations = users_df['occupation'].fillna(0).values
        user_features = np.zeros((self.num_users, 3), dtype=np.float32)        
        for idx in range(min(len(users_df), self.num_users)):
            user_features[idx, 0] = genders[idx] if idx < len(genders) else 0
            user_features[idx, 1] = ages[idx] if idx < len(ages) else 0
            user_features[idx, 2] = occupations[idx] if idx < len(occupations) else 0
        gender_onehot = np.eye(2)[user_features[:, 0].astype(int)]
        age_onehot = np.eye(7)[user_features[:, 1].astype(int)]
        occ_onehot = np.eye(21)[user_features[:, 2].astype(int) % 21]        
        user_features_onehot = np.concatenate([gender_onehot, age_onehot, occ_onehot], axis=1)        
        return torch.FloatTensor(user_features_onehot)

    def get_modal_features(self):
        return [self.visual_features, self.text_features, self.user_features]

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


class GCLKGDataset:
    def __init__(self, data_dir=DATA_DIR):
        self.data_dir = data_dir
        ratings = self._load_ratings()
        movies_df = self._load_movies()
        users_df = self._load_users()
        kg_file = os.path.join(data_dir, "kg_final.txt")
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

        ratings = ratings.sort_values('ts')
        train_end = int(0.7 * len(ratings))
        valid_end = int(0.8 * len(ratings))
        self.train_df = ratings.iloc[:train_end]
        self.valid_df = ratings.iloc[train_end:valid_end]
        self.test_df = ratings.iloc[valid_end:]

        self.kg = KnowledgeGraph(kg_file, self.item_map, {})
        self.mm_extractor = MultimodalFeatureExtractor(
            self.num_users, self.num_items, movies_df, users_df
        )
        self.modal_features = self.mm_extractor.get_modal_features()

        self._build_adj()

    def _load_ratings(self):
        ratings = pd.read_csv(
            os.path.join(self.data_dir, "ratings.txt"),
            sep='::', engine='python', header=None, encoding='latin-1',
            names=['user', 'item', 'rating', 'ts']
        )
        return ratings
    
    def _load_movies(self):
        """加载电影数据"""
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
        n_all = self.num_users + self.num_items
        edges = []
        for _, row in self.train_df.iterrows():
            u = row['uid']
            i = row['iid'] + self.num_users
            edges.append([u, i])
            edges.append([i, u])
        
        edges = np.array(edges).T
        values = np.ones(edges.shape[1])
        adj_mat = sp.coo_matrix((values, edges), shape=(n_all, n_all))
        rowsum = np.array(adj_mat.sum(1)).flatten()
        d_inv_sqrt = np.power(rowsum, -0.5)
        d_inv_sqrt[np.isinf(d_inv_sqrt) | np.isnan(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        self.norm_adj = d_mat_inv_sqrt.dot(adj_mat).dot(d_mat_inv_sqrt).tocsr()

    def get_loaders(self):
        train_set = ML1MDataset(self.train_df, self.num_items, negative_samples=1)
        valid_set = ML1MDataset(self.valid_df, self.num_items, negative_samples=1)
        test_set = ML1MDataset(self.test_df, self.num_items, negative_samples=1)

        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)
        return train_loader, valid_loader, test_loader

class ModalEncoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1)
        )
    
    def forward(self, x):
        return self.encoder(x)

class GCLKG(nn.Module):
    def __init__(self, n_users, n_items, emb_dim, norm_adj, kg, modal_features, n_layers=2):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_nodes = n_users + n_items
        self.emb_dim = emb_dim
        self.n_layers = n_layers
        self.kg = kg
        self.n_modalities = len(modal_features)
        self._modal_features = [f.to(DEVICE) for f in modal_features]
        self.user_id_emb = nn.Embedding(n_users, emb_dim)
        self.item_id_emb = nn.Embedding(n_items, emb_dim)
        nn.init.xavier_normal_(self.user_id_emb.weight, gain=1.0)
        nn.init.xavier_normal_(self.item_id_emb.weight, gain=1.0)

        self.modal_encoders = nn.ModuleList()
        for m, features in enumerate(modal_features):
            input_dim = features.shape[1]
            self.modal_encoders.append(ModalEncoder(input_dim, emb_dim))
        self.modal_fusion = nn.Sequential(
            nn.Linear(emb_dim * self.n_modalities, emb_dim),
            nn.LayerNorm(emb_dim),
            nn.LeakyReLU(0.2)
        )
        self.modal_attention = nn.Sequential(
            nn.Linear(emb_dim * 2, emb_dim),
            nn.Tanh(),
            nn.Linear(emb_dim, self.n_modalities),
            nn.Softmax(dim=-1)
        )
        
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
        self.sparse_adj = torch.sparse_coo_tensor(i, v, shape, dtype=torch.float32).coalesce()

    def _edge_dropout(self, dropout_rate=UI_EDGE_DROPOUT):
        adj_coo = self.norm_adj.tocoo()
        n_edges = len(adj_coo.data)
        keep_idx = np.random.choice(n_edges, size=int(n_edges * (1 - dropout_rate)), replace=False)
        rows = adj_coo.row[keep_idx]
        cols = adj_coo.col[keep_idx]
        values = adj_coo.data[keep_idx]
        new_adj = sp.coo_matrix((values, (rows, cols)), shape=adj_coo.shape)
        rowsum = np.array(new_adj.sum(1)).flatten()
        d_inv_sqrt = np.power(rowsum, -0.5)
        d_inv_sqrt[np.isinf(d_inv_sqrt) | np.isnan(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        new_norm_adj = d_mat_inv_sqrt.dot(new_adj).dot(d_mat_inv_sqrt).tocoo()        
        indices = np.vstack((new_norm_adj.row, new_norm_adj.col))
        i = torch.LongTensor(indices).to(DEVICE)
        v = torch.FloatTensor(new_norm_adj.data).to(DEVICE)
        return torch.sparse_coo_tensor(i, v, torch.Size(new_norm_adj.shape), dtype=torch.float32).coalesce()

    def _modal_masking(self, modal_emb, mask_ratio=0.2):
        mask = torch.rand_like(modal_emb) > mask_ratio
        return modal_emb * mask.float()

    def forward(self, adj=None, mask_modal=False):
        user_id_emb = self.user_id_emb.weight
        item_id_emb = self.item_id_emb.weight
        modal_embs = []
        for m, encoder in enumerate(self.modal_encoders):
            if m < 2:  
                item_modal = encoder(self._modal_features[m])
            else:  
                user_modal = encoder(self._modal_features[m])
                item_modal = user_modal.mean(dim=0, keepdim=True).expand(self.n_items, -1)
            
            if mask_modal:
                item_modal = self._modal_masking(item_modal, 0.2)
            modal_embs.append(item_modal)

        item_modal_fused = torch.stack(modal_embs, dim=1)  # [n_items, n_modal, emb_dim]

        user_item_att = torch.cat([
            user_id_emb.mean(dim=0, keepdim=True).expand(self.n_items, -1),
            item_id_emb
        ], dim=1)
        modal_weights = self.modal_attention(user_item_att)  # [n_items, n_modal]

        item_modal_weighted = (item_modal_fused * modal_weights.unsqueeze(-1)).sum(dim=1)

        user_emb = user_id_emb
        item_emb = item_id_emb + item_modal_weighted
        
        all_emb = torch.cat([user_emb, item_emb], dim=0)
        all_embs = [all_emb]
        
        for layer in range(self.n_layers):
            if adj is not None:
                all_emb = torch.sparse.mm(adj, all_emb)
            else:
                all_emb = torch.sparse.mm(self.sparse_adj, all_emb)
            all_embs.append(all_emb)

        out = torch.stack(all_embs, dim=0).mean(dim=0)
        
        user_emb = out[:self.n_users]
        item_emb = out[self.n_users:]
        
        return user_emb, item_emb

    def forward_multi_view(self):
        user_emb1, item_emb1 = self.forward(mask_modal=False)
        adj_aug = self._edge_dropout(UI_EDGE_DROPOUT)
        user_emb2, item_emb2 = self.forward(adj=adj_aug, mask_modal=True)
        
        return (user_emb1, item_emb1), (user_emb2, item_emb2)

    def predict(self, users, items):
        user_emb, item_emb = self.forward()
        u_emb = user_emb[users]
        i_emb = item_emb[items]
        logits = (u_emb * i_emb).sum(dim=-1)
        return logits

class Trainer:
    def __init__(self, model, dataset, train_loader, valid_loader, test_loader):
        self.model = model.to(DEVICE)
        self.dataset = dataset
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=LR, weight_decay=L2_REG
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=20, gamma=0.5
        )

    def info_nce_loss(self, emb1, emb2, temperature=TEMPERATURE):
        emb1 = F.normalize(emb1, dim=1)
        emb2 = F.normalize(emb2, dim=1)
        pos_sim = torch.sum(emb1 * emb2, dim=1) / temperature
        neg_sim = torch.matmul(emb1, emb2.t()) / temperature
        labels = torch.arange(len(emb1), device=DEVICE)
        
        loss = F.cross_entropy(neg_sim, labels)
        return loss

    def run_epoch(self, loader, train=True):
        if train:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0
        total_bpr = 0
        total_contrast = 0
        
        for (u, i_pos, i_neg), _ in loader:
            u = u.to(DEVICE)
            i_pos = i_pos.to(DEVICE)
            i_neg = i_neg.squeeze(1).to(DEVICE)
            if train:
                self.optimizer.zero_grad()
            (user_emb1, item_emb1), (user_emb2, item_emb2) = self.model.forward_multi_view()
            u_emb = user_emb1[u]
            i_pos_emb = item_emb1[i_pos]
            i_neg_emb = item_emb1[i_neg]
            
            pos_scores = (u_emb * i_pos_emb).sum(dim=-1)
            neg_scores = (u_emb * i_neg_emb).sum(dim=-1)
            bpr_loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10).mean()
            contrast_loss_user = self.info_nce_loss(user_emb1, user_emb2)
            contrast_loss_item = self.info_nce_loss(item_emb1, item_emb2)
            contrast_loss = (contrast_loss_user + contrast_loss_item) / 2
            loss = bpr_loss + CONTRAST_WEIGHT * contrast_loss

            if torch.isinf(loss) or torch.isnan(loss):
                continue

            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

            total_loss += loss.item() * len(u)
            total_bpr += bpr_loss.item() * len(u)
            total_contrast += contrast_loss.item() * len(u)

        n = len(loader.dataset)
        return total_loss / n, total_bpr / n, total_contrast / n

    def train(self, epochs=EPOCHS):
        best_val = float('inf')
        patience = 10
        patience_counter = 0
        
        for epoch in range(epochs):
            tr_loss, tr_bpr, tr_contrast = self.run_epoch(self.train_loader, True)
            if (epoch + 1) % 3 == 0 or epoch == epochs - 1:
                val_loss, val_bpr, val_contrast = self.run_epoch(self.valid_loader, False)
                self.scheduler.step()

                if not math.isinf(val_loss) and val_loss < best_val:
                    best_val = val_loss
                    torch.save(self.model.state_dict(), 'gclkg_best.pt')
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            else:
                print(f"Epoch {epoch+1:02d} | loss={tr_loss:.4f} (bpr={tr_bpr:.4f}, contrast={tr_contrast:.4f})")

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


