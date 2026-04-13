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

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EMB_DIM = 32
BATCH_SIZE = 2048
LR = 0.001
EPOCHS = 30
K = 20
KG_FILE = 'kg_final.txt'
USER_KG_FILE = 'user_kg.txt'
ML1M_DIR = "data"
L2_REG = 1e-5
N_LAYERS = 2
KG_WEIGHT = 0.25
TEMPERATURE = 0.2  
CONTRAST_WEIGHT = 0.05 
EDGE_DROPOUT = 0.2 

def set_seed(seed=2024):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class ML1MDataset(Dataset):
    def __init__(self, ratings_df, num_items, negative_samples=1):
        self.u = ratings_df['uid'].values.astype(np.int32)
        self.i = ratings_df['iid'].values.astype(np.float32)
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
        i_pos = torch.tensor(int(self.i[idx]), dtype=torch.long)
        user = self.u[idx]
        interacted = self.user_items.get(user, set())
        i_neg_list = []
        while len(i_neg_list) < self.negative_samples:
            i_neg = np.random.randint(0, self.num_items)
            if i_neg not in interacted:
                i_neg_list.append(i_neg)
        i_neg = torch.tensor(i_neg_list, dtype=torch.long)
        return (u_pos, i_pos, i_neg), (torch.tensor(1.0), torch.tensor(0.0))


class KGCLDataset:
    def __init__(self):
        ratings = pd.read_csv(os.path.join(ML1M_DIR, "ratings.dat"),
                              sep='::', engine='python', header=None,
                              names=['user', 'item', 'rating', 'ts'])
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

        kg = pd.read_csv(KG_FILE, sep='\t', header=None, names=['h', 'r', 't'])
        kg = kg[kg['h'].isin(self.item_map) & kg['t'].isin(self.item_map)]
        self.entities = set(kg['h']).union(set(kg['t']))
        self.rels = set(kg['r'])
        self.ent_map = {e: i + self.num_items for i, e in enumerate(self.entities)}
        self.rel_map = {r: i for i, r in enumerate(self.rels)}
        kg['h'] = kg['h'].map(self.item_map)
        kg['t'] = kg['t'].map(lambda x: self.ent_map[x] if x in self.ent_map else x)
        kg['r'] = kg['r'].map(self.rel_map)
        self.kg = kg.values
        self.num_entities = len(self.ent_map)
        self.num_rels = len(self.rel_map)

        self.user_kg = None
        if os.path.exists(USER_KG_FILE):
            user_kg = pd.read_csv(USER_KG_FILE, sep='\t', header=None, names=['h', 'r', 't'])
            user_rels = set(user_kg['r'])
            user_rel_offset = len(self.rel_map)
            for r in user_rels:
                if r not in self.rel_map:
                    self.rel_map[r] = user_rel_offset
                    user_rel_offset += 1
            user_kg['h'] = user_kg['h'].map(lambda x: self.user_map.get(x, x))
            user_kg['r'] = user_kg['r'].map(self.rel_map)
            self.user_kg = user_kg.values
            self.num_rels = len(self.rel_map)

        self._build_adj()

    def _build_adj(self):
        n_all = self.num_users + self.num_items + self.num_entities
        edges = []
        for _, row in self.train_df.iterrows():
            u = row['uid']
            i = row['iid'] + self.num_users
            edges.append([u, i])
            edges.append([i, u])
        for h, r, t in self.kg:
            h += self.num_users
            t += self.num_users
            edges.append([h, t])
            edges.append([t, h])
        
        edges = np.array(edges).T
        values = np.ones(edges.shape[1])
        self.adj_mat = sp.coo_matrix((values, edges), shape=(n_all, n_all))
        rowsum = np.array(self.adj_mat.sum(1)).flatten()
        d_inv_sqrt = np.power(rowsum, -0.5)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        self.norm_adj = d_mat_inv_sqrt.dot(self.adj_mat).dot(d_mat_inv_sqrt).tocsr()
        self.edges = edges

    def get_loaders(self):
        train_set = ML1MDataset(self.train_df, self.num_items, negative_samples=1)
        valid_set = ML1MDataset(self.valid_df, self.num_items, negative_samples=1)
        test_set = ML1MDataset(self.test_df, self.num_items, negative_samples=1)

        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
        valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)
        return train_loader, valid_loader, test_loader

class KGCL(nn.Module):
    def __init__(self, n_users, n_items, n_entities, n_rels, emb_dim, norm_adj, n_layers=2):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_entities = n_entities
        self.n_rels = n_rels
        self.n_nodes = n_users + n_items + n_entities
        self.emb_dim = emb_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(self.n_nodes, emb_dim)
        self.relation_emb = nn.Embedding(n_rels, emb_dim)
        nn.init.xavier_normal_(self.embedding.weight, gain=1.0)
        nn.init.xavier_normal_(self.relation_emb.weight, gain=1.0)

        self.gnn_layers = nn.ModuleList()
        for _ in range(n_layers):
            self.gnn_layers.append(nn.Linear(emb_dim, emb_dim))
        
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

    def _edge_dropout(self, dropout_rate):
        adj_coo = self.norm_adj.tocoo()
        n_edges = len(adj_coo.data)
        keep_idx = np.random.choice(n_edges, size=int(n_edges * (1 - dropout_rate)), replace=False)
        
        rows = adj_coo.row[keep_idx]
        cols = adj_coo.col[keep_idx]
        values = adj_coo.data[keep_idx]

        new_adj = sp.coo_matrix((values, (rows, cols)), shape=adj_coo.shape)
        rowsum = np.array(new_adj.sum(1)).flatten()
        d_inv_sqrt = np.power(rowsum, -0.5)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        new_norm_adj = d_mat_inv_sqrt.dot(new_adj).dot(d_mat_inv_sqrt).tocoo()

        indices = np.vstack((new_norm_adj.row, new_norm_adj.col))
        i = torch.LongTensor(indices).to(DEVICE)
        v = torch.FloatTensor(new_norm_adj.data).to(DEVICE)
        return torch.sparse_coo_tensor(i, v, torch.Size(new_norm_adj.shape), dtype=torch.float32).coalesce()

    def forward(self, adj=None):
        if adj is None:
            adj = self.adj_norm            
        ego_emb = self.embedding.weight
        all_emb = [ego_emb]        
        for layer in self.gnn_layers:
            side_emb = torch.sparse.mm(adj, ego_emb)
            ego_emb = layer(side_emb)
            ego_emb = F.leaky_relu(ego_emb)
            ego_emb = F.dropout(ego_emb, p=0.1, training=self.training)
            all_emb.append(ego_emb)

        out = torch.stack(all_emb, dim=0).mean(dim=0)        
        user_emb = out[:self.n_users]
        item_emb = out[self.n_users:self.n_users + self.n_items]
        return user_emb, item_emb

    def forward_multi_view(self):
        user_emb1, item_emb1 = self.forward()

        adj_aug = self._edge_dropout(EDGE_DROPOUT)
        user_emb2, item_emb2 = self.forward(adj_aug)
        
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
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )

    def contrastive_loss(self, emb1, emb2, temperature=TEMPERATURE):
        emb1 = F.normalize(emb1, dim=1)
        emb2 = F.normalize(emb2, dim=1)
        pos_sim = (emb1 * emb2).sum(dim=1)  # 正样本相似度

        neg_sim = torch.matmul(emb1, emb2.t())

        logits = neg_sim / temperature
        labels = torch.arange(len(emb1), device=DEVICE)
        
        loss = F.cross_entropy(logits, labels)
        return loss

    def kg_loss(self):
        kg = self.dataset.kg
        if len(kg) == 0:
            return torch.tensor(0.0, device=DEVICE)

        sample_size = min(256, len(kg))
        indices = np.random.choice(len(kg), sample_size, replace=False)
        
        h = torch.tensor([kg[i][0] + self.dataset.num_users for i in indices], device=DEVICE)
        r = torch.tensor([kg[i][1] for i in indices], device=DEVICE)
        t = torch.tensor([kg[i][2] + self.dataset.num_users for i in indices], device=DEVICE)
        neg_t = torch.randint(0, self.dataset.num_entities, (sample_size,), device=DEVICE)
        neg_t = neg_t + self.dataset.num_users
        h_emb = self.model.embedding(h)
        r_emb = self.model.relation_emb(r)
        t_emb = self.model.embedding(t)
        neg_t_emb = self.model.embedding(neg_t)
        pos_score = torch.norm(h_emb + r_emb - t_emb, p=2, dim=1)
        neg_score = torch.norm(h_emb + r_emb - neg_t_emb, p=2, dim=1)
        loss = -torch.log(torch.sigmoid(neg_score - pos_score) + 1e-10).mean()
        return loss

    def run_epoch(self, loader, train=True):
        if train:
            self.model.train()
        else:
            self.model.eval()
        total_loss = 0
        total_bpr = 0
        total_kg = 0
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
            contrast_loss_user = self.contrastive_loss(user_emb1, user_emb2)
            contrast_loss_item = self.contrastive_loss(item_emb1, item_emb2)
            contrast_loss = (contrast_loss_user + contrast_loss_item) / 2
            kg_loss_val = self.kg_loss() if train else torch.tensor(0.0, device=DEVICE)
            loss = bpr_loss + KG_WEIGHT * kg_loss_val + CONTRAST_WEIGHT * contrast_loss

            if torch.isinf(loss) or torch.isnan(loss):
                continue
            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

            total_loss += loss.item() * len(u)
            total_bpr += bpr_loss.item() * len(u)
            total_kg += kg_loss_val.item() * len(u)
            total_contrast += contrast_loss.item() * len(u)

        n = len(loader.dataset)
        return total_loss / n, total_bpr / n, total_kg / n, total_contrast / n

    def train(self, epochs=EPOCHS):
        best_val = float('inf')
        patience = 10
        patience_counter = 0
        
        for epoch in range(epochs):
            tr_loss, tr_bpr, tr_kg, tr_contrast = self.run_epoch(self.train_loader, True)
            val_loss, val_bpr, val_kg, val_contrast = self.run_epoch(self.valid_loader, False)            
            self.scheduler.step(val_loss)
            if not math.isinf(val_loss) and val_loss < best_val:
                best_val = val_loss
                torch.save(self.model.state_dict(), 'kgcl_best.pt')
                patience_counter = 0
            else:
                patience_counter += 1               
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

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

