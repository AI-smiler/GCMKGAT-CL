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
NEG_SAMPLE_NUM = 1
EVAL_BATCH_SIZE = 100

def set_seed(seed=2024):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def batch_neg_sampling(user_items, num_users, num_items, neg_sample_num):
    neg_samples = defaultdict(list)
    all_items = np.arange(num_items)
    for u in range(num_users):
        interacted = user_items.get(u, set())
        if len(interacted) >= num_items - neg_sample_num:
            neg_items = np.random.choice(list(interacted), neg_sample_num, replace=True)
        else:
            mask = np.ones(num_items, dtype=bool)
            mask[list(interacted)] = False
            neg_candidates = all_items[mask]
            neg_items = np.random.choice(neg_candidates, neg_sample_num, replace=False)
        neg_samples[u] = neg_items
    return neg_samples

class ML1MDataset(Dataset):
    def __init__(self, ratings_df, num_items, neg_samples):
        self.u = ratings_df['uid'].values.astype(np.int32)
        self.i = ratings_df['iid'].values.astype(np.int32)
        self.r = ratings_df['rating'].values.astype(np.float32)
        self.num_items = num_items
        self.neg_samples = neg_samples

    def __len__(self):
        return len(self.u)

    def __getitem__(self, idx):
        u_pos = torch.tensor(self.u[idx], dtype=torch.long)
        i_pos = torch.tensor(self.i[idx], dtype=torch.long)
        i_neg = torch.tensor(self.neg_samples[self.u[idx]], dtype=torch.long)
        return (u_pos, i_pos, i_neg), (torch.tensor(1.0), torch.tensor(0.0))


class KMCLDataset:
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

        kg = pd.read_csv(KG_FILE, sep='\t', header=None, names=['h', 'r', 't']) if os.path.exists(
            KG_FILE) else pd.DataFrame(columns=['h', 'r', 't'])
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
        self._precompute_neg_samples()

    def _build_adj(self):
        n_all = self.num_users + self.num_items + self.num_entities
        edges = []
        ui_edges = self.train_df[['uid', 'iid']].values
        ui_edges[:, 1] += self.num_users
        edges.extend(ui_edges.tolist())
        edges.extend(ui_edges[:, [1, 0]].tolist())

        kg_edges = self.kg[:, [0, 2]]
        kg_edges += self.num_users
        edges.extend(kg_edges.tolist())
        edges.extend(kg_edges[:, [1, 0]].tolist())

        edges = np.array(edges).T if edges else np.array([[], []])
        values = np.ones(edges.shape[1], dtype=np.float32) if edges.shape[1] > 0 else np.array([])
        self.adj_mat = sp.coo_matrix((values, edges), shape=(n_all, n_all), dtype=np.float32)

        rowsum = np.array(self.adj_mat.sum(1)).flatten()
        d_inv_sqrt = np.power(rowsum, -0.5, where=(rowsum != 0))
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt, dtype=np.float32)
        self.norm_adj = d_mat_inv_sqrt.dot(self.adj_mat).dot(d_mat_inv_sqrt).tocsr()

        self._convert_adj_to_tensor()

    def _convert_adj_to_tensor(self):
        adj_coo = self.norm_adj.tocoo()
        indices = torch.LongTensor(np.vstack((adj_coo.row, adj_coo.col))).to(DEVICE)
        values = torch.FloatTensor(adj_coo.data).to(DEVICE)
        shape = torch.Size(adj_coo.shape)
        self.adj_norm = torch.sparse_coo_tensor(indices, values, shape, dtype=torch.float32).coalesce()

    def _precompute_neg_samples(self):
        user_items = defaultdict(set)
        for u, i in zip(self.train_df['uid'], self.train_df['iid']):
            user_items[u].add(i)

        self.train_neg = batch_neg_sampling(user_items, self.num_users, self.num_items, NEG_SAMPLE_NUM)
        self.valid_neg = batch_neg_sampling(user_items, self.num_users, self.num_items, NEG_SAMPLE_NUM)
        self.test_neg = batch_neg_sampling(user_items, self.num_users, self.num_items, NEG_SAMPLE_NUM)

    def get_loaders(self):
        train_set = ML1MDataset(self.train_df, self.num_items, self.train_neg)
        valid_set = ML1MDataset(self.valid_df, self.num_items, self.valid_neg)
        test_set = ML1MDataset(self.test_df, self.num_items, self.test_neg)

        train_loader = DataLoader(
            train_set, batch_size=BATCH_SIZE, shuffle=True,
            pin_memory=True, num_workers=0 if os.name == 'nt' else 2
        )
        valid_loader = DataLoader(
            valid_set, batch_size=BATCH_SIZE, shuffle=False,
            pin_memory=True, num_workers=0 if os.name == 'nt' else 2
        )
        test_loader = DataLoader(
            test_set, batch_size=BATCH_SIZE, shuffle=False,
            pin_memory=True, num_workers=0 if os.name == 'nt' else 2
        )
        return train_loader, valid_loader, test_loader

class KMCL(nn.Module):
    def __init__(self, n_users, n_items, n_entities, n_rels, emb_dim, adj_norm, n_layers=2):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_entities = n_entities
        self.n_rels = n_rels
        self.n_nodes = n_users + n_items + n_entities
        self.emb_dim = emb_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(self.n_nodes, emb_dim)
        self.relation_emb = nn.Embedding(n_rels if n_rels > 0 else 1, emb_dim)
        nn.init.xavier_normal_(self.embedding.weight, gain=1.0)
        nn.init.xavier_normal_(self.relation_emb.weight, gain=1.0)

        self.gnn_layers = nn.ModuleList([nn.Linear(emb_dim, emb_dim) for _ in range(n_layers)])
        self.adj_norm = adj_norm
        self.to(DEVICE)

    def _edge_dropout(self, dropout_rate):
        adj_coo = self.adj_norm.coalesce()
        indices = adj_coo.indices().cpu().numpy()
        values = adj_coo.values().cpu().numpy()

        n_edges = len(values)
        keep_idx = np.random.choice(n_edges, size=int(n_edges * (1 - dropout_rate)), replace=False)

        new_indices = torch.LongTensor(indices[:, keep_idx]).to(DEVICE)
        new_values = torch.FloatTensor(values[keep_idx]).to(DEVICE)
        new_adj = torch.sparse_coo_tensor(
            new_indices, new_values, adj_coo.shape, dtype=torch.float32
        ).coalesce()
        return new_adj

    def forward(self, adj=None):
        adj = self.adj_norm if adj is None else adj
        ego_emb = self.embedding.weight
        all_emb = [ego_emb]

        for layer in self.gnn_layers:
            side_emb = torch.sparse.mm(adj, ego_emb)
            ego_emb = layer(side_emb)
            ego_emb = F.leaky_relu(ego_emb, inplace=True)
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

    @torch.no_grad()
    def predict(self, users, items):
        """预测指定用户对指定物品的评分（与KGCL一致接口）"""
        user_emb, item_emb = self.forward()
        u_emb = user_emb[users]
        i_emb = item_emb[items]
        logits = (u_emb * i_emb).sum(dim=-1)
        return logits

    @torch.no_grad()
    def predict_batch(self, users):
        """批量预测用户对所有物品的评分"""
        user_emb, item_emb = self.forward()
        u_emb = user_emb[users]
        scores = torch.matmul(u_emb, item_emb.t())
        return scores

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

        sim_matrix = torch.matmul(emb1, emb2.t()) / temperature
        batch_size = emb1.shape[0]
        labels = torch.arange(batch_size, device=DEVICE)

        loss = (F.cross_entropy(sim_matrix, labels) + F.cross_entropy(sim_matrix.t(), labels)) / 2
        return loss

    def kg_loss(self):
        kg = self.dataset.kg
        if len(kg) == 0:
            return torch.tensor(0.0, device=DEVICE)

        sample_size = min(512, len(kg))
        indices = np.random.choice(len(kg), sample_size, replace=False)
        h = torch.tensor(kg[indices, 0] + self.dataset.num_users, device=DEVICE, dtype=torch.long)
        r = torch.tensor(kg[indices, 1], device=DEVICE, dtype=torch.long)
        t = torch.tensor(kg[indices, 2] + self.dataset.num_users, device=DEVICE, dtype=torch.long)

        neg_t = torch.randint(
            self.dataset.num_users, self.dataset.num_users + self.dataset.num_entities,
            (sample_size,), device=DEVICE
        )

        h_emb = self.model.embedding(h)
        r_emb = self.model.relation_emb(r)
        t_emb = self.model.embedding(t)
        neg_t_emb = self.model.embedding(neg_t)

        pos_score = torch.norm(h_emb + r_emb - t_emb, p=2, dim=1)
        neg_score = torch.norm(h_emb + r_emb - neg_t_emb, p=2, dim=1)
        loss = -torch.log(torch.sigmoid(neg_score - pos_score) + 1e-10).mean()
        return loss

    def run_epoch(self, loader, train=True):
        self.model.train() if train else self.model.eval()
        total_loss = 0.0
        total_bpr = 0.0
        total_kg = 0.0
        total_contrast = 0.0
        n_samples = 0

        context = torch.no_grad() if not train else torch.enable_grad()
        with context:
            for (u, i_pos, i_neg), _ in loader:
                u = u.to(DEVICE, non_blocking=True)
                i_pos = i_pos.to(DEVICE, non_blocking=True)
                i_neg = i_neg.squeeze(1).to(DEVICE, non_blocking=True)

                (user_emb1, item_emb1), (user_emb2, item_emb2) = self.model.forward_multi_view()

                u_emb = user_emb1[u]
                pos_scores = (u_emb * item_emb1[i_pos]).sum(dim=-1)
                neg_scores = (u_emb * item_emb1[i_neg]).sum(dim=-1)
                bpr_loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10).mean()

                contrast_loss = (self.contrastive_loss(user_emb1, user_emb2) +
                                 self.contrastive_loss(item_emb1, item_emb2)) / 2

                kg_loss_val = self.kg_loss() if train else torch.tensor(0.0, device=DEVICE)

                loss = bpr_loss + KG_WEIGHT * kg_loss_val + CONTRAST_WEIGHT * contrast_loss

                if train:
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()

                batch_size = u.shape[0]
                total_loss += loss.item() * batch_size
                total_bpr += bpr_loss.item() * batch_size
                total_kg += kg_loss_val.item() * batch_size
                total_contrast += contrast_loss.item() * batch_size
                n_samples += batch_size

        avg_loss = total_loss / n_samples if n_samples > 0 else 0.0
        avg_bpr = total_bpr / n_samples if n_samples > 0 else 0.0
        avg_kg = total_kg / n_samples if n_samples > 0 else 0.0
        avg_contrast = total_contrast / n_samples if n_samples > 0 else 0.0
        return avg_loss, avg_bpr, avg_kg, avg_contrast

    def train(self, epochs=EPOCHS):
        best_val = float('inf')
        patience = 10
        patience_counter = 0

        for epoch in range(epochs):
            tr_loss, tr_bpr, tr_kg, tr_contrast = self.run_epoch(self.train_loader, True)
            val_loss, val_bpr, val_kg, val_contrast = self.run_epoch(self.valid_loader, False)

            self.scheduler.step(val_loss)

            print(
                f"Epoch {epoch + 1:02d} | loss={tr_loss:.4f} (bpr={tr_bpr:.4f}, kg={tr_kg:.4f}, contrast={tr_contrast:.4f}) | "
                f"val={val_loss:.4f}")

            if not math.isinf(val_loss) and val_loss < best_val:
                best_val = val_loss
                torch.save(self.model.state_dict(), 'kmcl_best.pt')
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

class Evaluator:
    def __init__(self, model, dataset):
        self.model = model.eval()
        self.dataset = dataset
        self.n_users = dataset.num_users
        self.n_items = dataset.num_items
        self.K = K
        self.all_items = torch.arange(self.n_items, device=DEVICE)

    def _recall_ndcg_mrr(self, df):
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
            if hit.any():
                rr = 1.0 / (np.where(hit)[0][0] + 1)
            else:
                rr = 0.0

            recalls.append(recall)
            ndcgs.append(ndcg)
            mrrs.append(rr)

        return np.mean(recalls), np.mean(ndcgs), np.mean(mrrs)

    def _coverage(self):
        scores = []
        for u in range(self.n_users):
            s = self.model.predict(torch.tensor([u] * self.n_items, device=DEVICE), self.all_items)
            _, topk = torch.topk(s, k=self.K)
            scores.extend(topk.cpu().numpy())
        return len(set(scores)) / self.n_items

    def evaluate(self, df):
        """评估主函数（返回与KGCL一致的指标字典）"""
        recall, ndcg, mrr = self._recall_ndcg_mrr(df)
        coverage = self._coverage()
        return dict(recall=recall, ndcg=ndcg, mrr=mrr, coverage=coverage)


