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
import warnings

warnings.filterwarnings('ignore')


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EMB_DIM = 32  # 降维提速
REL_DIM = 32
BATCH_SIZE = 2048
LR = 0.001
EPOCHS = 30 
K = 20
KG_FILE = 'kg_final.txt'
ML1M_DIR = "data"
L2_REG = 1e-5
N_LAYERS = 2 
KG_WEIGHT = 0.25
AGGREGATOR = "gcn"  # GCN比bi-interaction快
DROPOUT_RATE = 0.1
KG_ENABLE = True



def set_seed(seed=2024):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) if torch.cuda.is_available() else None


def check_file_exists(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在：{file_path}\n请检查路径是否正确！")


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
        self.user_items_np = {u: np.array(list(v)) for u, v in self.user_items.items()}

    def __len__(self):
        return len(self.u)

    def __getitem__(self, idx):
        u_pos = torch.tensor(self.u[idx], dtype=torch.long)
        i_pos = torch.tensor(self.i[idx], dtype=torch.long)
        user = self.u[idx]

        interacted = self.user_items_np.get(user, np.array([]))
        candidate_neg = np.random.randint(0, self.num_items, size=self.negative_samples * 2)
        mask = ~np.isin(candidate_neg, interacted)
        i_neg = candidate_neg[mask][:self.negative_samples]
        if len(i_neg) < self.negative_samples:
            padding = np.random.randint(0, self.num_items, size=self.negative_samples - len(i_neg))
            i_neg = np.concatenate([i_neg, padding])

        i_neg = torch.tensor(i_neg, dtype=torch.long)
        return (u_pos, i_pos, i_neg), (torch.tensor(1.0), torch.tensor(0.0))


class KGATDataset:
    def __init__(self):
        ratings_path = os.path.join(ML1M_DIR, "ratings.dat")
        check_file_exists(ratings_path)
        ratings = pd.read_csv(ratings_path, sep='::', engine='python', header=None,
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

        self.num_entities = 0
        self.num_rels = 0
        self.kg = np.array([])
        global KG_ENABLE
        if os.path.exists(KG_FILE) and os.path.getsize(KG_FILE) > 0:
            try:
                kg = pd.read_csv(KG_FILE, sep='\t', header=None, names=['h', 'r', 't'])
                kg = kg.dropna(subset=['h', 'r', 't'])
                self.entities = set(kg['h'].unique()).union(set(kg['t'].unique()))
                self.rels = set(kg['r'].unique())
                self.ent_map = {e: i + self.num_items for i, e in enumerate(self.entities)}
                self.rel_map = {r: i for i, r in enumerate(self.rels)}
                kg['h'] = kg['h'].map(self.ent_map)
                kg['t'] = kg['t'].map(self.ent_map)
                kg['r'] = kg['r'].map(self.rel_map)
                kg = kg.dropna(subset=['h', 'r', 't'])
                self.kg = kg.values
                self.num_entities = len(self.ent_map)
                self.num_rels = len(self.rel_map)
                print(f"KG加载成功：三元组{len(self.kg)} | 实体{self.num_entities} | 关系{self.num_rels}")
            except Exception as e:
                print(f"Warning：{e}")
                KG_ENABLE = False
        else:
            print(f"KG File is not exists")
            KG_ENABLE = False

        self._build_adj()

    def _build_adj(self):
        n_all = self.num_users + self.num_items + max(self.num_entities, 1)
        edges = []
        for _, row in self.train_df.iterrows():
            u = row['uid']
            i = row['iid'] + self.num_users
            edges.append([u, i])
            edges.append([i, u])
        if KG_ENABLE and len(self.kg) > 0:
            for h, r, t in self.kg:
                edges.append([h, t])
                edges.append([t, h])
        if len(edges) == 0:
            edges = [[0, 0]]
        edges = np.array(edges).T
        values = np.ones(edges.shape[1])
        self.adj_mat = sp.coo_matrix((values, edges), shape=(n_all, n_all))
        rowsum = np.array(self.adj_mat.sum(1)).flatten()
        rowsum[rowsum == 0] = 1e-8
        d_inv_sqrt = np.power(rowsum, -0.5)
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        self.norm_adj = d_mat_inv_sqrt.dot(self.adj_mat).dot(d_mat_inv_sqrt).tocsr()

    def get_loaders(self):
        train_set = ML1MDataset(self.train_df, self.num_items, negative_samples=1)
        valid_set = ML1MDataset(self.valid_df, self.num_items, negative_samples=1)
        test_set = ML1MDataset(self.test_df, self.num_items, negative_samples=1)
        train_loader = DataLoader(
            train_set, batch_size=BATCH_SIZE, shuffle=True,
            num_workers=4, pin_memory=True, drop_last=True
        )
        valid_loader = DataLoader(
            valid_set, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=4, pin_memory=True
        )
        test_loader = DataLoader(
            test_set, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=4, pin_memory=True
        )
        return train_loader, valid_loader, test_loader

class KGAT(nn.Module):
    def __init__(self, n_users, n_items, n_entities, n_rels, emb_dim, rel_dim,
                 norm_adj, n_layers=2, aggregator="gcn", dropout=0.1):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_entities = max(n_entities, 1)
        self.n_rels = max(n_rels, 1)
        self.n_nodes = n_users + n_items + self.n_entities
        self.emb_dim = emb_dim
        self.rel_dim = rel_dim
        self.n_layers = n_layers
        self.aggregator = aggregator
        self.dropout = nn.Dropout(dropout)

        self.ent_emb = nn.Embedding(self.n_nodes, emb_dim, dtype=torch.float32)
        self.rel_emb = nn.Embedding(self.n_rels, rel_dim, dtype=torch.float32)
        self.rel_proj = nn.Embedding(self.n_rels, emb_dim * rel_dim, dtype=torch.float32)

        self.tanh = nn.Tanh()
        self.leaky_relu = nn.LeakyReLU(0.2)

        if self.aggregator in ["gcn", "bi-interaction"]:
            self.W1 = nn.Linear(emb_dim, emb_dim, bias=False, dtype=torch.float32)
        if self.aggregator in ["graphsage", "bi-interaction"]:
            self.W2 = nn.Linear(2 * emb_dim, emb_dim, bias=False, dtype=torch.float32)
        if self.aggregator == "bi-interaction":
            self.W3 = nn.Linear(emb_dim, emb_dim, bias=False, dtype=torch.float32)

        nn.init.xavier_normal_(self.ent_emb.weight, gain=1.0)
        nn.init.xavier_normal_(self.rel_emb.weight, gain=1.0)
        nn.init.xavier_normal_(self.rel_proj.weight, gain=1.0)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=1.0)

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
        self.adj_sparse = torch.sparse_coo_tensor(i, v, shape, dtype=torch.float32).coalesce()

    def _transr_proj(self, h_emb, r_id):
        batch_size = h_emb.shape[0]
        w_r = self.rel_proj(r_id).reshape(batch_size, self.emb_dim, self.rel_dim)
        h_proj = torch.bmm(h_emb.unsqueeze(1), w_r).squeeze(1)
        return h_proj

    def knowledge_attention(self, h_emb, t_emb, r_id):
        if not KG_ENABLE:
            return torch.ones(h_emb.shape[0], 1, dtype=torch.float32).to(DEVICE)
        h_proj = self._transr_proj(h_emb, r_id)
        t_proj = self._transr_proj(t_emb, r_id)
        r_emb = self.rel_emb(r_id)
        attn_score = torch.sum(t_proj * self.tanh(h_proj + r_emb), dim=1, keepdim=True)
        attn_w = F.softmax(attn_score, dim=0)
        return attn_w

    def aggregate(self, ego_emb, neigh_emb):
        if self.aggregator == "gcn":
            agg_emb = self.leaky_relu(self.W1(ego_emb + neigh_emb))
        elif self.aggregator == "graphsage":
            concat_emb = torch.cat([ego_emb, neigh_emb], dim=1)
            agg_emb = self.leaky_relu(self.W2(concat_emb))
        elif self.aggregator == "bi-interaction":
            sum_emb = self.leaky_relu(self.W1(ego_emb + neigh_emb))
            mul_emb = self.leaky_relu(self.W3(ego_emb * neigh_emb))
            agg_emb = sum_emb + mul_emb
        else:
            raise ValueError(f"{self.aggregator}")
        return self.dropout(agg_emb)

    def forward(self):
        ego_emb = self.ent_emb.weight  # 已经是float32
        all_emb = [ego_emb]

        r_id = torch.zeros(self.n_nodes, dtype=torch.long, device=DEVICE)
        if self.n_rels > 0:
            r_id = r_id % self.n_rels  

        for _ in range(self.n_layers):
            neigh_emb = torch.sparse.mm(self.adj_sparse, ego_emb)
            attn_w = self.knowledge_attention(ego_emb, neigh_emb, r_id[:ego_emb.shape[0]])
            neigh_emb = neigh_emb * attn_w
            ego_emb = self.aggregate(ego_emb, neigh_emb)
            all_emb.append(ego_emb)

        final_emb = torch.cat(all_emb, dim=1)
        user_emb = final_emb[:self.n_users]
        item_emb = final_emb[self.n_users:self.n_users + self.n_items]
        del all_emb
        return user_emb, item_emb

    def transr_score(self, h_id, r_id, t_id):
        if not KG_ENABLE:
            return torch.zeros_like(h_id, dtype=torch.float32).to(DEVICE)
        h_emb = self.ent_emb(h_id)
        t_emb = self.ent_emb(t_id)
        r_emb = self.rel_emb(r_id)
        h_proj = self._transr_proj(h_emb, r_id)
        t_proj = self._transr_proj(t_emb, r_id)
        score = torch.norm(h_proj + r_emb - t_proj, p=2, dim=1)
        return score

    def predict(self, users, items):
        user_emb, item_emb = self.forward()
        u_emb = user_emb[users]
        i_emb = item_emb[items]
        logits = torch.sum(u_emb * i_emb, dim=-1)
        del u_emb, i_emb
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

    def kg_loss(self):
        if not KG_ENABLE or len(self.dataset.kg) == 0:
            return torch.tensor(0.0, dtype=torch.float32, device=DEVICE)
        sample_size = min(1024, len(self.dataset.kg))
        indices = np.random.choice(len(self.dataset.kg), sample_size, replace=False)
        kg_batch = self.dataset.kg[indices]
        h = torch.tensor(kg_batch[:, 0], device=DEVICE, dtype=torch.long)
        r = torch.tensor(kg_batch[:, 1], device=DEVICE, dtype=torch.long)
        t = torch.tensor(kg_batch[:, 2], device=DEVICE, dtype=torch.long)

        neg_t = torch.randint(
            self.dataset.num_users,
            self.dataset.num_users + self.dataset.num_entities,
            (sample_size,), device=DEVICE
        )

        pos_score = self.model.transr_score(h, r, t)
        neg_score = self.model.transr_score(h, r, neg_t)
        kg_loss = -torch.log(torch.sigmoid(neg_score - pos_score) + 1e-10).mean()
        return kg_loss

    def bpr_loss(self, u, i_pos, i_neg):
        pos_scores = self.model.predict(u, i_pos)
        neg_scores = self.model.predict(u, i_neg.squeeze(1))
        bpr_loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10).mean()
        return bpr_loss

    def run_epoch(self, loader, train=True):
        self.model.train() if train else self.model.eval()
        total_loss = 0.0
        batch_count = 0
        with torch.set_grad_enabled(train):
            for ((u, i_pos, i_neg), _) in loader:
                u = u.to(DEVICE, non_blocking=True)
                i_pos = i_pos.to(DEVICE, non_blocking=True)
                i_neg = i_neg.to(DEVICE, non_blocking=True)

                if train:
                    self.optimizer.zero_grad(set_to_none=True)
                    bpr_loss_val = self.bpr_loss(u, i_pos, i_neg)
                    kg_loss_val = self.kg_loss()
                    loss = bpr_loss_val + KG_WEIGHT * kg_loss_val

                    if not torch.isinf(loss) and not torch.isnan(loss):
                        loss.backward()
                        self.optimizer.step()
                        total_loss += loss.item() * len(u)
                else:
                    bpr_loss_val = self.bpr_loss(u, i_pos, i_neg)
                    loss = bpr_loss_val
                    if not torch.isinf(loss) and not torch.isnan(loss):
                        total_loss += loss.item() * len(u)

                batch_count += 1
                if batch_count % 500 == 0 and DEVICE.type == 'cuda':
                    torch.cuda.empty_cache()

        return total_loss / len(loader.dataset)

    def train(self, epochs=EPOCHS):
        best_val = float('inf')
        patience = 10
        patience_counter = 0
        for epoch in range(epochs):
            tr_loss = self.run_epoch(self.train_loader, True)
            val_loss = self.run_epoch(self.valid_loader, False)
            self.scheduler.step(val_loss)
            print(f"Epoch {epoch + 1:02d} | Train Loss: {tr_loss:.4f} | Val Loss: {val_loss:.4f}")

            if not math.isinf(val_loss) and val_loss < best_val:
                best_val = val_loss
                torch.save(self.model.state_dict(), 'kgat_best.pt')  # 统一模型保存名称
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break
            if DEVICE.type == 'cuda':
                torch.cuda.empty_cache()

class Evaluator:
    def __init__(self, model, dataset, K=20):
        self.model = model.eval()
        self.dataset = dataset
        self.n_users = dataset.num_users
        self.n_items = dataset.num_items
        self.K = K
        self.user_seen = {}
        for u in range(self.n_users):
            seen = self.dataset.train_df[self.dataset.train_df['uid'] == u]['iid'].tolist()
            self.user_seen[u] = np.array(seen, dtype=np.int32)

    def _recall_ndcg_mrr_batch(self, df, batch_size=256):
        user_pos = df.groupby('uid')['iid'].apply(list).to_dict()
        valid_users = [u for u in user_pos.keys() if len(user_pos[u]) > 0]
        if not valid_users:
            return 0.0, 0.0, 0.0

        recalls, ndcgs, mrrs = [], [], []
        with torch.no_grad():
            for idx in range(0, len(valid_users), batch_size):
                batch_users = valid_users[idx:idx + batch_size]
                batch_users_tensor = torch.tensor(batch_users, device=DEVICE)

                user_emb, item_emb = self.model.forward()
                batch_user_emb = user_emb[batch_users_tensor]
                scores = torch.matmul(batch_user_emb, item_emb.t())

                for i, u in enumerate(batch_users):
                    seen = self.user_seen.get(u, np.array([]))
                    if len(seen) > 0:
                        scores[i, seen] = -float('inf')

                _, topk_indices = torch.topk(scores, k=self.K, dim=1)
                topk_indices = topk_indices.cpu().numpy()

                for i, u in enumerate(batch_users):
                    pos = user_pos[u]
                    hit = np.isin(topk_indices[i], pos)
                    recall = hit.sum() / len(pos)
                    dcg = (hit / np.log2(np.arange(2, self.K + 2))).sum()
                    idcg = (1.0 / np.log2(np.arange(2, min(len(pos), self.K) + 2))).sum()
                    ndcg = dcg / (idcg + 1e-9)
                    mrr = 1.0 / (np.where(hit)[0][0] + 1) if hit.any() else 0.0

                    recalls.append(recall)
                    ndcgs.append(ndcg)
                    mrrs.append(mrr)

        return np.mean(recalls), np.mean(ndcgs), np.mean(mrrs)

    def _coverage(self, df=None):
        if df is not None:
            target_users = df['uid'].unique()
        else:
            target_users = range(self.n_users)

        with torch.no_grad():
            for idx in range(0, len(target_users), batch_size):
                batch_users = target_users[idx:idx + batch_size]
                batch_users_tensor = torch.tensor(batch_users, device=DEVICE)

                user_emb, item_emb = self.model.forward()
                batch_user_emb = user_emb[batch_users_tensor]
                scores = torch.matmul(batch_user_emb, item_emb.t())

                for i, u in enumerate(batch_users):
                    seen = self.user_seen.get(u, np.array([]))
                    if len(seen) > 0:
                        scores[i, seen] = -float('inf')

                _, topk = torch.topk(scores, k=self.K, dim=1)
                all_topk.extend(topk.cpu().numpy().flatten())

        return len(set(all_topk)) / self.n_items

    def evaluate(self, df):
        recall, ndcg, mrr = self._recall_ndcg_mrr_batch(df)
        coverage = self._coverage(df)  # 传入df，仅计算该批次用户的覆盖率
        return dict(recall=recall, ndcg=ndcg, mrr=mrr, coverage=coverage)

