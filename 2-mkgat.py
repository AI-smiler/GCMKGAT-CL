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
import random


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EMB_DIM = 32 
VISUAL_DIM = 2048  
TEXT_DIM = 300 
BATCH_SIZE = 2048
LR = 0.001
EPOCHS = 30
K = 10
KG_FILE = 'kg_final.txt'
ML1M_DIR = "data"
L2_REG = 1e-5
N_LAYERS = 2  
KG_WEIGHT = 0.25 


# ---------- 工具 ----------
def set_seed(seed=2024):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _L2_loss_mean(x):
    return torch.mean(torch.sum(torch.pow(x, 2), dim=1, keepdim=False) / 2.)


# ---------- 多模态实体编码器 (论文4.1.1) ----------
class MultimodalEntityEncoder(nn.Module):
    def __init__(self, emb_dim=EMB_DIM, visual_dim=VISUAL_DIM, text_dim=TEXT_DIM):
        super().__init__()
        self.emb_dim = emb_dim

        self.visual_projector = nn.Sequential(
            nn.Linear(visual_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, emb_dim)
        )

        self.text_projector = nn.Sequential(
            nn.Linear(text_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, emb_dim)
        )

        self.dense_transform = nn.Linear(emb_dim, emb_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, entity_emb, visual_feat=None, text_feat=None, modality='struct'):
        if modality == 'visual' and visual_feat is not None:
            encoded = self.visual_projector(visual_feat)
        elif modality == 'text' and text_feat is not None:
            encoded = self.text_projector(text_feat)
        else:
            encoded = entity_emb

        output = self.dense_transform(encoded)
        return output


class MKGATLayer(nn.Module):
    def __init__(self, emb_dim, num_relations):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_relations = num_relations

        self.relation_emb = nn.Embedding(num_relations, emb_dim)
        nn.init.xavier_normal_(self.relation_emb.weight)

        self.W1 = nn.Linear(emb_dim * 3, emb_dim) 
        self.W2 = nn.Linear(emb_dim, 1)  
        self.leaky_relu = nn.LeakyReLU(0.2)

        self.W_add = nn.Linear(emb_dim, emb_dim) 
        self.W_concat = nn.Linear(emb_dim * 2, emb_dim)  

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_normal_(self.W1.weight)
        nn.init.xavier_normal_(self.W2.weight)
        nn.init.xavier_normal_(self.W_add.weight)
        nn.init.xavier_normal_(self.W_concat.weight)

    def propagation(self, ego_emb, neighbor_emb, relation_ids):
        rel_emb = self.relation_emb(relation_ids)  # (num_edges, emb_dim)
        triplet_emb = torch.cat([ego_emb, rel_emb, neighbor_emb], dim=-1)
        triplet_transformed = self.W1(triplet_emb)
        attn_scores = self.leaky_relu(self.W2(triplet_transformed))  # (num_edges, 1)

        return triplet_transformed, attn_scores

    def aggregate(self, ego_emb, agg_emb, method='add'):
        if method == 'add':
            transformed_ego = self.W_add(ego_emb)
            output = transformed_ego + agg_emb
        else:  
            concat_vec = torch.cat([ego_emb, agg_emb], dim=-1)
            output = self.W_concat(concat_vec)

        return output

class MKGAT(nn.Module):
    def __init__(self, n_users, n_items, n_entities, n_rels, emb_dim,
                 norm_adj, n_layers=2, visual_dim=VISUAL_DIM, text_dim=TEXT_DIM):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_entities = n_entities
        self.n_nodes = n_users + n_items + n_entities
        self.emb_dim = emb_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(self.n_nodes, emb_dim)
        nn.init.xavier_normal_(self.embedding.weight)

        self.mm_encoder = MultimodalEntityEncoder(emb_dim, visual_dim, text_dim)

        self.kgat_layers = nn.ModuleList([
            MKGATLayer(emb_dim, n_rels) for _ in range(n_layers)
        ])

        self.kg_relation_emb = nn.Embedding(n_rels, emb_dim)
        nn.init.xavier_normal_(self.kg_relation_emb.weight)

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

    def forward_kg_embedding(self, visual_features=None, text_features=None):
        ego_emb = self.embedding.weight

        if visual_features is not None or text_features is not None:
            item_ids = torch.arange(self.n_items, device=DEVICE)
            base_item_emb = self.embedding.weight[self.n_users:self.n_users + self.n_items]

            if visual_features is not None:
                visual_emb = self.mm_encoder(base_item_emb, visual_feat=visual_features, modality='visual')
            else:
                visual_emb = base_item_emb

            if text_features is not None:
                text_emb = self.mm_encoder(base_item_emb, text_feat=text_features, modality='text')
            else:
                text_emb = base_item_emb

            fused_item_emb = (visual_emb + text_emb) / 2.0
            ego_emb = ego_emb.clone()
            ego_emb[self.n_users:self.n_users + self.n_items] = fused_item_emb

        all_embs = [ego_emb]
        current_emb = ego_emb

        for layer in self.kgat_layers:

            neighbor_emb = torch.sparse.mm(self.adj_norm, current_emb)

            aggregated = neighbor_emb

            new_emb = layer.aggregate(current_emb, aggregated, method='concat')
            new_emb = F.leaky_relu(new_emb)
            new_emb = F.dropout(new_emb, p=0.1, training=self.training)

            current_emb = new_emb
            all_embs.append(current_emb)

        final_emb = torch.cat(all_embs, dim=-1)  # (n_nodes, emb_dim * (n_layers + 1))

        return final_emb

    def forward_recommendation(self, visual_features=None, text_features=None):
        return self.forward_kg_embedding(visual_features, text_features)

    def calculate_kg_loss(self, h, r, t, neg_t):
        h_emb = self.embedding(h)
        r_emb = self.kg_relation_emb(r)
        t_emb = self.embedding(t)
        neg_t_emb = self.embedding(neg_t)

        pos_score = torch.norm(h_emb + r_emb - t_emb, p=2, dim=1)

        neg_score = torch.norm(h_emb + r_emb - neg_t_emb, p=2, dim=1)

        loss = -torch.log(torch.sigmoid(neg_score - pos_score) + 1e-10).mean()
        return loss

    def predict(self, users, items, visual_features=None, text_features=None):
        all_emb = self.forward_recommendation(visual_features, text_features)

        user_emb = all_emb[:self.n_users]
        item_emb = all_emb[self.n_users:self.n_users + self.n_items]

        u_emb = user_emb[users]  # (batch, emb_dim * (n_layers+1))
        i_emb = item_emb[items]

        scores = (u_emb * i_emb).sum(dim=-1)
        return scores


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


class MKGATDataset:

    def __init__(self):
        ratings_path = os.path.join(ML1M_DIR, "ratings.dat")

        if not os.path.exists(ratings_path):
            print(f"Warning: {ratings_path} not found. Using synthetic data for testing.")
            self._create_synthetic_data()
            return

        ratings = pd.read_csv(ratings_path,
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
        train_end = int(0.8 * len(ratings))
        valid_end = int(0.9 * len(ratings))
        self.train_df = ratings.iloc[:train_end]
        self.valid_df = ratings.iloc[train_end:valid_end]
        self.test_df = ratings.iloc[valid_end:]

        kg_path = KG_FILE
        if not os.path.exists(kg_path):
            print(f"Warning: {kg_path} not found. Using synthetic KG data.")
            self._create_synthetic_kg()
        else:
            kg = pd.read_csv(kg_path, sep='\t', header=None, names=['h', 'r', 't'])
            print(f"Loaded KG: {len(kg)} triples before filtering")
            kg = kg[kg['h'].isin(self.item_map) & kg['t'].isin(self.item_map)]
            print(f"Filtered KG: {len(kg)} triples after filtering")
            if len(kg) == 0:
                print("Warning: KG is empty after filtering. Using synthetic KG data.")
                self._create_synthetic_kg()
            else:
                self._process_kg(kg)

        self._load_multimodal_features()

        self._build_adj()

    def _create_synthetic_data(self, n_users=1000, n_items=500, n_interactions=50000):
        np.random.seed(42)

        self.num_users = n_users
        self.num_items = n_items
        self.user_map = {i: i for i in range(n_users)}
        self.item_map = {i: i for i in range(n_items)}

        users = np.random.randint(0, n_users, n_interactions)
        items = np.random.randint(0, n_items, n_interactions)
        ratings = np.ones(n_interactions)
        timestamps = np.arange(n_interactions)

        ratings_df = pd.DataFrame({
            'uid': users.astype(np.int32),
            'iid': items.astype(np.int32),
            'rating': ratings.astype(np.float32),
            'ts': timestamps
        })

        ratings_df = ratings_df.sort_values('ts')
        train_end = int(0.8 * len(ratings_df))
        valid_end = int(0.9 * len(ratings_df))
        self.train_df = ratings_df.iloc[:train_end]
        self.valid_df = ratings_df.iloc[train_end:valid_end]
        self.test_df = ratings_df.iloc[valid_end:]

        print(f"Created synthetic data: {n_users} users, {n_items} items, {n_interactions} interactions")
        self._create_synthetic_kg()

    def _create_synthetic_kg(self, n_entities=500, n_relations=10):
        """创建合成知识图谱"""
        np.random.seed(42)

        n_item_entities = min(self.num_items, 300)
        n_other_entities = n_entities

        self.entities = set(range(n_item_entities + n_other_entities))
        self.rels = set(range(n_relations))
        self.ent_map = {e: i + self.num_items for i, e in
                        enumerate(range(n_item_entities, n_item_entities + n_other_entities))}
        self.rel_map = {r: i for i, r in enumerate(range(n_relations))}

        n_triples = 10000
        heads = np.random.randint(0, n_item_entities, n_triples)
        relations = np.random.randint(0, n_relations, n_triples)
        tails = np.random.randint(0, n_item_entities + n_other_entities, n_triples)

        self.kg = pd.DataFrame({
            'h': heads.astype(np.int32),
            'r': relations.astype(np.int32),
            't': tails.astype(np.int32)
        })

        self.num_entities = len(self.ent_map)
        self.num_rels = len(self.rel_map)

        print(f"Created  KG: {self.num_entities} entities, {self.num_rels} relations, {n_triples} triples")

        self._create_synthetic_multimodal()
        self._build_adj()

    def _process_kg(self, kg):
        """处理知识图谱数据"""
        self.entities = set(kg['h']).union(set(kg['t']))
        self.rels = set(kg['r'])
        self.ent_map = {e: i + self.num_items for i, e in enumerate(self.entities)}
        self.rel_map = {r: i for i, r in enumerate(self.rels)}
        kg['h'] = kg['h'].map(self.item_map)
        kg['t'] = kg['t'].map(lambda x: self.ent_map[x] if x in self.ent_map else x)
        kg['r'] = kg['r'].map(self.rel_map)
        self.kg = kg.dropna()
        self.kg['h'] = self.kg['h'].astype(np.int32)
        self.kg['t'] = self.kg['t'].astype(np.int32)
        self.kg['r'] = self.kg['r'].astype(np.int32)
        self.num_entities = len(self.ent_map)
        self.num_rels = len(self.rel_map)

    def _load_multimodal_features(self):
        """加载多模态特征 (论文5.1.1)"""
        # 尝试从文件加载，否则使用随机特征模拟ResNet和Word2Vec特征
        visual_path = os.path.join(ML1M_DIR, "visual_features.npy")
        text_path = os.path.join(ML1M_DIR, "text_features.npy")

        if os.path.exists(visual_path) and os.path.exists(text_path):
            self.visual_features = torch.FloatTensor(np.load(visual_path)).to(DEVICE)
            self.text_features = torch.FloatTensor(np.load(text_path)).to(DEVICE)
            print(f"Loaded multimodal features: visual {self.visual_features.shape}, text {self.text_features.shape}")
        else:
            self._create_synthetic_multimodal()

    def _create_synthetic_multimodal(self):
        """创建合成多模态特征 (模拟ResNet2048和Word2Vec300)"""
        np.random.seed(42)
        # 模拟ResNet50输出的2048维特征
        self.visual_features = torch.randn(self.num_items, VISUAL_DIM, device=DEVICE) * 0.1
        # 模拟Word2Vec+SIF的300维特征
        self.text_features = torch.randn(self.num_items, TEXT_DIM, device=DEVICE) * 0.1
        print(f"Created synthetic multimodal features (ResNet50+Word2Vec): "
              f"visual {self.visual_features.shape}, text {self.text_features.shape}")

    def _build_adj(self):
        """构建协作知识图谱(CKG)的邻接矩阵 (论文Definition 3)"""
        n_all = self.num_users + self.num_items + self.num_entities
        edges = []

        # User-item交互边 (Interact关系)
        for _, row in self.train_df.iterrows():
            u = row['uid']
            i = row['iid'] + self.num_users
            edges.append([u, i])
            edges.append([i, u])  # 无向图

        # Item-entity KG边
        for _, row in self.kg.iterrows():
            h = row['h'] + self.num_users
            t = row['t'] + self.num_users
            edges.append([h, t])
            edges.append([t, h])

        edges = np.array(edges).T
        values = np.ones(edges.shape[1])
        self.adj_mat = sp.coo_matrix((values, edges), shape=(n_all, n_all))

        # 归一化邻接矩阵 (对称归一化)
        rowsum = np.array(self.adj_mat.sum(1)).flatten()
        rowsum[rowsum == 0] = 1
        d_inv_sqrt = np.power(rowsum, -0.5)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        self.norm_adj = d_mat_inv_sqrt.dot(self.adj_mat).dot(d_mat_inv_sqrt).tocsr()

    def get_loaders(self):
        train_set = ML1MDataset(self.train_df, self.num_items, negative_samples=1)
        valid_set = ML1MDataset(self.valid_df, self.num_items, negative_samples=1)
        test_set = ML1MDataset(self.test_df, self.num_items, negative_samples=1)

        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        return train_loader, valid_loader, test_loader

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
            self.optimizer, mode='max', factor=0.5, patience=5
        )

    def kg_loss(self):
        """计算KG损失 (论文4.1.3)"""
        kg = self.dataset.kg
        if len(kg) == 0:
            return torch.tensor(0.0, device=DEVICE)

        # 采样KG三元组
        sample_size = min(1024, len(kg))
        indices = np.random.choice(len(kg), sample_size, replace=False)

        h = torch.tensor([kg.iloc[i]['h'] + self.dataset.num_users for i in indices], device=DEVICE)
        r = torch.tensor([kg.iloc[i]['r'] for i in indices], device=DEVICE)
        t = torch.tensor([kg.iloc[i]['t'] + self.dataset.num_users for i in indices], device=DEVICE)

        # 负采样: 随机替换尾实体
        neg_t = torch.randint(0, self.dataset.num_entities, (sample_size,), device=DEVICE)
        neg_t = neg_t + self.dataset.num_users

        loss = self.model.calculate_kg_loss(h, r, t, neg_t)
        return loss

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

            # BPR损失 (论文公式11)
            pos_scores = self.model.predict(u, i_pos,
                                            self.dataset.visual_features,
                                            self.dataset.text_features)
            neg_scores = self.model.predict(u, i_neg,
                                            self.dataset.visual_features,
                                            self.dataset.text_features)
            bpr_loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10).mean()

            # L2正则化
            l2_reg = 0
            for param in self.model.parameters():
                l2_reg += torch.norm(param, p=2) ** 2
            l2_loss = L2_REG * l2_reg

            # KG损失 (交替训练)
            kg_loss_val = self.kg_loss() if train else torch.tensor(0.0, device=DEVICE)

            # 总损失 (论文: 交替优化，这里使用联合损失)
            loss = bpr_loss + l2_loss + KG_WEIGHT * kg_loss_val

            if torch.isinf(loss) or torch.isnan(loss):
                continue

            if train:
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item() * len(u)

        return total_loss / len(loader.dataset)

    def train(self, epochs=EPOCHS):
        best_ndcg = 0
        patience = 10
        patience_counter = 0

        for epoch in range(epochs):
            tr_loss = self.run_epoch(self.train_loader, True)

            # 验证
            evaluator = Evaluator(self.model, self.dataset)
            val_metrics = evaluator.evaluate(self.dataset.valid_df)
            val_ndcg = val_metrics['ndcg']

            self.scheduler.step(val_ndcg)

            print(f"Epoch {epoch + 1:02d} | train_loss={tr_loss:.4f} | "
                  f"val_recall@{K}={val_metrics['recall']:.4f} | "
                  f"val_ndcg@{K}={val_ndcg:.4f}")

            if val_ndcg > best_ndcg:
                best_ndcg = val_ndcg
                torch.save(self.model.state_dict(), 'mkgat_best.pt')
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
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
        """计算Recall@K和NDCG@K (论文5.1.2)"""
        user_pos = df.groupby('uid')['iid'].apply(list).to_dict()
        recalls, ndcgs, mrrs = [], [], []

        for u in tqdm(range(self.n_users), desc="Evaluating"):
            seen = set(self.dataset.train_df[self.dataset.train_df['uid'] == u]['iid'].tolist())
            pos = user_pos.get(u, [])
            if not pos:
                continue

            # 预测所有物品分数
            scores = self.model.predict(
                torch.tensor([u] * self.n_items, device=DEVICE),
                self.all_items,
                self.dataset.visual_features,
                self.dataset.text_features
            )
            scores[list(seen)] = -float('inf')  # 排除已交互物品

            _, topk = torch.topk(scores, k=self.K)
            topk = topk.cpu().numpy()
            hit = np.isin(topk, pos)

            # Recall@K
            recall = hit.sum() / len(pos)
            recalls.append(recall)

            # NDCG@K
            dcg = (hit / np.log2(np.arange(2, self.K + 2))).sum()
            idcg = (1.0 / np.log2(np.arange(2, min(len(pos), self.K) + 2))).sum()
            ndcg = dcg / (idcg + 1e-9)
            ndcgs.append(ndcg)

            # MRR
            rr = 1.0 / (np.where(hit)[0][0] + 1) if hit.any() else 0.0
            mrrs.append(rr)

        return np.mean(recalls), np.mean(ndcgs), np.mean(mrrs)

    def _coverage(self):
        """计算覆盖率"""
        scores = []
        for u in range(self.n_users):
            s = self.model.predict(
                torch.tensor([u] * self.n_items, device=DEVICE),
                self.all_items,
                self.dataset.visual_features,
                self.dataset.text_features
            )
            _, topk = torch.topk(s, k=self.K)
            scores.extend(topk.cpu().numpy())
        return len(set(scores)) / self.n_items

    def evaluate(self, df):
        recall, ndcg, mrr = self._recall_ndcg(df)
        coverage = self._coverage()
        return dict(recall=recall, ndcg=ndcg, mrr=mrr, coverage=coverage)
