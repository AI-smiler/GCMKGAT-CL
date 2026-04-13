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
from collections import defaultdict
import random
import warnings
import time
from tqdm import tqdm
import json

warnings.filterwarnings('ignore')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EMB_DIM = 32
BATCH_SIZE = 2048
LR = 0.001
EPOCHS = 30
K = 20
L2_REG = 1e-5
N_LAYERS = 2
KG_WEIGHT = 0.25
CONTRAST_WEIGHT = 0.05
MODAL_WEIGHT = 0.05
VIEW_CONTRAST_WEIGHT = 0.25
GATE_TEMPERATURE = 0.5
EDGE_DROPOUT_RATE = 0.1
KG_SAMPLE_SIZE = 256
BETA = 0.3
GAMMA = 0.2
TOP_K_NEIGHBORS = 16
CONTRAST_TEMP = 0.15
CONTRAST_SAMPLE_SIZE = 512
PATIENCE = 30
BPR_REG = 0.001

RATINGS_FILE = 'data/ratings.txt'
MOVIES_FILE = 'data/movies.txt'
USERS_FILE = 'data/users.txt'
KG_FILE = 'data/kg_final.txt'


def set_seed(seed=2024):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_embeddings(embeddings):
    return F.normalize(embeddings, p=2, dim=1)

class MultiHeadCrossModalAttention(nn.Module):
    def __init__(self, emb_dim, num_heads=2):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads
        self.q_proj = nn.Linear(emb_dim, emb_dim)
        self.k_proj = nn.Linear(emb_dim, emb_dim)
        self.v_proj = nn.Linear(emb_dim, emb_dim)
        self.out_proj = nn.Linear(emb_dim, emb_dim)

    def forward(self, visual_emb, text_emb, audio_emb=None):
        modalities = [visual_emb, text_emb]
        if audio_emb is not None:
            modalities.append(audio_emb)
        x = torch.stack(modalities, dim=1)
        batch_size, num_modalities, _ = x.shape
        q = self.q_proj(x).reshape(batch_size, num_modalities, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(batch_size, num_modalities, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(batch_size, num_modalities, self.num_heads, self.head_dim).transpose(1, 2)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, num_modalities, -1)
        output = self.out_proj(attn_output)
        return output.sum(dim=1)


class SparseKnowledgeAttention(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.emb_dim = emb_dim
        self.q_proj = nn.Linear(emb_dim, emb_dim)
        self.k_proj = nn.Linear(emb_dim, emb_dim)
        self.v_proj = nn.Linear(emb_dim, emb_dim)

    def forward(self, ego_emb, side_emb, rel_emb=None):
        q = self.q_proj(ego_emb)
        if rel_emb is not None:
            k = self.k_proj(side_emb * rel_emb)
        else:
            k = self.k_proj(side_emb)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.emb_dim)
        k_value = min(TOP_K_NEIGHBORS, attn_scores.shape[-1])
        topk_scores, topk_indices = torch.topk(attn_scores, k=k_value, dim=-1)
        attn_weights = F.softmax(topk_scores, dim=-1)
        v = self.v_proj(side_emb)
        n_nodes = q.shape[0]
        k_neighbors = topk_indices.shape[1]
        flat_indices = topk_indices.view(-1)
        topk_v = v[flat_indices].view(n_nodes, k_neighbors, self.emb_dim)
        aggregated = torch.sum(topk_v * attn_weights.unsqueeze(-1), dim=1)
        return aggregated


class StandardGNNLayer(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.emb_dim = emb_dim
        self.linear = nn.Linear(emb_dim, emb_dim)

    def forward(self, ego_emb, side_emb, rel_emb=None):
        aggregated = self.linear(side_emb)
        return aggregated

class GCMKGATCL_Ablation(nn.Module):
    def __init__(self, n_users, n_items, n_entities, n_rels, emb_dim,
                 norm_adj, movie_features, user_features, n_layers=2,
                 use_gated_module=True,          
                 use_cross_modal_attn=True,       
                 use_sparse_attention=True,       
                 use_contrastive_learning=True,   
                 use_hierarchical_sampling=True): 
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_entities = n_entities
        self.n_nodes = n_users + n_items + n_entities
        self.emb_dim = emb_dim
        self.n_layers = n_layers

        self.use_gated_module = use_gated_module
        self.use_cross_modal_attn = use_cross_modal_attn
        self.use_sparse_attention = use_sparse_attention
        self.use_contrastive_learning = use_contrastive_learning
        self.use_hierarchical_sampling = use_hierarchical_sampling

        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.item_emb = nn.Embedding(n_items, emb_dim)
        self.entity_emb = nn.Embedding(n_entities, emb_dim)
        self.relation_emb = nn.Embedding(n_rels, emb_dim)
        self.movie_encoder = nn.Sequential(
            nn.Linear(movie_features.shape[1], emb_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(emb_dim, emb_dim)
        )
        self.user_encoder = nn.Sequential(
            nn.Linear(user_features.shape[1], emb_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(emb_dim, emb_dim)
        )
        if self.use_cross_modal_attn:
            self.cross_modal_attn = MultiHeadCrossModalAttention(emb_dim)
        else:
            self.simple_mm_fusion = nn.Linear(emb_dim, emb_dim)

        if self.use_gated_module:
            self.movie_gate = nn.Sequential(nn.Linear(emb_dim * 2, emb_dim), nn.Sigmoid())
            self.user_gate = nn.Sequential(nn.Linear(emb_dim * 2, emb_dim), nn.Sigmoid())

        if self.use_sparse_attention:
            self.gnn_layers = nn.ModuleList([SparseKnowledgeAttention(emb_dim) for _ in range(n_layers)])
        else:
            self.gnn_layers = nn.ModuleList([StandardGNNLayer(emb_dim) for _ in range(n_layers)])
        
        self.layer_agg = nn.Linear(emb_dim * (n_layers + 1), emb_dim)

        if self.use_contrastive_learning:
            self.mm_proj = nn.Linear(emb_dim, emb_dim)
            self.id_proj = nn.Linear(emb_dim, emb_dim)
            self.view_proj = nn.Linear(emb_dim, emb_dim)

        self.register_buffer('movie_features', movie_features)
        self.register_buffer('user_features', user_features)

        adj_coo = norm_adj.tocoo()
        self.adj_norm = torch.sparse_coo_tensor(
            torch.from_numpy(np.vstack((adj_coo.row, adj_coo.col))).long().to(DEVICE),
            torch.from_numpy(adj_coo.data).float().to(DEVICE),
            torch.Size(adj_coo.shape), dtype=torch.float32
        ).coalesce()
        self.register_buffer('adj_row', torch.from_numpy(adj_coo.row).long().to(DEVICE))
        self.register_buffer('adj_col', torch.from_numpy(adj_coo.col).long().to(DEVICE))
        self.register_buffer('adj_data', torch.from_numpy(adj_coo.data).float().to(DEVICE))
        self.adj_shape = adj_coo.shape

        for m in [self.user_emb, self.item_emb, self.entity_emb, self.relation_emb]:
            nn.init.xavier_normal_(m.weight, gain=1.0)

        if self.use_contrastive_learning:
            for m in [self.mm_proj, self.id_proj, self.view_proj]:
                nn.init.xavier_normal_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        self.to(DEVICE)

    def _edge_dropout(self):
        n_edges = self.adj_row.shape[0]
        keep_idx = torch.randperm(n_edges, device=DEVICE)[:int(n_edges * (1 - EDGE_DROPOUT_RATE))]
        new_row = self.adj_row[keep_idx]
        new_col = self.adj_col[keep_idx]
        new_data = self.adj_data[keep_idx]
        temp_adj = sp.coo_matrix((new_data.cpu().numpy(), (new_row.cpu().numpy(), new_col.cpu().numpy())),
                                 shape=self.adj_shape)
        rowsum = np.array(temp_adj.sum(1)).flatten()
        rowsum[rowsum == 0] = 1
        d_inv_sqrt = np.power(rowsum, -0.5)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        norm_adj = sp.diags(d_inv_sqrt).dot(temp_adj).dot(sp.diags(d_inv_sqrt)).tocoo()
        return torch.sparse_coo_tensor(
            torch.from_numpy(np.vstack((norm_adj.row, norm_adj.col))).long().to(DEVICE),
            torch.from_numpy(norm_adj.data).float().to(DEVICE),
            torch.Size(norm_adj.shape), dtype=torch.float32
        ).coalesce()

    def fuse_embeddings(self):
        id_user_emb = self.user_emb.weight
        id_item_emb = self.item_emb.weight
        entity_emb = self.entity_emb.weight

        mm_item_emb = F.relu(self.movie_encoder(self.movie_features))
        mm_user_emb = F.relu(self.user_encoder(self.user_features))

        if self.use_cross_modal_attn:
            mm_item_emb = self.cross_modal_attn(mm_item_emb, mm_item_emb)
        else:
            mm_item_emb = F.relu(self.simple_mm_fusion(mm_item_emb))

        if self.use_gated_module:
            movie_gate_input = torch.cat([id_item_emb, mm_item_emb], dim=-1)
            movie_gate = torch.sigmoid(self.movie_gate(movie_gate_input) / GATE_TEMPERATURE)
            fused_item_emb = movie_gate * id_item_emb + (1 - movie_gate) * mm_item_emb + 0.1 * id_item_emb

            user_gate_input = torch.cat([id_user_emb, mm_user_emb], dim=-1)
            user_gate = torch.sigmoid(self.user_gate(user_gate_input) / GATE_TEMPERATURE)
            fused_user_emb = user_gate * id_user_emb + (1 - user_gate) * mm_user_emb + 0.1 * id_user_emb
        else:
            fused_item_emb = id_item_emb + mm_item_emb
            fused_user_emb = id_user_emb + mm_user_emb

        return fused_user_emb, fused_item_emb, entity_emb, id_item_emb, mm_item_emb

    def graph_convolution(self, user_emb, item_emb, entity_emb, adj=None):
        all_embeddings = torch.cat([user_emb, item_emb, entity_emb], dim=0)
        embeddings_list = [all_embeddings]
        adj = self.adj_norm if adj is None else adj
        for layer in self.gnn_layers:
            side_emb = torch.sparse.mm(adj, embeddings_list[-1])
            rel_emb_avg = self.relation_emb.weight.mean(dim=0).unsqueeze(0).repeat(side_emb.shape[0], 1)
            aggregated = layer(embeddings_list[-1], side_emb, rel_emb_avg)
            embeddings_list.append(F.relu(aggregated))
        stacked = torch.stack(embeddings_list, dim=1)
        final_emb = self.layer_agg(stacked.view(stacked.size(0), -1))
        return final_emb

    def forward(self, adj=None):
        fused_user_emb, fused_item_emb, entity_emb, id_item_emb, mm_item_emb = self.fuse_embeddings()
        final_emb = self.graph_convolution(fused_user_emb, fused_item_emb, entity_emb, adj)
        return (normalize_embeddings(final_emb[:self.n_users]),
                normalize_embeddings(final_emb[self.n_users:self.n_users + self.n_items]),
                id_item_emb, mm_item_emb)

    def forward_multi_view(self):
        emb1 = self.forward()
        adj_aug = self._edge_dropout()
        emb2 = self.forward(adj_aug)
        return (emb1[0], emb1[1]), (emb2[0], emb2[1])

    def compute_contrastive_loss(self, id_item_emb, mm_item_emb):
        if not self.use_contrastive_learning:
            return torch.tensor(0.0, device=DEVICE)
        
        n_items = min(CONTRAST_SAMPLE_SIZE, id_item_emb.size(0))
        indices = torch.randperm(id_item_emb.size(0))[:n_items]
        z_mm = normalize_embeddings(self.mm_proj(mm_item_emb[indices]))
        z_id = normalize_embeddings(self.id_proj(id_item_emb[indices]))

        similarity = torch.mm(z_mm, z_id.t()) / CONTRAST_TEMP
        similarity = similarity - similarity.max(dim=1, keepdim=True)[0].detach()
        labels = torch.arange(n_items, device=DEVICE)
        return (F.cross_entropy(similarity, labels) + F.cross_entropy(similarity.t(), labels)) / 2

    def compute_view_contrastive_loss(self, user_emb1, item_emb1, user_emb2, item_emb2):
        if not self.use_contrastive_learning:
            return torch.tensor(0.0, device=DEVICE)
        
        n_users = min(CONTRAST_SAMPLE_SIZE, user_emb1.size(0))
        idx = torch.randperm(user_emb1.size(0))[:n_users]
        z1 = normalize_embeddings(self.view_proj(user_emb1[idx]))
        z2 = normalize_embeddings(self.view_proj(user_emb2[idx]))
        sim = torch.mm(z1, z2.t()) / CONTRAST_TEMP
        labels = torch.arange(n_users, device=DEVICE)
        user_loss = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels)) / 2

        n_items = min(CONTRAST_SAMPLE_SIZE, item_emb1.size(0))
        idx = torch.randperm(item_emb1.size(0))[:n_items]
        z1 = normalize_embeddings(self.view_proj(item_emb1[idx]))
        z2 = normalize_embeddings(self.view_proj(item_emb2[idx]))
        sim = torch.mm(z1, z2.t()) / CONTRAST_TEMP
        labels = torch.arange(n_items, device=DEVICE)
        item_loss = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels)) / 2

        return (user_loss + item_loss) / 2

class NegativeSampler:
    def __init__(self, n_items, train_df, use_hierarchical=True, alpha=0.5):
        self.n_items = n_items
        self.use_hierarchical = use_hierarchical
        
        if self.use_hierarchical:
            item_counts = train_df['iid'].value_counts()
            item_pop = np.zeros(n_items, dtype=np.float32)
            item_pop[item_counts.index] = item_counts.values
            self.item_probs = np.power(item_pop + 1, alpha)
            self.item_probs /= self.item_probs.sum()
        else:
            self.item_probs = np.ones(n_items, dtype=np.float32) / n_items
        
        self.user_items = train_df.groupby('uid')['iid'].apply(set).to_dict()
        self.user_neg_probs = {}
        for u, items in self.user_items.items():
            mask = np.ones(n_items, dtype=bool)
            mask[list(items)] = False
            neg_probs = self.item_probs.copy()
            neg_probs[~mask] = 0
            if neg_probs.sum() > 0:
                self.user_neg_probs[u] = neg_probs / neg_probs.sum()

    def sample_batch(self, users, n_negs=1):
        negatives = np.empty((len(users), n_negs), dtype=np.int32)
        for idx, u in enumerate(users):
            if u in self.user_neg_probs:
                negatives[idx] = np.random.choice(self.n_items, size=n_negs, p=self.user_neg_probs[u])
            else:
                negatives[idx] = np.random.randint(0, self.n_items, size=n_negs)
        return negatives.tolist()

def evaluate(model, df, train_df, num_items, k=20, max_users=None):
    model.eval()
    user_pos = df.groupby('uid')['iid'].apply(list).to_dict()
    sampled_users = list(user_pos.keys())
    if max_users is not None and len(sampled_users) > max_users:
        sampled_users = sampled_users[:max_users]
    train_items_dict = defaultdict(set)
    for _, r in train_df.iterrows():
        train_items_dict[r['uid']].add(r['iid'])
    recalls, ndcgs, mrrs = [], [], []
    with torch.no_grad():
        user_emb, item_emb, _, _ = model.forward()
        batch_size = 256
        for i in range(0, len(sampled_users), batch_size):
            batch_users = sampled_users[i:i+batch_size]
            batch_user_emb = user_emb[batch_users]
            scores = torch.matmul(batch_user_emb, item_emb.t())
            for idx, u in enumerate(batch_users):
                pos = user_pos[u]
                if not pos:
                    continue

                user_scores = scores[idx].clone()
                if u in train_items_dict:
                    user_scores[list(train_items_dict[u])] = -float('inf')

                _, topk = torch.topk(user_scores, k=k)
                topk_items = topk.cpu().numpy()
                hits_mask = np.isin(topk_items, pos)

                recalls.append(hits_mask.sum() / len(pos))

                dcg = (hits_mask / np.log2(np.arange(2, k + 2))).sum()
                idcg = (1.0 / np.log2(np.arange(2, min(len(pos), k) + 2))).sum()
                ndcgs.append(dcg / (idcg + 1e-10))

                hit_idx = np.where(hits_mask)[0]
                mrrs.append(1.0 / (hit_idx[0] + 1) if len(hit_idx) > 0 else 0)

    return {
        'recall': np.mean(recalls),
        'ndcg': np.mean(ndcgs),
        'mrr': np.mean(mrrs),
    }

def train_model(model, train_df, valid_df, test_df, num_items, variant_name, 
                use_hierarchical_sampling=True):
    """训练单个消融变体"""
    print(f"\n{'='*70}")
    print(f"Training: {variant_name}")
    print(f"{'='*70}")
    print(f"  use_gated_module: {model.use_gated_module}")
    print(f"  use_cross_modal_attn: {model.use_cross_modal_attn}")
    print(f"  use_sparse_attention: {model.use_sparse_attention}")
    print(f"  use_contrastive_learning: {model.use_contrastive_learning}")
    print(f"  use_hierarchical_sampling: {use_hierarchical_sampling}")

    train_u, train_i = train_df['uid'].values, train_df['iid'].values
    neg_sampler = NegativeSampler(num_items, train_df, use_hierarchical=use_hierarchical_sampling)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=L2_REG)

    best_recall, best_epoch, patience_counter = 0, 0, 0
    best_metrics = None

    for epoch in range(EPOCHS):
        model.train()
        epoch_start = time.time()
        lr = LR * (0.5 + 0.5 * np.cos(np.pi * epoch / EPOCHS))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        indices = np.random.permutation(len(train_u))
        total_loss, n_batches = 0, 0

        for start in range(0, len(train_u), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(train_u))
            batch_idx = indices[start:end]
            u = torch.tensor(train_u[batch_idx], device=DEVICE)
            i_pos = torch.tensor(train_i[batch_idx], device=DEVICE)
            i_neg = torch.tensor(neg_sampler.sample_batch(train_u[batch_idx]), device=DEVICE)

            optimizer.zero_grad()
            user_emb, item_emb, id_item_emb, mm_item_emb = model.forward()

            pos_score = (user_emb[u] * item_emb[i_pos]).sum(dim=-1)
            neg_score = (user_emb[u] * item_emb[i_neg]).sum(dim=-1)
            bpr_loss = -torch.log(torch.sigmoid(pos_score - neg_score) + 1e-10).mean()

            bpr_reg = BPR_REG * (user_emb[u].norm(2).pow(2) + 
                                 item_emb[i_pos].norm(2).pow(2) + 
                                 item_emb[i_neg].norm(2).pow(2)) / (2 * len(u))

            cl_loss = model.compute_contrastive_loss(id_item_emb, mm_item_emb)

            (emb1, emb2), (emb3, emb4) = model.forward_multi_view()
            view_cl_loss = model.compute_view_contrastive_loss(emb1, emb2, emb3, emb4)

            mm_loss = F.mse_loss(mm_item_emb[i_pos], id_item_emb[i_pos])

            loss = bpr_loss + bpr_reg
            if model.use_contrastive_learning:
                loss += CONTRAST_WEIGHT * cl_loss + VIEW_CONTRAST_WEIGHT * view_cl_loss
            loss += MODAL_WEIGHT * mm_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        epoch_time = time.time() - epoch_start
        avg_loss = total_loss / n_batches

        val_metrics = evaluate(model, valid_df, train_df, num_items, k=K)

        if val_metrics['recall'] > best_recall:
            best_recall = val_metrics['recall']
            best_epoch = epoch + 1
            best_metrics = val_metrics
            patience_counter = 0
            # 保存最佳模型
            model_save_path = f'best_model_{variant_name.replace(" ", "_").replace("/", "_")}.pt'
            torch.save(model.state_dict(), model_save_path)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break

    model_save_path = f'best_model_{variant_name.replace(" ", "_").replace("/", "_")}.pt'
    if os.path.exists(model_save_path):
        model.load_state_dict(torch.load(model_save_path))
    
    test_metrics = evaluate(model, test_df, train_df, num_items, k=K)
    
    print(f"\n{variant_name} - Best Validation: Recall@{K}={best_recall:.4f}")
    print(f"{variant_name} - Test Results:")
    print(f"  Recall@{K}: {test_metrics['recall']:.4f}")
    print(f"  NDCG@{K}: {test_metrics['ndcg']:.4f}")
    print(f"  MRR@{K}: {test_metrics['mrr']:.4f}")

    return {
        'variant': variant_name,
        'best_epoch': best_epoch,
        'val_recall': best_recall,
        'test_recall': test_metrics['recall'],
        'test_ndcg': test_metrics['ndcg'],
        'test_mrr': test_metrics['mrr']
    }

def run_ablation_study():
    set_seed(2024)

    ratings = pd.read_csv(RATINGS_FILE, sep='::', engine='python', header=None,
                          names=['user', 'item', 'rating', 'ts'], encoding='latin-1')
    ratings = ratings[ratings['rating'] >= 4]

    user_map = {u: i for i, u in enumerate(ratings['user'].unique())}
    item_map = {i: j for j, i in enumerate(ratings['item'].unique())}
    ratings['uid'] = ratings['user'].map(user_map).astype(np.int32)
    ratings['iid'] = ratings['item'].map(item_map).astype(np.int32)
    num_users, num_items = len(user_map), len(item_map)

    ratings = ratings.sort_values('ts')
    train_df = ratings.iloc[:int(0.7 * len(ratings))].reset_index(drop=True)
    valid_df = ratings.iloc[int(0.7 * len(ratings)):int(0.8 * len(ratings))].reset_index(drop=True)
    test_df = ratings.iloc[int(0.8 * len(ratings)):].reset_index(drop=True)

    kg = pd.read_csv(KG_FILE, sep=r'\s+', header=None, names=['h', 'r', 't'], engine='python')
    kg = kg[kg['h'].isin(item_map)]
    ent_map = {}
    for e in set(kg['h']).union(set(kg['t'])):
        if e not in item_map:
            ent_map[e] = len(ent_map)
    rel_map = {r: i for i, r in enumerate(set(kg['r']))}
    kg['h'] = kg['h'].map(lambda x: item_map.get(x, ent_map.get(x, -1) + num_items if x in ent_map else -1))
    kg['t'] = kg['t'].map(lambda x: item_map.get(x, ent_map.get(x, -1) + num_items if x in ent_map else -1))
    kg['r'] = kg['r'].map(rel_map)
    kg = kg[(kg['h'] >= 0) & (kg['t'] >= 0)].astype(np.int32)
    num_entities, num_rels = len(ent_map), len(rel_map)

    movies = pd.read_csv(MOVIES_FILE, sep='::', engine='python', header=None,
                         names=['movie_id', 'title', 'genres'], encoding='latin-1')
    all_genres = set()
    for g in movies['genres']:
        if isinstance(g, str):
            all_genres.update(g.split('|'))
    genre_to_idx = {g: i for i, g in enumerate(sorted(all_genres))}
    movie_feat = np.zeros((num_items, len(genre_to_idx)), dtype=np.float32)
    valid_movies = movies[movies['movie_id'].isin(item_map)].copy()
    valid_movies['mid'] = valid_movies['movie_id'].map(item_map)
    for _, row in valid_movies.iterrows():
        mid = int(row['mid'])
        if isinstance(row['genres'], str):
            for g in row['genres'].split('|'):
                if g in genre_to_idx:
                    movie_feat[mid, genre_to_idx[g]] = 1.0
    movie_features = torch.FloatTensor(movie_feat).to(DEVICE)

    users = pd.read_csv(USERS_FILE, sep='::', engine='python', header=None,
                        names=['user_id', 'gender', 'age', 'occupation', 'zip'], encoding='latin-1')
    valid_users = users[users['user_id'].isin(user_map)].copy()
    valid_users = valid_users.sort_values('user_id').reset_index(drop=True)
    valid_users['uid'] = valid_users['user_id'].map(user_map)
    valid_users = valid_users.sort_values('uid').reset_index(drop=True)
    user_feat = np.zeros((num_users, 23), dtype=np.float32)
    user_feat[:, 0] = (valid_users['gender'].values == 'F').astype(np.float32)
    user_feat[:, 1] = valid_users['age'].values / 56.0
    for i, occ in enumerate(valid_users['occupation'].values):
        if 0 <= occ < 21:
            user_feat[i, 2 + occ] = 1.0
    user_features = torch.FloatTensor(user_feat).to(DEVICE)

    n_all = num_users + num_items + num_entities
    edges_row = np.concatenate([train_df['uid'].values, train_df['iid'].values + num_users,
                                kg['h'].values + num_users, kg['t'].values + num_users])
    edges_col = np.concatenate([train_df['iid'].values + num_users, train_df['uid'].values,
                                kg['t'].values + num_users, kg['h'].values + num_users])
    adj_mat = sp.coo_matrix((np.ones(len(edges_row)), (edges_row, edges_col)), shape=(n_all, n_all))
    rowsum = np.array(adj_mat.sum(1)).flatten()
    rowsum[rowsum == 0] = 1
    d_inv_sqrt = np.power(rowsum, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    norm_adj = sp.diags(d_inv_sqrt).dot(adj_mat).dot(sp.diags(d_inv_sqrt)).tocsr()

    ablation_configs = [
        {
            'name': 'Full Model',
            'use_gated_module': True,
            'use_cross_modal_attn': True,
            'use_sparse_attention': True,
            'use_contrastive_learning': True,
            'use_hierarchical_sampling': True
        },
        {
            'name': 'w/o Gated Module',
            'use_gated_module': False,
            'use_cross_modal_attn': True,
            'use_sparse_attention': True,
            'use_contrastive_learning': True,
            'use_hierarchical_sampling': True
        },
        {
            'name': 'w/o Cross-Modal Attention',
            'use_gated_module': True,
            'use_cross_modal_attn': False,
            'use_sparse_attention': True,
            'use_contrastive_learning': True,
            'use_hierarchical_sampling': True
        },
        {
            'name': 'w/o Sparse Attention',
            'use_gated_module': True,
            'use_cross_modal_attn': True,
            'use_sparse_attention': False,
            'use_contrastive_learning': True,
            'use_hierarchical_sampling': True
        },
        {
            'name': 'w/o Contrastive Learning',
            'use_gated_module': True,
            'use_cross_modal_attn': True,
            'use_sparse_attention': True,
            'use_contrastive_learning': False,
            'use_hierarchical_sampling': True
        },
        {
            'name': 'w/o Hierarchical Sampling',
            'use_gated_module': True,
            'use_cross_modal_attn': True,
            'use_sparse_attention': True,
            'use_contrastive_learning': True,
            'use_hierarchical_sampling': False
        }
    ]

    results = []
    for config in ablation_configs:
        model = GCMKGATCL_Ablation(
            num_users, num_items, num_entities, num_rels, EMB_DIM,
            norm_adj, movie_features, user_features, n_layers=N_LAYERS,
            use_gated_module=config['use_gated_module'],
            use_cross_modal_attn=config['use_cross_modal_attn'],
            use_sparse_attention=config['use_sparse_attention'],
            use_contrastive_learning=config['use_contrastive_learning']
        )
        
        result = train_model(
            model, train_df, valid_df, test_df, num_items, 
            config['name'], 
            use_hierarchical_sampling=config['use_hierarchical_sampling']
        )
        results.append(result)

    print("\n" + "="*70)
    print("ABLATION STUDY RESULTS SUMMARY")
    print("="*70)
    print(f"{'Variant':<35} {'Recall@20':>12}")
    print("-"*70)
    for r in results:
        print(f"{r['variant']:<35} {r['test_recall']:>12.4f}")
    print("="*70)

    results_df = pd.DataFrame(results)
    results_df.to_csv('ablation_study_results.csv', index=False)

    with open('ablation_study_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    return results

