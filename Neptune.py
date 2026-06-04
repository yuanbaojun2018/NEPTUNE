# -*- coding: utf-8 -*-
import os, sys, math, json, random, logging, datetime
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from tqdm import tqdm
from pathlib import Path



class NoisyTopkRouter(torch.nn.Module):
    def __init__(self, n_embed, num_experts, top_k):
        super(NoisyTopkRouter, self).__init__()
        self.top_k = top_k
        self.topkroute_linear = torch.nn.Linear(n_embed, num_experts)
        self.noise_linear = torch.nn.Linear(n_embed, num_experts)
    def forward(self, mh_output):
        logits = self.topkroute_linear(mh_output)
        noise_logits = self.noise_linear(mh_output)
        noise = torch.randn_like(logits) * torch.nn.functional.softplus(noise_logits)
        noisy_logits = logits + noise
        top_k_logits, indices = noisy_logits.topk(self.top_k, dim=-1)
        zeros = torch.full_like(noisy_logits, float('-inf'))
        sparse_logits = zeros.scatter(-1, indices, top_k_logits)
        router_output = torch.nn.functional.softmax(sparse_logits, dim=-1)

        # for aux loss
        prob = F.softmax(noisy_logits, dim=-1)                      # dense prob
        return router_output, indices, prob

class SparseMoE(torch.nn.Module):
    def __init__(self, n_embed, num_experts, top_k):
        super(SparseMoE, self).__init__()
        self.router = NoisyTopkRouter(n_embed, num_experts, top_k)
        self.experts = torch.nn.ModuleList([Expert(n_embed) for _ in range(num_experts)])
        self.top_k = top_k
    def forward(self, x):
        squeeze_back = False
        if x.dim() == 2:
            x = x.unsqueeze(1)
            squeeze_back = True
        gating_output, indices, dense_prob = self.router(x)
        final_output = torch.zeros_like(x)
        flat_x = x.view(-1, x.size(-1))
        flat_gating_output = gating_output.view(-1, gating_output.size(-1))
        flat_final = final_output.view(-1, final_output.size(-1))
        for i, expert in enumerate(self.experts):
            expert_mask = (indices == i).any(dim=-1)
            flat_mask = expert_mask.view(-1)
            if flat_mask.any():
                expert_input = flat_x[flat_mask]
                expert_output = expert(expert_input)
                gating_scores = flat_gating_output[flat_mask, i].unsqueeze(1)
                flat_final[flat_mask] += expert_output * gating_scores
        final_output = flat_final.view_as(final_output)
        self.last_aux = {
            "prob_mean": dense_prob.mean(dim=0),             # 每个专家平均分配概率
            "entropy": -(dense_prob * (dense_prob.clamp_min(1e-8)).log()).sum(dim=-1).mean()
        }
        return final_output.squeeze(1) if squeeze_back else final_output

# -----------------------------
# 配置
# -----------------------------
@dataclass
class Config:
    entity2id_path: str = "entity2id.txt"
    relation2id_path: str = "relation2id.txt"
    train_path: str = "train2id.txt"
    test_path: str = "test2id.txt"
    image_pth: str = "image_emb.pth"
    text_pth: str  = "text_emb.pth"
    n_embd: int = 512
    num_experts: int = 4
    top_k: int = 2
    lr: float = 1e-3
    weight_decay: float = 0.01
    batch_size: int = 1024
    num_workers: int = 2
    max_epochs: int = 10
    clip_grad: float = 1.0
    seed: int = 42
    ckpt_dir: str = "ckpts"
    eval_full_rank: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    type_constrain_path: str = ""   # 为空则不启用
    use_type_constrain: bool = True
    save_ckpt: bool = False          # ← 新增：是否保存 ckpt
    resume_from_ckpt: bool = False   # ← 新增：是否从 ckpt 恢复

# -----------------------------
# 工具函数（兼容首行数量 + 强韧 .pth 读入）
# -----------------------------
def set_seed(seed: int):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def _maybe_int_first_line(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip() != ""]
    if lines:
        # 若第一行形如 "79222" 这样的数量，跳过
        first = lines[0].split()[0]
        if first.isdigit():
            return lines[1:]
        try:
            _ = int(first)
            return lines[1:]
        except Exception:
            pass
    return lines

def load_id_map(path: str) -> Dict[int, str]:
    lines = _maybe_int_first_line(path)
    id2name = {}
    for ln in lines:
        parts = ln.split() if "\t" not in ln else ln.split("\t")
        if len(parts) < 2:  # 兼容空行/异常行
            continue
        # 主要格式：name id（name 可能包含尖括号和空格）
        name = " ".join(parts[:-1])
        try:
            idx = int(parts[-1])
            id2name[idx] = name
            continue
        except:
            pass
        # 次要格式：id name
        try:
            idx = int(parts[0]); name = " ".join(parts[1:])
            id2name[idx] = name
        except:
            continue
    return id2name

def load_triples(path: str) -> List[Tuple[int,int,int]]:
    lines = _maybe_int_first_line(path)
    triples = []
    for ln in lines:
        parts = ln.split() if "\t" not in ln else ln.split("\t")
        if len(parts) < 3: 
            continue
        h, r, t = int(parts[0]), int(parts[1]), int(parts[2])
        triples.append((h, r, t))
    return triples

def load_pth_matrix(pth_path: str) -> torch.Tensor:
    """
    兼容：torch.save 的新 zip 序列化（你给的 504B 开头）、老 pickle、以及 dict 包裹。
    优先返回浮点 Tensor [N, D]。
    """
    obj = torch.load(pth_path, map_location="cpu")
    # 直接 tensor
    if isinstance(obj, torch.Tensor):
        return obj.float()
    # numpy
    try:
        import numpy as np
        if isinstance(obj, np.ndarray):
            return torch.from_numpy(obj).float()
    except Exception:
        pass
    # dict（常见：{'emb': tensor} / {'features': tensor} / {'state_dict':..., 'emb':...}）
    if isinstance(obj, dict):
        # 常用键优先
        priority_keys = ["emb", "features", "image", "text", "data", "tensor", "array"]
        for k in priority_keys:
            if k in obj and isinstance(obj[k], torch.Tensor):
                return obj[k].float()
        # 退一步：找第一个张量
        for v in obj.values():
            if isinstance(v, torch.Tensor):
                return v.float()
        # 还有种情况：嵌套一层
        for v in obj.values():
            if isinstance(v, dict):
                for vv in v.values():
                    if isinstance(vv, torch.Tensor):
                        return vv.float()
    raise ValueError(f"Unrecognized .pth/.pt format: {pth_path}")



def _normalize_name(s: str) -> str:
    """把 <...> 去壳；URI 取最后一段；下划线→空格；去掉多余空白。"""
    if not isinstance(s, str):
        s = str(s)
    s = s.strip()
    if len(s) >= 2 and s[0] == "<" and s[-1] == ">":
        s = s[1:-1].strip()
    if "/" in s:
        s = s.rsplit("/", 1)[-1]
    s = s.replace("_", " ").strip()
    return s

def _read_clean_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    # 若首行是计数（纯数字），就跳过
    if lines and lines[0].replace("-", "").isdigit():
        lines = lines[1:]
    return lines

def load_entity_name2id(path: str) -> Dict[str, int]:
    """
    适配你的 entity2id.txt：第1列 name，第2列 id；允许 TAB 或 SPACE。
    """
    name2id = {}
    for ln in _read_clean_lines(path):
        parts = ln.split("\t") if "\t" in ln else ln.split()
        if len(parts) < 2:
            continue
        name = _normalize_name(" ".join(parts[:-1]))  # 更稳：允许名字里有空格
        try:
            idx = int(parts[-1])
        except Exception:
            # 如果第2列不是整数，尝试“首列是 id，其余是名字”格式（防御性）
            try:
                idx = int(parts[0])
                name = _normalize_name(" ".join(parts[1:]))
            except Exception:
                continue
        if name:
            name2id[name] = idx
    return name2id

def load_relation_name2id_if_any(path: str) -> Dict[str, int]:
    """
    若 relation2id.txt 含名字列则解析为 name->id；
    若两列都是 id（你的现状），则返回空字典 {}。
    """
    name2id = {}
    for ln in _read_clean_lines(path):
        parts = ln.split("\t") if "\t" in ln else ln.split()
        if len(parts) < 2:
            continue
        # 优先尝试“名字 + id”
        try:
            idx = int(parts[-1])
            name_candidate = " ".join(parts[:-1]).strip()
            # 如果“名字候选”其实也是纯数字（你的现状：两列 id），就跳过
            if name_candidate and not name_candidate.isdigit():
                name2id[_normalize_name(name_candidate)] = idx
            else:
                # 再尝试“id + 名字”
                try:
                    idx2 = int(parts[0])
                    name_candidate2 = " ".join(parts[1:]).strip()
                    if name_candidate2 and not name_candidate2.isdigit():
                        name2id[_normalize_name(name_candidate2)] = idx2
                except Exception:
                    pass
        except Exception:
            # 如果末列不是 id，尝试“id + 名字”
            try:
                idx = int(parts[0])
                name_candidate = " ".join(parts[1:]).strip()
                if name_candidate and not name_candidate.isdigit():
                    name2id[_normalize_name(name_candidate)] = idx
            except Exception:
                pass
    return name2id  # 若文件没有名字列，会是 {}

def _to_entity_id(tok: str, name2ent: Dict[str, int]) -> int:
    tok = str(tok).strip()
    # 纯数字
    if tok.isdigit():
        return int(tok)
    # 原样查
    if tok in name2ent:
        return name2ent[tok]
    # 规范化再查（<uri>, 下划线等）
    norm = _normalize_name(tok)
    if norm in name2ent:
        return name2ent[norm]
    raise ValueError(f"Unknown entity token '{tok}': not an int and not found in entity2id (name column).")

def _to_relation_id(tok: str, rel_name2id: Dict[str, int]) -> int:
    tok = str(tok).strip()
    # 关系优先按数字
    if tok.isdigit():
        return int(tok)
    # 如果 relation2id 有名字列，则支持名字解析
    if rel_name2id:
        if tok in rel_name2id:
            return rel_name2id[tok]
        norm = _normalize_name(tok)
        if norm in rel_name2id:
            return rel_name2id[norm]
    # 否则给出明确提示
    raise ValueError(
        f"Relation token '{tok}' is not an int, and relation2id.txt has no name column. "
        f"Please provide numeric relation IDs in triples, or update relation2id.txt to include names."
    )

# def load_triples_flexible(path: str,
#                           name2ent: Dict[str, int],
#                           rel_name2id: Dict[str, int]) -> List[Tuple[int, int, int]]:
#     triples = []
#     for ln in _read_clean_lines(path):
#         parts = ln.split("\t") if "\t" in ln else ln.split()
#         if len(parts) < 3:
#             continue
#         try:
#             h = _to_entity_id(parts[0], name2ent)
#             r = _to_relation_id(parts[1], rel_name2id)
#             t = _to_entity_id(parts[2], name2ent)
#             triples.append((h, r, t))
#         except Exception as e:
#             raise ValueError(f"[load_triples_flexible] {path}: bad line -> {ln}\n{e}")
#     return triples

def load_triples_flexible(path: str,
                          name2ent: Dict[str, int],
                          rel_name2id: Dict[str, int]) -> List[Tuple[int, int, int]]:
    """
    适配基准数据集的常见格式：head  tail  relation
    即 train2id/test2id 每行是：h_id  t_id  r_id
    解析后统一返回 (h, r, t) 形式，方便后续使用。
    """
    triples = []
    for ln in _read_clean_lines(path):
        parts = ln.split("\t") if "\t" in ln else ln.split()
        if len(parts) < 3:
            continue
        try:
            # ✅ 注意这里顺序：h, t, r
            h = _to_entity_id(parts[0], name2ent)       # 头实体
            t = _to_entity_id(parts[1], name2ent)       # 尾实体
            r = _to_relation_id(parts[2], rel_name2id)  # 关系
            triples.append((h, r, t))                   # 统一存成 (h, r, t)
        except Exception as e:
            raise ValueError(f"[load_triples_flexible] {path}: bad line -> {ln}\n{e}")
    return triples


# 简易 LoRA 线性
class LoRALinear(nn.Module):
    def __init__(self, in_f, out_f, r=8, alpha=16):
        super().__init__()
        self.down = nn.Linear(in_f, r, bias=False)
        self.up   = nn.Linear(r, out_f, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)
        self.scale = alpha / r
        self.frozen = True  # 冻结主干时可用

    def forward(self, x):
        return self.up(self.down(x)) * self.scale

class Expert(nn.Module):
    def __init__(self, n_embd, lora_r=8):
        super().__init__()
        self.fc1 = nn.Linear(n_embd, 4 * n_embd)
        self.fc2 = nn.Linear(4 * n_embd, n_embd)
        self.lora1 = LoRALinear(n_embd, 4 * n_embd, r=lora_r)
        self.lora2 = LoRALinear(4 * n_embd, n_embd, r=lora_r)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        y = self.fc1(x) + self.lora1(x)
        y = self.act(y)
        y = self.fc2(y) + self.lora2(y)
        return self.drop(y)


# -----------------------------
# 数据集（训练/评估均用全量尾实体打分）
# -----------------------------
class KGCDataset(Dataset):
    def __init__(self, triples: List[Tuple[int,int,int]]):
        self.data = triples
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        h, r, t = self.data[idx]
        return (torch.tensor(h, dtype=torch.long),
                torch.tensor(r, dtype=torch.long),
                torch.tensor(t, dtype=torch.long))



def filter_invalid_triples(triples: List[Tuple[int,int,int]], num_entities: int, num_relations: int,
                           stage: str) -> List[Tuple[int,int,int]]:
    """丢弃任何越界的 h/r/t，避免 r=9415 这类错误导致崩溃；打印过滤统计。"""
    before = len(triples)
    out = []
    for h, r, t in triples:
        if 0 <= h < num_entities and 0 <= r < num_relations and 0 <= t < num_entities:
            out.append((h, r, t))
    dropped = before - len(out)
    if dropped > 0:
        print(f"[WARN] {stage}: dropped {dropped}/{before} triples due to id out of range.")
        logging.info(f"[WARN] {stage}: dropped {dropped}/{before} triples due to id out of range.")
    return out

# -----------------------------
# 模型：两模态融合 + MoE 头
# -----------------------------
class MultiModalKGC(nn.Module):
    def __init__(self,
                 image_raw: torch.Tensor,   # [N, Di]
                 text_raw: torch.Tensor,    # [N, Dt]
                 num_relations: int,
                 n_embd: int,
                 num_experts: int,
                 top_k: int):
        super().__init__()
        assert image_raw.size(0) == text_raw.size(0), "image/text 行数不一致（必须与实体数一致）"
        self.register_buffer("image_raw", image_raw)
        self.register_buffer("text_raw",  text_raw)
        Di = image_raw.size(1)
        Dt = text_raw.size(1)
        self.img_proj = nn.Linear(Di, n_embd, bias=True)
        self.txt_proj = nn.Linear(Dt, n_embd, bias=True)
        # 可学习门控
        # self.log_a = nn.Parameter(torch.zeros(1))
        # self.log_b = nn.Parameter(torch.zeros(1))
        self.log_a = nn.Parameter(torch.tensor(1.0))
        self.log_b = nn.Parameter(torch.tensor(1.0))

        
        # 关系嵌入
        # self.rel_emb = nn.Embedding(num_relations, n_embd)
        self.rel_emb = nn.Embedding(num_relations * 2, n_embd)
        self.num_rel = num_relations  # 记录一下正向关系个数
        # (h ⊕ r) → n_embd
        self.hr_proj = nn.Linear(2 * n_embd, n_embd)
        # MoE
        self.moe = SparseMoE(n_embd, num_experts, top_k)

        self.logit_scale = nn.Parameter(torch.tensor(3.0))   # s 初值 ~ 3~5
        #self.margin = 0.10                                   # m 0.05~0.35 之间试
        self.margin = 0.20

        # __init__
        self.ent_residual = nn.Embedding(image_raw.size(0), n_embd)
        nn.init.zeros_(self.ent_residual.weight)
    
    # ===== 新增：同时返回四张实体表（fused / struct / image / text）=====
    def entity_tables_all(self):
        """
        返回四种实体表（未归一化）:
            fused  : (a*img + b*txt)/(a+b) + struct_res
            struct : 结构残差 ent_residual
            img    : 图像投影
            txt    : 文本投影
        训练时用它们分别计算 loss，评估仍然只用 fused。
        """
        img = self.img_proj(self.image_raw)   # [N, D]
        txt = self.txt_proj(self.text_raw)    # [N, D]
        a = F.softplus(self.log_a) + 1e-6
        b = F.softplus(self.log_b) + 1e-6
        struct = self.ent_residual.weight     # [N, D]

        fused = (a * img + b * txt) / (a + b) + struct
        return fused, struct, img, txt

    # ===== 新增：给定任一实体表，复用 MoE 打分 =====
    def _score_with_table(self, ent_table, h, r,
                          targets: Optional[torch.LongTensor] = None,
                          full_rank: bool = True):
        """
        通用打分函数：
            ent_table: [N, D]（未归一化）
            h, r, targets: [B]
        返回: {"logits": [B, N], "loss": CE}（若 targets 不为 None）
        """
        # 归一化实体表
        ent_table = F.normalize(ent_table, p=2, dim=-1)       # [N, D]

        # 取 head / rel
        head = ent_table.index_select(0, h)                   # [B, D]
        rel  = self.rel_emb(r)                                # [B, D]

        # (h ⊕ r) 走 MoE
        q = self.hr_proj(torch.cat([head, rel], dim=-1))      # [B, D]
        route_ctx = q + rel
        q = self.moe(route_ctx)                               # [B, D]
        q = F.normalize(q, p=2, dim=-1)

        # 全量实体打分
        logits = q @ ent_table.t()                            # [B, N] cosine

        # additive margin
        if targets is not None:
            idx = torch.arange(q.size(0), device=q.device)
            logits[idx, targets] = logits[idx, targets] - self.margin

        # logit scale
        logits = logits * F.softplus(self.logit_scale)


        if not full_rank:
            raise NotImplementedError("full_rank=True is required for exact metrics.")

        out = {"logits": logits}
        if targets is not None:
            out["loss"] = F.cross_entropy(logits, targets)
        return out

    def fused_entity_table(self) -> torch.Tensor:
        img = self.img_proj(self.image_raw)  # [N, D]
        txt = self.txt_proj(self.text_raw)   # [N, D]
        a = F.softplus(self.log_a) + 1e-6
        b = F.softplus(self.log_b) + 1e-6
        fused = (a * img + b * txt) / (a + b)
        fused = F.normalize(fused + self.ent_residual.weight, p=2, dim=-1)
        return fused

    def forward(self, h: torch.LongTensor, r: torch.LongTensor,
                targets: Optional[torch.LongTensor] = None, full_rank: bool = True):
        # 评估 & 原始代码默认使用 fused 实体表
        fused, _, _, _ = self.entity_tables_all()
        return self._score_with_table(fused, h, r, targets=targets, full_rank=full_rank)





@torch.no_grad()
def evaluate_one_direction(
    model: nn.Module,
    data_loader: DataLoader,
    device: str,
    *,
    direction: str,                               # 'tail' or 'head'
    sr2o: Optional[Dict[Tuple[int,int], set]] = None,
    tr2h: Optional[Dict[Tuple[int,int], set]] = None,
    rel2tails: Optional[Dict[int, set]] = None,
    rel2heads: Optional[Dict[int, set]] = None,
    dump_jsonl: Optional[str] = None,             # ← 新增：jsonl 输出路径
    topk_dump: int = 10,                          # ← 新增：每样本保存前 K 个预测
    id2ent: Optional[Dict[int, str]] = None,      # ← 新增：可选，存可读名字
    id2rel: Optional[Dict[int, str]] = None,      # ← 新增：可选，存可读名字
) -> Dict[str, float]:
    assert direction in ("tail", "head")
    model.eval()

    # === 禁用 type-constrain（关键两行）===
    # rel2tails = None
    # rel2heads = None

    def _is_main_process():
        return (not torch.distributed.is_available()
                or not torch.distributed.is_initialized()
                or torch.distributed.get_rank() == 0)

    hits1 = hits3 = hits10 = 0
    mrr_sum = 0.0
    n = 0

    # 仅主进程写 jsonl
    writer = None
    if dump_jsonl is not None and _is_main_process():
        Path(dump_jsonl).parent.mkdir(parents=True, exist_ok=True)
        writer = open(dump_jsonl, "a", encoding="utf-8")

    pbar = tqdm(
        data_loader,
        total=len(data_loader) if hasattr(data_loader, "__len__") else None,
        desc=f"[Eval-{direction.upper()}]",
        dynamic_ncols=True,
        leave=False,
        disable=not _is_main_process(),
    )

    for h, r, t in pbar:
        h = h.to(device); r = r.to(device); t = t.to(device)

        if direction == "tail":
            logits = model(h, r, targets=None, full_rank=True)["logits"]  # [B, N]
            gold = t
            if rel2tails is not None:
                raw = logits.clone()
                for i in range(h.size(0)):
                    allow = rel2tails.get(int(r[i]), None)
                    if allow:
                        logits[i, :] = float('-inf')
                        idx = torch.tensor(list(allow), device=logits.device, dtype=torch.long)
                        logits[i, idx] = raw[i, idx]
                        if int(gold[i]) not in allow:
                            logits[i, int(gold[i])] = raw[i, int(gold[i])]
                del raw
            if sr2o is not None:
                for i in range(h.size(0)):
                    others = sr2o.get((int(h[i]), int(r[i])), None)
                    if others:
                        for t_ex in others:
                            if t_ex != int(gold[i]):
                                logits[i, t_ex] = -1e9
            query_vec = h
        else:
            r_inv = r + model.num_rel
            logits = model(t, r_inv, targets=None, full_rank=True)["logits"]
            gold = h
            if rel2heads is not None:
                raw = logits.clone()
                for i in range(h.size(0)):
                    allow = rel2heads.get(int(r[i]), None)
                    if allow:
                        logits[i, :] = float('-inf')
                        idx = torch.tensor(list(allow), device=logits.device, dtype=torch.long)
                        logits[i, idx] = raw[i, idx]
                        if int(gold[i]) not in allow:
                            logits[i, int(gold[i])] = raw[i, int(gold[i])]
                del raw
            if tr2h is not None:
                for i in range(h.size(0)):
                    others = tr2h.get((int(t[i]), int(r[i])), None)
                    if others:
                        for h_ex in others:
                            if h_ex != int(gold[i]):
                                logits[i, h_ex] = -1e9
            query_vec = t

        # 排名与指标
        ranks = torch.argsort(torch.argsort(-logits, dim=1), dim=1)\
                 .gather(1, gold.view(-1,1)).squeeze(1) + 1
        hits1  += (ranks <= 1).sum().item()
        hits3  += (ranks <= 3).sum().item()
        hits10 += (ranks <= 10).sum().item()
        mrr_sum += (1.0 / ranks.float()).sum().item()
        n += h.size(0)

        # 仅主进程记录 JSONL（每样本一行）
        if writer is not None:
            # 取 top-k 预测
            topk_vals, topk_idx = torch.topk(logits, k=min(topk_dump, logits.size(1)), dim=1)
            topk_vals = topk_vals.detach().cpu().tolist()
            topk_idx  = topk_idx.detach().cpu().tolist()
            h_cpu = h.detach().cpu().tolist()
            r_cpu = r.detach().cpu().tolist()
            t_cpu = t.detach().cpu().tolist()
            ranks_cpu = ranks.detach().cpu().tolist()

            for i in range(len(h_cpu)):
                rec = {
                    "direction": direction,               # 'tail' / 'head'
                    "h_id": int(h_cpu[i]),
                    "r_id": int(r_cpu[i]),
                    "t_id": int(t_cpu[i]),
                    "query_id": int(query_vec[i].item()), # tail: h；head: t
                    "gold_id": int(gold[i].item()),
                    "gold_rank": int(ranks_cpu[i]),
                    "topk_pred": [
                        {"ent_id": int(eid), "score": float(s)}
                        for eid, s in zip(topk_idx[i], topk_vals[i])
                    ],
                }
                # 可选：可读名字
                if id2ent is not None:
                    rec["h_name"] = id2ent.get(rec["h_id"], str(rec["h_id"]))
                    rec["t_name"] = id2ent.get(rec["t_id"], str(rec["t_id"]))
                    rec["gold_name"] = id2ent.get(rec["gold_id"], str(rec["gold_id"]))
                    rec["topk_pred_names"] = [id2ent.get(int(e["ent_id"]), str(e["ent_id"])) for e in rec["topk_pred"]]
                if id2rel is not None:
                    rec["r_name"] = id2rel.get(rec["r_id"], str(rec["r_id"]))

                writer.write(json.dumps(rec, ensure_ascii=False) + "\n")

        pbar.set_postfix({
            "Hit@1":  f"{hits1 / max(n,1):.4f}",
            "Hit@3":  f"{hits3 / max(n,1):.4f}",
            "Hit@10": f"{hits10 / max(n,1):.4f}",
            "MRR":    f"{mrr_sum / max(n,1):.4f}",
        })

    pbar.close()
    if writer is not None:
        writer.close()

    return {
        "hit@1":  hits1 / n,
        "hit@3":  hits3 / n,
        "hit@10": hits10 / n,
        "MRR":    mrr_sum / n,
    }


@torch.no_grad()
def evaluate_bidir(
    model: nn.Module,
    data_loader: DataLoader,
    device: str,
    *,
    sr2o: Optional[Dict[Tuple[int,int], set]] = None,
    tr2h: Optional[Dict[Tuple[int,int], set]] = None,
    rel2tails: Optional[Dict[int, set]] = None,
    rel2heads: Optional[Dict[int, set]] = None,
    dump_dir: Optional[str] = None,         # ← 新增：目录
    epoch: Optional[int] = None,            # ← 新增：文件名带上 epoch
    id2ent: Optional[Dict[int, str]] = None,
    id2rel: Optional[Dict[int, str]] = None,
    topk_dump: int = 10,
) -> Dict[str, Dict[str, float]]:
    dump_tail = dump_head = None
    if dump_dir is not None:
        if epoch is None:
            dump_tail = os.path.join(dump_dir, "eval_tail.jsonl")
            dump_head = os.path.join(dump_dir, "eval_head.jsonl")
        else:
            dump_tail = os.path.join(dump_dir, f"eval_tail_epoch{epoch}.jsonl")
            dump_head = os.path.join(dump_dir, f"eval_head_epoch{epoch}.jsonl")

    tail_metrics = evaluate_one_direction(
        model, data_loader, device,
        direction="tail",
        sr2o=sr2o, tr2h=None,
        rel2tails=rel2tails, rel2heads=None,
        dump_jsonl=dump_tail, topk_dump=topk_dump,
        id2ent=id2ent, id2rel=id2rel,
    )
    head_metrics = evaluate_one_direction(
        model, data_loader, device,
        direction="head",
        sr2o=None, tr2h=tr2h,
        rel2tails=None, rel2heads=rel2heads,
        dump_jsonl=dump_head, topk_dump=topk_dump,
        id2ent=id2ent, id2rel=id2rel,
    )
    avg_metrics = {k: 0.5 * (tail_metrics[k] + head_metrics[k]) for k in tail_metrics.keys()}
    return {"tail": tail_metrics, "head": head_metrics, "final": avg_metrics}



def contrastive_loss(z1, z2, tau=0.1):
    z1 = F.normalize(z1, dim=-1); z2 = F.normalize(z2, dim=-1)
    sim = z1 @ z2.t() / tau
    labels = torch.arange(z1.size(0), device=z1.device)
    return F.cross_entropy(sim, labels)


def build_sr2o(all_triples):
    sr2o = defaultdict(set)
    for h, r, t in all_triples:
        sr2o[(h, r)].add(t)
    return sr2o

def build_tr2h(all_triples):
    tr2h = defaultdict(set)
    for h, r, t in all_triples:
        tr2h[(t, r)].add(h)
    return tr2h


def mask_logits_with_type_and_filter(
    logits: torch.Tensor,
    *,
    h: torch.LongTensor,
    r: torch.LongTensor,
    t: torch.LongTensor,
    gold: torch.LongTensor,
    direction: str,
    rel2tails: Optional[Dict[int, set]] = None,
    rel2heads: Optional[Dict[int, set]] = None,
    sr2o: Optional[Dict[Tuple[int,int], set]] = None,
    tr2h: Optional[Dict[Tuple[int,int], set]] = None,
) -> torch.Tensor:
    """
    在训练阶段对 logits 做和测试阶段一样的：
      - type constrain
      - filter ranking
    direction: "tail" 或 "head"
    logits: [B, N]
    """
    logits = logits.clone()
    B = h.size(0)

    # ========= 关键补丁：为当前 dtype 选择一个“非常小”的安全值 =========
    if torch.is_floating_point(logits):
        if logits.dtype == torch.float16:
            # fp16 最小有限值大约是 -65504，用这个就不会溢出
            very_neg = torch.finfo(torch.float16).min
        else:
            # fp32 / bf16 等用一个很小的数即可（不会溢出）
            very_neg = -1e9
    else:
        # 理论上 logits 不会是整型，这里只是兜底
        very_neg = -1e9
    # =======================================================

    if direction == "tail":
        # ---- type constrain: rel2tails ----
        if rel2tails is not None:
            raw = logits.clone()
            for i in range(B):
                allow = rel2tails.get(int(r[i]), None)
                if allow:
                    # 先全部置为 very_neg
                    logits[i, :] = very_neg
                    idx = torch.tensor(list(allow), device=logits.device, dtype=torch.long)
                    logits[i, idx] = raw[i, idx]
                    # 确保 gold 在集合里（万一 type 文件没涵盖）
                    if int(gold[i]) not in allow:
                        logits[i, int(gold[i])] = raw[i, int(gold[i])]
            del raw

        # ---- filter ranking: sr2o[(h,r)] ----
        if sr2o is not None:
            for i in range(B):
                others = sr2o.get((int(h[i]), int(r[i])), None)
                if others:
                    for t_ex in others:
                        if t_ex != int(gold[i]):
                            # 原来是 -1e9，这里改成 very_neg，fp16 不会溢出
                            logits[i, t_ex] = very_neg

    elif direction == "head":
        # ---- type constrain: rel2heads ----
        if rel2heads is not None:
            raw = logits.clone()
            for i in range(B):
                allow = rel2heads.get(int(r[i]), None)
                if allow:
                    logits[i, :] = very_neg
                    idx = torch.tensor(list(allow), device=logits.device, dtype=torch.long)
                    logits[i, idx] = raw[i, idx]
                    # 同样，保证 gold 在里面
                    if int(gold[i]) not in allow:
                        logits[i, int(gold[i])] = raw[i, int(gold[i])]
            del raw

        # ---- filter ranking: tr2h[(t,r)] ----
        if tr2h is not None:
            for i in range(B):
                others = tr2h.get((int(t[i]), int(r[i])), None)
                if others:
                    for h_ex in others:
                        if h_ex != int(gold[i]):
                            logits[i, h_ex] = very_neg

    return logits



# def mask_logits_with_type_and_filter(
#     logits: torch.Tensor,
#     *,
#     h: torch.LongTensor,
#     r: torch.LongTensor,
#     t: torch.LongTensor,
#     gold: torch.LongTensor,
#     direction: str,
#     rel2tails: Optional[Dict[int, set]] = None,
#     rel2heads: Optional[Dict[int, set]] = None,
#     sr2o: Optional[Dict[Tuple[int,int], set]] = None,
#     tr2h: Optional[Dict[Tuple[int,int], set]] = None,
# ) -> torch.Tensor:
#     """
#     在训练阶段对 logits 做和测试阶段一样的：
#       - type constrain
#       - filter ranking
#     direction: "tail" 或 "head"
#     logits: [B, N]
#     """
#     logits = logits.clone()
#     B = h.size(0)

#     if direction == "tail":
#         # ---- type constrain: rel2tails ----
#         if rel2tails is not None:
#             raw = logits.clone()
#             for i in range(B):
#                 allow = rel2tails.get(int(r[i]), None)
#                 if allow:
#                     logits[i, :] = float('-inf')
#                     idx = torch.tensor(list(allow), device=logits.device, dtype=torch.long)
#                     logits[i, idx] = raw[i, idx]
#                     # 确保 gold 在集合里（万一 type 文件没涵盖）
#                     if int(gold[i]) not in allow:
#                         logits[i, int(gold[i])] = raw[i, int(gold[i])]
#             del raw

#         # ---- filter ranking: sr2o[(h,r)] ----
#         if sr2o is not None:
#             for i in range(B):
#                 others = sr2o.get((int(h[i]), int(r[i])), None)
#                 if others:
#                     for t_ex in others:
#                         if t_ex != int(gold[i]):
#                             logits[i, t_ex] = -1e9

#     elif direction == "head":
#         # ---- type constrain: rel2heads ----
#         if rel2heads is not None:
#             raw = logits.clone()
#             for i in range(B):
#                 allow = rel2heads.get(int(r[i]), None)
#                 if allow:
#                     logits[i, :] = float('-inf')
#                     idx = torch.tensor(list(allow), device=logits.device, dtype=torch.long)
#                     logits[i, idx] = raw[i, idx]
#                     # 同样，保证 gold 在里面
#                     if int(gold[i]) not in allow:
#                         logits[i, int(gold[i])] = raw[i, int(gold[i])]
#             del raw

#         # ---- filter ranking: tr2h[(t,r)] ----
#         if tr2h is not None:
#             for i in range(B):
#                 others = tr2h.get((int(t[i]), int(r[i])), None)
#                 if others:
#                     for h_ex in others:
#                         if h_ex != int(gold[i]):
#                             logits[i, h_ex] = -1e9

#     return logits


def load_type_constrain(path: str):
    """
    解析 type_constrain.txt：
    第1行：关系总数（忽略即可）
    之后：两行一组：
        行1：rel_id <tab/space> 许多 head 候选实体 id
        行2：rel_id <tab/space> 许多 tail 候选实体 id
    返回：
        rel2heads: Dict[int, Set[int]]
        rel2tails: Dict[int, Set[int]]
    """
    rel2heads, rel2tails = {}, {}
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    if not lines:
        return {}, {}

    # 跳过第一行计数
    i = 1
    N = len(lines)
    while i < N:
        # 解析一行
        def _parse_line(s: str):
            parts = s.split() if "\t" not in s else s.split("\t")
            # parts[0] 是 rel_id，后面都是实体 id
            rid = int(parts[0])
            cands = []
            for x in parts[1:]:
                try:
                    cands.append(int(x))
                except:
                    pass
            return rid, set(cands)

        # 第 i 行：head 候选
        rid_h, heads = _parse_line(lines[i])
        # 第 i+1 行：tail 候选（有些文件可能缺行，做一下保护）
        tails = set()
        if i + 1 < N:
            rid_t, tails = _parse_line(lines[i + 1])
            # 正常情况下 rid_h == rid_t
        rel2heads[rid_h] = heads
        rel2tails[rid_h] = tails
        i += 2

    return rel2heads, rel2tails




def setup_run_dirs(base_dir: str = "runs", run_name: str = None):
    """
    创建本次运行的输出目录：{base_dir}/{run_name}/ ，包含 logs/ 和 preds/ 子目录。
    返回 (root_dir, logs_dir, preds_dir, run_name)。
    """
    if run_name is None:
        # 例如 2025-10-24_15-06-32
        run_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    root = Path(base_dir) / run_name
    logs = root / "logs"
    preds = root / "preds"
    logs.mkdir(parents=True, exist_ok=True)
    preds.mkdir(parents=True, exist_ok=True)
    return str(root), str(logs), str(preds), run_name


def setup_logger(logs_dir: str, run_name: str, rank0_only: bool = True):
    """
    配置 logging：同时写到控制台和 {logs_dir}/{run_name}.log。
    在 DDP 时只在 rank0 打印到控制台（文件仍可 rank0 写）。
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # 清理旧的 handlers（避免在 notebook 或多次调用时重复）
    for h in list(logger.handlers):
        logger.removeHandler(h)

    log_path = Path(logs_dir) / f"{run_name}.log"

    # File handler
    fh = logging.FileHandler(str(log_path), encoding="utf-8")
    fh.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Console handler（DDP 仅 rank0 打）
    def _is_main_process():
        return (not torch.distributed.is_available()
                or not torch.distributed.is_initialized()
                or torch.distributed.get_rank() == 0)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO if (not rank0_only or _is_main_process()) else logging.CRITICAL)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    logging.info(f"Log file -> {log_path}")
    return logger



# -----------------------------
# 训练入口
# -----------------------------
@dataclass
class TrainState:
    epoch: int
    best_final_hit1: float          # 用于早停：跟踪 FINAL Hit@1 的最好值
    patience: int                   # 连续未提升的 epoch 计数
    best_final_metrics: Dict[str, float]   # 在所有 epoch 上的 FINAL 指标逐项最大值（hit@1/3/10/MRR）


def main(cfg: Config):
    # === 运行目录 & 日志 ===
    run_root, logs_dir, preds_dir, run_name = setup_run_dirs(base_dir="runs")
    setup_logger(logs_dir, run_name)
    logging.info(f"Run dir: {run_root}")
    logging.info(f"Config: {cfg.__dict__}")

    set_seed(cfg.seed)
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    device = torch.device(cfg.device)

    # 1) 加载 id
    # id2ent = load_id_map(cfg.entity2id_path)
    # id2rel = load_id_map(cfg.relation2id_path)
    # num_entities = max(id2ent.keys()) + 1 if id2ent else None
    # num_relations = max(id2rel.keys()) + 1 if id2rel else None
    # assert num_entities is not None and num_relations is not None, "id 文件解析失败"
    # 可读名字（如果文件在就加载；没有就用 {}）
    id2ent = load_id_map(cfg.entity2id_path) if os.path.exists(cfg.entity2id_path) else {}
    id2rel = load_id_map(cfg.relation2id_path) if os.path.exists(cfg.relation2id_path) else {}

    # 先加载 embedding
    img_mat = load_pth_matrix(cfg.image_pth)   # [N, Di]
    txt_mat = load_pth_matrix(cfg.text_pth)    # [N, Dt]
    assert img_mat.size(0) == txt_mat.size(0), "image/text 行数必须一致"

    # 再用 embedding 行数作为实体总数（最可靠）
    num_entities = img_mat.size(0)


    def _maybe_fix_offset(triples, num_entities, num_relations):
        hs = [h for h,_,_ in triples]; rs = [r for _,r,_ in triples]; ts = [t for _,_,t in triples]
        if min(hs + ts) == 1 and max(hs + ts) == num_entities \
        and min(rs) == 1 and max(rs) == num_relations:
            # 看起来是 1-based，整体减 1
            return [(h-1, r-1, t-1) for h,r,t in triples], True
        return triples, False


    # 关系数用三元组里出现的最大 r + 1 来定（更稳）
    # train_triples_raw = load_triples(cfg.train_path)
    # test_triples_raw  = load_triples(cfg.test_path)
    # 读取实体名字→id（你的 entity2id：第一列是名字，第二列是id）
    ent_name2id = load_entity_name2id(cfg.entity2id_path) if os.path.exists(cfg.entity2id_path) else {}

    # 读取关系名字→id（如果 relation2id 里没有名字列，会返回 {}）
    rel_name2id = load_relation_name2id_if_any(cfg.relation2id_path) if os.path.exists(cfg.relation2id_path) else {}

    # 三元组：实体支持“名字/URI/数字”，关系优先数字；若出现关系名字但映射空，会报清晰错误
    train_triples_raw = load_triples_flexible(cfg.train_path, ent_name2id, rel_name2id)
    test_triples_raw  = load_triples_flexible(cfg.test_path,  ent_name2id, rel_name2id)
    
    max_r = max([r for _, r, _ in train_triples_raw + test_triples_raw])
    num_relations = max_r + 1



    # 先用上面新确定的 num_entities/num_relations
    train_triples_raw, fixed1 = _maybe_fix_offset(train_triples_raw, num_entities, num_relations)
    test_triples_raw,  fixed2 = _maybe_fix_offset(test_triples_raw,  num_entities, num_relations)



    # 2) 加载 triples 并做越界过滤（兼容 r=9415 等异常）
    # train_triples = filter_invalid_triples(load_triples(cfg.train_path), num_entities, num_relations, "train")
    # test_triples  = filter_invalid_triples(load_triples(cfg.test_path),  num_entities, num_relations, "test")
    train_triples  = filter_invalid_triples(train_triples_raw, num_entities, num_relations, "train")
    test_triples   = filter_invalid_triples(test_triples_raw,  num_entities, num_relations, "test")


    # 3) 加载 embedding（兼容 zip 序列化）
    img_mat = load_pth_matrix(cfg.image_pth)   # [N, Di]
    txt_mat = load_pth_matrix(cfg.text_pth)    # [N, Dt]
    assert img_mat.size(0) == num_entities and txt_mat.size(0) == num_entities, \
        f"embedding 行数与实体数不符: img={img_mat.size(0)}, txt={txt_mat.size(0)}, N={num_entities}"

    # 4) DataLoader
    train_loader = DataLoader(KGCDataset(train_triples),
                              batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True)
    test_loader  = DataLoader(KGCDataset(test_triples),
                              batch_size=cfg.batch_size, shuffle=False,
                              num_workers=cfg.num_workers, pin_memory=True)

    # 5) 模型/优化器
    model = MultiModalKGC(
        image_raw=img_mat.to(device),
        text_raw=txt_mat.to(device),
        num_relations=num_relations,
        n_embd=cfg.n_embd,
        num_experts=cfg.num_experts,
        top_k=cfg.top_k,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # 新增：Cosine 调度
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt,
        T_max=cfg.max_epochs,          # 或 len(train_loader) * cfg.max_epochs 更细粒度
        eta_min=cfg.lr * 0.1
    )

    # 6) 断点
    last_ckpt = os.path.join(cfg.ckpt_dir, "last.pt")

    state = TrainState(
        epoch=0,
        best_final_hit1=-1.0,
        patience=0,
        best_final_metrics={"hit@1": 0.0, "hit@3": 0.0, "hit@10": 0.0, "MRR": 0.0}
    )

    if cfg.resume_from_ckpt and os.path.exists(last_ckpt):
        ckpt = torch.load(last_ckpt, map_location=device)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        state.epoch = ckpt.get("epoch", 0)
        state.best_final_hit1 = ckpt.get("best_final_hit1", -1.0)
        state.patience = ckpt.get("patience", 0)
        state.best_final_metrics = ckpt.get(
            "best_final_metrics",
            {"hit@1": 0.0, "hit@3": 0.0, "hit@10": 0.0, "MRR": 0.0}
        )
        print(f"[INFO] Resumed from {last_ckpt} (epoch {state.epoch})")


    all_triples = train_triples + test_triples 
    sr2o = build_sr2o(all_triples)
    tr2h = build_tr2h(all_triples)
    rel2heads, rel2tails = {}, {}
    if getattr(cfg, "use_type_constrain", False) and getattr(cfg, "type_constrain_path", ""):
        rel2heads, rel2tails = load_type_constrain(cfg.type_constrain_path)
    # 7) 训练

    # ===== [2] 训练循环（替换你原来的 for epoch... 那段内部“for h,r,t in train_loader”这一小段）=====
    for epoch in range(state.epoch + 1, cfg.max_epochs + 1):
        model.train()
        running_loss, seen = 0.0, 0
        ema_loss = None  # 指标更稳的指数滑动均值
        lr_now = opt.param_groups[0]["lr"]

        # 用 tqdm 包裹 dataloader
        pbar = tqdm(
            train_loader,
            total=len(train_loader),
            desc=f"[Train] Epoch {epoch}",
            dynamic_ncols=True,
            leave=False,
        )

        for h, r, t in pbar:
            h = h.to(device); r = r.to(device); t = t.to(device)
            opt.zero_grad(set_to_none=True)

        #     with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
        #         # ===== 1) 取四种实体表（未归一化）=====
        #         fused_tab, struct_tab, img_tab, txt_tab = model.entity_tables_all()

                # # ===== 2) 主干：fused 模态，双向 KGC =====
                # # --- tail 方向（fused）---
                # out_t = model._score_with_table(fused_tab, h, r, targets=t, full_rank=True)
                # loss_tail = out_t["loss"]

                # # --- reciprocal head 方向（fused）---
                # r_inv = r + model.num_rel
                # out_h = model._score_with_table(fused_tab, t, r_inv, targets=h, full_rank=True)
                # loss_head = out_h["loss"]

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                fused_tab, struct_tab, img_tab, txt_tab = model.entity_tables_all()

                # ===== 2) 主干：fused 模态，双向 KGC =====
                # --- tail 方向（fused）---
                out_t = model._score_with_table(fused_tab, h, r, targets=t, full_rank=True)
                # out_t["logits"] 已经包含 margin + logit_scale
                logits_tail = mask_logits_with_type_and_filter(
                    out_t["logits"],
                    h=h, r=r, t=t, gold=t,
                    direction="tail",
                    rel2tails=rel2tails if cfg.use_type_constrain else None,
                    rel2heads=None,
                    sr2o=sr2o,
                    tr2h=None,
                )
                loss_tail = F.cross_entropy(logits_tail, t)

                # --- reciprocal head 方向（fused）---
                r_inv = r + model.num_rel
                out_h = model._score_with_table(fused_tab, t, r_inv, targets=h, full_rank=True)
                logits_head = mask_logits_with_type_and_filter(
                    out_h["logits"],
                    h=h, r=r, t=t, gold=h,
                    direction="head",
                    rel2tails=None,
                    rel2heads=rel2heads if cfg.use_type_constrain else None,
                    sr2o=None,
                    tr2h=tr2h,
                )
                loss_head = F.cross_entropy(logits_head, h)




                # ===== 3) 三种单模态：tail + head 两个方向 =====
                # --- struct-only ---
                out_t_struct = model._score_with_table(struct_tab, h, r, targets=t, full_rank=True)
                loss_tail_struct = out_t_struct["loss"]

                out_h_struct = model._score_with_table(struct_tab, t, r_inv, targets=h, full_rank=True)
                loss_head_struct = out_h_struct["loss"]

                # --- image-only ---
                out_t_img = model._score_with_table(img_tab, h, r, targets=t, full_rank=True)
                loss_tail_img = out_t_img["loss"]

                out_h_img = model._score_with_table(img_tab, t, r_inv, targets=h, full_rank=True)
                loss_head_img = out_h_img["loss"]

                # --- text-only ---
                out_t_txt = model._score_with_table(txt_tab, h, r, targets=t, full_rank=True)
                loss_tail_txt = out_t_txt["loss"]

                out_h_txt = model._score_with_table(txt_tab, t, r_inv, targets=h, full_rank=True)
                loss_head_txt = out_h_txt["loss"]

                # tail / head 三模态汇总
                loss_modal_tail = loss_tail_struct + loss_tail_img + loss_tail_txt
                loss_modal_head = loss_head_struct + loss_head_img + loss_head_txt

                # ===== 4) MoE 辅助项（负载均衡 + 熵正则，可选）=====
                aux = getattr(model.moe, "last_aux", None)
                aux_lb = torch.tensor(0.0, device=h.device)
                aux_ent = torch.tensor(0.0, device=h.device)
                if aux is not None:
                    aux_lb = (aux["prob_mean"] * aux["prob_mean"]).sum() * 0.005   # load balance
                    aux_ent = (- aux["entropy"]) * 0.0005                          # 增加熵 → 正则项取负号

                # ===== 5) 轻量对比学习（图/文对齐，可选）=====
                ent_emb_img = model.img_proj(model.image_raw[t])   # [B, D]
                ent_emb_txt = model.txt_proj(model.text_raw[t])    # [B, D]
                loss_ctr = contrastive_loss(ent_emb_img, ent_emb_txt) * 0.05

                # ===== 6) 总 loss：fused 双向 + 三模态双向 + aux + ctr =====
                #   λ 可以先设小一点，防止扰动太大
                lambda_modal = 0.2
                loss = (
                    loss_tail + loss_head                         # 主干 fused 双向
                    + lambda_modal * (loss_modal_tail + loss_modal_head)  # 三模态 tail+head 辅助
                    + aux_lb + aux_ent                            # MoE 正则
                    + loss_ctr                                    # 图文对比
                )

            scaler.scale(loss).backward()
            if cfg.clip_grad and cfg.clip_grad > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad)
            scaler.step(opt); scaler.update()

            batch_sz = h.size(0)
            running_loss += loss.item() * batch_sz
            seen += batch_sz

            # EMA 平滑显示更稳定
            cur = loss.item()
            ema_loss = cur if ema_loss is None else (0.9 * ema_loss + 0.1 * cur)

            pbar.set_postfix({
                "loss":    f"{ema_loss:.4f}",
                "tail_f":  f"{loss_tail.item():.4f}",
                "head_f":  f"{loss_head.item():.4f}",
                "tail_s":  f"{loss_tail_struct.item():.4f}",
                "head_s":  f"{loss_head_struct.item():.4f}",
                "tail_i":  f"{loss_tail_img.item():.4f}",
                "head_i":  f"{loss_head_img.item():.4f}",
                "tail_t":  f"{loss_tail_txt.item():.4f}",
                "head_t":  f"{loss_head_txt.item():.4f}",
                "ctr":     f"{loss_ctr.item():.4f}",
                "lr":      f"{lr_now:.2e}",
            })
        pbar.close()

        print("\n========== FINAL BEST OVER ALL EPOCHS (by FINAL metrics) ==========")
        print(f"Best FINAL Hit@1:  {state.best_final_metrics['hit@1']:.4f}")
        print(f"Best FINAL Hit@3:  {state.best_final_metrics['hit@3']:.4f}")
        print(f"Best FINAL Hit@10: {state.best_final_metrics['hit@10']:.4f}")
        print(f"Best FINAL MRR:    {state.best_final_metrics['MRR']:.4f}")
        print("==============================================================\n")
        
        logging.info("\n========== FINAL BEST OVER ALL EPOCHS (by FINAL metrics) ==========")
        logging.info(f"Best FINAL Hit@1:  {state.best_final_metrics['hit@1']:.4f}")
        logging.info(f"Best FINAL Hit@3:  {state.best_final_metrics['hit@3']:.4f}")
        logging.info(f"Best FINAL Hit@10: {state.best_final_metrics['hit@10']:.4f}")
        logging.info(f"Best FINAL MRR:    {state.best_final_metrics['MRR']:.4f}")
        logging.info("==============================================================\n")

        avg_loss = running_loss / max(seen, 1)
        print(f"[Epoch {epoch}] train_loss={avg_loss:.6f}")
        logging.info(f"[Epoch {epoch}] train_loss={avg_loss:.6f}")

        # ====== 评估（双向 + FINAL）======
        metrics_bi = evaluate_bidir(
            model, test_loader, device=device,
            sr2o=sr2o, tr2h=tr2h,
            rel2tails=rel2tails, rel2heads=rel2heads,  # ★ 用真正的 type constrain
            dump_dir=preds_dir, epoch=epoch,             # ← 新增
            id2ent=id2ent, id2rel=id2rel,                # ← 新增（可读性）
            topk_dump=10,                                # ← 可改
        )
        tail_m  = metrics_bi["tail"]
        head_m  = metrics_bi["head"]
        final_m = metrics_bi["final"]

        print(f"[Epoch {epoch}] Eval (TAIL ): Hit@1={tail_m['hit@1']:.4f}  Hit@3={tail_m['hit@3']:.4f}  Hit@10={tail_m['hit@10']:.4f}  MRR={tail_m['MRR']:.4f}")
        print(f"[Epoch {epoch}] Eval (HEAD ): Hit@1={head_m['hit@1']:.4f}  Hit@3={head_m['hit@3']:.4f}  Hit@10={head_m['hit@10']:.4f}  MRR={head_m['MRR']:.4f}")
        print(f"[Epoch {epoch}] Eval (FINAL): Hit@1={final_m['hit@1']:.4f}  Hit@3={final_m['hit@3']:.4f}  Hit@10={final_m['hit@10']:.4f}  MRR={final_m['MRR']:.4f}")

        logging.info(f"[Epoch {epoch}] Eval (TAIL ): Hit@1={tail_m['hit@1']:.4f}  Hit@3={tail_m['hit@3']:.4f}  Hit@10={tail_m['hit@10']:.4f}  MRR={tail_m['MRR']:.4f}")
        logging.info(f"[Epoch {epoch}] Eval (HEAD ): Hit@1={head_m['hit@1']:.4f}  Hit@3={head_m['hit@3']:.4f}  Hit@10={head_m['hit@10']:.4f}  MRR={head_m['MRR']:.4f}")
        logging.info(f"[Epoch {epoch}] Eval (FINAL): Hit@1={final_m['hit@1']:.4f}  Hit@3={final_m['hit@3']:.4f}  Hit@10={final_m['hit@10']:.4f}  MRR={final_m['MRR']:.4f}")


        # ====== 更新全程最优（逐项取最大，基于 FINAL）======
        for k in state.best_final_metrics.keys():
            if final_m[k] > state.best_final_metrics[k]:
                state.best_final_metrics[k] = final_m[k]

        # ====== 早停：连续 5 个 epoch FINAL Hit@1 未提升 ======
        if final_m["hit@1"] > state.best_final_hit1:
            state.best_final_hit1 = final_m["hit@1"]
            state.patience = 0
            improved = True
        else:
            state.patience += 1
            improved = False

        # ====== 可选：保存 ckpt（受开关控制）======
        state.epoch = epoch
        if cfg.save_ckpt:
            os.makedirs(cfg.ckpt_dir, exist_ok=True)
            last_ckpt = os.path.join(cfg.ckpt_dir, "last.pt")
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "metrics": metrics_bi,
                "best_final_hit1": state.best_final_hit1,
                "best_final_metrics": state.best_final_metrics,
                "patience": state.patience,
                "cfg": cfg.__dict__,
            }, last_ckpt)
            if improved:
                torch.save({
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "opt": opt.state_dict(),
                    "metrics": metrics_bi,
                    "best_final_hit1": state.best_final_hit1,
                    "best_final_metrics": state.best_final_metrics,
                    "patience": state.patience,
                    "cfg": cfg.__dict__,
                }, os.path.join(cfg.ckpt_dir, "best.pt"))
                print(f"[Epoch {epoch}] ✔ BEST FINAL Hit@1 updated to {state.best_final_hit1:.4f}")
                logging.info(f"[Epoch {epoch}] ✔ BEST FINAL Hit@1 updated to {state.best_final_hit1:.4f}")
            else:
                print(f"[Epoch {epoch}] (no FINAL Hit@1 improvement) patience={state.patience}/5")
                logging.info(f"[Epoch {epoch}] (no FINAL Hit@1 improvement) patience={state.patience}/5")
        else:
            # 不保存 ckpt 也给个早停日志
            if improved:
                print(f"[Epoch {epoch}] ✔ BEST FINAL Hit@1 updated to {state.best_final_hit1:.4f}")
                logging.info(f"[Epoch {epoch}] ✔ BEST FINAL Hit@1 updated to {state.best_final_hit1:.4f}")
            else:
                print(f"[Epoch {epoch}] (no FINAL Hit@1 improvement) patience={state.patience}/5")
                logging.info(f"[Epoch {epoch}] (no FINAL Hit@1 improvement) patience={state.patience}/5")

        # # ====== 提前停止 ======
        # if state.patience >= 5:
        #     print(f"[Early Stop] FINAL Hit@1 has not improved for 5 consecutive epochs. Stop training.")
        #     logging.info(f"[Early Stop] FINAL Hit@1 has not improved for 5 consecutive epochs. Stop training.")
        #     break

        # ====== （可选）每个 epoch 末尾打印“当前为止的全程最优 FINAL” ======
        print("\n[So far BEST FINAL over all epochs]")
        print(f"Hit@1={state.best_final_metrics['hit@1']:.4f}  "
            f"Hit@3={state.best_final_metrics['hit@3']:.4f}  "
            f"Hit@10={state.best_final_metrics['hit@10']:.4f}  "
            f"MRR={state.best_final_metrics['MRR']:.4f}\n")
        
        logging.info("\n[So far BEST FINAL over all epochs]")
        logging.info(f"Hit@1={state.best_final_metrics['hit@1']:.4f}  "
            f"Hit@3={state.best_final_metrics['hit@3']:.4f}  "
            f"Hit@10={state.best_final_metrics['hit@10']:.4f}  "
            f"MRR={state.best_final_metrics['MRR']:.4f}\n")
        
        scheduler.step()
        lr_now = opt.param_groups[0]["lr"]   # 让日志里显示最新 lr


if __name__ == "__main__":   #DB15K  MKG-W  MKG-Y  Kuai16K   #Kuai16K-visual.pth
    cfg = Config(
        entity2id_path="/home/nlp/kgc_data/example_data/structure/benchmarks/MKG-W/entity2id.txt",
        relation2id_path="/home/nlp/kgc_data/example_data/structure/benchmarks/MKG-W/relation2id.txt",
        train_path="/home/nlp/kgc_data/example_data/structure/benchmarks/MKG-W/train2id.txt",
        test_path="/home/nlp/kgc_data/example_data/structure/benchmarks/MKG-W/test2id.txt",
        image_pth="/home/nlp/kgc_data/example_data/images/MKG-W-visual.pth",
        text_pth="/home/nlp/kgc_data/example_data/text/MKG-W-textual.pth",
        type_constrain_path="/home/nlp/kgc_data/example_data/structure/benchmarks/MKG-W/type_constrain.txt",
        use_type_constrain=True,
        n_embd=512,
        num_experts=4,
        top_k=2,
        lr=1e-3,
        weight_decay=0.01,
        batch_size=1024,
        num_workers=2,
        max_epochs=1000,
        clip_grad=1.0,
        ckpt_dir="ckpts",
        eval_full_rank=True,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    main(cfg)
