import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "WeightPolicyNet", "NeighborStateProcessor",
    "NeighborReplay", "NeighborAgentW"
]
# ─────────────── 网络 ────────────────
class WeightPolicyNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int = 6):
        super().__init__()
        self.l1 = nn.Linear(state_dim, 128)
        self.l2 = nn.Linear(128, 64)
        self.l3 = nn.Linear(64, 32)
        self.logits = nn.Linear(32, action_dim)

        # Xavier 初始化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 根据输入和输出维度自动计算一个合适的分布区间
                nn.init.xavier_uniform_(m.weight)
                # 将每个线性层的偏置向量初始化为全零，保证在训练一开始时，偏置对激活输出没有额外偏倚。
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        # softmax 做完再 clip，避免 0/1
        p = torch.softmax(self.logits(x), dim=1)
        return torch.clamp(p, 1e-6, 1.0 - 1e-6)

class ValueNet(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.l1 = nn.Linear(state_dim, 128)
        self.l2 = nn.Linear(128, 64)
        self.l3 = nn.Linear(64, 32)
        self.out = nn.Linear(32, 1)
    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        # return torch.tanh(self.out(x)) * 10
        return torch.relu(self.out(x))
class NeighborStateProcessor:
    """
    state =
        [remain_orders_i           (N)
         idle_neighbors_i1..ik     (N*K)
         sg_t , ud_t , sg_{t-1}, ud_{t-1}, ...]  (2*hist_len)
    """
    def __init__(self, env, n_neighbors: int = 6, hist_len: int = 3):
        self.env  = env
        self.n    = env.n_valid_grids
        self.k    = n_neighbors
        self.hist_len = hist_len
        self._hist = []
        self.SCALE = 20.0            # 用于归一化订单数 / 空车数

    def get_state(self,
                  remain_vec: np.ndarray,
                  idle_snap: dict,
                  conflict_step: tuple) -> np.ndarray:

        # ①  邻居 idle  (N,K)
        nb_mat = np.zeros((self.n, self.k), np.float32)
        for i in range(self.n):
            for j, nb_node_id in enumerate(self.env.valid_neighbor_node_id[i][:self.k]):
                nb_mat[i, j] = idle_snap.get(nb_node_id, 0)

        # 归一化
        remain_vec = remain_vec.astype(np.float32) / self.SCALE
        nb_mat     = nb_mat / self.SCALE

        # ②  冲突历史
        self._hist.append(conflict_step)
        if len(self._hist) > self.hist_len:
            self._hist.pop(0)
        hist_arr = np.array(
            ([(0, 0)] * (self.hist_len - len(self._hist)) + self._hist),
            np.float32
        ).flatten()


        # ③  拼接
        return np.concatenate((remain_vec,  # remain_vec:(5796,)
                               nb_mat.flatten(), # nb_mat:(40572,)
                               hist_arr)) # hist_arr:(6,)


# ─────────────── Agent ────────────────
class NeighborAgentW:
    def __init__(self, state_dim, action_dim=6, lr=1e-4,
                 rho_max: float = 2.0, device=None):
        self.device = torch.device(device or (
            "cuda" if torch.cuda.is_available() else "cpu"))
        self.actor  = WeightPolicyNet(state_dim, action_dim).to(self.device)
        self.critic = ValueNet(state_dim).to(self.device)
        self.opt_a  = torch.optim.AdamW(self.actor.parameters(),  lr=lr, eps=1e-5)
        self.opt_c  = torch.optim.AdamW(self.critic.parameters(), lr=lr, eps=1e-5)
        self.rho_max= rho_max

        from collections import defaultdict
        self.metrics = defaultdict(list)  # {name: [v1,v2,...]}

    # -------- 行为 --------
    @torch.no_grad()
    def action(self, s_np, eps=0.0):
        s_t = torch.from_numpy(s_np).float().unsqueeze(0).to(self.device)
        w_pred = self.actor(s_t).squeeze(0)      # (6,)
        w = w_pred.clone()
        if np.random.rand() < eps:               # ε‑greedy (Dirichlet noise)
            noise = torch.from_numpy(np.random.dirichlet(np.ones(6)).astype(np.float32)).to(self.device)
            w = 0.5 * w + 0.5 * noise
            w = w / w.sum()
        return w.cpu().numpy()

    # -------- 更新 --------
    def update(self,
               s_np: np.ndarray,
               w_np: np.ndarray,
               r_np: np.ndarray,
               s_next_np: np.ndarray,
               gamma: float = 0.95):

        s      = torch.from_numpy(s_np).float().to(self.device)
        w_true = torch.from_numpy(w_np).float().to(self.device)
        r = torch.from_numpy(r_np / 1000.0).float().to(self.device)
        s_next = torch.from_numpy(s_next_np).float().to(self.device)
        # logp_b = torch.from_numpy(logp_beh_np).float().to(self.device)

        # ---------- Critic ----------
        v      = self.critic(s).squeeze(1)
        with torch.no_grad():
            v_next = self.critic(s_next).squeeze(1)
        # td = torch.clamp(r + gamma * v_next - v, -10.0, 10.0)
        td = torch.clamp(r + gamma * v_next - v, -50.0, 50.0)
        loss_c = td.pow(2).mean()
        self.opt_c.zero_grad()
        loss_c.backward()
        grad_norm_c = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 2.0)
        self.opt_c.step()

        # ---------- Actor (IS) ----------
        # 归一化 w_true
        w_true = torch.clamp(w_true / (w_true.sum(1, keepdim=True) + 1e-8), 0.0, 1.0)
        # p = self.actor(s)
        # logp_t  = (w_true * torch.log(p)).sum(1)  # 目标策略 logπ_tgt
        # rho = torch.exp(logp_t - logp_b)
        # if self.rho_max > 0:
        #     rho = torch.clamp(rho, 0.0, self.rho_max)
        # adv = td.detach()
        # entropy = -(w_true * torch.log(p)).sum(1).mean()
        # loss_a = -(rho * logp_t * adv).mean() - 0.01 * entropy



        # ---------- Actor (On-Policy A2C) ----------
        p = self.actor(s)
        logp_t = (w_true * torch.log(p)).sum(1)
        adv = td.detach()
        entropy = -(w_true * torch.log(p)).sum(1).mean()
        # 去掉 IS，直接用 on-policy 策略梯度
        loss_a = -(logp_t * adv).mean() - 0.01 * entropy



        self.opt_a.zero_grad();
        loss_a.backward();
        grad_norm_a = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 2.0)
        self.opt_a.step()

        self.metrics["loss_actor"].append(loss_a.item())
        self.metrics["loss_critic"].append(loss_c.item())
        self.metrics["grad_actor"].append(grad_norm_a.item())
        self.metrics["grad_critic"].append(grad_norm_c.item())
        self.metrics["value_mean"].append(v.mean().item())

        return (loss_a.item(), loss_c.item(),
                grad_norm_a.item(), grad_norm_c.item(),
                v.mean().item())


class NeighborReplay:
    def __init__(self, cap: int, batch: int, state_dim: int, action_dim: int = 6):
        self.cap, self.batch = int(cap), int(batch)
        self.ptr = self.size = 0
        self.s   = np.zeros((cap, state_dim),  np.float32)
        self.w   = np.zeros((cap, action_dim), np.float32)
        self.r   = np.zeros((cap,),            np.float32)
        self.ns  = np.zeros((cap, state_dim),  np.float32)

    def add(self, s, w, r, ns):
        idx = self.ptr
        self.s[idx], self.w[idx], self.r[idx], self.ns[idx] = (
            s, w, r, ns)
        self.ptr  = (self.ptr + 1) % self.cap
        self.size = min(self.size + 1, self.cap)

    def sample(self):
        k   = min(self.batch, self.size)
        idx = np.random.randint(0, self.size, size=k)
        return (self.s[idx], self.w[idx], self.r[idx],
                self.ns[idx])
