import random
import sys
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "WeightPolicyNet", "NeighborStateProcessor",
    "Stage2ActorReplay", "NeighborAgentW","Stage2ValueReplay"
]
# ─────────────── 网络 ────────────────
class WeightPolicyNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int = 6):
        super().__init__()
        self.l1 = nn.Linear(state_dim, 128)
        self.l2 = nn.Linear(128, 64)
        self.l3 = nn.Linear(64, 32)
        self.out = nn.Linear(32, action_dim)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def features(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        return x


    def logits(self, x):
        h = self.features(x)
        return self.out(h)

    def forward(self, x):
        logits = self.logits(x)
        return torch.softmax(logits, dim=-1)

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
        return self.out(x)
class NeighborStateProcessor:
    def __init__(self, env, n_neighbors: int = 6, hist_len: int = 3):
        self.env  = env
        self.n    = env.n_valid_grids
        self.k    = n_neighbors
        self.hist_len = hist_len
        self._hist = []
        self.SCALE = 20.0            # Normalizing order counts / idle vehicle counts
        self.HIST_SCALE = 500

    # 新增
    def get_state_for_source(self, s_id_node, remain_vec, idle_snap, conflict_step):
        # remain_vec: G 维 pending
        # Neighbor Idle: Take 6 neighbors of s_id (fill with 0 if invalid).
        nb_ids = []
        nb_idle_driver = []
        for neigh in s_id_node.neighbors:  # 可能有 None
            if neigh is None:
                nb_ids.append(-1)
                nb_idle_driver.append(0)
            else:
                nid = neigh.get_node_index()
                nb_ids.append(nid)
                nb_idle_driver.append(idle_snap.get(nid, 0))
        nb = np.asarray(nb_idle_driver, np.float32) / self.SCALE

        self._hist.append(conflict_step)
        if len(self._hist) > self.hist_len:
            self._hist.pop(0)

        hist_padded = [0.0] * (self.hist_len - len(self._hist)) + self._hist
        hist_arr = np.asarray(hist_padded, dtype=np.float32) / self.HIST_SCALE
        e_s = np.zeros(self.n, np.float32);
        e_s[s_id_node.get_node_index()] = 1.0

        return np.concatenate([remain_vec.astype(np.float32) / self.SCALE,
                               nb, hist_arr, e_s])

def _grad_global_norm(module) -> float:
    total = 0.0
    for p in module.parameters():
        if p.grad is not None:
            g = p.grad.detach()
            total += float(g.norm(2).item() ** 2)
    return total ** 0.5
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

    def masked_softmax(self,logits, mask):
        # Mask: 6 dimensions {0/1}
        logits = logits + (mask == 0) * (-1e9)
        return torch.softmax(logits, dim=-1)
    @torch.no_grad()
    def action(self, state_s_np, mask_np, eps=0.0, select='argmax'):
        # 1020
        s = torch.from_numpy(state_s_np).float().unsqueeze(0).to(self.device)
        logits = self.actor.logits(s)
        mask = torch.from_numpy(mask_np).to(self.device)  # 6 维
        prob = self.masked_softmax(logits, mask).squeeze(0)
        if np.random.rand() < eps:
            prob = 0.5 * prob + 0.5 * torch.from_numpy(np.random.dirichlet(np.ones(6))).to(prob)
            prob = prob / prob.sum()
        if select == 'argmax':
            order = torch.argsort(prob, descending=True).cpu().numpy().tolist()
        else:
            order = torch.multinomial(prob, num_samples=6, replacement=False).cpu().numpy().tolist()
        return prob.cpu().numpy(), order

    def update_value(self, states, rewards, next_states, gamma=0.95):
        """Critic；(s, r, s') -> MSE[ r + γV(s') - V(s) ]"""
        s = torch.from_numpy(states).float().to(self.device)  # [B, D]
        ns = torch.from_numpy(next_states).float().to(self.device)  # [B, D]
        r = torch.from_numpy(rewards).float().to(self.device)  # [B]

        v = self.critic(s).squeeze(-1)
        with torch.no_grad():
            v_next = self.critic(ns).squeeze(-1)

        td = r + gamma * v_next - v
        loss_c = (td ** 2).mean()

        self.opt_c.zero_grad(set_to_none=True)
        loss_c.backward()
        if hasattr(self, "grad_clip") and self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)

        self.opt_c.step()

        return loss_c
    def update_policy(self, states, attempts, masks, rewards, next_states,
                      gamma=0.95, beta=0.01):
        """
        Actor; Advantage = r + γV(s') - V(s)
        """
        eps = 1e-8
        s  = torch.from_numpy(states).float().to(self.device)
        ns = torch.from_numpy(next_states).float().to(self.device)
        r  = torch.from_numpy(rewards).float().to(self.device)

        with torch.no_grad():
            v      = self.critic(s).squeeze(-1)
            v_next = self.critic(ns).squeeze(-1)
            adv    = r + gamma * v_next - v  # [B]
            adv = adv * 2.0

        logliks, entropies, lens = [], [], []
        B = s.size(0)
        for i in range(B):
            s_i = s[i:i+1]                       # [1, D]
            seq = attempts[i]                    # List[int]
            ms  = masks[i]                       # List[np.ndarray(6,)]
            lens.append(len(seq))

            if len(seq) == 0:
                logliks.append(torch.tensor(0.0, device=self.device))
                entropies.append(torch.tensor(0.0, device=self.device))
                continue

            loglik_i, entropy_i = 0.0, 0.0
            for t, a_t in enumerate(seq):
                logits_t = self.actor.logits(s_i)                  # [1,6]
                mask_t_np = ms[t]
                mask_t = (torch.from_numpy(mask_t_np).float().to(self.device)
                          if not isinstance(mask_t_np, torch.Tensor)
                          else mask_t_np.float().to(self.device))
                mask_t = mask_t.unsqueeze(0)                       # [1,6]
                masked_logits = logits_t + (mask_t == 0).float() * (-1e9)
                prob_t = torch.softmax(masked_logits, dim=-1)      # [1,6]
                p_at  = prob_t[0, int(a_t)]
                loglik_i  = loglik_i + torch.log(p_at + eps)
                entropy_i = entropy_i - (prob_t * torch.log(prob_t + eps)).sum(dim=-1).squeeze(0)
            logliks.append(loglik_i)
            entropies.append(entropy_i)

        logliks   = torch.stack(logliks,   dim=0)  # [B]
        entropies = torch.stack(entropies, dim=0)  # [B]
        loss_a = -(adv * logliks).mean() - beta * entropies.mean()

        self.opt_a.zero_grad(set_to_none=True)
        loss_a.backward()

        if hasattr(self, "grad_clip") and self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)

        self.opt_a.step()
        return loss_a


class Stage2ActorReplay:
    def __init__(self, capacity: int = 200_000):
        self.buf = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buf)

    def add(self, state_s, attempts, masks_seq, reward_s, next_state_s):
        state_s      = np.asarray(state_s, dtype=np.float32)
        next_state_s = np.asarray(next_state_s, dtype=np.float32)
        # masks 每步都是 shape=(6,) 的 0/1
        masks_seq = [np.asarray(m, dtype=np.int64).reshape(6) for m in masks_seq]
        attempts  = [int(a) for a in attempts]
        reward_s  = float(reward_s)
        self.buf.append((state_s, attempts, masks_seq, reward_s, next_state_s))

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, k=min(batch_size, len(self.buf)))
        states, attempts, masks, rewards, next_states = [], [], [], [], []
        for s, aseq, mseq, r, ns in batch:
            states.append(s)
            attempts.append(aseq)
            masks.append(mseq)
            rewards.append(r)
            next_states.append(ns)
        return (np.stack(states, axis=0).astype(np.float32),   # [B, D2]
                attempts,                                      # List[List[int]] (变长)
                masks,                                         # List[List[np.ndarray(6)]]
                np.asarray(rewards, dtype=np.float32),         # [B]
                np.stack(next_states, axis=0).astype(np.float32))  # [B, D2]


class Stage2ValueReplay:
    def __init__(self, capacity: int = 200_000):
        from collections import deque
        self.buf = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buf)

    def add(self, state_s, reward_s, next_state_s):
        state_s      = np.asarray(state_s, dtype=np.float32)
        reward_s     = float(reward_s)
        next_state_s = np.asarray(next_state_s, dtype=np.float32)
        self.buf.append((state_s, reward_s, next_state_s))

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, k=min(batch_size, len(self.buf)))
        s, r, ns = zip(*batch)
        return (np.stack(s,  axis=0).astype(np.float32),
                np.asarray(r, dtype=np.float32),
                np.stack(ns, axis=0).astype(np.float32))