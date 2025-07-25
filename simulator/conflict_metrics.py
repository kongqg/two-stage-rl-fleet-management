# conflict_metrics.py
class ConflictMetrics:
    """统一管理冲突指标（同邻居抢占 & 无邻居可补）"""

    def __init__(self):
        # step-level
        self.same_grid_contention_step = 0   # 同邻居竞争
        self.unserved_demand_step      = 0   # 所有邻居皆无车
        # accumulated
        self.same_grid_contention_total = 0
        self.unserved_demand_total      = 0

    # ---------- 计数 ----------
    def reset_step(self):
        self.same_grid_contention_step = 0
        self.unserved_demand_step      = 0
        # self.unserved_demand_total     = 0
        # self.unserved_demand_total     = 0


    def add_same_grid_contention(self, n: int):
        self.same_grid_contention_step  += n
        self.same_grid_contention_total += n

    def add_unserved_demand(self, n: int):
        self.unserved_demand_step  += n
        self.unserved_demand_total += n

    # ---------- 读取 ----------
    def get_step(self):
        """返回 (same_grid_contention_step, unserved_demand_step)"""
        return self.same_grid_contention_step, self.unserved_demand_step

    def get_total(self):
        """返回 (same_grid_contention_total, unserved_demand_total)"""
        return self.same_grid_contention_total, self.unserved_demand_total
