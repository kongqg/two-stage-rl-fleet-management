# conflict_metrics.py
"""
1. Same-Neighbor Fill-In Conflict (`same_grid_contention_step`)
   *Definition:* The number of orders for which a neighbor originally had idle vehicles, but those vehicles were exhausted by earlier requests—forcing these orders to be fulfilled by other neighbors.

2. Total Conflict (`unserved_demand_step`)
   *Definition:* The total number of orders that remain unassigned after all neighbor requests have failed.
"""
class ConflictMetrics:

    def __init__(self):
        # step-level
        self.same_grid_contention_step = 0
        self.unserved_demand_step      = 0
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
        """ (same_grid_contention_step, unserved_demand_step)"""
        return self.same_grid_contention_step, self.unserved_demand_step

    def get_total(self):
        """ (same_grid_contention_total, unserved_demand_total)"""
        return self.same_grid_contention_total, self.unserved_demand_total
