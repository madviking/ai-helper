class CostTracker:
    def __init__(self):
        self.total_cost = 0

    def track_cost(self, cost, output_tokens, model):
        self.total_cost += cost

    def get_total_cost(self):
        return self.total_cost

    def reset_cost(self):
        self.total_cost = 0
