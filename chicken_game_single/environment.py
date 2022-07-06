import copy


class env:
    def __init__(self):
        self.game = [[1, 1], [0, 8]], [[8, 0], [-1, -1]]
        self.reward_a, self.reward_b = 0, 0

    def step(self, action_a, action_b):
        self.reward_a = self.game[action_a][action_b][0]
        self.reward_b = self.game[action_a][action_b][1]
        return copy.deepcopy(self.reward_a), copy.deepcopy(self.reward_b)
