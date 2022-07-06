import copy


class env:
    def __init__(self):
        self.game = []
        self.history_a, self.history_b = [], []
        self.init_a, self.init_b = [], []
        self.action_a, self.action_b = 0, 0
        self.reward_a, self.reward_b = 0, 0
        self.state, self.state_a, self.state_b = [], [], []
        self.done = False
        self.t, self.max_t = 0, 12

    def reset(self):
        self.t = 0
        self.done = False
        self.game = [[1, 1], [0, 8]], [[8, 0], [-1, -1]]
        self.history_a, self.history_b = [], []
        self.init_a, self.init_b = ['sw'], ['sw']
        return copy.deepcopy([self.init_a, self.init_b])

    def step(self, action_a, action_b):
        self.t += 1
        self.history_a.append(['sw', 'st'][action_a])
        self.history_b.append(['sw', 'st'][action_b])
        self.state = [self.history_a, self.history_b]
        self.reward_a = self.reward(['sw', 'st'][action_a], ['sw', 'st'][action_b])[0]
        self.reward_b = self.reward(['sw', 'st'][action_a], ['sw', 'st'][action_b])[1]
        self.done = self.is_done()
        return copy.deepcopy(self.state), copy.deepcopy(self.reward_a), copy.deepcopy(self.done)

    def step_sp(self, action_a, action_b):
        self.t += 1
        self.history_a.append(['sw', 'st'][action_a])
        self.history_b.append(['sw', 'st'][action_b])
        self.state_a = [self.history_a, self.history_b]
        self.state_b = [self.history_b, self.history_a]
        self.reward_a = self.reward(['sw', 'st'][action_a], ['sw', 'st'][action_b])[0]
        self.reward_b = self.reward(['sw', 'st'][action_a], ['sw', 'st'][action_b])[1]
        self.done = self.is_done()
        return copy.deepcopy(self.state_a), copy.deepcopy(self.state_b), copy.deepcopy(self.reward_a), copy.deepcopy(
            self.reward_b), copy.deepcopy(self.done)

    def reward(self, action_a, action_b):
        self.action_a = 0 if action_a == "sw" else 1
        self.action_b = 0 if action_b == "sw" else 1
        return copy.deepcopy([self.game[self.action_a][self.action_b][0], self.game[self.action_a][self.action_b][1]])

    def is_done(self):
        return True if self.t == self.max_t else False
