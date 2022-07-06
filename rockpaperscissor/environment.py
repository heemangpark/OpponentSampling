import copy


class env:
    def __init__(self):
        self.game = [[0, 0], [-1, 1], [1, -1]], [[1, -1], [0, 0], [-1, 1]], [[-1, 1], [1, -1], [0, 0]]
        self.history_a = []
        self.history_b = []
        self.action_a = ""
        self.action_b = ""
        self.state = []
        self.reward_a = 0
        self.reward_b = 0
        self.done = False
        self.t = 0
        self.maxt = 12

    def reset(self):
        self.t = 0
        self.done = False
        self.game = [[0, 0], [-1, 1], [1, -1]], [[1, -1], [0, 0], [-1, 1]], [[-1, 1], [1, -1], [0, 0]]
        self.history_a, self.history_b = ['rock'], ['rock']
        return copy.deepcopy([self.history_a, self.history_b])

    def step(self, action_a, action_b):
        self.t += 1
        self.history_a.append(['rock', 'paper', 'scissor'][action_a])
        self.history_b.append(['rock', 'paper', 'scissor'][action_b])
        self.state = [self.history_a, self.history_b]
        self.reward_a = self.reward(['rock', 'paper', 'scissor'][action_a], ['rock', 'paper', 'scissor'][action_b])[0]
        self.reward_b = self.reward(['rock', 'paper', 'scissor'][action_a], ['rock', 'paper', 'scissor'][action_b])[1]
        self.done = self.is_done()
        return copy.deepcopy(self.state), self.reward_a, self.done

    def reward(self, action_a, action_b):
        if action_a == 'rock':
            self.action_a = 0
        elif action_a == 'paper':
            self.action_a = 1
        else:
            self.action_a = 2
        if action_b == 'rock':
            self.action_b = 0
        elif action_b == 'paper':
            self.action_b = 1
        else:
            self.action_b = 2
        reward_a = self.game[self.action_a][self.action_b][0]
        reward_b = self.game[self.action_a][self.action_b][1]
        return [reward_a, reward_b]

    def is_done(self):
        return True if self.t == self.maxt else False
