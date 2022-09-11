import numpy as np


class Agent:
    def __init__(self, gamma=0.99):
        self.gamma = gamma

        self.sum_space = list(range(4, 22))
        self.dealer_show_card_space = list(range(1, 11))
        self.ace_space = [False, True]
        self.action_space = [0, 1]  # 0: stick, 1: hit

        self.V = np.zeros(
            (len(self.sum_space), len(self.dealer_show_card_space), len(self.ace_space))
        )
        self.returns = {}
        self.init_returns()
        self.states_visited = np.zeros_like(
            self.V, dtype="bool"
        )  # whether or not it is a first-visit to the state

        self.state_space = [
            (x, y, z)
            for x in self.sum_space
            for y in self.dealer_show_card_space
            for z in self.ace_space
        ]

        self.memory = []

        # self.init_vals()

    def init_returns(self):
        for total in range(len(self.sum_space)):
            for dealer_show_card in range(len(self.dealer_show_card_space)):
                for ace in range(len(self.ace_space)):
                    self.returns[(total, dealer_show_card, ace)] = []

    def policy(self, state):
        total, _, _ = state
        return 0 if total >= 20 else 1

    def update_V(self):
        for idx, (state, _) in enumerate(self.memory):
            G = 0
            cur_sum, cur_dealer_show_card, cur_ace = state
            if not self.states_visited[cur_sum - 4][cur_dealer_show_card - 1][
                int(cur_ace)
            ]:
                self.states_visited[cur_sum - 4][cur_dealer_show_card - 1][
                    int(cur_ace)
                ] = True
                for t, (_, reward_t) in enumerate(self.memory[idx:]):
                    G += self.gamma**t * reward_t
                # print((cur_sum - 4, cur_dealer_show_card - 1, int(cur_ace)))
                # print(self.returns.keys())
                    self.returns[
                    (cur_sum - 4, cur_dealer_show_card - 1, int(cur_ace))
                ].append(
                    G
                )  # check and update these lines since replaced dictionary with numpy array
        for state, _ in self.memory:
            cur_sum, cur_dealer_show_card, cur_ace = state
            self.V[cur_sum - 4][cur_dealer_show_card - 1][int(cur_ace)] = np.mean(
                self.returns[(cur_sum - 4, cur_dealer_show_card - 1, int(cur_ace))]
            )

        self.states_visited.fill(False)
        self.memory.clear()
