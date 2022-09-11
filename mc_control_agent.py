import numpy as np


class Agent:
    def __init__(self, eps=0.1, gamma=0.99) -> None:

        self.sum_space = list(range(4, 22))
        self.dealer_show_card_space = list(range(1, 11))
        self.ace_space = [False, True]
        self.action_space = [0, 1]  # 0: stick, 1: hit

        self.eps = eps
        self.gamma = gamma

        self.state_space = np.array(
            [
                (x, y, z)
                for x in self.sum_space
                for y in self.dealer_show_card_space
                for z in self.ace_space
            ]
        )
        self.Q = np.zeros(
            (
                len(self.sum_space),
                len(self.dealer_show_card_space),
                len(self.ace_space),
                len(self.action_space),
            )
        )

        self.pairs_visited = np.zeros_like(self.Q, dtype="bool")
        self.policy = np.ones(
            (
                len(self.sum_space),
                len(self.dealer_show_card_space),
                len(self.ace_space),
                len(self.action_space),
            ),
            dtype="float32",
        )
        for a in self.action_space:
            self.policy[:, :, :, a] /= len(self.action_space)
        # print(np.sum(self.policy))
        self.memory = []
        self.returns = {}
        self.init_returns()

    def init_returns(self):
        for total in range(len(self.sum_space)):
            for dealer_show_card in range(len(self.dealer_show_card_space)):
                for ace in range(len(self.ace_space)):
                    for action in self.action_space:
                        self.returns[(total, dealer_show_card, int(ace)), action] = []

    def choose_action(self, state):
        total, dealer_show_card, ace = state
        # print(self.policy[total - 4, dealer_show_card - 1, int(ace)])
        action = np.random.choice(
            self.action_space, p=self.policy[total - 4, dealer_show_card - 1, int(ace)]
        )
        return action

    def update_Q(self):
        for idx, (state, action, _) in enumerate(self.memory):
            G = 0
            total, dealer_show_card, ace = state
            if not self.pairs_visited[
                total - 4, dealer_show_card - 1, int(ace), action
            ]:
                self.pairs_visited[total - 4, dealer_show_card - 1, int(ace)] = 1
                for t, (_, _, reward_t) in enumerate(self.memory[idx:]):
                    G += self.gamma**t * reward_t
                    self.returns[
                        (total - 4, dealer_show_card - 1, int(ace)), action
                    ].append(G)

        for state, action, _ in self.memory:
            total, dealer_show_card, ace = state
            self.Q[total - 4, dealer_show_card - 1, int(ace), action] = np.mean(
                self.returns[(total - 4, dealer_show_card - 1, int(ace)), action]
            )
            self.update_policy(state)

        self.pairs_visited.fill(False)
        self.memory.clear()

    def update_policy(self, state):
        total, dealer_show_card, ace = state
        a_max = np.argmax(
            self.Q[total - 4, dealer_show_card - 1, int(ace), a]
            for a in self.action_space
        )
        n_actions = len(self.action_space)
        self.policy[total - 4, dealer_show_card - 1, int(ace), :] = self.eps / n_actions
        self.policy[total - 4, dealer_show_card - 1, int(ace), a_max] = (
            1 - self.eps + self.eps / n_actions
        )
        # for action in self.action_space:
        #     self.policy[total - 4, dealer_show_card - 1, int(ace), action] = (
        #         1 - self.eps + self.eps / n_actions
        #         if action == a_max
        #         else self.eps / n_actions
        #     )
