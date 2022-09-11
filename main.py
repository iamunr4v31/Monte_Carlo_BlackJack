import gym
from agent import Agent

if __name__ == "__main__":
    env = gym.make("Blackjack-v1")
    agent = Agent()
    n_episodes = 50000
    for eps in range(1, n_episodes + 1):
        state, _ = env.reset()
        # print(state)
        # print(env.step(env.action_space.sample()))
        done = False
        while not done:
            action = agent.policy(state)
            next_state, reward, done, _, _ = env.step(action)
            agent.memory.append((state, reward))
            state = next_state
        agent.update_V()
        if eps % 1000 == 0:
            print(f"{eps} episodes done")
    print(agent.V[21 - 4, 3 - 1, 1])
    print(agent.V[4 - 4, 1 - 1, 0])
