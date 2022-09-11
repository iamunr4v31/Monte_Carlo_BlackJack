from mc_control_agent import Agent
import gym
import matplotlib.pyplot as plt

if __name__ == "__main__":
    env = gym.make("Blackjack-v1")
    agent = Agent(eps=0.001)
    n_episodes = 500000
    history_dict = {-1: 0, 0: 0, 1: 0}
    win_rates = []
    for i in range(1, n_episodes + 1):
        if i % 1000 == 0:
            print(f"{i} episodes done")
            win_rates.append(history_dict[1] / i)
            print(f"Win %: {win_rates[-1] if win_rates else 0}")
        observation, _ = env.reset()

        done = False
        while not done:
            action = agent.choose_action(observation)
            next_state, reward, done, _, _ = env.step(action)
            agent.memory.append((observation, action, reward))
            observation = next_state

        agent.update_Q()
        if reward == 1:
            history_dict[1] += 1
        elif reward == 0:
            history_dict[0] += 1
        else:
            history_dict[-1] += 1
    plt.plot(win_rates)
    plt.show()
