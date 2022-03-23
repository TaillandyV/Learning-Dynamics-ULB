from typing import Tuple, Optional

from matplotlib import pyplot as plt

import gym
import numpy as np
from gym import Env
from numpy import ndarray

from agent import QLearnerAgent

def run_episode(env: Env, agent: QLearnerAgent, training: bool, gamma) -> float:
    """
    Interact with the environment for one episode using actions derived from the q_table and the action_selector.

    :param env: The gym environment.
    :param agent: The agent.
    :param training: If true the q_table will be updated using q-learning. The flag is also passed to the action selector.
    :param gamma: The discount factor.
    :return: The cumulative discounted reward.
    """
    done = False
    obs = env.reset()
    cum_reward = 0.
    t = 0
    while not done:
        action = agent.act(obs, training)
        new_obs, reward, done, _ = env.step(action)
        if training:
            agent.learn(obs, action, reward, done, new_obs)
        obs = new_obs
        cum_reward += gamma ** t * reward
        t += 1
    return cum_reward


def train(env: Env, gamma: float, num_episodes: int, evaluate_every: int, num_evaluation_episodes: int,
          alpha: float, epsilon_max: Optional[float] = None, epsilon_min: Optional[float] = None,
          epsilon_decay: Optional[float] = None) -> Tuple[QLearnerAgent, ndarray, ndarray]:
    """
    Training loop.
    
    :param env: The gym environment.
    :param gamma: The discount factor.
    :param num_episodes: Number of episodes to train.
    :param evaluate_every: Evaluation frequency.
    :param num_evaluation_episodes: Number of episodes for evaluation.
    :param alpha: Learning rate.
    :param epsilon_max: The maximum epsilon of epsilon-greedy.
    :param epsilon_min: The minimum epsilon of epsilon-greedy.
    :param epsilon_decay: The decay factor of epsilon-greedy.
    :return: Tuple containing the agent, the returns of all training episodes and averaged evaluation return of
            each evaluation.
    """
    digits = len(str(num_episodes))
    agent = QLearnerAgent(env.observation_space.n, env.action_space.n, alpha, gamma, epsilon_max,
                          epsilon_min, epsilon_decay)
    evaluation_returns = np.zeros(num_episodes // evaluate_every)
    returns = np.zeros(num_episodes)
    for episode in range(num_episodes):
        returns[episode] = run_episode(env, agent, True, gamma)

        if (episode + 1) % evaluate_every == 0:
            evaluation_step = episode // evaluate_every
            cum_rewards_eval = np.zeros(num_evaluation_episodes)
            for eval_episode in range(num_evaluation_episodes):
                cum_rewards_eval[eval_episode] = run_episode(env, agent, False, gamma)
            evaluation_returns[evaluation_step] = np.mean(cum_rewards_eval)
            print(f"Episode {(episode + 1): >{digits}}/{num_episodes:0{digits}}:\t"
                  f"Averaged evaluation return {evaluation_returns[evaluation_step]:0.3}")
    return agent, returns, evaluation_returns


if __name__ == '__main__':
    env = gym.make('FrozenLake-v1')

    nb_runs=20
    num_episodes=30000
    evaluate_every=1000

    average_returns=np.zeros(num_episodes)
    average_eval_return=np.zeros((num_episodes//evaluate_every))
    std_eval_return=np.zeros(((num_episodes//evaluate_every),nb_runs))
    
    best_actions=[]
    
    for j in range (nb_runs):
        trained,returns,eval_return=train(env,1.,num_episodes,evaluate_every,32,0.0165,1.,0.,0.99974)
        best_action=np.chararray((trained.q_table.shape[0],1),5,True)
        holes_layout={5,7,11,12}

        for i in range(trained.q_table.shape[0]):
            if(i in holes_layout):
                action_string="Hole"
            elif(i==trained.q_table.shape[0]-1):
                action_string="End"
            else:
                action=trained.q_table[i].argmax()
                if(action == 0):
                    action_string="Left"
                elif(action == 1):
                    action_string="Down"
                elif(action == 2):
                    action_string="Right"
                else:
                    action_string="Up"
            best_action[i]=action_string
        best_action=np.reshape(best_action,(4,4))
        best_actions.append(best_action)

        average_returns+=returns
        average_eval_return+=eval_return
        std_eval_return[:,j]=eval_return
    
    for i in range (nb_runs):
        print("run ",i," strategy:")
        print(best_actions[i])

    average_returns/=nb_runs
    average_eval_return/=nb_runs

    average_for_plot=average_returns.shape[0]//100
    average_returns_reduced=np.zeros(average_for_plot)

    for i in range(average_for_plot):
        average_returns_reduced[i]=np.mean(average_returns[i*100:(i+1)*100])


    plt.plot(list(range(0,num_episodes,100)),average_returns_reduced)
    plt.errorbar(list(range(0,num_episodes,evaluate_every)),average_eval_return,yerr=np.std(std_eval_return,1),color='black')
    plt.xlabel("Episodes")
    plt.ylabel("Average return")
    plt.show()
    #plt.savefig("runner.png")      
    env.close()