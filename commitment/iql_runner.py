from typing import Tuple, List

import numpy as np
from numpy import ndarray

from iql_agent import IQLAgent
from matrix_games import MatrixGame
from matplotlib import pyplot as plt


def train_iql(env: MatrixGame, t_max: int, evaluate_every: int, num_evaluation_episodes: int,
              epsilon_max: float = None, epsilon_min: float = None,
              epsilon_decay: float = None) -> Tuple[List[IQLAgent], ndarray, ndarray, list]:
    """
    Training loop.

    :param env: The gym environment.
    :param t_max: The number of timesteps.
    :param evaluate_every: Evaluation frequency.
    :param num_evaluation_episodes: Number of episodes for evaluation.
    :param epsilon_max: The maximum epsilon of epsilon-greedy.
    :param epsilon_min: The minimum epsilon of epsilon-greedy.
    :param epsilon_decay: The decay factor of epsilon-greedy.
    :return: Tuple containing the list of agents, the returns of all training episodes, the averaged evaluation
    return of each evaluation, and the list of the greedy joint action of each evaluation.
    """
    agents = [IQLAgent(env.num_actions, epsilon_max, epsilon_min, epsilon_decay)
              for _ in range(env.num_agents)]
    E_joint=[]
    digits = len(str(t_max))
    evaluation_returns = np.zeros(t_max // evaluate_every)
    returns = np.zeros(t_max)
    for episode in range(t_max):
        returns[episode] = run_round(env,agents, True)
        if (episode + 1) % evaluate_every == 0:
            evaluation_step = episode // evaluate_every
            cum_rewards_eval = np.zeros(num_evaluation_episodes)
            for eval_episode in range(num_evaluation_episodes):
                cum_rewards_eval[eval_episode] = run_round(env,agents, False)
            evaluation_returns[evaluation_step] = np.mean(cum_rewards_eval)
            E_joint.append((agents[0].greedy_action(),agents[1].greedy_action())) #We may need to make it more "ok" as we assume 2 here
            print(f"Episode {(episode + 1): >{digits}}/{t_max:0{digits}}:\t"
                  f"Averaged evaluation return {evaluation_returns[evaluation_step]:0.3}")
    return agents, returns, evaluation_returns,E_joint

def run_round(env: MatrixGame, agents: IQLAgent, training: bool) -> int:
    """
    runs a round of the game

    :param env: The MatrixGame environment.
    :param agents: The list of Independant Q-learning agents
    :param training: If true the q_table will be updated using q-learning. The flag is also passed to the action selector.
    :return: the reward as a int
    """
    actions = np.zeros(env.num_agents,dtype=int)
    for agents_nb in range(env.num_agents):
        actions[agents_nb] = agents[agents_nb].act(training)
    reward = env.act(actions)
    if(training):
        for agents_nb in range(env.num_agents):
            agents[agents_nb].learn(actions[agents_nb],reward)
    return reward

if __name__ == '__main__':
    env = MatrixGame()
    nb_sim = 100
    t_max=1000
    evaluate_every=100
    E_joints=[]
    for sim in range(nb_sim):
        agents,returns,evaluation_return,E_joint=train_iql(env,t_max,evaluate_every,20,0.5,0.,0.6)
        E_joints.append(E_joint)

    prob_opti={}
    for sim in E_joints:
        for action in sim:
            prob_opti[action]=prob_opti.get(action,0)+1
    
    prob_occurence=[]
    greedy_act=[]
    for step in range(t_max//evaluate_every):
        occurence_joint_compare={}
        for run in range(nb_sim):
            action=E_joints[run][step]
            occurence_joint_compare[action]=occurence_joint_compare.get(action,0)+1
        mst_common_act=max(occurence_joint_compare,key=occurence_joint_compare.get)
        greedy_act.append(mst_common_act)
        prob_occurence.append(occurence_joint_compare[mst_common_act])

    for step in range(t_max//evaluate_every):
        prob_occurence[step]/=nb_sim
    
    fig, ax = plt.subplots()
    ax.bar(list(range(0,t_max,evaluate_every)),prob_occurence,width=50)
    ax.set_xticks(list(range(0,t_max,evaluate_every)), labels=greedy_act)
    plt.xlabel("Joint for each evaluation time step")
    plt.ylabel("Probability")
    plt.ylim(0,1.05)
    plt.show()
    #plt.savefig("runner_iql_compare.png") to store the img 

    opti_action=env.get_best_action()
    if(opti_action in prob_opti):
        nb_eval_per_sim=t_max/evaluate_every
        prob=prob_opti[opti_action]/(nb_sim*nb_eval_per_sim)
        print("Joint action: ",opti_action," with a prob of ",prob)
    else:
        print("The action has not been considered by the agents")

    


