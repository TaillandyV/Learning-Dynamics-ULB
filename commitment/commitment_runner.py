from typing import List, Tuple

from commitment_agent import CommitmentAgent
from matrix_games import MatrixGame

import numpy as np
from matplotlib import pyplot as plt


def train_commitment(env: MatrixGame, t_max: int, n_min: int, n_init: int,
                     p: float, evaluate_every: int) -> Tuple[List[CommitmentAgent], List]:
    """
    Training loop.

    :param env: The gym environment.
    :param t_max: The number of timesteps.
    :param n_min: The threshold of number of samples to consider the average reward of a commitment sequence reliable.
    :param n_init: The number of sequences to initialize by exploring.
    :param p: The probability to start a new sequence by exploring randomly and uniformly.
    :param evaluate_every: Frequency of evaluation to get the current greedy action.
    :return: Tuple containing the list of trained agents and a list containing the greedy joint actions at each evaluation step.
    """
    agents = [CommitmentAgent(env.num_actions, t_max, n_min, n_init, p) for _ in range(env.num_agents)]
    digits = len(str(t_max))
    evaluation_returns = np.zeros(t_max // evaluate_every)
    returns = np.zeros(t_max)
    E_joint=[]
    for episode in range(t_max):
        returns[episode] = run_round(env,agents, True, episode)
        if (episode + 1) % evaluate_every == 0:
            evaluation_step = episode // evaluate_every
            evaluation_returns[evaluation_step] = run_round(env,agents, False, episode)
            E_joint.append((agents[0].greedy_action(),agents[1].greedy_action()))
            print(f"Episode {(episode + 1): >{digits}}/{t_max:0{digits}}:\t"
                  f"Averaged evaluation return {evaluation_returns[evaluation_step]:0.3}")
    return agents,E_joint

def run_round(env: MatrixGame, agents: CommitmentAgent, training: bool, episode: int) -> int:
    """
    runs a round of the game

    :param env: The MatrixGame environment.
    :param agents: The list of Independant Commitment agents
    :param training: If true the agents will learn this step
    :return: the reward as a int
    """
    actions = np.zeros(env.num_agents,dtype=int)
    if(training):
        for agents_nb in range(env.num_agents):
            actions[agents_nb] = agents[agents_nb].act(episode)
        reward = env.act(actions)
        for agents_nb in range(env.num_agents):
            agents[agents_nb].learn(episode,reward)
    else:
        for agents_nb in range(env.num_agents):
            actions[agents_nb] = agents[agents_nb].greedy_action()
        reward = env.act(actions)

    return reward


if __name__ == '__main__':
    env = MatrixGame()
    nb_sim=100
    t_max=1000
    n_min=10
    n_init=10
    evaluate_every=100
    p = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    opti_action=env.get_best_action()
    prob_p=[]
    compare_prob=[]
    for prob in p:
        avg_prob=[]
        joints_converge=[]
        for i in range(nb_sim):
            trained_agents,E_joint=train_commitment(env,t_max,n_min,n_init,prob,evaluate_every)
            joint_act=(trained_agents[0].greedy_action(),trained_agents[1].greedy_action())
            joints_converge.append(joint_act)
            if(prob == 0.9):
                compare_prob.append(E_joint)
        prob_opti={}
        for action in joints_converge:
            prob_opti[action]=prob_opti.get(action,0)+1
        if(opti_action in prob_opti):
            avg_prob.append(prob_opti[opti_action]/(nb_sim))
        else:
            avg_prob.append(0)
        prob_p.append(avg_prob)

    prob_occurence=[]
    greedy_act=[]
    for step in range(t_max//evaluate_every):
        occurence_joint_compare={}
        for run in range(nb_sim):
            action=compare_prob[run][step]
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
    #plt.savefig("runner_commitment_compare.png") 

    plt.clf()

    plt.plot(p,prob_p)
    plt.xticks(p)
    plt.ylim(0,1.05)
    plt.show()
    #plt.savefig("runner_commitment.png") 

  

    
        

