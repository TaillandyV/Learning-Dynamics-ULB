from typing import Tuple
from numpy import ndarray
import numpy as np


def create_seq(start: int, stop: int, step: int):
    """
    Generator that yield an elements of a seq with an increasing given step 

    :param start: the first element of the sequence
    :param stop: the maximum possible element of a sequence
    :param step: the first step 
    """
    value = start
    while value < stop:
        yield value
        value += step
        step += 1
    

def generate_t_to_seq(t_max: int) -> Tuple[dict, ndarray]:
    """
    Function that generates the different commitment sequences.

    Example
    --------------------------
    >>> (id_to_seq, t_to_seq) = generate_t_to_seq(10)
    >>> id_to_seq
    {0: array([0, 2, 5, 9]), 1: array([1, 4, 8]), 2: array([3, 7]), 3: array([6])}
    >>> t_to_seq
    array([0, 1, 0, 2, 1, 0, 3, 2, 1, 0])

    :param t_max: The number of timesteps.
    :return: (id_to_seq, t_to_seq) with id_to_seq a dictionary mapping
            sequences number to an array containing all its timesteps. t_to_seq is a array of size t_max with t_to_seq[t]
            being the sequence number of the timestep t.
    """
    nbr_seq=0
    seqs=[]
    for i in range(t_max):
        nbr_seq+=i
        new_seq=list(create_seq(nbr_seq, t_max, i+2))
        if(len(new_seq)!=0):
            seqs.append(new_seq)
        else:
            break
    
    id_to_seq={}

    for nb_seq in range(len(seqs)):
        id_to_seq[nb_seq]=seqs[nb_seq]
    
    t_to_seq=np.zeros(t_max,dtype=int)

    for keys in id_to_seq:
        fill=id_to_seq[keys]
        for position in fill:
            t_to_seq[position]=keys
   
    return id_to_seq,t_to_seq

class CommitmentAgent:
    """
    The agent class that applies commitment sequences.
    """

    def __init__(self, num_actions: int, t_max: int, n_min: int, n_init: int, p: float):
        """
        :param num_actions: The number of actions.
        :param t_max: The number of timesteps.
        :param n_min: The threshold of number of samples to consider the average reward of a commitment sequence reliable.
        :param n_init: The number of sequences to initialize by exploring.
        :param p: The probability to start a new sequence by exploring randomly and uniformly.
        """
        self.num_actions = num_actions
        self.t_max = t_max
        self.n_min = n_min
        self.n_init = n_init
        self.p = p
        self.id_to_seq,self.t_to_seq = generate_t_to_seq(t_max)
        self.sequence_action={}
        self.sequence_reward={}
        self.avg_reward={}

    def act(self, t: int) -> int:
        """
        Method that return the action to perform at timestep t.

        :param t: The timestep.
        :return: The action.
        """
        if(self.t_to_seq[t] not in self.sequence_action): #not a known sequence
            if(self.t_to_seq[t]<self.n_min): #below the n_min
                action = np.random.randint(0,self.num_actions)
                self.sequence_action[self.t_to_seq[t]] = action
                return action

            else:
                if(np.random.rand()<self.p):
                    action = np.random.randint(0,self.num_actions)
                    self.sequence_action[self.t_to_seq[t]] = action
                    return action

                else:
                    action = self.greedy_action()
                    self.sequence_action[self.t_to_seq[t]] = action
                    return action
        
        else:
            return self.sequence_action[self.t_to_seq[t]]

    def learn(self, t: int, reward: float) -> None:
        """
        Learning method using the fact that at timestep t the agent got a specific reward.

        :param t: The timestep.
        :param reward: The reward
        """
        if(self.t_to_seq[t] not in self.sequence_reward):  
            self.sequence_reward[self.t_to_seq[t]] = [reward]
        else:
            if(len(self.sequence_reward[self.t_to_seq[t]]) < self.n_min-1):
                self.sequence_reward[self.t_to_seq[t]].append(reward)

            else:
                self.sequence_reward[self.t_to_seq[t]].append(reward)
                self.avg_reward[self.t_to_seq[t]] = np.mean(self.sequence_reward[self.t_to_seq[t]])


    def greedy_action(self) -> int:
        """
        Return the action of the best sequence. If there are no sequences of length bigger than n_min, the returned
        action should be -1.
        :return: The action.
        """
        if(len(self.avg_reward)==0): #the avg is not of len 0 if at least one sequence as been observed for more than n_min.
            return -1
        else:
            return self.sequence_action[max(self.avg_reward, key=self.avg_reward.get)]
            

