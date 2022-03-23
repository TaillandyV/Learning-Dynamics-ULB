import numpy as np
import random


class IQLAgent:
    """
    The agent class for exercise 1.
    """

    def __init__(self,
                 num_actions: int,
                 epsilon_max: float = None,
                 epsilon_min: float = None,
                 epsilon_decay: float = None):
        """
        :param num_actions: Number of actions.
        :param epsilon_max: The maximum epsilon of epsilon-greedy.
        :param epsilon_min: The minimum epsilon of epsilon-greedy.
        :param epsilon_decay: The decay factor of epsilon-greedy.
        """
        self.q_table = np.zeros((num_actions,))
        self.count = np.zeros((num_actions,), dtype=int)
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.epsilon_max = epsilon_max
        self.epsilon = epsilon_max

        self.num_actions=num_actions #added so that the agent knows the available options but not what they do

    def greedy_action(self) -> int:
        """
        Return the greedy action.

        :param observation: The observation.
        :return: The action.
        """
        return self.q_table.argmax()


    def act(self, training: bool = True) -> int:
        """
        Return the action.

        :param training: Boolean flag for training.
        :return: The action.
        """
        if(training):
            if(np.random.rand()<self.epsilon):
                return np.random.randint(0,self.num_actions)
            else:
                return self.greedy_action()
        else:
            return self.greedy_action()

    def learn(self, act: int, rew: float) -> None:
        """
        Update the Q-Value.

        :param obs: The observation.
        :param act: The action.
        :param rew: The reward.
        """
        self.count[act]+=1
        self.q_table[act] += (1./self.count[act])*(rew-self.q_table[act])
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
