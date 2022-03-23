from typing import Iterable
import random
import numpy as np

class MatrixGame:
    """
    Matrix Game environment.
    """

    def __init__(self):
        """
        Matrix Game environment.
        """
        self.num_agents = 2
        self.num_actions = 3
        #We want to create a 3*3 matrix but because this is fixed, just like the values, we may directly construct it
        self.matrix=np.array([[{16,22,-5},{4,6,-100},{10,20,-30}],
                         [{4,6,-100},{25,0,-4},{10,5,3}],
                         [{8,12,-20},{10,20,-30},{4,5,6}]],
                         dtype=object)

    def act(self, action: Iterable[int]):
        """
        Method to perform an action in the Matrix Game and obtain the associated reward.
        :param action: The joint action.
        :return: The reward.
        """
        return random.sample(self.matrix[action[0]][action[1]],1)[0]

    def get_best_action(self):
        """
        Method to get the joint action that gives the best average rewards
        :return: The joint action.
        """
        actions_reward={}
        for i in range(self.matrix.shape[0]):
            for j in range(self.matrix.shape[1]):
                set=self.matrix[i][j]
                avg_reward=0
                for reward in set:
                    avg_reward+=reward
                actions_reward[(i,j)]=avg_reward/len(set)
        return max(actions_reward,key=actions_reward.get)
