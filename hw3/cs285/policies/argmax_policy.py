import numpy as np


class ArgMaxPolicy(object):

    def __init__(self, critic):
        self.critic = critic

    def get_action(self, obs):
        if len(obs.shape) > 3:
            observation = obs
        else:
            observation = obs[None]
        
        ## TODO return the action that maxinmizes the Q-value 
        # at the current observation as the output
        
        # argmax(-1) returns the index of last dimension (action, in this case) 
        actions = self.critic.qa_values(observation).argmax(-1)

        return actions.squeeze()