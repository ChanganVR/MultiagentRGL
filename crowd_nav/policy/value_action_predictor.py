import torch
import torch.nn as nn
from crowd_nav.policy.helpers import mlp


class ValueActionPredictor(nn.Module):
    def __init__(self, config, graph_model_val, graph_model_act=None, shared_gcn=True):
        super().__init__()
        self.shared_gcn = shared_gcn
        if shared_gcn:
            self.graph_model = graph_model_val
        else:
            self.graph_model_val = graph_model_val
            assert graph_model_act is not None, "void graph_model_act"
            self.graph_model_act = graph_model_act

        self.value_network = mlp(config.gcn.X_dim, config.rgl_ppo.value_network_dims)
        self.action_network = mlp(config.gcn.X_dim, config.rgl_ppo.value_network_dims[:-1] +
                                  [config.action_space.rotation_samples * config.action_space.speed_samples + 1])
        
    def convert_to_mean_and_cov(self, action_feats): # action_feats: (batch, 5) 5->(mu_x, mu_y, s_x^2, s_y^2, s_xy)
        mu = action_feats[:,:2] # (batch, 2)
        cov = torch.cat((action_feats[:,2:3], action_feats[:,-1:], action_feats[:,-1:], action_feats[:,3:4]), dim=-1)\
                   .view(-1,2,2) # (batch, 2, 2)
        return mu, cov

    def forward(self, state):
        """ Embed state into a latent space. Take the first row of the feature matrix as state representation.
        """
        assert len(state[0].shape) == 3
        assert len(state[1].shape) == 3

        # only use the feature of robot node as state representation
        if self.shared_gcn:
            state_embedding = self.graph_model(state)[:, 0, :]
            value = self.value_network(state_embedding)
            action_feat = self.action_network(state_embedding)
        else:
            state_embedding_val = self.graph_model_val(state)[:, 0, :]
            state_embedding_act = self.graph_model_act(state)[:, 0, :]
            value = self.value_network(state_embedding_val)
            action_feat = self.action_network(state_embedding_act)
            mu, cov = self.convert_to_mean_and_cov(action_feat)
            
        return value, mu, cov
