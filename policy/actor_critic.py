import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from argparse import Namespace
from torch.distributions.normal import Normal
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torchscale.architecture.retnet import RetNetRelPos
from torchscale.component.multiscale_retention import MultiScaleRetention, theta_shift


def mlp(sizes, activation=nn.ReLU, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        activ = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), activ()]

    return nn.Sequential(*layers)

# *** Trick: Orthogonal Initialization ***
use_orthogonal_init = True
def orthogonal_init(module, gain=np.sqrt(2)):
    for layer in module.modules():
        if isinstance(layer, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.GRU, nn.LSTM, nn.MultiheadAttention)):
            for name, param in layer.named_parameters():
                if 'weight' in name:
                    nn.init.orthogonal_(param, gain)
                elif 'bias' in name:
                    nn.init.constant_(param, 0.0)


class EgoStateEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, device):
        super().__init__()

        self.mlp_net = nn.Sequential(
            nn.Linear(input_dim, output_dim, device=device),
            nn.LayerNorm(output_dim, device=device),
            nn.GELU(),
            nn.Linear(output_dim, output_dim, device=device),
            nn.LayerNorm(output_dim, device=device),
            nn.GELU()
            )
    
        if use_orthogonal_init:
            orthogonal_init(self.mlp_net)
    
    def forward(self, ego_state_batch):
        ego_state_hn_batch = self.mlp_net(ego_state_batch)

        return ego_state_hn_batch


class RawSensorEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, device):
        super().__init__()

        self.input_dim = input_dim
        self.out_channel_1 = 32
        self.out_channel_2 = 32
        self.output_dim = output_dim
        self.kernel_size_1 = 5
        self.stride_1 = 2
        self.kernel_size_2 = 3
        self.stride_2 = 2
        self.device = device

        self.cnn_net_1 = nn.Sequential(
            nn.Conv1d(input_dim[0], self.out_channel_1, self.kernel_size_1, self.stride_1, device=device),
            nn.ReLU()
            )
        self.cnn_net_2 = nn.Sequential(
            nn.Conv1d(self.out_channel_1, self.out_channel_2, self.kernel_size_2, self.stride_2, device=device),
            nn.ReLU()
            )
        net_1_out_dim = int(np.floor((input_dim[1]-self.kernel_size_1)/self.stride_1+1))
        net_2_out_dim = int(np.floor((net_1_out_dim-self.kernel_size_2)/self.stride_2+1))
        self.output_net = nn.Sequential(
            nn.Linear(self.out_channel_2*net_2_out_dim, output_dim, device=device),
            nn.ReLU()
            )
        
        if use_orthogonal_init:
            orthogonal_init(self.cnn_net_1)
            orthogonal_init(self.cnn_net_2)
            orthogonal_init(self.output_net)

    def forward(self, sensor_obs_batch):
        sensor_obs_hn_batch = self.cnn_net_1(sensor_obs_batch)
        sensor_obs_hn_batch = self.cnn_net_2(sensor_obs_hn_batch)
        sensor_obs_hn_batch = sensor_obs_hn_batch.view(sensor_obs_hn_batch.size(0), -1)
        sensor_obs_hn_batch = self.output_net(sensor_obs_hn_batch)

        return sensor_obs_hn_batch


class SensorEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, device, mode='biGRU'):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.device = device
        self.mode = mode

        if mode == 'GRU':
            self.rnn_net = nn.GRU(input_dim, hidden_dim, batch_first=True, device=device)
        elif mode == 'biGRU':
            self.rnn_net = nn.GRU(input_dim, hidden_dim, batch_first=True, bidirectional=True, device=device)
        elif mode == 'LSTM':
            self.rnn_net = nn.LSTM(input_dim, hidden_dim, batch_first=True, device=device)
        elif mode == 'biLSTM':
            self.rnn_net = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True, device=device)
        
        if use_orthogonal_init:
            orthogonal_init(self.rnn_net)

    def forward(self, sensor_obs_batch):
        sensor_obs_hn_batch = torch.zeros(sensor_obs_batch.size(0), self.hidden_dim, device=self.device)
        obstacle_num_batch = (sensor_obs_batch != 0).any(dim=-1).sum(dim=-1).cpu()
        obstacle_num_batch.masked_fill_(obstacle_num_batch == 0, 1)
        batch_valid = obstacle_num_batch > 0
        if batch_valid.any():
            sensor_obs_batch_valid = sensor_obs_batch[batch_valid]
            obstacle_num_batch_valid = obstacle_num_batch[batch_valid]
            sensor_obs_batch_valid = pack_padded_sequence(sensor_obs_batch_valid, obstacle_num_batch_valid, batch_first=True, enforce_sorted=False)
            if self.mode == 'GRU' or self.mode == 'biGRU':
                output, hn = self.rnn_net(sensor_obs_batch_valid)
            elif self.mode == 'LSTM' or self.mode == 'biLSTM':
                output, (hn, cn) = self.rnn_net(sensor_obs_batch_valid)
            sensor_obs_hn_batch_valid = torch.sum(hn, 0)
            sensor_obs_hn_batch[batch_valid] = sensor_obs_hn_batch_valid
        
        return sensor_obs_hn_batch


class AgentEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, device, mode='biGRU'):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.device = device
        self.mode = mode

        if mode == 'GRU':
            self.rnn_net = nn.GRU(input_dim, hidden_dim, batch_first=True, device=device)
        elif mode == 'biGRU':
            self.rnn_net = nn.GRU(input_dim, hidden_dim, batch_first=True, bidirectional=True, device=device)
        elif mode == 'LSTM':
            self.rnn_net = nn.LSTM(input_dim, hidden_dim, batch_first=True, device=device)
        elif mode == 'biLSTM':
            self.rnn_net = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True, device=device)
        
        if use_orthogonal_init:
            orthogonal_init(self.rnn_net)

    def forward(self, agent_obs_batch):
        agent_obs_hn_batch = torch.zeros(agent_obs_batch.size(0), self.hidden_dim, device=self.device)
        agent_num_batch = (agent_obs_batch != 0).any(dim=-1).sum(dim=-1).cpu()
        agent_num_batch.masked_fill_(agent_num_batch == 0, 1)
        batch_valid = agent_num_batch > 0
        if batch_valid.any():
            agent_obs_batch_valid = agent_obs_batch[batch_valid]
            agent_num_batch_valid = agent_num_batch[batch_valid]
            agent_obs_batch_valid = pack_padded_sequence(agent_obs_batch_valid, agent_num_batch_valid, batch_first=True, enforce_sorted=False)
            if self.mode == 'GRU' or self.mode == 'biGRU':
                output, hn = self.rnn_net(agent_obs_batch_valid)
            elif self.mode == 'LSTM' or self.mode == 'biLSTM':
                output, (hn, cn) = self.rnn_net(agent_obs_batch_valid)
            agent_obs_hn_batch_valid = torch.sum(hn, 0)
            agent_obs_hn_batch[batch_valid] = agent_obs_hn_batch_valid
        
        return agent_obs_hn_batch


class SpatialEncoder(nn.Module):
    def __init__(self, ego_state_dim, sensor_obs_dim, agent_obs_dim, costmap_obs_dim, obs_hn_dim, obs_type, unit_num, device):
        super().__init__()

        self.obs_type = obs_type
        self.device = device

        self.ego_state_encoder = EgoStateEncoder(ego_state_dim, obs_hn_dim, device=device)
        if 's' in obs_type:
            if isinstance(sensor_obs_dim, tuple):
                self.sensor_encoder = RawSensorEncoder(sensor_obs_dim, obs_hn_dim, device=device)
            elif isinstance(sensor_obs_dim, int):
                self.sensor_encoder = SensorEncoder(sensor_obs_dim, obs_hn_dim, device=device, mode='biGRU')
        if 'a' in obs_type:
            self.agent_encoder = AgentEncoder(agent_obs_dim, obs_hn_dim, device=device, mode='biGRU')

        self.att_nets = nn.ModuleList(
            nn.Sequential(
                nn.Linear(obs_hn_dim * (1+len(obs_type)), obs_hn_dim, device=device),
                nn.GELU(),
                nn.Linear(obs_hn_dim, (1+len(obs_type)), device=device),
                )
            for _ in range(unit_num)
            )
        # self.ln = nn.LayerNorm(obs_hn_dim, device=device)
        self.fusion_net = nn.Sequential(
            nn.Linear(obs_hn_dim * unit_num, obs_hn_dim, device=device),
            nn.GELU(),
            # nn.Linear(obs_hn_dim, obs_hn_dim, device=device)
            )
        
        if use_orthogonal_init:
            orthogonal_init(self.att_nets)
            orthogonal_init(self.fusion_net)

    def forward(self, ego_state_list, sensor_obs_list, agent_obs_list, costmap_obs_list):
        ego_state_batch = torch.stack(ego_state_list).to(self.device)
        ego_state_hn_batch = self.ego_state_encoder(ego_state_batch).unsqueeze(1)
        ego_state_mask = torch.zeros(len(ego_state_list), dtype=torch.bool, device=self.device).unsqueeze(1)

        obs_hn_batch = ego_state_hn_batch
        obs_mask = ego_state_mask

        if 's' in self.obs_type:
            sensor_obs_batch = pad_sequence(sensor_obs_list, batch_first=True).to(self.device)
            sensor_obs_hn_batch = self.sensor_encoder(sensor_obs_batch).unsqueeze(1)
            sensor_obs_mask = torch.as_tensor([torch.all(obs == 0) for obs in sensor_obs_list], dtype=torch.bool, device=self.device).unsqueeze(1)
            obs_hn_batch = torch.cat((obs_hn_batch, sensor_obs_hn_batch), 1)
            obs_mask = torch.cat((obs_mask, sensor_obs_mask), 1)
        if 'a' in self.obs_type:
            agent_obs_batch = pad_sequence(agent_obs_list, batch_first=True).to(self.device)
            agent_obs_hn_batch = self.agent_encoder(agent_obs_batch).unsqueeze(1)
            agent_obs_mask = torch.as_tensor([torch.all(obs == 0) for obs in agent_obs_list], dtype=torch.bool, device=self.device).unsqueeze(1)
            obs_hn_batch = torch.cat((obs_hn_batch, agent_obs_hn_batch), 1)
            obs_mask = torch.cat((obs_mask, agent_obs_mask), 1)

        x = obs_hn_batch.reshape(obs_hn_batch.size(0), -1)
        att = torch.cat([att_net(x).unsqueeze(1) for att_net in self.att_nets], dim=1)
        att = att.masked_fill(obs_mask.unsqueeze(1), -float('inf'))
        att = F.softmax(att, dim=-1)
        y = torch.matmul(att, obs_hn_batch)
        
        # y = self.ln(y)
        y = y.reshape(y.size(0), -1)
        # y_ = y
        z = self.fusion_net(y)
        # z = z + y_

        return z


class TemporalEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, device):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.device = device

        self.rnn_net = nn.LSTM(input_dim, hidden_dim, batch_first=True, device=device)
        self.ln = nn.LayerNorm(hidden_dim, device=device)

        if use_orthogonal_init:
            orthogonal_init(self.rnn_net)

    def forward(self, obs_batch):
        output, (hn, cn) = self.rnn_net(obs_batch)
        obs_hn_batch = torch.sum(hn, 0)
        obs_hn_batch = self.ln(obs_hn_batch)

        return obs_hn_batch
    

class Observer(nn.Module):
    def __init__(self, policy_name, ego_state_dim, sensor_obs_dim, agent_obs_dim, costmap_obs_dim, obs_hn_dim, obs_dim, obs_type, ST_type, device):
        super().__init__()
        
        self.policy_name = policy_name
        self.obs_type = obs_type
        self.ST_type = ST_type
        self.device = device
        
        if 'S' in self.ST_type:
            if policy_name == 'Ours':
                self.spatial_encoder = SpatialEncoder(ego_state_dim, sensor_obs_dim, agent_obs_dim, costmap_obs_dim, obs_hn_dim, obs_type, unit_num=1, device=device)
        else:
            if 's' in self.obs_type:
                if isinstance(sensor_obs_dim, tuple):
                    self.sensor_encoder = RawSensorEncoder(sensor_obs_dim, obs_hn_dim, device=device)
                elif isinstance(sensor_obs_dim, int):
                    self.sensor_encoder = SensorEncoder(sensor_obs_dim, obs_hn_dim, device=device, mode='biGRU')
            if 'a' in self.obs_type:
                self.agent_encoder = AgentEncoder(agent_obs_dim, obs_hn_dim, device=device, mode='biGRU')
        self.ln = nn.LayerNorm(obs_dim, device=device)
        if 'T' in self.ST_type:
            if policy_name == 'Ours':
                self.temporal_encoder = TemporalEncoder(obs_dim, obs_dim, device=device)

    def encoder(self, obs_list_list):
        if 'T' in self.ST_type:
            all_obs_list = [obs for obs_list in obs_list_list for obs in obs_list]
        else:
            all_obs_list = [obs_list[-1] for obs_list in obs_list_list]
        all_obs_list = list(zip(*all_obs_list))
        ego_state_list, sensor_obs_list, agent_obs_list, costmap_obs_list = all_obs_list[0], all_obs_list[1], all_obs_list[2], []
        
        ego_state_batch = torch.stack(ego_state_list).to(self.device)

        if 'S' in self.ST_type:
            if self.policy_name == 'Ours':
                obs_hn_batch = self.spatial_encoder(ego_state_list, sensor_obs_list, agent_obs_list, costmap_obs_list)
            obs_hn_batch = torch.cat((ego_state_batch, obs_hn_batch), 1)
        else:
            obs_hn_batch = ego_state_batch
            if 's' in self.obs_type:
                sensor_obs_batch = pad_sequence(sensor_obs_list, batch_first=True).to(self.device)
                sensor_obs_hn_batch = self.sensor_encoder(sensor_obs_batch)
                obs_hn_batch = torch.cat((obs_hn_batch, sensor_obs_hn_batch), 1)
            if 'a' in self.obs_type:
                agent_obs_batch = pad_sequence(agent_obs_list, batch_first=True).to(self.device)
                agent_obs_hn_batch = self.agent_encoder(agent_obs_batch)
                obs_hn_batch = torch.cat((obs_hn_batch, agent_obs_hn_batch), 1)
            
        obs_hn_batch = self.ln(obs_hn_batch)

        if 'T' in self.ST_type:
            obs_hn_batch = obs_hn_batch.reshape(len(obs_list_list), -1, obs_hn_batch.shape[-1])
            obs_hn_batch = self.temporal_encoder(obs_hn_batch)
            
        return obs_hn_batch


class PPOActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, observer, device):
        super().__init__()

        self.observer = observer
        self.device = device
        
        self.pi_net = mlp([obs_dim] + list(hidden_sizes), nn.ReLU, nn.ReLU).to(device)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim).to(device)
        self.log_std = nn.Parameter(-torch.ones(act_dim, device=device))
        # self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim).to(device)
        
        if use_orthogonal_init:
            orthogonal_init(self.pi_net)
            orthogonal_init(self.mu_layer, 0.01)
            # orthogonal_init(self.log_std_layer, 0.01)
    
    def forward(self, obs, act=None, deterministic=False):
        obs = self.observer.encoder(obs)
        net_out = self.pi_net(obs)
        mu = self.mu_layer(net_out)
        std = torch.exp(self.log_std)
        # log_std = self.log_std_layer(net_out)
        # log_std = torch.clamp(log_std, -20, 2)
        # std = torch.exp(log_std)
        pi = Normal(mu, std)

        if act is None:
            if deterministic:
                raw_act = mu
            else:
                raw_act = pi.rsample()
            act = torch.tanh(raw_act)
            logp_act = pi.log_prob(raw_act).sum(-1) - torch.log(1 - act.pow(2) + 1e-6).sum(-1)
        else:
            act_clipped = torch.clamp(act, -1.0 + 1e-6, 1.0 - 1e-6)
            raw_act = torch.atanh(act_clipped)
            logp_act = pi.log_prob(raw_act).sum(-1) - torch.log(1 - act_clipped.pow(2) + 1e-6).sum(-1)

        return pi, act, logp_act


class PPOCritic(nn.Module):
    def __init__(self, obs_dim, hidden_sizes, observer, device):
        super().__init__()

        self.observer = observer
        self.device = device

        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], nn.ReLU, nn.Identity).to(device)

        if use_orthogonal_init:
            for layer in self.v_net.modules():
                if isinstance(layer, nn.Linear):
                    if layer.out_features == 1:
                        orthogonal_init(layer, 1.0)
                    else:
                        orthogonal_init(layer)

    def forward(self, obs):
        obs = self.observer.encoder(obs)
        net_out = self.v_net(obs)
        val = net_out.squeeze(-1)

        return val


class PPOActorCritic(nn.Module):
    def __init__(self, env, policy_name, ego_state_dim=6, sensor_obs_dim=7, agent_obs_dim=7, costmap_obs_dim=(1, 100, 100),
                 obs_hn_dim=256, hidden_sizes=(256, 256), obs_type='a', ST_type='S', device=None):
        super().__init__()

        self.name = policy_name
        self.device = device
        
        local_obs_type = obs_type
        if policy_name == 'Ours':
            if 'S' in ST_type:
                local_obs_dim = ego_state_dim + obs_hn_dim
            else:
                local_obs_dim = ego_state_dim + obs_hn_dim * len(local_obs_type)
        else:
            local_obs_dim = ego_state_dim + obs_hn_dim

        act_dim = env.action_space.shape[0]

        observer = Observer(policy_name, ego_state_dim, sensor_obs_dim, agent_obs_dim, costmap_obs_dim, obs_hn_dim, local_obs_dim, local_obs_type, ST_type, device=device)
        self.pi = PPOActor(local_obs_dim, act_dim, hidden_sizes, observer=observer, device=device)
        self.v = PPOCritic(local_obs_dim, hidden_sizes, observer=observer, device=device)

    def step(self, obs):
        with torch.no_grad():
            _, act, logp_act = self.pi(obs, None)
            val = self.v(obs)
        return act.cpu().numpy(), val.cpu().numpy(), logp_act.cpu().numpy()

    def act(self, obs):
        with torch.no_grad():
            _, act, _ = self.pi(obs, None, True)
        return act.cpu().numpy()