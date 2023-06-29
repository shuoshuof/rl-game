import numpy as np

from utils import Environment
import cv2
import win32api
from tqdm import tqdm
from A2C import *
import torch
import time
from torch.utils.tensorboard import SummaryWriter
def train_on_policy_agent(env, agent, num_episodes):
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
                state = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, _ = env.step(action)
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    state = next_state
                    episode_return += reward
                return_list.append(episode_return)
                agent.update(transition_dict)
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list
def state_transformer(state):
    # 保证双方的状态都是对等的
    p0_state = state[:6]
    p1_state = state[0:4]+state[6:]
    return np.array(p0_state),np.array(p1_state)
def action_transformer(p0_action_idx,p1_action_idx):
    idx2action = [[0,-1],[0,1],[-1,0],[1,0],[0,0]]
    p0_action = idx2action[p0_action_idx]
    p1_action = idx2action[p1_action_idx]
    return p0_action,p1_action
def train(env,agent,num_episodes,writer):
    reward_list = []
    with tqdm(total=num_episodes, desc='训练进度') as pbar:
        for episode in range(num_episodes):
            total_reward =0
            transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
            env.reset()
            done = False
            while not done:
                state = env.get_state()
                p0_state, p1_state = state_transformer(state)

                p1_action_idx = agent.take_action(p1_state)

                _,p1_action = action_transformer(0,p1_action_idx)

                img,next_state, done,reward = env.update([0,0],p1_action,draw=False)

                p0_next_state, p1_next_state = state_transformer(next_state)

                p0_reward, p1_reward = reward
                transition_dict['states'].append(p1_next_state)
                transition_dict['actions'].append(p1_action_idx)
                transition_dict['next_states'].append(p1_next_state)
                transition_dict['rewards'].append(p1_reward)
                transition_dict['dones'].append(done)
                total_reward+=p1_reward
            reward_list.append(total_reward)
            agent.update(transition_dict)
            pbar.update(1)
            if (episode + 1) % 10 == 0:
                pbar.set_postfix({
                    'episode':
                    '%d' % episode,
                    'reward':
                    '%.3f' % np.mean(reward_list[-10:])
                })
            writer.add_scalars('reward',
                               tag_scalar_dict={'reward':reward_list[-1],},
                               global_step=episode+1)
if __name__ =='__main__':
    actor_lr = 1e-3
    critic_lr = 1e-2
    num_episodes = 100000
    hidden_dim = 128
    gamma = 0.98
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
        "cpu")

    log_path = './logs/{}'.format(time.strftime('%Y_%m_%d_%H_%M'))
    writer = SummaryWriter(log_path,flush_secs=60)

    env = Environment()
    state_dim = 6 # (bx,by,vx,vy,px,py)
    action_dim = 5 # (上，下，左，右，静止)
    agent = ActorCritic(state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                        gamma, device)


    train(env,agent,num_episodes,writer)