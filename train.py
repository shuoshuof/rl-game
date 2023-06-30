# -*- coding: utf-8 -*-
# time: 2023/6/28 16:27
# file: train.py
# author: shuoshuof

import numpy as np

from utils import Environment,action_transformer,state_transformer
import cv2
import win32api
from tqdm import tqdm
from A2C import *
import torch
import time
from torch.utils.tensorboard import SummaryWriter
import imageio
def controler(draw):
    if win32api.GetKeyState(ord('J')) < 0:
        return True
    elif win32api.GetKeyState(ord('K')) < 0:
        cv2.destroyAllWindows()
        return False
    else:
        return draw
def save_git(env,agent,episode):
    env.reset()
    done = False
    all_frames = []
    while not done:
        state = env.get_state()
        p0_state, p1_state = state_transformer(state)

        p1_action_idx = agent.take_action(p1_state)

        _, p1_action = action_transformer(0, p1_action_idx)

        img, next_state, done, reward,frame_num = env.update([0, 0], p1_action, draw=True)

        p0_next_state, p1_next_state = state_transformer(next_state)
        all_frames.append(img)
    imageio.mimsave('./result.gif', all_frames,duration=30)
def train(env,agent,num_episodes,writer):
    reward_list = []
    frame_num_list = []
    draw = False
    with tqdm(total=num_episodes, desc='训练进度') as pbar:
        for episode in range(num_episodes):
            total_reward =0
            transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
            env.reset()
            done = False
            frame_num =None
            while not done:
                state = env.get_state()
                p0_state, p1_state = state_transformer(state)

                p1_action_idx = agent.take_action(p1_state)

                _,p1_action = action_transformer(0,p1_action_idx)

                draw = controler(draw)
                img,next_state, done,reward,frame_num = env.update([0,0],p1_action,draw=draw)

                p0_next_state, p1_next_state = state_transformer(next_state)

                p0_reward, p1_reward = reward
                transition_dict['states'].append(p1_next_state)
                transition_dict['actions'].append(p1_action_idx)
                transition_dict['next_states'].append(p1_next_state)
                transition_dict['rewards'].append(p1_reward)
                transition_dict['dones'].append(done)
                total_reward+=p1_reward
                if draw:
                    cv2.imshow('img', img)
                    cv2.waitKey(1)
            #print(total_reward)
            reward_list.append(total_reward)
            frame_num_list.append(frame_num)
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
                                   tag_scalar_dict={'reward':np.mean(reward_list[-10:]),},
                                   global_step=episode+1)
                writer.add_scalars('frame',
                                   tag_scalar_dict={'frame':np.mean(frame_num_list[-10:]),},
                                   global_step=episode+1)

            if (episode+1)%100==0:
                save_git(env, agent, episode + 1)
                writer.add_scalars('avg_frame',
                                   tag_scalar_dict={'frame': np.mean(frame_num_list[-100:]), },
                                   global_step=episode + 1)
                agent.save(dir='./models')
                writer.add_scalars('avg_reward',
                                   tag_scalar_dict={'reward': np.mean(reward_list[-100:]), },
                                   global_step=(episode + 1)/100)

if __name__ =='__main__':
    actor_lr = 1e-4
    critic_lr = 1e-4
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
    agent.load('./models')

    train(env,agent,num_episodes,writer)