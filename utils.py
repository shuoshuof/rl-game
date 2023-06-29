# -*- coding: utf-8 -*-
# time: 2023/6/27 11:16
# file: utils.py
# author: shuoshuof
import numpy as np
import cv2
import win32api
import random
class ball:
    def __init__(self,r,x,y,v_x,v_y):
        self.x = x
        self.y = y
        self.v_x = v_x
        self.v_y = v_y
        self.belong = int(self.v_y >0)
        self.r = r
    def update(self):
        self.x += self.v_x
        self.y += self.v_y
    def wall_collision(self):
        self.v_x = -self.v_x
    def racket_collision(self):
        self.v_y = -self.v_y
        self.v_x+=random.uniform(-0.2,0.2)
        self.belong = 1-self.belong
class racket:
    def __init__(self,player_id,x,y,R,W,r=5,w=15):
        self.player_id = player_id
        self.x= x
        self.y= y
        self.r = r
        self.w = w
        # 背景大小
        self.R = R
        self.W = W
        self.update_Collision_domain()
        self.reward =0
        self.num_beat=0
    def update_Collision_domain(self):
        self.pt_lt = (int(self.x-self.w//2),int(self.y-self.r//2))
        self.pt_rb = (int(self.x+self.w//2),int(self.y+self.r//2))
    def update(self,dx,dy):
        self.x = max(0,min(self.x+dx,self.W-1))
        if self.player_id ==1:
            self.y = max(self.R//2,min(self.y+dy,self.R-1))
        else:
            self.y = max(0,min(self.y+dy,self.R//2))
        self.update_Collision_domain()
    def get_reward(self):
        #print('玩家{}击球'.format(self.player_id))
        pass
    def detect(self,bx,by,belong):

        if self.pt_lt[0] <= bx and bx <= self.pt_rb[0] and self.pt_lt[1] <= by and by <= self.pt_rb[1] and belong==self.player_id:
            self.get_reward()
            self.num_beat+=1
            return True
        else:
            return False
class Environment:
    def __init__(self,b_r=120,b_w=80):
        self.r = b_r
        self.w = b_w
        # self.player0_color = 255
        # self.player1_color = 200
        self.Init()
    def Init(self):
        # self.ball = ball(r=3,x=self.w//2,y=self.r//2,v_x=0,v_y=-1)
        self.ball_init()
        # self.player0 =racket(player_id=0,x=self.w//2,y=5,R=self.r,W=self.w)
        self.player0 = racket(player_id=0, x=self.w // 2, y=5, R=self.r, W=self.w,w=self.w)

        self.player1 =racket(player_id=1,x=self.w//2,y=self.r-5,R=self.r,W=self.w)
        self.point_score=[0,0]
        self.frame =0
    def ball_init(self):
        b_vx = random.uniform(-1,1)
        self.ball = ball(r=3,x=self.w//2,y=5,v_x=b_vx,v_y=-1)
    def reset(self):
        self.Init()
    def get_state(self):
        bx,by = self.ball.x/self.w,self.ball.y/self.r
        v_x,v_y = self.ball.v_x,self.ball.v_y
        p0_x,p0_y = self.player0.x/self.w,self.player0.y/self.r
        p1_x,p1_y = self.player1.x/self.w,self.player1.y/self.r
        state = [bx,by,v_x,v_y,p0_x,p0_y,p1_x,p1_y]
        return state
    def update(self,player0_action:list,player1_action:list,draw=False):
        # state = self.get_state()
        self.player0.update(*player0_action)
        self.player1.update(*player1_action)
        self.ball.update()
        img = None
        if draw:
            img = self.draw()

        next_state = self.get_state()

        reward = self.collision_detect()
        done =False
        if max(self.point_score)>20 or self.player0.num_beat>20 or self.player1.num_beat>20:
            done = True
        self.frame+=1
        return img,next_state,done,reward,self.frame
    def draw(self):
        img = np.zeros((self.r, self.w))
        bx,by,r = self.ball.x,self.ball.y,self.ball.r
        cv2.circle(img,(int(bx),int(by)),radius=r,color=255,thickness=-1)

        # x,y,r,w = self.player0.x,self.player0.y,self.player0.r,self.player0.w
        pt0_lt = self.player0.pt_lt
        pt0_rb = self.player0.pt_rb
        cv2.rectangle(img,pt0_lt,pt0_rb,color=255,thickness=-1)

        # x, y, r, w = self.player1.x, self.player1.y, self.player1.r, self.player1.w
        pt1_lt = self.player1.pt_lt
        pt1_rb = self.player1.pt_rb

        cv2.rectangle(img, pt1_lt, pt1_rb, color=255, thickness=-1)
        return img
    def collision_detect(self):
        reward = [0,0]
        bx,by,r = self.ball.x,self.ball.y,self.ball.r
        # 碰撞到左右边界
        if bx <=0 or bx >= self.w-1:
            self.ball.wall_collision()
        # 球拍碰撞
        if self.player0.detect(bx,by,belong=self.ball.belong):
            self.ball.racket_collision()
            #reward[0]=1
        if self.player1.detect(bx,by,belong=self.ball.belong):
            self.ball.racket_collision()
            #reward[1] = 1
        if by <=0:
            self.point_score[1]+=1
            reward[0] = -1
            reward[1] = 1
            self.ball_init()
        if by >= self.r:
            self.point_score[0]+=1
            reward[1] = -1
            reward[0] =1
            self.ball_init()
        #print(self.point_score)
        return reward
def control():
    action =[0,0]
    if win32api.GetKeyState(ord('W'))<0:
        action[1] -= 1
    if win32api.GetKeyState(ord('S'))<0:
        action[1] += 1
    if win32api.GetKeyState(ord('A'))<0:
        action[0] -= 1
    if win32api.GetKeyState(ord('D'))<0:
        action[0] += 1
    return action
def state_transformer(state):
    # 保证双方的状态都是对等的
    bx, by, v_x, v_y, p0_x, p0_y, p1_x, p1_y =state
    p0_state = state[1-bx,1-by,-v_x,-v_y,1-p0_x,1-p0_y]
    p1_state = state[bx, by, v_x, v_y,p1_x, p1_y]
    return np.array(p0_state),np.array(p1_state)
def action_transformer(p0_action_idx,p1_action_idx):
    idx2action_1 = [[0,-1],[0,1],[-1,0],[1,0],[0,0]]
    idx2action_0 = [[0, 1], [0, -1], [1, 0], [-1, 0], [0, 0]]
    p0_action = idx2action_0[p0_action_idx]
    p1_action = idx2action_1[p1_action_idx]
    return p0_action,p1_action
def controler(draw):
    if win32api.GetKeyState(ord('J')) < 0:
        return True
    elif win32api.GetKeyState(ord('K')) < 0:
        cv2.destroyAllWindows()
        return False
    else:
        return draw

if __name__ == "__main__":

    env  =Environment()
    total_reward = 0
    while True:
        state = env.get_state()
        p0_state, p1_state = state_transformer(state)

        action = control()
        img,next_state, done,reward,frame = env.update([0,0],action,draw=True)

        p0_reward, p1_reward = reward

        total_reward += p1_reward
        #print(total_reward)
        # print(p1_state)
        if done:
            print(frame)
            total_reward = 0
            env.reset()
            break
        img = cv2.resize(img,(img.shape[1]*4,img.shape[0]*4))
        cv2.imshow('img',img)
        cv2.waitKey(1)





        
        



