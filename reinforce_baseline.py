'''
Monte Carlo REINFORCE Policy Gradient algorithm with Baseline
Author: Varshil Gandhi
Date: 16th December, 2018
Algo Source: David Silver Lectures & Rich Sutton RL Book 
'''

import os
from tqdm import tqdm
import gym
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import argparse
from policy_model import MLP_model
from value_model import Value_model
from utils import *


# Parameters
def argsparser():
    parser = argparse.ArgumentParser('Tensorflow implementation of Reinforce Monte Carlo Policy Gradient Algorithm')
    parser.add_argument('--env_id', help = 'environment_ID', default = 'LunarLander-v2')
    parser.add_argument('--tf_seed', help='Tensorflow seed', type=int, default=1)
    parser.add_argument('--np_seed', help='Numpy seed', type=int, default=1)
    parser.add_argument('--test_env_seed_no', help='number of ENV seeds ', type = int, default=5)
    parser.add_argument('--gamma', help = 'Discount Factor',type=float, default= 0.99)
    parser.add_argument('--num_ep', help = 'No of Episodic Iterations',type=int, default= 5000)
    parser.add_argument('--v_learning_rate', help = 'Step size for value model',type=float, default= 0.01)
    parser.add_argument('--p_learning_rate', help = 'Step size for policy model',type=float, default= 0.004)
    parser.add_argument('--render', help = 'Rendering of the episodes',type=bool, default= True)
    parser.add_argument('--save_period', help = 'Model saving interval',type=int,default= 20)
    parser.add_argument('--V_MODEL_DIR', help= 'TF value model saving', type = str, default='v_model/')
    parser.add_argument('--P_MODEL_DIR', help= 'TF policy model saving', type = str, default='p_model/')
    parser.add_argument('--SAVE_DIR', help= 'Complete test + train log store', type = str, default='Saved_Exps/')
    parser.add_argument('--test_ep', help = 'No of test episodes per seed',type=int,default= 100)
    parser.add_argument('--task', type= str, choices=['train','test'])
    parser.add_argument('--reward_monitor', type= int, default= -1000) 
    parser.add_argument('-e','--exp_no', help='Required for folder save type any +ve integer', type= int)
    if (parser.parse_args().exp_no is None and parser.parse_args().task == 'test'):
        parser.error('--exp_no is required for folder defination')

    return parser.parse_args()



def baseline_train(p_model, v_model, traj, episode_no, avg_reward, args):
    
    training_epochs = len(traj['values'])
    print('Episode Frames = ', training_epochs)
    
    avg_v_cost = 0
    avg_p_cost = 0
    
    # Calculating Delta
    vec = agent_value(v_model, traj['observations'])	# ? * 1
    vec = vec.T											# 1 * ?
    traj['delta'] = traj['values'] - vec[0] 			# traj['values'] = (?,)  # vec[0] = (?,)
    													# Therefore traj['delta'] = (?,)
    
    # Training session for value approximator
    
    _, c1 = v_model.sess.run([v_model.optimizer, v_model.loss], feed_dict={
            v_model.tf_observations : np.array(traj['observations']),
            v_model.tf_values : np.array(traj['values'])
        })
    avg_v_cost += c1

    _, c2 = p_model.sess.run([p_model.optimizer, p_model.loss], feed_dict={
            p_model.tf_observations : np.array(traj['observations']),
            p_model.tf_actions : np.array(traj['actions']),
            p_model.tf_values : np.array(traj['delta'])
        })    
    avg_p_cost += c2      
    
    if (episode_no % args.save_period) == 0 :
        if not os.path.exists(args.V_MODEL_DIR):
            os.makedirs(args.V_MODEL_DIR)
        if not os.path.exists(args.P_MODEL_DIR):
            os.makedirs(args.P_MODEL_DIR)
        
        if (avg_reward > args.reward_monitor):
            v_model.saver.save(v_model.sess, args.V_MODEL_DIR+'agent.ckpt')
            p_model.saver.save(p_model.sess, args.P_MODEL_DIR+'agent.ckpt')
            args.reward_monitor = avg_reward
        
    # print("Optimization on single trajectory data completed! New policy learnt")
    print("Episode ",episode_no,"\n Value model training loss =", avg_v_cost, "\n Policy model training loss =", avg_p_cost,'\n')
    
    return avg_v_cost, avg_p_cost
        
    
def test(args):
        
    env = gym.make(args.env_id)
    
    test_P_agent = MLP_model(env.action_space.n, env.observation_space.shape[0], args.p_learning_rate, args.tf_seed)
    test_V_agent = Value_model( env.observation_space.shape[0], args.v_learning_rate, args.tf_seed)
    
    restore_model(test_V_agent, args.V_MODEL_DIR)
    restore_model(test_P_agent, args.P_MODEL_DIR)
    
    a = np.arange(40)
    np.random.shuffle(a)
    seeds = a[:args.test_env_seed_no] 
    
    introduction_print_file('test_log.txt', args)
    
    # Initialize reward recorder (Over 5 different Seeds)
    for seed in seeds:
        test_reward_vec = []
        env.seed(int(seed))
        for num_test_ep in tqdm(range(args.test_ep)):
            test_ep_reward = 0
            obs = env.reset()
            done = False
            while not done:
                action = test_action(test_P_agent, obs)
                next_obs, reward, done, _ = env.step(action)
                if args.render:
                    env.render()
                test_ep_reward += reward
                obs = next_obs
            test_reward_vec.append(test_ep_reward)
            with open('test_log.txt', 'a') as f:
                print(test_ep_reward,',', num_test_ep,',', seed, file=f)
        
        mean_reward = np.mean(np.array(test_reward_vec))
        error = np.std(np.array(test_reward_vec))
        print("Model performance estimation over 100 trajectories on ",args.test_env_seed_no, "different seed: Average Reward observed = ", mean_reward, '+-', error)
        
        with open('test_log.txt', 'a') as f:
            print("Model performance estimation over 100 trajectories on different seeds: Average Reward observed = ", mean_reward, '+-', error, file=f)

    env.close()
        
    srcs = ["test_log.txt", "reward_log.txt", "v_model", "p_model", "logfile.txt", 'Fig_1.png', 'Fig_2.png']
    dest = str(args.SAVE_DIR+"Test_"+str(args.exp_no)+'/')
    
    if not os.path.exists(dest):
        os.makedirs(dest)
    else:
        assert (True),"Folder already exists. Risk of overwriting data"
    
    for src in srcs:
        move(src, dest)
    
    return mean_reward, error




def main(args):
    np.random.seed(args.np_seed)
    tf.set_random_seed(args.tf_seed)
    
    env = gym.make(args.env_id)
    
    # defining the models and parameters
    policy_agent = MLP_model(env.action_space.n, env.observation_space.shape[0], args.p_learning_rate, args.tf_seed)
    value_agent = Value_model(env.observation_space.shape[0], args.p_learning_rate, args.tf_seed)
    print(np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))
    # Make log file
    make_log_file(args, value_agent, policy_agent)
    
    episode_value = []
    avg_reward_vec = []
    
    open('reward_log.txt', 'w').close()

    # each episode will have an updated policy pi(s,a)
    for num_episode in range(args.num_ep):
        ep_reward = 0
        avg_reward = 0
        state = env.reset()
        traj = {'observations':[],'actions':[]}
        reward_vec = []
        done = False
        if (num_episode %100) == 0:
            args.v_learning_rate *= 1.01
            args.p_learning_rate *= 1.01
            print(args.v_learning_rate,args.p_learning_rate)
        
        while not done:
            action = agent_action(policy_agent, state)
            next_state, reward, done, _ = env.step(action)
            if args.render:
                env.render()
            traj['observations'].append(state)
            traj['actions'].append(action)
            reward_vec.append(reward)
            ep_reward += reward
            state = next_state
            
        traj['values'] = oc_stateValues(reward_vec, args.gamma)
        print('Net reward in episode no: ',num_episode,' = ',ep_reward)
        episode_value.append(ep_reward)
        
        # To store 100 consecutive episodes avg reward
        avg_reward = avg_100_reward(episode_value)
        avg_reward_vec.append(avg_reward)
        
        # Train
        V_cost, P_cost = baseline_train(policy_agent, value_agent, traj, num_episode, avg_reward, args)
        
        # File Data Transfer
        with open('reward_log.txt', 'a') as f:
            print(num_episode,',', ep_reward,',', avg_reward,',', V_cost ,',', P_cost, file=f)            
    
    env.close()	
    	
    	
if __name__ == '__main__':
    args = argsparser()   
    if args.task == 'train':
        main(args)
    if args.task == 'test':
        test(args)
