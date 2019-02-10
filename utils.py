'''
Common code blocks.
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import shutil,os

def agent_action(p_model, observation):   
    prob = p_model.sess.run(p_model.action_probs, feed_dict = {p_model.tf_observations: [observation]})
    action = np.random.choice(np.arange(len(prob[0])), p = prob[0])
    return action

def test_action(p_model, observation):
    prob = p_model.sess.run(p_model.action_probs, feed_dict = {p_model.tf_observations: [observation]})
    action = np.argmax(prob[0])
    return action

def agent_value(v_model, observations):   
    pred_value = v_model.sess.run(v_model.value, feed_dict = {v_model.tf_observations: observations})
    return pred_value
    

def oc_stateValues(reward, GAMMA):
    l = len(reward)
    for i in range(l-1):
        reward[l-i-2] += (GAMMA * reward[l-i-1]) 
    reward = np.array(reward)
#    reward -= np.mean(reward)
#    reward /= np.std(reward)
    return reward


def restore_model(model, MODEL_DIR):
    model.saver.restore(model.sess, MODEL_DIR + 'agent.ckpt')
    

def introduction_print_file(file_name, args):
    with open(file_name, 'w') as f:
        print('Test Results of Agent on ', args.env_id,' with 100 episodes of each seed', file =f)
        

def avg_100_reward(episode_reward_vec):
    l = len(episode_reward_vec)
    if l<=100:
        return np.mean(np.array(episode_reward_vec))
    vec_100 = episode_reward_vec[l-100:l]
    avg_value = np.mean(np.array(vec_100))
    return avg_value
    
     
def move(src, dest):
    shutil.move(src,dest)
    
    
def make_log_file(args, v_model, p_model):
    with open('logfile.txt', 'w') as f:
        print("Monte Carlo REINFORCE Policy Gradient with Baseline algorithm \nAuthor: Varshil Gandhi \nDate: 16th December, 2018 \nAlgo Source: David Silver Lectures & Rich Sutton RL Book", file = f)
        print('\nThe variables used in the experiment:', file = f)
        print('-----------------------------------------------', file = f)
        for keys, values in vars(args).items():
            print('| ',keys, ' : ' , values, ' |', file= f)
            
        print('\nValue Model Parameters', file = f)
        print('_____________________________________________', file= f)
        print('| Neurons in the First Hidden Layer | ', v_model.n_hidden_1,'  |', file = f)
        print('| Neurons in the Second Hidden Layer| ', v_model.n_hidden_2,'  |', file = f) 
        print('| Std Deviation in weights of Layers| ', v_model.std_dev, '|', file = f)
        print('| Constant in bias vectors of Layers| ', v_model.bias_constant, ' |', file = f)
        print('| Regularization Constant           | ', v_model.reg, ' |', file = f)
        print('_____________________________________________', file = f)
        
        print('\nPolicy Model Parameters', file = f)
        print('_____________________________________________', file= f)
        print('| Neurons in the First Hidden Layer | ', p_model.n_hidden_1,'  |', file = f)
        print('| Neurons in the Second Hidden Layer| ', p_model.n_hidden_2,'  |', file = f) 
        print('| Std Deviation in weights of Layers| ', p_model.std_dev, '|', file = f)
        print('| Constant in bias vectors of Layers| ', p_model.bias_constant, ' |', file = f)
        print('_____________________________________________', file = f)

