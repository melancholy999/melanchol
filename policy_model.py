# Tensorflow low level implementation of multilayer perceptron as Policy Function Approximator

import tensorflow as tf
import numpy as np

class MLP_model():
    
    def __init__(self, n_action, n_observation, LEARNING_RATE, seed):
        
        tf.set_random_seed(seed)
        np.random.seed(seed)
        self.n_input = n_observation
        self.n_classes = n_action
        self.learning_rate = LEARNING_RATE
        self.model = self.tensorflowModel()
        

        
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    
    def tensorflowModel(self): 
        # network parameters
        self.n_hidden_1 = 32
        self.n_hidden_2 = 32
        self.std_dev = 0.2
        self.bias_constant = 0
        self.reg = 0		# Optional
        
        with tf.name_scope('inputs'):
            self.tf_observations = tf.placeholder(tf.float32,[None,self.n_input],name = 'observation')
            self.tf_actions = tf.placeholder(tf.int32, [None,], name = 'num_actions')
            self.tf_values = tf.placeholder(tf.float32, [None,], name ='state_values')

        
        self.weights = {
                'h1': tf.Variable(tf.random_normal([self.n_input,self.n_hidden_1], stddev=self.std_dev)),
                'h2': tf.Variable(tf.random_normal([self.n_hidden_1,self.n_hidden_2],stddev=self.std_dev)),
                'out': tf.Variable(tf.random_normal([self.n_hidden_2,self.n_classes],stddev=self.std_dev))
            }

        
        self.biases = {
                'b1': tf.Variable(tf.constant(self.bias_constant , shape = [self.n_hidden_1])),
                'b2': tf.Variable(tf.constant(self.bias_constant , shape = [self.n_hidden_2])),
                'out': tf.Variable(tf.constant(self.bias_constant , shape = [self.n_classes])),
            }

        # nn layers outputs
        logits = self.multilayer_perceptron()
        
        # softmax
        self.action_probs = tf.nn.softmax(logits,name = 'action')
        
        
        with tf.name_scope('loss'):
            self.regularizer = tf.nn.l2_loss(self.weights['h1'])+tf.nn.l2_loss(self.weights['h2'])+tf.nn.l2_loss(self.weights['out'])
            loss_fn = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits, labels = self.tf_actions)
            self.loss= tf.reduce_mean(tf.multiply(loss_fn, self.tf_values)) + (self.reg * self.regularizer) 
            
        with tf.name_scope('train'):
            self.optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate).minimize(self.loss)
        
    
    def multilayer_perceptron(self):
        # Hidden Layer with ReLU activation
        layer1 = tf.add(tf.matmul(self.tf_observations,self.weights['h1']),self.biases['b1'])
        layer1 = tf.nn.tanh(layer1)
        #Hidden Layer with ReLU activation
        layer2 = tf.add(tf.matmul(layer1,self.weights['h2']),self.biases['b2'])
        layer2 = tf.nn.tanh(layer2)
        # Output layer with linear activation
        outlayer = tf.matmul(layer2,self.weights['out']) + self.biases['out']
        return outlayer

