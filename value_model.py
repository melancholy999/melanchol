# Tensorflow implementation of MLP as Value Function Approximator
# Activation Function: RELU


import tensorflow as tf
import numpy as np

class Value_model():
    
    def __init__(self, n_observation, LEARNING_RATE, seed):
        tf.set_random_seed(seed)
        np.random.seed(seed)
        self.n_input = n_observation
        self.reg = 0.1
        
        self.learning_rate = LEARNING_RATE
        self.model = self.tensorflowModel()
        
        
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
    
    def tensorflowModel(self):
        # network parameters
        self.n_hidden_1 = 32	
        self.n_hidden_2 = 16
        self.n_classes = 1
        self.std_dev = 0.2
        self.bias_constant = 0.0
        
        with tf.name_scope('inputs'):
            self.tf_observations = tf.placeholder(tf.float32,[None,self.n_input],name = 'observation')
            self.tf_values = tf.placeholder(tf.float32, [None,], name ='state_values')

        
        self.weights = {
                'h1': tf.Variable(tf.random_normal([self.n_input,self.n_hidden_1],stddev=self.std_dev)),
                'h2': tf.Variable(tf.random_normal([self.n_hidden_1,self.n_hidden_2],stddev=self.std_dev)),
                'out': tf.Variable(tf.random_normal([self.n_hidden_2,self.n_classes], stddev=self.std_dev))
            }
        
        self.biases = {
                'b1': tf.Variable(tf.constant(self.bias_constant, shape = [self.n_hidden_1])),
                'b2': tf.Variable(tf.constant(self.bias_constant, shape = [self.n_hidden_2])),
                'out': tf.Variable(tf.constant(self.bias_constant, shape = [self.n_classes])),
            }
        
        
        logits = self.multilayer_perceptron()
        
        self.value = logits
        
        with tf.name_scope('loss'):
            self.regularizer = tf.nn.l2_loss(self.weights['h1'])+tf.nn.l2_loss(self.weights['h2'])+tf.nn.l2_loss(self.weights['out'])
            self.loss = tf.losses.mean_squared_error(labels = [self.tf_values], predictions = tf.transpose(logits)) + (self.reg * self.regularizer) 
        
        with tf.name_scope('train'):
            self.optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate).minimize(self.loss)
    
    def multilayer_perceptron(self):
        # Hidden Layer with ReLU activation
        layer1 = tf.add(tf.matmul(self.tf_observations,self.weights['h1']),self.biases['b1'])
        layer1 = tf.nn.relu(layer1)
        #Hidden Layer with ReLU activation
        layer2 = tf.add(tf.matmul(layer1,self.weights['h2']),self.biases['b2'])
        layer2 = tf.nn.relu(layer2)
        # Output layer with linear activation
        outlayer = tf.matmul(layer2,self.weights['out']) + self.biases['out']
        return outlayer
