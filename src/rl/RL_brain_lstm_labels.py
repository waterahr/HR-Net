import numpy as np
import tensorflow as tf

# reproducible
np.random.seed(1)
tf.set_random_seed(1)

class PolicyGradient:
    def __init__(self,
             n_actions,
             n_features,
             learning_rate = 0.01,
             reward_decay = 0.1,
             output_graph = False,
             weights = None,
             init_step = 1,
             reward_type = "ma_as_reward_one_choose"):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.weights = weights
        self.step = init_step
        self.reward_type = reward_type
        
        #Stores the current episode status, actions and rewards
        self.ep_obs, self.ep_as, self.ep_rs = [], [], []
        
        self._build_net()
        gpuConfig = tf.ConfigProto(allow_soft_placement=True)
        gpuConfig.gpu_options.allow_growth = True  
        self.sess = tf.Session(config=gpuConfig)
        self.sess.run(tf.global_variables_initializer())
        if weights is not None:
            print(self.saver.restore(self.sess, weights))
        
        if output_graph:
            tf.summary.FileWriter("../../results/logs/RL_brain_lstm_labels", self.sess.graph)
        
    def _build_net(self):
        with tf.name_scope('inputs'):
            #current episode state/observation
            self.tf_obs = tf.placeholder(tf.float32, [None, self.n_features, self.n_actions], name="observations")
            #current choosed episode action
            self.tf_acts = tf.placeholder(tf.int32, [None, self.n_features], name="action_nums")
            #reward/action value of current spisode's current step/action
            self.tf_rews = tf.placeholder(tf.float32, [None,], name="action_value")
            
        lstm_cell_1 = tf.contrib.rnn.BasicLSTMCell(num_units = 128)
        lstm_cell_2 = tf.contrib.rnn.BasicLSTMCell(num_units = 64)
        lstm_cell_3 = tf.contrib.rnn.BasicLSTMCell(num_units = self.n_actions)
        lstm_cell = tf.contrib.rnn.MultiRNNCell(cells = [lstm_cell_1, lstm_cell_2, lstm_cell_3])
        
        prob_logs, _ = tf.nn.dynamic_rnn(cell=lstm_cell, inputs=self.tf_obs, dtype=tf.float32)
        #print(prob_logs)###Tensor("concat:0", shape=(?, 61, 3), dtype=float32)
        
        self.probs = tf.nn.softmax(prob_logs, axis=-1, name = 'act_prob')###[None, 61, 3]
        ###print(self.probs)###Tensor("act_prob:0", shape=(?, 61, 3), dtype=float32)

        with tf.name_scope('saver'):
            self.saver = tf.train.Saver(max_to_keep=4, keep_checkpoint_every_n_hours=2)
        
        with tf.name_scope('loss'):
            #neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.all_act_prob,labels =self.tf_acts)

            #print(self.probs[:, 0, :])###Tensor("loss/strided_slice:0", shape=(?, 3), dtype=float32)
            #print(self.tf_acts)###Tensor("inputs/action_nums:0", shape=(?, 61), dtype=int32)
            #print(tf.one_hot(indices=self.tf_acts[:, 0], depth=self.n_actions))###Tensor("loss/one_hot:0", shape=(?, 3), dtype=float32)
            neg_log_prob = tf.reduce_sum(-tf.log(self.probs[:, 0, :]) * tf.one_hot(indices=self.tf_acts[:, 0], depth=self.n_actions), axis=1)
            for i in range(1, self.n_features):
                neg_log_prob += tf.reduce_sum(-tf.log(self.probs[:, i, :]) * tf.one_hot(indices=self.tf_acts[:, i], depth=self.n_actions), axis=1)
            loss = tf.reduce_mean(neg_log_prob * self.tf_rews)
        
        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)
            
    
    def choose_action(self, observation):
        prob_weights = self.sess.run(self.probs, feed_dict={self.tf_obs:observation[np.newaxis, :, :]})###[None, 61, 3]
        action = []###[61,]
        for i in range(prob_weights.shape[1]):
            action.append(np.random.choice(range(prob_weights.shape[2]), p=prob_weights[0, i].ravel()))
        return action
    
    def store_transition(self, s, a, r):
        self.ep_obs.append(s)###[61, 3]
        self.ep_as.append(a)###[61,]
        self.ep_rs.append(r)###[,]
        
    def learn(self):
        discounted_ep_rs_norm = self._discounted_and_norm_rewards()
        print("discounted_ep_rs_norm: ", discounted_ep_rs_norm)
        
        self.sess.run(self.train_op, feed_dict={
            self.tf_obs:np.stack(self.ep_obs),
            self.tf_acts:np.array(self.ep_as),
            self.tf_rews:discounted_ep_rs_norm,
        })
        infor = self.saver.save(self.sess, 
                        '../models/imagenet_models/rl_lstm/'+self.reward_type+'/RL_brain_lstm_labels-step'+str(self.step),
                        global_step=1, 
                        write_meta_graph=False)
        print(infor)
        
        self.ep_obs.clear()
        self.ep_as.clear()
        self.ep_rs.clear()
        self.step += 1
        return discounted_ep_rs_norm
    
    def _discounted_and_norm_rewards(self):
        discounted_ep_rs = np.zeros_like(self.ep_rs)
        running_add = 0
        for t in reversed(range(0, len(self.ep_rs))):
            running_add = running_add * self.gamma + self.ep_rs[t]
            discounted_ep_rs[t] = running_add

        if np.shape(discounted_ep_rs)[0] != 1:
            discounted_ep_rs -= np.mean(discounted_ep_rs)
            discounted_ep_rs /= np.std(discounted_ep_rs)
        return discounted_ep_rs