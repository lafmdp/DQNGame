from collections import deque
import tensorflow as tf
import cv2
import numpy as np
import random

class DQN():
    def __init__(self, n_actions, batch_size = 128, epislon = 0.9, lamda =  0.9, capcity = 30000):
        self.n_actions = n_actions
        self.replay = deque()
        self.batch_size = batch_size
        self.epislon = epislon
        self.capcity = capcity
        self.lamda = lamda

        self.input_image = tf.placeholder("float",[None, 80, 100, 4])
        self.argmax = tf.placeholder("float", [None, self.n_actions])
        self.target = tf.placeholder("float", [None])  # q_target

        self.sess = tf.Session()
        self._build_network()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def _build_network(self):
        # create the convolution netural network
        Weight = {
            'w1' : tf.Variable(tf.truncated_normal([8,8,4,16],stddev=0.1),dtype=tf.float32),
            'w2' : tf.Variable(tf.truncated_normal([4,4,16,32],stddev=0.1),dtype=tf.float32),
            'w3' : tf.Variable(tf.truncated_normal([4,4,32,64],stddev=0.1),dtype=tf.float32),
            'fc' : tf.Variable(tf.truncated_normal([8320, 64])),
            'out': tf.Variable(tf.truncated_normal([64, self.n_actions],stddev=0.1),dtype=tf.float32)
        }

        biases = {
            'b1' : tf.zeros([16], dtype=tf.float32),
            'b2' : tf.zeros([32],dtype=tf.float32),
            'b3' : tf.zeros([64],dtype=tf.float32),
            'fc' : tf.zeros([64],dtype=tf.float32),
            'out': tf.zeros([self.n_actions],dtype=tf.float32)
        }

        con1 = tf.nn.relu(tf.nn.conv2d(self.input_image, Weight['w1'], strides=[1,2,2,1],padding='SAME'),name='con1')
        con2 = tf.nn.relu(tf.nn.conv2d(con1, Weight['w2'], strides=[1,2,2,1],padding='SAME'),name='con2')
        con3 = tf.nn.relu(tf.nn.conv2d(con2, Weight['w3'], strides=[1,2,2,1],padding='SAME'),name='con3')

        con_flat = tf.reshape(con3, [-1, 8320])
        fc4 = tf.nn.relu(tf.matmul(con_flat, Weight['fc']) + biases['fc'],name='fc4')

        self.pred_action = tf.nn.softmax(tf.matmul(fc4, Weight['out']) + biases['out'])

        # define loss function
        action =tf.reduce_sum(tf.multiply(self.pred_action, self.argmax), reduction_indices=1)
        loss = tf.reduce_mean(tf.square(action - self.target))
        self.train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

    def choose_action(self, image):
        action = np.zeros([self.n_actions], dtype=np.int32)

        image = cv2.cvtColor(cv2.resize(image, (100, 80)), cv2.COLOR_BGR2GRAY)
        # 转换为二值
        ret, image = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)
        # the normal data we can caculate
        image = np.stack((image,image,image,image), axis=2)

        image = np.reshape(image,[1,80,100,4])
        pred = self.sess.run(self.pred_action[0],feed_dict={self.input_image : image})

        if np.random.random() < self.epislon:
            maxID = np.argmax(pred)
        else:
            maxID = np.random.randint(0,self.n_actions - 1)

        action[maxID] = 1

        return list(action)

    def train_network(self):
        minibatch = random.sample(self.replay, self.batch_size)

        image1_batch = [d[0] for d in minibatch]
        action_batch = [d[1] for d in minibatch]
        reward_batch = [d[2] for d in minibatch]
        image2_batch = [d[3] for d in minibatch]

        out_batch = self.sess.run(self.pred_action, feed_dict=[image2_batch])

        target_batch = []
        for i in range(0, len(out_batch)):
            target_batch.append(reward_batch[i] + self.lamda * np.max(out_batch[i]))

        feed = {
            self.argmax : action_batch,
            self.target : target_batch,
            self.input_image : image1_batch
        }

        self.sess.run(self.train_step, feed_dict=feed)

    def save_model(self):
        self.saver.save(self.sess,save_path='Model\\atari.cpk')

    def restore_model(self):
        self.saver.restore(self.sess,save_path='Model\\atari.cpk')

    def save_replay(self, image1, action, reward, image2):

        self.replay.append((image1, action, reward, image2))

        if len(self.replay) > self.capcity:
            self.replay.popleft()

