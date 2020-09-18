import gym
import shutil
import numpy as np
import  tensorflow as tf
import os
import threading
from matplotlib import pyplot as plt

env_name = "Pendulum-v0"
intro_env = gym.make(env_name)

n_states = intro_env.observation_space.shape[0]
n_actions = intro_env.action_space.shape[0]
action_bound = [intro_env.action_space.low, intro_env.action_space.high]

global_scope = "GlobalNetwork"
n_actor_neurons = 256
n_critic_neurons = 128

log_dir = "./logs/"

MAX_EPISODE = 2000
MAX_STEPS = 200
update_interval = 10
discount_factor = 0.9
ent_coef = 0.01

n_workers = os.cpu_count()
global_running_rewards =[]
episode = 0


class ActorCritic:

    def __init__(self, name):
        self.scope = name
        self.actor_opt = tf.train.RMSPropOptimizer(0.0001)
        self.critic_opt = tf.train.RMSPropOptimizer(0.001)

        if self.scope == global_scope:
            with tf.variable_scope(self.scope):
                self.state_p = tf.placeholder( dtype=tf.float32, shape=[None, n_states],name="state_placeholder" )
                self.a_params, self.c_params = self.build_model()[-2:]
        else:
            with tf.variable_scope(self.scope):

                self.state_p = tf.placeholder( dtype=tf.float32,shape=[None, n_states], name="state_placeholder" )
                self.v_target_p = tf.placeholder(dtype = tf.float32, shape = [None, 1], name = "v_target_p")
                self.actions_p = tf.placeholder( dtype=tf.float32, shape=[None, n_actions], name="actions_p" )

                self.mu, self.sigma, self.v, self.a_params, self.c_params = self.build_model()

                with tf.name_scope("wrap_actor"):
                    self.mu *= action_bound[1]
                    self.sigma += 1e-4

                a_dist = tf.distributions.Normal(self.mu, self.sigma, name = "actor_distribution")
                adv_error = tf.subtract(self.v_target_p, self.v)

                with tf.name_scope("c_loss"):
                    critic_loss = tf.reduce_mean(tf.square(adv_error))

                with tf.name_scope( "a_loss" ):
                    entropy = a_dist.entropy()
                    a_loss = a_dist.log_prob(self.actions_p ) * tf.stop_gradient( adv_error )
                    actor_loss = - tf.reduce_mean( a_loss + ent_coef * entropy )

                with tf.name_scope("grads"):
                    actor_grads = tf.gradients(actor_loss, self.a_params)
                    critic_grads = tf.gradients( critic_loss, self.c_params )

                with tf.variable_scope("choose_action"):
                    self.action = tf.clip_by_value(tf.squeeze(a_dist.sample(n_actions), axis=0),
                                                   action_bound[0],
                                                   action_bound[1])

            with tf.name_scope("sync"):
                with tf.name_scope("pull"):
                    self.pull_actor_op = [local_params.assign(global_params) for local_params, global_params in
                                              zip(self.a_params, global_ac.a_params)]
                    self.pull_critic_op = [local_params.assign( global_params ) for local_params, global_params in
                                               zip( self.c_params, global_ac.c_params )]
                    with tf.name_scope( "push" ):
                        self.push_actor_op = self.actor_opt.apply_gradients( zip( actor_grads,
                                                                                  global_ac.a_params ) )

                        self.push_critic_op = self.critic_opt.apply_gradients( zip( critic_grads,
                                                                                    global_ac.c_params ) )

    def push_global(self, states, actions, value):

        sess.run([self.push_actor_op, self.push_critic_op],
                 feed_dict={self.state_p: states,
                            self.actions_p: actions,
                            self.v_target_p: value})

    def pull_global(self):
        sess.run([self.pull_actor_op, self.pull_critic_op])

    def build_model(self):

        with tf.variable_scope("actor"):

            actor_layer = tf.layers.dense(self.state_p,
                                          units = n_actor_neurons,
                                          activation = tf.nn.relu,
                                          kernel_initializer = tf.initializers.he_normal(),
                                          name = "actor_hidden_layer")
            mu = tf.layers.dense(actor_layer,
                                 units = n_actions,
                                 activation = tf.nn.tanh,
                                 kernel_initializer = tf.contrib.layers.xavier_initializer(),
                                 name = "mu_layer")
            sigma = tf.layers.dense(actor_layer,
                                    units = n_actions,
                                    activation = tf.nn.softplus,
                                    kernel_initializer = tf.contrib.layers.xavier_initializer(),
                                    name = "sigma_layer")

        with tf.variable_scope("critic"):
            critic_layer = tf.layers.dense(self.state_p ,
                                           units = n_critic_neurons,
                                           activation = tf.nn.relu,
                                           kernel_initializer = tf.initializers.he_normal(),
                                           name = "critic_hidden_layer")
            value = tf.layers.dense(critic_layer,
                                    units = 1,
                                    activation = None,
                                    kernel_initializer = tf.contrib.layers.xavier_initializer(),
                                    name = "Value_layer")

        actor_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope + "/actor")
        critic_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope + "/critic")


        return mu, sigma, value, actor_params, critic_params


    def select_action(self, state):
        state = np.expand_dims(state, axis = 0)
        action = sess.run(self.action, feed_dict={self.state_p: state})[0]
        return  action

    def get_value(self, state):
        state = np.expand_dims(state, axis = 0)
        value = sess.run(self.v, feed_dict={self.state_p: state})[0, 0]
        return  value

class Worker:
    def __init__(self, name):
        self.name = name
        self.env = gym.make(env_name)
        self.ac = ActorCritic(self.name)

    def work(self):

        global episode, global_running_rewards

        actions = []
        states = []
        rewards = []

        """To implement it is so important to use <<while>> and <<global episode>>
           Because in <<for>> version each agent sees only its own episode and its own MAX_EPISODE
           while on the other hand all of them have to see an unique MAX_EPISODE and share <<episode>> number.
        """

        while not coord.should_stop() and episode < MAX_EPISODE:
        # for episode in range(MAX_EPISODE): # Not Asynchronous
            state = self.env.reset()
            episode_reward = 0

            for step in range(1, MAX_STEPS + 1):
                # if self.name == "W_0":
                #     self.env.render()
                action = self.ac.select_action(state)
                next_state, reward, done, _ = self.env.step(action)
                done = True if done or step == MAX_STEPS else False

                states.append(state)
                actions.append(action)
                rewards.append((reward + 8) /8)

                episode_reward += reward

                if done or step % update_interval == 0:
                    if done:
                        v_target = 0
                    else:
                        v_target = self.ac.get_value( next_state )

                    v_targets = []
                    for r in rewards[::-1]:
                        v_target = r + discount_factor * v_target
                        v_targets.append( v_target )
                    v_targets.reverse()

                    actions = np.vstack(actions)
                    states = np.vstack(states)
                    v_targets = np.vstack(v_targets)

                    self.ac.push_global(states, actions, v_targets)
                    self.ac.pull_global()
                    actions = []
                    states = []
                    rewards = []
                if done:
                    if len(global_running_rewards) == 0:
                        global_running_rewards.append(episode_reward)
                    else:
                        global_running_rewards.append(0.9 * global_running_rewards[-1] + 0.1 * episode_reward)
                    print(
                        self.name,
                        "Ep:", episode,
                        "| Ep_r: %i" % global_running_rewards[-1],
                        )

                    break
                state = next_state
            episode += 1

            # print("Episode No.{},Episode reward:{}".format(episode, int(episode_reward)))


if __name__ == '__main__':
    print('n_states: ', n_states)
    print('n_actions: ', n_actions)
    print('action_bounds: ', action_bound)
    print('cpu_count: ', n_workers)

    train = False
    if train:
        with tf.device('/cpu:0'):
            global_ac = ActorCritic(global_scope)
            workers = []
            for i in range(n_workers):
                workers.append(Worker("W_{}".format(i)))
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()

        threads = []
        for worker in workers:
            thread = threading.Thread(target=worker.work)
            thread.start()
            threads.append(thread)

        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)
        tf.summary.FileWriter(log_dir, sess.graph)

        coord.join(threads)
        saver = tf.train.Saver()
        saver.save(sess, 'model_test_now/pendulum.ckpt')
        plt.figure()
        plt.plot(np.arange(len(global_running_rewards)), global_running_rewards)
        plt.xlabel('step')
        plt.ylabel('Total moving reward')
        plt.savefig('rewards_1.png')
        plt.show()
    else:
        sess = tf.Session()
        with tf.device('/cpu:0'):
            global_ac = ActorCritic(global_scope)
            agent = Worker("W_2")
        saver = tf.train.Saver()
        saver.restore(sess, 'model_test_now/pendulum.ckpt')
        env = wrappers.Monitor(agent.env, "./videos", video_callable=lambda episode_id: True, force=True)
        state = env.reset()
        for _ in range(5000):
            agent.env.render()
            x = input()
            action = agent.ac.select_action(state)
            next_state, reward, done, info = env.step(action)
            state = next_state
            if done:
                state = env.reset()
                print('env has been reset')
