from __future__ import print_function

import warnings

warnings.filterwarnings("ignore")
import time
import numpy as np
import tensorflow as tf
from absl import app
from absl import flags

from config import get_config
from env import Environment
from game import CFRRL_Game
from model import Network

FLAGS = flags.FLAGS
# flags.DEFINE_string('ckpt', '', 'apply a specific checkpoint')
flags.DEFINE_string('ckpt', 'ckpt-36', 'apply a specific checkpoint')
# flags.DEFINE_boolean('eval_delay', False, 'evaluate delay or not')
flags.DEFINE_boolean('eval_delay', True, 'evaluate delay or not')
flags.DEFINE_boolean('central_included', True, 'central link included or not')


def sim(config, network, game):
    start = time.time()
    for tm_idx in game.tm_indexes:
        state = game.get_state(tm_idx)
        if config.method == 'actor_critic':
            policy = network.actor_predict(np.expand_dims(state, 0)).numpy()[0]
        elif config.method == 'pure_policy':
            policy = network.policy_predict(np.expand_dims(state, 0)).numpy()[0]
        actions = policy.argsort()[-game.max_moves:]
        game.evaluate(tm_idx, actions, eval_delay=FLAGS.eval_delay, central=FLAGS.central_included)
    end = time.time()
    print("\nTest time: {} mins {} secs".format((end - start) // 60, (end - start) % 60))


def main(_):
    # Using cpu for testing
    tf.config.experimental.set_visible_devices([], 'GPU')
    tf.get_logger().setLevel('INFO')
    config = get_config(FLAGS) or FLAGS
    env = Environment(config, is_training=False)
    game = CFRRL_Game(config, env)
    network = Network(config, game.state_dims, game.action_dim, game.max_moves)
    step = network.restore_ckpt(FLAGS.ckpt)
    if config.method == 'actor_critic':
        learning_rate = network.lr_schedule(network.actor_optimizer.iterations.numpy()).numpy()
    elif config.method == 'pure_policy':
        learning_rate = network.lr_schedule(network.optimizer.iterations.numpy()).numpy()
    print('\nstep %d, learning rate: %f\n' % (step, learning_rate))

    sim(config, network, game)


if __name__ == '__main__':
    app.run(main)
