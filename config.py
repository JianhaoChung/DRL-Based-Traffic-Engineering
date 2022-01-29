class NetworkConfig(object):
    scale = 100
    max_step = 1000 * scale
    initial_learning_rate = 0.0001
    learning_rate_decay_rate = 0.96
    learning_rate_decay_step = 5 * scale
    moving_average_decay = 0.9999
    entropy_weight = 0.1

    save_step = 10 * scale

    '''
    max_to_keep: the number of checkpoints to keep. Unless preserved by keep_checkpoint_every_n_hours, checkpoints will be deleted from the active set, oldest first, until only max_to_keep checkpoints remain. 
    If None, no checkpoints are deleted and everything stays in the active set. Note that max_to_keep=None will keep all checkpoint paths in memory and in the checkpoint state protocol buffer on disk.
    '''

    max_to_keep = 1000

    Conv2D_out = 128
    Dense_out = 128
    optimizer = 'RMSprop'
    # optimizer = 'Adam'

    logit_clipping = 10  # 10 or 0, = 0 means logit clipping is disabled


class Config(NetworkConfig):
    version = 'TE_v2'
    project_name = 'CFR-RL'

    method = 'actor_critic'
    # method = 'pure_policy'

    model_type = 'Conv'

    topology_name = ['Abilene', 'Ebone', 'Sprintlink', 'Tiscali']
    topology_file = topology_name[0]

    traffic_file = 'TM'
    test_traffic_file = 'TM2'

    tm_history = 1

    max_moves = 10  # percentage

    # For pure policy
    baseline = 'avg'  # avg, best

    central_links_nums = 10

    partial_tm_zeroing = False  # For scheme 1.0

    suffix = ['alpha', 'alpha+', 'beta', 'beta+', 'beta++', 'gamma', 'deta', 'epsilon']
    model_name_suffix = suffix[1]

    scheme = ['baseline'] + suffix

    cf_influence = 0
    critical_links = 5  # 5 by default


def get_config(FLAGS):
    config = Config

    for k, v in FLAGS.__flags.items():
        if hasattr(config, k):
            setattr(config, k, v.value)

    return config
