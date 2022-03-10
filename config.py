class NetworkConfig(object):
    # scale = 100
    scale = 50
    max_step = 1000 * scale
    initial_learning_rate = 0.0001
    learning_rate_decay_rate = 0.96
    learning_rate_decay_step = 5 * scale
    moving_average_decay = 0.9999
    entropy_weight = 0.1

    save_step = 10 * scale
    max_to_keep = 1000

    Conv2D_out = 128
    Dense_out = 128

    optimizer = 'RMSprop'
    # optimizer = 'Adam'

    logit_clipping = 10  # 10 or 0, = 0 means logit clipping is disabled


class Config(NetworkConfig):
    version = 'TE_v2'
    project_name = 'CFR-RL'
    # method = 'actor_critic'
    method = 'pure_policy'

    model_type = 'Conv'

    topology_name = ['Abilene', 'Ebone', 'Sprintlink', 'Tiscali']
    topology_file = topology_name[0]
    traffic_file = 'TM'
    test_traffic_file = 'TM2'
    tm_history = 1
    max_moves = 10  # percentage

    # For pure policy
    baseline = 'avg'  # avg, best
    # baseline = 'best'  # avg, best

    partial_tm_zeroing = False  # False by default
    # partial_tm_zeroing = True

    suffix = ['baseline', 'alpha',  'alpha+', 'alpha++', 'beta',
              'beta+', 'beta++', 'beta+++', 'betas+++', 'beta++++',
              'delta']

    debug_suffix = ['debug', 'debug+', 'debug++', 'debug+++', 'debug++++']

    scheme = model_name_suffix = suffix[1]
    # scheme = model_name_suffix =debug_suffix[-1]

    # scheme_explore = ''
    scheme_explore = 'lastK_centralized_sample'  # lastK_sample

    central_links_nums = 10
    cf_influence = 1    # 1 by default
    central_influence = 1  # 1 by deafault

    critical_links = 5  # 5 by default


def get_config(FLAGS):
    config = Config
    for k, v in FLAGS.__flags.items():
        if hasattr(config, k):
            setattr(config, k, v.value)

    return config
