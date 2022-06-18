class NetworkConfig(object):
    scale = 50
    max_step = 1000 * scale
    save_step = 10 * scale
    max_to_keep = 1000
    initial_learning_rate = 0.0001
    learning_rate_decay_rate = 0.96
    learning_rate_decay_step = 5 * scale
    moving_average_decay = 0.9999
    entropy_weight = 0.1
    Conv2D_out = 128
    Dense_out = 128
    optimizer = 'RMSprop'
    logit_clipping = 10  # 10 or 0, = 0 means logit clipping is disabled


class Config(NetworkConfig):
    version = 'TE_v2'
    project_name = 'CFR-RL'
    method = 'pure_policy'
    model_type = 'Conv'
    traffic_file = 'TM'
    test_traffic_file = 'TM2'
    tm_history = 1
    baseline = 'avg'  # avg or best;

    topology_name = ['Abilene', 'Ebone']
    topology_idx = 0

    max_moves_list = [5, 10, 15, 20]  # different ratio of OD flows to reroute
    percentage_idx = 3

    scheme_list = ['baseline', 'alpha_update']  # baseline means CFR-RL and alpha_update means PKE-DRL
    scheme_idx = 1

    central_flow_sampling_ratio_list = [0.3, 0.5, 0.6, 0.7, 0.75, 0.8, 1]
    cfr_idx = -3

    scheme = scheme_list[scheme_idx]
    topology_file = topology_name[topology_idx]
    max_moves = max_moves_list[percentage_idx]
    central_flow_sampling_ratio = central_flow_sampling_ratio_list[cfr_idx]

    scheme_suffix = None

    if scheme is not 'baseline':
        scheme_suffix = 'lastK_centralized_sample'

    label_name = ['PKE-DRL', 'CFR-RL', 'TopK Critical', 'TopK Cum-Centrality',
                  'TopK Centralized', 'TopK', 'ECMP']

    # round = None # round = 1
    round = 2

    # Bottleneck Flows Configuration (Just Ignore) #
    central_links_nums = 10
    cf_influence = 1  # 1 by default
    central_influence = 1  # 1 by deafault
    partial_tm_zeroing = False  # False by default
    critical_links = 5  # 5 by default


def get_config(FLAGS):
    config = Config
    for k, v in FLAGS.__flags.items():
        if hasattr(config, k):
            setattr(config, k, v.value)
    return config
