class NetworkConfig(object):
    # scale = 100
    scale = 50
    # scale = 5 # demo
    max_step = 1000 * scale
    # save_step = 10 * scale
    save_step = 10 * scale
    max_to_keep = 1000
    initial_learning_rate = 0.0001
    learning_rate_decay_rate = 0.96
    learning_rate_decay_step = 5 * scale
    moving_average_decay = 0.9999
    entropy_weight = 0.1
    Conv2D_out = 128
    Dense_out = 128
    optimizer = 'RMSprop'  # 'Adam'
    logit_clipping = 10  # 10 or 0, = 0 means logit clipping is disabled


class Config(NetworkConfig):
    version = 'TE_v2'
    project_name = 'CFR-RL'
    # method = 'actor_critic'
    method = 'pure_policy'
    model_type = 'Conv'
    topology_name = ['Abilene', 'Ebone', 'Sprintlink', 'Tiscali']
    topology_idx = 1
    topology_file = topology_name[topology_idx]
    traffic_file = 'TM'
    test_traffic_file = 'TM2'
    tm_history = 1
    baseline = 'avg'  # avg, best;  For pure policy

    # Schemes
    suffix = ['baseline', 'alpha', 'alpha_update', 'alpha+']
    scheme_idx = 2
    scheme = model_name_suffix = suffix[scheme_idx]

    max_moves_list = [5, 10, 15, 20, 30]
    percentage_idx = 2
    max_moves = max_moves_list[percentage_idx]  # 10 for percentage default

    central_flow_sampling_ratio_list = [0.3, 0.5, 0.6, 0.7, 0.75, 0.8, 1]
    cfr_idx = -3
    central_flow_sampling_ratio = central_flow_sampling_ratio_list[cfr_idx]

    # More Details of Schemes (Flow Hibird Sampling) #
    scheme_list = [None, 'lastK_centralized_sample', 'lastK_sample']
    scheme_idx = 1
    scheme_explore = scheme_list[scheme_idx]

    # Bottleneck Flows Configuration #
    central_links_nums = 10
    cf_influence = 1  # 1 by default
    central_influence = 1  # 1 by deafault
    partial_tm_zeroing = False  # False by default
    critical_links = 5  # 5 by default

    label_name = ['PKE-DRL', 'CFR-RL', 'TopK Critical', 'TopK Cum-Centrality', 'TopK Centralized', 'TopK', 'ECMP']


def get_config(FLAGS):
    config = Config
    for k, v in FLAGS.__flags.items():
        if hasattr(config, k):
            setattr(config, k, v.value)
    return config
