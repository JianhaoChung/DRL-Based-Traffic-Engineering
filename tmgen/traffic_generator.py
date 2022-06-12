import pandas
from tmgen.models import exp_tm
from tmgen.models import random_gravity_tm
from tmgen.models import uniform_tm

from tmgen import TrafficMatrix


def uniform_tm_generator(path, name, node_num, low, high, num_epochs):
    """
    Parameters:
    num_nodes – number of points-of-presence
    low – lowest allowed value
    high – highest allowed value
    num_epochs – number of
    Returns:
    TrafficMatrix object
    :return:
    """
    tm = uniform_tm(node_num, low, high, num_epochs)
    # print(tm)
    traffic_matrix_saver(tm, path, name)


def exponential_tm_generator(path, name, node_num, mean_traffic, num_epochs):
    """
    Exponential traffic matrix. Values are drawn from an exponential distribution, with a mean value of mean_traffic.
    Parameters:
    num_nodes – number of nodes in the network
    mean_traffic – mean value of the distribution
    num_epochs – number of epochs in the traffic matrix
    Returns:
    TrafficMatrix object
    """
    tm = exp_tm(node_num, mean_traffic, num_epochs)
    traffic_matrix_saver(tm, path, name)


def random_gravity_tm_generator(num_nodes, mean_traffic):
    """
    :parameters:
    num_nodes – number of nodes in the network
    mean_traffic – average traffic volume between a pair of nodes
    :return: a new TrafficMatrix
    """
    return random_gravity_tm(num_nodes, mean_traffic)


def traffic_matrix_loader(path, name):
    return TrafficMatrix.from_pickle(path + '/' + name + '.pickle')


def traffic_matrix_saver(tm, path, name):
    tm.to_pickle(path + '/' + name + '.pickle')
    tm.to_csv(path + '/' + name + '.csv')


def tm_between_an_OD_flow(tm, ingress, egress):
    return tm.between(ingress, egress)


def tm_transpose_and_save(file, save_path=None, num_nodes=23, diagonal_drop=False):
    df = pandas.read_csv(file, header=None, index_col=False)
    df = df.transpose()
    # print(df.info, df.shape)
    if diagonal_drop:
        zero_position = [i * num_nodes + i for i in range(num_nodes)]
        # print(zero_position)
        df = df.drop(zero_position, axis=1)
        # df = df.loc[0:, 0:]
        print(df.info, df.shape)

    df.to_csv(save_path, sep=' ', header=False, index=False)


def random_sample_TM(file1, file2):
    df_e = pandas.read_csv(file1, header=None, index_col=False)
    df_u = pandas.read_csv(file2, header=None, index_col=False)

    train_e = df_e.sample(frac=0.7, replace=False)
    test_e = df_e[~df_e.index.isin(train_e.index)]

    train_u = df_u.sample(frac=0.7, replace=False)
    test_u = df_u[~df_u.index.isin(train_u.index)]

    train = train_e.append(train_u)
    test = test_e.append(test_u)

    train_saver = '/home/scnu-go/ProjectsSCNU/CFR-RL/data/EboneTM.csv'
    test_saver = '/home/scnu-go/ProjectsSCNU/CFR-RL/data/EboneTM2.csv'

    train.to_csv(train_saver,  header=False, index=False)
    test.to_csv(test_saver,  header=False, index=False)


if __name__ == '__main__':
    EET = '/home/scnu-go/ProjectsSCNU/CFR-RL/data/Ebone_ExponentialTM.csv'
    EUT = '/home/scnu-go/ProjectsSCNU/CFR-RL/data/Ebone_UniformTM.csv'
    random_sample_TM(EET, EUT)

    exit(1)

    traffic_data_path = '/home/scnu-go/ProjectsSCNU/CFR-RL/data/tmgen_raw/'
    tm_save_path = '/home/scnu-go/ProjectsSCNU/CFR-RL/data/'
    Topology_Name = ['Ebone', 'Sprintlink', 'Tiscali']

    Topo_Idx = 0
    node_num = 23
    num_epochs = 50  # 30
    # print_detail = True
    print_detail = False

    uniform_traffic_data = Topology_Name[Topo_Idx] + '_UniformTraffic'
    exponential_traffic_data = Topology_Name[Topo_Idx] + '_ExponentialTraffic'

    uniform_tm_generator(traffic_data_path, uniform_traffic_data,
                         node_num=node_num, num_epochs=num_epochs, low=400, high=600)  # 30， 500

    exponential_tm_generator(traffic_data_path, exponential_traffic_data,
                             node_num=node_num, mean_traffic=400, num_epochs=num_epochs)  # 100

    if print_detail:
        uniform_tm = traffic_matrix_loader(tm_save_path, uniform_traffic_data)
        exponential_tm = traffic_matrix_loader(tm_save_path, exponential_traffic_data)
        print('num_nodes: {}, num_epochs: {}'
              .format(uniform_tm.num_nodes(), uniform_tm.num_epochs()))
        # print(uniform_tm.at_time(1))
        print(tm_between_an_OD_flow(uniform_tm, 0, 1))

    uniform_csv_file = traffic_data_path + uniform_traffic_data + '.csv'
    exponential_csv_file = traffic_data_path + exponential_traffic_data + '.csv'

    uniform_tm_name = Topology_Name[Topo_Idx] + '_UniformTM'
    exponential_tm_name = Topology_Name[Topo_Idx] + '_ExponentialTM'

    # uniform_tm_name = Topology_Name[Topo_Idx] + '_UniformTM2'
    # exponential_tm_name = Topology_Name[Topo_Idx] + '_ExponentialTM2'

    tm_transpose_and_save(file=uniform_csv_file, save_path=tm_save_path + uniform_tm_name + '.csv')
    tm_transpose_and_save(file=exponential_csv_file, save_path=tm_save_path + exponential_tm_name + '.csv')
