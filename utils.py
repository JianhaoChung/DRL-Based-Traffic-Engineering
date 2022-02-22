import os
import numpy as np
import networkx as nx
from config import get_config
from absl import flags

FLAGS = flags.FLAGS


class utility:
    def __init__(self, config, data_dir='./data/'):
        self.topology_file = data_dir + config.topology_file
        self.shortest_paths_file = self.topology_file + '_shortest_paths'
        self.DG = nx.DiGraph()
        self.config = config
        self.central_links_nums = config.central_links_nums

    def calculate_linksets(self):
        # print('[*] Loading topology...', self.topology_file)
        f = open(self.topology_file, 'r')
        header = f.readline()
        self.num_nodes = int(header[header.find(':') + 2:header.find('\t')])
        self.num_links = int(header[header.find(':', 10) + 2:])
        f.readline()
        self.link_idx_to_sd = {}
        for line in f:
            link = line.split('\t')
            i, s, d, w, c = link
            self.link_idx_to_sd[int(i)] = (int(s), int(d))
        self.link_sets = []

        for e in self.link_idx_to_sd:
            # print(e, ': ', self.link_idx_to_sd[e])
            self.link_sets.append(self.link_idx_to_sd[e])
        f.close()

    def calcualte_shortest_paths_links(self):
        self.pair_idx_to_sd = []
        self.pair_sd_to_idx = {}
        # Shortest paths
        self.shortest_paths = []
        self.shortest_paths_format = []
        if os.path.exists(self.shortest_paths_file):
            # print('[*] Loading shortest paths...', self.shortest_paths_file)
            f = open(self.shortest_paths_file, 'r')
            self.num_pairs = 0
            for line in f:
                sd = line[:line.find(':')]
                s = int(sd[:sd.find('-')])
                d = int(sd[sd.find('>') + 1:])
                self.pair_idx_to_sd.append((s, d))
                self.pair_sd_to_idx[(s, d)] = self.num_pairs
                self.num_pairs += 1
                self.shortest_paths.append([])
                paths = line[line.find(':') + 1:].strip()[1:-1]
                while paths != '':
                    idx = paths.find(']')
                    path = paths[1:idx]
                    # print(path)
                    node_path = np.array(path.split(',')).astype(np.int16)

                    self.shortest_paths_format.append(path)

                    assert node_path.size == np.unique(node_path).size
                    self.shortest_paths[-1].append(node_path)
                    paths = paths[idx + 3:]

        self.shortest_paths_links = []
        for path in self.shortest_paths_format:
            path = list(map(int, list(path.split(', '))))
            temp = []
            for i in range(len(path) - 1):
                temp.append((path[i], path[i + 1]))
            self.shortest_paths_links.append(temp)
            # print(self.shortest_paths_links)

        # for idx in range(len(self.shortest_paths_links)):
        #     print(self.shortest_paths_format[idx], ": ", self.shortest_paths_links[idx])

    def calcualte_sorted_link_centralization(self, print_=False):
        self.path_links_counter = [idx for idx in range(len(self.link_sets))]
        for idx, link in enumerate(self.link_sets):
            for path_link in self.shortest_paths_links:
                if link in path_link:
                    self.path_links_counter[idx] += 1

        # sorted_link_idx = list(np.argsort(self.path_links_counter))
        sorted_link_idx = sorted(range(len(self.path_links_counter)), key=lambda k: self.path_links_counter[k],
                                 reverse=True)
        self.topk_centralized_links = [self.link_sets[idx] for idx in sorted_link_idx[:self.central_links_nums]]

        link_centrality_mapper = {}
        for idx, link_count in enumerate(self.path_links_counter):
            link_centrality_mapper[self.link_sets[idx]] = link_count

        if print_:
            # print(sorted_link_idx)
            print("Topo link and its count in the shortest paths of OD flows: ", end=' ')
            print(link_centrality_mapper)
            print('\nTopk centralized links: {}'.format(self.topk_centralized_links))
        return link_centrality_mapper

    def scaling_action_space(self, central_influence=1, print_=False):
        self.calculate_linksets()
        self.calcualte_shortest_paths_links()
        link_centrality_mapper = self.calcualte_sorted_link_centralization(print_)
        self.shortest_path_tag = [0 for i in range(len(self.shortest_paths))]
        action_space = []

        for idx, path in enumerate(self.shortest_paths_links):
            if len(set(path).intersection(self.topk_centralized_links)) >= central_influence:
                self.shortest_path_tag[idx] = 1
        for idx, val in enumerate(self.shortest_path_tag):
            if val == 1:
                action_space.append(idx)

        # print('OD flows indexes| target-flows-to-select(1): {}, non-targets(0): {}'
        #       .format(self.shortest_path_tag.count(1), self.shortest_path_tag.count(0)))

        # print(self.shortest_path_tag)

        partial_tm_zeroing = [pair for idx, pair in enumerate(self.pair_idx_to_sd) if self.shortest_path_tag[idx] == 0]

        if print_:
            print(
                "\nTotal number of selected centralized flows: {}\nIndex of OD flows to select critical flows: {}".format(
                    len(action_space), action_space))

        return link_centrality_mapper, self.topk_centralized_links, partial_tm_zeroing, action_space


if __name__ == '__main__':
    config = get_config(FLAGS) or FLAGS
    utility(config=config).scaling_action_space(config.central_influence, print_=True)
