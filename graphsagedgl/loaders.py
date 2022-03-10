import json
import dgl
import torch
import torchtext


class JSONDatasetLoader(object):
    def __init__(self, path, a_type, b_type, a2b_type, b2a_type):
        self.path = path
        self.a_type = a_type
        self.b_type = b_type
        self.a2b_type = a2b_type
        self.b2a_type = b2a_type
        self.graph = None
        self.train_graph = None
        self.test_graph = None

    def create_graph(self):
        a2b = (self.a_type, self.a2b_type, self.b_type)
        b2a = (self.b_type, self.b2a_type, self.a_type)
        edges_per_type = {
            a2b: ([], []),
            b2a: ([], [])
        }

        unique_a_nodes = set()
        unique_b_nodes = set()
        idx2a = []
        idx2b = []
        a2idx = {}
        b2idx = {}

        with open(self.path, 'r') as f:
            for line in f:
                data = json.loads(line)
                if data[self.a_type] in unique_a_nodes:
                    a_idx = a2idx[data[self.a_type]]
                else:
                    a_idx = len(unique_a_nodes)
                    unique_a_nodes.add(data[self.a_type])
                    idx2a.append(data[self.a_type])
                    a2idx[data[self.a_type]] = a_idx
                if data[self.b_type] in unique_b_nodes:
                    b_idx = b2idx[data[self.b_type]]
                else:
                    b_idx = len(unique_b_nodes)
                    unique_b_nodes.add(data[self.b_type])
                    idx2b.append(data[self.b_type])
                    b2idx[data[self.b_type]] = b_idx

                edges_per_type[a2b][0].append(a_idx)
                edges_per_type[a2b][1].append(b_idx)
                edges_per_type[b2a][0].append(b_idx)
                edges_per_type[b2a][1].append(a_idx)

        self.graph = dgl.heterograph(
            edges_per_type,
            {self.a_type: len(unique_a_nodes), self.b_type: len(unique_b_nodes)}
            )
        return self.graph

    def add_features(self, feature_names):
        features = {feature_name: [] for feature_name in feature_names}
        with open(self.path, 'r') as f:
            for line in f:
                data = json.loads(line)
                for feature_name in feature_names:
                    features[feature_name].append(data[feature_name])
        for feature_name in features:
            first_val = features[feature_name][0]
            if isinstance(first_val, int):
                self.graph.edges[self.a2b_type].data[feature_name] = torch.IntTensor(features[feature_name])
                self.graph.edges[self.b2a_type].data[feature_name] = torch.IntTensor(features[feature_name])
            elif isinstance(first_val, float):
                self.graph.edges[self.a2b_type].data[feature_name] = torch.FloatTensor(features[feature_name])
                self.graph.edges[self.b2a_type].data[feature_name] = torch.FloatTensor(features[feature_name])
            elif isinstance(first_val, str):
                tokenizer = torchtext.data.utils.get_tokenizer("basic_english")
                tokenized = [tokenizer(line) for line in features[feature_name]]
                vocab = torchtext.vocab.build_vocab_from_iterator(tokenized)
                numericalized = []
                max_len = 0
                for x in features[feature_name]:
                    numericalized.append([vocab[token] for token in tokenizer(x)])
                    if len(numericalized[-1]) > max_len:
                        max_len = len(numericalized[-1])
                padded = torch.zeros(len(numericalized), max_len) - 1
                for i, x in enumerate(numericalized):
                    padded[i, :len(x)] = torch.IntTensor(x)
                self.graph.edges[self.a2b_type].data[feature_name] = padded
                self.graph.edges[self.b2a_type].data[feature_name] = padded

    def train_test_split(self, nodes_to_split, test_size=.2):
        if nodes_to_split not in [self.a_type, self.b_type]:
            raise ValueError(f"nodes_to_split must be either '{self.a_type}' or '{self.b_type}'")
        if nodes_to_split == self.a_type:
            nodes_to_keep = self.b_type
        else:
            nodes_to_keep = self.a_type

        total_size = self.graph.nodes(nodes_to_split).size(0)
        if type(test_size) == float:
            test_size = int(test_size * total_size)
        if test_size > total_size:
            raise ValueError(f"test_size must be less than {total_size}")
        train_size = total_size - test_size
        perm = torch.randperm(total_size)
        train_idx = perm[:train_size]
        test_idx = perm[train_size:]
        self.train_graph = self.graph.subgraph(
            {nodes_to_split: train_idx, nodes_to_keep: self.graph.nodes(nodes_to_keep)}
        )
        self.test_graph = self.graph.subgraph(
            {nodes_to_split: test_idx, nodes_to_keep: self.graph.nodes(nodes_to_keep)}
        )
        return self.train_graph, self.test_graph


class RedditDatasetLoader(JSONDatasetLoader):
    def __init__(self, path):
        super().__init__(path, 'author', 'subreddit', 'posted_to', 'posted_by')

    def add_reddit_features(self):
        feature_names = ['body']
        self.add_features(feature_names)

    def train_test_split(self, test_size=.2):
        return super().train_test_split('author', test_size)
