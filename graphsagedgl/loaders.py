import json
import dgl


class JSONDatasetLoader(object):
    def __init__(self, path, a_type, b_type, a2b_type, b2a_type):
        self.path = path
        self.a_type = a_type
        self.b_type = b_type
        self.a2b_type = a2b_type
        self.b2a_type = b2a_type
        self.graph = None

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


class RedditDatasetLoader(JSONDatasetLoader):
    def __init__(self, path):
        super().__init__(path, 'author', 'subreddit', 'posted_to', 'posted_by')
