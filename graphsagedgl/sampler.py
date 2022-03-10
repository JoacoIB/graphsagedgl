import torch
import dgl


class Node2NodeSampler(torch.utils.data.IterableDataset):
    def __init__(self, graph, batch_size, a_type, b_type, a2b_type, b2a_type):
        self.graph = graph
        self.batch_size = batch_size
        self.a_type = a_type
        self.b_type = b_type
        self.a2b_type = a2b_type
        self.b2a_type = b2a_type

    def __iter__(self):
        while True:
            anchor = torch.randint(0, self.graph.number_of_nodes(self.a_type), (self.batch_size,))
            positive = dgl.sampling.random_walk(
                self.graph,
                anchor,
                metapath=[self.a2b_type, self.b2a_type]
            )[0][:, 2]
            negative = torch.randint(0, self.graph.number_of_nodes(self.a_type), (self.batch_size,))
            yield anchor, positive, negative
