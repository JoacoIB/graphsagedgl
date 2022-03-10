import unittest
from graphsagedgl.loaders import RedditDatasetLoader, JSONDatasetLoader


class TestLoading(unittest.TestCase):
    def test_load_graph(self):
        loader = RedditDatasetLoader('tests/data/loader_data.json')
        g = loader.create_graph()
        self.assertEqual(
            g.number_of_nodes('author'),
            2
        )
        self.assertEqual(
            g.number_of_nodes('subreddit'),
            3
        )
        self.assertEqual(
            g.number_of_edges(),
            10
        )

    def test_load_reddit_features(self):
        loader = RedditDatasetLoader('tests/data/loader_data.json')
        _ = loader.create_graph()
        loader.add_reddit_features()
        self.assertEqual(loader.graph.edges[loader.a2b_type].data["body"].shape, (5, 4))
        self.assertEqual(loader.graph.edges[loader.b2a_type].data["body"].shape, (5, 4))

    def test_load_custom_features(self):
        loader = JSONDatasetLoader('tests/data/loader_data.json', 'author', 'subreddit', 'posted_to', 'posted_by')
        _ = loader.create_graph()
        loader.add_features(["score", "rate"])
        self.assertEqual(loader.graph.edges[loader.a2b_type].data["score"].shape, (5,))
        self.assertEqual(loader.graph.edges[loader.b2a_type].data["score"].shape, (5,))
        self.assertEqual(loader.graph.edges[loader.a2b_type].data["rate"].shape, (5,))
        self.assertEqual(loader.graph.edges[loader.b2a_type].data["rate"].shape, (5,))


class TestSpliting(unittest.TestCase):
    def test_train_test_split(self):
        loader = RedditDatasetLoader('tests/data/split_data.json')
        _ = loader.create_graph()
        train_graph, test_graph = loader.train_test_split()
        self.assertEqual(train_graph.nodes('author').shape, (8,))
        self.assertEqual(train_graph.nodes('subreddit').shape, (2,))
        self.assertEqual(test_graph.nodes('author').shape, (2,))
        self.assertEqual(test_graph.nodes('subreddit').shape, (2,))

    def test_unknown_node_type(self):
        loader = JSONDatasetLoader('tests/data/split_data.json', 'author', 'subreddit', 'posted_to', 'posted_by')
        _ = loader.create_graph()
        with self.assertRaises(ValueError):
            loader.train_test_split(nodes_to_split="unknown")

    def test_split_a_type(self):
        loader = JSONDatasetLoader('tests/data/split_data.json', 'author', 'subreddit', 'posted_to', 'posted_by')
        _ = loader.create_graph()
        train_graph, test_graph = loader.train_test_split(nodes_to_split="subreddit", test_size=.5)
        self.assertEqual(train_graph.nodes('author').shape, (10,))
        self.assertEqual(train_graph.nodes('subreddit').shape, (1,))
        self.assertEqual(test_graph.nodes('author').shape, (10,))
        self.assertEqual(test_graph.nodes('subreddit').shape, (1,))

    def test_test_size_too_big(self):
        loader = RedditDatasetLoader('tests/data/split_data.json')
        _ = loader.create_graph()
        with self.assertRaises(ValueError):
            loader.train_test_split(test_size=15)
