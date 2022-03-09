import unittest
from graphsagedgl.loaders import RedditDatasetLoader, JSONDatasetLoader


class TestRedditDatasetLoader(unittest.TestCase):
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
