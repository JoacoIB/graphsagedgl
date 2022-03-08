import unittest
from graphsagedgl.loaders import RedditDatasetLoader


class TestRedditDatasetLoader(unittest.TestCase):
    def test_load_dataset(self):
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
