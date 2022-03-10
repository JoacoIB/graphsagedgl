import unittest
from graphsagedgl.sampler import Node2NodeSampler
from graphsagedgl.loaders import RedditDatasetLoader


class TestNode2NodeSampler(unittest.TestCase):
    def test_sampler(self):
        loader = RedditDatasetLoader('tests/data/sampler_data.json')
        graph = loader.create_graph()
        sampler = Node2NodeSampler(
            graph=graph,
            batch_size=2,
            a_type="author",
            b_type="subreddit",
            a2b_type="posted_to",
            b2a_type="posted_by"
        )
        for anchor, positive, negative in sampler:
            self.assertEqual(anchor.shape, (2,))
            self.assertEqual(positive.shape, (2,))
            self.assertEqual(negative.shape, (2,))
            break
