import random
import unittest
from hamming_index import HammingIndex


class HammingTests(unittest.TestCase):
    def test_radius_search_matches_brute_force_with_collisions_and_boundaries(self):
        rng = random.Random(20261002)
        values = [rng.getrandbits(64) for _ in range(100)] + [0, 0, (1<<64)-1, 1, 3, 15]
        tree = HammingIndex()
        for i, value in enumerate(values): tree.add(value, i)
        for query in [0, 7, (1<<64)-1, *values[:5]]:
            for radius in [0, 1, 4, 32, 64]:
                expected = sorted(((query^v).bit_count(), i) for i,v in enumerate(values) if (query^v).bit_count() <= radius)
                self.assertEqual(sorted(tree.within(query, radius)), expected)

    def test_invalid_values_and_radius_rejected(self):
        tree = HammingIndex()
        self.assertEqual(tree.within(0, 4), [])
        for value in (-1, 1<<64, True, 1.5):
            with self.assertRaises(ValueError): tree.add(value, "id")
            with self.assertRaises(ValueError): tree.within(value, 4)
        for radius in (-1, 65, True):
            with self.assertRaises(ValueError): tree.within(0, radius)


if __name__ == "__main__": unittest.main()
