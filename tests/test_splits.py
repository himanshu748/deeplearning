import unittest

from src.splits import split_pairs, validate_split_ratios


class SplitPairsTest(unittest.TestCase):
    def test_split_pairs_is_deterministic_and_complete(self):
        pairs = list(range(20))

        first = split_pairs(pairs, train_ratio=0.7, val_ratio=0.15, seed=7)
        second = split_pairs(pairs, train_ratio=0.7, val_ratio=0.15, seed=7)

        self.assertEqual(first, second)
        self.assertEqual(sum(len(part) for part in first), len(pairs))
        self.assertEqual(set(first[0] + first[1] + first[2]), set(pairs))
        self.assertTrue(all(first))

    def test_split_pairs_rejects_tiny_datasets(self):
        with self.assertRaisesRegex(ValueError, "At least 3"):
            split_pairs([1, 2])

    def test_validate_split_ratios_rejects_invalid_ratios(self):
        invalid = [(0, 0.15), (0.7, 0), (0.8, 0.2), (1, 0.1)]

        for train_ratio, val_ratio in invalid:
            with self.subTest(train_ratio=train_ratio, val_ratio=val_ratio):
                with self.assertRaises(ValueError):
                    validate_split_ratios(train_ratio, val_ratio)


if __name__ == "__main__":
    unittest.main()
