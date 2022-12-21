import unittest
from pathlib import Path
from mxio import dataset


class DatasetTestCases(unittest.TestCase):

    def test_summarize_range(self):
        summarized = dataset.summarize_sequence([1, 2, 3, 4])
        expected = [(1, 4)]
        self.assertEqual(summarized, expected, f'Range not summarized {summarized=} != {expected=}')

    def test_summarize_skips(self):
        summarized = dataset.summarize_sequence([1, 2, 3, 5, 6, 7])
        expected = [(1, 3), (5, 7)]
        self.assertEqual(summarized, expected, f'Range not summarized {summarized=} != {expected=}')

    def test_summarize_singles(self):
        summarized = dataset.summarize_sequence([1, 5, 7])
        expected = [(1, 1), (5, 5), (7, 7)]
        self.assertEqual(summarized, expected, f'Range not summarized {summarized=} != {expected=}')

    def test_tags_from_path(self):
        file_name = Path(__file__)
        expected = ('Python script', 'ASCII text executable')
        tags = dataset.get_tags(file_name)
        self.assertEqual(tags, expected, f'Magic tags do not match {tags=} != {expected=}.')

if __name__ == '__main__':
    unittest.main()