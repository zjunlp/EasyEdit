import unittest

from easyeditor.evaluate.evaluate_utils import exact_match_score


class ExactMatchScoreTests(unittest.TestCase):
    def test_empty_normalized_answers_do_not_match(self):
        for prediction, target in ((".", "the"), ("", "an"), ("the", "?")):
            self.assertFalse(exact_match_score(prediction, target))

    def test_nonempty_answers_keep_existing_behavior(self):
        self.assertTrue(exact_match_score("The Eiffel Tower!", "eiffel tower"))
        self.assertFalse(exact_match_score("London", "Paris"))
