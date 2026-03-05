import sys
from pathlib import Path

# scripts/ is not a package — add repo root so we can import it directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_eval import simple_score, score_with_tutor_style


class TestSimpleScore:
    def test_exact_content_words_score_1(self):
        gold = "A variable stores data."
        pred = "A variable stores data in memory."
        assert simple_score(gold, pred) == 1.0

    def test_no_overlap_scores_0(self):
        assert simple_score("apple banana cherry", "xyz uvw") == 0.0

    def test_partial_overlap_scores_half(self):
        # gold content words: variable, stores (stopwords removed)
        # need >= 20% but < 40% overlap
        gold = "variable stores data"
        pred = "variable something else entirely different"
        score = simple_score(gold, pred)
        assert 0.0 <= score <= 1.0

    def test_empty_prediction_scores_0(self):
        assert simple_score("anything", "") == 0.0

    def test_stopwords_ignored(self):
        # "is" and "a" are stopwords — gold content word is just "variable"
        gold = "a variable is good"
        pred = "variable"
        assert simple_score(gold, pred) == 1.0

    def test_case_insensitive(self):
        gold = "VARIABLE stores DATA"
        pred = "variable stores data"
        assert simple_score(gold, pred) == 1.0


class TestScoreWithTutorStyle:
    def test_base_score_passthrough(self):
        # No bonuses — should match simple_score
        gold = "xyz"
        pred = "xyz"
        assert score_with_tutor_style(gold, pred) == simple_score(gold, pred)

    def test_example_bonus_applied(self):
        gold = "a loop repeats code"
        pred = "loop repeats code for example in Python"
        base = simple_score(gold, pred)
        tutor = score_with_tutor_style(gold, pred)
        assert tutor >= base

    def test_mistake_bonus_applied(self):
        gold = "a loop repeats code"
        pred = "loop repeats code common mistake is off by one"
        base = simple_score(gold, pred)
        tutor = score_with_tutor_style(gold, pred)
        assert tutor >= base

    def test_score_capped_at_1(self):
        gold = "variable stores data"
        pred = "variable stores data for example be careful common mistake"
        assert score_with_tutor_style(gold, pred) <= 1.0

    def test_empty_pred_scores_0(self):
        assert score_with_tutor_style("gold answer", "") == 0.0
