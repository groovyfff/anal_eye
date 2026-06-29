from src.logic.composite_score import CompositeScore

def test_rsi_score_is_low_for_neutral_rsi() -> None:
    scorer = CompositeScore({'weights': {'rsi_score': 1.0}})
    score = scorer.calculate(features={'rsi': 50.0})
    assert score == 0.0

def test_rsi_score_is_high_for_rsi_extremes() -> None:
    scorer = CompositeScore({'weights': {'rsi_score': 1.0}})
    low_score = scorer.calculate(features={'rsi': 0.0})
    high_score = scorer.calculate(features={'rsi': 100.0})
    assert low_score == 1.0
    assert high_score == 1.0
