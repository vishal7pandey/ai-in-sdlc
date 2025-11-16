"""Tests for the EntityExtractor helper."""

from src.agents.extraction import EntityExtractor


def test_entity_extractor_finds_actor_action_and_condition() -> None:
    extractor = EntityExtractor()
    text = "Users should be able to log in with email when accessing secure pages"

    actors = extractor.extract_actors(text)
    action = extractor.extract_action(text)
    condition = extractor.extract_condition(text)

    assert "users" in actors
    assert "log in" in action.lower()
    assert condition is not None and "when accessing" in condition.lower()
