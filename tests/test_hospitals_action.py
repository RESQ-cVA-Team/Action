from src.actions.actions.hospitals_action import _build_pagination_buttons
from src.actions.helpers.hospital import extract_hospital_filters


class _StubTracker:
    def __init__(self, latest_message, slots=None):
        self.latest_message = latest_message
        self._slots = slots or {}

    def get_slot(self, key):
        return self._slots.get(key)


def test_extract_hospital_filters_treats_numeric_limit_like_offset_with_fixed_page_size() -> None:
    tracker = _StubTracker(
        latest_message={"metadata": {"limit": 100}, "entities": []},
        slots={},
    )

    filters = extract_hospital_filters(tracker)

    assert filters["limit"] == 50
    assert filters["offset"] == 100


def test_extract_hospital_filters_uses_explicit_offset_when_present() -> None:
    tracker = _StubTracker(
        latest_message={"metadata": {"limit": 100, "offset": 50}, "entities": []},
        slots={},
    )

    filters = extract_hospital_filters(tracker)

    assert filters["limit"] == 50
    assert filters["offset"] == 50


def test_build_pagination_buttons_first_page_has_only_next() -> None:
    buttons = _build_pagination_buttons(offset=0, limit=50, total_count=120, language="en")

    assert len(buttons) == 1
    assert buttons[0]["title"] == "Next"
    assert buttons[0]["payload"] == '/list_hospitals{"offset": 50}'


def test_build_pagination_buttons_middle_page_has_previous_and_next() -> None:
    buttons = _build_pagination_buttons(offset=50, limit=50, total_count=180, language="en")

    assert len(buttons) == 2
    assert buttons[0]["title"] == "Previous"
    assert buttons[0]["payload"] == '/list_hospitals{"offset": 0}'
    assert buttons[1]["title"] == "Next"
    assert buttons[1]["payload"] == '/list_hospitals{"offset": 100}'


def test_build_pagination_buttons_last_page_has_only_previous() -> None:
    buttons = _build_pagination_buttons(offset=100, limit=50, total_count=120, language="en")

    assert len(buttons) == 1
    assert buttons[0]["title"] == "Previous"
    assert buttons[0]["payload"] == '/list_hospitals{"offset": 50}'
