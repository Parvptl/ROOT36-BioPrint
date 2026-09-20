import sys
from pathlib import Path

import numpy as np

# The adapter puts backend/ on sys.path itself, but this module has to import
# it first. Resolved from __file__ rather than hardcoded, so the tests run from
# any checkout and inside the project venv.
_BACKEND = Path(__file__).resolve().parents[2]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from evaluation.ml.aalto_adapter import extract_features, parse_aalto_file

def test_adapter_basic_extraction(tmp_path):
    tsv_content = """PARTICIPANT_ID	TEST_SECTION_ID	SENTENCE	USER_INPUT	KEYSTROKE_ID	PRESS_TIME	RELEASE_TIME	LETTER	KEYCODE
100001	1090979	Was wondering	Was wondering	51891207	1000	1100	SHIFT	16
100001	1090979	Was wondering	Was wondering	51891208	1050	1150	W	87
100001	1090979	Was wondering	Was wondering	51891209	1200	1300	a	65
100001	1090979	Was wondering	Was wondering	51891210	1350	1450	s	83
100001	1090979	Was wondering	Was wondering	51891211	1500	1600	 	32
100001	1090979	Was wondering	Was wondering	51891212	1650	1750	w	87
100001	1090979	Was wondering	Was wondering	51891213	1800	1900	BKSP	8
100001	1090979	Was wondering	Was wondering	51891214	1950	2050	w	87
"""
    f = tmp_path / "100001_keystrokes.txt"
    f.write_text(tsv_content, encoding='utf-8')
    
    views = list(parse_aalto_file(str(f)))
    assert len(views) == 1
    view = views[0]
    
    # 8 presses
    assert len(view.presses) == 8
    
    # Check shift is parsed
    assert view.presses[0].code == "Shift"
    assert view.presses[0].key_class == "shift"
    
    # Check space is char
    assert view.presses[4].code == "Space"
    assert view.presses[4].key_class == "char"
    
    # Check backspace is backspace
    assert view.presses[6].code == "Backspace"
    assert view.presses[6].key_class == "backspace"
    
    features = extract_features(view)
    
    # Flight time between W (up at 1150) and a (down at 1200) = 50ms
    # a (up 1300) to s (down 1350) = 50ms
    # Thus kbd_flight_median should be 50ms
    assert features["kbd_flight_median"] == 50.0
    
    # Dwells are all 100ms
    assert features["kbd_dwell_median"] == 100.0
    
    # kbd_shift_right_ratio must be NaN
    assert np.isnan(features["kbd_shift_right_ratio"])
    
def test_adapter_rollover(tmp_path):
    tsv_content = """PARTICIPANT_ID	TEST_SECTION_ID	SENTENCE	USER_INPUT	KEYSTROKE_ID	PRESS_TIME	RELEASE_TIME	LETTER	KEYCODE
100001	1090979	abc	abc	1	1000	1200	a	65
100001	1090979	abc	abc	2	1100	1300	b	66
"""
    f = tmp_path / "100001_keystrokes.txt"
    f.write_text(tsv_content, encoding='utf-8')
    
    view = list(parse_aalto_file(str(f)))[0]
    features = extract_features(view)
    
    # Flight between a (up 1200) and b (down 1100) = -100ms (rollover)
    assert features["kbd_flight_median"] == -100.0
    assert features["kbd_flight_negative_frac"] == 1.0

def test_missing_or_invalid_file(tmp_path):
    f = tmp_path / "empty_keystrokes.txt"
    f.write_text("NOT_A_VALID_TSV", encoding='utf-8')
    views = list(parse_aalto_file(str(f)))
    assert len(views) == 0
