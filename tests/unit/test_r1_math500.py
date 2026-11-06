import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from wsp.evaluate.math500 import extract_final_box


def test_last_box_and_thinking():
    assert extract_final_box('draft \\boxed{7} final \\boxed{\\frac{1}{2}}')=='\\frac{1}{2}'
    assert extract_final_box('<think>\\boxed{99}</think>answer \\boxed{3}')=='3'
    assert extract_final_box('incomplete \\boxed{1') is None
    assert extract_final_box('answer \\boxed 5 $')=='5'
    assert extract_final_box('none') is None
