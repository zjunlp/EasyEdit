"""
python utils/test_helpers.py
"""

import torch as t

from .helpers import (
    add_vector_from_position,
    find_last_subtensor_position,
    find_instruction_end_postion,
)


def test_add_vector_from_position():
    matrix = t.tensor([[1, 2], [3, 4], [5, 6]], dtype=t.float32)
    vector = t.tensor([1, 1], dtype=t.float32)
    position_ids = t.tensor([1, 2, 3])
    result = add_vector_from_position(
        matrix, vector, position_ids, from_pos=2
    )
    expected = t.tensor([[1, 2], [4, 5], [6, 7]], dtype=t.float32)
    assert t.allclose(result, expected)


def test_find_last_subtensor_position():
    tensor = t.tensor([1, 2, 3, 4, 5, 1, 2, 3])
    sub_tensor = t.tensor([1, 2, 3])
    result = find_last_subtensor_position(tensor, sub_tensor)
    assert result == 5

    # Test case where sub_tensor isn't in tensor
    sub_tensor = t.tensor([6, 7])
    result = find_last_subtensor_position(tensor, sub_tensor)
    assert result == -1


def test_find_instruction_end_position():
    tokens = t.tensor([1, 2, 3, 4, 5, 1, 2, 3, 1, 2, 3])
    end_str = t.tensor([1, 2, 3])
    result = find_instruction_end_postion(tokens, end_str)
    assert result == 10

    # Test case where end_str isn't in tokens
    end_str = t.tensor([6, 7])
    result = find_instruction_end_postion(tokens, end_str)
    assert result == -1
