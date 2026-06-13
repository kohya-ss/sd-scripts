import pytest
import math
from library.lora_util import parse_blocks


def test_single_value():
    # Test single numeric value
    result = parse_blocks("1.0")
    assert len(result) == 19
    assert all(val == 1.0 for val in result), "set all values to 1.0 when default value is 1.0"

    # Test zero
    result = parse_blocks("0")
    assert len(result) == 19
    assert all(val == 0.0 for val in result), "set all values to 0.0 when default value is 0"

    # Test negative value
    result = parse_blocks("-0.5")
    assert len(result) == 19
    assert all(val == -0.5 for val in result), "set all values to -0.5 when default value is -0.5"


def test_explicit_list():
    # Test exact length list
    result = parse_blocks("[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1]")
    assert len(result) == 19
    assert result[0] == 0.1
    assert result[9] == 1.0
    assert result[18] == 0.1

    # Test shorter list that repeats
    result = parse_blocks("[0.0,0.5,1.0]")
    assert len(result) == 19
    assert result[0] == 0.0
    assert result[1] == 0.5
    assert result[2] == 1.0
    assert result[3] == 0.0  # Pattern repeats
    assert result[4] == 0.5

    # Test longer list that gets truncated
    result = parse_blocks("[" + ",".join(["0.5"] * 25) + "]")
    assert len(result) == 19
    assert all(val == 0.5 for val in result)


def test_default_with_overrides():
    # Test default value with single index override
    result = parse_blocks("1.0,0:0.5")
    assert len(result) == 19
    assert result[0] == 0.5
    assert all(val == 1.0 for val in result[1:])

    # Test default with multiple index overrides
    result = parse_blocks("0.5,1:0.7,5:0.9,10:0.3")
    assert len(result) == 19
    assert result[0] == 0.5  # Default value
    assert result[1] == 0.7  # Override
    assert result[5] == 0.9  # Override
    assert result[10] == 0.3  # Override
    assert result[18] == 0.5  # Default value

    # Test without default value (should use 0.0)
    result = parse_blocks("3:0.8")
    assert len(result) == 19
    assert result[3] == 0.8
    assert all(val == 0.0 for i, val in enumerate(result) if i != 3)


def test_range_overrides():
    # Test simple range
    result = parse_blocks("1-5:0.7")
    assert len(result) == 19
    assert all(result[i] == 0.7 for i in range(1, 6))
    assert all(val == 0.0 for i, val in enumerate(result) if i < 1 or i > 5)

    # Test multiple ranges
    result = parse_blocks("0.1,1-3:0.5,7-9:0.8")
    assert len(result) == 19
    assert all(result[i] == 0.5 for i in range(1, 4))
    assert all(result[i] == 0.8 for i in range(7, 10))
    assert result[0] == 0.1  # Default
    assert result[6] == 0.1  # Default
    assert result[18] == 0.1  # Default


def test_cos_function():
    # Test cos over range
    result = parse_blocks("1-5:cos")
    assert len(result) == 19
    # Calculate expected values for cosine function
    expected_cos = [(1 + math.cos(i / (5 - 1) * math.pi)) / 2 for i in range(5)]
    for i in range(1, 6):
        assert result[i] == pytest.approx(expected_cos[i - 1])

    # Test parameterized cos
    result = parse_blocks("3-7:cos(0.2,0.8)")
    assert len(result) == 19
    # Cos goes from 1 to 0 over π, scaled to range 0.2 to 0.8
    for i in range(5):
        normalized = (1 + math.cos(i / (5 - 1) * math.pi)) / 2
        expected = 0.2 + normalized * (0.8 - 0.2)
        assert result[i + 3] == pytest.approx(expected)


def test_sin_function():
    # Test sin over range
    result = parse_blocks("2-6:sin")
    assert len(result) == 19
    # Calculate expected values for sine function
    expected_sin = [math.sin(i / (6 - 2) * (math.pi / 2)) for i in range(5)]
    for i in range(2, 7):
        assert result[i] == pytest.approx(expected_sin[i - 2])

    # Test parameterized sin
    result = parse_blocks("4-8:sin(0.3,0.9)")
    assert len(result) == 19
    # Sin goes from 0 to 1 over π/2, scaled to range 0.3 to 0.9
    for i in range(5):
        normalized = math.sin(i / (5 - 1) * (math.pi / 2))
        expected = 0.3 + normalized * (0.9 - 0.3)
        assert result[i + 4] == pytest.approx(expected)


def test_linear_function():
    # Test linear over range
    result = parse_blocks("3-7:linear")
    assert len(result) == 19
    # Calculate expected values for linear function (0 to 1)
    expected_linear = [i / (7 - 3) for i in range(5)]
    for i in range(3, 8):
        assert result[i] == pytest.approx(expected_linear[i - 3])

    # Test parameterized linear
    result = parse_blocks("5-9:linear(0.4,0.7)")
    assert len(result) == 19
    # Linear goes from 0.4 to 0.7
    for i in range(5):
        t = i / 4  # normalized position
        expected = 0.4 + t * (0.7 - 0.4)
        assert result[i + 5] == pytest.approx(expected)


def test_reverse_linear_function():
    # Test reverse_linear over range
    result = parse_blocks("2-6:reverse_linear")
    assert len(result) == 19
    # Calculate expected values for reverse linear function (1 to 0)
    expected_reverse = [1 - i / (6 - 2) for i in range(5)]
    for i in range(2, 7):
        assert result[i] == pytest.approx(expected_reverse[i - 2])

    # Test parameterized reverse_linear
    result = parse_blocks("10-15:reverse_linear(0.8,0.2)")
    assert len(result) == 19
    # Reverse linear goes from 0.2 to 0.8 (reversed)
    for i in range(6):
        t = i / 5  # normalized position
        expected = 0.2 + t * (0.8 - 0.2)
        assert result[i + 10] == pytest.approx(expected)


def test_custom_length():
    # Test with custom length
    result = parse_blocks("1.0", length=5)
    assert len(result) == 5
    assert all(val == 1.0 for val in result)

    # Test list with custom length
    result = parse_blocks("[0.1,0.2,0.3]", length=10)
    assert len(result) == 10
    assert result[0] == 0.1
    assert result[3] == 0.1  # Pattern repeats

    # Test ranges with custom length
    result = parse_blocks("1-3:0.5", length=7)
    assert len(result) == 7
    assert all(result[i] == 0.5 for i in range(1, 4))
    assert result[0] == 0.0
    assert result[6] == 0.0


def test_custom_default():
    # Test with custom default value
    result = parse_blocks("1:0.5", default=0.2)
    assert len(result) == 19
    assert result[1] == 0.5
    assert result[0] == 0.2
    assert result[18] == 0.2

    # Test overriding default value
    result = parse_blocks("0.7,1:0.5", default=0.2)
    assert len(result) == 19
    assert result[1] == 0.5
    assert result[0] == 0.7  # Explicitly set default
    assert result[18] == 0.7


def test_out_of_bounds_indices():
    # Test negative indices (should be ignored)
    result = parse_blocks("-5:0.9")
    assert len(result) == 19
    assert all(val == 0.0 for val in result), "Negative index should be ignored"

    # Test indices beyond length
    result = parse_blocks("25:0.8")
    assert len(result) == 19
    assert all(val == 0.0 for val in result), "Indices above the max length should be ignored"

    # Test range partially out of bounds
    result = parse_blocks("17-22:0.7")
    assert len(result) == 19
    assert result[17] == 0.7
    assert result[18] == 0.7
    # Indices 19-22 would be out of bounds


def test_mixed_patterns():
    # Test combining different formats
    result = parse_blocks("0.3,2:0.8,5-8:cos,10-15:linear(0.1,0.9)")
    assert len(result) == 19
    assert result[0] == 0.3  # Default
    assert result[2] == 0.8  # Single index

    # Check cos values
    cos_range = range(5, 9)
    expected_cos = [(1 + math.cos(i / (8 - 5) * math.pi)) / 2 for i in range(4)]
    for i, idx in enumerate(cos_range):
        assert result[idx] == pytest.approx(expected_cos[i])

    # Check linear values
    linear_range = range(10, 16)
    for i, idx in enumerate(linear_range):
        t = i / 5  # normalized position
        expected = 0.1 + t * (0.9 - 0.1)
        assert result[idx] == pytest.approx(expected)


def test_edge_cases():
    # Test empty string
    result = parse_blocks("")
    assert len(result) == 19
    assert all(val == 0.0 for val in result)

    # Test whitespace
    result = parse_blocks("  ")
    assert len(result) == 19
    assert all(val == 0.0 for val in result)

    # Test empty list
    result = parse_blocks("[]")
    assert len(result) == 19
    assert all(val == 0.0 for val in result)

    # Test single-item range
    result = parse_blocks("5-5:0.7")
    assert len(result) == 19
    assert result[5] == 0.7
    assert result[4] == 0.0
    assert result[6] == 0.0

    # Test function with single-item range
    result = parse_blocks("7-7:cos")
    assert len(result) == 19
    assert result[7] == 1.0  # When range is single point, cos at position 0 is 1

    # Test overlapping ranges
    result = parse_blocks("1-5:0.3,3-7:0.8")
    assert len(result) == 19
    assert result[1] == 0.3
    assert result[2] == 0.3
    assert result[3] == 0.8  # Later definition overwrites
    assert result[4] == 0.8  # Later definition overwrites
    assert result[5] == 0.8  # Later definition overwrites
    assert result[7] == 0.8
    assert result[8] == 0.0


def test_malformed_input():
    # Test malformed list
    result = parse_blocks("[0.1,0.2,")
    assert len(result) == 19
    assert all(val == 0.0 for val in result), "malformed list"

    # Test invalid end range
    result = parse_blocks("5-:0.7")
    assert len(result) == 19
    assert result[5] == 0.7
    assert result[6] == 0.0

    # Test invalid start range, indices should never be negative
    result = parse_blocks("-5:0.7")
    assert len(result) == 19
    assert all(val == 0.0 for val in result), "invalid start range, indices should never be negative"

    # Test invalid function
    result = parse_blocks("1-5:unknown_func")
    assert len(result) == 19
    assert all(val == 0.0 for val in result), "Function name not recognized"

    # Test invalid function parameters
    result = parse_blocks("1-5:cos(invalid,0.8)")
    assert len(result) == 19
    assert all(val == 0.0 for val in result), "Invalid parameters"
