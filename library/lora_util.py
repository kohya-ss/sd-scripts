from collections.abc import MutableSequence
import re
import math
import warnings
from typing import Optional, Union


def parse_blocks(input_str: Optional[Union[str, float]], length=19, default: Optional[float]=0.0) -> MutableSequence[Optional[float]]:
    """
    Parse different formats of block specifications and return a list of values.

    Args:
        input_str (str): The input string after the '=' sign
        length (int): The desired length of the output list (default: 19)

    Returns:
        list: A list of float values with the specified length
    """
    input_str = f"{input_str}" if not isinstance(input_str, str) else input_str.strip() 
    result = [default] * length  # Initialize with default

    if input_str == "":
        return [default] * length

    # Case: Single value (e.g., "1.0" or "-1.0")
    if re.match(r'^-?\d+(\.\d+)?$', input_str):
        value = float(input_str)
        return [value] * length

    # Case: Explicit list (e.g., "[0,0,1,1,0.9,0.8,0.6]")
    if input_str.startswith("[") and input_str.endswith("]"):
        if input_str[1:-1].strip() == "":
            return [default] * length

        # Use regex to properly split on commas while handling negative numbers
        values = [float(x) for x in re.findall(r'-?\d+(?:\.\d+)?', input_str)]
        # If list is shorter than required length, repeat the pattern
        if len(values) < length:
            values = (values * (length // len(values) + 1))[:length]
        # If list is longer than required length, truncate
        return values[:length]

    # Pre-process to handle function parameters with commas
    # Replace function parameters with placeholders
    function_params = {}
    placeholder_counter = 0
    
    def replace_function(match):
        nonlocal placeholder_counter
        func_with_params = match.group(0)
        placeholder = f"FUNC_PLACEHOLDER_{placeholder_counter}"
        function_params[placeholder] = func_with_params
        placeholder_counter += 1
        return placeholder

    # Find function calls with parameters and replace them
    preprocessed_str = re.sub(r'\w+\([^)]+\)', replace_function, input_str)

    # Case: Default value with specific overrides (e.g., "1.0,0:0.5")
    parts = preprocessed_str.split(',')
    default_value = default
    
    # Check if the first part is a default value (no colon)
    if ':' not in parts[0] and re.match(r'^-?\d+(\.\d+)?$', parts[0]):
        default_value = float(parts[0])
        parts = parts[1:]  # Remove the default value from parts
        # Fill the result with the default value
        result = [default_value] * length
    
    # Process the remaining parts as ranges or single indices
    for part in parts:
        if ':' not in part:
            continue  # Skip parts without colon (should only be the default value)
        
        indices_part, value_part = part.split(':')

        # Restore any function placeholders
        for placeholder, original in function_params.items():
            if placeholder in value_part:
                value_part = value_part.replace(placeholder, original)
        
        # Handle range (e.g., "10-18" or "-5-10")
        if '-' in indices_part and not indices_part.startswith('-'):
            # This is a range with a dash (not just a negative number)
            range_parts = indices_part.split('-', 1)  # Split on first dash only

            # Handle potential negative values in the range
            if range_parts[0] == '':
                # Handle case like "-5-10" (from -5 to 10)
                start_idx = int('-' + range_parts[1].split('-')[0])
                end_idx = int(range_parts[1].split('-')[1])
            else:
                # Normal case like "5-10" or "-5-(-3)"
                start_idx = int(range_parts[0])
                end_idx_str = range_parts[1]
                
                # Handle potentially complex end index expressions
                if end_idx_str.startswith('(') and end_idx_str.endswith(')'):
                    # Handle expressions like "(-3)"
                    end_idx = eval(end_idx_str)
                else:
                    print("end_idx_str", end_idx_str)
                    # If end str is blank, set to start idx
                    if end_idx_str == "":
                        warnings.warn("Range end was missing, setting to start of range")
                        end_idx = start_idx
                    else:
                        end_idx = int(end_idx_str)

            # Make sure indices are within bounds
            start_idx = max(0, min(start_idx, length-1))
            end_idx = max(0, min(end_idx, length-1))
            range_length = end_idx - start_idx + 1
            
            # Check if we have a function with parameters
            # Checking function and 2 numbers (float and int) separated by ,
            # cos(0.2, 0.8), cos(0, 1.0), cos(1, 0.1)
            func_match = re.match(r'(\w+)\((\d+|\d+\.\d+),(\d+|\d+\.\d+)\)', value_part)
            if func_match:
                func_name = func_match.group(1)
                start_val = float(func_match.group(2))
                end_val = float(func_match.group(3))
                
                if func_name == 'cos':
                    # Implement parameterized cosine
                    for i in range(range_length):
                        # Calculate position in the range from 0 to π (half a period)
                        position = i / (range_length - 1) * math.pi if range_length > 1 else 0
                        # Cosine from 1 at 0 to 0 at π, scaled to requested range
                        normalized_value = (1 + math.cos(position)) / 2
                        # Scale and shift to the requested start and end values
                        value = start_val + normalized_value * (end_val - start_val)
                        if start_idx + i < length:
                            result[start_idx + i] = value
                
                elif func_name == 'sin':
                    # Implement parameterized sine
                    for i in range(range_length):
                        # Calculate position in the range from 0 to π/2 (quarter period)
                        position = i / (range_length - 1) * (math.pi/2) if range_length > 1 else 0
                        # Sine from 0 at 0 to 1 at π/2, scaled to requested range
                        normalized_value = math.sin(position)
                        # Scale and shift to the requested start and end values
                        value = start_val + normalized_value * (end_val - start_val)
                        if start_idx + i < length:
                            result[start_idx + i] = value
                
                elif func_name == 'linear':
                    # Implement parameterized linear function
                    for i in range(range_length):
                        # Linear interpolation from start_val to end_val
                        t = i / (range_length - 1) if range_length > 1 else 0
                        value = start_val + t * (end_val - start_val)
                        if start_idx + i < length:
                            result[start_idx + i] = value
                
                elif func_name == 'reverse_linear':
                    # Implement parameterized reverse linear function
                    for i in range(range_length):
                        # Linear interpolation from end_val to start_val
                        t = i / (range_length - 1) if range_length > 1 else 0
                        value = end_val + t * (start_val - end_val)
                        if start_idx + i < length:
                            result[start_idx + i] = value
            
            # Handle non-parameterized functions
            elif value_part == 'cos':
                # Default cosine from 1 to 0
                for i in range(range_length):
                    position = i / (range_length - 1) * math.pi if range_length > 1 else 0
                    value = (1 + math.cos(position)) / 2
                    if start_idx + i < length:
                        result[start_idx + i] = value
            
            elif value_part == 'sin':
                # Default sine from 0 to 1
                for i in range(range_length):
                    position = i / (range_length - 1) * (math.pi/2) if range_length > 1 else 0
                    value = math.sin(position)
                    if start_idx + i < length:
                        result[start_idx + i] = value
            
            elif value_part == 'linear':
                # Default linear from 0 to 1
                for i in range(range_length):
                    value = i / (range_length - 1) if range_length > 1 else 0
                    if start_idx + i < length:
                        result[start_idx + i] = value
            
            elif value_part == 'reverse_linear':
                # Default reverse linear from 1 to 0
                for i in range(range_length):
                    value = 1 - (i / (range_length - 1) if range_length > 1 else 0)
                    if start_idx + i < length:
                        result[start_idx + i] = value
            
            else:
                # Regular numeric value
                try:
                    value = float(value_part)
                    for i in range(start_idx, end_idx + 1):
                        if 0 <= i < length:
                            result[i] = value
                except ValueError:
                    warnings.warn(f"Could not parse value '{value_part}'")
        
        # Handle single index (e.g., "1")
        else:
            try:
                index = int(indices_part)
                if 0 <= index < length:
                    # Check if we have a function with parameters (unlikely for single index)
                    if '(' in value_part and ')' in value_part:
                        warnings.warn("Functions with parameters not supported for single indices: {part}")
                        continue
                    
                    # Assuming a single index won't have a function pattern, just a value
                    value = float(value_part)
                    result[index] = value
            except ValueError:
                raise RuntimeError(f"Could not parse index '{indices_part}'")

    return result


