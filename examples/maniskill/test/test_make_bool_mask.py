def make_bool_mask(*dims: int) -> tuple[bool, ...]:
    """Make a boolean mask for the given dimensions.

    Example:
        make_bool_mask(2, -2, 2) == (True, True, False, False, True, True)
        make_bool_mask(2, 0, 2) == (True, True, True, True)

    Args:
        dims: The dimensions to make the mask for.

    Returns:
        A tuple of booleans.
    """
    result = []
    for dim in dims:
        if dim > 0:
            result.extend([True] * (dim))
        else:
            result.extend([False] * (-dim))
    return tuple(result)


# Example 1: Basic usage
mask = make_bool_mask(3, -2, 1)
print(mask)  # Output: (True, True, True, False, False, True)

# Example 2: Using with zero
mask = make_bool_mask(2, 0, 2)
print(mask)  # Output: (True, True, True, True)

# Example 3: All positive numbers
mask = make_bool_mask(1, 2, 3)
print(mask)  # Output: (True, True, True, True, True, True)

# Example 4: All negative numbers
mask = make_bool_mask(-1, -2, -3)
print(mask)  # Output: (False, False, False, False, False, False
import ipdb

ipdb.set_trace()
