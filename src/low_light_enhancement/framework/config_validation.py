from __future__ import annotations

import math
from typing import Any


def require_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a bool. Got {value!r}.")

    return value


def require_finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite float, not bool.")

    result = float(value)

    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite. Got {result}.")

    return result


def require_positive_float(value: Any, name: str) -> float:
    result = require_finite_float(value, name)

    if result <= 0.0:
        raise ValueError(f"{name} must be > 0. Got {result}.")

    return result


def require_non_negative_float(value: Any, name: str) -> float:
    result = require_finite_float(value, name)

    if result < 0.0:
        raise ValueError(f"{name} must be >= 0. Got {result}.")

    return result


def require_probability(value: Any, name: str) -> float:
    result = require_finite_float(value, name)

    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]. Got {result}.")

    return result


def require_fraction(value: Any, name: str) -> float:
    result = require_finite_float(value, name)

    if not 0.0 < result < 1.0:
        raise ValueError(f"{name} must be in (0, 1). Got {result}.")

    return result


def require_positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive int, not bool.")

    if not isinstance(value, int):
        raise ValueError(f"{name} must be a positive int. Got {value!r}.")

    if value < 1:
        raise ValueError(f"{name} must be >= 1. Got {value}.")

    return value


def require_non_negative_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a non-negative int, not bool.")

    if not isinstance(value, int):
        raise ValueError(f"{name} must be a non-negative int. Got {value!r}.")

    if value < 0:
        raise ValueError(f"{name} must be >= 0. Got {value}.")

    return value


def require_non_negative_range(
    min_value: Any,
    max_value: Any,
    name: str
) -> tuple[float, float]:
    return require_range(
        min_value,
        max_value,
        name,
        lower_bound=0.0,
        lower_bound_inclusive=True
    )


def require_positive_range(
    min_value: Any,
    max_value: Any,
    name: str
) -> tuple[float, float]:
    return require_range(
        min_value,
        max_value,
        name,
        lower_bound=0.0,
        lower_bound_inclusive=False
    )


def require_scale_range(
    min_value: Any,
    max_value: Any,
    name: str
) -> tuple[float, float]:
    return require_range(
        min_value,
        max_value,
        name,
        lower_bound=0.0,
        lower_bound_inclusive=False,
        upper_bound=1.0,
        upper_bound_inclusive=True
    )


def require_range(
    min_value: Any,
    max_value: Any,
    name: str,
    *,  # next options are keyword-only
    lower_bound: float,
    lower_bound_inclusive: bool,
    upper_bound: float | None = None,
    upper_bound_inclusive: bool = True
) -> tuple[float, float]:
    min_result = require_finite_float(min_value, f"{name}_min")
    max_result = require_finite_float(max_value, f"{name}_max")

    if min_result > max_result:
        raise ValueError(
            f"{name} min must be <= max. "
            f"Got min={min_result}, max={max_result}."
        )

    if not _respects_bound(
        min_result,
        lower_bound,
        inclusive=lower_bound_inclusive,
        lower=True
    ) or not _respects_bound(
        max_result,
        lower_bound,
        inclusive=lower_bound_inclusive,
        lower=True
    ):
        operator = ">=" if lower_bound_inclusive else ">"

        raise ValueError(
            f"{name} values must be {operator} {lower_bound}. "
            f"Got min={min_result}, max={max_result}."
        )

    if upper_bound is not None and (
        not _respects_bound(
            min_result,
            upper_bound,
            inclusive=upper_bound_inclusive,
            lower=False
        )
        or not _respects_bound(
            max_result,
            upper_bound,
            inclusive=upper_bound_inclusive,
            lower=False
        )
    ):
        operator = "<=" if upper_bound_inclusive else "<"

        raise ValueError(
            f"{name} values must be {operator} {upper_bound}. "
            f"Got min={min_result}, max={max_result}."
        )

    return min_result, max_result


def require_output_size(value: Any, name: str) -> tuple[int, int]:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an int or a pair of ints, not bool.")

    if isinstance(value, int):
        size = require_positive_int(value, name)
        return size, size

    if isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise ValueError(f"{name} must have exactly two values.")

        width = require_positive_int(value[0], f"{name}[0]")
        height = require_positive_int(value[1], f"{name}[1]")

        return width, height

    raise ValueError(f"{name} must be an int or a pair of ints. Got {value!r}.")


def _respects_bound(
    value: float,
    bound: float,
    *,
    inclusive: bool,
    lower: bool
) -> bool:
    if lower:
        return value >= bound if inclusive else value > bound

    return value <= bound if inclusive else value < bound