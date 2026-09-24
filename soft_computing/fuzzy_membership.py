def clamp(value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    return max(min_val, min(max_val, value))

def triangular(x: float, a: float, b: float, c: float) -> float:
    if x < a or x > c:
        return 0.0
    if x <= b:
        if a == b:
            return 1.0
        return (x - a) / (b - a)
    if x > b:
        if b == c:
            return 1.0
        return (c - x) / (c - b)
    return 0.0

def trapezoidal(x: float, a: float, b: float, c: float, d: float) -> float:
    if x < a or x > d:
        return 0.0
    if x <= b:
        if a == b:
            return 1.0
        return (x - a) / (b - a)
    if b < x < c:
        return 1.0
    if x >= c:
        if c == d:
            return 1.0
        return (d - x) / (d - c)
    return 0.0

# Input memberships
def low_input(x: float) -> float:
    return trapezoidal(x, 0.0, 0.0, 0.20, 0.40)

def medium_input(x: float) -> float:
    return triangular(x, 0.20, 0.50, 0.80)

def high_input(x: float) -> float:
    return trapezoidal(x, 0.60, 0.80, 1.0, 1.0)
