import random
import math


def sample_torus(r_outer, r_inner, x_inner, y_inner):
    """
    Sample uniformly from (x, y) satisfiying:

       x**2 + y**2 <= r_outer**2

       (x-x_inner)**2 + (y-y_inner)**2 > r_inner**2

    Assumes that the inner circle lies inside the outer circle;
    i.e., that hypot(x_inner, y_inner) <= r_outer - r_inner.
    """
    # Sample from a normal annulus with radii r_inner and r_outer.
    rad = math.sqrt(random.uniform(r_inner**2, r_outer**2))
    angle = random.uniform(-math.pi, math.pi)
    x, y = rad*math.cos(angle), rad*math.sin(angle)

    # If we're inside the forbidden hole, reflect.
    if math.hypot(x - x_inner, y - y_inner) < r_inner:
        x, y = x_inner - x, y_inner - y

    return x, y