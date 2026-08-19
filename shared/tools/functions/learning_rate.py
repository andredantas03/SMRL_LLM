import math

def learning_rate_schedule(t, amax, amin, Tw, Tc):
    if t < Tw:
        at = t * amax / Tw
    elif t >= Tw and t <= Tc:
        at = amin + 0.5 * (1 + math.cos((t - Tw) * math.pi / (Tc - Tw))) * (amax - amin)
    else:
        at = amin
    return at