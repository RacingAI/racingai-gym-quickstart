from turtle import done
import numpy as np

def check_race_conditions(obs):
    if 1 in obs['lap_counts']:
        return True
    elif 1 in obs['collisions']:
        return True
    else:
        return False

    # Also return some race information