import numpy as np

from DragonBallEnv import DragonBallEnv
from typing import List, Tuple
import heapdict

class Node:
    def __init__(self, state, step, cost=0, terminated=False, parentNode=None) -> None:
        self.state = state
        self.step = step
        self.cost = cost
        self.terminated = terminated
        self.parentNode = parentNode
        self.g = cost
        if parentNode is not None:
            self.g += parentNode.g

    #TODO: check if we need a printing method.
            
            
class BFSAgent():
    def __init__(self) -> None:
        raise NotImplementedError

    def search(self, env: DragonBallEnv) -> Tuple[List[int], float, int]:
        raise NotImplementedError


class WeightedAStarAgent():
    def __init__(self) -> None:
        raise NotImplementedError

    def search(self, env: DragonBallEnv, h_weight) -> Tuple[List[int], float, int]:
        raise NotImplementedError



class AStarEpsilonAgent():
    def __init__(self) -> None:
        raise NotImplementedError
        
    def ssearch(self, env: DragonBallEnv, epsilon: int) -> Tuple[List[int], float, int]:
        raise NotImplementedError