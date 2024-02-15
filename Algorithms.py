from collections import deque

import numpy as np

from DragonBallEnv import DragonBallEnv
from typing import List, Tuple, Any
import heapdict


class Node:
    def __init__(self, state, step, cost=0, terminated=False, parentNode=None) -> None:
        self.state = state
        self.step = step  # this is the action taken to reach the current node
        self.cost = cost
        self.terminated = terminated
        self.parentNode = parentNode
        self.expanded = False
        self.total_cost = cost
        if parentNode is not None:
            self.total_cost += parentNode.total_cost

    # TODO: check if we need a printing method.


class Agent:
    def __init__(self):
        self.env = None

    @staticmethod
    def solution(node: Node) -> Tuple[List[int], int]:
        actions = []
        total_cost = node.total_cost
        while node.parentNode is not None:
            actions.append(node.step)
            node = node.parentNode
        actions.reverse()
        return actions, total_cost


class BFSAgent(Agent):
    def __init__(self) -> None:
        super().__init__()

    def animation(self, epochs: int, state: int, action: List[int], total_cost: int) -> None:
        clear_output(wait=True)
        print(self.env.render())
        print(f"Timestep: {epochs}")
        print(f"State: {state}")
        print(f"Action: {action}")
        print(f"Total Cost: {total_cost}")
        time.sleep(1)

    def search(self, env: DragonBallEnv) -> Tuple[List[int], float, int]:
        self.env = env
        self.env.reset()
        # open contains nodes
        open: deque = deque()
        # close contains states
        close = set()

        curr_state = env.get_initial_state()
        # TODO: check if cost for initial state is 0 or 1
        current_node = Node(curr_state, None, 1)
        open.append(current_node)

        # initial state will not be the final goal since we are promised to have 2 balls to collect.

        while len(open) != 0:
            curr_node = open.pop()
            close.add(curr_node.state)
            # def __init__(self, state, step, cost=0, terminated=False, parentNode=None) -> None:
            for action, (succ_state, cost, terminated) in env.succ(curr_state).items():
                succ_node = Node(succ_state, action, cost, terminated, curr_node)
                if succ_state not in close and succ_node not in open:
                    self.env.step(action)


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
