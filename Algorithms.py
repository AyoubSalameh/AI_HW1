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
    def __repr__(self) -> str:
        return f"{self.state}"


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
        open = []
        # close contains states
        close = []

        curr_state = self.env.get_initial_state()
        # TODO: check if cost for initial state is 0 or 1
        curr_node = Node(curr_state, None, 0, False, None)
        open.append(curr_node)

        # initial state will not be the final goal since we are promised to have 2 balls to collect.

        while len(open) != 0:
            curr_node = open.pop(0)
            print("curr node:", curr_node)
            close.append(curr_node.state)
            for action, (succ_state, cost, terminated) in env.succ(curr_node.state).items():
                succ_node = Node(succ_state, action, cost, terminated, curr_node)
                succ_node.state = (succ_node.state[0], curr_node.state[1], curr_node.state[2])
                #print("operation " + str(action))
                """checking if we reached a db """
                if succ_state[0] == self.env.d1[0]:
                    succ_node.state = (succ_node.state[0], True, curr_node.state[2])
                if succ_state[0] == self.env.d2[0]:
                    succ_node.state = (succ_node.state[0], curr_node.state[1], True)



                if (succ_node.state not in close) and (succ_node.state not in [ item.state for item in open]):
                    if self.env.is_final_state(succ_node.state):
                        print("************FINAL STATE*************" + str(succ_state))
                        (path, total_cost) = self.solution(succ_node)
                        return (path, total_cost, len(close))
                    #not a hole and not same state
                    if terminated or succ_state == curr_node.state:
                        continue
                    open.append(succ_node)

        return ([1],1,1)


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
