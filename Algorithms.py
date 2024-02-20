from collections import deque

import numpy as np

from DragonBallEnv import DragonBallEnv
from typing import List, Tuple, Any
import heapdict


class Node:
    def __init__(self, state, step=None, cost=0, terminated=False, parent_node=None) -> None:
        if parent_node is not None:
            self.state = (state[0], parent_node.state[1], parent_node.state[2])
        else:
            self.state = state

        self.step = step  # this is the action taken to reach the current node
        self.cost = cost
        self.terminated = terminated
        self.parentNode = parent_node
        self.total_cost = cost
        if parent_node is not None:
            self.total_cost += parent_node.total_cost

    def __repr__(self) -> str:
        """
            overloads the print to print the state of the Node when calling print(node)
        """
        return f"{self.state}"


class Agent:
    def __init__(self):
        self.env = None

    @staticmethod
    def solution(final_node: Node) -> Tuple[List[int], int]:
        actions = []
        total_cost = final_node.total_cost
        while final_node.parentNode is not None:
            actions.append(final_node.step)
            final_node = final_node.parentNode
        actions.reverse()
        return actions, total_cost


class BFSAgent(Agent):
    def __init__(self) -> None:
        super().__init__()
        self.open: List[Node] = []  # open contains nodes
        self.close: List[Tuple[int, int, int]] = []  # close contains states

    def update_node_state_if_db(self, node: Node, curr_cell: int) -> None:
        if curr_cell == self.env.d1[0]:
            node.state = (node.state[0], True, node.state[2])
        if curr_cell == self.env.d2[0]:
            node.state = (node.state[0], node.state[1], True)

    def initialize(self, env) -> None:
        self.env = env
        self.env.reset()
        self.open = []
        self.close = []

    def search(self, env: DragonBallEnv) -> Tuple[List[int], float, int]:
        self.initialize(env)
        # not sure if it is legal for the bool to be at the start, but just in case
        curr_state = self.env.get_initial_state()
        curr_node = Node(curr_state)
        self.update_node_state_if_db(curr_node, curr_state[0])
        self.open.append(curr_node)

        while len(self.open) != 0:
            curr_node = self.open.pop(0)
            # print("curr node:", curr_node)
            self.close.append(curr_node.state)
            for action, (succ_state, cost, terminated) in env.succ(curr_node.state).items():
                #not adding to open in case we are stuck in a hole
                if succ_state == None:
                    continue
                succ_node = Node(succ_state, action, cost, terminated, curr_node)
                self.update_node_state_if_db(succ_node, succ_state[0])

                if (succ_node.state not in self.close) and (succ_node.state not in [item.state for item in self.open]):
                    if self.env.is_final_state(succ_node.state):
                        # print("************FINAL STATE*************" + str(succ_node.state))
                        (path, total_cost) = self.solution(succ_node)
                        return path, total_cost, len(self.close)

                    #if we are in the same state, continue
                    #if we are in cell number that is the same cell number as final state, continue
                    #maybe we should change succ_state to succ_node.state
                    if succ_state == curr_node.state or (succ_state[0] in [item[0] for item in self.env.get_goal_states()]):
                        continue

                    self.open.append(succ_node)

        return None


class WeightedAStarAgent():
    def __init__(self) -> None:
        raise NotImplementedError

    ''' implementing a method that calculates the heuristic value for each state, by searching for the min manhattan distance between
    need to think abt what we can do if we got the ball but h is to the ball we just got'''
    def calculate_hueristic(self):


    def search(self, env: DragonBallEnv, h_weight) -> Tuple[List[int], float, int]:
        raise NotImplementedError


class AStarEpsilonAgent():
    def __init__(self) -> None:
        raise NotImplementedError

    def ssearch(self, env: DragonBallEnv, epsilon: int) -> Tuple[List[int], float, int]:
        raise NotImplementedError
