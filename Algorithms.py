from collections import deque

import numpy as np

from DragonBallEnv import DragonBallEnv
from typing import List, Tuple, Any
import heapdict


class Node:
    def __init__(self, state, step=None, cost=0, terminated=False, parent_node=None, h=0, f=0) -> None:
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

        self.h = h
        self.f = f

    def __repr__(self) -> str:
        """
            overloads the print to print the state of the Node when calling print(node)
        """
        return f"{self.state}"

    def __eq__(self, other):
        # i think we need this in order for the focal and open to understand they have the same node
        return self.state == other.state

    def __hash__(self):
        # Combine the hash values of relevant attributes
        return hash(self.state)


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

    def update_node_state_if_db(self, node: Node, curr_cell: int) -> None:
        if curr_cell == self.env.d1[0]:
            node.state = (node.state[0], True, node.state[2])
        if curr_cell == self.env.d2[0]:
            node.state = (node.state[0], node.state[1], True)

    def is_not_final_g(self, state: Tuple[int, bool, bool]) -> bool:
        row, col = self.env.to_row_col(state)
        return self.env.desc[row, col] == b"G" and not self.env.is_final_state(state)

    def calculate_heuristic(self, state: Tuple[int, int, int]):
        # getting the coordinates for each needed point.
        srow, scol = self.env.to_row_col(state)
        d1row, d1col = self.env.to_row_col(self.env.d1)
        d2row, d2col = self.env.to_row_col(self.env.d2)
        ret = np.inf

        # d1 collected but d2 not
        if state[1] is True and state[2] is False:
            ret = min(abs(srow - d2row) + abs(scol - d2col), ret)
        # the opposite
        elif state[1] is False and state[2] is True:
            ret = min(abs(srow - d1row) + abs(scol - d1col), ret)
        # neither collected
        elif state[1] is False and state[2] is False:
            ret = min(abs(srow - d1row) + abs(scol - d1col), abs(srow - d2row) + abs(scol - d2col))

        else:
            # goal is given as a list
            for state in self.env.goals:
                grow, gcol = self.env.to_row_col(state)
                ret = min(ret, abs(srow - grow) + abs(scol - gcol))

        return ret


class BFSAgent(Agent):
    def __init__(self) -> None:
        super().__init__()
        self.open: List[Node] = []  # open contains nodes
        self.close: List[Tuple[int, int, int]] = []  # close contains states

    def initialize(self, env) -> None:
        self.env = env
        self.env.reset()
        self.open = []
        self.close = []

    def search(self, env: DragonBallEnv) -> Tuple[List[int], float, int]:
        self.initialize(env)
        curr_state = self.env.get_initial_state()
        curr_node = Node(curr_state)
        self.update_node_state_if_db(curr_node, curr_state[0])
        self.open.append(curr_node)
        expanded = 0
        while len(self.open) != 0:
            curr_node = self.open.pop(0)
            self.close.append(curr_node.state)
            expanded += 1
            for action, (succ_state, cost, terminated) in env.succ(curr_node.state).items():
                if succ_state == None:
                    # not expanding child of hole
                    continue
                succ_node = Node(succ_state, action, cost, terminated, curr_node)
                self.update_node_state_if_db(succ_node, succ_state[0])

                if self.is_not_final_g(succ_node.parentNode.state) is True:
                    # not expanding child of G that is not a solution
                    continue

                if (succ_node.state not in self.close) and (succ_node.state not in [item.state for item in self.open]):
                    if self.env.is_final_state(succ_node.state):
                        (path, total_cost) = self.solution(succ_node)
                        return path, total_cost, expanded
                    self.open.append(succ_node)

        return None


class WeightedAStarAgent(Agent):
    def __init__(self) -> None:
        super().__init__()
        self.open = heapdict.heapdict()  # open contains nodes hd[Node1] = priority1
        self.close = set()  # close contains states
        self.expanded = 0

    def initialize(self, env) -> None:
        self.env = env
        self.env.reset()
        self.expanded = 0
        self.open = heapdict.heapdict()  # open contains nodes hd[Node1] = priority1
        self.close = set()  # close contains states
        # might need to add more things

    def search(self, env: DragonBallEnv, h_weight) -> Tuple[List[int], float, int]:
        self.initialize(env)
        initial_state = self.env.get_initial_state()

        h = self.calculate_heuristic(initial_state)
        f = (h_weight * h) + (1 - h_weight) * 0

        initial_node = Node(initial_state, h=h, f=f)

        self.update_node_state_if_db(initial_node, initial_state[0])
        self.open[initial_node] = (initial_node.f, initial_node.state)

        while len(self.open) > 0:
            current_node = self.open.popitem()[0]  # popitem():Remove and return the (key, priority) pair
            self.close.add(current_node.state)

            # in case we found a final state
            if self.env.is_final_state(current_node.state):
                (path, total_cost) = self.solution(current_node)
                return path, total_cost, self.expanded  # len(self.close)?

            self.expanded += 1
            for action, (succ_state, cost, terminated) in env.succ(current_node.state).items():
                if succ_state is None:
                    continue
                succ_node = Node(succ_state, action, cost, terminated, current_node)
                self.update_node_state_if_db(succ_node, succ_state[0])
                succ_node.h = self.calculate_heuristic(succ_node.state)
                succ_node.f = (h_weight * succ_node.h) + (1 - h_weight) * succ_node.total_cost

                if self.is_not_final_g(succ_node.parentNode.state) is True:
                    # not expanding child of G that is not a solution
                    continue

                if succ_node.state in [item for item in self.close]:
                    # if node is in close, no need to check the g again. heuristic is consistent
                    continue

                # now we need to see if the succ_node.state is in open or not
                if succ_node.state not in [item.state for item in self.open.keys()]:
                    self.open[succ_node] = (succ_node.f, succ_node.state)
                # if it is in open, check if f val is lower
                else:
                    for item in self.open.keys():
                        if item.state == succ_node.state and item.f > succ_node.f:
                            self.open.pop(item)
                            self.open[succ_node] = (succ_node.f, succ_node.state)
                            break

        return None


class AStarEpsilonAgent(Agent):
    def __init__(self) -> None:
        super().__init__()
        self.open = heapdict.heapdict()  # open contains nodes hd[Node1] = priority1
        self.close = set()  # close contains states
        self.expanded = 0

    def compute_focal(self, epsilon) -> heapdict:
        min_f_val = self.open.peekitem()[0].f
        focal = heapdict.heapdict()
        for node in self.open.keys():
            if node.f <= (1 + epsilon) * min_f_val:
                # https://piazza.com/class/lrurdsbmuiww0/post/111
                focal[node] = (node.total_cost, node.state)
        return focal

    def initialize(self, env) -> None:
        self.env = env
        self.env.reset()
        self.open = heapdict.heapdict()  # open contains nodes hd[Node1] = priority1
        self.close = set()  # close contains states

    def search(self, env: DragonBallEnv, epsilon: int) -> Tuple[List[int], float, int]:
        # in the notebook it is written they will not test the expanded here.
        self.initialize(env)
        initial_state = self.env.get_initial_state()
        h = self.calculate_heuristic(initial_state)
        f = h + 0
        initial_node = Node(initial_state, h=h, f=f)
        self.update_node_state_if_db(initial_node, initial_state[0])
        self.open[initial_node] = (initial_node.f, initial_node.state)

        while len(self.open) > 0:
            current_node = (self.compute_focal(epsilon)).popitem()[0]
            self.open.pop(current_node)
            self.close.add(current_node.state)

            # in case we found a final state
            if self.env.is_final_state(current_node.state):
                (path, total_cost) = self.solution(current_node)
                return path, total_cost, self.expanded  # len(self.close)?

            self.expanded += 1
            for action, (succ_state, cost, terminated) in env.succ(current_node.state).items():
                if succ_state is None:
                    continue
                succ_node = Node(succ_state, action, cost, terminated, current_node)
                self.update_node_state_if_db(succ_node, succ_state[0])
                succ_node.h = self.calculate_heuristic(succ_node.state)
                succ_node.f = succ_node.h + succ_node.total_cost

                if self.is_not_final_g(succ_node.parentNode.state) is True:
                    # not expanding child of G that is not a solution
                    continue

                # if node is in close, no need to check the g again. heuristic is consistent
                if succ_node.state in [item for item in self.close]:
                    continue

                # now we need to see if the succ_node.state is in open or not
                if succ_node.state not in [item.state for item in self.open.keys()]:
                    self.open[succ_node] = (succ_node.f, succ_node.state)
                # if it is in open, check if f val is lower
                else:
                    for item in self.open.keys():
                        if item.state == succ_node.state and item.f > succ_node.f:
                            self.open.pop(item)
                            self.open[succ_node] = (succ_node.f, succ_node.state)
                            break

        return None
