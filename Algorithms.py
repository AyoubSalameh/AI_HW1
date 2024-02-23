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
                # not adding to open in case we are stuck in a hole
                if succ_state == None:
                    continue
                succ_node = Node(succ_state, action, cost, terminated, curr_node)
                self.update_node_state_if_db(succ_node, succ_state[0])

                if (succ_node.state not in self.close) and (succ_node.state not in [item.state for item in self.open]):
                    if self.env.is_final_state(succ_node.state):
                        # print("************FINAL STATE*************" + str(succ_node.state))
                        (path, total_cost) = self.solution(succ_node)
                        return path, total_cost, len(self.close)

                    # if we are in the same state, continue
                    # if we are in cell number that is the same cell number as final state, continue
                    # maybe we should change succ_state to succ_node.state
                    if succ_state == curr_node.state or (
                            succ_state[0] in [item[0] for item in self.env.get_goal_states()]):
                        continue

                    self.open.append(succ_node)

        return None


class WeightedAStarAgent(Agent):
    def __init__(self) -> None:
        super().__init__()
        self.open = heapdict.heapdict()  # open contains nodes hd[Node1] = priority1
        self.close = set()  # close contains states
        self.expanded = 0
        # might need to add more things

    def initialize(self, env) -> None:
        self.env = env
        self.env.reset()
        self.expanded = 0
        self.open = heapdict.heapdict()  # open contains nodes hd[Node1] = priority1
        self.close = set()  # close contains states
        # might need to add more things

    def calculate_heuristic(self, state: Tuple[int, int, int]):
        # getting the coordinates for each needed point.
        srow, scol = self.env.to_row_col(state)
        d1row, d1col = self.env.to_row_col(self.env.d1)
        d2row, d2col = self.env.to_row_col(self.env.d2)
        ret = np.inf

        # d1 collected but d2 not
        if state[1] == True and state[2] == False:
            ret = min(abs(srow - d2row) + abs(scol - d2col), ret)
        # the opposite
        elif state[1] == False and state[2] == True:
            ret = min(abs(srow - d1row) + abs(scol - d1col), ret)
        # neither collected
        elif state[1] == False and state[2] == False:
            ret = min(abs(srow - d1row) + abs(scol - d1col), abs(srow - d2row) + abs(scol - d2col))

        else:
            # goal is given as a list
            for state in self.env.goals:
                grow, gcol = self.env.to_row_col(state)
                ret = min(ret, abs(srow - grow) + abs(scol - gcol))

        return ret

    def search(self, env: DragonBallEnv, h_weight) -> Tuple[List[int], float, int]:
        self.initialize(env)
        initial_state = self.env.get_initial_state()

        h = self.calculate_heuristic(initial_state)
        f = (h_weight * h) + (1 - h_weight) * 0

        initial_node = Node(initial_state, h=h, f=f)
        self.update_node_state_if_db(initial_node, initial_state[0])
        self.open[initial_node] = (initial_node.f, initial_node.state[0])

        while len(self.open) > 0:
            current_node = self.open.popitem()[0]  # popitem():Remove and return the (key, priority) pair
            self.close.add(current_node.state)

            # in case we found a final state
            if self.env.is_final_state(current_node.state):
                (path, total_cost) = self.solution(current_node)
                return path, total_cost, self.expanded  # len(self.close)?

            self.expanded += 1
            for action, (succ_state, cost, terminated) in env.succ(current_node.state).items():
                succ_node = Node(succ_state, action, cost, terminated, current_node)
                self.update_node_state_if_db(succ_node, succ_state[0])
                succ_node.h = self.calculate_heuristic(succ_node.state)

                succ_node.f = (h_weight * succ_node.h) + (1 - h_weight) * succ_node.total_cost

                # if it is goal but didnt collect all dragon balls, or is hole. i guess
                # TODO: check if we need to add these to closed. meaning if we need to exapnd them
                if terminated is True and self.env.is_final_state(succ_node.state) is False:
                    continue
                # maybe this line is useless
                if succ_node.state == current_node.state:
                    continue

                # if node is in close, no need to check the g again. heuristic is consistent
                if succ_node.state in [item for item in self.close]:
                    continue

                # now we need to see if the succ_node.state is in open or not
                if succ_node.state not in [item.state for item in self.open.keys()]:
                    self.open[succ_node] = (succ_node.f, succ_node.state[0])
                # if it is in open, check if f val is lower
                else:
                    for item in self.open.keys():
                        if item.state == succ_node.state and item.f > succ_node.f:
                            self.open.pop(item)
                            self.open[succ_node] = (succ_node.f, succ_node.state[0])
                            break

        return None


class AStarEpsilonAgent(Agent):
    def __init__(self) -> None:
        super().__init__()
        self.open = heapdict.heapdict()  # open contains nodes hd[Node1] = priority1
        self.close = set()  # close contains states
        self.expanded = 0

    def calculate_heuristic(self, state: Tuple[int, int, int]):
        # getting the coordinates for each needed point.
        srow, scol = self.env.to_row_col(state)
        d1row, d1col = self.env.to_row_col(self.env.d1)
        d2row, d2col = self.env.to_row_col(self.env.d2)
        ret = np.inf

        # d1 collected but d2 not
        if state[1] == True and state[2] == False:
            ret = min(abs(srow - d2row) + abs(scol - d2col), ret)
        # the opposite
        elif state[1] == False and state[2] == True:
            ret = min(abs(srow - d1row) + abs(scol - d1col), ret)
        # neither collected
        elif state[1] == False and state[2] == False:
            ret = min(abs(srow - d1row) + abs(scol - d1col), abs(srow - d2row) + abs(scol - d2col))

        else:
            # goal is given as a list
            for state in self.env.goals:
                grow, gcol = self.env.to_row_col(state)
                ret = min(ret, abs(srow - grow) + abs(scol - gcol))

        return ret

    def compute_focal(self, epsilon) -> heapdict:
        min_f_val = self.open.peekitem()[0].f
        focal = heapdict.heapdict()
        for node in self.open.keys():
            if node.f <= (1 + epsilon) * min_f_val:
                # https://piazza.com/class/lrurdsbmuiww0/post/111
                focal[node] = node.total_cost
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
        self.open[initial_node] = (initial_node.f, initial_node.state[0])

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
                succ_node = Node(succ_state, action, cost, terminated, current_node)
                self.update_node_state_if_db(succ_node, succ_state[0])
                succ_node.h = self.calculate_heuristic(succ_node.state)
                succ_node.f = succ_node.h + succ_node.total_cost

                # if it is goal but didnt collect all dragon balls, or is hole. i guess
                # TODO: check if we need to add these to closed. meaning if we need to exapnd them
                if terminated is True and self.env.is_final_state(succ_node.state) is False:
                    continue
                # maybe this line is useless
                if succ_node.state == current_node.state:
                    continue

                # if node is in close, no need to check the g again. heuristic is consistent
                if succ_node.state in [item for item in self.close]:
                    continue

                # now we need to see if the succ_node.state is in open or not
                if succ_node.state not in [item.state for item in self.open.keys()]:
                    self.open[succ_node] = (succ_node.f, succ_node.state[0])
                # if it is in open, check if f val is lower
                else:
                    for item in self.open.keys():
                        if item.state == succ_node.state and item.f > succ_node.f:
                            self.open.pop(item)
                            self.open[succ_node] = (succ_node.f, succ_node.state[0])
                            break

        return None
