#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Evolution.py

"""
Author:             Romain Claveau
Version:            0 (unstable)
Python version:     3.11.7
Dependencies:       Tree (local), random, copy
License:            CC BY-NC-SA (https://creativecommons.org/licenses/by-nc-sa/4.0/)

Reporting an issue:  https://github.com/RomainClaveau/KinePycs/issues
"""

"""
==============================================================
Evolution of Mathematical Expression to match a given function
==============================================================

Creating and evolving Trees representing mathematical expressions until
reaching a function being close to a given objective function.

This module is inspired from Miles Cranmer's guidelines (https://github.com/MilesCranmer/PySR)
and its associated paper (https://arxiv.org/pdf/2305.01582.pdf).
"""

import Tree
from random import *
from copy import *
from numpy import *
from scipy import *

class Evolution:

    Niter = 100       # Iterations
    Npop = 1000     # Total population
    Ngroup = 10      # Population in each group (for tournaments)

    Complexity = 3  # Initial complexity

    Pcross = 0.01   # Probability of crossover
    Pfittest = 0.9  # Probability of selecting the fittest individual

    Population = [] # Population
    Best = set()    # Best

    Tracks = {}

    Constants = {}  # Constants

    Best_score = 1e99
    Best_tree = None

    """
    Initializing the Evolution
    """
    def __init__(self, objective: list, dimension: int = 1):

        # Checking the arguments' type
        if not isinstance(objective, (list)): raise TypeError("`objective` should be a list.")
        if not isinstance(dimension, (int)): raise TypeError("`dimension` should be an integer.")

        # Checking arguments' value
        if dimension <= 0: raise ValueError("`dimension` should higher than zero.")
        if len(objective) == 0 or len(objective[0]) != dimension + 1: raise ValueError("`objective` does not have the required shape.")

        # Saving
        self.Objective = objective
        self.Dimension = dimension

        # Initializing the population
        self.initialize()

    """
    Initializing the population at fixed complexity
    """
    def initialize(self):
        
        # Creating the population
        while len(self.Population) != self.Npop:

            # Creating the initial tree
            T = Tree.Tree(dimension=self.Dimension)
            T.add_node(type="var", value="x0")

            for i in range(self.Complexity):
                # Randomly selecting a node
                _id = choice(list(T.Nodes.keys()))

                # Randomly selecting an operation
                _op = choice(list(Tree._Operations.keys()))
                if _op == "id": continue

                # Randomly selecting an appending type
                _at = choice(["var", "cst"])

                # Randomly selecting an appending row
                _ar = -1

                # Applying on the node
                T.apply_on_node(node=_id, operation=_op, appending_type=_at, appending_row=_ar)

                # Saving if we reached the desired complexity
                if len(T.Nodes.keys()) == self.Complexity:
                    T.lambdify(update=True)
                    self.Population.append(deepcopy(T))
                    break

        # Initializing the constants
        # Generating uniformly a float between -10 and 10.
        for _k, _v in enumerate(self.Population):
            if len(_v.sConstants) != 0:
                self.Constants[_k] = {_c: uniform(-10, 10) for _c in _v.sConstants}

    """
    Computing the error - Several options are natively supported.
        1) cumul:   Cumulative error
    """
    def error(self, expr: int, type: str = "cumul") -> float:

        # NOTE: The objective data should be shaped as
        #   [[x0, y0, z0, u0, ...], [x1, y1, z1, u1, ...], ...]
        # Meaning that data.shape = (n, dim) where
        #   n is the number of observations (points)
        #   dim the number of variables
        
        # The last variable is assumed to be expressed through all the others
        #   i.e. w = f(x, y, z, u, ...)

        # Retrieving the expression
        f = self.Population[expr].lambdify(update=True)

        # Retrieving the constants
        if expr in self.Constants:
            csts = list(self.Constants[expr].values())
        else:
            csts = []

        exps = []
        calcs = []

        # Calculating
        try:
            for obs in self.Objective:
                exps.append(obs[-1])
                calcs.append(f(*[*obs[:-1], *csts]))
        except Exception as e:
            return 1e99

        # Computing error
        match type:
            case "cumul": return sum([abs(exps[i] - calcs[i]) for i in range(len(exps))])
            case other: pass

    def optimize(self, id: int) -> None:
        tree = self.Population[id]

        # Retrieving the expression
        f = tree.lambdify(update=True)

        # Retrieving the constants
        if id in self.Constants:
            csts = list(self.Constants[id].values())
        else:
            return None

        def eq(args):
            err = 0.0

            for obs in self.Objective:
                try:
                    err += abs(obs[-1] - f(*[*obs[:-1], *args]))
                except Exception:
                    err = 1e99
                    break

            return err

        sol = optimize.minimize(eq, x0=list(csts), method="Nelder-Mead", options={"adaptive": True})

        if sol.fun < self.Best_score:
            self.Best_score = sol.fun
            self.Best_tree = deepcopy(tree)

        for _n, _k in enumerate(self.Constants[id]):
            self.Constants[id][_k] = sol.x[_n]

        return None

    """
    Running a tournament
    """
    def tournament(self, group: list) -> int:
        scores = []

        for _id in group:
            score = self.error(_id)
            scores.append([_id, score])

        # Sorting by increasing scores
        scores.sort(key=lambda x: x[1])

        for _expr in scores:
            if uniform(0, 1) < self.Pfittest:
                return _expr[0]

    """
    Mutating an expression
    """
    def mutate(self, id: int) -> None:

        tree = deepcopy(self.Population[id])

        # Several choices
        #   1) `mutate`:    mutating a random node
        #   2) `apply`:     applying an operation on a node
        #   3) `delete`:    deleting a branching from a given node
        #   4) `append`:    generating and appending a new tree onto top node
        #   5) `random`:    generating a new tree
        #   6) `nothing`:   doing nothing

        # TODO: exclude `ind` from acceptable operations
        
        what_to_do = choice([
            #"mutate", "apply", "delete", "append", "random", "nothing"
            "mutate", "apply", "delete", "nothing"
        ])
        
        match what_to_do:
            # Mutating a random node
            case "mutate":
                # Selecting a node
                _id = choice(list(tree.Nodes.keys()))

                # Selecting a type
                _t = choice(["op", "var", "cst"])

                # Default appendable type (not used for `var` and `cst`)
                _at = "var"

                # Default appendable row (not used for `var` and `cst`)
                _ar = -1

                # Selecting a value
                if _t == "op":
                    _v = choice(list(Tree._Operations.keys()))
                    _at = choice(["var", "cst"])

                    if _at == "var":
                        _ar = choice(tree.sVars)
                    else:
                        _ar = -1

                elif _t == "var":
                    _v = choice(tree.sVars)
                else:
                    # Retrieving the last constant introduced
                    if len(tree.sConstants) == 0:
                        _cst = 0
                    else:
                        _cst = int(tree.sConstants[-1][1:]) + 1

                    _cst_id = choice(range(-1, len(tree.sConstants)))

                    if _cst_id == -1:
                        # Adding the constant
                        tree.sConstants.append(f"c{_cst}")
                        _v = _cst
                    else:
                        _v = int(tree.sConstants[_cst_id][1:])
                
                # Applying mutation
                try:
                    tree.mutate_node(node=_id, type=_t, value=_v, appending_type=_at, appending_row=_ar)
                except Exception:
                    return None
            
            # Applying a random operation on a node
            case "apply":
                # Selecting a node
                _id = choice(list(tree.Nodes.keys()))

                # Selecting an operation
                _op = choice(list(Tree._Operations.keys()))

                # Selecting an appendable type
                _at = choice(["var", "cst"])

                # Selecting an appendable row
                if _at == "var":
                    _ar = randint(0, self.Dimension - 1)
                else:
                    _ar = -1

                # Applying the operation
                tree.apply_on_node(node=_id, operation=_op, appending_type=_at, appending_row=_ar)

            # Deleting a branching from a random node
            case "delete":
                # Selecting a node
                _id = choice(list(tree.Nodes.keys()))

                # Rejecting the move if the node is the highest
                if _id == [_n for _n, _v in tree.Parents.items() if len(_v) == 0][0]:
                    return None

                # Deleting the branching, without deleting the parent node
                tree.delete_branching_from(node=_id, include=False)
            
            # Generating a random tree (of complexity 3) and appending it on top (similar to `crossover`)
            case "append":
                pass

            # Generating a random tree with a higher complexity
            case "random":
                pass
            
            # Speak for itself, doing nothing at all
            case "nothing":
                return None

        # Accepting the move or not?
        # Creating temporary element
        self.Population.append(deepcopy(tree))
        
        if len(tree.sConstants) != 0:
            self.Constants["tmp"] = {_c: uniform(-10, 10) for _c in tree.sConstants}
        
        # Computing the score difference

        # NOTE: In some cases, Δs may be equals to NaN or Inf, which is rejected.
        # Δs is defined as new_score - old_score.
        # A better (lower) score leads to Δs < 1.
        try:
            Δs = self.error(-1) - self.error(id)
        except Exception as e:
            print(e)

        # Computing the complexity difference
        ΔL = len(tree.Nodes.keys()) - len(self.Population[id].Nodes)

        if isfinite(Δs) is False:
            # Deleting the temporary element
            del self.Population[-1]
            
            if len(tree.sConstants) != 0:
                del self.Constants["tmp"]

            return None

        alpha = exp(Δs / self.Temperature)

        if isfinite(alpha) is False:
            # Deleting the temporary element
            del self.Population[-1]
            
            if len(tree.sConstants) != 0:
                del self.Constants["tmp"]

            return None

        # Accepting the move or not
        if uniform(0, 1) < alpha:
            if id in self.Constants and len(tree.sConstants) == 0:
                del self.Constants[id]

            self.Population[id] = deepcopy(tree)

            if "tmp" in self.Constants:
                self.Constants[id] = deepcopy(self.Constants["tmp"])

        # Deleting the temporary element
        del self.Population[-1]
        
        if len(tree.sConstants) != 0:
            del self.Constants["tmp"]

        

        return None

    """
    Performing a crossover
    """
    def crossover(self, A: int, B: int) -> None:
        pass
    
    """
    Running the Evolution - At each step, we proceed to several operations.

        1) Generating randomly homogeneous groups of the total population
        2) Making "tournaments" in each group and keeping the "fittest"
        3) Calculating the probability to "crossover"
            Proba > Pcross: We are mutating the fittest of each group
            Proba < Pcross: We are performing a crossover between two winners of tournaments
        4) We are replacing the old formula by their mutated counterpart as long as it is accepted
            
            The move is accepted according to three parameters:
                a) The complexity difference between the two structures;
                b) The score difference
                c) The "Temperature"
        
        "Temperature" should be understood in the context of Simulated Annealing (see https://en.wikipedia.org/wiki/Simulated_annealing):
        We first (on the first iteration) apply a very temperature, allowing to explore various configuration, and decrease it slowly, at
        each step. This allows to generally find an approximate solution to the global minimum (i.e. we are seeking).

        /!\ Several parameters are impacting directly the "precision" and the "performance" of the algorithm, and there is no universal values
        which allow both of them. For more information, please visit ...
    """
    def run(self):
        _n = 0

        self.Temperature = 1.0

        while _n < self.Niter:

            print(_n, self.Best_score)

            if self.Best_tree is not None:
                print(self.Best_tree.stringify())

            self.Best = set()

            # Generating groups of population
            ids = [i for i in range(self.Npop)]
            shuffle(ids)

            groups = [ids[i:i + self.Ngroup] for i in range(0, len(ids), self.Ngroup)]

            # For each group, we are performing a tournament
            # We are saving the fittest of each group into another list
            for _g in groups:
                _winner = self.tournament(_g)
                self.Best.add(_winner)

            # For each winner, we mutate it or perform a crossover with another winner
            for _winner in self.Best:

                # End of the set
                if _winner is None:
                    break

                if uniform(0, 1) > self.Pcross:
                    self.mutate(_winner)
                else:
                    _cross = choice(list(self.Best))
                    self.crossover(_winner, _cross)

            # Slight optimization (for constants)
            for _b in self.Best:
                self.optimize(_b)

            self.Temperature -= 1.0 / self.Niter
            _n += 1

x = linspace(0, 1)
y = exp(-x**2) * x

objective = list(column_stack((x, y)))
Evolution = Evolution(objective=objective, dimension=1)
Evolution.run()
print(Evolution.Best)