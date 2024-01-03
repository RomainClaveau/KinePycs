#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Evolution.py

"""
Author:             Romain Claveau
Version:            1 (stable)
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
from numpy import *
from random import *
from copy import *
from scipy.optimize import minimize

class Evolution:

    # NOTE: ...
    # TODO: ...

    # Numbers
    # Must have Npopulation mod Nisland = 0
    # and Nisland mod Ntournament = 0
    Npopulation = 1000  # Total population
    Nisland = 10        # Number of expressions inside an island
    Ntournament = 2    # Number of expressions during a tournament
    Niterations = 100    # Number of iterations before leaving the evolution process

    # Dictionary containing all the expressions
    # For each expression, we save:
    #   the tree
    #   the constants (if any)
    #   the score
    #   the number of mutations
    Population = {}

    # Dictionary of islands
    # Each island is independent of others, but elements may migrate
    # Prevents hasty over-specialization
    Islands = {}

    # Set containing the best expressions' id
    Best_overall = set()
    Best_islands = {}

    # Dictionary keeping track of the expressions' mutations
    Tracks = {}

    # Best overall expression
    Best_score = 1e99
    Best_id = None

    # Probabilities
    Proba_cross = 0.01      # Performing a crossover
    Proba_fit = 0.9         # Selecting the fittest
    Proba_migration = 0.05   # Performing a migration

    Verbose = True

    """
    Initializing the Evolution class - Defining the data to match and the system's dimension

    Arguments
    =========

        .(list) objective:  the function whose expression should be matched
                            should be shaped as (n, dim + 1)
                            where n is the number of observables
                            and dim the number of independent variables
        .(int) dimension:   the number of independent variables
                            e.g. for y(x) = x, the dimension is 1.

    Returns
    =======

        .None
    """
    def __init__(self, objective: list, dimension: int) -> None:
        
        # Checking the arguments' type
        if not isinstance(objective, (list)): raise TypeError("`objective` should be a list.")
        if not isinstance(dimension, (int)): raise TypeError("`dimension` should be an integer.")

        # Checking arguments' value
        if dimension <= 0: raise ValueError("`dimension` should higher than zero.")
        if len(objective) == 0 or len(objective[0]) != dimension + 1: raise ValueError("`objective` does not have the required shape.")

        # Saving the function to match and the associated dimension
        self.Objective = objective
        self.Dimension = dimension

        # Generating the initial population
        self.__generate__()

    """
    Generate the population and populating the islands

    Arguments
    =========

        .None

    Returns
    =======

        .None
    """
    def __generate__(self) -> None:
        
        # Generating the population
        for _id in range(self.Npopulation):

            # Generating a random tree
            _complexity, _constants, _tree = self.generate_tree(3)

            # Saving the expression
            self.Population[_id] = {
                "complexity": _complexity,
                "constants": _constants,
                "tree": _tree
            }

        # Populating the islands
        ids = list(self.Population.keys())
        shuffle(ids)

        for _n, _id in enumerate(ids):

            _k = int(_n / self.Nisland)

            if _k not in self.Islands:
                self.Islands[_k] = []

            self.Islands[_k].append(_id)

    """
    Generate a random tree with a given complexity

    Arguments
    =========

        .(int) complexity: The given complexity (number of nodes)

    Returns
    =======

        .(tuple) expr:  The tuple containing three elements
                        1) The complexity
                        2) The constants
                        3) The tree
    """
    def generate_tree(self, complexity: int) -> tuple:

        # Generating the initial node
        tree = Tree.Tree(dimension=self.Dimension)
        tree.add_node(type="var", value="x0")

        # Applying random operations until reaching the desired complexity
        while len(tree.Nodes) <= complexity:
            self.random_operation(tree)

        # Fixing the constants' values (if any)
        constants = {_k: uniform(-1, 1) for _k in tree.sConstants}

        # Returning the complexity, constants and tree
        return complexity, constants, tree
    
    """
    Generating a random operation on a tree

    Arguments
    =========

        .(object) tree: the tree to be operated on

    Returns
    =======

        .None
    """
    def random_operation(self, tree: object) -> None:

        # Selecting a node
        _id = choice(list(tree.Nodes.keys()))

        # Selecting an operation
        _op = choice(list(Tree._Operations.keys()))

        # Selecting an appendable type
        _at = choice(["var", "cst"])

        # Selecting an appendable row
        if _at == "var":
            _ar = choice(range(0, self.Dimension))
        else:
            _ar = -1

        # Applying the operation
        tree.apply_on_node(node=_id, operation=_op, appending_type=_at, appending_row=_ar)

    """
    Calculating the error
    """
    def error(self, expr: int or object) -> float:

        if isinstance(expr, (int)):

            # Loading the expression's function
            f = self.Population[expr]["tree"].lambdify(update=True)

            # Loading the constants
            constants = self.Population[expr]["constants"].values()

        else:
            
            # Loading the expression's function
            f = expr.lambdify(update=True)
            
            # Loading the constants
            constants = [uniform(-1, 1) for _cst in expr.sConstants]

        err = 0.0

        for _obs in self.Objective:

            # NOTE: _obs = [x0, x1, ..., xn]
            # we consider [x0, x1, ..., xn-1] to be the variables
            # and xn the observable such that
            # xn = f(x0, x1, ..., xn-1)

            try:
                err += abs(f(*[*_obs[:-1], *constants]) - _obs[-1])
            except Exception:
                return 1e99

        # Checking whether we get NaN of Inf
        if isfinite(err):
            return err
        else:
            return 1e99

    """
    Running a tournament

    Arguments
    =========

        .(list) exprs: List of expressions to be tested

    Returns
    =======

        .(tuple) results: The score and the id of the best expression
    """
    def tournament(self, exprs: list) -> tuple:

        results = []

        for _expr in exprs:
            err = self.error(_expr)
            results.append([err, _expr])

        results.sort(key=lambda x: x[0])

        for _res in results:
            if uniform(0, 1) < self.Proba_fit:
                return _res

        return results[0]

    """
    Optimizing the constants

    Arguments
    =========

        .(int) expr: The expression's id

    Returns
    =======

        .(tuple) results: Returning the score and the optimized constants
    """
    def optimize(self, expr: int) -> tuple:

        # Loading the expression's function
        f = self.Population[expr]["tree"].lambdify(update=True)

        # Loading the constants
        constants_init = self.Population[expr]["constants"].values()

        # Defining the equation to solve
        def Equation(csts):
            err = 0.0

            for _obs in self.Objective:

                # NOTE: _obs = [x0, x1, ..., xn]
                # we consider [x0, x1, ..., xn-1] to be the variables
                # and xn the observable such that
                # xn = f(x0, x1, ..., xn-1)

                try:
                    err += abs(f(*[*_obs[:-1], *csts]) - _obs[-1])
                except Exception:
                    return 1e99

            return err

        # Lauching optimization through Nelder-Mead algorithm
        # Enforcing the `adaptive` to `True` for multidimensional optimization
        try:
            sol = minimize(Equation, x0=list(constants_init), method="Nelder-Mead", options={"adaptive": True})
        except Exception:
            return 1e99, constants_init
        
        return sol.fun, sol.x

    """
    Mutating an expression
    """
    def mutate(self, expr: int) -> None:
        
        # Actions
        #   1) `switch`: mutate an operator
        #   2) `prepend`: append / prepend a node
        #   3) `apply`: apply an operation on a node
        #   4) `delete`: replacing a branching by a constant or a variable
        #   5) `new`: generate a new tree
        #   6) `nothing`: no modification
        what_to_do = choice([
            "switch", "prepend", "apply", "delete", "new", "nothing"
        ])

        # New tree
        tree = None

        match what_to_do:
            case "switch": tree = self.mutation_switch(expr)
            case "prepend": tree = self.mutation_prepend(expr)
            case "apply": tree = self.mutation_apply(expr)
            case "delete": tree = self.mutation_delete(expr)
            case "new": tree = self.mutation_new(expr)
            case "nothing": pass

        # Modifications were done on the tree
        if tree is not None:
            
            # Computing score
            old_score = self.error(expr)
            new_score = self.error(tree)

            # Computing complexity
            old_complexity = len(self.Population[expr]["tree"].Nodes)
            new_complexity = len(tree.Nodes)

            # Computing complexity's occurrence
            occurrences = {}

            for _expr in self.Population.values():
                _s = len(_expr["tree"].Nodes)
                
                if _s not in occurrences:
                    occurrences[_s] = 0

                occurrences[_s] += 1

            if old_complexity not in occurrences:
                occurrences[old_complexity] = 0

            if new_complexity not in occurrences:
                occurrences[new_complexity] = 0

            old_occ = occurrences[old_complexity]
            new_occ = occurrences[new_complexity]

            # Accepting or not the move
            q_annealing = exp((old_score - new_score) / max(1e-15, self.Temperature))
            q_occurrence = old_occ / max(1e-15, new_occ)
            q_complexity = old_complexity / max(1e-15, new_complexity)

            # The higher, the better
            q_a = old_complexity / old_score
            q_b = new_complexity / new_score

            q_annealing = exp(-(q_a - q_b) / self.Temperature)

            # If we accept the move
            if uniform(0, 1) < q_annealing:
                
                # Retrieving constants (with random values)
                constants = {_k: uniform(-1, 1) for _k in tree.sConstants}

                # Saving it
                self.Population[expr]["tree"] = deepcopy(tree)
                self.Population[expr]["constants"] = deepcopy(constants)


    def mutation_switch(self, expr: int) -> object:

        # Selecting a random operation node
        _id = choice([_k for _k, _v in self.Population[expr]["tree"].Nodes.items() if _v["type"] == "op"])
        _op = self.Population[expr]["tree"].Nodes[_id]["value"]

        # Selecting another operation with the same arity
        _op_new = choice([_k for _k, _v in Tree._Operations.items() if _v["arity"] == Tree._Operations[_op]["arity"] and _k != _op])

        # Creating a copy of the actual tree
        tree = deepcopy(self.Population[expr]["tree"])

        # Applying the modification
        tree.Nodes[_id]["value"] = _op_new

        return tree

    def mutation_prepend(self, expr: int) -> object:

        # Creating a copy of the actual tree
        tree = deepcopy(self.Population[expr]["tree"])

        # Generating a random expression (of complexity between 1 and 3)
        _, _, branch = self.generate_tree(complexity=randint(1, 3))
        
        # We are prepending an expression at the top of the actual tree
        if uniform(0, 1) < 0.5:
            tree.prepend_tree(node=-1, tree=branch)
        
        # Otherwise, we replace a bottom node
        else:
            _id = choice([_node for _node, _v in tree.Children.items() if len(_v) == 0])
            tree.prepend_tree(node=_id, tree=branch)

        return tree

    def mutation_apply(self, expr: int) -> object:

        # Creating a copy of the actual tree
        tree = deepcopy(self.Population[expr]["tree"])

        # Applying a random operation on the tree
        self.random_operation(tree)

        return tree

    def mutation_delete(self, expr: int) -> object:

        # Creating a copy of the actual tree
        tree = deepcopy(self.Population[expr]["tree"])

        # Selecting a random node (except the highest one)
        _id = choice(list(tree.Nodes.keys()))

        while len(tree.Parents[_id]) == 0:
            _id = choice(list(tree.Nodes.keys()))

        # Retrieving the branch parent
        _parent = [_node for _node, _v in tree.Children.items() if _id in _v][0]

        # Deleting the branching
        tree.delete_branching_from(node=_id, include=True)

        # Adding a node
        _nt = choice(["var", "cst"])

        if _nt == "var":
            _nr = choice(tree.sVars)
        else:
            if len(tree.sConstants) != 0:
                _nr = f"c{int(tree.sConstants[-1][1:]) + 1}"
            else:
                _nr = "c0"

        _node = tree.add_node(type=_nt, value=_nr)
        tree.add_edge([_node, _parent])

        return tree

    def mutation_new(self, expr: int) -> object:

        complexity = choice(list(range(3, 5)))

        _, _, tree = self.generate_tree(complexity)
        
        return tree

    def crossover(self, _a: int, _b: int) -> None:

        # Selecting random nodes for each element
        _id_a = choice(list(self.Population[_a]["tree"].Nodes.keys()))
        _id_b = choice(list(self.Population[_b]["tree"].Nodes.keys()))

        # Creating copies
        _tree_a = deepcopy(self.Population[_a]["tree"])
        _tree_b = deepcopy(self.Population[_b]["tree"])

        # Creating branches
        _branch_a = _tree_a.branch_to_tree(_id_a)
        _branch_b = _tree_b.branch_to_tree(_id_b)

        # Updating ids in the case the selected node is the highest one
        if _id_a == [_k for _k, _v in _tree_a.Parents.items() if len(_v) == 0][0]:
            _id_a = -1

        if _id_b == [_k for _k, _v in _tree_b.Parents.items() if len(_v) == 0][0]:
            _id_b = -1

        # Proceeding to replacements
        _tree_a.prepend_tree(_id_a, _branch_b)
        _tree_b.prepend_tree(_id_b, _branch_a)

        # Accepting or not the move

        # Computing scores
        score_a = self.error(_a)
        score_b = self.error(_b)
        score_new_a = self.error(_tree_a)
        score_new_b = self.error(_tree_b)

        # Updating A
        if score_a < score_new_a:
            self.Population[_a]["tree"] = deepcopy(_tree_a)
            self.Population[_a]["constants"] = {_k: uniform(-1, 1) for _k in _tree_a.sConstants}

        # Updating B
        if score_b < score_new_b:
            self.Population[_b]["tree"] = deepcopy(_tree_b)
            self.Population[_b]["constants"] = {_k: uniform(-1, 1) for _k in _tree_b.sConstants}

    def migration(self):

        # Selecting two islands
        _isl_a = choice(list(self.Islands.keys()))
        _isl_b = choice(list(self.Islands.keys()))

        while _isl_a == _isl_b:
            _isl_a = choice(list(self.Islands.keys()))
            _isl_b = choice(list(self.Islands.keys()))
        
        # Selecting an element in each island
        _id_a = choice(self.Islands[_isl_a])
        _id_b = choice(self.Islands[_isl_b])

        # Performing the exchange
        _store = deepcopy(self.Population[_id_b])

        self.Population[_id_b] = deepcopy(self.Population[_id_a])
        self.Population[_id_a] = _store


    """
    Running the evolution - At each step, we proceed to several operations.

    Arguments
    =========

        .(float) tol: the error to reach before leaving the evolution process

    Returns
    =======

        .None
    """
    def run(self, tol: float = 1e-5) -> None:

        self.Temperature = 1.0

        for _iter in range(self.Niterations):
            
            # Iterating over islands
            for _island_id, _island in self.Islands.items():

                # Shuffling the order of the island
                _exprs = _island[:]
                shuffle(_exprs)

                groups = {}

                # Creating groups for tournaments
                for _n, _k in enumerate(_exprs):
                    _id = int(_n / self.Ntournament)

                    if _id not in groups:
                        groups[_id] = []

                    groups[_id].append(_k)

                #
                # 1) Running a tournament per group
                #
                winners = []

                for _k, _g in groups.items():
                    best_score, best_expr = self.tournament(_g)

                    winners.append(best_expr)

                    if _island_id not in self.Best_islands:
                        self.Best_islands[_island_id] = {
                            "score": best_score,
                            "expr": best_expr
                        }

                    if best_score < self.Best_islands[_island_id]["score"]:
                        self.Best_islands[_island_id] = {
                            "score": best_score,
                            "expr": best_expr
                        }

                # Retrieving the island's best
                best = self.Best_islands[_island_id]["expr"]

                # Optimizing constants for island's best expression
                score, constants = self.optimize(best)

                # Updating the score
                self.Best_islands[_island_id]["score"] = score

                # Saving the constants (if any)
                for _k, _v in enumerate(constants):
                    csts = list(self.Population[best]["constants"].keys())
                    self.Population[best]["constants"][csts[_k]] = _v

                #
                # 2) Mutating groups winners
                #
                for _w in winners:
                    self.mutate(_w)

                #
                # 3) Performing crossover in the island
                #
                if uniform(0, 1) < self.Proba_cross:
                    
                    # Selecting two random elements
                    _a = choice(winners)
                    _b = choice(winners)

                    while _a == _b:
                        _a = choice(winners)
                        _b = choice(winners)

                    # Performing a crossover
                    self.crossover(_a, _b)

            #
            # 4) Performing crossover in the ocean
            #
            if uniform(0, 1) < self.Proba_cross:
                    
                # Selecting two random islands
                _a = choice(list(self.Best_islands.keys()))
                _b = choice(list(self.Best_islands.keys()))

                while _a == _b:
                    _a = choice(list(self.Best_islands.keys()))
                    _b = choice(list(self.Best_islands.keys()))

                # Performing a crossover
                self.crossover(self.Best_islands[_a]["expr"], self.Best_islands[_b]["expr"])

            #
            # 5) Performing migrations
            #
            self.migration()

            #
            # 6) Optimizing constants
            #
            
            for _n in self.Population.keys():
                
                _score, _constants = self.optimize(_n)

                # Saving constants
                _ks = list(self.Population[_n]["constants"].keys())

                for _id, _ in enumerate(_ks):
                    self.Population[_n]["constants"][_ks[_id]] = _constants[_id]

            #
            # 7) Retrieving the best
            #
            best = sorted(
                list(self.Best_islands.values()),
                key=lambda x: x["score"]
            )[0]

            best_expr = best["expr"]
            best_score = best["score"]

            print(self.Population[best_expr]["tree"].stringify(update=True))
            print(best_score)

            # Updating the temperature
            self.Temperature -= 1.0 / self.Niterations

            if self.Temperature < 0:
                self.Temperature = 0