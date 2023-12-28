#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Tree.py

"""
Author:             Romain Claveau
Version:            0
Python version:     3.11.7
Dependencies:       numpy, re
License:            CC BY-NC-SA (https://creativecommons.org/licenses/by-nc-sa/4.0/)

Reporting an issue:  https://github.com/RomainClaveau/KinePycs/issues
"""

"""
=============================================
Mathematical expression `Tree` representation
=============================================

Creating a `Tree` representing a mathematical expression
that could be modified through operations on nodes.

Operations
==========

    .add_node       -   Adding a node into the tree
    .add_edge       -   Adding an edge between two nodes
    .delete_edge    -   Removing an edge
    .apply_on_node  -   Apply an mathematical operation on a node
    .stringify      -   Return a string representing the mathematical Tree
    .lambdify       -   Return a callable function representing the mathematical Tree

Example
=======

>> T = Tree(dimension=1)
>> id_node_0 = T.add_node(value='x0', type='var')
    --> f(X) = x0
>> id_nodes = T.apply_on_node(node=id_node_0, operation='plus', appending_type='cst', appending_row=-1)
    --> f(X) = c0 + x0
>> T.apply_on_node(node=id_nodes[0], operation='div', appending_type='var', appending_row=-1)
    --> f(X) = c1 / (c0 + x0)

Final structure
---------------

    .nodes
        ..0: X
        ..1: plus
        ..2: c0
        ..3: div
        ..4: c1
    .edges
        ..0 -> 1
        ..2 -> 1
        ..1 -> 3
        ..4 -> 3
"""

from numpy import *
from re import *

"""
Basics operations to be applied on variables or constants

Available operations
====================
    .exp --> exp(x)
    .log --> log(x) (using `e` as a basis)
    .neg --> -x
    .inv --> 1 / x (unprotected against x = 0 singularity)
    .abs --> |x|
    .sqrt --> √(x) (unprotected to prevent complex values)
    .pow --> x**y (unprotected to prevent complex values)
    .plus --> x + y
    .minus --> x - y
    .mult --> x * y
    .div --> x / y (unprotected against y = 0 singularity)
"""
_Operations = {
    # Identity operation to avoid stringify and lambdify 
    # to fail whenever these are applied on no-operation trees like `f(x) = x`
    "id": {"arity": 1, "func": lambda x: x},

    "exp": {"arity": 1, "func": lambda x: exp(x)},
    "log": {"arity": 1, "func": lambda x: log(x)},
    "neg": {"arity": 1, "func": lambda x: -x},
    "inv": {"arity": 1, "func": lambda x: 1 / x},
    "abs": {"arity": 1, "func": lambda x: abs(x)},
    "sqrt": {"arity": 1, "func": lambda x: sqrt(x)},

    "pow": {"arity": 2, "func": lambda x, y: power(x, y)},
    "plus": {"arity": 2, "func": lambda x, y: x + y},
    "minus": {"arity": 2, "func": lambda x, y: x - y},
    "mult": {"arity": 2, "func": lambda x, y: x * y},
    "div": {"arity": 2, "func": lambda x, y: x / y},
}

"""
Custom errors

Types
=====
    .DuplicateError raised when an `id` duplication is detected
    .OperatorError  raised when an operator is not in the predefined list `_Operations`
    .VariableError  raised when a variable is not in the variables list {x0, ..., xn} with n being the system's dimension
    .SizeError      raised when an edge is used to connect more than two nodes
    .OverError      raised when a node is overconnected (> 3 legs when arity is 2 or > 2 legs when arity is 1)
    .SelfError      raised when a node is connected with itself
    .SameError      raised when two nodes are already connected
    .NodeError      raised when trying to connect node that does not exist
    .EdgeError      raised when the specified edge does not exist
    .OperatorError  raised when the specified operation does not exist
    .AppendError    raised when the specified type is not appendable (i.e. others types than `var` and `cst`)
    .RowError       raised when the specified row is out-of-range
    .StringifyError raised when the string representation of the Tree is incomplete (not all nodes are represented)
    .LoopError      raised when the parsing loop is stuck or does not converge to a complete tree string representation
    .LambdifyError  raised when the parsed string representation of the Tree could not be evaluated as a function
"""
class DuplicateError(Exception): pass
class OperatorError(Exception): pass
class VariableError(Exception): pass
class SizeError(Exception): pass
class OverError(Exception): pass
class SelfError(Exception): pass
class SameError(Exception): pass
class NodeError(Exception): pass
class EdgeError(Exception): pass
class OperatorError(Exception): pass
class AppendError(Exception): pass
class RowError(Exception): pass
class StringifyError(Exception): pass
class LoopError(Exception): pass
class LambdifyError(Exception): pass

class Tree:
    """
    Initializing the tree

    Arguments
    =========

        .(int) dimension: dimension of the system

    Returns
    =======

        .None
        .(Exception) TypeError: if the specified `dimension` is below 1

    Example
    =======

        >> T = Tree(dimension=2)
            --> Studying y = f(x0, x1)
    """
    def __init__(self, dimension) -> None or Exception:

        # Creating all the Tree's properties
        self.Dimension = None    # Dimension of the system

        self.Nodes = {}          # Nodes values
        self.Edges = []          # Edges list

        self.Expr = None         # String representation of the formula
        self.Lambda = None       # Lambda function of the formula

        self.sVars = []          # Variables symbols
        self.sConstants = []     # Constants symbols

        self.Parents = {}        # Parent of each node (if any)
        self.Children = {}       # Children(s) of each node (if any)

        self.iVars = []          # Variables list
        self.iConstants = []     # Constants list
        self.iOperators = []     # Operators list

        self.Stringified = None  # Stringified expression of the tree (string)
        self.Lambdified = None

        # Checking if dimension >= 0
        if not isinstance(dimension, (int)) or int(dimension) <= 0:
            raise TypeError("`dimension` should be an integer higher than zero.")
        
        # Saving the system's dimension
        self.Dimension = dimension

        # For each dimension, we associate a variable
        # y = f(x0, ..., xn)
        self.sVars = [f"x{i}" for i in range(0, self.Dimension)]

        return None
    
    """
    Adding a node into the tree

    Arguments
    =========

        .(str) value: value of the node, as a string
        .(str) type: type of the node, as a string
            .."var": variable
            .."cst": constant
            .."op": operator
    
    Returns
    =======

        .(int) node_id:                 id of the created node
        .(Exception) TypeError:         when the node `value` or `type` is not a string
        .(Exception) ValueError:        when the type of node do not match that of a variable (var), constant (cst)
                                        or of an operator (op)
        .(Exception) VariableError:     when the variable `value` is not in the variables list
        .(Exception) DuplicateError:    when an object duplication is detected
        .(Exception) OperatorError:     when the specified operator is not in the predefined operations list

    Example
    =======

        >> T = Tree(dimension=1)
        >> T.add_node(value='x0', type='var')
    """
    def add_node(self, value: str, type: str) -> int or Exception:

        # Checking arguments
        if not isinstance(value, (str)): raise TypeError("Not a string")
        if not isinstance(type, (str)): raise TypeError("Not a string")
        if type not in ("var", "cst", "op"): raise ValueError("`type` should be `var`, `cst` or `op`")

        # Retrieving the last id in the nodes' list
        if len(self.Nodes.keys()) == 0:
            _id = 0
        else:
            _id = list(self.Nodes.keys())[-1] + 1

        # Adding the node into the nodes' list
        self.Nodes[_id] = {"value": value, "type": type}

        # Adding the node into its type list
        match type:
            case "var":
                if value not in self.sVars: raise VariableError(f"Variable `{value}` does not exist.")
                if _id in self.iVars: raise DuplicateError(f"Object `{_id}` is duplicated.")
                self.iVars.append(_id)
            case "cst":
                if _id in self.iConstants: raise DuplicateError(f"Object `{_id}` is duplicated.")
                self.iConstants.append(_id)
                if value not in self.sConstants: self.sConstants.append(f"c{value}")
            case "op":
                if value not in _Operations: raise OperatorError(f"Operation `{value}` does not exist.")
                if _id in self.iOperators: raise DuplicateError(f"Object `{_id}` is duplicated.")
                self.iOperators.append(_id)

        # Adding the node to the parents and the children list
        self.Parents[_id] = []
        self.Children[_id] = []

        # Returning the newly created node's id
        return _id

    """
    Adding an edge (or multiple edges) between two (or more) nodes

    Arguments
    =========

        .(list) edges: pairwise list of nodes to be connected

    Returns
    =======

        .None
        .(Exception) TypeError: when the provided `edges` is not a list nor pairwise
        .(Exception) SizeError: when the edge does not connect exactly two nodes
        .(Exception) OverError: when a node is overconnected (more than three legs for an arity of two
                                or more than two legs for an arity of one)
        .(Exception) SelfError: when a node is connected to itself
        .(Exception) SameError: when an edge already exists
        .(Exception) NodeError: when a node does not exist
    """
    def add_edge(self, edges: list) -> None or Exception:

        # Checking if `edges` is a list
        if not isinstance(edges, (list)): raise TypeError("`edges` should be a list.")

        # Checking if `edges` does contain more than one edge
        if isinstance(edges[0], (list)):
            # We expect edges = [[A, B], [C, D], ...]
            for edge in edges:
                self.add_edge(edge)
        else:
            # We expect edges = [A, B]

            # Checking if the edge connects exactly two nodes
            if len(edges) != 2: raise SizeError("`edges` should connect two nodes.")

            # Checking if the edge does not connect a node with itself
            if edges[0] == edges[1]: raise SelfError("Could not connect a node with itself.")

            # Checking if the edge does not already exist
            if edges in self.Edges: raise SameError("Edge already exists.")

            # Checking if each node exist
            if not all([node in self.Nodes for node in edges]): raise NodeError("Specified Node does not exist.")

            # Checking for connections
            for node in edges:
                if node in self.iOperators and _Operations[self.Nodes[node]["value"]]["arity"] == 2:
                    # Connections <= 3
                    if len(self.Children[node]) + len(self.Parents[node]) > 3:
                        raise OverError(f"Node `{node}` is overconnected.")
                else:
                    # Connections <= 2
                    if len(self.Children[node]) + len(self.Parents[node]) > 2:
                        raise OverError(f"Node `{node}` is overconnected.")

            # Adding the edge
            # Assuming the tree to be directed
            # edges = [A, B] <==> A -> B
            # with A = child and B = parent
            self.Edges.append(edges)

            # Updating children and parents list
            self.Parents[edges[0]].append(edges[1])
            self.Children[edges[1]].append(edges[0])

            return None

    """
    Deleting an edge

    Arguments
    =========

        .(list) edge: Edge to be deleted

    Returns
    =======

        .None
        .(Exception) EdgeError: The specified edge does not exist
    """
    def delete_edge(self, edge: list) -> None or Exception:

        if edge not in self.Edges: raise EdgeError("Edge does not exist.")

        self.Parents[edge[0]].remove(edge[1])
        self.Children[edge[1]].remove(edge[0])
        self.Edges.remove(edge)

        return None

    """
    Applying an operation on a node

    Arguments
    =========

        .(int) node:            Node's id
        .(str) operation:       Operation to be applied on the node
        .(str) appending_type:  Node's type to be appended while applying the operation on the node
        .(int) appending_row:   Select the n-th row of the appended type (either `cst` or `var`)

    Returns
    =======

        .None
        .(Exception) TypeError:     The variable's type is incorrect
        .(Exception) NodeError:     The node does not exist
        .(Exception) OperatorError: The operation does not exist
        .(Exception) AppendError:   The specified type is not appendable
        .(Exception) RowError:      The specified row is out-of-range
    """
    def apply_on_node(self, node: int, operation: str, appending_type: str = "var", appending_row: int = -1) -> None or Exception:

        # Checking arguments' type
        if not isinstance(node, (int)): raise TypeError("`node` should be an integer")
        if not isinstance(operation, (str)): raise TypeError("`operation` should be a string")
        if not isinstance(appending_type, (str)): raise TypeError("`appending_type` should be a string")
        if not isinstance(appending_row, (int)): raise TypeError("`appending_row` should be an integer")

        # Checking arguments' value
        if node not in self.Nodes: raise NodeError("Specified Node does not exist.")
        if operation not in _Operations: raise OperatorError(f"Operation `{operation}` does not exist.")
        if appending_type not in ("var", "cst"): raise AppendError(f"{appending_type} is not an appendable type.")

        # Adding the node
        _id = self.add_node(value=operation, type="op")

        # Updating the edges
        if len(self.Parents[node]) != 0:
            _p = self.Parents[node][-1]

            self.delete_edge([node, _p])
            self.add_edge([_id, _p])

        # Adding the edge
        self.add_edge([node, _id])

        # Checking if the operation arity is greater than one
        if _Operations[operation]["arity"] == 2:

            # Creating a new constant in the case when appending_row = -1
            if appending_type == "cst":
                if appending_row == -1:
                    if len(self.sConstants) == 0:
                        _id_append = 0
                    else:
                        _id_append = len(self.sConstants)

                    appending_row = _id_append

                    # Adding
                    self.sConstants.append(f"c{appending_row}")
                
                if f"c{appending_row}" not in self.sConstants: raise RowError("Specified row does not match any element.")

                # Adding the appending's node
                _id2 = self.add_node(value=f"c{appending_row}", type=appending_type)

            if appending_type == "var":
                if appending_row == -1:
                    appending_row = int(self.sVars[-1][1:])

                if f"x{appending_row}" not in self.sVars: raise RowError("Specified row does not match any element.")

                # Adding the appending's node
                _id2 = self.add_node(value=f"x{appending_row}", type=appending_type)

            # Adding the new edge between the operator and the appended node
            self.add_edge([_id2, _id])

        return _id

    """
    Stringify the tree - Returning a mathematical expression representing the tree

    Arguments
    =========

        .(bool) update: Forcing (True) or not (False) the update of the stringified expression

    Returns
    =======

        .(str) tree_str:                Stringified tree (almost) ready for interpretation
        .(Exception) StringifyError:    Not all nodes are accounted for in the stringified expression
        .(Exception) LoopError:         Whether an infinite loop is detected or if it does not lead to
                                        a complete expression for the tree string representation
    """
    def stringify(self, update=False) -> str or Exception:

        if self.Stringified is not None and update is False:
            return self.Stringified

        # NOTE: the current `stringify` implementation may be not the more efficient
        # as it behaves as O(N log(N)) but is better than a double `for` loop which
        # behave as O(N^2), where N is the number of `blocks`.
        
        # Operator blocks (parent) encapsulating nodes (children)
        blocks = {}

        tree_str = None

        # Operation nodes always have child (or children)
        for _op in self.iOperators:
            # Creating the block corresponding to `_op` which refers to its children
            blocks[f"${_op}$"] = f"${_op}$({','.join([f'${str(_val)}$' for _val in self.Children[_op]])})"

            # Retrieving the highest operator
            if len(self.Parents[_op]) == 0:
                tree_str = blocks[f"${_op}$"]
                del blocks[f"${_op}$"]

        # Initializing a loop counter to avoid infinite `while` looping
        _c = 0

        while len(blocks) > 0:
            for _id in dict(blocks):

                # Detecting operations which are not injected yet
                if tree_str.find(_id) != -1 and tree_str.find(_id + "(") == -1:
                    tree_str = tree_str.replace(_id, blocks[_id])
                    del blocks[_id]

            _c += 1

            # Emergency exit from infinite looping
            if _c > len(self.iOperators): raise LoopError("Exiting because of an infinite loop detected.")

        # If all replacements were performed, `blocks` should be of length 1
        if len(blocks) == 0:

            # Checking that all nodes are represented in the final expression
            if sorted([int(_x) for _x in findall(r'\$(\d+)\$', tree_str)]) != sorted(self.Nodes.keys()):
                raise StringifyError("Incomplete representation of the Tree.")
            
            # Now, we are replacing the variables $...$ by their true value
            for _id, _val in self.Nodes.items():
                tree_str = tree_str.replace(f"${_id}$", _val["value"])

            self.Stringified = tree_str

            return tree_str

        # We may assume that replacement/injection is not complete
        raise LoopError("Failed to stringify the tree.")

    """
    Lambdify the stringified expression of the tree - Returning a ready-to-use function as a lambda function
    in the form lambda x0, ..., xn, c0, ..., cn: f(*args) with {x0, ..., xn} the variables and {c0, ..., cn}
    the constants

    Arguments
    =========

        .(bool) update: Forcing (True) or not (False) the update of the lambdified expression

    Returns
    =======

        .(callable) tree_lambda:    the lambda function representing the tree
        .(Exception) LambdifyError: the expression could not be converted into a callable function
    """
    def lambdify(self, update=False) -> callable or Exception:

        # NOTE: the lambdify function does use the `EVAL` instruction which may be insecure when applied
        # to arbitrary and unescaped content. Be sure to exclusively pass expressions stemming from the Tree
        # class as it should only contains safe expressions.

        # For now, to avoid any prompt injection, several things/tips:
        #   1) Check every new mathematical expression added to the list of operations
        #   2) Tree's variables are not protected against out-of-the-class modification

        # Please, bear in mind this caution before executing any (possibly unsafe) content.
        # Hopefully, security will be enhanced in forthcoming versions.

        
        if self.Lambdified is not None and update is False:
            return self.Lambdified

        if self.Stringified is None or update is True:
            self.stringify(update=True)

        tree_expr = self.Stringified

        # Avoid looping over all operations
        for _op in set(findall(r'([a-zA-Z]+)\(', tree_expr)):

            # Avoiding possible name's overlapping
            tree_expr = tree_expr.replace(f"{_op}(", f"_Operations['{_op}']['func'](")

        # Concatenate arguments for the lambda function formatting
        _params = ",".join(self.sVars + self.sConstants)

        # Formatting the lambda function
        tree_lambda_as_str = f"lambda {_params}: {tree_expr}"

        # Checking if the generated `tree_lambda_as_str` could be evaluated
        try:
            self.Lambdified = eval(tree_lambda_as_str)
        except LambdifyError as e:
            raise LambdifyError("Could not evaluate the expression.")

        # Should return a function
        return self.Lambdified

"""
Timeit
======

    # > 793.764 op / sec
    >> T = Tree(dimension=1)                    x 1             7.700000423938036e-06   s
    >> T = Tree(dimension=1)                    x 1.000         0.0013624999992316589   s
    >> T = Tree(dimension=1)                    x 1.000.000     1.2598193000012543      s

    # ~ 503.198 op / sec
    >> T.add_node(value="x0", type="var")       x 1             1.3700002455152571e-05  s
    >> T.add_node(value="x0", type="var")       x 1.000         0.0019834000049741007   s
    >> T.add_node(value="x0", type="var")       x 1.000.000     1.987286800002039       s

    # ~ 46.098 op / sec
    >> T.apply_on_node(id0, "mult", "cst", -1)  x 1             0.00010540000221226364  s
    >> T.apply_on_node(id0, "mult", "cst", -1)  x 1.000         0.024609899999632034    s
    >> T.apply_on_node(id0, "mult", "cst", -1)  x 1.000.000     21.692502400001104      s

    # ~ 30.826 op / sec
    >> T.stringify()                            x 1             8.920000254875049e-05   s
    >> T.stringify()                            x 1.000         0.03659950000292156     s
    >> T.stringify()                            x 1.000.000     32.43970980000449       s

    # ~ 12.209 op / sec
    >> T.lambdify()                             x 1             0.00014659999578725547  s
    >> T.lambdify()                             x 1.000         0.09187039999960689     s
    >> T.lambdify()                             x 1.000.000     81.90408880000177       s
"""