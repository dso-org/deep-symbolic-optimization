import numpy as np
from sympy import symbols, lambdify
from sympy.parsing.sympy_parser import parse_expr

from dsr import utils as U

"""Symbolic expression class including vectorized evaluation"""
class Expression:

    def __init__(self, traversal=None, tree=None, order="preorder"):

        self.order = order

        if tree is None:
            # self.size = len(traversal)
            self.tree = build_tree(traversal.copy(), order=order) # type: Node            

        elif traversal is None:
            self.tree = tree

        else:
            raise ValueError("Must provide either traversal or expression tree")

        self.depth = self.tree.depth()

        self.expr = parse_expr(self.tree.__repr__()) # SymPy symbolic expression
        self.x = symbols(' '.join('x{}'.format(i+1) for i in range(U.n_inputs)))

        self.np_lambda = lambdify(self.x, self.expr, modules=np) # Vectorized lambda function

        self.loss = lambda X, y : np.mean(np.square(self.np_lambda(*X.T) - y))

    def __repr__(self):
        return self.expr.__repr__()

"""Dataset class defined by ground truth expression"""
class Dataset:

    def __init__(self, ground_truth, n_samples=600, domain=(-5, 5), np_seed=0):

        self.ground_truth = ground_truth
        self.rng = np.random.RandomState(seed=np_seed)
        
        # x = tuple(sorted(self.ground_truth.free_symbols, key=str)) # Sort symbols by str representation
        x = symbols(' '.join('x{}'.format(i+1) for i in range(U.n_inputs)))
        np_lambda = lambdify(x, self.ground_truth, modules=np)

        self.X = self.rng.uniform(low=domain[0], high=domain[1], size=(n_samples, len(x)))
        self.y = np_lambda(*self.X.T)


"""Basic tree class supporting printing"""
class Node:

    def __init__(self, val):
        self.val = val
        self.children = []

    def depth(self):
        if len(self.children) == 0:
            return 1
        return max([c.depth() for c in self.children]) + 1

    def size(self):
        if len(self.children) == 0:
            return 1
        return sum([c.size() for c in self.children]) + 1

    def preorder_traversal(self):
        return [self.val] + [c.preorder_traversal() for c in self.children]

    def __repr__(self):
        children_repr = ",".join(repr(child) for child in self.children)
        if len(self.children) == 0:
            return U.library[self.val] # Avoids unnecessary parantheses, e.g. x1()
        return "{}({})".format(U.library[self.val], children_repr)

"""Recursively builds tree from pre-order traversal"""
def build_tree(traversal, order="preorder"):

    if order == "preorder":
        val = traversal.pop(0)
        n_children = U.n_children[val]
        node = Node(val)
        for _ in range(n_children):
            node.children.append(build_tree(traversal))
        return node

    elif order == "postorder":
        raise NotImplementedError

    elif order == "inorder":
        raise NotImplementedError

if __name__ == "__main__":

    # Example: (x0 + x1) * sin(x1 * x2)
    traversal = U.convert(["Mul", "Add", "x0", "x1", "sin", "Mul", "x1", "x2"])
    expr = Expression(traversal=traversal)

    if traversal == ["Mul", "Add", "x0", "x1", "sin", "Mul", "x1", "x2"]:
        # Assert tree is correct
        node = expr.tree
        assert U.library[node.val] == "Mul"
        assert len(node.children) == 2
        assert U.library[node.children[0].val] == "Add"
        assert U.library[node.children[1].val] == "sin"
        assert len(node.children[0].children) == 2
        assert U.library[node.children[0].children[0].val] == "x0"
        assert U.library[node.children[0].children[1].val] == "x1"
        assert len(node.children[1].children) == 1
        assert U.library[node.children[1].children[0].val] == "Mul"
        assert len(node.children[1].children[0].children) == 2
        assert U.library[node.children[1].children[0].children[0].val] == "x1"
        assert U.library[node.children[1].children[0].children[1].val] == "x2"

        # Assert printout is correct
        assert repr(node) == "Mul(Add(x0,x1),sin(Mul(x1,x2)))"

    ground_truth = parse_expr("Mul(Add(x0,x1),sin(Mul(x1,x2)))")
    dataset = Dataset(ground_truth)

    print("Ground truth:", dataset.ground_truth)
    print("Expression evaluated:", expr.expr)
    print("Loss:", expr.loss(dataset.X, dataset.y))

    print(U.convert(U.flatten(expr.tree.preorder_traversal())))