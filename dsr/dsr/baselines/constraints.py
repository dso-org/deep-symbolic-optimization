"""Defines constraints for GP individuals, to be used as decorators for
evolutionary operations."""

from dsr.functions import UNARY_TOKENS, BINARY_TOKENS

TRIG_TOKENS = ["sin", "cos", "tan", "csc", "sec", "cot"]

# Define inverse tokens
INVERSE_TOKENS = {
    "exp" : "log",
    "neg" : "neg",
    "inv" : "inv",
    "sqrt" : "n2"
}

# Add inverse trig functions
INVERSE_TOKENS.update({
    t : "arc" + t for t in TRIG_TOKENS
    })

# Add reverse
INVERSE_TOKENS.update({
    v : k for k, v in INVERSE_TOKENS.items()
    })

DEBUG = False


def check_inv(ind):
    """Returns True if two sequential tokens are inverse unary operators."""

    names = [node.name for node in ind]
    for i, name in enumerate(names[:-1]):
        if name in INVERSE_TOKENS and names[i+1] == INVERSE_TOKENS[name]:
            if DEBUG:
                print("Constrained inverse:", ind)
            return True
    return False


def check_const(ind):
    """Returns True if children of a parent are all const tokens."""

    names = [node.name for node in ind]
    for i, name in enumerate(names):
        if name in UNARY_TOKENS and names[i+1] == "const":
            if DEBUG:
                print("Constrained const (unary)", ind)
            return True
        if name in BINARY_TOKENS and names[i+1] == "const" and names[i+1] == "const":
            if DEBUG:
                print(print("Constrained const (binary)", ind))
            return True
    return False


def check_trig(ind):
    """Returns True if a descendant of a trig operator is another trig
    operator."""
    
    names = [node.name for node in ind]
    trig_descendant = False # True when current node is a descendant of a trig operator
    trig_dangling = None # Number of unselected nodes in trig subtree
    for i, name in enumerate(names):
        if name in TRIG_TOKENS:
            if trig_descendant:
                if DEBUG:
                    print("Constrained trig:", ind)
                return True
            trig_descendant = True
            trig_dangling = 1
        elif trig_descendant:
            if name in BINARY_TOKENS:
                trig_dangling += 1
            elif name not in UNARY_TOKENS:
                trig_dangling -= 1
            if trig_dangling == 0:
                trig_descendant = False
    return False


def make_check_min_len(min_length):
    """Creates closure for minimum length constraint"""

    def check_min_len(ind):
        """Returns True if individual is less than minimum length"""

        if len(ind) < min_length:
            if DEBUG:
                print("Constrained min len: {} (length {})".format(ind, len(ind)))
            return True

        return False

    return check_min_len


def make_check_max_len(max_length):
    """Creates closure for maximum length constraint"""

    def check_max_len(ind):
        """Returns True if individual is greater than maximum length"""

        if len(ind) > max_length:
            if DEBUG:
                print("Constrained max len: {} (length {})".format(ind, len(ind)))
            return True

        return False

    return check_max_len


def make_check_num_const(max_const):
    """Creates closure for maximum number of constants constraint"""

    def check_num_const(ind):
        """Returns True if individual has more than max_const const tokens"""

        num_const = len([t for t in ind if t.name == "const"])
        if num_const > max_const:
            if DEBUG:
                print("Constrained max const: {} ({} consts)".format(ind, num_const))
            return True

        return False

    return check_num_const
