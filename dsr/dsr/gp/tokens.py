import numpy as np
from dsr.program import Program,  _finish_tokens
from collections import defaultdict
from dsr.subroutines import jit_parents_siblings_at_once

try:
    from deap import gp
    from deap import base
    from deap import tools
    from deap import creator
    from deap import algorithms
except ImportError:
    gp          = None
    base        = None
    tools       = None
    creator     = None
    algorithms  = None

# Define the name of type for any types. This is a DEAP widget thingy. 
__type__ = object

try:
    CONST_TOKEN = Program.library.names.index("const")
except:
    CONST_TOKEN = None

r"""
    Fast special case version of below. This is mainly used during constraint 
    checking. 
"""
def opt_DEAP_to_math_tokens(individual):

    tokens = np.array([i.token for i in individual], dtype=np.int32)
    
    return tokens

r"""
    This is a base class for accessing DEAP and interfacing it with DSR. 
        
    These are pure symblic components which relate to any symblic task. These are not purely task agnostic and
    are kept seprate from core.
"""
def DEAP_to_math_tokens(individual, tokens_size):
        
    assert gp is not None, "Must import Deap GP library to use method. You may need to install it."
    assert isinstance(individual, gp.PrimitiveTree), "Program tokens should be a Deap GP PrimativeTree object."

    l                   = min(len(individual), tokens_size)
  
    tokens              = np.zeros(tokens_size, dtype=np.int32)
    arities             = np.empty(l)
    optimized_consts    = []
    
    if CONST_TOKEN is not None:
        for i in range(l): 
            ind         = individual[i]
            tokens[i]   = ind.token
            arities[i]  = ind.arity
            if ind.token == CONST_TOKEN: optimized_consts.append(ind.value) 
    else:   
        for i in range(l): tokens[i], arities[i] = individual[i].token, individual[i].arity
        
    dangling            = 1 + np.cumsum(arities - 1) 
    expr_length         = 1 + np.argmax(dangling == 0)
    
    '''
        Here we return the tokens as a list of indexable integers as well as a list of library token objects. 
        We primarily need to library token objects if we want to keep track of optimized mutable constants 
    '''
    return tokens, optimized_consts, expr_length


def math_tokens_to_DEAP(tokens, primitive_set):
    """
    Transforms DSR standard tokens into DEAP format tokens.

    DSR and DEAP format are very similar, but we need to translate it over. 

    Parameters
    ----------
    tokens : list of integers
        A list of integers corresponding to tokens in the library. The list
        defines an expression's pre-order traversal. "Dangling" programs are
        completed with repeated "x1" until the expression completes.

    primitive_set : gp.PrimitiveSet
        This should contain the list of primitives we will use. One way to create this is:
        
            # Create the primitive set
            pset = gp.PrimitiveSet("MAIN", dataset.X_train.shape[1])

            # Add input variables
            rename_kwargs = {"ARG{}".format(i) : "x{}".format(i + 1) for i in range(dataset.n_input_var)}
            pset.renameArguments(**rename_kwargs)

            # Add primitives
            for k, v in function_map.items():
                if k in dataset.function_set:
                    pset.addPrimitive(v.function, v.arity, name=v.name) 

    Returns
    _______
    individual : gp.PrimitiveTree
        This is a specialized list that contains points to element from primitive_set that were mapped based 
        on the translation of the tokens. 
    """
        
    assert gp is not None, "Must import Deap GP library to use method. You may need to install it."
    assert isinstance(tokens, np.ndarray), "Raw tokens are supplied as a numpy array."
    assert isinstance(primitive_set, PrimitiveSet), "You need to supply a valid primitive set for translation."
    assert Program.library is not None, "You have to have an initial program class to supply library token conversions."
    
    '''
        Truncate expressions that complete early; extend ones that don't complete
    '''
    tokens      = _finish_tokens(tokens)
    plist       = []      
    mc_count    = 0 
    
    for t in tokens:
        
        node = Program.library[t]

        if node.name == "const":
            '''
                NUMBER - Blank floating point constant. 
                    
                    Typically this is a constant parameter we want to optimize.
            '''
            try:
                # Optimizable consts are not tracked like other terminals in DSR.
                # We just need to make sure we keep them in order. Naming is arbitrary. 
                cname   = "mutable_const_{}".format(mc_count)
                p       = primitive_set.mapping[cname]
                if node.value is not None:
                    p.value = np.float(node.value)
                else:
                    p.value = np.float(1.0)
                plist.append(p)
                mc_count += 1
            except ValueError:
                print("ERROR: Cannot add mutable \"const\" from DEAP primitve set")
                
        elif node.arity == 0 and node.input_var is None:
            '''
                NUMBER - Library supplied floating point constant. 
                    
                    This is a constant the user sets and should not change. 
            '''
            try:
                # The DSR node name is stored in the string to make it easier to map back from DEAP
                # later. 
                p       = primitive_set.mapping["user_const_{}".format(node.name)]
                p.value = node.function()
                plist.append(p)
            except ValueError:
                print("ERROR: Cannot add user \"const\" from DEAP primitve set")
                
        elif node.input_var is not None:
            '''
                NUMBER - Values from input X at location given by value in node
                
                    This is usually the raw data point numerical values. Its value should not change. 
            '''
            try:
                # Here we use x{} rather than ARG{} since we renamed it by mapping. 
                plist.append(primitive_set.mapping[node.name])
            except ValueError:
                print("ERROR: Cannot add argument value \"x{}\" from DEAP primitve set".format(node))
                
        else:
            '''
                FUNCTION - Name should map from Program. Be sure to add all function map items into PrimativeSet before call. 
                
                    This is any common function with a name like "sin" or "log". 
                    We assume right now all functions work on floating points. 
            '''
            try:
                plist.append(primitive_set.mapping[node.name])
            except ValueError:
                print("ERROR: Cannot add function \"{}\" from DEAP primitve set".format(node.name))
            
    individual = gp.PrimitiveTree(plist)
    
    '''
        Look. You've got it all wrong. You don't need to follow me. 
        You don't need to follow anybody! You've got to think for yourselves. 
        You're all individuals! 
    '''
    return individual


def individual_to_dsr_aps(individual, library):
    r"""
        This will convert a deap individual to a DSR action, parent, sibling group.
    """ 
        
    # Get the action tokens from individuals 
    actions             = opt_DEAP_to_math_tokens(individual)
    # Add one dim at the front to be (1 x L)
    actions             = np.expand_dims(actions,axis=0) 
    # Get the parent/siblings for 
    parent, sibling     = jit_parents_siblings_at_once(actions, arities=library.arities, parent_adjust=library.parent_adjust)
    
    return actions, parent, sibling


class Primitive(object):
    """Class that encapsulates a primitive and when called with arguments it
    returns the Python code to call the primitive with the arguments.

        >>> pr = Primitive("mul", (int, int), int)
        >>> pr.format(1, 2)
        'mul(1, 2)'
    """
    __slots__ = ('name', 'arity', 'args', 'ret', 'seq', 'token')

    def __init__(self, name, args, ret, token):
        assert isinstance(token,int)
        self.name   = name
        self.arity  = len(args)
        self.args   = args
        self.ret    = ret
        args        = ", ".join(map("{{{0}}}".format, range(self.arity)))
        self.seq    = "{name}({args})".format(name=self.name, args=args)
        self.token  = token # DSR Token library number

    def format(self, *args):
        return self.seq.format(*args)

    def __eq__(self, other):
        if type(self) is type(other):
            return all(getattr(self, slot) == getattr(other, slot)
                       for slot in self.__slots__)
        else:
            return NotImplemented


class Terminal(object):
    """Class that encapsulates terminal primitive in expression. Terminals can
    be values or 0-arity functions.
    """
    __slots__ = ('name', 'value', 'ret', 'conv_fct', 'token')

    def __init__(self, terminal, symbolic, ret, token):
        assert isinstance(token,int)
        self.ret        = ret
        self.value      = terminal
        self.name       = str(terminal)
        self.conv_fct   = str if symbolic else repr
        self.token      = token # DSR Token library number

    @property
    def arity(self):
        return 0

    def format(self):
        return self.conv_fct(self.value)

    def __eq__(self, other):
        if type(self) is type(other):
            return all(getattr(self, slot) == getattr(other, slot)
                       for slot in self.__slots__)
        else:
            return NotImplemented


class Ephemeral(Terminal):
    """Class that encapsulates a terminal which value is set when the
    object is created. To mutate the value, a new object has to be
    generated. This is an abstract base class. When subclassing, a
    staticmethod 'func' must be defined.
    """

    def __init__(self):
        Terminal.__init__(self, self.func(), symbolic=False, ret=self.ret, token=CONST_TOKEN)

    @staticmethod
    def func():
        """Return a random value used to define the ephemeral state.
        """
        raise NotImplementedError


class PrimitiveSetTyped(object):
    """Class that contains the primitives that can be used to solve a
    Strongly Typed GP problem. The set also defined the researched
    function return type, and input arguments type and number.
    """

    def __init__(self, name, in_types, ret_type, prefix="ARG"):
              
        self.terminals = defaultdict(list)
        self.primitives = defaultdict(list)
        self.arguments = []
        # setting "__builtins__" to None avoid the context
        # being polluted by builtins function when evaluating
        # GP expression.
        self.context = {"__builtins__": None}
        self.mapping = dict()
        self.terms_count = 0
        self.prims_count = 0

        self.name = name
        self.ret = ret_type
        self.ins = in_types
    
        for i, type_ in enumerate(in_types):
            arg_str = "{prefix}{index}".format(prefix=prefix, index=i)
            self.arguments.append(arg_str)
            # Each variable token ID is just its number as the first n tokens
            term = Terminal(arg_str, True, ret=type_, token=i) 
            self._add(term)
            self.terms_count += 1

    def renameArguments(self, **kargs):
        """Rename function arguments with new names from *kargs*.
        """
        for i, old_name in enumerate(self.arguments):
            if old_name in kargs:
                new_name = kargs[old_name]
                self.arguments[i] = new_name
                self.mapping[new_name] = self.mapping[old_name]
                self.mapping[new_name].value = new_name
                del self.mapping[old_name]

    def _add(self, prim):
        def addType(dict_, ret_type):
            if ret_type not in dict_:
                new_list = []
                for type_, list_ in dict_.items():
                    if issubclass(type_, ret_type):
                        for item in list_:
                            if item not in new_list:
                                new_list.append(item)
                dict_[ret_type] = new_list

        addType(self.primitives, prim.ret)
        addType(self.terminals, prim.ret)

        self.mapping[prim.name] = prim
        if isinstance(prim, Primitive):
            for type_ in prim.args:
                addType(self.primitives, type_)
                addType(self.terminals, type_)
            dict_ = self.primitives
        else:
            dict_ = self.terminals

        for type_ in dict_:
            if issubclass(prim.ret, type_):
                dict_[type_].append(prim)

    def addPrimitive(self, primitive, in_types, ret_type, name):
        """Add a primitive to the set.

        :param primitive: callable object or a function.
        :param in_types: list of primitives arguments' type
        :param ret_type: type returned by the primitive.
        :param name: alternative name for the primitive instead
                     of its __name__ attribute.
        """        
        prim = Primitive(name, in_types, ret=ret_type, token=Program.library.names.index(name))

        assert name not in self.context or \
               self.context[name] is primitive, \
            "Primitives are required to have a unique name. " \
            "Consider using the argument 'name' to rename your " \
            "second '%s' primitive." % (name,)

        self._add(prim)
        self.context[prim.name] = primitive
        self.prims_count += 1

    def addTerminal(self, terminal, ret_type, name):
        """Add a terminal to the set. Terminals can be named
        using the optional *name* argument. This should be
        used : to define named constant (i.e.: pi); to speed the
        evaluation time when the object is long to build; when
        the object does not have a __repr__ functions that returns
        the code to build the object; when the object class is
        not a Python built-in.

        :param terminal: Object, or a function with no arguments.
        :param ret_type: Type of the terminal.
        :param name: defines the name of the terminal in the expression.
        """
        symbolic = False
        
        assert name not in self.context, \
            "Terminals are required to have a unique name. " \
            "Consider using the argument 'name' to rename your " \
            "second %s terminal." % (name,)

        if name is not None:
            self.context[name] = terminal
            terminal = name
            symbolic = True
        elif terminal in (True, False):
            # To support True and False terminals with Python 2.
            self.context[str(terminal)] = terminal

        if name.startswith("user_const_"):
            prim = Terminal(terminal, symbolic, ret=ret_type, token=Program.library.names.index(name.split('_')[2]))
        elif name.startswith("mutable_const_"):
            prim = Terminal(terminal, symbolic, ret=ret_type, token=CONST_TOKEN)
        else:
            # We don't support other types at the moment
            raise ValueError
        
        self._add(prim)
        self.terms_count += 1
        
    def addEphemeralConstant(self, name, ephemeral, ret_type):
        """Add an ephemeral constant to the set. An ephemeral constant
        is a no argument function that returns a random value. The value
        of the constant is constant for a Tree, but may differ from one
        Tree to another.

        :param name: name used to refers to this ephemeral type.
        :param ephemeral: function with no arguments returning a random value.
        :param ret_type: type of the object returned by *ephemeral*.
        """
        module_gp = globals()
        if name not in module_gp:
            class_ = type(name, (Ephemeral,), {'func': staticmethod(ephemeral),
                                               'ret': ret_type})
            module_gp[name] = class_
        else:
            class_ = module_gp[name]
            if issubclass(class_, Ephemeral):
                if class_.func is not ephemeral:
                    raise Exception("Ephemerals with different functions should "
                                    "be named differently, even between psets.")
                elif class_.ret is not ret_type:
                    raise Exception("Ephemerals with the same name and function "
                                    "should have the same type, even between psets.")
            else:
                raise Exception("Ephemerals should be named differently "
                                "than classes defined in the gp module.")

        self._add(class_)
        self.terms_count += 1
        
    @property
    def terminalRatio(self):
        """Return the ratio of the number of terminals on the number of all
        kind of primitives.
        """
        return self.terms_count / float(self.terms_count + self.prims_count)


class PrimitiveSet(PrimitiveSetTyped):
    """Class same as :class:`~deap.gp.PrimitiveSetTyped`, except there is no
    definition of type.
    """

    def __init__(self, name, arity, prefix="ARG"):
        args = [__type__] * arity
        PrimitiveSetTyped.__init__(self, name, args, ret_type=__type__, prefix=prefix)

    def addPrimitive(self, primitive, arity, name):
        """Add primitive *primitive* with arity *arity* to the set.
        If a name *name* is provided, it will replace the attribute __name__
        attribute to represent/identify the primitive.
        """
        assert arity > 0, "arity should be >= 1"
        args = [__type__] * arity
        PrimitiveSetTyped.addPrimitive(self, primitive, args, ret_type=__type__, name=name)

    def addTerminal(self, terminal, name):
        """Add a terminal to the set."""
        PrimitiveSetTyped.addTerminal(self, terminal, ret_type=__type__, name=name)

    def addEphemeralConstant(self, name, ephemeral):
        """Add an ephemeral constant to the set."""
        PrimitiveSetTyped.addEphemeralConstant(self, name, ephemeral, ret_type=__type__)

