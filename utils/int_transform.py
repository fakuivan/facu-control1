from sympy import Expr, Derivative, Function, Add, Mul, laplace_transform, And, Eq, Piecewise, re
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, TypeVar, Union, cast
from functools import partial
from more_itertools import partition

def traverse_linop(op: Callable[[Expr], Expr],
                   ccheck: Callable[[Expr], bool],
                   expr: Expr) -> Expr:
    """
    Traverses an expression tree applying op with the linear operator rules
    """
    rself: Callable[[Expr], Expr] = partial(traverse_linop, op, ccheck)
    # sub1 + sub2
    if expr.func == Add:
        return Add(*map(rself, expr.args))
    # const * sub1 * sub2
    if expr.func == Mul:
        variables, [*constants] = partition(ccheck, expr.args)
        if len(constants) > 0:
            return Mul(*constants, rself(Mul(*variables)))
    return op(expr)

def laplace_transform_f(fmap: Dict[Function, Function],
                        timevar,
                        svar,
                        expr: Expr,
                        czero=True) -> Expr:
    """
    Transforms a function expression in time space to laplace space with
    support for derivatives, can be composed with traverse_linop to apply
    the transform to inner expressions
    """
    L = lambda expr: laplace_transform(expr, timevar, svar)
    if expr.func == Derivative:
        f_expr, (var, order) = expr.args
        # for now only match univariate functions 
        f, (f_var,) = f_expr.func, f_expr.args
        if var == timevar == f_var and f in fmap:
            return laplace_transform_diff(
                f, fmap[f], timevar, svar, order, czero=czero)
    if expr.func in fmap:
        return fmap[expr.func]
    return L(expr)

def laplace_transform_diff(f: Function,
                           F: Function,
                           t: Expr,
                           s: Expr,
                           order: int,
                           czero=True) -> Expr:
    """
    Maps the derivative of a function f(t) with laplace transform F(s)
    of any positive or zero order, optionally with intial conditions set
    to zero if czero == True.
    """
    head = F(s)*s**order
    diffs = [Derivative(f(t), t, n).subs(t, 0) for n in range(order)]
    # TODO: Verify ROC, for now assume 0.
    return (head, 0, And(*[Eq(diff, 0) for diff in diffs])) if czero else \
           (head - Add(*[s**n*diff for n, diff in enumerate(reversed(diffs))]), 0, True)


T = TypeVar("T")
S = TypeVar("S")
def floating_reducer(reducer: Callable[[T, S], T], initial: T) -> \
        tuple[Callable[[S], T], Callable[[], T]]:
    """
    Creates a pair of impure functions that close over a floating
    accumulator. ``reducer`` is used by the updater to update this
    state.
    """
    accum = initial
    def update(value: S) -> T:
        nonlocal accum
        accum = reducer(accum, value)
        return accum
    
    # Should we invalidate ``update`` after reading the accumulator?
    def get() -> T:
        nonlocal accum
        return accum

    return update, get

def laplace_transform_extended(expr: Expr, t: Expr, s: Expr, 
                               fmap: Dict[Function, Function],
                               czero=True, noconds=False):
    """
    Laplace transform extended to handle function symbols and their
    derivatives.
    """
    update_roc, get_roc = floating_reducer(max, 0)
    append_cond, get_cond = floating_reducer(And, True)

    def L(expr):
        result = laplace_transform_f(fmap, t, s, expr, czero=czero)
        if not isinstance(result, tuple):
            return result
        transform, roc, cond = result
        append_cond(cond)
        update_roc(roc)
        return transform

    result = traverse_linop(L, lambda expr: expr.is_constant(t), expr)
    return result if noconds else (result, get_roc(), get_cond())

def lt_as_piecewise(result: Union[Tuple[Expr, Expr, Expr], Expr], s: Expr):
    """
    Takes the result of a Laplace transform and converts the tuple output
    to a piecewise function where ROC and other conditions have to be met
    """
    if not isinstance(result, tuple):
        return result
    transform, roc, cond = result
    return Piecewise((transform, And(re(s) > roc, cond)))

T=TypeVar("T")
def list_from_iv_pairs(pairs: Iterable[Tuple[int, T]]) -> List[T]:
    """
    Constructs a list from (index, value) pairs,
    errors in the case the list would contain "holes"
    """
    my_list: List[T] = []
    last_index = -1
    for i, v in sorted(pairs, key=lambda pair: pair[0]):
        if i-last_index != 1:
            raise KeyError(
                f"List indexes must be contiguous, "
                f"expected {last_index+1}, got {i}")
        my_list.append(v)
        last_index = i
    return my_list

def permute_args(map: Dict[Union[str, int], Union[str, int]],
                 args: Tuple[Any, ...],
                 kwargs: Dict[str, Any]):
    """
    Permutes a list of arguments and a dictionary of kwargs matching
    the mapping in ``map``.
    A mapping of ``{0: 1, 1: 0, 2: "my_flag", "moved_flag": 3}`` maps:
        func("first", "second", "third", moved_flag="flag")
    to:
        func("second", "first", "flag", my_flag="third")
    """
    merged_args = {**{k: v for k, v in enumerate(args)} , **kwargs}
    # partitioning by args that map to positionals or kwargs
    # (typed python is ass)
    kwarg_map, args_map = cast(
        Tuple[Iterable[Tuple[Union[int, str], str]],
                Iterable[Tuple[Union[int, str], int]]],
        partition(lambda pair: isinstance(pair[1], int), map.items()))
    mapped_kwargs = {v: merged_args[k] for k, v in kwarg_map if k in merged_args}
    mapped_args = list_from_iv_pairs(
        (v, merged_args[k]) for k, v in args_map if k in merged_args)
    return mapped_args, mapped_kwargs

def compose(outer,
            if_flag: Optional[Tuple[str, bool]]=None,
            connect_args: Dict[Union[str, int], Union[str, int]] = {}):
    """
    Funny function to compose functions while allowing the outer function
    to access arguments of the inner function. if_flag allows the result
    of the inner call to be returned without being composed, this essentially
    allows the outer function to be an optional transformation on the intermediate
    value.
    This can be used to decorate general functions, adding extra functionality
    like with our extended laplace transform and the function that processes
    roc and other constaints into a piecewise function. This:

        @compose(lt_as_piecewise, ("as_piecewise", False), {3: 1})
        def laplace_transform_f(...): ...
    
    Would be equivalent to:

        def laplace_transform_f_extra(*args, **kwargs):
            if not kwargs.get("as_piecewise", False):
                return laplace_transform_f(*args, **kwargs)
            s = args[3]
            return lt_as_piecewise(laplace_transform_f(*args, **kwargs), s)

    Since the argument order for inner is preserved, nested compossition is
    possible. Changes to the return value propagate "upwards" trough decorators

    Sadly since python's type hinting system is not expressive enough to even
    corrctly type a simple function composition with inner taking more than
    one argument, this gives Pyright and mypy a stroke, hence why I might not
    use it.
    """
    flag_name, default = if_flag if if_flag is not None else (None, None)
    def build_composed(inner):
        def composed(*args, **kwargs):
            intermediate = inner(*args, **kwargs)
            if flag_name is not None and kwargs.get(flag_name, default):
                margs, mkwargs = permute_args(connect_args, args, kwargs)
                return outer(intermediate, *margs, **mkwargs)
            return intermediate
        return composed
    return build_composed

