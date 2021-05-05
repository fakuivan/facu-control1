from typing import Callable, Dict, Iterable, Iterator, NamedTuple, Tuple, TypeVar
from more_itertools import split_before
from itertools import chain, accumulate
from functools import partial
ichain = chain.from_iterable

class SVGPoint(NamedTuple):
    x: float
    y: float

    def __add__(self, other: 'SVGPoint') -> 'SVGPoint':
        return SVGPoint(self.x + other.x, self.y + other.y)

parsers: Dict[str, Callable[[SVGPoint, str], SVGPoint]] = {
    'l': lambda _, values: SVGPoint(*parse_float_pair(values)),
    'h': lambda cursor, values: SVGPoint(float(values), cursor.y),
    'v': lambda cursor, values: SVGPoint(cursor.x, float(values))
}
# m should be handled differently, for now this will suffice
parsers['m'] = parsers['l']

def parse_float_pair(text: str) -> Tuple[float, float]:
    x, y = map(float, text.split(','))
    return x, y

T = TypeVar('T')
def skip_iter(iterable: Iterable[T]) -> Iterator[T]:
    iterable = iter(iterable)
    next(iterable)
    return iterable

def parse_path_command(name: str, *args: str, initial = SVGPoint(0, 0)) -> Iterator[SVGPoint]:
    if parser := parsers.get(name, None):
        accum = lambda points: skip_iter(accumulate(points, initial=initial))
        return accum(map(partial(parser, SVGPoint(0, 0)), args))
    
    if parser := parsers.get(name.lower(), None):
        return skip_iter(accumulate(args, parser, initial=initial))

    raise ValueError(f'Invalid path command {name!r}')

def is_path_command(cmd: str):
    return cmd.lower() in parsers
    
def parse_svg_path_commands(expr: str, initial = SVGPoint(0, 0)) -> Iterator[SVGPoint]:
    for name, *args in split_before(expr.split(" "), is_path_command):
        for point in parse_path_command(name, *args, initial=initial):
            yield (initial := point)
