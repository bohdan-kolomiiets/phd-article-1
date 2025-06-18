from typing import Iterable, TypeVar, Callable
from collections import defaultdict

T = TypeVar("T")
K = TypeVar("K")

def group_by(items: Iterable[T], key_selector: Callable[[T], K]) -> dict[K, list[T]]:
    groups = defaultdict(list)
    for item in items:
        key = key_selector(item)
        groups[key].append(item)
    return dict(groups)