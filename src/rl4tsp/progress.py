from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import TypeVar

T = TypeVar("T")


def iter_progress(
    iterable: Iterable[T],
    *,
    enabled: bool = False,
    total: int | None = None,
    desc: str | None = None,
    unit: str = "it",
) -> Iterable[T]:
    if not enabled:
        return iterable
    try:
        from tqdm import tqdm
    except Exception:
        return iterable
    return tqdm(iterable, total=total, desc=desc, unit=unit, dynamic_ncols=True)


def exhaust(iterator: Iterable[T]) -> Iterator[T]:
    yield from iterator
