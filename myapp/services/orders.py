"""Toy order service — existing business logic. No LangChain imports."""

from dataclasses import dataclass
from datetime import date, timedelta


class OrderNotFound(LookupError):
    pass


@dataclass(frozen=True)
class Order:
    id: str
    status: str
    eta: date
    total: float


_DB = {
    "A123": Order("A123", "shipped", date.today() + timedelta(days=2), 49.99),
    "B456": Order("B456", "processing", date.today() + timedelta(days=5), 19.00),
}


def get_order(order_id: str) -> Order:
    order = _DB.get(order_id)
    if order is None:
        raise OrderNotFound(order_id)
    return order
