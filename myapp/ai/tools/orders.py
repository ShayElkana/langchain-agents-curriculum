"""Tools are thin adapters over the service layer — never raw DB access."""

from langchain.tools import tool

from myapp.services import orders


@tool
def get_order_status(order_id: str) -> str:
    """Look up the current status of a customer order by its ID."""
    try:
        order = orders.get_order(order_id)
    except orders.OrderNotFound:
        return (
            f"No order found with ID {order_id!r}. "
            "Ask the customer to re-check the ID (letter + three digits)."
        )
    return f"Order {order.id}: {order.status}, ETA {order.eta:%Y-%m-%d}, total ${order.total:.2f}"
