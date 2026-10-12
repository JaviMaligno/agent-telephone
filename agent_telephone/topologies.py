"""Topologies as explicit edge sequences over role ids.

`distance` counts edges from the original message: user -> lead is distance 1 (nominal).
`info_distance` = 1 + the smallest distance at which the writer had received source-derived
content: in hub_relay the lead still holds the brief when it writes to r2, so that message is
nominally at distance 4 but at information distance 2. `direction` is "down" for delegation
towards the endpoint and "up" for a hand-back to a caller (losses there do not leave the system).
The receiver of the last edge produces the endpoint artifact; it is always the last role
(the role that owns the deliverable, e.g. the implementer), whatever the topology.
"""
from __future__ import annotations

from typing import NamedTuple

USER = "user"


class Edge(NamedTuple):
    sender: str
    receiver: str
    distance: int
    direction: str = "down"
    info_distance: int = 0


def _annotate(hops: list[tuple[str, str, str]]) -> list[Edge]:
    """(sender, receiver, direction) in order -> edges with nominal and information distance."""
    last, closest, out = {USER: 0}, {USER: 0}, []
    for a, b, direction in hops:
        d, info = last[a] + 1, closest[a] + 1
        last[b], closest[b] = d, min(closest.get(b, info), info)
        out.append(Edge(a, b, d, direction, info))
    return out


def min_info_distance(edges: list[Edge], node: str) -> int:
    """Information distance of whatever `node` writes next (e.g. the endpoint artifact)."""
    return min(e.info_distance for e in edges if e.receiver == node) + 1


def _need(roles: list[str], n: int, name: str) -> None:
    if len(roles) < n:
        raise ValueError(f"{name} needs at least {n} roles, got {len(roles)}")


def hier_chain(roles: list[str]) -> list[Edge]:
    """user -> lead -> r1 -> r2 -> r3 (nested delegation over all roles, in order)."""
    _need(roles, 2, "hier_chain")
    path = [USER, *roles]
    return _annotate([(a, b, "down") for a, b in zip(path, path[1:])])


def hub_relay(roles: list[str]) -> list[Edge]:
    """user -> lead -> r1 -> lead -> r2 (the lead rewrites r1's output for r2); r2 is the last role."""
    _need(roles, 3, "hub_relay")
    lead, r1, r2 = roles[0], roles[1], roles[-1]
    return _annotate([(USER, lead, "down"), (lead, r1, "down"), (r1, lead, "up"), (lead, r2, "down")])


def peer_route(roles: list[str]) -> list[Edge]:
    """user -> A -> B -> C, where B is not the owner and forwards to C (the last role)."""
    _need(roles, 3, "peer_route")
    a, b, c = roles[0], roles[1], roles[-1]
    return _annotate([(USER, a, "down"), (a, b, "down"), (b, c, "down")])


TOPOLOGIES = {"hier_chain": hier_chain, "hub_relay": hub_relay, "peer_route": peer_route}


def build(name: str, roles: list[str]) -> list[Edge]:
    return TOPOLOGIES[name](roles)
