import pytest

from agent_telephone.topologies import Edge, hier_chain, hub_relay, peer_route


def test_hier_chain():
    assert hier_chain(["lead", "pm", "arch", "impl"]) == [
        Edge("user", "lead", 1, "down", 1), Edge("lead", "pm", 2, "down", 2), Edge("pm", "arch", 3, "down", 3),
        Edge("arch", "impl", 4, "down", 4)]


def test_hub_relay():
    edges = hub_relay(["lead", "pm", "impl"])
    assert [(e.sender, e.receiver) for e in edges] == [("user", "lead"), ("lead", "pm"), ("pm", "lead"), ("lead", "impl")]
    assert [e.distance for e in edges] == [1, 2, 3, 4]
    # the lead writes to impl holding the brief (received at 1): information distance 2; pm -> lead is a hand-back
    assert [e.info_distance for e in edges] == [1, 2, 3, 2]
    assert [e.direction for e in edges] == ["down", "down", "up", "down"]


def test_peer_route():
    assert [e.distance for e in peer_route(["a", "b", "c"])] == [1, 2, 3]
    assert peer_route(["a", "b", "c"])[-1] == Edge("b", "c", 3, "down", 3)


def test_too_few_roles():
    with pytest.raises(ValueError):
        hub_relay(["lead", "pm"])


def test_endpoint_author_is_the_last_role():
    roles = ["lead", "pm", "arch", "impl"]
    for topo in (hier_chain, hub_relay, peer_route):
        assert topo(roles)[-1].receiver == "impl"
    assert [e.receiver for e in hub_relay(roles)] == ["lead", "pm", "lead", "impl"]
