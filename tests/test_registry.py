from __future__ import annotations

from pdeobs.registry import Registry


class FakeEntryPoint:
    name = "bundle"

    def load(self):
        def register():
            return {"first": object(), "second": object()}

        return register


class FakeEntryPoints(list):
    def select(self, **kwargs):
        assert kwargs == {"group": "pdeobs.test"}
        return self


def test_discovery_supports_registration_bundle_hooks(monkeypatch) -> None:
    monkeypatch.setattr(
        "pdeobs.registry.metadata.entry_points",
        lambda: FakeEntryPoints([FakeEntryPoint()]),
    )
    registry = Registry("test", entry_point_group="pdeobs.test")
    loaded = registry.discover(on_error="raise")

    assert loaded == ("first", "second")
    assert registry.names() == ("first", "second")
