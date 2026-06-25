"""
The Capability Registry must be built FROM the dispatcher's own tool descriptors,
so what the system advertises can never drift from what it can actually run.
"""

from __future__ import annotations

from app.core.tools.dispatcher import ToolDispatcher
from app.core.registry.builtin import build_registry_from_dispatcher


def _registry(tmp_path):
    return build_registry_from_dispatcher(ToolDispatcher(workspace_dir=str(tmp_path)))


def test_builtin_capabilities_are_discoverable(tmp_path):
    reg = _registry(tmp_path)
    assert reg.has_capability("read the contents of a file")
    assert reg.has_capability("show the git diff")
    assert reg.has_capability("fetch a web page over http")


def test_every_entry_is_well_formed(tmp_path):
    reg = _registry(tmp_path)
    assert reg.all()
    for e in reg.all():
        assert e.id and e.name and e.description
        assert e.side_effects in {"none", "read", "write", "network"}
        assert isinstance(e.params_schema, dict)


def test_registry_cannot_drift_from_dispatcher(tmp_path):
    # Every registered capability must belong to a tool the dispatcher exposes.
    disp = ToolDispatcher(workspace_dir=str(tmp_path))
    reg = build_registry_from_dispatcher(disp)
    roots = {e.id.split(".")[0] for e in reg.all()}
    assert roots <= set(disp.available_tools)


def test_write_capability_flagged_as_write_side_effect(tmp_path):
    reg = _registry(tmp_path)
    write_caps = [e for e in reg.all() if e.side_effects == "write"]
    # file write is a mutating capability and must be marked as such (governance).
    assert any("write" in e.id for e in write_caps)
