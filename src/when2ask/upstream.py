from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping

from .schema import ToolSpec, load_tool_specs

PRIMARY_API_TO_PLUGIN = {
    "GorillaFileSystem": "gfs",
    "TravelAPI": "travel",
    "TradingBot": "trading",
    "VehicleControlAPI": "vehicle",
    "DocumentPlugin": "document",
}


def load_tool_specs_for_sample(
    upstream_root: str | Path,
    sample: Mapping[str, Any],
) -> dict[str, ToolSpec]:
    """Load the public ClarifyBench tool registry without creating an LLM provider.

    The upstream simulation updates data-dependent argument domains from the current
    sample context before agent execution. We mirror that behavior here because
    SAGE's viability score depends directly on domain cardinality.
    """
    root = Path(upstream_root).resolve()
    primary_api = sample.get("primary_api")
    if primary_api not in PRIMARY_API_TO_PLUGIN:
        raise ValueError(f"unsupported or missing primary_api: {primary_api!r}")

    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    from core.plugin_manager import PluginManager
    from core.tool_registry import ToolRegistry

    plugin_name = PRIMARY_API_TO_PLUGIN[str(primary_api)]
    manager = PluginManager(plugin_config_dir=str(root / "config" / "plugins"))
    if not manager.load_plugin(plugin_name):
        raise RuntimeError(
            f"failed to load upstream plugin {plugin_name!r} from {root}"
        )

    plugin = manager.get_plugin(plugin_name)
    if plugin is None:
        raise RuntimeError(f"upstream plugin {plugin_name!r} was not registered")

    if "initial_config" in sample and hasattr(plugin, "initialize_from_config"):
        plugin.initialize_from_config(sample["initial_config"])

    registry = ToolRegistry(manager)

    context = dict(sample)
    if "initial_config" in sample:
        context["initial_config"] = sample["initial_config"]
    registry.update_domain_from_data(context)

    tool_dicts = [tool.to_dict() for tool in registry.get_all_tools()]
    return load_tool_specs(tool_dicts)
