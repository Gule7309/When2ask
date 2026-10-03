from __future__ import annotations

from dataclasses import dataclass
from math import inf
from typing import Any, Mapping


@dataclass(frozen=True)
class DomainSpec:
    """Minimal argument-domain representation used by the SAGE reconstruction."""

    kind: str
    values: Any = None
    data_dependent: bool = False

    @property
    def size(self) -> float:
        if self.kind == "finite":
            if self.values is None:
                raise ValueError("finite domain requires values")
            return float(len(self.values))
        if self.kind == "boolean":
            return 2.0
        if self.kind == "numeric_range":
            if self.values is None or len(self.values) != 2:
                raise ValueError("numeric_range requires [start, end]")
            start, end = self.values
            return float(max(1, end - start + 1))
        return inf

    @classmethod
    def from_upstream_dict(cls, data: Mapping[str, Any]) -> "DomainSpec":
        kind = str(data.get("type", "string"))
        values = data.get("values")
        if values is None and "range" in data:
            values = data["range"]
        return cls(
            kind=kind,
            values=values,
            data_dependent=bool(data.get("data_dependent", False)),
        )


@dataclass(frozen=True)
class ArgumentSpec:
    name: str
    domain: DomainSpec
    required: bool = True
    description: str = ""

    @classmethod
    def from_upstream_dict(cls, data: Mapping[str, Any]) -> "ArgumentSpec":
        return cls(
            name=str(data["name"]),
            domain=DomainSpec.from_upstream_dict(data.get("domain", {})),
            required=bool(data.get("required", True)),
            description=str(data.get("description", "")),
        )


@dataclass(frozen=True)
class ToolSpec:
    name: str
    arguments: tuple[ArgumentSpec, ...]
    description: str = ""

    @property
    def argument_map(self) -> dict[str, ArgumentSpec]:
        return {arg.name: arg for arg in self.arguments}

    @classmethod
    def from_upstream_dict(cls, data: Mapping[str, Any]) -> "ToolSpec":
        return cls(
            name=str(data["name"]),
            description=str(data.get("description", "")),
            arguments=tuple(
                ArgumentSpec.from_upstream_dict(arg)
                for arg in data.get("arguments", [])
            ),
        )


def load_tool_specs(tool_dicts: list[Mapping[str, Any]]) -> dict[str, ToolSpec]:
    return {tool.name: tool for tool in map(ToolSpec.from_upstream_dict, tool_dicts)}
