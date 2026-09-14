"""Flat config loader. Reads a YAML file into a single namespace and
applies dotted-key CLI overrides (e.g. `optim.lr_g=0.0001`)."""

import argparse
from dataclasses import dataclass, field

import yaml


@dataclass
class Config:
    values: dict = field(default_factory=dict)

    def __getattr__(self, name):
        try:
            return self.values[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __getitem__(self, name):
        return self.values[name]

    def get(self, name, default=None):
        return self.values.get(name, default)


def _flatten(prefix: str, node: dict, out: dict) -> None:
    for key, value in node.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            _flatten(full_key, value, out)
        else:
            out[full_key] = value


def load_config(path: str, overrides: list[str] | None = None) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    flat: dict = {}
    _flatten("", raw, flat)

    for override in overrides or []:
        key, _, value = override.partition("=")
        flat[key] = yaml.safe_load(value)

    return Config(values=flat)


def build_arg_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default="configs/default.yaml", help="Path to a YAML config file")
    parser.add_argument("--set", nargs="*", default=[], metavar="key=value",
                         help="Dotted-key overrides, e.g. optim.lr_g=0.0001")
    return parser
