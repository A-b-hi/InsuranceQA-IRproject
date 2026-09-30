"""Test helpers."""
import types

import config


def make_cfg(**overrides):
    """Copy of config with some values replaced (e.g. empty directories for missing-resource tests)."""
    values = {k: getattr(config, k) for k in dir(config) if k.isupper()}
    values.update(overrides)
    return types.SimpleNamespace(**values)
