# Unsloth Notebooks - Notebooks for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

"""pytest conftest: GPU-free harness so `import unsloth` survives on a
CPU-only CI runner.

Mirrors the unslothai/unsloth-zoo conftest pattern (PR #639, #642):
``unsloth_zoo.device_type.get_device_type()`` raises
``NotImplementedError`` on a runner with no CUDA / XPU / HIP visible,
which in turn fails ``import unsloth_zoo`` (and therefore
``import unsloth``). Pre-load ``unsloth_zoo.device_type`` under a
temporarily-True ``torch.cuda.is_available()`` so the ``@cache``
permanently captures ``"cuda"`` and the import chain finishes; tests
that touch real tensors run on CPU as normal.

Once the harness is in place we eagerly ``import unsloth`` so every
``fix_*`` / ``patch_*`` in ``unsloth/import_fixes.py`` activates BEFORE
pytest collects the API drift tests. Without this step
``test_notebook_api_drift.py`` would probe an unpatched runtime state
and false-fail on entirely-healthy symbol setups (the exact gap fixed
in unslothai/unsloth#5421).
"""
from __future__ import annotations

import importlib.util
import os
import sys
import types


def _has_real_accelerator() -> bool:
    try:
        import torch
    except Exception:
        return False
    for probe in (
        lambda: hasattr(torch, "cuda") and torch.cuda.is_available(),
        lambda: hasattr(torch, "xpu") and torch.xpu.is_available(),
        lambda: hasattr(torch, "accelerator") and torch.accelerator.is_available(),
    ):
        try:
            if probe():
                return True
        except Exception:
            pass
    return False


def _preload_device_type(package: str, prereqs: tuple[str, ...] = ()) -> bool:
    """Pre-load <package>.device_type under a mocked
    torch.cuda.is_available() == True so its @cache permanently
    captures "cuda". prereqs lists submodule names of <package> that
    must be loaded first (e.g. 'utils' for unsloth_zoo)."""
    target = f"{package}.device_type"
    if target in sys.modules:
        return True
    pkg_spec = importlib.util.find_spec(package)
    if pkg_spec is None or not pkg_spec.submodule_search_locations:
        return False
    pkg_path = pkg_spec.submodule_search_locations[0]

    skeleton_already = package in sys.modules
    if not skeleton_already:
        skel = types.ModuleType(package)
        skel.__path__ = [pkg_path]
        skel.__spec__ = pkg_spec
        skel.__package__ = package
        sys.modules[package] = skel

    try:
        for prereq in prereqs:
            full = f"{package}.{prereq}"
            if full in sys.modules:
                continue
            prereq_path = os.path.join(pkg_path, f"{prereq}.py")
            prereq_spec = importlib.util.spec_from_file_location(full, prereq_path)
            prereq_mod = importlib.util.module_from_spec(prereq_spec)
            sys.modules[full] = prereq_mod
            prereq_spec.loader.exec_module(prereq_mod)

        device_type_path = os.path.join(pkg_path, "device_type.py")
        dt_spec = importlib.util.spec_from_file_location(target, device_type_path)
        dt_mod = importlib.util.module_from_spec(dt_spec)
        sys.modules[target] = dt_mod

        import torch

        _orig_is_avail = torch.cuda.is_available
        torch.cuda.is_available = lambda: True  # type: ignore[assignment]
        try:
            dt_spec.loader.exec_module(dt_mod)
        finally:
            torch.cuda.is_available = _orig_is_avail
    except Exception:
        sys.modules.pop(target, None)
        return False
    finally:
        if not skeleton_already:
            sys.modules.pop(package, None)

    return True


def _patch_torch_cuda_for_import() -> None:
    """Stub torch.cuda.* probes that fire at IMPORT time of unsloth /
    unsloth_zoo when DEVICE_TYPE was forced to "cuda" above. These are
    queries, not real GPU work, so returning plausible Ampere values
    lets the import chain finish; tests run on CPU as normal."""
    try:
        import torch.cuda.memory as _cuda_memory  # type: ignore

        _cuda_memory.mem_get_info = lambda *a, **k: (0, 80 * 1024**3)
    except Exception:
        pass
    try:
        import torch

        torch.cuda.get_device_capability = lambda *a, **k: (8, 0)
        torch.cuda.is_bf16_supported = lambda *a, **k: True
    except Exception:
        pass


def _install_device_type_stub(name: str) -> None:
    stub = types.ModuleType(name)
    stub.DEVICE_TYPE = "cuda"
    stub.DEVICE_TYPE_TORCH = "cuda"
    stub.DEVICE_COUNT = 1
    stub.ALLOW_PREQUANTIZED_MODELS = False
    stub.is_hip = lambda: False
    stub.get_device_type = lambda: "cuda"
    stub.get_device_count = lambda: 1
    stub.device_synchronize = lambda *a, **k: None
    stub.device_empty_cache = lambda *a, **k: None
    stub.device_is_bf16_supported = lambda *a, **k: False
    sys.modules[name] = stub


if not _has_real_accelerator():
    if not _preload_device_type("unsloth_zoo", prereqs=("utils",)):
        _install_device_type_stub("unsloth_zoo.device_type")
    if not _preload_device_type("unsloth"):
        _install_device_type_stub("unsloth.device_type")
    _patch_torch_cuda_for_import()


# Apply ALL upstream-drift fixes (vllm GuidedDecodingParams alias, triton
# CompiledKernel attr wrap, peft transformers_weight_conversion stub,
# etc.) by triggering `import unsloth`. Mirrors unsloth's own
# tests/conftest.py pattern after PR #5421.
try:
    import unsloth  # noqa: F401  # runs unsloth/import_fixes.py
except Exception:
    # unsloth not installed (security-only suites) or import failed; drift
    # detectors will surface any pathology the missing patches would have
    # hidden.
    pass
