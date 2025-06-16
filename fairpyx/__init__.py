"""Wrapper enabling `import fairpyx` directly from a fresh clone.

The repository layout is:

fairpyx/       <- *this* directory (no real code originally)
    fairpyx/   <- actual python package with __init__.py, modules, algorithms …

When Python searches for the *top-level* package ``fairpyx`` it finds this
folder first, but because an ``__init__.py`` now exists we must *delegate* all
imports to the inner real package so that statements such as::

    from fairpyx import Instance

work transparently.  We do so by:
1. Importing ``fairpyx.fairpyx`` (the real package).
2. Re-registering it in ``sys.modules`` under the top-level key ``fairpyx``.
3. Mirroring its submodules so that ``import fairpyx.instances`` succeeds.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys

_inner_dir = pathlib.Path(__file__).parent / 'fairpyx'
_inner_init = _inner_dir / '__init__.py'

spec = importlib.util.spec_from_file_location(__name__, _inner_init, submodule_search_locations=[str(_inner_dir)])
module = importlib.util.module_from_spec(spec)
# Register early so that inner package imports succeed
sys.modules[__name__] = module
spec.loader.exec_module(module)  # type: ignore

# Expose names
globals().update({k: v for k, v in module.__dict__.items() if not k.startswith('_')})

# cleanup
del importlib, pathlib, sys, _inner_dir, _inner_init, spec, module

import importlib
import pathlib
import sys

# ---------------------------------------------------------------------------
# 1) Import the *real* inner package
# ---------------------------------------------------------------------------
_inner_pkg_name = __name__ + '.fairpyx'
_real_pkg = importlib.import_module(_inner_pkg_name)

# ---------------------------------------------------------------------------
# 2) Expose it as if it were the top-level package
# ---------------------------------------------------------------------------
sys.modules[__name__] = _real_pkg  # ``import fairpyx`` now returns the real pkg

# 3) Mirror submodules: ``fairpyx.instances`` -> actually ``fairpyx.fairpyx.instances``
_prefix_old = _inner_pkg_name  # e.g. 'fairpyx.fairpyx'
_prefix_new = __name__         # 'fairpyx'
for mod_name, mod in list(sys.modules.items()):
    if mod_name.startswith(_prefix_old):
        mirror_name = mod_name.replace(_prefix_old, _prefix_new, 1)
        sys.modules[mirror_name] = mod

# Finally, pollute *this* module's namespace with the inner package attributes so
# ``from fairpyx import Instance`` works.
globals().update({k: v for k, v in _real_pkg.__dict__.items() if not k.startswith('_')})

del importlib, pathlib, sys, _inner_pkg_name, _real_pkg, _prefix_old, _prefix_new
