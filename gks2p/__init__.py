__all__ = ["preprocess", "mkops", "datasets", "mkops_general", "suite2p_temporal_smoothing"]

# Lazy import helper: import heavy submodules only when accessed as attributes on
# the package, e.g. `gks2p.preprocess` or `from gks2p import preprocess`.
import importlib
from types import ModuleType

_LAZY_SUBMODULES = {
	"preprocess": "gks2p.preprocess",
	"mkops": "gks2p.mkops",
	"datasets": "gks2p.datasets",
	"mkops_general": "gks2p.mkops_general",
	"suite2p_temporal_smoothing": "gks2p.suite2p_temporal_smoothing",
}


def __getattr__(name: str) -> ModuleType:
	"""Lazy-load a submodule when accessed as an attribute on the package.

	Example: `from gks2p import preprocess` will trigger importlib to import
	`gks2p.preprocess` the first time `preprocess` is accessed.
	"""
	if name in _LAZY_SUBMODULES:
		module = importlib.import_module(_LAZY_SUBMODULES[name])
		globals()[name] = module
		return module
	raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
	# Expose lazy names in dir(gks2p)
	return sorted(list(globals().keys()) + list(_LAZY_SUBMODULES.keys()))
