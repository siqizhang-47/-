"""NumPy 2.0 compatibility shim.

`torch_timeseries` (and some other deps) still use aliases that NumPy 2.0
removed, e.g. `np.Inf`, `np.NaN`, `np.float_`. Importing this module restores
them so the framework runs on NumPy >= 2.0 without editing the installed
package. Import it *first*, before any framework import.
"""
import numpy as np

_ALIASES = {
    "Inf": "inf", "Infinity": "inf", "NINF": "inf",
    "NaN": "nan", "NAN": "nan",
    "PINF": "inf", "infty": "inf",
}
for _old, _new in _ALIASES.items():
    if not hasattr(np, _old) and hasattr(np, _new):
        val = getattr(np, _new)
        setattr(np, _old, -val if _old == "NINF" else val)

# removed scalar type aliases -> map to builtins / numpy dtypes
for _old, _val in {
    "float_": np.float64, "int_": np.int64, "complex_": np.complex128,
    "bool8": np.bool_, "object_": object, "str_": np.str_,
}.items():
    if not hasattr(np, _old):
        setattr(np, _old, _val)
