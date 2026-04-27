"""NumPy compatibility helpers for third-party extensions.

Some pycocotools builds still reference ``numpy.NPY_OWNDATA`` which was
removed from NumPy's public Python API in newer releases.
"""

import numpy as np


# Kept for compatibility with extensions expecting this constant.
# Equivalent to NPY_ARRAY_OWNDATA in NumPy C-API headers.
if not hasattr(np, "NPY_OWNDATA"):
    np.NPY_OWNDATA = 0x0004
