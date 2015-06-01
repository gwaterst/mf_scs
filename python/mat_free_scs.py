#!/usr/bin/env python
import _mat_free_scs_direct
import _mat_free_scs_indirect
from warnings import warn
from scipy import sparse


def solve(probdata, cone, **opts):
    """
    solves convex cone problems

    @return dictionary with solution with keys:
         'x' - primal solution
         's' - primal slack solution
         'y' - dual solution
         'info' - information dictionary
    """
    if not 'A' in probdata or not 'b' in probdata or not 'c' in probdata:
        raise TypeError("Missing one or more of A, b, c from data dictionary")
    A = probdata['A']
    b = probdata['b']
    c = probdata['c']

    warm = {}
    if 'x' in probdata:
        warm['x'] = probdata['x']
    if 'y' in probdata:
        warm['y'] = probdata['y']
    if 's' in probdata:
        warm['s'] = probdata['s']

    if A is None or b is None or c is None:
        raise TypeError("Incomplete data specification")
    if not sparse.issparse(A):
        raise TypeError("A is required to be a sparse matrix")
    if not sparse.isspmatrix_csc(A):
        warn("Converting A to a CSC (compressed sparse column) matrix; may take a while.")
        A = A.tocsc()

    if sparse.issparse(b):
        b = b.toDense()

    if sparse.issparse(c):
        c = c.toDense()

    # Set m and n based on b and c rather than A.
    m = b.size
    n = c.size

    Adata, Aindices, Acolptr = A.data, A.indices, A.indptr
    # Convert keys to upper case.
    new_opts = {}
    for key, val in opts.items():
        new_opts[key.upper()] = opts[key]
    opts = new_opts
    if opts.get("USE_INDIRECT", False):
        # HACK transfer from probdata to opts.
        for key in ["Amul", "ATmul", "getDE", "getM"]:
            if key in probdata:
                opts[key] = probdata[key]
        return _mat_free_scs_indirect.csolve((m, n), Adata, Aindices, Acolptr, b, c, cone, opts, warm)
    else:
        return _mat_free_scs_direct.csolve((m, n), Adata, Aindices, Acolptr, b, c, cone, opts, warm)
