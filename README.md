This is a special matrix-free version of SCS. See [these papers](http://stanford.edu/~boyd/papers/abs_ops.html) for further details on matrix-free solvers.

### Using matrix-free scs in Python

To create the Python interface, the following lines of code should work:
```
cd <scs-directory>/python
python setup.py install
```
You may need `sudo` privileges for a global installation. 

After installing the scs interface, you must import the module with
```
import mat_free_scs
```

Matrix-free SCS can only be called via CVXPY.
