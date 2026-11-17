"""A tiny Python calculator used to test how KíLín reads code.

This file is a test fixture; it is read by KíLín, never imported.

"""


def py_function_B(x):
    """Double a number.

    Parameters
    ----------
    x: float
        Number to double.

    Returns
    -------
    float
        Twice ``x``.

    """

    return 2 * x


def py_function_A(a, b):
    """Double a, then add b.

    Parameters
    ----------
    a: float
        Number that is doubled.
    b: float
        Number that is added.

    Returns
    -------
    float
        ``2 * a + b``.

    """

    return py_function_B(a) + b


class PyBaseClass:
    """Store one value.

    Parameters
    ----------
    value: float
        Stored value.

    """

    def __init__(self, value):
        self.value = value


class PyChildClass(PyBaseClass):
    """Add numbers to the stored value, doubled."""

    def py_method_C(self, b):
        """Return ``py_function_A(self.value, b)``.

        Parameters
        ----------
        b: float
            Number that is added.

        Returns
        -------
        float
            Twice the stored value plus ``b``.

        """

        return py_function_A(self.value, b)
