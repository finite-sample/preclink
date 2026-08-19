"""Sphinx configuration — fleet standard via py-canon."""

from py_canon.sphinx import configure

# preclink's public API is typed in terms of pandas/numpy objects, so the
# fleet's python-only intersphinx map is not enough to resolve them. Overrides
# replace rather than merge, so python has to be repeated here.
configure(
    globals(),
    intersphinx_mapping={
        "python": ("https://docs.python.org/3", None),
        "pandas": ("https://pandas.pydata.org/docs/", None),
        "numpy": ("https://numpy.org/doc/stable/", None),
    },
)
