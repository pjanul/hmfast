import os
import re
import sys

sys.path.insert(0, os.path.abspath("../../src"))

project = "hmfast"
copyright = "2025, The hmfast developers"
author = "Patrick Janulewicz, Licong Xu, Boris Bolliet"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "numpydoc",
    "sphinx.ext.mathjax",
]

templates_path = ["_templates"]
exclude_patterns = ["_build"]

autosummary_generate = True
autosummary_imported_members = False

autodoc_member_order = "bysource"
autoclass_content = "class"

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "private-members": False,  # Ensures private methods are not shown
    "special-members": "",     # Do not include __init__ or other dunder methods
    "show-inheritance": True,
    "inherited-members": True,
}

numpydoc_show_class_members = False
numpydoc_class_members_toctree = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "jax": ("https://jax.readthedocs.io/en/latest", None),
}

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]


def _sci_notation(match):
    """Reformat one long-decimal float literal (e.g. '10000000000.0') as scientific
    notation (e.g. '1e10'), preserving its value exactly (no rounding-driven change)."""
    mantissa, exp = f"{float(match.group(0)):e}".split("e")
    return f"{mantissa.rstrip('0').rstrip('.')}e{int(exp)}"


def _shorten_large_float_defaults(app, what, name, obj, options, signature, return_annotation):
    """Rendered signatures show default floats via Python's repr, which spells out large
    round numbers (mass ranges, HOD masses, ...) digit by digit instead of using
    scientific notation; reformat any float literal with 5+ integer digits."""
    if signature:
        signature = re.sub(r"\d{5,}\.\d+", _sci_notation, signature)
    return signature, return_annotation


def setup(app):
    app.connect("autodoc-process-signature", _shorten_large_float_defaults)
