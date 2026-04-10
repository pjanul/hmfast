import os
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
]

templates_path = ["_templates"]
exclude_patterns = ["_build"]

autosummary_generate = True
autosummary_imported_members = False

autodoc_member_order = "bysource"
autoclass_content = "both"

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "inherited-members": False,
    "exclude-members": "tree_flatten,tree_unflatten",
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

# --- Hide tree_flatten and tree_unflatten everywhere ---
def skip_tree_methods(app, what, name, obj, skip, options):
    if name in ("tree_flatten", "tree_unflatten"):
        return True
    return skip

def setup(app):
    app.connect("autodoc-skip-member", skip_tree_methods)