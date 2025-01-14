# -*- coding: utf-8 -*-
# This file is execfile()d with the current directory set to its
# containing dir.
import importlib
import os
import sys
from unittest.mock import Mock

from mphys.utils.docs._utils.patch import do_monkeypatch
from mphys.utils.docs.config_params import MOCK_MODULES

# Only mock the ones that don't import.
for mod_name in MOCK_MODULES:
    try:
        importlib.import_module(mod_name)
    except ImportError:
        sys.modules[mod_name] = Mock()

# start off running the monkeypatch to keep options/parameters
# usable in docstring for autodoc.
do_monkeypatch()

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('./_exts'))

# -- General configuration ------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
needs_sphinx = '1.5'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinxcontrib.bibtex',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'numpydoc',
    'embed_code',
    'embed_options',
    'embed_compare',
    'embed_shell_cmd',
    'embed_bibtex',
    'embed_n2',
    'embed_pregen_n2',
    'tags'
]

bibtex_bibfiles=['references/papers_using_mphys.bib']
bibtex_default_style='plain'

numpydoc_show_class_members = False

# autodoc_default_flags = ['members']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
# source_suffix = ['.rst', '.md']
source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'


# -- Project information -----------------------------------------------------

project = 'MPhys'
copyright = '2022, NASA'
author = 'NASA'


# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', '_srcdocs/dev']
absp = os.path.join('.', '_srcdocs')
sys.path.insert(0, os.path.abspath(absp))

packages = [
    'mphys',
]

autoclass_content = 'both'
autodoc_member_order = 'bysource'
autosummary_generate = True

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = '_theme'

# Add any paths that contain custom themes here, relative to this directory.
html_theme_path = ['.']

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
#html_static_path = ['_static']

# Output file base name for HTML help builder.
htmlhelp_basename = 'OpenMDAOdoc'

# Customize sidebar
html_sidebars = {
   '**': ['localtoc.html', 'globaltoc.html', 'searchbox.html']
}

html_extra_path = ['_n2html']
