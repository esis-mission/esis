# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

package_path = os.path.abspath('../')
sys.path.insert(0, package_path)
os.environ['PYTHONPATH'] = ';'.join((package_path, os.environ.get('PYTHONPATH', '')))

# -- Project information -----------------------------------------------------

project = 'ESIS'
copyright = '2025, Roy T. Smart, Charles C. Kankelborg, Jacob D. Parker'
author = 'Roy T. Smart, Charles C. Kankelborg, Jacob D. Parker'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.napoleon',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.inheritance_diagram',
    'sphinx.ext.viewcode',
    'sphinxcontrib.bibtex',
    'jupyter_sphinx',
    'nbsphinx',
    'sphinx_codeautolink',
    'sphinx_favicon'
]
autosummary_generate = True  # Turn on sphinx.ext.autosummary
autosummary_imported_members = True
autosummary_ignore_module_all = False
# autoclass_content = 'both'
autodoc_typehints = "description"

graphviz_output_format = 'png'
inheritance_graph_attrs = dict(rankdir='TB')

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# This pattern also affects html_static_path and html_extra_path.
# directories to ignore when looking for source files.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.

html_theme = 'pydata_sphinx_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

html_theme_options = {
    "logo": {
        "image_light": "_static/logo_black.png",
        "image_dark": "_static/logo_white.png",
    },
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/esis-mission/esis",
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/euv-snapshot-imaging-spectrograph/",
            "icon": "fa-brands fa-python",
        },
    ],
    "analytics": {
        "google_analytics_id": "G-CQ9NDQQWKM",
    }
}

favicons = [
    dict(href="favicon_io/favicon-16x16.png"),
    dict(href="favicon_io/favicon-32x32.png"),
    dict(
        rel="apple-touch-icon",
        href="favicon_io/apple-touch-icon.png"
    )
]

# https://github.com/readthedocs/readthedocs.org/issues/2569
master_doc = 'index'

bibtex_bibfiles = ['refs.bib']
bibtex_default_style = 'plain'
bibtex_reference_style = 'author_year'

nbsphinx_execute = 'always'

codeautolink_custom_blocks = {"jupyter-execute": None}

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'ipython': ('https://ipython.readthedocs.io/en/stable/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'matplotlib': ('https://matplotlib.org/stable', None),
    'astropy': ('https://docs.astropy.org/en/stable/', None),
    'astroscrappy': ('https://astroscrappy.readthedocs.io/en/stable/', None),
    'named_arrays': ('https://named-arrays.readthedocs.io/en/stable/', None),
    'optika': ('https://optika.readthedocs.io/en/stable/', None),
    'msfc_ccd': ('https://msfc-ccd.readthedocs.io/en/stable/', None),
}
