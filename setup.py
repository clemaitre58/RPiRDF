import os
from setuptools import setup, find_packages

ver_file = os.path.join('DescGlob', '_version.py')
with open(ver_file) as f:
    exec(f.read())

PACKAGES = find_packages()

CLASSIFIERS = ["Development Status :: 3 - Alpha",
               "Environment :: Console",
               "Intended Audience :: Science/Research",
               "License :: OSI Approved :: MIT License",
               "Operating System :: OS Independent",
               "Programming Language :: Python",
               "Topic :: Scientific/Engineering"]

# Description should be a one-liner:
description = "DescGlob : a package for global image descriptors"
# Long description will go up on the pypi page
long_description = """
DescGlob :
================.
DescGlob is a packakage which contain some implementation of globale image
descriptors like :

- fourrier1
- ...
"""

NAME = "DescGlob"
MAINTAINER = "Cedric LEMAITRE"
MAINTAINER_EMAIL = "c.lemaitre58@gmail.com"
DESCRIPTION = description
LONG_DESCRIPTION = long_description
URL = "https://github.com/clemaitre58/RPiRDF"
DOWNLOAD_URL = ""
LICENSE = "MIT"
AUTHOR = "Cedric LEMAITRE"
AUTHOR_EMAIL = "c.lemaitre58@gmail.com"
PLATFORMS = "OS Independent"
VERSION = __version__
PACKAGE_DATA = {}
REQUIRES = ["numpy", "numba"]

opts = dict(name=NAME,
            maintainer=MAINTAINER,
            maintainer_email=MAINTAINER_EMAIL,
            description=DESCRIPTION,
            long_description=LONG_DESCRIPTION,
            url=URL,
            download_url=DOWNLOAD_URL,
            license=LICENSE,
            classifiers=CLASSIFIERS,
            author=AUTHOR,
            author_email=AUTHOR_EMAIL,
            platforms=PLATFORMS,
            version=VERSION,
            packages=PACKAGES,
            package_data=PACKAGE_DATA,
            install_requires=REQUIRES)


if __name__ == '__main__':
    setup(**opts)
