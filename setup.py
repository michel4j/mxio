import os
from setuptools import find_packages, setup

with open(os.path.join(os.path.dirname(__file__), 'README.md')) as readme:
    long_description = readme.read()

with open(os.path.join(os.path.dirname(__file__), 'requirements.txt')) as f:
    requirements = f.read().splitlines()

# allow setup.py to be run from any path
os.chdir(os.path.normpath(os.path.join(os.path.abspath(__file__), os.pardir)))


def my_version():
    from setuptools_scm.version import get_local_dirty_tag

    def clean_scheme(version):
        return get_local_dirty_tag(version) if version.dirty else ''

    def version_scheme(version):
        return str(version.format_with('{tag}.{distance}'))

    return {'local_scheme': clean_scheme, 'version_scheme': version_scheme}



setup(
    name='mxio',
    use_scm_version=my_version,
    packages=find_packages(),
    url='https://github.com/michel4j/mxio',
    include_package_data=True,
    license='MIT',
    author='Michel Fodje',
    author_email='michel4j@gmail.com',
    description='A Simple X-ray Area Detector Data IO Library',
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=requirements,
    entry_points={
        'mxio.plugins': [
            'CBF = mxio.formats.cbf',
            'HDF5 = mxio.formats.hdf5',
            'MARCCD = mxio.formats.marccd',
            'RAXIS = mxio.formats.raxis',
            'SMV = mxio.formats.smv',
            'NEXUS = mxio.formats.nexus',
            'MAR345 = mxio.formats.mar345',
        ]
    },
    classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
    ],
)
