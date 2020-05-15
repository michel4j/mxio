import os
from setuptools import find_packages, setup

with open(os.path.join(os.path.dirname(__file__), 'README.md')) as readme:
    long_description = readme.read()

with open(os.path.join(os.path.dirname(__file__), 'requirements.txt')) as f:
    requirements = f.read().splitlines()

# allow setup.py to be run from any path
os.chdir(os.path.normpath(os.path.join(os.path.abspath(__file__), os.pardir)))


def package_version():
    from mxio.version import get_version
    return get_version()


setup(
    name='mxio',
    version=package_version(),
    packages=find_packages(),
    url='https://github.com/michel4j/mxio',
    include_package_data=True,
    license='MIT',
    author='Michel Fodje',
    author_email='michel4j@gmail.com',
    description='A Simple MX Diffraction Image Library',
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=requirements,
    classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.7',
    ],
)
