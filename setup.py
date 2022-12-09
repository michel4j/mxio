import os
from setuptools import find_packages, setup

with open(os.path.join(os.path.dirname(__file__), 'README.md')) as readme:
    long_description = readme.read()

with open(os.path.join(os.path.dirname(__file__), 'requirements.txt')) as f:
    requirements = f.read().splitlines()

# allow setup.py to be run from any path
os.chdir(os.path.normpath(os.path.join(os.path.abspath(__file__), os.pardir)))


setup(
    name='mxio',
    use_scm_version=True,
    packages=find_packages(),
    url='https://github.com/michel4j/mxio',
    include_package_data=True,
    license='MIT',
    author='Michel Fodje',
    author_email='michel4j@gmail.com',
    description='A Simple MX Diffraction Image Library',
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=requirements+ [
        'importlib-metadata ~= 1.0 ; python_version < "3.8"', 'setuptools_scm'
    ],
    classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
    ],
)
