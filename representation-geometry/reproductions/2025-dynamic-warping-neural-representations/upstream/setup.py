from setuptools import setup, find_packages

from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='riemannian_dynamics',
    packages=find_packages(exclude=['tests*']),
    version='0.0.1',
    package_data = {
        '': ['*.ttf'],
    },

    description='Code for RNNs perform task computations by warping neural representations',
    long_description=long_description,
    long_description_content_type='text/markdown',

    url='https://github.com/',
    author='Anon',
    license='MIT',
    install_requires=['jax',
                      'equinox',
                      'quax',
                      'diffrax',
                      'tqdm',
                      'scipy',
                      'matplotlib',
                      'optax',
                      'PyYAML',
                      'notebook',
                      'ipywidgets'
                      ],
    python_requires='>=3.10',
    classifiers=[
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Mathematics',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.10',
    ],
)
