#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

setup(
    author="Hrayr Muradyan",
    author_email='hrayrmuradyan20@gmail.com',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    install_requires=['pandas', 'numpy', 'scikit-learn', 'matplotlib'],
    description="An easy-to-use package for customer profile creation in Python.",
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='CustomerProfile',
    name='CustomerProfile',
    packages=find_packages(include=['CustomerProfile', 'CustomerProfile.*']),
    test_suite='tests',
    url='https://github.com/HrayrMuradyan/CustomerProfile',
    version='0.1.1',
    zip_safe=False,
)
