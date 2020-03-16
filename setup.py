from setuptools import setup, find_packages


setup(
    name='tf_custom_cxx_op',
    version='0.0.0',
    packages=find_packages('src/python'),
    package_dir={'': 'src/python'},
    install_requires=[
        'tensorflow',
        'numpy',
        'scipy',
        'absl-py',
    ],
    zip_safe=False,
)
