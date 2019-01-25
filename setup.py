import setuptools

setuptools.setup(
    name='libgsea',
    version='0.1.0',
    author='Antony B Holmes',
    author_email='antony.b.holmes@gmail.com',
    description='Library for working with GSEA plots.',
    url='https://github.com/antonybholmes/libgsea',
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    test_suite='nose.collector',
    tests_require=['nose'],
)
