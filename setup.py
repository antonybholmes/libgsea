import setuptools

setuptools.setup(
    name='libgsea',
    version='0.6.0',
    author='Antony B Holmes',
    author_email='antony@antonybholmes.com',
    description='Library for GSEA including extended GSEA.',
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
