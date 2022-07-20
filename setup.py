import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="perceval-quandela",
    version="0.0.1",
    author="Perceval@Quandela.com",
    author_email="Perceval@Quandela.com",
    description="A powerful Quantum Photonic Framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Quandela/Perceval",
    project_urls={
        "Documentation" : "https://perceval.quandela.net/docs/",
        "Source": "https://github.com/Quandela/Perceval",
        "Tracker": "https://github.com/Quandela/Perceval/issues"
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=['perceval', 'perceval.components', 'perceval.backends', 'perceval.utils', 'perceval.utils.renderer',
              'perceval.lib.phys', 'perceval.lib.symb', 'perceval.algorithm', 'perceval.converters'],
    install_requires=['sympy', 'numpy', 'scipy', 'tabulate', 'matplotlib', 'quandelibc>=0.5.2'],
    setup_requires=["scmver"],
    extras_require={"test": ["pytest", "pytest-cov"]},
    python_requires=">=3.6",
    scmver=True
)
