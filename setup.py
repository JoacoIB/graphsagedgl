import setuptools
with open("README.md", "r") as fh:
    long_description = fh.read()
setuptools.setup(
    name='graphsagedgl',
    version='0.1.0',
    scripts=[],
    author="Joaquin Ignacio Barotto",
    author_email="joaquin.barotto@gmail.com",
    description="GraphSage Implementation using Deep Graph Library.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/joacoib/graphsagedgl",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)"
    ],
)
