import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()


__version__ = "0.0.0"

REPO_NAME = "ikp-mlops"
AUTHOR_USER_NAME = "ArtitayaN"
SRC_REPO = "fraud_prediction"
AUTHOR_EMAIL = "arnamphimai@gmail.com"


setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="A small python package for fraud prediction app",
    long_description=long_description,
    long_description_content="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.24.3",
        "pandas>=1.5.0",
        "scikit-learn>=1.2.0"
    ],
    packages=setuptools.find_packages(where="src")
)