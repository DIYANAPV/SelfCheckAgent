from setuptools import setup, find_packages

setup(
    name="SemanticNgram",
    version="0.1.0",
    author="Diyana Muhammed",
    author_email="diyanapv@gmail.com",
    description="A semantic language model package with n-gram support and Word2Vec integration",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/DIYANAPV/Semantic_N-gram",
    packages=find_packages(include=["SemanticNgram", "SemanticNgram.*"]),  # Includes subpackages like `download_word2vec`
    include_package_data=True,  # Ensures non-Python files are included
    package_data={
        "SemanticNgram": ["model/*"],  # Includes files in the `model` folder
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "spacy",
        "gensim",
        "nltk",
        "scipy",
        "numpy",
        "tqdm",  # Add tqdm for progress bar support
        "requests",  # Add requests for downloading files
    ],
    python_requires=">=3.7",
)
