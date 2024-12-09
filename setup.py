'''from setuptools import setup, find_packages
from setuptools.command.install import install
import subprocess

class PostInstallCommand(install):
    """Post-installation for installing the Spacy model."""
    def run(self):
        install.run(self)
        subprocess.call(['python', '-m', 'spacy', 'download', 'en_core_web_sm'])'''

from setuptools import setup, find_packages
from setuptools.command.install import install
import os
import subprocess

class PostInstallCommand(install):
    """Post-installation command to download required resources."""
    def run(self):
        install.run(self)
        print("Running post-install script to download resources...")
        subprocess.call(["python", "download_resources.py"])

setup(
    name="my_package",
    version="0.1",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "spacy",
        "numpy",
        "nltk",
        "scipy",
        "gensim",
    ],
    cmdclass={
        'install': PostInstallCommand,
    },
    entry_points={
        "console_scripts": [
            "download-resources=my_package.download_resources:download_word2vec",
        ]
    },
)


setup(
    name="Semantic_Ngram",  # Name of your package
    version="0.1",  # Initial version of your package
    author="diyana",  # Author of the package
    author_email="diyanapv@gmail.com",  # Author's email
    description="A package for semantic n-gram models with Word2Vec",  # Short description of the package
    long_description=open('README.md').read(),  # Detailed description (from README.md)
    long_description_content_type='text/markdown',  # Markdown format for README
    url="https://github.com/DIYANAPV/Semantic_Ngram.git",  # URL to your GitHub repository
    packages=find_packages(),  # This will automatically find all sub-packages
    install_requires=[
        'spacy',
        'gensim',
        'nltk',
        'scipy',
        'tqdm',
        'requests'
    ],

    python_requires='>=3.6',  # Minimum Python version required
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  # You can change this to your license
        "Operating System :: OS Independent",
    ],
    include_package_data=True,  # Includes non-Python files specified in MANIFEST.in (if any)
    entry_points={  # If you have CLI commands, define them here
        'console_scripts': [
            "download-resources=my_package.download_resources:download_word2vec",
            #'semantic-ngram=SemanticNgram.main:main',  # Example command (adjust based on your entry point)
        ],
    },
    cmdclass={
        'install': PostInstallCommand,
    },
)
