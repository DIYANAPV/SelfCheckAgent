from setuptools import setup, find_packages
from setuptools.command.install import install
import subprocess

class PostInstallCommand(install):
    """Post-installation for installing the Spacy model."""
    def run(self):
        install.run(self)
        subprocess.call(['python', '-m', 'spacy', 'download', 'en_core_web_sm'])

setup(
    name="selfcheckagent",  # Updated package name
    version="0.1.1",  # Incremented version
    author="Diyana",  # Author of the package
    author_email="diyanapv@gmail.com",
    description="A self-check agent for contextual and symbolic consistency.",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url="https://github.com/DIYANAPV/SelfCheckAgent",  # Updated repository URL

    # Include selfcheckagent package and its modules
    packages=find_packages(include=["selfcheckagent", "selfcheckagent.*"]),

    install_requires=[
        'spacy',
        'torch',
        'transformers',
        'numpy',
        'pandas',
        'tqdm',
        'scipy',
        'nltk',
        'gensim',
    ],

    python_requires='>=3.6',  # Minimum Python version

    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],

    include_package_data=True,  # Includes non-Python files if any

    entry_points={
        'console_scripts': [
            'contextual-agent=selfcheckagent.contextual_agent:main',  # Example CLI for contextual agent
            'symbolic-agent=selfcheckagent.symbolic_agent:semantic_model_predict',  # Example CLI for symbolic model
        ],
    },

    cmdclass={
        'install': PostInstallCommand,  # Installs Spacy's model
    },
)
