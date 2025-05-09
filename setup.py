from setuptools import setup, find_packages
from setuptools.command.install import install
import subprocess


class PostInstallCommand(install):
    """Post-installation command for downloading SpaCy model and ensuring Hugging Face dependencies."""
    def run(self):
        install.run(self)
        print("Downloading SpaCy model: en_core_web_sm...")
        subprocess.call(['python', '-m', 'spacy', 'download', 'en_core_web_sm'])


setup(
    name="selfcheckagent",  # Package name
    version="0.1.2",  # Updated version
    author="hiddennode",  # Package author
    author_email="hiddennode@gmail.com",
    description="A self-check agent for contextual, specialized, and symbolic consistency.",
    long_description=open('README.md', encoding="utf-8").read(),
    long_description_content_type='text/markdown',
    url="-",  # Repository URL

    # Packages included
    packages=find_packages(include=["selfcheckagent", "selfcheckagent.*"]),

    install_requires=[
        'spacy>=3.0',
        'torch>=1.8',
        'transformers>=4.0',
        'numpy',
        'pandas',
        'tqdm',
        'scipy',
        'nltk',
        'gensim',
    ],

    python_requires='>=3.8',  # Minimum Python version

    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],

    include_package_data=True,


    entry_points={
    'console_scripts': [
        'contextual-agent=selfcheckagent.Contextual_Consistency_Agent:main',  # CLI for ContextualAgent
        'nli-agent=selfcheckagent.Specialized_Detection_Agent:main',          # CLI for SelfCheckNLI
        'semantic-agent=selfcheckagent.Semantic_Agent:main',                  # CLI for Semantic Model
    ],
},


    cmdclass={
        'install': PostInstallCommand,  # Custom post-installation script
    },
)

