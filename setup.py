from setuptools import setup, find_packages

setup(
    name="shellgroq",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pocketgroq",
    ],
    entry_points={
        "console_scripts": [
            "shellgpt=Main:main",
        ],
    },
    author="Coder",
    author_email="your.email@example.com",
    description="A shell interface for Groq AI",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/shellgpt",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
