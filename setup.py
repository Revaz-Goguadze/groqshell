from setuptools import setup, find_packages
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

VERSION = "0.0.2"
DESCRIPTION = "Command-line interface for Groq AI models"
LONG_DESCRIPTION = long_description

# Setting up
setup(
    name="groqshell-cli",
    version=VERSION,
    author="Johnny Cage",
    author_email="<johnnycage111@gmail.com>",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    install_requires=["groq", "colorama", "rich", "pygments"],
    entry_points={
        "console_scripts": [
            "groqshell=groqshell.mainrich:main",
        ],
    },
    keywords=["python", "groq", "ai", "cli", "shell", "interface"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ],
)
