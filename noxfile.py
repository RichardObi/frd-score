""" Nox configuration file for automation of linting and testing.

Adopted and adjusted from https://github.com/mseitzer/pytorch-fid
"""

import nox

LOCATIONS = ("src/", "tests/", "noxfile.py", "setup.py")


@nox.session(python=["3.8", "3.9", "3.10", "3.11", "3.12"])
def tests(session):
    session.install("setuptools")
    session.install("-r", "requirements.in")
    session.install("numpy") #get numpy's latest version based on python version (e.g. python3.12)
    session.install(".")
    session.install("pytest")
    session.install("pytest-mock")
    session.run("pytest", *session.posargs)

@nox.session
def lint(session):
    session.install("flake8")
    session.install("flake8-bugbear")
    session.install("flake8-isort")
    session.install("black==24.3.0")
    session.install("isort")

    args = session.posargs or LOCATIONS
    session.run("black", "--check", "--diff", *args)
    session.run("isort")
    session.run("flake8", *args)

