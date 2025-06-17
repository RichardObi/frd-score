""" Nox configuration file for automation of linting and testing.

Adopted and adjusted from https://github.com/mseitzer/pytorch-fid
"""

import nox

LOCATIONS = ("frd_v1/", "frd_v0/", "frd_v1/", "frd_v1/src/", "frd_v0/tests/", "frd_v0/src/", "frd_v0/tests/",  "src/", "tests/", "noxfile.py", "setup.py")


#@nox.session(python=["3.9"])
@nox.session(python=["3.9", "3.10", "3.11", "3.12"])
def tests(session):
    session.install("setuptools")
    session.install("numpy")
    #session.install("-r", "requirements.in")
    session.install("-r", "frd_v1/requirements.txt")
    session.install("-r", "frd_v0/requirements.in")
    # Note: Pyradiomics might need to be installed from github in order for compatibility with python versions >3.9
    # https://github.com/AIM-Harvard/pyradiomics/issues/903
    session.install("frd_v0/.")
    #session.install("frd_v1/.")

    # Install PyRadiomics from GitHub (main branch): This is in case previous installs of the frd dir have incompatible versions (e.g. numpy <2.0)
    session.install("git+https://github.com/Radiomics/pyradiomics.git@master#egg=pyradiomics")
    session.install("numpy") #get numpy's again, but now overwriting requirements.in file to get numpy's latest version based on python version (e.g. python3.12)

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

