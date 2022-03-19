"""Nox sessions.

Nox is used to run testing, linting, and formatting
in a consistent way.
"""
import tempfile

import nox

nox.options.sessions = "lint", "tests"
LOCATIONS = "src", "tests", "noxfile.py", "conf.py"


@nox.session(python=["3.8", "3.9"])
def tests(session):
    """Runs the test suite."""
    args = session.posargs or ["--cov"]
    session.run("poetry", "install", "--no-dev", external=True)
    install_with_constraints(session, "coverage[toml]", "pytest", "pytest-cov", "torch")
    session.run("pytest", *args)


@nox.session(python=["3.8", "3.9"])
def doctests(session):
    """Runs the documentation tests with xdoctest."""
    args = session.posargs or ["all"]
    session.run("poetry", "install", "--no-dev", external=True)
    install_with_constraints(session, "xdoctest")
    session.run("xdoctest", "-m", "panndas", *args)


@nox.session(python=["3.8", "3.9"])
def docs(session):
    """Build the docs with sphinx."""
    session.run("poetry", "install", "--no-dev", external=True)
    install_with_constraints(session, "sphinx")
    session.run("sphinx-build", ".", "docs/")
    session.run("sphinx-apidoc", "-o", "docs/", "src/panndas")


@nox.session(python=["3.8", "3.9"])
def coverage(session):
    """Runs the coverage check and reports it to codecov."""
    install_with_constraints(session, "coverage[toml]", "codecov")
    session.run("coverage", "xml", "--fail-under=0")
    session.run("codecov", *session.posargs)


@nox.session(python=["3.8", "3.9"])
def lint(session):
    """Lints all Python files with flake8 + extensions."""
    args = session.posargs or LOCATIONS
    install_with_constraints(
        session,
        "flake8",
        "flake8-bandit",
        "flake8-black",
        "flake8-docstrings",
        "flake8-bugbear",
        "flake8-import-order",
        "darglint",
    )
    session.run("flake8", *args)


@nox.session(python=["3.8", "3.9"])
def fmt(session):
    """Runs the black code formatter."""
    args = session.posargs or LOCATIONS
    install_with_constraints(session, "black")
    session.run("black", *args)


# add versioning of dev dependencies
def install_with_constraints(session, *args, **kwargs):
    """Installs packages using Poetry's lock file as a constraint."""
    with tempfile.NamedTemporaryFile() as requirements:
        session.run(
            "poetry",
            "export",
            "--dev",
            "--format=requirements.txt",
            "--without-hashes",
            f"--output={requirements.name}",
            external=True,
        )
        session.install(f"--constraint={requirements.name}", *args, **kwargs)
