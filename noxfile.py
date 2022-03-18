import tempfile

import nox

nox.options.sessions = "lint", "tests"
LOCATIONS = "src", "tests", "noxfile.py"


@nox.session(python=["3.8", "3.9"])
def tests(session):
    args = session.posargs or ["--cov"]
    session.run("poetry", "install", "--no-dev", external=True)
    install_with_constraints(session, "coverage[toml]", "pytest", "pytest-cov", "torch")
    session.run("pytest", *args)


@nox.session(python=["3.8", "3.9"])
def lint(session):
    args = session.posargs or LOCATIONS
    install_with_constraints(
        session,
        "flake8",
        "flake8-bandit",
        "flake8-black",
        "flake8-bugbear",
        "flake8-import-order",
    )
    session.run("flake8", *args)


@nox.session(python=["3.9"])
def fmt(session):
    args = session.posargs or LOCATIONS
    install_with_constraints(session, "black")
    session.run("black", *args)


# add versioning of dev dependencies
def install_with_constraints(session, *args, **kwargs):
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
