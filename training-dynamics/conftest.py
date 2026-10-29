"""Course-wide pytest configuration.  [AI]

`uv run pytest a0-prereqs --solutions` runs every test against the reference implementations
in a0-prereqs/solutions/*_solution.py instead of your code. It works by setting the
environment variable TDYN_SOLUTIONS=1, which tdyn.swap.impl (called by every
tests/adapters.py) reads; `TDYN_SOLUTIONS=1 uv run pytest a0-prereqs` is equivalent.
"""
import os


def pytest_addoption(parser):
    parser.addoption("--solutions", action="store_true", default=False,
                     help="run the tests against solutions/*_solution.py (sets TDYN_SOLUTIONS=1)")


def pytest_configure(config):
    if config.getoption("--solutions"):
        os.environ["TDYN_SOLUTIONS"] = "1"


def pytest_report_header(config):
    which = "reference solutions" if os.environ.get("TDYN_SOLUTIONS") == "1" else "your code"
    return f"tdyn: testing {which}"
