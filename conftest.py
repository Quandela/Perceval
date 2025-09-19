import pytest


def pytest_addoption(parser):
    parser.addoption("--skip-long-test", action="store_true",
                     help="do not run long tests")
    parser.addoption("--save_figs", action="store_true",
                     help="use to generate new test figures")


def pytest_runtest_setup(item):
    if 'long_test' in item.keywords and item.config.getoption("--skip-long-test"):
        pytest.skip("ignored per user request")
