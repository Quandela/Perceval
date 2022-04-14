import pytest


def pytest_addoption(parser):
    parser.addoption("--long_test", action="store_true",
                     help="run long tests")
    parser.addoption("--save_figs", action="store_true",
                     help="use to generate new test figures")


def pytest_runtest_setup(item):
    if 'long_test' in item.keywords and not item.config.getoption("--long_test"):
        pytest.skip("need --long_test option to run this test")
