import pytest


def pytest_addoption(parser):
    parser.addoption("--long_test", action="store_true",
                     help="run long tests")


def pytest_runtest_setup(item):
    if 'long_test' in item.keywords and not item.config.getoption("--long_test"):
        pytest.skip("need --long_test option to run this test")