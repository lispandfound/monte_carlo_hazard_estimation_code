import pytest
from hypothesis import Phase, settings


# 1. Define the custom command line argument
def pytest_addoption(parser):
    parser.addoption(
        "--smoke",
        action="store_true",
        default=False,
        help="Run hypothesis tests with 1 example",
    )


# 2. Register the profiles
settings.register_profile(
    "smoke", max_examples=1, phases=[Phase.generate], deadline=None
)


# 3. Load the profile based on the flag
def pytest_configure(config):
    if config.getoption("--smoke"):
        settings.load_profile("smoke")
