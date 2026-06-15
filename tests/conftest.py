"""Pytest configuration.

Integration tests (marked ``@pytest.mark.integration``) need a live Postgres
and/or network. They are skipped by default so the suite is green offline; set
``RUN_INTEGRATION=1`` (with the docker stack up) to run them.
"""

from __future__ import annotations

import os

import pytest


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip integration-marked tests unless ``RUN_INTEGRATION`` is set."""
    if os.environ.get("RUN_INTEGRATION"):
        return
    skip = pytest.mark.skip(reason="integration test; set RUN_INTEGRATION=1 with Postgres up")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)
