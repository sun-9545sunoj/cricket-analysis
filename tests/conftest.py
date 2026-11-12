import os

import pytest

from src import models
from src.etl import ingest

FIXTURE_ZIP = os.path.join(os.path.dirname(__file__), "fixtures", "india_nz_matches.zip")


@pytest.fixture(scope="session", autouse=True)
def fixture_database(tmp_path_factory):
    """Builds a throwaway database and models from 45 real India v New Zealand matches (2019-2026)."""
    mp = pytest.MonkeyPatch()
    mp.setenv("CRICKET_DATA_DIR", str(tmp_path_factory.mktemp("data")))
    # The fixture is far smaller than the real dataset, so lower the clustering threshold
    mp.setitem(models.CLUSTER_MIN_BALLS, "T20", 60)
    mp.setitem(models.CLUSTER_MIN_BALLS, "ODI", 100)
    count = ingest([FIXTURE_ZIP])
    report = models.train_all()
    yield {"count": count, "report": report}
    mp.undo()
