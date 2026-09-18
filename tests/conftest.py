import json
import pytest
from sentiment_forecast.synthetic import write_fixture
from sentiment_forecast.data import (
    read_csv,
    trends_table,
    reviews_table,
    coverage_table,
)


@pytest.fixture
def dataset(tmp_path):
    folder = tmp_path / "inputs"
    path = write_fixture(folder)
    config = json.loads(path.read_text())
    return (
        config,
        trends_table(read_csv(folder / "trends.csv")),
        reviews_table(read_csv(folder / "reviews.csv"), "levis-505"),
        coverage_table(read_csv(folder / "coverage.csv")),
    )
