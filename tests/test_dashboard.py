"""Every dashboard page must render without raising, using the real artifacts."""
import json

import pytest
from streamlit.testing.v1 import AppTest

from src import config

PAGES = ["home", "dataset", "architecture", "clients", "training", "evaluation", "prediction", "privacy"]

pytestmark = pytest.mark.skipif(
    not (config.REPORTS_DIR / "experiment_summary.json").exists(),
    reason="experiment artifacts missing; run `python main.py all`")


def page(name: str) -> AppTest:
    """Build a tiny Streamlit script that imports one view module and renders it."""
    script = (
        f"import sys\nsys.path.insert(0, r'{config.PROJECT_ROOT}')\n"
        f"from dashboard.views import {name}\n{name}.render()\n"
    )
    return AppTest.from_string(script, default_timeout=180)


@pytest.mark.parametrize("name", PAGES)
def test_page_renders_without_exception(name):
    at = page(name).run()
    assert not at.exception, [e.message for e in at.exception]
    assert not at.error, [e.value for e in at.error]


def test_prediction_form_matches_schema_and_predicts():
    at = page("prediction").run()
    assert not at.exception
    schema = json.loads((config.MODELS_DIR / "full" / "schema.json").read_text())
    widget_keys = {w.key.split("_", 1)[1] for w in list(at.selectbox) + list(at.number_input)
                   if w.key and w.key.startswith("full_")}
    assert widget_keys == set(schema["input_columns"])
    at.button[0].click().run()
    assert not at.exception, [e.message for e in at.exception]
    assert "Probability of Pass" in " ".join(m.label for m in at.metric)


@pytest.mark.parametrize("name", ["training", "evaluation"])
def test_switch_to_early_variant(name):
    at = page(name).run()
    at.radio[0].set_value("early").run()
    assert not at.exception, [e.message for e in at.exception]
