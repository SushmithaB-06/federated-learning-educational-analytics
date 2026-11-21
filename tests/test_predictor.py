"""Tests for the schema-driven predictor (uses the real saved artifacts)."""
import numpy as np
import pytest

from src import config
from src.data import load, preprocess
from src.models.train import predict_proba
from src.utils.predictor import MODEL_CHOICES, StudentPredictor, ValidationError

pytestmark = pytest.mark.skipif(
    not (config.MODELS_DIR / "full" / "fedavg_iid.pt").exists(),
    reason="trained models not present; run `python main.py all`")


@pytest.fixture(scope="module")
def predictor():
    return StudentPredictor("full", "fedavg_iid")


def test_default_student_is_valid_and_predicts(predictor):
    pred = predictor.predict(predictor.default_student())
    assert pred.label in {"Pass", "Fail"}
    assert abs(pred.probability_pass + pred.probability_fail - 1) < 1e-6
    assert 0 <= pred.probability_pass <= 1


def test_form_fields_match_training_schema(predictor):
    assert set(predictor.fields) == set(predictor.input_columns)
    assert "G3" not in predictor.fields
    assert "G1" in predictor.fields and "G2" in predictor.fields
    early = StudentPredictor("early", "fedavg_iid")
    assert "G1" not in early.fields and "G2" not in early.fields


def test_inference_matches_training_transform(predictor):
    """Transforming a raw test student must reproduce the stored processed row exactly."""
    df = load.load_raw_dataset()
    _, test_df = preprocess.split_raw(df)
    data = preprocess.load_processed("full")
    row = test_df.iloc[3]
    student = {c: (row[c] if c in load.CATEGORICAL_COLUMNS else int(row[c])) for c in predictor.input_columns}
    x = predictor.transform(student)
    np.testing.assert_allclose(x[0], data.X_test[3], atol=1e-6)
    p_direct = predict_proba(predictor.model, data.X_test[[3]])[0]
    assert abs(predictor.predict(student).probability_pass - p_direct) < 1e-6


def test_validation_reports_all_problems(predictor):
    student = predictor.default_student()
    student["age"] = 99
    student["sex"] = "X"
    del student["school"]
    student["G3"] = 15
    with pytest.raises(ValidationError) as exc:
        predictor.predict(student)
    msgs = " ".join(exc.value.problems)
    assert "age" in msgs and "sex" in msgs and "school" in msgs and "G3" in msgs
    assert len(exc.value.problems) == 4


def test_rejects_non_integer_and_text_numbers(predictor):
    student = predictor.default_student()
    student["absences"] = "many"
    with pytest.raises(ValidationError, match="absences"):
        predictor.predict(student)
    student["absences"] = 2.5
    with pytest.raises(ValidationError, match="whole number"):
        predictor.predict(student)


def test_all_model_choices_load():
    for key in MODEL_CHOICES:
        p = StudentPredictor("full", key)
        assert p.model.input_dim == 58


def test_sensitivity_table(predictor):
    tbl = predictor.sensitivity(predictor.default_student(), top_k=5)
    assert len(tbl) == 5
    assert (tbl["abs_swing"].diff().dropna() <= 1e-12).all()  # sorted descending
    assert 0 <= tbl.attrs["baseline_p_pass"] <= 1


def test_high_grades_more_likely_to_pass_than_low_grades(predictor):
    """Sanity check on the learned model, not a formal guarantee."""
    s = predictor.default_student()
    good = dict(s, G1=18, G2=18, failures=0)
    bad = dict(s, G1=4, G2=4, failures=3)
    assert predictor.predict(good).probability_pass > predictor.predict(bad).probability_pass
