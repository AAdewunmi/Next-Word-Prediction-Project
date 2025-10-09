import pytest
from predictor.forms import PredictionForm
from predictor.services.predictor import get_required_seq_length


@pytest.mark.django_db
def test_form_validates_exact_token_count(monkeypatch):
    monkeypatch.setattr("predictor.forms.get_required_seq_length", lambda: 5)
    form = PredictionForm(data={"seed_text": "a b c d e"})
    assert form.is_valid()

    form2 = PredictionForm(data={"seed_text": "a b c d"})
    assert not form2.is_valid()
