import pytest
from django.urls import reverse


@pytest.mark.django_db
def test_home_get(client):
    resp = client.get(reverse("predictor:home"))
    assert resp.status_code == 200
    assert b"Next" in resp.content


@pytest.mark.django_db
def test_home_post_success(client, monkeypatch):
    # Avoid loading TF; stub predictor
    from predictor.services import predictor as svc
    class Fake:
        def get_seq_length(self): return 3
        def predict(self, text, n_words=12): return text + " next", 7
    monkeypatch.setattr(svc, "get_predictor", lambda: Fake())

    data = {"seed_text": "one two three"}
    resp = client.post(reverse("predictor:home"), data)
    assert resp.status_code == 200
    assert b"one two three next" in resp.content
