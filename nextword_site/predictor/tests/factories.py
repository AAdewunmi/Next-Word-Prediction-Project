import factory
from predictor.models import PredictionLog


class PredictionLogFactory(factory.django.DjangoModelFactory):
    class Meta:
        model = PredictionLog

    input_text = factory.Faker("sentence")
    cleaned_text = factory.Faker("sentence")
    seq_length_used = 50
    generated_text = factory.Faker("sentence")
    success = True
    error_message = ""
    latency_ms = 12


# Create your tests here.
