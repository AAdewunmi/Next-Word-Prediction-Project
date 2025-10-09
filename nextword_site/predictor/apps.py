from django.apps import AppConfig

class PredictorConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "predictor"                 # <-- must match the package directory name
    verbose_name = "Next-Word Prediction"

