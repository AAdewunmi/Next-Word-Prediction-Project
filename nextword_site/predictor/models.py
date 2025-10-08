from django.db import models

class PredictionLog(models.Model):
    """
    Store each prediction request/response for analytics and auditing.
    """
    input_text = models.TextField(help_text="Raw user input (pre-validation).")
    cleaned_text = models.TextField(blank=True, help_text="Normalized text used for tokenization.")
    seq_length_used = models.PositiveIntegerField(default=50)
    generated_text = models.TextField(blank=True)
    success = models.BooleanField(default=False)
    error_message = models.TextField(blank=True)
    latency_ms = models.PositiveIntegerField(default=0)
    client_ip = models.GenericIPAddressField(null=True, blank=True)
    user_agent = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [models.Index(fields=["created_at", "success"])]
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"Prediction #{self.pk} success={self.success}"

