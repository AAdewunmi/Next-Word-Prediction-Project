from django import forms
from django.core.exceptions import ValidationError
from .services.predictor import get_required_seq_length

class PredictionForm(forms.Form):
    """
    Single text input that must contain exactly `seq_length` tokens.
    """
    seed_text = forms.CharField(
        label="Seed Text",
        widget=forms.Textarea(attrs={"rows": 4, "placeholder": "Enter exactly 50 words..."}),
        help_text="Enter exactly the required number of words (default 50) used to seed the next-word predictor.",
    )

    def clean_seed_text(self):
        text = self.cleaned_data["seed_text"]
        # Split on whitespace; do not alter text yet (preprocessing happens server-side)
        tokens = [t for t in text.split() if t.strip()]
        required = get_required_seq_length()
        if len(tokens) != required:
            raise ValidationError(f"Please provide exactly {required} words (got {len(tokens)}).")
        return text
