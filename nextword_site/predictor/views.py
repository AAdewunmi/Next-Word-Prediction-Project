from django.http import HttpRequest, HttpResponse
from django.shortcuts import render
from django.views.decorators.http import require_http_methods

from .forms import PredictionForm
from .models import PredictionLog
from .services.predictor import get_predictor
from .utils.text import normalize_text_ascii_letters_only


@require_http_methods(["GET", "POST"])
def home(request: HttpRequest) -> HttpResponse:
    """
    Home page with input form and result panel.
    """
    form = PredictionForm(request.POST or None)
    context = {
        "form": form,
        "result": None,
        "error": None,
        "seq_length": get_predictor().get_seq_length(),
    }

    if request.method == "POST" and form.is_valid():
        raw = form.cleaned_data["seed_text"]
        plog = PredictionLog(input_text=raw)
        try:
            generated, latency_ms = get_predictor().predict(raw, n_words=12)
            plog.cleaned_text = normalize_text_ascii_letters_only(raw)
            plog.seq_length_used = get_predictor().get_seq_length()
            plog.generated_text = generated
            plog.success = True
            plog.latency_ms = latency_ms
            plog.client_ip = request.META.get("REMOTE_ADDR")
            plog.user_agent = request.META.get("HTTP_USER_AGENT", "")
            plog.save()
            context["result"] = generated
        except Exception as exc:
            plog.success = False
            plog.error_message = str(exc)
            plog.save()
            context["error"] = f"Prediction failed: {exc}"

    return render(request, "predictor/home.html", context)


# Create your views here.
