from django.shortcuts import render
from .services import predictor as svc


def home(request):
    """
    Render the homepage and (on POST) run the next-word prediction.

    Context keys:
      - seed_text: the user-provided seed (echoed back into the textarea)
      - prediction_text: full text returned by the model (seed + generated)
      - elapsed_ms: inference time in milliseconds (optional display)
    """
    ctx = {
        "seed_text": "",
        "prediction_text": "",
        "elapsed_ms": None,
    }

    if request.method == "POST":
        seed_text = (request.POST.get("seed_text") or "").strip()
        ctx["seed_text"] = seed_text

        if seed_text:
            # Load the predictor only when needed (keeps GET light, helps tests).
            p = svc.get_predictor()
            # IMPORTANT: return the FULL string so tests can assert on HTML content
            full_text, elapsed_ms = p.predict(seed_text, n_words=12)
            ctx["prediction_text"] = full_text
            ctx["elapsed_ms"] = elapsed_ms

    return render(request, "predictor/home.html", ctx)

