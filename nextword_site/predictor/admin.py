from django.contrib import admin
from .models import PredictionLog
# Register your models here.
from django.contrib import admin
from .models import PredictionLog


@admin.register(PredictionLog)
class PredictionLogAdmin(admin.ModelAdmin):
    list_display = ("id", "success", "seq_length_used", "latency_ms", "created_at")
    list_filter = ("success", "created_at")
    search_fields = ("input_text", "generated_text", "error_message")
    readonly_fields = ("created_at",)
