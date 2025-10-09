from django.urls import path
from .views import home

app_name = "predictor"

urlpatterns = [
    path("", home, name="home"),
]
