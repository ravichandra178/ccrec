#urls here
from django.urls import path
from .views import DetailView

urlpatterns = [
    path('', DetailView, name = 'detail'),
]