from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name="Home-Page"),
    path('base/', views.base, name="Base-Page"),
    path('information/', views.information, name="Information-Page"),
]