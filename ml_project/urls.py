from django.contrib import admin
from django.urls import path, include

# This is the clean, correct configuration.
# The development server will handle static files automatically.
urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('predictor.urls')),
]
