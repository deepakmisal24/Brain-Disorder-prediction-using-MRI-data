from django.urls import path
from . import views

urlpatterns = [
    # URL for the main landing page
    path('', views.index, name='index'),
    
    # URL for the Alzheimer's prediction page
    path('alzheimers/', views.alzheimers_page, name='alzheimers_page'),
    
    # URL for the Brain Tumor prediction page
    path('braintumor/', views.braintumor_page, name='braintumor_page'),
    
    # URL for the Parkinson's prediction page
    path('parkinsons/', views.parkinsons_page, name='parkinsons_page'),
    
    # We will also need URLs to handle the actual predictions later
    # For example: path('predict/alzheimers/numerical/', views.predict_alz_numerical, name='predict_alz_numerical'),
]
