from django.urls import path
from . import views

urlpatterns = [
    path('', views.mainpage, name='mainpage'),  # Root URL points to the main page view
    path('moreinfo/', views.more_info, name='moreinfo'),  # More info page
    path('home/', views.home, name='home'),  # Prediction form page
    path('predict/', views.predict, name='predict'),  # Handle the prediction logic
    path('Symptoms/', views.Symptoms, name='Symptoms'),  # User Symptoms page
    path('prevent/', views.prevent, name='prevent'),  # User prevention & treatment page
    # Add other paths as needed
]
