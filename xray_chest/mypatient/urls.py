from django.urls import path
from mypatient import views

urlpatterns = [
    path('', views.home),
    path('detail', views.detail),
    path('patient_info', views.patient_info),
    # path('patient_info', views.patient_info),
    path('update', views.update),
    path('delete', views.delete),
    path('search', views.search),
    path('register/', views.regi),
    path('receipt/', views.receipt),
    path('Atelectasis/', views.Atelectasis),
    path('Cardiomegaly/', views.Cardiomegaly),
    path('Edema/', views.Edema),
    path('Effusion2/', views.Effusion2),
    path('Fibrosis/', views.Fibrosis),
    path('Pneumonia/', views.Pneumonia),
    path('Pneumothorax/', views.Pneumothorax),
    path('Tuberculosis/', views.Tuberculosis),

]