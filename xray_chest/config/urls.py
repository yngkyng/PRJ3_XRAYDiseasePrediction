from django.contrib import admin
from django.urls import path, include
from config import views
from mypatient import views as view

urlpatterns = [
    path("admin/", admin.site.urls),
    path('',views.home),
    path('mymember/', include('mymember.urls')),
    path('mypatient/', include('mypatient.urls')),
    path('mymember/Atelectasis', view.Atelectasis, name='Atelectasis'),
    path('mymember/Cardiomegaly/', view.Cardiomegaly, name='Cardiomegaly'),
    path('mymember/Edema/', view.Edema, name='Edema'),
    path('mymember/Effusion2/', view.Effusion2, name='Effusion2'),
    path('mymember/Fibrosis/', view.Fibrosis, name='Fibrosis'),
    path('mymember/Pneumonia/', view.Pneumonia, name='Pneumonia'),
    path('mymember/Pneumothorax/', view.Pneumothorax, name='Pneumothorax'),
    path('mymember/Tuberculosis/', view.Tuberculosis, name='Tuberculosis'),
    path('mymember/Total/', view.Multiclassification, name='Total'),
]
