from django.contrib import admin
from mypatient.models import Patient


class PatientAdmin(admin.ModelAdmin):
    list_display = ( "idx", "name", "age","height", "weight",
                     "blood_type","last_visit","memo", "picture_url")


admin.site.register(Patient, PatientAdmin)