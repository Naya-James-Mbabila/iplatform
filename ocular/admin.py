from django.contrib import admin
from .models import PatientProfile, OcularImage, DoctorDiagnosis

@admin.register(PatientProfile)
class PatientProfileAdmin(admin.ModelAdmin):
    list_display = ('user', 'phone', 'location')
    search_fields = ('user__username', 'phone')

@admin.register(OcularImage)
class OcularImageAdmin(admin.ModelAdmin):
    list_display = ('id', 'patient', 'uploaded_at', 'is_analyzed')
    list_filter = ('is_analyzed', 'uploaded_at')
    raw_id_fields = ('patient',)

@admin.register(DoctorDiagnosis)
class DoctorDiagnosisAdmin(admin.ModelAdmin):
    list_display = ('image', 'doctor', 'created_at', 'short_diagnosis')
    list_filter = ('created_at',)
    raw_id_fields = ('image', 'doctor')
    
    def short_diagnosis(self, obj):
        return f"{obj.diagnosis_text[:50]}..." if len(obj.diagnosis_text) > 50 else obj.diagnosis_text
    short_diagnosis.short_description = 'Diagnosis'