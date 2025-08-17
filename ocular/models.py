from django.db import models
from django.contrib.auth.models import User

class PatientProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    phone = models.CharField(max_length=15, blank=True, null=True)
    location = models.CharField(max_length=100, blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Profile: {self.user.username}"

class OcularImage(models.Model):
    patient = models.ForeignKey(User, on_delete=models.CASCADE)
    image = models.ImageField(upload_to='ocular_images/')
    notes = models.TextField(blank=True, null=True, help_text="Optional notes about the image")
    uploaded_at = models.DateTimeField(auto_now_add=True)
    is_analyzed = models.BooleanField(default=False)  # Just a flag for doctor review

    class Meta:
        ordering = ['-uploaded_at']

    def __str__(self):
        return f"Image by {self.patient.username} - {self.uploaded_at.date()}"

class DoctorDiagnosis(models.Model):
    image = models.OneToOneField(OcularImage, on_delete=models.CASCADE, related_name='doctor_diagnosis')
    doctor = models.ForeignKey(User, on_delete=models.CASCADE, limit_choices_to={'is_staff': True})
    diagnosis_text = models.TextField(help_text="Doctor's diagnosis and recommendations")
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Diagnosis by {self.doctor.username} for Image {self.image.id}"