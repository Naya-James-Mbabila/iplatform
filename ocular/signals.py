from django.db.models.signals import post_save
from django.dispatch import receiver

def register_signals():
    from django.contrib.auth import get_user_model
    User = get_user_model()
    
    @receiver(post_save, sender=User)
    def create_patient_profile(sender, instance, created, **kwargs):
        if created:
            from .models import PatientProfile
            PatientProfile.objects.create(user=instance)

    @receiver(post_save, sender=User)
    def save_patient_profile(sender, instance, **kwargs):
        if hasattr(instance, 'patientprofile'):
            instance.patientprofile.save()