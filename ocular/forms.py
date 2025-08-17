from django import forms
from .models import OcularImage, DoctorDiagnosis

class ImageUploadForm(forms.ModelForm):
    class Meta:
        model = OcularImage
        fields = ['image', 'notes']
        widgets = {
            'image': forms.FileInput(attrs={
                'class': 'form-control',
                'accept': 'image/*'
            }),
            'notes': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 3,
                'placeholder': 'Optional notes about the image...'
            })
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Make image field not required so camera uploads can work
        self.fields['image'].required = False

class DiagnosisForm(forms.ModelForm):
    class Meta:
        model = DoctorDiagnosis
        fields = ['diagnosis_text']
        widgets = {
            'diagnosis_text': forms.Textarea(attrs={
                'class': 'form-control', 
                'rows': 4,
                'placeholder': 'Enter your diagnosis and recommendations...'
            })
        }