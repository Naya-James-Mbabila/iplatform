from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib.auth import login
from django.contrib.auth.forms import UserCreationForm
from django.contrib import messages
from .models import OcularImage, DoctorDiagnosis
from .forms import ImageUploadForm, DiagnosisForm
from django.core.files.base import ContentFile
import base64
import re

@login_required
def dashboard(request):
    if request.user.is_staff:
        pending_images = OcularImage.objects.filter(is_analyzed=False)
        return render(request, 'ocular/doctor_dashboard.html', {'images': pending_images})
    else:
        patient_images = OcularImage.objects.filter(patient=request.user)
        return render(request, 'ocular/patient_dashboard.html', {'images': patient_images})

@login_required
def upload_view(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        
        # Check if we have camera image data
        camera_image_data = request.POST.get('camera_image', '').strip()
        has_camera_image = bool(camera_image_data)
        has_file_upload = bool(request.FILES.get('image'))
        
        # Validate that we have either camera image or file upload
        if not has_camera_image and not has_file_upload:
            messages.error(request, "Please provide an image either by uploading a file or using the camera")
            return render(request, 'ocular/upload.html', {'form': form})
        
        # For camera uploads, we don't need the form to be valid for the image field
        # For file uploads, we need the form to be valid
        form_valid = form.is_valid()
        
        if (has_camera_image) or (has_file_upload and form_valid):
            try:
                # Create new OcularImage instance
                image = OcularImage()
                image.patient = request.user
                image.is_analyzed = False
                
                # Set notes if provided
                if form_valid and form.cleaned_data.get('notes'):
                    image.notes = form.cleaned_data['notes']
                elif request.POST.get('notes'):
                    image.notes = request.POST.get('notes', '')

                # Handle camera image
                if has_camera_image:
                    try:
                        # Validate data URL format
                        if not re.match(r'^data:image/(jpeg|jpg|png|gif);base64,', camera_image_data):
                            messages.error(request, "Invalid image format from camera")
                            return render(request, 'ocular/upload.html', {'form': form})
                        
                        # Extract format and base64 data
                        format_part, imgstr = camera_image_data.split(';base64,', 1)
                        ext = format_part.split('/')[-1].lower()
                        
                        # Handle jpg vs jpeg
                        if ext == 'jpg':
                            ext = 'jpeg'
                        
                        # Decode base64 image
                        image_file = ContentFile(
                            base64.b64decode(imgstr), 
                            name=f'camera_capture_{request.user.id}.{ext}'
                        )
                        image.image.save(f'camera_capture_{request.user.id}.{ext}', image_file)
                        
                    except Exception as e:
                        messages.error(request, f"Failed to process camera image: {str(e)}")
                        return render(request, 'ocular/upload.html', {'form': form})
                
                # Handle file upload
                elif has_file_upload and form_valid:
                    image.image = form.cleaned_data['image']
                
                # Save the image
                image.save()
                messages.success(request, 'Image uploaded successfully!')
                return redirect('dashboard')
                
            except Exception as e:
                messages.error(request, f"Failed to save image: {str(e)}")
                return render(request, 'ocular/upload.html', {'form': form})
        else:
            # Form validation failed for file upload
            messages.error(request, "Please correct the errors below")
    else:
        form = ImageUploadForm()
    
    return render(request, 'ocular/upload.html', {'form': form})
    
@login_required
def analysis_view(request, image_id):
    image = get_object_or_404(OcularImage, id=image_id)
    diagnosis = DoctorDiagnosis.objects.filter(image=image).first()

    if request.user.is_staff and request.method == 'POST':
        form = DiagnosisForm(request.POST, instance=diagnosis)
        if form.is_valid():
            diagnosis = form.save(commit=False)
            diagnosis.image = image
            diagnosis.doctor = request.user
            diagnosis.save()
            image.is_analyzed = True
            image.save()
            messages.success(request, 'Diagnosis saved successfully!')
            return redirect('dashboard')
    else:
        form = DiagnosisForm(instance=diagnosis) if request.user.is_staff else None

    return render(request, 'ocular/analysis.html', {
        'image': image,
        'form': form,
        'diagnosis': diagnosis
    })

def register(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            messages.success(request, 'Registration successful!')
            return redirect('dashboard')
    else:
        form = UserCreationForm()
    return render(request, 'ocular/register.html', {'form': form})