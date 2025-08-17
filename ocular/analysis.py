import cv2
import numpy as np
import os
import requests
import base64
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from PIL import Image
from django.conf import settings
from django.core.files.storage import default_storage
import tempfile
import json

logger = logging.getLogger(__name__)

class EyeAnalyzer:
    def __init__(self, hf_token: str = "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"):
        self.device = "cpu"  # Use CPU for Django deployment
        
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_eye.xml'
            self.eye_cascade = cv2.CascadeClassifier(cascade_path)
            if self.eye_cascade.empty():
                self.eye_cascade = None
                logger.warning("Eye cascade classifier could not be loaded")
        except Exception as e:
            self.eye_cascade = None
            logger.error(f"Error loading cascade: {e}")
        
        self.thresholds = {
            'redness': {'mild': 8, 'moderate': 20, 'severe': 35},
            'yellowing': {'mild': 5, 'moderate': 15, 'severe': 25},
            'pallor': {'mild': 20, 'moderate': 40, 'severe': 60}
        }
        
        self.hf_token = hf_token
        self.ai_enabled = hf_token is not None

    def analyze_eyes(self, django_image_field) -> Dict:
        """Main method to analyze Django image field"""
        try:
            # Create temporary file from Django image field
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                # Reset file pointer and read data
                django_image_field.seek(0)
                temp_file.write(django_image_field.read())
                temp_file_path = temp_file.name
            
            try:
                # Analyze the temporary image file
                results = self._analyze_image_file(temp_file_path)
                return self._format_for_django_template(results)
            finally:
                # Clean up temporary file
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
                    
        except Exception as e:
            logger.error(f"Error analyzing Django image: {e}")
            return {
                "error": f"Analysis failed: {str(e)}",
                "success": False
            }

    def analyze_image_path(self, image_path: str) -> Dict:
        """Analyze image from file path"""
        try:
            if not os.path.exists(image_path):
                return {"error": "Image file not found", "success": False}
                
            results = self._analyze_image_file(image_path)
            return self._format_for_django_template(results)
            
        except Exception as e:
            logger.error(f"Error analyzing image path: {e}")
            return {
                "error": f"Analysis failed: {str(e)}",
                "success": False
            }

    def _analyze_image_file(self, image_path: str) -> Dict:
        """Core image analysis logic adapted from your original code"""
        # Load and validate image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Could not load image - file may be corrupted or invalid format")
        
        logger.info(f"Image loaded successfully: {image.shape}")
        
        # Find eye regions using your original method
        eye_centers = self._find_eye_centers(image)
        if not eye_centers:
            logger.warning("No eyes detected in image")
            return {"error": "No eyes detected in the image - please ensure the image shows clear eye regions"}
        
        logger.info(f"Found {len(eye_centers)} eye centers: {eye_centers}")
        
        # Analyze each eye using simplified version of your original method
        eye_analyses = []
        for i, center in enumerate(eye_centers):
            eye_data = self._analyze_eye_simplified(image, center, f"Eye {i+1}")
            if eye_data:
                eye_analyses.append(eye_data)
        
        if not eye_analyses:
            return {"error": "Could not analyze any eye regions - image quality may be insufficient"}
        
        # Generate overall assessment using your original logic
        overall_assessment = self._generate_clinical_report(eye_analyses)
        
        # Get AI explanation if enabled
        if self.ai_enabled:
            ai_explanation = self._get_hf_explanation(overall_assessment)
            overall_assessment['ai_explanation'] = ai_explanation
        
        return {
            "success": True,
            "eye_analyses": eye_analyses,
            "overall_assessment": overall_assessment,
            "metadata": {
                "eyes_detected": len(eye_analyses),
                "image_shape": image.shape,
                "analysis_timestamp": datetime.now()
            }
        }

    def _find_eye_centers(self, image: np.ndarray) -> List[Tuple[int, int]]:
        """Your original eye detection method"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Try cascade classifier first if available
        if self.eye_cascade is not None:
            try:
                eyes = self.eye_cascade.detectMultiScale(
                    gray, 
                    scaleFactor=1.1, 
                    minNeighbors=4,
                    minSize=(30, 30),
                    maxSize=(200, 200)
                )
                if len(eyes) > 0:
                    centers = [(x + w//2, y + h//2) for (x, y, w, h) in eyes]
                    logger.info(f"Cascade detected {len(centers)} eyes")
                    return centers[:2]  # Limit to 2 eyes max
            except Exception as e:
                logger.warning(f"Cascade detection failed: {e}")
        
        # Fallback to estimated positions (your original fallback)
        h, w = gray.shape
        estimated_eyes = [
            (w//3, h//3),      # Left eye (adjusted from your original)
            (2*w//3, h//3)     # Right eye (adjusted from your original)
        ]
        logger.info("Using estimated eye positions")
        return estimated_eyes

    def _analyze_eye_simplified(self, image: np.ndarray, center: Tuple[int, int], 
                               label: str) -> Optional[Dict]:
        """Simplified version of your original eye analysis without SAM"""
        try:
            h, w = image.shape[:2]
            cx, cy = center
            
            # Calculate analysis region (from your original method)
            box_size = min(w, h) // 6
            x1 = max(0, cx - box_size)
            y1 = max(0, cy - box_size)
            x2 = min(w, cx + box_size)
            y2 = min(h, cy + box_size)
            
            # Validate region size
            if x2 - x1 < 30 or y2 - y1 < 30:
                logger.warning(f"Eye region too small for {label}: {x2-x1}x{y2-y1}")
                return None
            
            # Extract eye region
            eye_region = image[y1:y2, x1:x2]
            logger.info(f"Analyzing {label} region: {eye_region.shape}")
            
            # Create simplified masks using color-based segmentation
            sclera_mask, conjunctiva_mask, iris_mask = self._segment_parts_simplified(eye_region)
            
            # Perform clinical analysis using your original methods
            analysis_results = self._clinical_analysis(eye_region, sclera_mask, conjunctiva_mask)
            
            return {
                'label': label,
                'center': center,
                'bbox': (x1, y1, x2-x1, y2-y1),
                'region': eye_region,
                'analysis': analysis_results,
                'clinical_notes': self._interpret_results(analysis_results),
                'masks': {
                    'sclera': sclera_mask,
                    'conjunctiva': conjunctiva_mask,
                    'iris': iris_mask
                }
            }
            
        except Exception as e:
            logger.error(f"Error analyzing eye {label}: {e}")
            return None

    def _segment_parts_simplified(self, eye_region: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Simplified version of your original segmentation without SAM"""
        try:
            h, w = eye_region.shape[:2]
            
            # Create eye mask based on region
            eye_mask = np.ones((h, w), dtype=np.uint8) * 255
            
            # HSV conversion for color-based segmentation (from your original)
            hsv = cv2.cvtColor(eye_region, cv2.COLOR_BGR2HSV)
            
            # Detect iris (dark regions) - simplified version of your original
            gray = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
            _, dark_thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Morphological operations to clean up
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            iris_mask = cv2.morphologyEx(dark_thresh, cv2.MORPH_CLOSE, kernel)
            iris_mask = cv2.morphologyEx(iris_mask, cv2.MORPH_OPEN, kernel)
            
            # Sclera detection (white regions) - from your original
            white_mask = cv2.inRange(hsv, np.array([0, 0, 180]), np.array([180, 30, 255]))
            sclera_mask = cv2.bitwise_and(white_mask, eye_mask)
            sclera_mask = cv2.bitwise_and(sclera_mask, cv2.bitwise_not(iris_mask))
            
            # Conjunctiva detection (reddish regions) - from your original
            red_mask1 = cv2.inRange(hsv, np.array([0, 30, 30]), np.array([10, 255, 255]))
            red_mask2 = cv2.inRange(hsv, np.array([170, 30, 30]), np.array([180, 255, 255]))
            conjunctiva_mask = cv2.bitwise_or(red_mask1, red_mask2)
            conjunctiva_mask = cv2.bitwise_and(conjunctiva_mask, eye_mask)
            
            return sclera_mask, conjunctiva_mask, iris_mask
            
        except Exception as e:
            logger.error(f"Error in simplified segmentation: {e}")
            # Return empty masks
            h, w = eye_region.shape[:2]
            return (np.zeros((h, w), dtype=np.uint8), 
                    np.zeros((h, w), dtype=np.uint8), 
                    np.zeros((h, w), dtype=np.uint8))

    def _clinical_analysis(self, eye_region: np.ndarray, 
                          sclera_mask: np.ndarray, 
                          conjunctiva_mask: np.ndarray) -> Dict:
        """Your original clinical analysis methods"""
        return {
            'redness': self._calculate_redness(sclera_mask, conjunctiva_mask),
            'yellowing': self._calculate_yellowing(eye_region, sclera_mask),
            'pallor': self._calculate_pallor(eye_region, conjunctiva_mask)
        }

    def _calculate_redness(self, sclera_mask: np.ndarray, 
                          conjunctiva_mask: np.ndarray) -> Dict:
        """Your original redness calculation"""
        total_area = np.sum((sclera_mask > 0) | (conjunctiva_mask > 0))
        red_area = np.sum(conjunctiva_mask > 0)
        
        if total_area == 0:
            return {'score': 0, 'severity': 'normal', 'confidence': 0.5}
        
        score = min(100, (red_area / total_area) * 100 * 1.5)
        
        # Check for scleral involvement
        sclera_red = np.sum((sclera_mask > 0) & (conjunctiva_mask > 0))
        if sclera_red > 0:
            sclera_ratio = sclera_red / np.sum(sclera_mask > 0)
            score = min(100, score + sclera_ratio * 20)
        
        return {
            'score': round(score, 1),
            'severity': self._assess_severity('redness', score),
            'confidence': min(0.95, 0.5 + (total_area / 2000))
        }

    def _calculate_yellowing(self, eye_region: np.ndarray, 
                            sclera_mask: np.ndarray) -> Dict:
        """Your original yellowing calculation"""
        sclera_pixels = np.sum(sclera_mask > 0)
        if sclera_pixels < 50:
            return {'score': 0, 'severity': 'normal', 'confidence': 0.3}
        
        hsv = cv2.cvtColor(eye_region, cv2.COLOR_BGR2HSV)
        yellow_mask = cv2.inRange(hsv, np.array([20, 30, 80]), np.array([40, 255, 255]))
        yellow_pixels = np.sum(cv2.bitwise_and(yellow_mask, sclera_mask) > 0)
        
        score = min(100, (yellow_pixels / sclera_pixels) * 100 * 1.2)
        return {
            'score': round(score, 1),
            'severity': self._assess_severity('yellowing', score),
            'confidence': min(0.9, 0.4 + (sclera_pixels / 1500))
        }

    def _calculate_pallor(self, eye_region: np.ndarray, 
                         conjunctiva_mask: np.ndarray) -> Dict:
        """Your original pallor calculation"""
        conj_pixels = np.sum(conjunctiva_mask > 0)
        if conj_pixels < 30:
            return {'score': 0, 'severity': 'normal', 'confidence': 0.5}
        
        lab = cv2.cvtColor(eye_region, cv2.COLOR_BGR2LAB)
        l_channel = lab[:,:,0][conjunctiva_mask > 0]
        avg_lightness = np.mean(l_channel)
        score = max(0, min(100, (160 - avg_lightness) * 2))
        
        return {
            'score': round(score, 1),
            'severity': self._assess_severity('pallor', score),
            'confidence': min(0.85, 0.4 + (conj_pixels / 1000))
        }

    def _assess_severity(self, metric: str, score: float) -> str:
        """Your original severity assessment"""
        if score < self.thresholds[metric]['mild']:
            return 'normal'
        elif score < self.thresholds[metric]['moderate']:
            return 'mild'
        elif score < self.thresholds[metric]['severe']:
            return 'moderate'
        return 'severe'

    def _interpret_results(self, analysis: Dict) -> List[str]:
        """Your original results interpretation"""
        notes = []
        r = analysis['redness']
        y = analysis['yellowing']
        p = analysis['pallor']
        
        if r['severity'] != 'normal':
            notes.append(f"Conjunctival {r['severity']} hyperemia (score: {r['score']}/100)")
        if y['severity'] != 'normal':
            notes.append(f"{y['severity'].title()} scleral icterus (score: {y['score']}/100)")
        if p['severity'] != 'normal':
            notes.append(f"{p['severity'].title()} conjunctival pallor (score: {p['score']}/100)")
        
        return notes or ["No abnormal findings detected"]

    def _generate_clinical_report(self, eye_analyses: List[Dict]) -> Dict:
        """Generate clinical report adapted from your original _generate_report"""
        if not eye_analyses:
            return {
                "status": "error",
                "risk_level": "unknown",
                "concerns": ["Unable to analyze eye regions"],
                "recommendations": ["Retake image with better lighting and positioning"],
                "scores": {"redness": 0.0, "yellowing": 0.0, "pallor": 0.0}
            }
        
        # Calculate average scores
        total_redness = sum(eye['analysis']['redness']['score'] for eye in eye_analyses)
        total_yellowing = sum(eye['analysis']['yellowing']['score'] for eye in eye_analyses)
        total_pallor = sum(eye['analysis']['pallor']['score'] for eye in eye_analyses)
        
        num_eyes = len(eye_analyses)
        avg_scores = {
            "redness": round(total_redness / num_eyes, 1),
            "yellowing": round(total_yellowing / num_eyes, 1),
            "pallor": round(total_pallor / num_eyes, 1)
        }
        
        # Generate findings and recommendations using your original logic
        findings = []
        recommendations = set()
        
        for eye in eye_analyses:
            findings.extend(eye['clinical_notes'])
            r = eye['analysis']['redness']['severity']
            y = eye['analysis']['yellowing']['severity']
            p = eye['analysis']['pallor']['severity']
            
            # Your original recommendation logic
            if r == 'severe':
                recommendations.add("Urgent ophthalmology consultation recommended")
            elif r == 'moderate':
                recommendations.add("Consider topical anti-inflammatory treatment")
            if y in ('moderate', 'severe'):
                recommendations.add("Liver function tests recommended")
            if p in ('moderate', 'severe'):
                recommendations.add("Complete blood count and iron studies recommended")
        
        # Determine overall status and risk level
        severe_findings = any(
            eye['analysis'][metric]['severity'] == 'severe'
            for eye in eye_analyses
            for metric in ['redness', 'yellowing', 'pallor']
        )
        
        moderate_findings = any(
            eye['analysis'][metric]['severity'] == 'moderate'
            for eye in eye_analyses
            for metric in ['redness', 'yellowing', 'pallor']
        )
        
        if severe_findings:
            status = "significant_abnormality"
            risk_level = "high"
        elif moderate_findings:
            status = "mild_abnormality"
            risk_level = "moderate"
        else:
            status = "normal"
            risk_level = "low"
        
        # Default recommendations
        if not recommendations:
            recommendations.add("Continue routine monitoring")
        if risk_level == "high":
            recommendations.add("Seek immediate medical attention")
        elif risk_level == "moderate":
            recommendations.add("Schedule appointment with healthcare provider")
        
        return {
            "status": status,
            "risk_level": risk_level,
            "concerns": list(set(findings)),
            "recommendations": sorted(list(recommendations)),
            "scores": avg_scores
        }

    def _get_hf_explanation(self, assessment: Dict) -> str:
        """Your original AI explanation method"""
        if not self.ai_enabled:
            return "AI explanation not available"
            
        try:
            status_text = assessment['status'].replace('_', ' ').title()
            concerns_text = "; ".join(assessment['concerns'][:3])
            
            prompt = f"""Provide a brief, professional medical explanation for these eye analysis results:
            
Status: {status_text}
Key findings: {concerns_text}
Risk level: {assessment['risk_level']}

Please explain in simple terms what these findings might indicate, keeping the explanation under 80 words and reassuring where appropriate."""

            headers = {"Authorization": f"Bearer {self.hf_token}"}
            API_URL = "https://api-inference.huggingface.co/models/microsoft/DialoGPT-medium"
            payload = {
                "inputs": prompt,
                "parameters": {"max_length": 100, "temperature": 0.7}
            }
            
            response = requests.post(API_URL, headers=headers, json=payload, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    explanation = result[0].get('generated_text', '').replace(prompt, '').strip()
                    return explanation[:300] if explanation else "Analysis completed successfully."
                    
            return "Professional analysis completed. Please review the detailed findings."
            
        except Exception as e:
            logger.error(f"Error getting AI explanation: {e}")
            return "Computer vision analysis completed successfully."

    def _format_for_django_template(self, results: Dict) -> Dict:
        """Format results to match Django template expectations"""
        if not results.get("success", False):
            return results
            
        # Create the structure expected by the Django template
        formatted_results = {
            "success": True,
            "overall_assessment": results["overall_assessment"],
            "eye_data": results["eye_analyses"],  # Template expects eye_data
            "metadata": results["metadata"]
        }
        
        # Add AI explanation if present
        if "ai_explanation" in results["overall_assessment"]:
            formatted_results["ai_explanation"] = results["overall_assessment"]["ai_explanation"]
            
        return formatted_results


# Convenience functions for Django views
def analyze_uploaded_image(image_field, hf_token=None):
    """Analyze uploaded Django image field"""
    analyzer = EyeAnalyzer(hf_token=hf_token)
    return analyzer.analyze_eyes(image_field)


def analyze_image_file(file_path, hf_token=None):
    """Analyze image from file path"""
    analyzer = EyeAnalyzer(hf_token=hf_token)
    return analyzer.analyze_image_path(file_path)


# Test function for debugging
def test_analyzer(image_path):
    """Test the analyzer with debug output"""
    print(f"Testing analyzer with image: {image_path}")
    analyzer = EyeAnalyzer()
    
    # Test if image can be loaded
    import cv2
    test_image = cv2.imread(image_path)
    if test_image is None:
        print("ERROR: Cannot load image")
        return None
    else:
        print(f"Image loaded successfully: {test_image.shape}")
    
    # Run analysis
    results = analyzer.analyze_image_path(image_path)
    
    # Print results
    print("\n=== ANALYSIS RESULTS ===")
    print(json.dumps(results, indent=2, default=str))
    
    return results