"""
Medical Imaging Analysis Service
Advanced image processing and AI analysis for medical images
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import pydicom
import logging
from typing import Dict, List, Tuple, Optional, Any
import base64
import io
import magic
from datetime import datetime
import hashlib

logger = logging.getLogger(__name__)

class MedicalImageProcessor:
    """Advanced medical image processing and analysis"""
    
    def __init__(self):
        self.supported_formats = {
            'dicom': ['.dcm', '.dicom'],
            'standard': ['.jpg', '.jpeg', '.png', '.tiff', '.bmp'],
            'raw': ['.raw', '.nii', '.nifti']
        }
        
        self.image_modalities = {
            'xray': ['x-ray', 'radiograph', 'chest', 'bone'],
            'ct': ['ct', 'computed tomography', 'cat scan'],
            'mri': ['mri', 'magnetic resonance', 'fmri'],
            'ultrasound': ['ultrasound', 'sonogram', 'doppler'],
            'mammography': ['mammogram', 'breast'],
            'fluoroscopy': ['fluoroscopy', 'angiogram'],
            'nuclear': ['pet', 'spect', 'nuclear medicine']
        }
    
    def process_medical_image(self, image_data: str, metadata: Dict = None) -> Dict:
        """Process uploaded medical image for AI analysis"""
        try:
            # Decode image
            image = self._decode_image(image_data)
            if not image:
                return {'error': 'Failed to decode image'}
            
            # Detect image properties
            image_info = self._analyze_image_properties(image)
            
            # Enhance image for analysis
            enhanced_image = self._enhance_medical_image(image)
            
            # Extract DICOM metadata if applicable
            dicom_metadata = self._extract_dicom_metadata(image_data, metadata)
            
            # Prepare for AI analysis
            processed_data = {
                'original_image': image,
                'enhanced_image': enhanced_image,
                'image_info': image_info,
                'dicom_metadata': dicom_metadata,
                'processing_timestamp': datetime.utcnow().isoformat()
            }
            
            return {
                'success': True,
                'processed_data': processed_data,
                'image_quality': self._assess_image_quality(image),
                'ai_ready': True
            }
            
        except Exception as e:
            logger.error(f"Image processing failed: {e}")
            return {'error': f'Image processing failed: {str(e)}'}
    
    def _decode_image(self, image_data: str) -> Optional[Image.Image]:
        """Decode various image formats"""
        try:
            # Handle base64 data URLs
            if image_data.startswith('data:image'):
                image_data = image_data.split(',')[1]
            
            # Decode base64
            image_bytes = base64.b64decode(image_data)
            
            # Try to open as standard image
            try:
                image = Image.open(io.BytesIO(image_bytes))
                return image.convert('RGB')
            except Exception:
                # Try as DICOM
                return self._decode_dicom_image(image_bytes)
                
        except Exception as e:
            logger.error(f"Image decoding failed: {e}")
            return None
    
    def _decode_dicom_image(self, image_bytes: bytes) -> Optional[Image.Image]:
        """Decode DICOM image files"""
        try:
            # Read DICOM file
            dicom_data = pydicom.dcmread(io.BytesIO(image_bytes))
            
            # Extract pixel array
            pixel_array = dicom_data.pixel_array
            
            # Normalize to 0-255 range
            if pixel_array.max() > 255:
                pixel_array = ((pixel_array - pixel_array.min()) / 
                              (pixel_array.max() - pixel_array.min()) * 255).astype(np.uint8)
            
            # Convert to PIL Image
            if len(pixel_array.shape) == 2:  # Grayscale
                image = Image.fromarray(pixel_array, mode='L')
                return image.convert('RGB')
            else:  # Color
                return Image.fromarray(pixel_array)
                
        except Exception as e:
            logger.error(f"DICOM decoding failed: {e}")
            return None
    
    def _analyze_image_properties(self, image: Image.Image) -> Dict:
        """Analyze basic image properties"""
        return {
            'dimensions': image.size,
            'mode': image.mode,
            'format': image.format,
            'has_transparency': image.mode in ('RGBA', 'LA'),
            'estimated_modality': self._estimate_modality(image),
            'contrast_analysis': self._analyze_contrast(image),
            'noise_level': self._estimate_noise_level(image)
        }
    
    def _enhance_medical_image(self, image: Image.Image) -> Image.Image:
        """Enhance medical image for better AI analysis"""
        try:
            # Convert to numpy array for OpenCV processing
            img_array = np.array(image)
            
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            if len(img_array.shape) == 3:
                # Convert to LAB color space for better contrast enhancement
                lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                lab[:, :, 0] = clahe.apply(lab[:, :, 0])
                enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            else:
                # Grayscale image
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                enhanced = clahe.apply(img_array)
            
            # Noise reduction
            enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
            
            # Convert back to PIL Image
            return Image.fromarray(enhanced)
            
        except Exception as e:
            logger.warning(f"Image enhancement failed: {e}")
            return image  # Return original if enhancement fails
    
    def _extract_dicom_metadata(self, image_data: str, metadata: Dict = None) -> Dict:
        """Extract comprehensive DICOM metadata"""
        try:
            if not metadata:
                return {}
            
            # Try to extract DICOM metadata
            image_bytes = base64.b64decode(image_data.split(',')[1] if ',' in image_data else image_data)
            dicom_data = pydicom.dcmread(io.BytesIO(image_bytes))
            
            return {
                'patient_info': {
                    'patient_id': str(dicom_data.get('PatientID', 'Unknown')),
                    'patient_name': str(dicom_data.get('PatientName', 'Unknown')),
                    'patient_age': str(dicom_data.get('PatientAge', 'Unknown')),
                    'patient_sex': str(dicom_data.get('PatientSex', 'Unknown'))
                },
                'study_info': {
                    'study_date': str(dicom_data.get('StudyDate', 'Unknown')),
                    'study_time': str(dicom_data.get('StudyTime', 'Unknown')),
                    'study_description': str(dicom_data.get('StudyDescription', 'Unknown')),
                    'modality': str(dicom_data.get('Modality', 'Unknown')),
                    'body_part': str(dicom_data.get('BodyPartExamined', 'Unknown'))
                },
                'acquisition_info': {
                    'institution': str(dicom_data.get('InstitutionName', 'Unknown')),
                    'manufacturer': str(dicom_data.get('Manufacturer', 'Unknown')),
                    'model': str(dicom_data.get('ManufacturerModelName', 'Unknown')),
                    'slice_thickness': str(dicom_data.get('SliceThickness', 'Unknown')),
                    'pixel_spacing': str(dicom_data.get('PixelSpacing', 'Unknown'))
                },
                'image_info': {
                    'rows': int(dicom_data.get('Rows', 0)),
                    'columns': int(dicom_data.get('Columns', 0)),
                    'bits_allocated': int(dicom_data.get('BitsAllocated', 0)),
                    'photometric_interpretation': str(dicom_data.get('PhotometricInterpretation', 'Unknown'))
                }
            }
            
        except Exception as e:
            logger.warning(f"DICOM metadata extraction failed: {e}")
            return {}
    
    def _estimate_modality(self, image: Image.Image) -> str:
        """Estimate imaging modality from image characteristics"""
        # Convert to grayscale for analysis
        gray_image = image.convert('L')
        img_array = np.array(gray_image)
        
        # Analyze image characteristics
        mean_intensity = np.mean(img_array)
        std_intensity = np.std(img_array)
        
        # Simple heuristics for modality detection
        if mean_intensity < 50 and std_intensity > 40:
            return "Likely X-Ray (high contrast, dark background)"
        elif mean_intensity > 100 and std_intensity < 30:
            return "Likely MRI (moderate contrast, gray matter)"
        elif std_intensity > 50:
            return "Likely CT (variable contrast)"
        else:
            return "Unknown modality"
    
    def _analyze_contrast(self, image: Image.Image) -> Dict:
        """Analyze image contrast characteristics"""
        gray_image = image.convert('L')
        img_array = np.array(gray_image)
        
        return {
            'mean_intensity': float(np.mean(img_array)),
            'std_intensity': float(np.std(img_array)),
            'min_intensity': int(np.min(img_array)),
            'max_intensity': int(np.max(img_array)),
            'contrast_ratio': float(np.std(img_array) / np.mean(img_array)) if np.mean(img_array) > 0 else 0
        }
    
    def _estimate_noise_level(self, image: Image.Image) -> str:
        """Estimate noise level in medical image"""
        gray_image = image.convert('L')
        img_array = np.array(gray_image)
        
        # Use Laplacian variance to estimate noise
        laplacian_var = cv2.Laplacian(img_array, cv2.CV_64F).var()
        
        if laplacian_var > 1000:
            return "High noise"
        elif laplacian_var > 500:
            return "Moderate noise"
        else:
            return "Low noise"
    
    def _assess_image_quality(self, image: Image.Image) -> Dict:
        """Assess medical image quality with ML/DL compatibility scoring"""
        contrast_info = self._analyze_contrast(image)
        noise_level = self._estimate_noise_level(image)

        # Enhanced quality scoring with ML/DL weights
        quality_score = 0.0

        # Contrast scoring (ML-enhanced)
        if 30 < contrast_info['contrast_ratio'] < 100:
            quality_score += 0.4
        elif contrast_info['contrast_ratio'] >= 20:
            quality_score += 0.2

        # Noise scoring (ML-enhanced)
        if noise_level == "Low noise":
            quality_score += 0.3
        elif noise_level == "Moderate noise":
            quality_score += 0.1

        # Resolution scoring (ML-enhanced)
        total_pixels = image.size[0] * image.size[1]
        if total_pixels > 1000000:  # > 1MP
            quality_score += 0.4  # Increased weight for ML compatibility
        elif total_pixels > 500000:  # > 0.5MP
            quality_score += 0.2

        # ML/DL compatibility scoring
        ml_compatibility = self._assess_ml_compatibility(image, contrast_info, noise_level)
        quality_score += ml_compatibility * 0.3  # 30% weight for ML compatibility

        # Determine quality level with ML consideration
        if quality_score >= 0.9:
            quality_level = "Excellent (Optimal for ML/DL analysis)"
        elif quality_score >= 0.7:
            quality_level = "Good (Suitable for ML/DL analysis)"
        elif quality_score >= 0.5:
            quality_level = "Fair (May require ML preprocessing)"
        else:
            quality_level = "Poor (Requires ML preprocessing)"

        return {
            'quality_score': quality_score,
            'quality_level': quality_level,
            'diagnostic_quality': quality_score >= 0.6,
            'ml_compatibility': ml_compatibility,
            'recommendations': self._get_quality_recommendations_ml_enhanced(quality_score, noise_level, ml_compatibility)
        }

    def _assess_ml_compatibility(self, image: Image.Image, contrast_info: Dict, noise_level: str) -> float:
        """Assess ML/DL model compatibility score (0-1)"""
        compatibility_score = 0.0

        # Size compatibility (ML models prefer certain resolutions)
        width, height = image.size
        if 224 <= width <= 1024 and 224 <= height <= 1024:
            compatibility_score += 0.4
        elif 128 <= width <= 2048 and 128 <= height <= 2048:
            compatibility_score += 0.2

        # Contrast compatibility (ML models need good contrast)
        if contrast_info['contrast_ratio'] > 0.3:
            compatibility_score += 0.3
        elif contrast_info['contrast_ratio'] > 0.1:
            compatibility_score += 0.1

        # Noise compatibility (ML models sensitive to noise)
        if noise_level == "Low noise":
            compatibility_score += 0.3
        elif noise_level == "Moderate noise":
            compatibility_score += 0.1

        return min(compatibility_score, 1.0)

    def _get_quality_recommendations_ml_enhanced(self, quality_score: float, noise_level: str, ml_compatibility: float) -> List[str]:
        """Get ML-enhanced recommendations for image quality improvement"""
        recommendations = []

        if quality_score < 0.5:
            recommendations.append("Image quality is poor for ML analysis - consider retaking")

        if ml_compatibility < 0.5:
            recommendations.append("Image not optimal for ML/DL analysis - preprocessing recommended")

        if noise_level == "High noise":
            recommendations.append("High noise detected - may affect ML model accuracy")

        if quality_score < 0.7:
            recommendations.append("Image may not be optimal for automated ML analysis")
            recommendations.append("Radiologist review strongly recommended for ML-assisted diagnosis")

        if ml_compatibility >= 0.7:
            recommendations.append("Image suitable for ML/DL analysis")
        else:
            recommendations.append("Consider ML preprocessing before analysis")

        return recommendations
medical_image_processor = MedicalImageProcessor()