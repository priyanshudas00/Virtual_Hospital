import os
import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageEnhance
import PyPDF2
import magic
import hashlib
import logging
from typing import Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class MedicalFileProcessor:
    """Advanced medical file processing"""
    
    def __init__(self):
        self.supported_formats = {
            'pdf': ['.pdf'],
            'image': ['.jpg', '.jpeg', '.png', '.tiff', '.bmp'],
            'dicom': ['.dcm', '.dicom']
        }
    
    def process_uploaded_file(self, file_path: str, file_type: str) -> Dict:
        """Process uploaded medical file"""
        try:
            # Analyze file properties
            file_info = self._analyze_file(file_path)
            
            if not file_info['valid']:
                return file_info
            
            # Route to appropriate processor
            if file_info['category'] == 'pdf':
                return self._process_pdf(file_path)
            elif file_info['category'] == 'image':
                return self._process_image(file_path)
            else:
                return {'success': False, 'error': 'Unsupported file type'}
                
        except Exception as e:
            logger.error(f"File processing failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _analyze_file(self, file_path: str) -> Dict:
        """Analyze file properties"""
        try:
            if not os.path.exists(file_path):
                return {'valid': False, 'error': 'File not found'}
            
            # Get file stats
            file_size = os.path.getsize(file_path)
            if file_size > 50 * 1024 * 1024:  # 50MB limit
                return {'valid': False, 'error': 'File too large (max 50MB)'}
            
            # Detect file type
            file_ext = os.path.splitext(file_path)[1].lower()
            
            # Categorize file
            if file_ext in self.supported_formats['pdf']:
                category = 'pdf'
            elif file_ext in self.supported_formats['image']:
                category = 'image'
            elif file_ext in self.supported_formats['dicom']:
                category = 'dicom'
            else:
                category = 'unknown'
            
            return {
                'valid': True,
                'category': category,
                'size': file_size,
                'extension': file_ext,
                'hash': self._calculate_hash(file_path)
            }
            
        except Exception as e:
            return {'valid': False, 'error': str(e)}
    
    def _process_pdf(self, file_path: str) -> Dict:
        """Extract text from PDF medical reports"""
        try:
            extracted_text = ""
            
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for page in pdf_reader.pages:
                    extracted_text += page.extract_text() + "\n"
            
            # Clean extracted text
            cleaned_text = self._clean_medical_text(extracted_text)
            
            # Extract structured data
            structured_data = self._extract_medical_data(cleaned_text)
            
            return {
                'success': True,
                'extracted_text': cleaned_text,
                'structured_data': structured_data,
                'extraction_method': 'PyPDF2',
                'text_quality': self._assess_text_quality(cleaned_text)
            }
            
        except Exception as e:
            logger.error(f"PDF processing failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _process_image(self, file_path: str) -> Dict:
        """Process medical images with OCR"""
        try:
            # Load and enhance image
            image = cv2.imread(file_path)
            enhanced_image = self._enhance_medical_image(image)
            
            # Extract text using OCR
            extracted_text = pytesseract.image_to_string(
                enhanced_image,
                config='--oem 3 --psm 6'
            )
            
            # Convert to base64 for AI analysis
            _, buffer = cv2.imencode('.jpg', enhanced_image)
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            
            return {
                'success': True,
                'image_data': f"data:image/jpeg;base64,{image_base64}",
                'extracted_text': self._clean_medical_text(extracted_text),
                'image_enhanced': True,
                'ready_for_ai': True
            }
            
        except Exception as e:
            logger.error(f"Image processing failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _enhance_medical_image(self, image: np.ndarray) -> np.ndarray:
        """Enhance medical images for better analysis"""
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Apply CLAHE for contrast enhancement
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            
            # Noise reduction
            denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
            
            # Convert back to 3-channel
            if len(denoised.shape) == 2:
                enhanced_final = cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)
            else:
                enhanced_final = denoised
            
            return enhanced_final
            
        except Exception as e:
            logger.warning(f"Image enhancement failed: {e}")
            return image
    
    def _clean_medical_text(self, text: str) -> str:
        """Clean extracted medical text"""
        if not text:
            return ""
        
        # Remove excessive whitespace
        cleaned = ' '.join(text.split())
        
        # Fix common OCR errors in medical terms
        corrections = {
            'Hemog1obin': 'Hemoglobin',
            'G1ucose': 'Glucose',
            'Cho1esterol': 'Cholesterol',
            'Creatinine': 'Creatinine'
        }
        
        for error, correction in corrections.items():
            cleaned = cleaned.replace(error, correction)
        
        return cleaned
    
    def _extract_medical_data(self, text: str) -> Dict:
        """Extract structured medical data from text"""
        structured = {
            'lab_values': [],
            'vital_signs': [],
            'medications': [],
            'diagnoses': [],
            'dates': []
        }
        
        # Extract lab values using regex
        import re
        lab_pattern = r'([A-Za-z\s]+):\s*([0-9.]+)\s*([A-Za-z/%]+)'
        lab_matches = re.findall(lab_pattern, text)
        
        for match in lab_matches:
            structured['lab_values'].append({
                'parameter': match[0].strip(),
                'value': match[1],
                'unit': match[2]
            })
        
        # Extract dates
        date_pattern = r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b'
        dates = re.findall(date_pattern, text)
        structured['dates'] = list(set(dates))
        
        return structured
    
    def _assess_text_quality(self, text: str) -> Dict:
        """Assess quality of extracted text"""
        if not text:
            return {'quality': 'poor', 'score': 0.0}
        
        word_count = len(text.split())
        medical_keywords = ['patient', 'test', 'result', 'normal', 'abnormal', 'blood']
        medical_count = sum(1 for word in text.lower().split() if word in medical_keywords)
        
        quality_score = min((word_count / 50) * 0.5 + (medical_count / word_count) * 0.5, 1.0)
        
        if quality_score >= 0.8:
            quality = 'excellent'
        elif quality_score >= 0.6:
            quality = 'good'
        elif quality_score >= 0.4:
            quality = 'fair'
        else:
            quality = 'poor'
        
        return {
            'quality': quality,
            'score': quality_score,
            'word_count': word_count,
            'medical_density': medical_count / max(word_count, 1)
        }
    
    def _calculate_hash(self, file_path: str) -> str:
        """Calculate file hash for integrity"""
        try:
            hash_sha256 = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()[:16]
        except:
            return ""

# Global file processor
file_processor = MedicalFileProcessor()