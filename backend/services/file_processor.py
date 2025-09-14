import os
import logging
import cv2
import numpy as np
import pytesseract
from PIL import Image
import PyPDF2
import fitz  # PyMuPDF
from io import BytesIO
import base64
from typing import Dict, Optional, Tuple
import magic
import hashlib

logger = logging.getLogger(__name__)

class MedicalFileProcessor:
    def __init__(self):
        self.supported_formats = {
            'pdf': ['.pdf'],
            'image': ['.jpg', '.jpeg', '.png', '.tiff', '.bmp'],
            'dicom': ['.dcm', '.dicom']
        }
        
        # OCR configuration
        self.ocr_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,:-/()[]{}+=<>?!@#$%^&*_|~`"\' '
    
    def process_uploaded_file(self, file_path: str, file_type: str = None) -> Dict:
        """Process uploaded medical file and extract text"""
        try:
            # Detect file type if not provided
            if not file_type:
                file_type = self._detect_file_type(file_path)
            
            # Validate file
            validation_result = self._validate_medical_file(file_path, file_type)
            if not validation_result['valid']:
                return validation_result
            
            # Extract text based on file type
            extracted_data = {}
            
            if file_type == 'pdf':
                extracted_data = self._extract_from_pdf(file_path)
            elif file_type == 'image':
                extracted_data = self._extract_from_image(file_path)
            elif file_type == 'dicom':
                extracted_data = self._extract_from_dicom(file_path)
            else:
                return {
                    'success': False,
                    'error': f'Unsupported file type: {file_type}'
                }
            
            # Enhance extracted data
            extracted_data.update({
                'file_path': file_path,
                'file_type': file_type,
                'file_size': os.path.getsize(file_path),
                'file_hash': self._calculate_file_hash(file_path),
                'processing_timestamp': datetime.utcnow().isoformat(),
                'extraction_method': self._get_extraction_method(file_type)
            })
            
            return extracted_data
            
        except Exception as e:
            logger.error(f"File processing failed: {e}")
            return {
                'success': False,
                'error': f'File processing failed: {str(e)}'
            }
    
    def _detect_file_type(self, file_path: str) -> str:
        """Detect file type using magic numbers"""
        try:
            mime_type = magic.from_file(file_path, mime=True)
            
            if mime_type == 'application/pdf':
                return 'pdf'
            elif mime_type.startswith('image/'):
                return 'image'
            elif 'dicom' in mime_type.lower():
                return 'dicom'
            else:
                # Fallback to extension
                ext = os.path.splitext(file_path)[1].lower()
                for file_type, extensions in self.supported_formats.items():
                    if ext in extensions:
                        return file_type
                
                return 'unknown'
                
        except Exception as e:
            logger.warning(f"File type detection failed: {e}")
            return 'unknown'
    
    def _validate_medical_file(self, file_path: str, file_type: str) -> Dict:
        """Validate uploaded medical file"""
        try:
            # Check file exists
            if not os.path.exists(file_path):
                return {'valid': False, 'error': 'File not found'}
            
            # Check file size (max 50MB)
            file_size = os.path.getsize(file_path)
            if file_size > 50 * 1024 * 1024:
                return {'valid': False, 'error': 'File too large (max 50MB)'}
            
            # Check file type
            if file_type not in self.supported_formats:
                return {'valid': False, 'error': f'Unsupported file type: {file_type}'}
            
            # Additional validation based on file type
            if file_type == 'pdf':
                if not self._validate_pdf(file_path):
                    return {'valid': False, 'error': 'Invalid or corrupted PDF file'}
            elif file_type == 'image':
                if not self._validate_image(file_path):
                    return {'valid': False, 'error': 'Invalid or corrupted image file'}
            
            return {'valid': True}
            
        except Exception as e:
            return {'valid': False, 'error': f'Validation failed: {str(e)}'}
    
    def _extract_from_pdf(self, file_path: str) -> Dict:
        """Extract text from PDF files using multiple methods"""
        extracted_text = ""
        metadata = {}
        
        try:
            # Method 1: PyMuPDF (better for complex layouts)
            doc = fitz.open(file_path)
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text()
                extracted_text += f"\n--- Page {page_num + 1} ---\n{text}"
            
            metadata = {
                'total_pages': len(doc),
                'extraction_method': 'PyMuPDF',
                'document_info': doc.metadata
            }
            
            doc.close()
            
        except Exception as e:
            logger.warning(f"PyMuPDF extraction failed: {e}")
            
            # Method 2: PyPDF2 (fallback)
            try:
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    
                    for page_num, page in enumerate(pdf_reader.pages):
                        text = page.extract_text()
                        extracted_text += f"\n--- Page {page_num + 1} ---\n{text}"
                    
                    metadata = {
                        'total_pages': len(pdf_reader.pages),
                        'extraction_method': 'PyPDF2'
                    }
                    
            except Exception as e2:
                logger.error(f"PDF extraction completely failed: {e2}")
                return {
                    'success': False,
                    'error': f'PDF extraction failed: {str(e2)}'
                }
        
        # Clean and structure the text
        cleaned_text = self._clean_medical_text(extracted_text)
        structured_data = self._structure_medical_text(cleaned_text)
        
        return {
            'success': True,
            'extracted_text': cleaned_text,
            'structured_data': structured_data,
            'metadata': metadata,
            'text_length': len(cleaned_text),
            'confidence': self._assess_extraction_quality(cleaned_text)
        }
    
    def _extract_from_image(self, file_path: str) -> Dict:
        """Extract text from medical images using OCR"""
        try:
            # Load and preprocess image
            image = cv2.imread(file_path)
            if image is None:
                return {'success': False, 'error': 'Could not load image'}
            
            # Image preprocessing for better OCR
            processed_image = self._preprocess_medical_image(image)
            
            # Extract text using Tesseract OCR
            extracted_text = pytesseract.image_to_string(
                processed_image, 
                config=self.ocr_config
            )
            
            # Get image metadata
            pil_image = Image.open(file_path)
            metadata = {
                'image_size': pil_image.size,
                'image_mode': pil_image.mode,
                'extraction_method': 'Tesseract OCR',
                'preprocessing_applied': True
            }
            
            # Clean and structure the text
            cleaned_text = self._clean_medical_text(extracted_text)
            structured_data = self._structure_medical_text(cleaned_text)
            
            return {
                'success': True,
                'extracted_text': cleaned_text,
                'structured_data': structured_data,
                'metadata': metadata,
                'text_length': len(cleaned_text),
                'confidence': self._assess_extraction_quality(cleaned_text)
            }
            
        except Exception as e:
            logger.error(f"Image OCR failed: {e}")
            return {
                'success': False,
                'error': f'Image text extraction failed: {str(e)}'
            }
    
    def _extract_from_dicom(self, file_path: str) -> Dict:
        """Extract metadata and text from DICOM files"""
        try:
            import pydicom
            
            # Read DICOM file
            dicom_data = pydicom.dcmread(file_path)
            
            # Extract metadata
            metadata = {
                'patient_name': str(dicom_data.get('PatientName', 'Unknown')),
                'study_date': str(dicom_data.get('StudyDate', 'Unknown')),
                'modality': str(dicom_data.get('Modality', 'Unknown')),
                'body_part': str(dicom_data.get('BodyPartExamined', 'Unknown')),
                'study_description': str(dicom_data.get('StudyDescription', 'Unknown')),
                'institution': str(dicom_data.get('InstitutionName', 'Unknown'))
            }
            
            # Create text summary
            extracted_text = f"""
DICOM Medical Imaging Report

Patient: {metadata['patient_name']}
Study Date: {metadata['study_date']}
Modality: {metadata['modality']}
Body Part: {metadata['body_part']}
Study Description: {metadata['study_description']}
Institution: {metadata['institution']}

Image Dimensions: {dicom_data.pixel_array.shape if hasattr(dicom_data, 'pixel_array') else 'Not available'}
"""
            
            return {
                'success': True,
                'extracted_text': extracted_text.strip(),
                'structured_data': metadata,
                'metadata': {
                    'extraction_method': 'DICOM metadata',
                    'dicom_tags': len(dicom_data)
                },
                'text_length': len(extracted_text),
                'confidence': 0.95  # High confidence for DICOM metadata
            }
            
        except Exception as e:
            logger.error(f"DICOM extraction failed: {e}")
            return {
                'success': False,
                'error': f'DICOM extraction failed: {str(e)}'
            }
    
    def _preprocess_medical_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess medical images for better OCR"""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Noise reduction
        denoised = cv2.medianBlur(gray, 3)
        
        # Contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        # Thresholding for better text recognition
        _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return thresh
    
    def _clean_medical_text(self, text: str) -> str:
        """Clean and normalize extracted medical text"""
        if not text:
            return ""
        
        # Remove excessive whitespace
        cleaned = ' '.join(text.split())
        
        # Fix common OCR errors in medical text
        medical_corrections = {
            'Hemog1obin': 'Hemoglobin',
            'Creatinine': 'Creatinine',
            'Cho1esterol': 'Cholesterol',
            'G1ucose': 'Glucose',
            'Trig1ycerides': 'Triglycerides',
            'Thyroid': 'Thyroid',
            'Vitamin': 'Vitamin',
            'Protein': 'Protein',
            'Albumin': 'Albumin',
            'Bilirubin': 'Bilirubin'
        }
        
        for error, correction in medical_corrections.items():
            cleaned = cleaned.replace(error, correction)
        
        return cleaned
    
    def _structure_medical_text(self, text: str) -> Dict:
        """Structure medical text into categories"""
        structured = {
            'lab_values': [],
            'diagnoses': [],
            'medications': [],
            'recommendations': [],
            'dates': [],
            'doctor_notes': []
        }
        
        # Extract lab values (pattern: Name: Value Unit)
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
        
        # Extract medication mentions
        med_keywords = ['tablet', 'capsule', 'mg', 'ml', 'dose', 'medication', 'drug', 'prescription']
        lines = text.split('\n')
        
        for line in lines:
            if any(keyword in line.lower() for keyword in med_keywords):
                structured['medications'].append(line.strip())
        
        return structured
    
    def _assess_extraction_quality(self, text: str) -> float:
        """Assess quality of text extraction"""
        if not text:
            return 0.0
        
        # Basic quality metrics
        word_count = len(text.split())
        char_count = len(text)
        
        # Check for medical keywords
        medical_keywords = [
            'patient', 'diagnosis', 'treatment', 'medication', 'test', 'result',
            'normal', 'abnormal', 'blood', 'urine', 'scan', 'x-ray', 'mri'
        ]
        
        medical_word_count = sum(1 for word in text.lower().split() if word in medical_keywords)
        medical_ratio = medical_word_count / max(word_count, 1)
        
        # Quality score calculation
        if word_count < 10:
            return 0.2
        elif medical_ratio > 0.1:
            return min(0.9, 0.5 + medical_ratio)
        else:
            return 0.4
    
    def _validate_pdf(self, file_path: str) -> bool:
        """Validate PDF file"""
        try:
            with open(file_path, 'rb') as file:
                PyPDF2.PdfReader(file)
            return True
        except:
            return False
    
    def _validate_image(self, file_path: str) -> bool:
        """Validate image file"""
        try:
            Image.open(file_path)
            return True
        except:
            return False
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA-256 hash of file"""
        try:
            hash_sha256 = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except:
            return ""
    
    def _get_extraction_method(self, file_type: str) -> str:
        """Get extraction method description"""
        methods = {
            'pdf': 'PyMuPDF + PyPDF2 text extraction',
            'image': 'Tesseract OCR with preprocessing',
            'dicom': 'DICOM metadata extraction'
        }
        return methods.get(file_type, 'Unknown method')

# Global file processor instance
file_processor = MedicalFileProcessor()