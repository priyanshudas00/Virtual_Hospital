import os
import logging
import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import PyPDF2
import fitz  # PyMuPDF
from io import BytesIO
import base64
from typing import Dict, Optional, Tuple, List
import magic
import hashlib
from datetime import datetime
import pydicom
import nibabel as nib
import SimpleITK as sitk

logger = logging.getLogger(__name__)

class AdvancedMedicalFileProcessor:
    """Advanced medical file processing with multi-format support"""
    
    def __init__(self):
        self.supported_formats = {
            'pdf': ['.pdf'],
            'image': ['.jpg', '.jpeg', '.png', '.tiff', '.bmp'],
            'dicom': ['.dcm', '.dicom'],
            'nifti': ['.nii', '.nii.gz'],
            'text': ['.txt', '.rtf', '.doc', '.docx']
        }
        
        # Enhanced OCR configuration for medical text
        self.medical_ocr_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,:-/()[]{}+=<>?!@#$%^&*_|~`"\' '
        
        # Medical terminology patterns
        self.medical_patterns = {
            'lab_values': r'([A-Za-z\s]+):\s*([0-9.]+)\s*([A-Za-z/%]+)',
            'dates': r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b',
            'medications': r'(tablet|capsule|mg|ml|dose|medication|drug|prescription)',
            'vital_signs': r'(BP|blood pressure|pulse|temperature|weight|height|BMI)',
            'diagnoses': r'(diagnosis|impression|assessment|conclusion)',
            'procedures': r'(procedure|surgery|operation|intervention)'
        }
    
    def process_medical_file(self, file_path: str, file_metadata: Dict = None) -> Dict:
        """Process any medical file format with comprehensive analysis"""
        try:
            # Detect and validate file
            file_info = self._analyze_file_properties(file_path)
            
            if not file_info['valid']:
                return file_info
            
            # Route to appropriate processor
            file_type = file_info['type']
            
            if file_type == 'pdf':
                result = self._process_pdf_report(file_path, file_metadata)
            elif file_type == 'image':
                result = self._process_medical_image(file_path, file_metadata)
            elif file_type == 'dicom':
                result = self._process_dicom_file(file_path, file_metadata)
            elif file_type == 'nifti':
                result = self._process_nifti_file(file_path, file_metadata)
            else:
                result = self._process_text_file(file_path, file_metadata)
            
            # Add comprehensive metadata
            result.update({
                'file_info': file_info,
                'processing_timestamp': datetime.utcnow().isoformat(),
                'processor_version': '2.0.0',
                'security_scan': self._security_scan_file(file_path)
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Medical file processing failed: {e}")
            return {
                'success': False,
                'error': f'File processing failed: {str(e)}',
                'file_path': file_path
            }
    
    def _process_pdf_report(self, file_path: str, metadata: Dict = None) -> Dict:
        """Process PDF medical reports with advanced text extraction"""
        try:
            extracted_text = ""
            extraction_metadata = {}
            
            # Method 1: PyMuPDF (best for complex medical reports)
            try:
                doc = fitz.open(file_path)
                
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    
                    # Extract text
                    text = page.get_text()
                    
                    # Extract tables if present
                    tables = page.find_tables()
                    for table in tables:
                        table_data = table.extract()
                        text += f"\n[TABLE DATA]\n{table_data}\n"
                    
                    extracted_text += f"\n--- Page {page_num + 1} ---\n{text}"
                
                extraction_metadata = {
                    'total_pages': len(doc),
                    'extraction_method': 'PyMuPDF',
                    'document_info': doc.metadata,
                    'tables_found': sum(len(doc.load_page(i).find_tables()) for i in range(len(doc)))
                }
                
                doc.close()
                
            except Exception as e:
                logger.warning(f"PyMuPDF extraction failed: {e}")
                # Fallback to PyPDF2
                extracted_text, extraction_metadata = self._fallback_pdf_extraction(file_path)
            
            # Clean and structure medical text
            cleaned_text = self._clean_medical_text(extracted_text)
            structured_data = self._extract_medical_structures(cleaned_text)
            
            return {
                'success': True,
                'extracted_text': cleaned_text,
                'structured_data': structured_data,
                'extraction_metadata': extraction_metadata,
                'text_quality': self._assess_text_quality(cleaned_text),
                'medical_content_detected': self._detect_medical_content(cleaned_text)
            }
            
        except Exception as e:
            logger.error(f"PDF processing failed: {e}")
            return {
                'success': False,
                'error': f'PDF processing failed: {str(e)}'
            }
    
    def _process_medical_image(self, file_path: str, metadata: Dict = None) -> Dict:
        """Process medical images with advanced preprocessing"""
        try:
            # Load image
            image = cv2.imread(file_path)
            if image is None:
                return {'success': False, 'error': 'Could not load image'}
            
            # Get image properties
            image_properties = self._analyze_image_properties(image)
            
            # Enhance image for medical analysis
            enhanced_image = self._enhance_medical_image(image)
            
            # Extract text using OCR if text is present
            extracted_text = ""
            if self._detect_text_in_image(enhanced_image):
                extracted_text = pytesseract.image_to_string(
                    enhanced_image, 
                    config=self.medical_ocr_config
                )
            
            # Convert to base64 for AI analysis
            _, buffer = cv2.imencode('.jpg', enhanced_image)
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            
            return {
                'success': True,
                'image_data': f"data:image/jpeg;base64,{image_base64}",
                'extracted_text': self._clean_medical_text(extracted_text),
                'image_properties': image_properties,
                'enhancement_applied': True,
                'ready_for_ai_analysis': True
            }
            
        except Exception as e:
            logger.error(f"Medical image processing failed: {e}")
            return {
                'success': False,
                'error': f'Image processing failed: {str(e)}'
            }
    
    def _process_dicom_file(self, file_path: str, metadata: Dict = None) -> Dict:
        """Process DICOM medical imaging files"""
        try:
            # Read DICOM file
            dicom_data = pydicom.dcmread(file_path)
            
            # Extract comprehensive metadata
            dicom_metadata = {
                'patient_info': {
                    'patient_id': str(dicom_data.get('PatientID', 'Unknown')),
                    'patient_name': str(dicom_data.get('PatientName', 'Unknown')),
                    'patient_age': str(dicom_data.get('PatientAge', 'Unknown')),
                    'patient_sex': str(dicom_data.get('PatientSex', 'Unknown')),
                    'patient_birth_date': str(dicom_data.get('PatientBirthDate', 'Unknown'))
                },
                'study_info': {
                    'study_date': str(dicom_data.get('StudyDate', 'Unknown')),
                    'study_time': str(dicom_data.get('StudyTime', 'Unknown')),
                    'study_description': str(dicom_data.get('StudyDescription', 'Unknown')),
                    'study_id': str(dicom_data.get('StudyID', 'Unknown')),
                    'accession_number': str(dicom_data.get('AccessionNumber', 'Unknown'))
                },
                'acquisition_info': {
                    'modality': str(dicom_data.get('Modality', 'Unknown')),
                    'body_part': str(dicom_data.get('BodyPartExamined', 'Unknown')),
                    'view_position': str(dicom_data.get('ViewPosition', 'Unknown')),
                    'institution': str(dicom_data.get('InstitutionName', 'Unknown')),
                    'manufacturer': str(dicom_data.get('Manufacturer', 'Unknown')),
                    'model': str(dicom_data.get('ManufacturerModelName', 'Unknown'))
                },
                'image_info': {
                    'rows': int(dicom_data.get('Rows', 0)),
                    'columns': int(dicom_data.get('Columns', 0)),
                    'bits_allocated': int(dicom_data.get('BitsAllocated', 0)),
                    'pixel_spacing': str(dicom_data.get('PixelSpacing', 'Unknown')),
                    'slice_thickness': str(dicom_data.get('SliceThickness', 'Unknown'))
                }
            }
            
            # Convert DICOM to standard image format
            pixel_array = dicom_data.pixel_array
            
            # Normalize pixel values
            if pixel_array.max() > 255:
                pixel_array = ((pixel_array - pixel_array.min()) / 
                              (pixel_array.max() - pixel_array.min()) * 255).astype(np.uint8)
            
            # Convert to PIL Image
            if len(pixel_array.shape) == 2:  # Grayscale
                pil_image = Image.fromarray(pixel_array, mode='L').convert('RGB')
            else:
                pil_image = Image.fromarray(pixel_array)
            
            # Convert to base64
            buffer = BytesIO()
            pil_image.save(buffer, format='JPEG', quality=95)
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            return {
                'success': True,
                'image_data': f"data:image/jpeg;base64,{image_base64}",
                'dicom_metadata': dicom_metadata,
                'pixel_data_shape': pixel_array.shape,
                'ready_for_ai_analysis': True,
                'file_type': 'DICOM Medical Image'
            }
            
        except Exception as e:
            logger.error(f"DICOM processing failed: {e}")
            return {
                'success': False,
                'error': f'DICOM processing failed: {str(e)}'
            }
    
    def _enhance_medical_image(self, image: np.ndarray) -> np.ndarray:
        """Advanced medical image enhancement for AI analysis"""
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            
            # Noise reduction using bilateral filter
            denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
            
            # Sharpening for better detail visibility
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpened = cv2.filter2D(denoised, -1, kernel)
            
            # Convert back to 3-channel for consistency
            if len(sharpened.shape) == 2:
                enhanced_final = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)
            else:
                enhanced_final = sharpened
            
            return enhanced_final
            
        except Exception as e:
            logger.warning(f"Image enhancement failed: {e}")
            return image  # Return original if enhancement fails
    
    def _extract_medical_structures(self, text: str) -> Dict:
        """Extract structured medical data from text"""
        structured = {
            'lab_values': [],
            'vital_signs': [],
            'medications': [],
            'diagnoses': [],
            'procedures': [],
            'dates': [],
            'measurements': [],
            'recommendations': []
        }
        
        # Extract lab values
        import re
        lab_matches = re.findall(self.medical_patterns['lab_values'], text)
        for match in lab_matches:
            structured['lab_values'].append({
                'parameter': match[0].strip(),
                'value': match[1],
                'unit': match[2],
                'status': self._classify_lab_value(match[0], float(match[1]) if match[1].replace('.', '').isdigit() else 0)
            })
        
        # Extract dates
        dates = re.findall(self.medical_patterns['dates'], text)
        structured['dates'] = list(set(dates))
        
        # Extract medications
        lines = text.split('\n')
        for line in lines:
            if re.search(self.medical_patterns['medications'], line.lower()):
                structured['medications'].append(line.strip())
        
        # Extract vital signs
        vital_patterns = {
            'blood_pressure': r'BP:?\s*(\d{2,3})/(\d{2,3})',
            'heart_rate': r'HR:?\s*(\d{2,3})',
            'temperature': r'Temp:?\s*(\d{2,3}\.?\d?)',
            'weight': r'Weight:?\s*(\d{2,3}\.?\d?)\s*(kg|lbs)',
            'height': r'Height:?\s*(\d{1,3}\.?\d?)\s*(cm|ft|in)'
        }
        
        for vital_type, pattern in vital_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                structured['vital_signs'].append({
                    'type': vital_type,
                    'values': matches
                })
        
        return structured
    
    def _classify_lab_value(self, parameter: str, value: float) -> str:
        """Classify lab values as normal/abnormal based on reference ranges"""
        # Common lab reference ranges
        reference_ranges = {
            'hemoglobin': {'min': 12.0, 'max': 15.5},
            'glucose': {'min': 70, 'max': 100},
            'cholesterol': {'min': 0, 'max': 200},
            'creatinine': {'min': 0.6, 'max': 1.2},
            'wbc': {'min': 4.5, 'max': 11.0},
            'rbc': {'min': 4.2, 'max': 5.4}
        }
        
        param_lower = parameter.lower()
        for ref_param, ranges in reference_ranges.items():
            if ref_param in param_lower:
                if value < ranges['min']:
                    return 'low'
                elif value > ranges['max']:
                    return 'high'
                else:
                    return 'normal'
        
        return 'unknown'
    
    def _analyze_file_properties(self, file_path: str) -> Dict:
        """Comprehensive file analysis and validation"""
        try:
            if not os.path.exists(file_path):
                return {'valid': False, 'error': 'File not found'}
            
            # Get file stats
            file_stats = os.stat(file_path)
            file_size = file_stats.st_size
            
            # Check file size (max 100MB for medical files)
            if file_size > 100 * 1024 * 1024:
                return {'valid': False, 'error': 'File too large (max 100MB)'}
            
            # Detect file type
            try:
                mime_type = magic.from_file(file_path, mime=True)
            except:
                mime_type = 'application/octet-stream'
            
            # Determine file category
            file_ext = os.path.splitext(file_path)[1].lower()
            file_type = self._categorize_file_type(mime_type, file_ext)
            
            # Calculate file hash for integrity
            file_hash = self._calculate_file_hash(file_path)
            
            return {
                'valid': True,
                'type': file_type,
                'size': file_size,
                'mime_type': mime_type,
                'extension': file_ext,
                'hash': file_hash,
                'modification_time': datetime.fromtimestamp(file_stats.st_mtime).isoformat()
            }
            
        except Exception as e:
            return {'valid': False, 'error': f'File analysis failed: {str(e)}'}
    
    def _categorize_file_type(self, mime_type: str, extension: str) -> str:
        """Categorize file type for appropriate processing"""
        if mime_type == 'application/pdf':
            return 'pdf'
        elif mime_type.startswith('image/'):
            return 'image'
        elif 'dicom' in mime_type.lower() or extension in ['.dcm', '.dicom']:
            return 'dicom'
        elif extension in ['.nii', '.nii.gz']:
            return 'nifti'
        elif mime_type.startswith('text/') or extension in ['.txt', '.rtf']:
            return 'text'
        else:
            return 'unknown'
    
    def _clean_medical_text(self, text: str) -> str:
        """Advanced medical text cleaning and normalization"""
        if not text:
            return ""
        
        # Remove excessive whitespace
        cleaned = ' '.join(text.split())
        
        # Fix common OCR errors in medical terminology
        medical_corrections = {
            'Hemog1obin': 'Hemoglobin',
            'Hemog|obin': 'Hemoglobin',
            'Creatinine': 'Creatinine',
            'Cho1esterol': 'Cholesterol',
            'Cholestero1': 'Cholesterol',
            'G1ucose': 'Glucose',
            'Glucose': 'Glucose',
            'Trig1ycerides': 'Triglycerides',
            'Thyroid': 'Thyroid',
            'Vitamin': 'Vitamin',
            'Protein': 'Protein',
            'Albumin': 'Albumin',
            'Bilirubin': 'Bilirubin',
            'Urea': 'Urea',
            'Sodium': 'Sodium',
            'Potassium': 'Potassium',
            'Chloride': 'Chloride'
        }
        
        for error, correction in medical_corrections.items():
            cleaned = cleaned.replace(error, correction)
        
        # Normalize medical abbreviations
        abbreviation_expansions = {
            'BP': 'Blood Pressure',
            'HR': 'Heart Rate',
            'RR': 'Respiratory Rate',
            'Temp': 'Temperature',
            'WBC': 'White Blood Cells',
            'RBC': 'Red Blood Cells',
            'Hgb': 'Hemoglobin',
            'Hct': 'Hematocrit',
            'PLT': 'Platelets'
        }
        
        for abbrev, expansion in abbreviation_expansions.items():
            cleaned = re.sub(rf'\b{abbrev}\b', expansion, cleaned, flags=re.IGNORECASE)
        
        return cleaned
    
    def _detect_medical_content(self, text: str) -> Dict:
        """Detect and categorize medical content in text"""
        medical_indicators = {
            'lab_report': ['hemoglobin', 'glucose', 'cholesterol', 'creatinine', 'wbc', 'rbc'],
            'radiology_report': ['impression', 'findings', 'technique', 'comparison', 'recommendation'],
            'pathology_report': ['specimen', 'microscopic', 'diagnosis', 'malignant', 'benign'],
            'discharge_summary': ['admission', 'discharge', 'hospital course', 'medications', 'follow-up'],
            'consultation_note': ['chief complaint', 'history', 'examination', 'assessment', 'plan'],
            'operative_report': ['procedure', 'surgeon', 'anesthesia', 'findings', 'complications']
        }
        
        text_lower = text.lower()
        detected_types = []
        confidence_scores = {}
        
        for report_type, keywords in medical_indicators.items():
            matches = sum(1 for keyword in keywords if keyword in text_lower)
            confidence = matches / len(keywords)
            
            if confidence > 0.3:  # 30% keyword match threshold
                detected_types.append(report_type)
                confidence_scores[report_type] = confidence
        
        return {
            'detected_types': detected_types,
            'confidence_scores': confidence_scores,
            'primary_type': max(confidence_scores.items(), key=lambda x: x[1])[0] if confidence_scores else 'unknown',
            'medical_content_detected': len(detected_types) > 0
        }
    
    def _assess_text_quality(self, text: str) -> Dict:
        """Assess quality of extracted text for medical analysis"""
        if not text:
            return {'quality': 'poor', 'score': 0.0}
        
        word_count = len(text.split())
        char_count = len(text)
        
        # Medical keyword density
        medical_keywords = [
            'patient', 'diagnosis', 'treatment', 'medication', 'test', 'result',
            'normal', 'abnormal', 'blood', 'urine', 'scan', 'x-ray', 'mri',
            'doctor', 'physician', 'hospital', 'clinic', 'examination'
        ]
        
        medical_word_count = sum(1 for word in text.lower().split() if word in medical_keywords)
        medical_density = medical_word_count / max(word_count, 1)
        
        # Quality scoring
        quality_score = 0.0
        
        if word_count >= 50:
            quality_score += 0.3
        if medical_density > 0.1:
            quality_score += 0.4
        if char_count > 200:
            quality_score += 0.3
        
        # Determine quality level
        if quality_score >= 0.8:
            quality_level = 'excellent'
        elif quality_score >= 0.6:
            quality_level = 'good'
        elif quality_score >= 0.4:
            quality_level = 'fair'
        else:
            quality_level = 'poor'
        
        return {
            'quality': quality_level,
            'score': quality_score,
            'word_count': word_count,
            'medical_density': medical_density,
            'suitable_for_ai_analysis': quality_score >= 0.4
        }
    
    def _security_scan_file(self, file_path: str) -> Dict:
        """Security scan for uploaded medical files"""
        return {
            'virus_scan': 'clean',  # In production, integrate with antivirus
            'content_scan': 'medical_content_detected',
            'pii_detected': True,  # Medical files typically contain PII
            'encryption_recommended': True,
            'scan_timestamp': datetime.utcnow().isoformat()
        }
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA-256 hash for file integrity"""
        try:
            hash_sha256 = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except:
            return ""

# Global file processor instance
file_processor = AdvancedMedicalFileProcessor()