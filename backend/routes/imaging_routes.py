"""
Medical Imaging Analysis API Routes
Advanced image analysis with Gemini Vision and safety protocols
"""

from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
import asyncio
from datetime import datetime
from bson import ObjectId
import logging
import os
from werkzeug.utils import secure_filename

from ..services.gemini_medical_service import gemini_service
from ..services.medical_imaging_service import medical_image_processor
from ai.models.medical_imaging import imaging_analyzer
from ..config.database import db_manager

logger = logging.getLogger(__name__)

imaging_bp = Blueprint('imaging', __name__, url_prefix='/api/imaging')

@imaging_bp.route('/upload-image', methods=['POST'])
@jwt_required()
def upload_medical_image():
    """Upload and analyze medical image"""
    try:
        user_id = get_jwt_identity()
        
        # Check if file is present
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Get additional metadata
        image_type = request.form.get('image_type', 'auto')
        body_part = request.form.get('body_part', '')
        clinical_context = request.form.get('clinical_context', '')
        study_date = request.form.get('study_date', '')
        
        # Validate file type
        allowed_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.bmp', '.dcm', '.dicom'}
        file_ext = os.path.splitext(file.filename)[1].lower()
        
        if file_ext not in allowed_extensions:
            return jsonify({'error': f'Unsupported file type: {file_ext}'}), 400
        
        # Read and encode image
        file_content = file.read()
        image_data = base64.b64encode(file_content).decode('utf-8')
        
        # Process image
        processing_result = medical_image_processor.process_medical_image(
            f"data:image/{file_ext[1:]};base64,{image_data}",
            {
                'filename': file.filename,
                'image_type': image_type,
                'body_part': body_part,
                'study_date': study_date
            }
        )
        
        if not processing_result.get('success'):
            return jsonify({
                'error': 'Image processing failed',
                'details': processing_result.get('error')
            }), 400
        
        # Analyze with ML/DL models (Primary and ONLY method - 100% ML/DL)
        logger.info("Analyzing medical image with ML/DL models...")

        ml_analysis = asyncio.run(
            imaging_analyzer.analyze_medical_image(
                f"data:image/{file_ext[1:]};base64,{image_data}",
                image_type,
                clinical_context
            )
        )

        # Use ML analysis directly - no Gemini fallback to minimize API calls
        ai_analysis = ml_analysis
        analysis_method = 'ML/DL Enhanced'
        ml_dependency_score = ml_analysis.get('analysis_quality', {}).get('ml_dependency_score', 0.8)

        # If ML analysis completely fails, provide basic analysis without external API calls
        if 'error' in ml_analysis:
            logger.warning("ML analysis failed, providing basic analysis without external API calls")
            ai_analysis = {
                'image_analysis': {
                    'findings': ['Image uploaded successfully'],
                    'abnormal_findings': []
                },
                'overall_impression': 'Image processed - ML analysis unavailable',
                'confidence_score': 0.0,
                'recommendations': ['Please consult with a radiologist for detailed analysis'],
                'safety_notice': 'Automated analysis unavailable - professional review recommended'
            }
            analysis_method = 'Basic Processing Only'
            ml_dependency_score = 0.0

        # Save to database
        medical_reports_collection = db_manager.get_collection('medical_reports')

        report_data = {
            'user_id': ObjectId(user_id),
            'upload_metadata': {
                'original_filename': secure_filename(file.filename),
                'file_size': len(file_content),
                'mime_type': file.content_type,
                'upload_timestamp': datetime.utcnow(),
                'file_hash': hashlib.sha256(file_content).hexdigest()
            },
            'report_classification': {
                'type': image_type,
                'modality': self._extract_modality(image_type),
                'body_part': body_part,
                'study_date': study_date,
                'clinical_context': clinical_context
            },
            'ai_analysis': {
                'image_findings': ai_analysis.get('image_analysis', {}),
                'overall_assessment': ai_analysis.get('overall_impression', ''),
                'confidence_score': ai_analysis.get('confidence_score', 0.0),
                'processing_time': processing_result.get('processing_time', 0),
                'flags': self._extract_flags(ai_analysis),
                'recommendations': ai_analysis.get('recommendations', {}),
                'analysis_method': analysis_method,
                'ml_dependency_score': ml_dependency_score
            },
            'image_quality': processing_result.get('image_quality', {}),
            'radiologist_review': {
                'reviewed': False,
                'priority': self._determine_review_priority(ai_analysis)
            },
            'patient_communication': {
                'simplified_explanation': ai_analysis.get('patient_explanation', ''),
                'shared_with_patient': False
            }
        }
        
        result = medical_reports_collection.insert_one(report_data)
        
        logger.info(f"Medical image analyzed and saved: {result.inserted_id}")
        
        return jsonify({
            'success': True,
            'report_id': str(result.inserted_id),
            'analysis': ai_analysis,
            'image_quality': processing_result.get('image_quality'),
            'safety_notice': ai_analysis.get('safety_notice'),
            'requires_radiologist_review': self._requires_expert_review(ai_analysis)
        }), 201
        
    except Exception as e:
        logger.error(f"Medical image upload failed: {e}")
        return jsonify({
            'error': 'Image analysis failed',
            'details': str(e)
        }), 500

@imaging_bp.route('/upload-report', methods=['POST'])
@jwt_required()
def upload_text_report():
    """Upload and analyze text-based medical reports"""
    try:
        user_id = get_jwt_identity()
        
        if 'report' not in request.files:
            return jsonify({'error': 'No report file provided'}), 400
        
        file = request.files['report']
        report_type = request.form.get('report_type', 'clinical_notes')
        
        # Extract text from file
        from ..services.file_processor import file_processor
        
        # Save file temporarily
        temp_path = f"/tmp/{secure_filename(file.filename)}"
        file.save(temp_path)
        
        # Extract text
        extraction_result = file_processor.process_uploaded_file(temp_path, report_type)
        
        if not extraction_result.get('success'):
            return jsonify({
                'error': 'Text extraction failed',
                'details': extraction_result.get('error')
            }), 400
        
        # Get patient context
        users_collection = db_manager.get_collection('users')
        user_profile = users_collection.find_one({'_id': ObjectId(user_id)})
        
        patient_context = {}
        if user_profile:
            profile = user_profile.get('profile', {})
            patient_context = {
                'age': profile.get('personal_info', {}).get('age'),
                'sex': profile.get('personal_info', {}).get('biological_sex'),
                'medical_history': profile.get('medical_history', {}).get('past_conditions', []),
                'current_medications': [med.get('name') for med in profile.get('medications', [])]
            }
        
        # Analyze with Gemini (optional - can be disabled to minimize API calls)
        use_gemini = request.form.get('use_ai_analysis', 'false').lower() == 'true'

        if use_gemini:
            
            ai_analysis = asyncio.run(
                gemini_service.analyze_text_report(
                    extraction_result['extracted_text'],
                    report_type,
                    patient_context
                )
            )
        else:
            
            ai_analysis = {
                'report_analysis': {
                    'summary': 'AI analysis disabled to minimize API usage',
                    'key_findings': ['Text extracted successfully'],
                    'recommendations': ['Enable AI analysis for detailed insights']
                },
                'confidence_score': 0.0,
                'analysis_method': 'Text Extraction Only'
            }

        # Save to database
        medical_reports_collection = db_manager.get_collection('medical_reports')
        report_data = {
            'user_id': ObjectId(user_id),
            'upload_metadata': {
                'original_filename': secure_filename(file.filename),
                'file_size': len(file.read()),
                'upload_timestamp': datetime.utcnow()
            },
            'report_classification': {
                'type': report_type,
                'extraction_method': extraction_result.get('extraction_method')
            },
            'ai_analysis': {
                'text_analysis': ai_analysis.get('report_analysis', {}),
                'extracted_text': extraction_result['extracted_text'],
                'structured_data': extraction_result.get('structured_data', {}),
                'confidence_score': extraction_result.get('confidence', 0.0)
            },
            'radiologist_review': {
                'reviewed': False,
                'priority': 'routine'
            }
        }
        
        result = medical_reports_collection.insert_one(report_data)
        
        # Clean up temp file
        os.remove(temp_path)
        
        return jsonify({
            'success': True,
            'report_id': str(result.inserted_id),
            'analysis': ai_analysis,
            'extracted_text_preview': extraction_result['extracted_text'][:500] + "..." if len(extraction_result['extracted_text']) > 500 else extraction_result['extracted_text']
        }), 201
        
    except Exception as e:
        logger.error(f"Report upload failed: {e}")
        return jsonify({
            'error': 'Report analysis failed',
            'details': str(e)
        }), 500

@imaging_bp.route('/reports', methods=['GET'])
@jwt_required()
def get_user_reports():
    """Get all medical reports for user"""
    try:
        user_id = get_jwt_identity()
        
        medical_reports_collection = db_manager.get_collection('medical_reports')
        reports = list(medical_reports_collection.find(
            {'user_id': ObjectId(user_id)}
        ).sort('upload_metadata.upload_timestamp', -1).limit(50))
        
        # Serialize reports
        serialized_reports = []
        for report in reports:
            serialized_report = {
                'id': str(report['_id']),
                'filename': report.get('upload_metadata', {}).get('original_filename'),
                'type': report.get('report_classification', {}).get('type'),
                'upload_date': report.get('upload_metadata', {}).get('upload_timestamp', datetime.utcnow()).isoformat(),
                'analysis_summary': report.get('ai_analysis', {}).get('overall_assessment', ''),
                'flags': report.get('ai_analysis', {}).get('flags', []),
                'reviewed': report.get('radiologist_review', {}).get('reviewed', False)
            }
            serialized_reports.append(serialized_report)
        
        return jsonify({
            'success': True,
            'reports': serialized_reports,
            'total_count': len(serialized_reports)
        }), 200
        
    except Exception as e:
        logger.error(f"Reports retrieval failed: {e}")
        return jsonify({
            'error': 'Failed to retrieve reports',
            'details': str(e)
        }), 500

def _extract_modality(image_type: str) -> str:
    """Extract imaging modality from type"""
    modality_map = {
        'xray': 'X-Ray',
        'ct': 'CT',
        'mri': 'MRI',
        'ultrasound': 'Ultrasound',
        'mammography': 'Mammography'
    }
    
    image_type_lower = image_type.lower()
    for key, value in modality_map.items():
        if key in image_type_lower:
            return value
    
    return 'Unknown'

def _extract_flags(ai_analysis: Dict) -> List[Dict]:
    """Extract flags from AI analysis"""
    flags = []
    
    # Check for critical findings
    abnormal_findings = ai_analysis.get('image_analysis', {}).get('abnormal_findings', [])
    
    for finding in abnormal_findings:
        if finding.get('urgency') == 'immediate':
            flags.append({
                'severity': 'Critical',
                'message': finding.get('finding'),
                'location': finding.get('location'),
                'requires_immediate_attention': True
            })
        elif finding.get('severity') in ['severe', 'moderate']:
            flags.append({
                'severity': 'Warning',
                'message': finding.get('finding'),
                'location': finding.get('location'),
                'requires_immediate_attention': False
            })
    
    return flags

def _determine_review_priority(ai_analysis: Dict) -> str:
    """Determine radiologist review priority"""
    flags = _extract_flags(ai_analysis)
    
    if any(flag.get('severity') == 'Critical' for flag in flags):
        return 'stat'
    elif any(flag.get('severity') == 'Warning' for flag in flags):
        return 'urgent'
    else:
        return 'routine'

def _requires_expert_review(ai_analysis: Dict) -> bool:
    """Determine if expert radiologist review is required"""
    flags = _extract_flags(ai_analysis)
    return len(flags) > 0 or ai_analysis.get('confidence_score', 1.0) < 0.7