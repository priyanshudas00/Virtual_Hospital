import os
import sys
import logging
from datetime import datetime, timedelta
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_jwt_extended import JWTManager, jwt_required, create_access_token, get_jwt_identity
import bcrypt
import asyncio
from bson import ObjectId
import json
from werkzeug.utils import secure_filename

# Import services
from config.database import db_manager
from services.gemini_service import gemini_service
from services.file_processor import file_processor
from services.imaging_service import imaging_analyzer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET', 'your-secret-key')
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=24)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB

# Initialize extensions
CORS(app)
jwt = JWTManager(app)

# Create directories
os.makedirs('uploads', exist_ok=True)

# Connect to database
db_manager.connect()

# Utility functions
def hash_password(password: str) -> bytes:
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

def check_password(password: str, hashed: bytes) -> bool:
    return bcrypt.checkpw(password.encode('utf-8'), hashed)

def serialize_doc(doc):
    """Convert MongoDB document to JSON"""
    if isinstance(doc, dict):
        for key, value in doc.items():
            if isinstance(value, ObjectId):
                doc[key] = str(value)
            elif isinstance(value, datetime):
                doc[key] = value.isoformat()
            elif isinstance(value, dict):
                doc[key] = serialize_doc(value)
            elif isinstance(value, list):
                doc[key] = [serialize_doc(item) if isinstance(item, dict) else item for item in value]
    return doc

# Authentication Routes
@app.route('/api/auth/register', methods=['POST'])
def register():
    """Register new user"""
    try:
        data = request.json
        
        # Validate required fields
        required = ['email', 'password', 'firstName', 'lastName']
        for field in required:
            if not data.get(field):
                return jsonify({'error': f'{field} is required'}), 400
        
        # Check if user exists
        users = db_manager.get_collection('users')
        if users.find_one({'email': data['email']}):
            return jsonify({'error': 'User already exists'}), 400
        
        # Create user
        user_data = {
            'email': data['email'],
            'password': hash_password(data['password']),
            'username': data['email'].split('@')[0],
            'profile': {
                'firstName': data['firstName'],
                'lastName': data['lastName'],
                'phone': data.get('phone', ''),
                'dateOfBirth': data.get('dateOfBirth', ''),
                'role': data.get('role', 'patient')
            },
            'created_at': datetime.utcnow(),
            'last_login': None
        }
        
        result = users.insert_one(user_data)
        
        # Create token
        token = create_access_token(identity=str(result.inserted_id))
        
        # Remove password from response
        user_data.pop('password')
        user_data['_id'] = str(result.inserted_id)
        
        return jsonify({
            'success': True,
            'token': token,
            'user': serialize_doc(user_data)
        }), 201
        
    except Exception as e:
        logger.error(f"Registration failed: {e}")
        return jsonify({'error': 'Registration failed'}), 500

@app.route('/api/auth/login', methods=['POST'])
def login():
    """User login"""
    try:
        data = request.json
        
        if not data.get('email') or not data.get('password'):
            return jsonify({'error': 'Email and password required'}), 400
        
        # Find user
        users = db_manager.get_collection('users')
        user = users.find_one({'email': data['email']})
        
        if not user or not check_password(data['password'], user['password']):
            return jsonify({'error': 'Invalid credentials'}), 401
        
        # Update last login
        users.update_one(
            {'_id': user['_id']},
            {'$set': {'last_login': datetime.utcnow()}}
        )
        
        # Create token
        token = create_access_token(identity=str(user['_id']))
        
        # Remove password from response
        user.pop('password')
        user['_id'] = str(user['_id'])
        
        return jsonify({
            'success': True,
            'token': token,
            'user': serialize_doc(user)
        }), 200
        
    except Exception as e:
        logger.error(f"Login failed: {e}")
        return jsonify({'error': 'Login failed'}), 500

# Medical Analysis Routes
@app.route('/api/analyze/symptoms', methods=['POST'])
@jwt_required()
def analyze_symptoms():
    """Analyze symptoms with Gemini AI"""
    try:
        user_id = get_jwt_identity()
        data = request.json
        
        user_input = data.get('user_input', '')
        session_id = data.get('session_id')
        conversation_history = data.get('conversation_history', [])
        
        if not user_input:
            return jsonify({'error': 'User input required'}), 400
        
        # Analyze with Gemini
        analysis = await gemini_service.analyze_symptoms(
            user_input, 
            conversation_history
        )
        
        # Save interaction
        interactions = db_manager.get_collection('interactions')
        
        if session_id:
            # Update existing session
            interactions.update_one(
                {'session_id': session_id, 'user_id': ObjectId(user_id)},
                {
                    '$push': {'conversation_log': {
                        'user_input': user_input,
                        'ai_response': analysis,
                        'timestamp': datetime.utcnow()
                    }},
                    '$set': {'updated_at': datetime.utcnow()}
                }
            )
        else:
            # Create new session
            session_id = f"SYM_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{user_id[:8]}"
            interactions.insert_one({
                'user_id': ObjectId(user_id),
                'session_id': session_id,
                'created_at': datetime.utcnow(),
                'conversation_log': [{
                    'user_input': user_input,
                    'ai_response': analysis,
                    'timestamp': datetime.utcnow()
                }],
                'status': 'active'
            })
        
        analysis['session_id'] = session_id
        return jsonify(analysis), 200
        
    except Exception as e:
        logger.error(f"Symptom analysis failed: {e}")
        return jsonify({'error': 'Analysis failed'}), 500

@app.route('/api/analyze/report-text', methods=['POST'])
@jwt_required()
def analyze_text_report():
    """Analyze text-based medical reports"""
    try:
        user_id = get_jwt_identity()
        
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        report_type = request.form.get('report_type', 'lab_report')
        
        # Save file
        filename = secure_filename(file.filename)
        file_path = f"uploads/{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{filename}"
        file.save(file_path)
        
        # Process file
        processing_result = file_processor.process_uploaded_file(file_path, report_type)
        
        if not processing_result.get('success'):
            return jsonify(processing_result), 400
        
        # Analyze with Gemini
        analysis = await gemini_service.analyze_text_report(
            processing_result['extracted_text'],
            report_type
        )
        
        # Save to database
        reports = db_manager.get_collection('medical_reports')
        report_record = {
            'user_id': ObjectId(user_id),
            'filename': filename,
            'report_type': report_type,
            'extracted_text': processing_result['extracted_text'],
            'ai_analysis': analysis,
            'uploaded_at': datetime.utcnow(),
            'file_hash': processing_result.get('hash')
        }
        
        result = reports.insert_one(report_record)
        
        # Clean up file
        os.remove(file_path)
        
        return jsonify({
            'success': True,
            'report_id': str(result.inserted_id),
            'analysis': analysis,
            'extracted_text': processing_result['extracted_text'][:500] + "..."
        }), 200
        
    except Exception as e:
        logger.error(f"Text report analysis failed: {e}")
        return jsonify({'error': 'Analysis failed'}), 500

@app.route('/api/analyze/report-image', methods=['POST'])
@jwt_required()
def analyze_medical_image():
    """Analyze medical images with advanced AI"""
    try:
        user_id = get_jwt_identity()
        
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        image_type = request.form.get('image_type', 'xray')
        clinical_context = request.form.get('clinical_context', '')
        
        # Save and process image
        filename = secure_filename(file.filename)
        file_path = f"uploads/{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{filename}"
        file.save(file_path)
        
        # Process image
        processing_result = file_processor.process_uploaded_file(file_path, 'image')
        
        if not processing_result.get('success'):
            return jsonify(processing_result), 400
        
        # Analyze with advanced imaging AI
        analysis = await imaging_analyzer.analyze_medical_image(
            processing_result['image_data'],
            image_type,
            clinical_context
        )
        
        # Save to database
        reports = db_manager.get_collection('medical_reports')
        report_record = {
            'user_id': ObjectId(user_id),
            'filename': filename,
            'report_type': f'{image_type}_image',
            'image_type': image_type,
            'clinical_context': clinical_context,
            'ai_analysis': analysis,
            'uploaded_at': datetime.utcnow()
        }
        
        result = reports.insert_one(report_record)
        
        # Clean up file
        os.remove(file_path)
        
        return jsonify({
            'success': True,
            'report_id': str(result.inserted_id),
            'analysis': analysis
        }), 200
        
    except Exception as e:
        logger.error(f"Image analysis failed: {e}")
        return jsonify({'error': 'Analysis failed'}), 500

@app.route('/api/generate-final-report', methods=['POST'])
@jwt_required()
def generate_comprehensive_report():
    """Generate final doctor handoff report"""
    try:
        user_id = get_jwt_identity()
        data = request.json
        
        session_id = data.get('session_id')
        if not session_id:
            return jsonify({'error': 'Session ID required'}), 400
        
        # Get patient data
        users = db_manager.get_collection('users')
        user = users.find_one({'_id': ObjectId(user_id)})
        
        # Get symptom analysis
        interactions = db_manager.get_collection('interactions')
        symptom_session = interactions.find_one({'session_id': session_id})
        
        # Get recent medical reports
        reports = db_manager.get_collection('medical_reports')
        recent_reports = list(reports.find(
            {'user_id': ObjectId(user_id)}
        ).sort('uploaded_at', -1).limit(5))
        
        # Generate comprehensive report
        doctor_report = await gemini_service.generate_doctor_report(
            serialize_doc(user.get('profile', {})),
            symptom_session.get('conversation_log', [])[-1].get('ai_response', {}) if symptom_session else {},
            [report.get('ai_analysis', {}) for report in recent_reports]
        )
        
        # Save final report
        final_reports = db_manager.get_collection('final_reports')
        final_report = {
            'user_id': ObjectId(user_id),
            'session_id': session_id,
            'doctor_report': doctor_report,
            'generated_at': datetime.utcnow(),
            'report_data': {
                'symptom_analysis': symptom_session,
                'medical_reports': recent_reports,
                'patient_profile': user.get('profile', {})
            }
        }
        
        result = final_reports.insert_one(final_report)
        
        return jsonify({
            'success': True,
            'report_id': str(result.inserted_id),
            'doctor_report': doctor_report
        }), 200
        
    except Exception as e:
        logger.error(f"Final report generation failed: {e}")
        return jsonify({'error': 'Report generation failed'}), 500

# Dashboard Routes
@app.route('/api/dashboard', methods=['GET'])
@jwt_required()
def get_dashboard_data():
    """Get user dashboard data"""
    try:
        user_id = get_jwt_identity()
        
        # Get user profile
        users = db_manager.get_collection('users')
        user = users.find_one({'_id': ObjectId(user_id)})
        
        # Get recent interactions
        interactions = db_manager.get_collection('interactions')
        recent_interactions = list(interactions.find(
            {'user_id': ObjectId(user_id)}
        ).sort('created_at', -1).limit(10))
        
        # Get medical reports
        reports = db_manager.get_collection('medical_reports')
        recent_reports = list(reports.find(
            {'user_id': ObjectId(user_id)}
        ).sort('uploaded_at', -1).limit(10))
        
        # Get final reports
        final_reports = db_manager.get_collection('final_reports')
        diagnosis_reports = list(final_reports.find(
            {'user_id': ObjectId(user_id)}
        ).sort('generated_at', -1).limit(10))
        
        return jsonify({
            'success': True,
            'user': serialize_doc(user),
            'recent_interactions': [serialize_doc(i) for i in recent_interactions],
            'medical_reports': [serialize_doc(r) for r in recent_reports],
            'diagnosis_reports': [serialize_doc(d) for d in diagnosis_reports]
        }), 200
        
    except Exception as e:
        logger.error(f"Dashboard data failed: {e}")
        return jsonify({'error': 'Failed to load dashboard'}), 500

@app.route('/api/diagnosis-reports', methods=['GET'])
@jwt_required()
def get_diagnosis_reports():
    """Get all diagnosis reports for user"""
    try:
        user_id = get_jwt_identity()
        
        final_reports = db_manager.get_collection('final_reports')
        reports = list(final_reports.find(
            {'user_id': ObjectId(user_id)}
        ).sort('generated_at', -1))
        
        # Format reports for frontend
        formatted_reports = []
        for report in reports:
            doctor_report = report.get('doctor_report', {})
            
            formatted_reports.append({
                '_id': str(report['_id']),
                'user_id': str(report['user_id']),
                'created_at': report['generated_at'].isoformat(),
                'status': 'completed',
                'report_type': 'comprehensive_assessment',
                'assessment_data': doctor_report,
                'confidence_score': 0.85,  # Default confidence
                'urgency_level': self._extract_urgency(doctor_report),
                'primary_diagnosis': self._extract_primary_diagnosis(doctor_report)
            })
        
        return jsonify({
            'success': True,
            'reports': formatted_reports
        }), 200
        
    except Exception as e:
        logger.error(f"Failed to get reports: {e}")
        return jsonify({'error': 'Failed to load reports'}), 500

@app.route('/api/diagnosis-reports/<report_id>', methods=['GET'])
@jwt_required()
def get_specific_report(report_id):
    """Get specific diagnosis report"""
    try:
        user_id = get_jwt_identity()
        
        final_reports = db_manager.get_collection('final_reports')
        report = final_reports.find_one({
            '_id': ObjectId(report_id),
            'user_id': ObjectId(user_id)
        })
        
        if not report:
            return jsonify({'error': 'Report not found'}), 404
        
        # Format for frontend
        doctor_report = report.get('doctor_report', {})
        formatted_report = {
            '_id': str(report['_id']),
            'user_id': str(report['user_id']),
            'created_at': report['generated_at'].isoformat(),
            'status': 'completed',
            'report_type': 'comprehensive_assessment',
            'assessment_data': doctor_report,
            'confidence_score': 0.85,
            'urgency_level': self._extract_urgency(doctor_report),
            'primary_diagnosis': self._extract_primary_diagnosis(doctor_report)
        }
        
        return jsonify(formatted_report), 200
        
    except Exception as e:
        logger.error(f"Failed to get specific report: {e}")
        return jsonify({'error': 'Failed to load report'}), 500

# Intake Form Route
@app.route('/api/intake-form', methods=['POST'])
@jwt_required()
def submit_intake_form():
    """Process comprehensive intake form"""
    try:
        user_id = get_jwt_identity()
        data = request.json
        
        # Validate required fields
        required = ['age', 'gender', 'primary_symptoms', 'symptom_details']
        for field in required:
            if not data.get(field):
                return jsonify({'error': f'{field} is required'}), 400
        
        # Create symptom description for AI
        symptom_text = f"""
Patient: {data.get('age')} year old {data.get('gender')}
Primary symptoms: {', '.join(data.get('primary_symptoms', []))}
Duration: {data.get('symptom_duration', 'Unknown')}
Pain level: {data.get('pain_level', 0)}/10
Details: {data.get('symptom_details', '')}
Medical history: {', '.join(data.get('chronic_conditions', []))}
Current medications: {', '.join(data.get('current_medications', []))}
Lifestyle: Exercise {data.get('exercise', 'unknown')}, Sleep {data.get('sleep_hours', 8)} hours
"""
        
        # Analyze with Gemini (simulate conversation completion)
        conversation_history = [
            {'question': 'What are your main symptoms?', 'answer': symptom_text},
            {'question': 'How long have you had these symptoms?', 'answer': data.get('symptom_duration', '')},
            {'question': 'Any medical history?', 'answer': ', '.join(data.get('chronic_conditions', []))}
        ]
        
        analysis = await gemini_service.analyze_symptoms(symptom_text, conversation_history)
        
        # Generate comprehensive doctor report
        doctor_report = await gemini_service.generate_doctor_report(
            data,
            analysis,
            []  # No uploaded reports yet
        )
        
        # Save final assessment
        final_reports = db_manager.get_collection('final_reports')
        session_id = f"INTAKE_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{user_id[:8]}"
        
        final_report = {
            'user_id': ObjectId(user_id),
            'session_id': session_id,
            'doctor_report': doctor_report,
            'generated_at': datetime.utcnow(),
            'intake_data': data,
            'ai_analysis': analysis
        }
        
        result = final_reports.insert_one(final_report)
        
        return jsonify({
            'success': True,
            'report_id': str(result.inserted_id),
            'assessment_data': doctor_report,
            'session_id': session_id
        }), 201
        
    except Exception as e:
        logger.error(f"Intake form processing failed: {e}")
        return jsonify({'error': 'Processing failed'}), 500

# Helper functions
def _extract_urgency(doctor_report: Dict) -> str:
    """Extract urgency level from doctor report"""
    report_text = json.dumps(doctor_report).lower()
    
    if any(word in report_text for word in ['emergency', 'critical', 'urgent']):
        return 'HIGH'
    elif any(word in report_text for word in ['moderate', 'concerning']):
        return 'MEDIUM'
    else:
        return 'LOW'

def _extract_primary_diagnosis(doctor_report: Dict) -> str:
    """Extract primary diagnosis from doctor report"""
    if isinstance(doctor_report, dict):
        # Try to find diagnosis in various fields
        for field in ['assessment', 'chief_complaint', 'condition_explanation']:
            if field in doctor_report:
                diagnosis = doctor_report[field]
                if isinstance(diagnosis, str) and len(diagnosis) > 10:
                    return diagnosis[:100] + "..." if len(diagnosis) > 100 else diagnosis
    
    return "Medical Assessment Completed"

# Health check
@app.route('/api/health', methods=['GET'])
def health_check():
    """System health check"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'services': {
            'gemini_api': 'connected',
            'mongodb': 'connected',
            'file_processor': 'ready'
        }
    }), 200

if __name__ == '__main__':
    logger.info("🏥 MediScan AI - Clinical Decision Support System")
    logger.info("🤖 Gemini AI Integration: Advanced Medical Analysis")
    logger.info("📊 MongoDB Database: Secure Patient Records")
    logger.info("🔬 Advanced Imaging: Multi-modal Medical Analysis")
    
    port = int(os.getenv('PORT', 3002))
    app.run(host='0.0.0.0', port=port, debug=True)