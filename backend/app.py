import os
import sys
import logging
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from flask_jwt_extended import JWTManager, jwt_required, create_access_token, get_jwt_identity
import bcrypt
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
from bson import ObjectId
import traceback

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import custom modules
from config.database import db_manager
from services.gemini_service import gemini_service
from services.file_processor import file_processor
from services.healthcare_finder import healthcare_finder
from models.patient import PatientModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET')
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=24)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

# Initialize extensions
CORS(app, origins=["http://localhost:5173", "https://localhost:5173"])
jwt = JWTManager(app)

# Initialize database connections
try:
    db_manager.connect_mongodb()
    db_manager.connect_redis()
    logger.info("🏥 Database connections established")
except Exception as e:
    logger.error(f"❌ Database connection failed: {e}")
    sys.exit(1)

# Initialize patient model
patient_model = PatientModel(db_manager)

# Thread pool for async operations
executor = ThreadPoolExecutor(max_workers=10)

# Utility functions
def hash_password(password: str) -> bytes:
    """Hash password using bcrypt"""
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

def check_password(password: str, hashed: bytes) -> bool:
    """Check password against hash"""
    return bcrypt.checkpw(password.encode('utf-8'), hashed)

def serialize_mongo_doc(doc):
    """Convert MongoDB document to JSON serializable format"""
    if isinstance(doc, dict):
        for key, value in doc.items():
            if isinstance(value, ObjectId):
                doc[key] = str(value)
            elif isinstance(value, datetime):
                doc[key] = value.isoformat()
            elif isinstance(value, dict):
                doc[key] = serialize_mongo_doc(value)
            elif isinstance(value, list):
                doc[key] = [serialize_mongo_doc(item) if isinstance(item, dict) else item for item in value]
    return doc

# Authentication Routes
@app.route('/api/auth/register', methods=['POST'])
def register():
    """Register new user with comprehensive profile"""
    try:
        data = request.json
        
        # Validate required fields
        required_fields = ['email', 'password', 'firstName', 'lastName']
        for field in required_fields:
            if not data.get(field):
                return jsonify({'error': f'{field} is required'}), 400
        
        # Check if user exists
        users_collection = db_manager.get_collection('users')
        if users_collection.find_one({'email': data['email']}):
            return jsonify({'error': 'User already exists'}), 400
        
        # Hash password
        hashed_password = hash_password(data['password'])
        
        # Create comprehensive user profile
        user_data = {
            'email': data['email'],
            'password': hashed_password,
            'personal_info': {
                'first_name': data['firstName'],
                'last_name': data['lastName'],
                'phone': data.get('phone', ''),
                'date_of_birth': data.get('dateOfBirth', ''),
                'gender': data.get('gender', ''),
                'address': data.get('address', ''),
                'emergency_contact': data.get('emergencyContact', '')
            },
            'medical_info': {
                'blood_type': data.get('bloodType', ''),
                'height': data.get('height', 0),
                'weight': data.get('weight', 0),
                'allergies': data.get('allergies', []),
                'chronic_conditions': data.get('chronicConditions', []),
                'current_medications': data.get('currentMedications', []),
                'family_history': data.get('familyHistory', [])
            },
            'insurance_info': {
                'provider': data.get('insuranceProvider', ''),
                'policy_number': data.get('insuranceNumber', ''),
                'coverage_type': data.get('coverageType', ''),
                'financial_capability': data.get('financialCapability', 'medium')
            },
            'preferences': {
                'language': data.get('preferredLanguage', 'English'),
                'communication_method': data.get('communicationMethod', 'email'),
                'privacy_level': data.get('privacyLevel', 'standard')
            },
            'account_info': {
                'role': data.get('role', 'patient'),
                'account_status': 'active',
                'email_verified': False,
                'profile_completion': 0.6  # Initial completion based on registration data
            },
            'created_at': datetime.utcnow(),
            'updated_at': datetime.utcnow(),
            'last_login': None
        }
        
        result = users_collection.insert_one(user_data)
        
        # Create JWT token
        access_token = create_access_token(
            identity=str(result.inserted_id),
            additional_claims={
                'email': data['email'],
                'role': user_data['account_info']['role']
            }
        )
        
        # Remove password from response
        user_data.pop('password')
        user_data['_id'] = str(result.inserted_id)
        user_data = serialize_mongo_doc(user_data)
        
        logger.info(f"New user registered: {data['email']}")
        
        return jsonify({
            'success': True,
            'token': access_token,
            'user': user_data,
            'message': 'Registration successful'
        }), 201
        
    except Exception as e:
        logger.error(f"Registration error: {e}")
        return jsonify({
            'success': False,
            'error': 'Registration failed',
            'details': str(e)
        }), 500

@app.route('/api/auth/login', methods=['POST'])
def login():
    """Authenticate user and return JWT token"""
    try:
        data = request.json
        
        if not data.get('email') or not data.get('password'):
            return jsonify({'error': 'Email and password are required'}), 400
        
        # Find user
        users_collection = db_manager.get_collection('users')
        user = users_collection.find_one({'email': data['email']})
        
        if not user or not check_password(data['password'], user['password']):
            return jsonify({'error': 'Invalid credentials'}), 401
        
        # Update last login
        users_collection.update_one(
            {'_id': user['_id']},
            {'$set': {'last_login': datetime.utcnow()}}
        )
        
        # Create JWT token
        access_token = create_access_token(
            identity=str(user['_id']),
            additional_claims={
                'email': user['email'],
                'role': user.get('account_info', {}).get('role', 'patient')
            }
        )
        
        # Remove password from response
        user.pop('password')
        user['_id'] = str(user['_id'])
        user = serialize_mongo_doc(user)
        
        logger.info(f"User logged in: {data['email']}")
        
        return jsonify({
            'success': True,
            'token': access_token,
            'user': user,
            'message': 'Login successful'
        }), 200
        
    except Exception as e:
        logger.error(f"Login error: {e}")
        return jsonify({
            'success': False,
            'error': 'Login failed',
            'details': str(e)
        }), 500

# Patient Intake Form Routes
@app.route('/api/intake-form', methods=['POST'])
@jwt_required()
def submit_intake_form():
    """Process comprehensive intake form with AI analysis"""
    try:
        user_id = get_jwt_identity()
        data = request.json
        
        logger.info(f"Processing intake form for user: {user_id}")
        
        # Save intake form to database
        intake_form = asyncio.run(patient_model.save_intake_form(user_id, data))
        
        # Prepare data for Gemini AI analysis
        patient_data = {
            'user_id': user_id,
            'age': data.get('age'),
            'gender': data.get('gender'),
            'location': data.get('location'),
            'occupation': data.get('occupation'),
            'symptoms': data.get('symptoms', []),
            'symptom_duration': data.get('symptom_duration'),
            'pain_level': data.get('pain_level', 0),
            'urgency': data.get('urgency', 'medium'),
            'medical_history': data.get('medical_history', []),
            'current_medications': data.get('current_medications', []),
            'allergies': data.get('allergies_text', ''),
            'surgeries': data.get('surgeries_text', ''),
            'family_history': data.get('family_history_text', ''),
            'sleep_hours': data.get('sleep_hours'),
            'exercise': data.get('exercise'),
            'diet': data.get('diet'),
            'smoking': data.get('smoking'),
            'alcohol': data.get('alcohol'),
            'stress_level': data.get('stress_level'),
            'insurance_provider': data.get('insurance_provider'),
            'financial_capability': data.get('financial_capability')
        }
        
        # Get AI assessment from Gemini
        logger.info("Requesting AI assessment from Gemini...")
        assessment_start_time = datetime.utcnow()
        
        ai_assessment = asyncio.run(gemini_service.get_initial_assessment(patient_data))
        
        assessment_end_time = datetime.utcnow()
        processing_time = (assessment_end_time - assessment_start_time).total_seconds()
        
        # Add processing metadata
        if 'metadata' not in ai_assessment:
            ai_assessment['metadata'] = {}
        ai_assessment['metadata']['processing_time'] = processing_time
        
        # Save AI assessment to database
        ai_report = asyncio.run(patient_model.save_ai_assessment(
            user_id, 
            str(intake_form['_id']), 
            ai_assessment,
            'initial_assessment'
        ))
        
        logger.info(f"AI assessment completed in {processing_time:.2f} seconds")
        
        return jsonify({
            'success': True,
            'intake_form_id': str(intake_form['_id']),
            'ai_report_id': str(ai_report['_id']),
            'assessment_data': ai_assessment,
            'processing_time': processing_time,
            'message': 'Intake form processed successfully'
        }), 201
        
    except Exception as e:
        logger.error(f"Intake form processing failed: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': 'Failed to process intake form',
            'details': str(e)
        }), 500

@app.route('/api/diagnosis-reports', methods=['GET'])
@jwt_required()
def get_diagnosis_reports():
    """Get all diagnosis reports for user"""
    try:
        user_id = get_jwt_identity()
        
        # Get AI reports from database
        reports_collection = db_manager.get_collection('ai_reports')
        reports = list(reports_collection.find(
            {'user_id': ObjectId(user_id)}
        ).sort('created_at', -1).limit(50))
        
        # Serialize reports
        serialized_reports = [serialize_mongo_doc(report) for report in reports]
        
        return jsonify({
            'success': True,
            'reports': serialized_reports,
            'total_count': len(serialized_reports)
        }), 200
        
    except Exception as e:
        logger.error(f"Failed to get diagnosis reports: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to retrieve reports',
            'details': str(e)
        }), 500

@app.route('/api/diagnosis-reports/<report_id>', methods=['GET'])
@jwt_required()
def get_specific_diagnosis_report(report_id):
    """Get specific diagnosis report"""
    try:
        user_id = get_jwt_identity()
        
        reports_collection = db_manager.get_collection('ai_reports')
        report = reports_collection.find_one({
            '_id': ObjectId(report_id),
            'user_id': ObjectId(user_id)
        })
        
        if not report:
            return jsonify({'error': 'Report not found'}), 404
        
        serialized_report = serialize_mongo_doc(report)
        
        return jsonify(serialized_report), 200
        
    except Exception as e:
        logger.error(f"Failed to get specific report: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to retrieve report',
            'details': str(e)
        }), 500

# Medical Report Upload Routes
@app.route('/api/upload-medical-report', methods=['POST'])
@jwt_required()
def upload_medical_report():
    """Upload and analyze medical reports with AI"""
    try:
        user_id = get_jwt_identity()
        
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        report_type = request.form.get('type', 'general')
        description = request.form.get('description', '')
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Save file securely
        upload_dir = os.path.join('uploads', user_id)
        os.makedirs(upload_dir, exist_ok=True)
        
        filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
        file_path = os.path.join(upload_dir, filename)
        file.save(file_path)
        
        logger.info(f"File uploaded: {file_path}")
        
        # Process file and extract text
        extraction_result = file_processor.process_uploaded_file(file_path, report_type)
        
        if not extraction_result.get('success'):
            return jsonify({
                'success': False,
                'error': extraction_result.get('error', 'File processing failed')
            }), 400
        
        # Get patient context for AI analysis
        intake_collection = db_manager.get_collection('intake_forms')
        latest_intake = intake_collection.find_one(
            {'user_id': ObjectId(user_id)},
            sort=[('created_at', -1)]
        )
        
        reports_collection = db_manager.get_collection('ai_reports')
        latest_assessment = reports_collection.find_one(
            {'user_id': ObjectId(user_id)},
            sort=[('created_at', -1)]
        )
        
        # Prepare context for AI analysis
        patient_context = {}
        if latest_intake:
            patient_context = {
                'age': latest_intake.get('basic_info', {}).get('age'),
                'gender': latest_intake.get('basic_info', {}).get('gender'),
                'current_symptoms': latest_intake.get('symptoms', {}).get('primary_symptoms', []),
                'medical_history': latest_intake.get('medical_history', {}),
                'current_medications': latest_intake.get('medical_history', {}).get('current_medications', [])
            }
        
        previous_assessment = latest_assessment.get('assessment_data') if latest_assessment else None
        
        # Analyze report with Gemini AI
        report_data = {
            'extracted_text': extraction_result['extracted_text'],
            'structured_data': extraction_result.get('structured_data', {}),
            'report_type': report_type,
            'upload_date': datetime.utcnow().isoformat(),
            'upload_id': str(ObjectId())
        }
        
        logger.info("Analyzing medical report with Gemini AI...")
        ai_analysis = asyncio.run(gemini_service.analyze_medical_report(
            report_data, 
            patient_context, 
            previous_assessment
        ))
        
        # Save upload record with AI analysis
        uploads_collection = db_manager.get_collection('medical_uploads')
        upload_record = {
            'user_id': ObjectId(user_id),
            'filename': filename,
            'original_filename': file.filename,
            'file_path': file_path,
            'report_type': report_type,
            'description': description,
            'extraction_result': extraction_result,
            'ai_analysis': ai_analysis,
            'file_metadata': {
                'size': os.path.getsize(file_path),
                'mime_type': file.content_type,
                'upload_timestamp': datetime.utcnow()
            },
            'created_at': datetime.utcnow(),
            'status': 'processed'
        }
        
        upload_result = uploads_collection.insert_one(upload_record)
        
        logger.info(f"Medical report processed and analyzed: {upload_result.inserted_id}")
        
        return jsonify({
            'success': True,
            'upload_id': str(upload_result.inserted_id),
            'extracted_text': extraction_result['extracted_text'][:500] + "..." if len(extraction_result['extracted_text']) > 500 else extraction_result['extracted_text'],
            'ai_analysis': ai_analysis,
            'file_info': {
                'filename': filename,
                'size': upload_record['file_metadata']['size'],
                'type': report_type
            },
            'message': 'Medical report uploaded and analyzed successfully'
        }), 201
        
    except Exception as e:
        logger.error(f"Medical report upload failed: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': 'Upload and analysis failed',
            'details': str(e)
        }), 500

# Healthcare Provider Search Routes
@app.route('/api/find-healthcare', methods=['POST'])
@jwt_required()
def find_healthcare_providers():
    """Find healthcare providers based on patient needs"""
    try:
        user_id = get_jwt_identity()
        data = request.json
        
        # Get latest medical assessment for context
        reports_collection = db_manager.get_collection('ai_reports')
        latest_assessment = reports_collection.find_one(
            {'user_id': ObjectId(user_id)},
            sort=[('created_at', -1)]
        )
        
        medical_profile = {}
        if latest_assessment:
            assessment_data = latest_assessment.get('assessment_data', {})
            medical_profile = {
                'primary_diagnosis': assessment_data.get('preliminary_assessment', {}).get('primary_diagnosis'),
                'urgency_level': assessment_data.get('preliminary_assessment', {}).get('urgency_level'),
                'specialist_needed': assessment_data.get('referral_recommendations', {}).get('specialist_needed', []),
                'financial_capability': data.get('financial_capability', 'medium'),
                'location': data.get('location', '')
            }
        
        # Search for healthcare providers
        logger.info(f"Searching healthcare providers for user: {user_id}")
        
        search_results = asyncio.run(healthcare_finder.find_providers(
            data.get('location', ''),
            medical_profile,
            data.get('financial_capability', 'medium'),
            data.get('urgency', 'routine')
        ))
        
        if not search_results.get('success'):
            return jsonify(search_results), 400
        
        # Get provider recommendations from Gemini AI
        if search_results['providers']:
            logger.info("Getting AI provider recommendations...")
            ai_recommendations = asyncio.run(gemini_service.get_provider_recommendations(
                medical_profile,
                search_results['providers']
            ))
            search_results['ai_recommendations'] = ai_recommendations
        
        logger.info(f"Found {search_results.get('total_found', 0)} healthcare providers")
        
        return jsonify(search_results), 200
        
    except Exception as e:
        logger.error(f"Healthcare provider search failed: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': 'Provider search failed',
            'details': str(e)
        }), 500

# Dashboard Routes
@app.route('/api/dashboard', methods=['GET'])
@jwt_required()
def get_dashboard_data():
    """Get comprehensive dashboard data for patient"""
    try:
        user_id = get_jwt_identity()
        
        # Get recent intake forms
        intake_collection = db_manager.get_collection('intake_forms')
        recent_forms = list(intake_collection.find(
            {'user_id': ObjectId(user_id)}
        ).sort('created_at', -1).limit(10))
        
        # Get AI reports
        reports_collection = db_manager.get_collection('ai_reports')
        ai_reports = list(reports_collection.find(
            {'user_id': ObjectId(user_id)}
        ).sort('created_at', -1).limit(20))
        
        # Get medical uploads
        uploads_collection = db_manager.get_collection('medical_uploads')
        medical_uploads = list(uploads_collection.find(
            {'user_id': ObjectId(user_id)}
        ).sort('created_at', -1).limit(15))
        
        # Get user profile
        users_collection = db_manager.get_collection('users')
        user_profile = users_collection.find_one({'_id': ObjectId(user_id)})
        
        # Calculate health metrics
        health_metrics = calculate_health_metrics(ai_reports, medical_uploads)
        
        # Serialize all data
        dashboard_data = {
            'user_profile': serialize_mongo_doc(user_profile) if user_profile else None,
            'recent_forms': [serialize_mongo_doc(form) for form in recent_forms],
            'ai_reports': [serialize_mongo_doc(report) for report in ai_reports],
            'medical_uploads': [serialize_mongo_doc(upload) for upload in medical_uploads],
            'health_metrics': health_metrics,
            'summary': {
                'total_assessments': len(ai_reports),
                'total_uploads': len(medical_uploads),
                'last_assessment': ai_reports[0]['created_at'].isoformat() if ai_reports else None,
                'health_score': health_metrics.get('overall_score', 75),
                'risk_level': health_metrics.get('risk_level', 'Low')
            }
        }
        
        return jsonify({
            'success': True,
            'data': dashboard_data
        }), 200
        
    except Exception as e:
        logger.error(f"Dashboard data retrieval failed: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to load dashboard data',
            'details': str(e)
        }), 500

def calculate_health_metrics(ai_reports: list, medical_uploads: list) -> Dict:
    """Calculate health metrics from patient data"""
    try:
        metrics = {
            'overall_score': 75,  # Default score
            'risk_level': 'Low',
            'trend': 'Stable',
            'last_assessment_date': None,
            'improvement_areas': [],
            'positive_indicators': []
        }
        
        if ai_reports:
            latest_report = ai_reports[0]
            assessment_data = latest_report.get('assessment_data', {})
            
            # Extract urgency level
            urgency = assessment_data.get('preliminary_assessment', {}).get('urgency_level', 'LOW')
            
            # Calculate score based on urgency and other factors
            if urgency == 'EMERGENCY':
                metrics['overall_score'] = 30
                metrics['risk_level'] = 'Critical'
            elif urgency == 'HIGH':
                metrics['overall_score'] = 50
                metrics['risk_level'] = 'High'
            elif urgency == 'MEDIUM':
                metrics['overall_score'] = 70
                metrics['risk_level'] = 'Medium'
            else:
                metrics['overall_score'] = 85
                metrics['risk_level'] = 'Low'
            
            metrics['last_assessment_date'] = latest_report['created_at'].isoformat()
            
            # Extract improvement areas
            lifestyle_recs = assessment_data.get('lifestyle_recommendations', {})
            if lifestyle_recs.get('diet_modifications'):
                metrics['improvement_areas'].append('Diet optimization')
            if lifestyle_recs.get('exercise_plan'):
                metrics['improvement_areas'].append('Physical activity')
            if lifestyle_recs.get('stress_management'):
                metrics['improvement_areas'].append('Stress management')
        
        return metrics
        
    except Exception as e:
        logger.error(f"Health metrics calculation failed: {e}")
        return {
            'overall_score': 75,
            'risk_level': 'Unknown',
            'trend': 'Stable'
        }

# Health check endpoint
@app.route('/api/health', methods=['GET'])
def health_check():
    """System health check"""
    try:
        # Test database connection
        db_manager.get_collection('users').find_one()
        
        # Test Gemini API
        gemini_status = 'configured' if gemini_service.api_key else 'not_configured'
        
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'services': {
                'mongodb': 'connected',
                'redis': 'connected' if db_manager.redis_client else 'not_available',
                'gemini_api': gemini_status,
                'file_processor': 'ready',
                'healthcare_finder': 'ready'
            },
            'version': '2.0.0'
        }), 200
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

@jwt.expired_token_loader
def expired_token_callback(jwt_header, jwt_payload):
    return jsonify({'error': 'Token has expired'}), 401

@jwt.invalid_token_loader
def invalid_token_callback(error):
    return jsonify({'error': 'Invalid token'}), 401

if __name__ == '__main__':
    # Create upload directories
    os.makedirs('uploads', exist_ok=True)
    
    logger.info("🏥" + "=" * 80)
    logger.info("🤖 AI VIRTUAL HOSPITAL - PRODUCTION BACKEND")
    logger.info("🏥" + "=" * 80)
    logger.info("🔗 Gemini AI Integration: Enabled")
    logger.info("🗄️ MongoDB Database: Connected")
    logger.info("📁 File Processing: OCR + Text Extraction")
    logger.info("🌍 Healthcare Finder: Google Maps + Database")
    logger.info("🔐 JWT Authentication: Enabled")
    logger.info("📊 Real-time Analytics: Enabled")
    logger.info("🏥" + "=" * 80)
    
    # Run Flask app
    port = int(os.getenv('PORT', 3002))
    app.run(
        host='0.0.0.0', 
        port=port, 
        debug=os.getenv('NODE_ENV') != 'production',
        threaded=True
    )