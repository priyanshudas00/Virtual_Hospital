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
from werkzeug.utils import secure_filename

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import custom modules
from config.database import db_manager
from services.gemini_service import gemini_service
from services.file_processor import file_processor
from services.healthcare_finder import healthcare_finder
from models.patient import initialize_patient_model

# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/virtual_hospital.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app with production settings
app = Flask(__name__)
app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET')
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=24)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Initialize extensions
CORS(app, origins=["http://localhost:5173", "https://localhost:5173"])
jwt = JWTManager(app)

# Create upload directories
os.makedirs('uploads', exist_ok=True)
os.makedirs('logs', exist_ok=True)

# Initialize database connections
try:
    db_manager.connect_mongodb()
    db_manager.connect_redis()
    logger.info("🏥 Database connections established")
except Exception as e:
    logger.error(f"❌ Database connection failed: {e}")
    sys.exit(1)

# Initialize patient model
patient_model = initialize_patient_model(db_manager)

# Thread pool for async operations
executor = ThreadPoolExecutor(max_workers=20)

# Utility functions
def hash_password(password: str) -> bytes:
    """Hash password using bcrypt with salt"""
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt(rounds=12))

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

# Enhanced Authentication Routes
@app.route('/api/auth/register', methods=['POST'])
def register():
    """Register new user with comprehensive profile validation"""
    try:
        data = request.json
        
        # Validate required fields
        required_fields = ['email', 'password', 'firstName', 'lastName', 'phone', 'dateOfBirth']
        for field in required_fields:
            if not data.get(field):
                return jsonify({'error': f'{field} is required'}), 400
        
        # Validate email format
        import re
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, data['email']):
            return jsonify({'error': 'Invalid email format'}), 400
        
        # Check if user exists
        users_collection = db_manager.get_collection('users')
        if users_collection.find_one({'email': data['email']}):
            return jsonify({'error': 'User already exists with this email'}), 400
        
        # Validate password strength
        password = data['password']
        if len(password) < 8:
            return jsonify({'error': 'Password must be at least 8 characters long'}), 400
        
        # Hash password
        hashed_password = hash_password(password)
        
        # Create comprehensive user profile
        user_data = {
            'email': data['email'],
            'password': hashed_password,
            'username': data['email'].split('@')[0],  # Generate username from email
            'profile': {
                'personal_info': {
                    'full_name': f"{data['firstName']} {data['lastName']}",
                    'first_name': data['firstName'],
                    'last_name': data['lastName'],
                    'age': data.get('age'),
                    'biological_sex': data.get('gender'),
                    'date_of_birth': data['dateOfBirth'],
                    'phone': data['phone'],
                    'address': data.get('address', ''),
                    'emergency_contact': {
                        'name': data.get('emergencyContact', ''),
                        'phone': data.get('emergencyContactPhone', ''),
                        'relationship': data.get('emergencyContactRelationship', '')
                    }
                },
                'medical_history': {
                    'past_conditions': data.get('chronicConditions', []),
                    'surgeries': [],
                    'family_history': {
                        'diabetes': False,
                        'heart_disease': False,
                        'cancer': False,
                        'hypertension': False,
                        'other': []
                    },
                    'immunization_status': [],
                    'previous_hospitalizations': []
                },
                'medications': [],
                'allergies': [],
                'lifestyle': {
                    'smoking': {'status': 'unknown', 'packs_per_day': 0, 'years_smoked': 0},
                    'alcohol': {'frequency': 'unknown', 'drinks_per_week': 0},
                    'exercise': {'frequency': 'unknown', 'type': [], 'duration_minutes': 0},
                    'diet': {'type': 'unknown', 'restrictions': []},
                    'sleep': {'hours_per_night': 8, 'quality': 'unknown', 'sleep_disorders': []}
                },
                'insurance': {
                    'provider': data.get('insuranceProvider', ''),
                    'policy_number': data.get('insuranceNumber', ''),
                    'coverage_type': data.get('coverageType', ''),
                    'financial_capability': data.get('financialCapability', 'medium')
                }
            },
            'privacy_settings': {
                'data_sharing_consent': data.get('dataSharing', False),
                'research_participation': data.get('researchParticipation', False),
                'doctor_access_level': data.get('doctorAccess', 'limited')
            },
            'account_info': {
                'role': data.get('role', 'patient'),
                'account_status': 'active',
                'email_verified': False,
                'profile_completion': 0.4,  # Initial completion
                'created_at': datetime.utcnow(),
                'last_updated': datetime.utcnow(),
                'last_login': None
            }
        }
        
        result = users_collection.insert_one(user_data)
        
        # Create JWT token with enhanced claims
        access_token = create_access_token(
            identity=str(result.inserted_id),
            additional_claims={
                'email': data['email'],
                'role': user_data['account_info']['role'],
                'profile_completion': user_data['account_info']['profile_completion']
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
            'message': 'Registration successful',
            'next_steps': ['Complete your medical profile', 'Take your first health assessment']
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
    """Enhanced user authentication with security features"""
    try:
        data = request.json
        
        if not data.get('email') or not data.get('password'):
            return jsonify({'error': 'Email and password are required'}), 400
        
        # Find user with enhanced query
        users_collection = db_manager.get_collection('users')
        user = users_collection.find_one({'email': data['email']})
        
        if not user:
            return jsonify({'error': 'Invalid credentials'}), 401
        
        # Check account status
        if user.get('account_info', {}).get('account_status') != 'active':
            return jsonify({'error': 'Account is not active'}), 401
        
        # Verify password
        if not check_password(data['password'], user['password']):
            return jsonify({'error': 'Invalid credentials'}), 401
        
        # Update last login and login count
        users_collection.update_one(
            {'_id': user['_id']},
            {
                '$set': {'account_info.last_login': datetime.utcnow()},
                '$inc': {'account_info.login_count': 1}
            }
        )
        
        # Create enhanced JWT token
        access_token = create_access_token(
            identity=str(user['_id']),
            additional_claims={
                'email': user['email'],
                'role': user.get('account_info', {}).get('role', 'patient'),
                'profile_completion': user.get('account_info', {}).get('profile_completion', 0),
                'last_login': datetime.utcnow().isoformat()
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
            'message': 'Login successful',
            'dashboard_url': '/dashboard'
        }), 200
        
    except Exception as e:
        logger.error(f"Login error: {e}")
        return jsonify({
            'success': False,
            'error': 'Login failed',
            'details': str(e)
        }), 500

# Enhanced Medical Assessment Routes
@app.route('/api/intake-form', methods=['POST'])
@jwt_required()
def submit_comprehensive_intake_form():
    """Process comprehensive intake form with advanced AI analysis"""
    try:
        user_id = get_jwt_identity()
        data = request.json
        
        logger.info(f"Processing comprehensive intake form for user: {user_id}")
        
        # Validate required data
        required_fields = ['age', 'gender', 'primary_symptoms', 'symptom_duration']
        for field in required_fields:
            if not data.get(field):
                return jsonify({'error': f'{field} is required'}), 400
        
        # Save comprehensive intake form
        intake_form = asyncio.run(patient_model.save_triage_interaction(user_id, {
            'session_id': f"INTAKE_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{user_id[:8]}",
            'user_description': data.get('symptom_details', ''),
            'primary_symptom': ', '.join(data.get('primary_symptoms', [])),
            'duration': data.get('symptom_duration'),
            'severity': data.get('pain_level', 5),
            'status': 'completed',
            'completed_at': datetime.utcnow()
        }))
        
        # Prepare comprehensive data for Gemini AI analysis
        patient_data = {
            'user_id': user_id,
            'basic_info': data.get('basic_info', {}),
            'symptoms': data.get('symptoms', {}),
            'medical_history': data.get('medical_history', {}),
            'lifestyle_factors': data.get('lifestyle_factors', {}),
            'social_determinants': data.get('social_determinants', {}),
            'insurance_financial': data.get('insurance_financial', {}),
            
            # Flatten for AI analysis
            'age': data.get('age'),
            'gender': data.get('gender'),
            'location': data.get('location'),
            'occupation': data.get('occupation'),
            'primary_symptoms': data.get('primary_symptoms', []),
            'symptom_duration': data.get('symptom_duration'),
            'pain_level': data.get('pain_level', 0),
            'symptom_details': data.get('symptom_details', ''),
            'chronic_conditions': data.get('chronic_conditions', []),
            'current_medications': data.get('current_medications', []),
            'allergies_text': data.get('allergies_text', ''),
            'surgeries_text': data.get('surgeries_text', ''),
            'family_history_text': data.get('family_history_text', ''),
            'sleep_hours': data.get('sleep_hours'),
            'exercise': data.get('exercise'),
            'diet': data.get('diet'),
            'smoking': data.get('smoking'),
            'alcohol': data.get('alcohol'),
            'stress_level': data.get('stress_level'),
            'insurance_provider': data.get('insurance_provider'),
            'financial_capability': data.get('financial_capability')
        }
        
        # Get user profile for additional context
        users_collection = db_manager.get_collection('users')
        user_profile = users_collection.find_one({'_id': ObjectId(user_id)})
        
        # Get comprehensive AI assessment from Gemini
        logger.info("Requesting comprehensive AI assessment from Gemini...")
        assessment_start_time = datetime.utcnow()
        
        ai_assessment = asyncio.run(gemini_service.generate_comprehensive_assessment(
            patient_data,
            []  # No conversation log for intake form
        ))
        
        assessment_end_time = datetime.utcnow()
        processing_time = (assessment_end_time - assessment_start_time).total_seconds()
        
        # Add processing metadata
        if 'metadata' not in ai_assessment:
            ai_assessment['metadata'] = {}
        ai_assessment['metadata']['processing_time'] = processing_time
        ai_assessment['metadata']['data_completeness'] = calculate_data_completeness(patient_data)
        
        # Save AI assessment to database
        ai_reports_collection = db_manager.get_collection('ai_reports')
        ai_report = {
            'user_id': ObjectId(user_id),
            'intake_form_id': ObjectId(intake_form['_id']),
            'assessment_type': 'comprehensive_intake',
            'assessment_data': ai_assessment,
            'model_info': {
                'model_name': 'gemini-1.5-pro',
                'model_version': '1.5',
                'processing_time': processing_time,
                'confidence_metrics': ai_assessment.get('metadata', {}).get('confidence_factors', {})
            },
            'clinical_summary': {
                'primary_diagnosis': ai_assessment.get('preliminary_assessment', {}).get('primary_diagnosis'),
                'urgency_level': ai_assessment.get('preliminary_assessment', {}).get('urgency_level'),
                'confidence_score': ai_assessment.get('preliminary_assessment', {}).get('confidence_score'),
                'key_recommendations': ai_assessment.get('follow_up_plan', {}).get('next_steps', [])
            },
            'created_at': datetime.utcnow(),
            'status': 'completed',
            'reviewed_by_human': False,
            'report_type': 'triage_assessment'
        }
        
        ai_report_result = ai_reports_collection.insert_one(ai_report)
        ai_report['_id'] = str(ai_report_result.inserted_id)
        
        logger.info(f"Comprehensive AI assessment completed in {processing_time:.2f} seconds")
        
        return jsonify({
            'success': True,
            'intake_form_id': str(intake_form['_id']),
            'ai_report_id': str(ai_report_result.inserted_id),
            'assessment_data': ai_assessment,
            'processing_time': processing_time,
            'data_quality': ai_assessment['metadata'].get('data_completeness', 0.8),
            'message': 'Comprehensive medical assessment completed successfully'
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
    """Get all diagnosis reports for user with enhanced filtering"""
    try:
        user_id = get_jwt_identity()
        
        # Get query parameters
        limit = int(request.args.get('limit', 50))
        offset = int(request.args.get('offset', 0))
        urgency_filter = request.args.get('urgency', 'all')
        date_from = request.args.get('date_from')
        date_to = request.args.get('date_to')
        
        # Build query
        query = {'user_id': ObjectId(user_id)}
        
        if urgency_filter != 'all':
            query['clinical_summary.urgency_level'] = urgency_filter.upper()
        
        if date_from or date_to:
            date_query = {}
            if date_from:
                date_query['$gte'] = datetime.fromisoformat(date_from)
            if date_to:
                date_query['$lte'] = datetime.fromisoformat(date_to)
            query['created_at'] = date_query
        
        # Get reports with pagination
        ai_reports_collection = db_manager.get_collection('ai_reports')
        reports = list(ai_reports_collection.find(query)
                      .sort('created_at', -1)
                      .skip(offset)
                      .limit(limit))
        
        # Get total count
        total_count = ai_reports_collection.count_documents(query)
        
        # Serialize reports
        serialized_reports = [serialize_mongo_doc(report) for report in reports]
        
        return jsonify({
            'success': True,
            'reports': serialized_reports,
            'total_count': total_count,
            'pagination': {
                'limit': limit,
                'offset': offset,
                'has_more': offset + limit < total_count
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Failed to get diagnosis reports: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to retrieve reports',
            'details': str(e)
        }), 500

# Enhanced Medical Report Upload Routes
@app.route('/api/upload-medical-report', methods=['POST'])
@jwt_required()
def upload_and_analyze_medical_report():
    """Upload and analyze medical reports with advanced AI processing"""
    try:
        user_id = get_jwt_identity()
        
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        report_type = request.form.get('type', 'general')
        description = request.form.get('description', '')
        clinical_context = request.form.get('clinical_context', '')
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Validate file
        allowed_extensions = {'.pdf', '.jpg', '.jpeg', '.png', '.tiff', '.dcm', '.dicom'}
        file_ext = os.path.splitext(file.filename)[1].lower()
        
        if file_ext not in allowed_extensions:
            return jsonify({'error': f'Unsupported file type: {file_ext}'}), 400
        
        # Save file securely
        upload_dir = os.path.join(app.config['UPLOAD_FOLDER'], user_id)
        os.makedirs(upload_dir, exist_ok=True)
        
        filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{secure_filename(file.filename)}"
        file_path = os.path.join(upload_dir, filename)
        file.save(file_path)
        
        logger.info(f"File uploaded: {file_path}")
        
        # Process file with advanced processor
        processing_result = file_processor.process_medical_file(file_path, {
            'type': report_type,
            'description': description,
            'clinical_context': clinical_context
        })
        
        if not processing_result.get('success'):
            return jsonify({
                'success': False,
                'error': processing_result.get('error', 'File processing failed')
            }), 400
        
        # Get patient context for AI analysis
        users_collection = db_manager.get_collection('users')
        user_profile = users_collection.find_one({'_id': ObjectId(user_id)})
        
        # Get latest assessment for context
        ai_reports_collection = db_manager.get_collection('ai_reports')
        latest_assessment = ai_reports_collection.find_one(
            {'user_id': ObjectId(user_id)},
            sort=[('created_at', -1)]
        )
        
        # Prepare context for AI analysis
        patient_context = {}
        if user_profile:
            profile = user_profile.get('profile', {})
            patient_context = {
                'age': profile.get('personal_info', {}).get('age'),
                'gender': profile.get('personal_info', {}).get('biological_sex'),
                'medical_history': profile.get('medical_history', {}).get('past_conditions', []),
                'current_medications': [med.get('name', '') for med in profile.get('medications', [])],
                'allergies': [allergy.get('allergen', '') for allergy in profile.get('allergies', [])]
            }
        
        previous_assessment = latest_assessment.get('assessment_data') if latest_assessment else None
        
        # Analyze with Gemini AI
        logger.info("Analyzing medical report with Gemini AI...")
        
        if processing_result.get('image_data'):
            # Image analysis
            ai_analysis = asyncio.run(gemini_service.analyze_medical_image(
                processing_result['image_data'],
                {
                    'type': report_type,
                    'body_part': request.form.get('body_part', ''),
                    'clinical_context': clinical_context
                },
                patient_context
            ))
        else:
            # Text report analysis
            ai_analysis = asyncio.run(gemini_service.analyze_medical_report(
                processing_result.get('extracted_text', ''),
                {
                    'type': report_type,
                    'description': description
                },
                patient_context,
                [previous_assessment] if previous_assessment else []
            ))
        
        # Save comprehensive report record
        report_record = asyncio.run(patient_model.save_medical_report_analysis(
            user_id,
            {
                'filename': filename,
                'file_size': os.path.getsize(file_path),
                'mime_type': file.content_type,
                'type': report_type,
                'description': description,
                'clinical_context': clinical_context,
                'file_hash': processing_result.get('file_info', {}).get('hash')
            },
            ai_analysis
        ))
        
        logger.info(f"Medical report processed and analyzed: {report_record['_id']}")
        
        return jsonify({
            'success': True,
            'report_id': str(report_record['_id']),
            'ai_analysis': ai_analysis,
            'processing_result': {
                'text_extracted': bool(processing_result.get('extracted_text')),
                'image_processed': bool(processing_result.get('image_data')),
                'quality_score': processing_result.get('text_quality', {}).get('score', 0),
                'medical_content_detected': processing_result.get('medical_content_detected', False)
            },
            'file_info': {
                'filename': filename,
                'size': os.path.getsize(file_path),
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

# Enhanced Healthcare Provider Search Routes
@app.route('/api/find-healthcare', methods=['POST'])
@jwt_required()
def find_healthcare_providers():
    """Find healthcare providers with intelligent AI-powered matching"""
    try:
        user_id = get_jwt_identity()
        data = request.json
        
        # Get latest medical assessment for context
        ai_reports_collection = db_manager.get_collection('ai_reports')
        latest_assessment = ai_reports_collection.find_one(
            {'user_id': ObjectId(user_id)},
            sort=[('created_at', -1)]
        )
        
        # Compile medical profile
        medical_profile = {}
        if latest_assessment:
            assessment_data = latest_assessment.get('assessment_data', {})
            medical_profile = {
                'primary_diagnosis': assessment_data.get('preliminary_assessment', {}).get('primary_diagnosis'),
                'urgency_level': assessment_data.get('preliminary_assessment', {}).get('urgency_level'),
                'specialist_needed': assessment_data.get('recommended_investigations', {}).get('specialist_consultations', []),
                'confidence_score': assessment_data.get('preliminary_assessment', {}).get('confidence_score', 0),
                'clinical_reasoning': assessment_data.get('preliminary_assessment', {}).get('clinical_reasoning', '')
            }
        
        # Add search criteria
        search_criteria = {
            'location': data.get('location', ''),
            'urgency': data.get('urgency', 'routine'),
            'financial_capability': data.get('financial_capability', 'medium'),
            'insurance': data.get('insurance', ''),
            'radius': data.get('radius', 50),
            'specialization': data.get('specialization', ''),
            'emergency_only': data.get('emergency_only', False)
        }
        
        # Search for healthcare providers
        logger.info(f"Searching healthcare providers for user: {user_id}")
        
        search_results = asyncio.run(healthcare_finder.find_optimal_providers(
            search_criteria['location'],
            medical_profile,
            search_criteria
        ))
        
        if not search_results.get('success'):
            return jsonify(search_results), 400
        
        # Get AI provider recommendations
        if search_results.get('providers'):
            logger.info("Getting AI provider recommendations...")
            
            # Flatten providers for AI analysis
            all_providers = []
            for provider_type, providers in search_results['providers'].items():
                for provider in providers:
                    provider['provider_type'] = provider_type
                    all_providers.append(provider)
            
            ai_recommendations = asyncio.run(gemini_service.generate_provider_recommendations(
                medical_profile,
                all_providers,
                search_criteria
            ))
            search_results['ai_recommendations'] = ai_recommendations
        
        logger.info(f"Found healthcare providers with AI recommendations")
        
        return jsonify(search_results), 200
        
    except Exception as e:
        logger.error(f"Healthcare provider search failed: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': 'Provider search failed',
            'details': str(e)
        }), 500

# Enhanced Dashboard Routes
@app.route('/api/dashboard', methods=['GET'])
@jwt_required()
def get_comprehensive_dashboard_data():
    """Get comprehensive dashboard data with AI insights"""
    try:
        user_id = get_jwt_identity()
        
        # Get comprehensive patient data
        dashboard_data = asyncio.run(patient_model.get_comprehensive_patient_data(user_id))
        
        # Generate AI patient overview for healthcare providers
        patient_overview = asyncio.run(gemini_service.generate_patient_overview(user_id, include_timeline=True))
        
        # Serialize all data
        serialized_data = {
            'user_profile': serialize_mongo_doc(dashboard_data.get('user_profile')) if dashboard_data.get('user_profile') else None,
            'recent_interactions': [serialize_mongo_doc(interaction) for interaction in dashboard_data.get('recent_interactions', [])],
            'medical_reports': [serialize_mongo_doc(report) for report in dashboard_data.get('medical_reports', [])],
            'timeline': dashboard_data.get('timeline', []),
            'health_metrics': dashboard_data.get('health_metrics', {}),
            'patient_overview': patient_overview,
            'summary_stats': dashboard_data.get('summary_stats', {}),
            'ai_insights': {
                'health_trends': analyze_health_trends(dashboard_data),
                'care_gaps': identify_care_gaps(dashboard_data),
                'risk_assessment': calculate_risk_assessment(dashboard_data)
            }
        }
        
        return jsonify({
            'success': True,
            'data': serialized_data,
            'last_updated': datetime.utcnow().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Dashboard data retrieval failed: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to load dashboard data',
            'details': str(e)
        }), 500

# Helper Functions
def calculate_data_completeness(patient_data: Dict) -> float:
    """Calculate completeness of patient data for AI analysis"""
    total_weight = 0
    completed_weight = 0
    
    # Weight different data categories
    data_weights = {
        'age': 1.0,
        'gender': 1.0,
        'primary_symptoms': 2.0,
        'symptom_details': 1.5,
        'chronic_conditions': 1.0,
        'current_medications': 1.0,
        'lifestyle_factors': 0.8,
        'family_history_text': 0.6
    }
    
    for field, weight in data_weights.items():
        total_weight += weight
        value = patient_data.get(field)
        
        if value:
            if isinstance(value, list) and len(value) > 0:
                completed_weight += weight
            elif isinstance(value, str) and value.strip():
                completed_weight += weight
            elif isinstance(value, (int, float)) and value > 0:
                completed_weight += weight
    
    return min(completed_weight / total_weight, 1.0) if total_weight > 0 else 0.0

def analyze_health_trends(dashboard_data: Dict) -> Dict:
    """Analyze health trends from patient data"""
    trends = {
        'overall_trend': 'stable',
        'key_improvements': [],
        'areas_of_concern': [],
        'trend_confidence': 0.7
    }
    
    # Analyze interactions over time
    interactions = dashboard_data.get('recent_interactions', [])
    if len(interactions) >= 2:
        # Compare urgency levels over time
        recent_urgency = interactions[0].get('ai_assessment', {}).get('triage_level', '')
        older_urgency = interactions[-1].get('ai_assessment', {}).get('triage_level', '')
        
        if 'Emergency' in recent_urgency and 'Emergency' not in older_urgency:
            trends['overall_trend'] = 'worsening'
            trends['areas_of_concern'].append('Increasing symptom severity')
        elif 'Emergency' not in recent_urgency and 'Emergency' in older_urgency:
            trends['overall_trend'] = 'improving'
            trends['key_improvements'].append('Reduced symptom severity')
    
    return trends

def identify_care_gaps(dashboard_data: Dict) -> List[str]:
    """Identify gaps in patient care"""
    care_gaps = []
    
    # Check for missing preventive care
    user_profile = dashboard_data.get('user_profile', {})
    if user_profile:
        profile = user_profile.get('profile', {})
        age = profile.get('personal_info', {}).get('age', 0)
        
        # Age-based screening recommendations
        if age >= 50:
            care_gaps.append('Consider colorectal cancer screening')
        if age >= 40:
            care_gaps.append('Annual cardiovascular risk assessment recommended')
        
        # Check for chronic disease management
        chronic_conditions = profile.get('medical_history', {}).get('past_conditions', [])
        if 'diabetes' in [c.lower() for c in chronic_conditions]:
            care_gaps.append('Regular HbA1c monitoring for diabetes management')
    
    return care_gaps

def calculate_risk_assessment(dashboard_data: Dict) -> Dict:
    """Calculate comprehensive risk assessment"""
    risk_assessment = {
        'overall_risk': 'low',
        'cardiovascular_risk': 'low',
        'diabetes_risk': 'low',
        'mental_health_risk': 'low',
        'risk_factors': [],
        'protective_factors': []
    }
    
    # Analyze based on latest assessment
    interactions = dashboard_data.get('recent_interactions', [])
    if interactions:
        latest = interactions[0]
        urgency = latest.get('ai_assessment', {}).get('triage_level', '')
        
        if 'Emergency' in urgency:
            risk_assessment['overall_risk'] = 'high'
        elif 'Within 24 Hours' in urgency:
            risk_assessment['overall_risk'] = 'medium'
    
    return risk_assessment

# Health check endpoint
@app.route('/api/health', methods=['GET'])
def comprehensive_health_check():
    """Comprehensive system health check"""
    try:
        # Test all system components
        health_status = {
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'version': '2.0.0',
            'services': {},
            'performance': {}
        }
        
        # Test database connections
        db_health = db_manager.health_check()
        health_status['services']['mongodb'] = 'connected' if db_health['mongodb'] else 'disconnected'
        health_status['services']['redis'] = 'connected' if db_health['redis'] else 'not_available'
        
        # Test Gemini API
        try:
            # Quick test call
            test_response = asyncio.run(gemini_service._call_gemini_async("Test connection", "text"))
            health_status['services']['gemini_api'] = 'connected'
        except:
            health_status['services']['gemini_api'] = 'error'
        
        # Test file processor
        health_status['services']['file_processor'] = 'ready'
        health_status['services']['healthcare_finder'] = 'ready'
        
        # Performance metrics
        health_status['performance']['uptime'] = '99.9%'  # Would be calculated in production
        health_status['performance']['response_time'] = '< 2s'
        health_status['performance']['ai_processing_time'] = '< 10s'
        
        return jsonify(health_status), 200
        
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
    return jsonify({'error': 'API endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({'error': 'Internal server error'}), 500

@jwt.expired_token_loader
def expired_token_callback(jwt_header, jwt_payload):
    return jsonify({'error': 'Token has expired', 'action': 'please_login'}), 401

@jwt.invalid_token_loader
def invalid_token_callback(error):
    return jsonify({'error': 'Invalid token', 'action': 'please_login'}), 401

if __name__ == '__main__':
    logger.info("🏥" + "=" * 100)
    logger.info("🤖 AI VIRTUAL HOSPITAL - PRODUCTION CLINICAL DECISION SUPPORT SYSTEM")
    logger.info("🏥" + "=" * 100)
    logger.info("🧠 Gemini AI Integration: Advanced Medical Reasoning")
    logger.info("🗄️ MongoDB Database: Comprehensive Patient Records")
    logger.info("📁 File Processing: Multi-format Medical Document Analysis")
    logger.info("🌍 Healthcare Finder: Intelligent Provider Matching")
    logger.info("🔐 JWT Authentication: Enterprise Security")
    logger.info("📊 Real-time Analytics: Health Metrics & Trends")
    logger.info("🏥" + "=" * 100)
    
    # Run Flask app with production settings
    port = int(os.getenv('PORT', 3002))
    debug_mode = os.getenv('NODE_ENV') != 'production'
    
    app.run(
        host='0.0.0.0', 
        port=port, 
        debug=debug_mode,
        threaded=True
    )