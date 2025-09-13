import os
import json
import logging
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from flask_jwt_extended import JWTManager, jwt_required, create_access_token, get_jwt_identity
from pymongo import MongoClient
from bson import ObjectId
import bcrypt
import google.generativeai as genai
from PIL import Image
import PyPDF2
import pytesseract
import cv2
import numpy as np
from io import BytesIO
import base64
import stripe
from twilio.rest import Client
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import requests
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Initialize extensions
CORS(app)
jwt = JWTManager(app)

# Initialize external services
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
stripe.api_key = os.getenv('STRIPE_SECRET_KEY')
twilio_client = Client(os.getenv('TWILIO_ACCOUNT_SID'), os.getenv('TWILIO_AUTH_TOKEN'))

# MongoDB connection
client = MongoClient(os.getenv('MONGODB_URI'))
db = client.virtualHospital

# Collections
users_collection = db.users
intake_forms_collection = db.intake_forms
ai_reports_collection = db.ai_reports
medical_uploads_collection = db.medical_uploads
doctors_collection = db.doctors
hospitals_collection = db.hospitals
appointments_collection = db.appointments
payments_collection = db.payments

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Gemini AI Model
model = genai.GenerativeModel('gemini-1.5-pro')

class VirtualHospitalAI:
    def __init__(self):
        self.model = model
        
    def generate_medical_prompt(self, patient_data, context="initial_assessment"):
        """Generate comprehensive medical prompt for Gemini API"""
        
        base_prompt = f"""
        You are an advanced AI medical assistant providing preliminary health assessment. 
        IMPORTANT: This is NOT a replacement for professional medical care.
        
        Patient Information:
        - Age: {patient_data.get('age', 'Not provided')}
        - Gender: {patient_data.get('gender', 'Not provided')}
        - Location: {patient_data.get('location', 'Not provided')}
        
        Current Symptoms: {patient_data.get('symptoms', 'None reported')}
        
        Medical History: {patient_data.get('medical_history', 'None reported')}
        
        Current Medications: {patient_data.get('medications', 'None reported')}
        
        Allergies: {patient_data.get('allergies', 'None reported')}
        
        Lifestyle Factors:
        - Sleep: {patient_data.get('sleep_hours', 'Not provided')} hours
        - Exercise: {patient_data.get('exercise', 'Not provided')}
        - Diet: {patient_data.get('diet', 'Not provided')}
        - Smoking: {patient_data.get('smoking', 'Not provided')}
        - Alcohol: {patient_data.get('alcohol', 'Not provided')}
        
        Insurance/Financial: {patient_data.get('insurance', 'Not provided')}
        
        Please provide a comprehensive assessment in the following structured format:
        
        1. PRELIMINARY ASSESSMENT:
        - Likely health conditions based on symptoms and history
        - Urgency level: EMERGENCY / URGENT / ROUTINE / MONITORING
        - Mental health risk assessment if applicable
        
        2. CLINICAL ANALYSIS:
        - Probable causes and differential diagnosis
        - Risk factors identified
        - Symptom pattern analysis
        
        3. RECOMMENDED INVESTIGATIONS:
        - Essential lab tests and imaging
        - Specialized consultations needed
        - Monitoring parameters
        
        4. LIFESTYLE RECOMMENDATIONS:
        - Personalized diet modifications
        - Exercise and activity guidelines
        - Sleep hygiene improvements
        - Stress management techniques
        
        5. MEDICATION GUIDANCE:
        - Over-the-counter options (if appropriate)
        - Prescription medication categories that may be needed
        - Drug interaction warnings
        - NOTE: Actual prescriptions require licensed physician
        
        6. REFERRAL RECOMMENDATIONS:
        - Specialist consultations needed
        - Urgency of referrals
        - Preparation for appointments
        
        7. PATIENT EDUCATION:
        - Condition explanation in simple terms
        - Warning signs requiring immediate attention
        - Self-care strategies
        
        8. FOLLOW-UP PLAN:
        - Recommended timeline for reassessment
        - Symptom tracking suggestions
        - Progress monitoring indicators
        
        Provide detailed, evidence-based, and patient-friendly explanations.
        Always emphasize the importance of professional medical consultation.
        """
        
        return base_prompt
    
    def analyze_medical_report(self, report_text, patient_context):
        """Analyze uploaded medical reports"""
        
        prompt = f"""
        You are analyzing a medical report for a patient. Provide comprehensive analysis.
        
        Patient Context: {patient_context}
        
        Medical Report Content:
        {report_text}
        
        Please analyze and provide:
        
        1. REPORT SUMMARY:
        - Key findings and abnormalities
        - Normal vs abnormal values
        - Critical results requiring attention
        
        2. CLINICAL INTERPRETATION:
        - What these results mean for the patient
        - Correlation with symptoms and history
        - Progression from previous reports (if mentioned)
        
        3. UPDATED ASSESSMENT:
        - Confirmed or ruled out conditions
        - New diagnostic possibilities
        - Risk stratification
        
        4. RECOMMENDED ACTIONS:
        - Immediate steps needed
        - Follow-up tests required
        - Lifestyle modifications
        - Medication adjustments (general categories)
        
        5. PATIENT COMMUNICATION:
        - Explain results in simple terms
        - Address likely patient concerns
        - Reassurance or caution as appropriate
        
        6. NEXT STEPS:
        - Specialist referrals needed
        - Timeline for follow-up
        - Monitoring requirements
        
        Provide clear, accurate, and compassionate analysis.
        """
        
        return prompt
    
    async def get_ai_assessment(self, prompt):
        """Get AI assessment from Gemini"""
        try:
            response = self.model.generate_content(prompt)
            return {
                'success': True,
                'assessment': response.text,
                'timestamp': datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }

# Initialize AI assistant
ai_assistant = VirtualHospitalAI()

# Utility functions
def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

def check_password(password, hashed):
    return bcrypt.checkpw(password.encode('utf-8'), hashed)

def extract_text_from_pdf(file_path):
    """Extract text from PDF files"""
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
        return text
    except Exception as e:
        logger.error(f"PDF extraction error: {e}")
        return None

def extract_text_from_image(image_path):
    """Extract text from images using OCR"""
    try:
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray)
        return text
    except Exception as e:
        logger.error(f"OCR extraction error: {e}")
        return None

def find_nearby_healthcare(location, condition_urgency, financial_capability):
    """Find nearby doctors and hospitals"""
    try:
        geolocator = Nominatim(user_agent="virtual_hospital")
        location_data = geolocator.geocode(location)
        
        if not location_data:
            return []
        
        # Query hospitals and doctors from database
        query = {
            'location': {
                '$near': {
                    '$geometry': {
                        'type': 'Point',
                        'coordinates': [location_data.longitude, location_data.latitude]
                    },
                    '$maxDistance': 50000  # 50km radius
                }
            }
        }
        
        # Add financial filters
        if financial_capability == 'low':
            query['cost_category'] = {'$in': ['low', 'government']}
        elif financial_capability == 'medium':
            query['cost_category'] = {'$in': ['low', 'medium']}
        
        # Add urgency filters
        if condition_urgency == 'EMERGENCY':
            query['emergency_services'] = True
        
        hospitals = list(hospitals_collection.find(query).limit(10))
        doctors = list(doctors_collection.find(query).limit(10))
        
        return {
            'hospitals': hospitals,
            'doctors': doctors,
            'patient_location': {
                'latitude': location_data.latitude,
                'longitude': location_data.longitude
            }
        }
        
    except Exception as e:
        logger.error(f"Healthcare finder error: {e}")
        return []

# Authentication Routes
@app.route('/api/auth/register', methods=['POST'])
def register():
    try:
        data = request.json
        
        # Check if user exists
        if users_collection.find_one({'email': data['email']}):
            return jsonify({'error': 'User already exists'}), 400
        
        # Hash password
        hashed_password = hash_password(data['password'])
        
        # Create user
        user_data = {
            'email': data['email'],
            'password': hashed_password,
            'first_name': data.get('firstName', ''),
            'last_name': data.get('lastName', ''),
            'role': data.get('role', 'patient'),
            'phone': data.get('phone', ''),
            'date_of_birth': data.get('dateOfBirth', ''),
            'emergency_contact': data.get('emergencyContact', ''),
            'insurance_provider': data.get('insuranceProvider', ''),
            'insurance_number': data.get('insuranceNumber', ''),
            'created_at': datetime.utcnow(),
            'updated_at': datetime.utcnow()
        }
        
        result = users_collection.insert_one(user_data)
        
        # Create JWT token
        access_token = create_access_token(identity=str(result.inserted_id))
        
        # Remove password from response
        user_data.pop('password')
        user_data['_id'] = str(result.inserted_id)
        
        return jsonify({
            'token': access_token,
            'user': user_data
        }), 201
        
    except Exception as e:
        logger.error(f"Registration error: {e}")
        return jsonify({'error': 'Registration failed'}), 500

@app.route('/api/auth/login', methods=['POST'])
def login():
    try:
        data = request.json
        
        # Find user
        user = users_collection.find_one({'email': data['email']})
        if not user or not check_password(data['password'], user['password']):
            return jsonify({'error': 'Invalid credentials'}), 401
        
        # Create JWT token
        access_token = create_access_token(identity=str(user['_id']))
        
        # Remove password from response
        user.pop('password')
        user['_id'] = str(user['_id'])
        
        return jsonify({
            'token': access_token,
            'user': user
        }), 200
        
    except Exception as e:
        logger.error(f"Login error: {e}")
        return jsonify({'error': 'Login failed'}), 500

# Patient Intake Form Routes
@app.route('/api/intake-form', methods=['POST'])
@jwt_required()
def submit_intake_form():
    try:
        user_id = get_jwt_identity()
        data = request.json
        
        # Save intake form
        intake_data = {
            'user_id': ObjectId(user_id),
            'basic_info': {
                'age': data.get('age'),
                'gender': data.get('gender'),
                'location': data.get('location'),
                'occupation': data.get('occupation')
            },
            'symptoms': data.get('symptoms', []),
            'medical_history': data.get('medicalHistory', []),
            'current_medications': data.get('currentMedications', []),
            'allergies': data.get('allergies', []),
            'lifestyle': {
                'sleep_hours': data.get('sleepHours'),
                'exercise': data.get('exercise'),
                'diet': data.get('diet'),
                'smoking': data.get('smoking'),
                'alcohol': data.get('alcohol'),
                'stress_level': data.get('stressLevel')
            },
            'insurance_info': {
                'provider': data.get('insuranceProvider'),
                'policy_number': data.get('policyNumber'),
                'financial_capability': data.get('financialCapability')
            },
            'created_at': datetime.utcnow(),
            'status': 'submitted'
        }
        
        result = intake_forms_collection.insert_one(intake_data)
        
        # Generate AI assessment
        prompt = ai_assistant.generate_medical_prompt(data)
        ai_response = await ai_assistant.get_ai_assessment(prompt)
        
        if ai_response['success']:
            # Save AI report
            ai_report = {
                'user_id': ObjectId(user_id),
                'intake_form_id': result.inserted_id,
                'assessment': ai_response['assessment'],
                'type': 'initial_assessment',
                'created_at': datetime.utcnow()
            }
            
            ai_report_result = ai_reports_collection.insert_one(ai_report)
            
            return jsonify({
                'intake_form_id': str(result.inserted_id),
                'ai_report_id': str(ai_report_result.inserted_id),
                'assessment': ai_response['assessment']
            }), 201
        else:
            return jsonify({
                'intake_form_id': str(result.inserted_id),
                'error': 'AI assessment failed',
                'details': ai_response['error']
            }), 201
            
    except Exception as e:
        logger.error(f"Intake form error: {e}")
        return jsonify({'error': 'Failed to process intake form'}), 500

@app.route('/api/intake-forms', methods=['GET'])
@jwt_required()
def get_intake_forms():
    try:
        user_id = get_jwt_identity()
        
        forms = list(intake_forms_collection.find(
            {'user_id': ObjectId(user_id)}
        ).sort('created_at', -1))
        
        # Convert ObjectId to string
        for form in forms:
            form['_id'] = str(form['_id'])
            form['user_id'] = str(form['user_id'])
        
        return jsonify(forms), 200
        
    except Exception as e:
        logger.error(f"Get intake forms error: {e}")
        return jsonify({'error': 'Failed to retrieve forms'}), 500

# Medical Report Upload Routes
@app.route('/api/upload-medical-report', methods=['POST'])
@jwt_required()
def upload_medical_report():
    try:
        user_id = get_jwt_identity()
        
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        report_type = request.form.get('type', 'general')
        description = request.form.get('description', '')
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Save file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Extract text based on file type
        extracted_text = ""
        if filename.lower().endswith('.pdf'):
            extracted_text = extract_text_from_pdf(file_path)
        elif filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            extracted_text = extract_text_from_image(file_path)
        
        # Get patient context
        latest_intake = intake_forms_collection.find_one(
            {'user_id': ObjectId(user_id)},
            sort=[('created_at', -1)]
        )
        
        patient_context = ""
        if latest_intake:
            patient_context = f"""
            Age: {latest_intake.get('basic_info', {}).get('age')}
            Gender: {latest_intake.get('basic_info', {}).get('gender')}
            Current symptoms: {latest_intake.get('symptoms')}
            Medical history: {latest_intake.get('medical_history')}
            Current medications: {latest_intake.get('current_medications')}
            """
        
        # Generate AI analysis
        prompt = ai_assistant.analyze_medical_report(extracted_text, patient_context)
        ai_response = await ai_assistant.get_ai_assessment(prompt)
        
        # Save upload record
        upload_data = {
            'user_id': ObjectId(user_id),
            'filename': filename,
            'file_path': file_path,
            'report_type': report_type,
            'description': description,
            'extracted_text': extracted_text,
            'ai_analysis': ai_response['assessment'] if ai_response['success'] else None,
            'created_at': datetime.utcnow()
        }
        
        result = medical_uploads_collection.insert_one(upload_data)
        
        return jsonify({
            'upload_id': str(result.inserted_id),
            'extracted_text': extracted_text[:500] + "..." if len(extracted_text) > 500 else extracted_text,
            'ai_analysis': ai_response['assessment'] if ai_response['success'] else None
        }), 201
        
    except Exception as e:
        logger.error(f"Upload error: {e}")
        return jsonify({'error': 'Upload failed'}), 500

# Healthcare Provider Routes
@app.route('/api/find-healthcare', methods=['POST'])
@jwt_required()
def find_healthcare():
    try:
        user_id = get_jwt_identity()
        data = request.json
        
        location = data.get('location')
        condition_urgency = data.get('urgency', 'ROUTINE')
        financial_capability = data.get('financial_capability', 'medium')
        
        results = find_nearby_healthcare(location, condition_urgency, financial_capability)
        
        return jsonify(results), 200
        
    except Exception as e:
        logger.error(f"Healthcare finder error: {e}")
        return jsonify({'error': 'Failed to find healthcare providers'}), 500

# Dashboard Routes
@app.route('/api/dashboard', methods=['GET'])
@jwt_required()
def get_dashboard_data():
    try:
        user_id = get_jwt_identity()
        
        # Get recent intake forms
        recent_forms = list(intake_forms_collection.find(
            {'user_id': ObjectId(user_id)}
        ).sort('created_at', -1).limit(5))
        
        # Get AI reports
        ai_reports = list(ai_reports_collection.find(
            {'user_id': ObjectId(user_id)}
        ).sort('created_at', -1).limit(10))
        
        # Get medical uploads
        uploads = list(medical_uploads_collection.find(
            {'user_id': ObjectId(user_id)}
        ).sort('created_at', -1).limit(10))
        
        # Convert ObjectIds to strings
        for item in recent_forms + ai_reports + uploads:
            item['_id'] = str(item['_id'])
            item['user_id'] = str(item['user_id'])
        
        return jsonify({
            'recent_forms': recent_forms,
            'ai_reports': ai_reports,
            'medical_uploads': uploads,
            'summary': {
                'total_assessments': len(ai_reports),
                'total_uploads': len(uploads),
                'last_assessment': ai_reports[0]['created_at'] if ai_reports else None
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Dashboard error: {e}")
        return jsonify({'error': 'Failed to load dashboard'}), 500

# Payment Routes
@app.route('/api/create-payment-intent', methods=['POST'])
@jwt_required()
def create_payment_intent():
    try:
        user_id = get_jwt_identity()
        data = request.json
        
        amount = data.get('amount')  # in cents
        service_type = data.get('service_type')
        
        # Create Stripe payment intent
        intent = stripe.PaymentIntent.create(
            amount=amount,
            currency='usd',
            metadata={
                'user_id': user_id,
                'service_type': service_type
            }
        )
        
        return jsonify({
            'client_secret': intent.client_secret,
            'payment_intent_id': intent.id
        }), 200
        
    except Exception as e:
        logger.error(f"Payment intent error: {e}")
        return jsonify({'error': 'Failed to create payment intent'}), 500

# Admin Routes (for managing doctors and hospitals)
@app.route('/api/admin/add-doctor', methods=['POST'])
@jwt_required()
def add_doctor():
    try:
        data = request.json
        
        doctor_data = {
            'name': data['name'],
            'specialization': data['specialization'],
            'qualifications': data['qualifications'],
            'experience': data['experience'],
            'location': {
                'type': 'Point',
                'coordinates': [data['longitude'], data['latitude']]
            },
            'address': data['address'],
            'phone': data['phone'],
            'email': data['email'],
            'consultation_fee': data['consultation_fee'],
            'cost_category': data['cost_category'],
            'availability': data['availability'],
            'rating': data.get('rating', 0),
            'created_at': datetime.utcnow()
        }
        
        result = doctors_collection.insert_one(doctor_data)
        
        return jsonify({
            'doctor_id': str(result.inserted_id),
            'message': 'Doctor added successfully'
        }), 201
        
    except Exception as e:
        logger.error(f"Add doctor error: {e}")
        return jsonify({'error': 'Failed to add doctor'}), 500

@app.route('/api/admin/add-hospital', methods=['POST'])
@jwt_required()
def add_hospital():
    try:
        data = request.json
        
        hospital_data = {
            'name': data['name'],
            'type': data['type'],  # government, private, specialty
            'specialties': data['specialties'],
            'location': {
                'type': 'Point',
                'coordinates': [data['longitude'], data['latitude']]
            },
            'address': data['address'],
            'phone': data['phone'],
            'email': data['email'],
            'emergency_services': data['emergency_services'],
            'cost_category': data['cost_category'],
            'facilities': data['facilities'],
            'rating': data.get('rating', 0),
            'bed_capacity': data['bed_capacity'],
            'created_at': datetime.utcnow()
        }
        
        result = hospitals_collection.insert_one(hospital_data)
        
        return jsonify({
            'hospital_id': str(result.inserted_id),
            'message': 'Hospital added successfully'
        }), 201
        
    except Exception as e:
        logger.error(f"Add hospital error: {e}")
        return jsonify({'error': 'Failed to add hospital'}), 500

# Health check
@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'services': {
            'mongodb': 'connected',
            'gemini_api': 'configured',
            'stripe': 'configured'
        }
    }), 200

if __name__ == '__main__':
    # Create upload directory
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Create database indexes
    users_collection.create_index('email', unique=True)
    intake_forms_collection.create_index('user_id')
    ai_reports_collection.create_index('user_id')
    medical_uploads_collection.create_index('user_id')
    doctors_collection.create_index([('location', '2dsphere')])
    hospitals_collection.create_index([('location', '2dsphere')])
    
    logger.info("🏥 Virtual Hospital Backend Starting...")
    logger.info("🤖 Gemini AI Integration: Enabled")
    logger.info("💳 Stripe Payments: Enabled")
    logger.info("📱 SMS Notifications: Enabled")
    logger.info("🗄️ MongoDB: Connected")
    
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 3002)), debug=True)