"""
Medical Triage API Routes
Dynamic symptom checking and intelligent triage system
"""

from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
import asyncio
from datetime import datetime
from bson import ObjectId
import logging

from ..services.gemini_medical_service import gemini_service
from ai.models.symptom_analyzer import SymptomAnalyzer
from ..config.database import db_manager
from ..models.patient import PatientModel

# Initialize ML symptom analyzer
ml_symptom_analyzer = SymptomAnalyzer()

logger = logging.getLogger(__name__)

triage_bp = Blueprint('triage', __name__, url_prefix='/api/triage')

@triage_bp.route('/start-session', methods=['POST'])
@jwt_required()
def start_triage_session():
    """Start new triage session with initial complaint"""
    try:
        user_id = get_jwt_identity()
        data = request.json
        
        initial_complaint = data.get('complaint', '').strip()
        if not initial_complaint:
            return jsonify({'error': 'Initial complaint is required'}), 400
        
        # Emergency check (keep Gemini for critical safety)
        if gemini_service.safety.check_emergency(initial_complaint):
            return jsonify({
                'emergency_detected': True,
                'response': gemini_service.safety.EMERGENCY_RESPONSE,
                'action': 'EMERGENCY_PROTOCOL'
            }), 200

        # Create new interaction session
        session_data = {
            'user_id': ObjectId(user_id),
            'session_id': f"TRIAGE_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{user_id[:8]}",
            'started_at': datetime.utcnow(),
            'status': 'active',
            'initial_complaint': {
                'primary_symptom': data.get('primary_symptom', ''),
                'user_description': initial_complaint,
                'duration': data.get('duration', ''),
                'severity': data.get('severity', 5),
                'onset': data.get('onset', '')
            },
            'conversation_log': [],
            'ai_assessment': {},
            'doctor_report': {}
        }

        # Save to database
        interactions_collection = db_manager.get_collection('interactions')
        result = interactions_collection.insert_one(session_data)
        session_data['_id'] = str(result.inserted_id)

        # Get initial triage assessment from ML analyzer (minimize Gemini API calls)
        logger.info("Performing initial triage with ML/DL models...")
        ml_assessment = ml_symptom_analyzer.analyze_symptoms(
            initial_complaint,
            patient_age=data.get('age', 30),
            patient_gender=data.get('gender', 'unknown')
        )

        # Generate triage questions based on ML assessment
        triage_response = {
            'assessment': ml_assessment,
            'questions': _generate_triage_questions_ml_based(ml_assessment),
            'urgency_level': ml_assessment.get('urgency', 'Medium'),
            'disclaimer': 'This is an AI-assisted assessment. Please consult healthcare professionals for medical advice.',
            'ml_dependency_score': ml_assessment.get('mlDependencyScore', 0.0),
            'analysis_method': ml_assessment.get('analysisMethod', 'ML/DL Enhanced')
        }
        if triage_response.get('emergency_detected'):
            return jsonify(triage_response), 200
        
        # Update session with initial questions
        if 'questions' in triage_response:
            interactions_collection.update_one(
                {'_id': result.inserted_id},
                {'$set': {'current_questions': triage_response['questions']}}
            )
        
        return jsonify({
            'success': True,
            'session_id': session_data['session_id'],
            'interaction_id': str(result.inserted_id),
            'questions': triage_response.get('questions', []),
            'disclaimer': gemini_service.safety.get_medical_disclaimer()
        }), 201
        
    except Exception as e:
        logger.error(f"Triage session start failed: {e}")
        return jsonify({
            'error': 'Failed to start triage session',
            'details': str(e)
        }), 500

@triage_bp.route('/answer-question', methods=['POST'])
@jwt_required()
def answer_triage_question():
    """Process user answer and get next questions"""
    try:
        user_id = get_jwt_identity()
        data = request.json
        
        interaction_id = data.get('interaction_id')
        question = data.get('question')
        answer = data.get('answer', '').strip()
        
        if not all([interaction_id, question, answer]):
            return jsonify({'error': 'Missing required fields'}), 400
        
        # Get interaction from database
        interactions_collection = db_manager.get_collection('interactions')
        interaction = interactions_collection.find_one({
            '_id': ObjectId(interaction_id),
            'user_id': ObjectId(user_id)
        })
        
        if not interaction:
            return jsonify({'error': 'Interaction not found'}), 404
        
        # Emergency check on answer
        if gemini_service.safety.check_emergency(answer):
            return jsonify({
                'emergency_detected': True,
                'response': gemini_service.safety.EMERGENCY_RESPONSE,
                'action': 'EMERGENCY_PROTOCOL'
            }), 200
        
        # Add answer to conversation log
        conversation_entry = {
            'sequence': len(interaction.get('conversation_log', [])) + 1,
            'ai_question': question,
            'user_answer': answer,
            'timestamp': datetime.utcnow(),
            'question_type': data.get('question_type', 'general')
        }
        
        # Update conversation log
        interactions_collection.update_one(
            {'_id': ObjectId(interaction_id)},
            {'$push': {'conversation_log': conversation_entry}}
        )
        
        # Get updated conversation history
        updated_interaction = interactions_collection.find_one({'_id': ObjectId(interaction_id)})
        conversation_history = updated_interaction.get('conversation_log', [])
        
        # Check if we have enough information for assessment
        if len(conversation_history) >= 5:  # After 5 questions, provide assessment
            # Get patient profile for context
            users_collection = db_manager.get_collection('users')
            user_profile = users_collection.find_one({'_id': ObjectId(user_id)})
            
            # Generate final assessment
            assessment = asyncio.run(
                gemini_service.generate_final_assessment(
                    updated_interaction, 
                    user_profile.get('profile', {}) if user_profile else {}
                )
            )
            
            # Update interaction with assessment
            interactions_collection.update_one(
                {'_id': ObjectId(interaction_id)},
                {
                    '$set': {
                        'ai_assessment': assessment.get('triage_assessment', {}),
                        'doctor_report': assessment.get('doctor_report', {}),
                        'patient_education': assessment.get('patient_education', {}),
                        'completed_at': datetime.utcnow(),
                        'status': 'completed'
                    }
                }
            )
            
            return jsonify({
                'assessment_complete': True,
                'assessment': assessment,
                'session_id': interaction['session_id'],
                'interaction_id': interaction_id
            }), 200
        
        else:
            # Get next questions
            initial_complaint = interaction['initial_complaint']['user_description']
            next_questions = asyncio.run(
                gemini_service.perform_triage(initial_complaint, conversation_history)
            )
            
            return jsonify({
                'assessment_complete': False,
                'questions': next_questions.get('questions', []),
                'progress': len(conversation_history) / 5 * 100,  # Progress percentage
                'conversation_count': len(conversation_history)
            }), 200
        
    except Exception as e:
        logger.error(f"Question answering failed: {e}")
        return jsonify({
            'error': 'Failed to process answer',
            'details': str(e)
        }), 500

@triage_bp.route('/get-assessment/<interaction_id>', methods=['GET'])
@jwt_required()
def get_triage_assessment(interaction_id):
    """Get completed triage assessment"""
    try:
        user_id = get_jwt_identity()
        
        interactions_collection = db_manager.get_collection('interactions')
        interaction = interactions_collection.find_one({
            '_id': ObjectId(interaction_id),
            'user_id': ObjectId(user_id),
            'status': 'completed'
        })
        
        if not interaction:
            return jsonify({'error': 'Assessment not found or not completed'}), 404
        
        # Serialize MongoDB document
        def serialize_doc(doc):
            if isinstance(doc, ObjectId):
                return str(doc)
            elif isinstance(doc, datetime):
                return doc.isoformat()
            elif isinstance(doc, dict):
                return {k: serialize_doc(v) for k, v in doc.items()}
            elif isinstance(doc, list):
                return [serialize_doc(item) for item in doc]
            return doc
        
        serialized_interaction = serialize_doc(interaction)
        
        return jsonify({
            'success': True,
            'assessment': serialized_interaction.get('ai_assessment', {}),
            'doctor_report': serialized_interaction.get('doctor_report', {}),
            'patient_education': serialized_interaction.get('patient_education', {}),
            'conversation_summary': {
                'total_questions': len(serialized_interaction.get('conversation_log', [])),
                'session_duration': self._calculate_session_duration(interaction),
                'initial_complaint': serialized_interaction.get('initial_complaint', {})
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Assessment retrieval failed: {e}")
        return jsonify({
            'error': 'Failed to retrieve assessment',
            'details': str(e)
        }), 500

@triage_bp.route('/sessions', methods=['GET'])
@jwt_required()
def get_user_sessions():
    """Get all triage sessions for user"""
    try:
        user_id = get_jwt_identity()
        
        interactions_collection = db_manager.get_collection('interactions')
        sessions = list(interactions_collection.find(
            {'user_id': ObjectId(user_id)}
        ).sort('started_at', -1).limit(50))
        
        # Serialize sessions
        serialized_sessions = []
        for session in sessions:
            serialized_session = {
                'id': str(session['_id']),
                'session_id': session.get('session_id'),
                'started_at': session['started_at'].isoformat(),
                'status': session.get('status'),
                'initial_complaint': session.get('initial_complaint', {}),
                'assessment_summary': {
                    'primary_condition': session.get('ai_assessment', {}).get('likely_conditions', [{}])[0].get('condition', 'Unknown'),
                    'triage_level': session.get('ai_assessment', {}).get('triage_level', 'Unknown'),
                    'urgency_score': session.get('ai_assessment', {}).get('urgency_score', 0)
                }
            }
            
            if session.get('completed_at'):
                serialized_session['completed_at'] = session['completed_at'].isoformat()
            
            serialized_sessions.append(serialized_session)
        
        return jsonify({
            'success': True,
            'sessions': serialized_sessions,
            'total_count': len(serialized_sessions)
        }), 200
        
    except Exception as e:
        logger.error(f"Sessions retrieval failed: {e}")
        return jsonify({
            'error': 'Failed to retrieve sessions',
            'details': str(e)
        }), 500

def _calculate_session_duration(interaction: Dict) -> str:
    """Calculate session duration"""
    try:
        start_time = interaction.get('started_at')
        end_time = interaction.get('completed_at')
        
        if start_time and end_time:
            duration = end_time - start_time
            minutes = int(duration.total_seconds() / 60)
            return f"{minutes} minutes"
        
        return "In progress"
        
    except Exception:
        return "Unknown"

def _generate_triage_questions_ml_based(ml_assessment: Dict) -> List[str]:
    """Generate triage questions based on ML assessment to minimize external API calls"""
    questions = []
    urgency = ml_assessment.get('urgency', 'Medium')
    primary_diagnosis = ml_assessment.get('primaryDiagnosis', '').lower()
    extracted_symptoms = ml_assessment.get('extractedSymptoms', [])

    # Base questions for all cases
    questions.append("How long have you been experiencing these symptoms?")
    questions.append("On a scale of 1-10, how severe are your symptoms?")

    # ML-driven questions based on diagnosis
    if 'pain' in primary_diagnosis or 'ache' in primary_diagnosis:
        questions.extend([
            "Where exactly is the pain located?",
            "Does the pain radiate to other areas?",
            "What makes the pain better or worse?"
        ])
    elif 'fever' in extracted_symptoms:
        questions.extend([
            "What is your highest temperature?",
            "Do you have any associated symptoms like chills or sweating?"
        ])
    elif 'cough' in extracted_symptoms:
        questions.extend([
            "Is your cough productive (bringing up mucus)?",
            "Do you have shortness of breath?"
        ])
    elif 'headache' in extracted_symptoms:
        questions.extend([
            "Where is the headache located?",
            "Is it throbbing, constant, or does it come and go?"
        ])

    # Urgency-based questions
    if urgency == 'High':
        questions.extend([
            "Are you experiencing chest pain or pressure?",
            "Do you have difficulty breathing?",
            "Have you lost consciousness or had seizures?"
        ])
    elif urgency == 'Low':
        questions.extend([
            "Have these symptoms occurred before?",
            "What home remedies have you tried?"
        ])

    # Limit to 5 questions to keep it manageable
    return questions[:5]
