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
from ..config.database import db_manager
from ..models.patient import PatientModel

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
        
        # Emergency check
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
        
        # Get initial triage questions from Gemini
        triage_response = asyncio.run(
            gemini_service.perform_triage(initial_complaint, [])
        )
        
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