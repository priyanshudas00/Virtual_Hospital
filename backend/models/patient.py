from datetime import datetime
from typing import Dict, List, Optional, Any
from bson import ObjectId
import logging

logger = logging.getLogger(__name__)

class PatientModel:
    def __init__(self, db_manager):
        self.db = db_manager
        self.collection = db_manager.get_collection('patients')
        self.intake_collection = db_manager.get_collection('intake_forms')
        self.reports_collection = db_manager.get_collection('ai_reports')
        self.uploads_collection = db_manager.get_collection('medical_uploads')
    
    async def create_patient_profile(self, user_data: Dict) -> Dict:
        """Create comprehensive patient profile"""
        try:
            patient_profile = {
                'user_id': user_data['user_id'],
                'personal_info': {
                    'first_name': user_data.get('first_name'),
                    'last_name': user_data.get('last_name'),
                    'email': user_data.get('email'),
                    'phone': user_data.get('phone'),
                    'date_of_birth': user_data.get('date_of_birth'),
                    'gender': user_data.get('gender'),
                    'address': user_data.get('address'),
                    'emergency_contact': user_data.get('emergency_contact')
                },
                'medical_profile': {
                    'blood_type': user_data.get('blood_type'),
                    'height': user_data.get('height'),
                    'weight': user_data.get('weight'),
                    'chronic_conditions': user_data.get('chronic_conditions', []),
                    'allergies': user_data.get('allergies', []),
                    'current_medications': user_data.get('current_medications', []),
                    'family_history': user_data.get('family_history', []),
                    'surgical_history': user_data.get('surgical_history', [])
                },
                'insurance_info': {
                    'provider': user_data.get('insurance_provider'),
                    'policy_number': user_data.get('insurance_number'),
                    'coverage_type': user_data.get('coverage_type'),
                    'financial_capability': user_data.get('financial_capability', 'medium')
                },
                'preferences': {
                    'preferred_language': user_data.get('preferred_language', 'English'),
                    'communication_method': user_data.get('communication_method', 'email'),
                    'privacy_settings': user_data.get('privacy_settings', {})
                },
                'created_at': datetime.utcnow(),
                'updated_at': datetime.utcnow(),
                'status': 'active'
            }
            
            result = self.collection.insert_one(patient_profile)
            patient_profile['_id'] = str(result.inserted_id)
            
            logger.info(f"Patient profile created: {result.inserted_id}")
            return patient_profile
            
        except Exception as e:
            logger.error(f"Patient profile creation failed: {e}")
            raise
    
    async def save_intake_form(self, user_id: str, intake_data: Dict) -> Dict:
        """Save comprehensive intake form data"""
        try:
            intake_form = {
                'user_id': ObjectId(user_id),
                'form_version': '2.0',
                'basic_info': {
                    'age': intake_data.get('age'),
                    'gender': intake_data.get('gender'),
                    'location': intake_data.get('location'),
                    'occupation': intake_data.get('occupation'),
                    'marital_status': intake_data.get('marital_status'),
                    'education_level': intake_data.get('education_level')
                },
                'symptoms': {
                    'primary_symptoms': intake_data.get('symptoms', []),
                    'symptom_duration': intake_data.get('symptom_duration'),
                    'pain_level': intake_data.get('pain_level', 0),
                    'symptom_details': intake_data.get('symptom_details', ''),
                    'onset_pattern': intake_data.get('onset_pattern'),
                    'aggravating_factors': intake_data.get('aggravating_factors', []),
                    'relieving_factors': intake_data.get('relieving_factors', [])
                },
                'medical_history': {
                    'past_conditions': intake_data.get('medical_history', []),
                    'current_medications': intake_data.get('current_medications', []),
                    'allergies': intake_data.get('allergies_text', ''),
                    'surgeries': intake_data.get('surgeries_text', ''),
                    'family_history': intake_data.get('family_history_text', ''),
                    'immunization_status': intake_data.get('immunizations', []),
                    'previous_hospitalizations': intake_data.get('hospitalizations', [])
                },
                'lifestyle_factors': {
                    'sleep_hours': intake_data.get('sleep_hours'),
                    'sleep_quality': intake_data.get('sleep_quality'),
                    'exercise_frequency': intake_data.get('exercise'),
                    'exercise_type': intake_data.get('exercise_type'),
                    'diet_type': intake_data.get('diet'),
                    'smoking_status': intake_data.get('smoking'),
                    'alcohol_consumption': intake_data.get('alcohol'),
                    'stress_level': intake_data.get('stress_level'),
                    'work_environment': intake_data.get('work_environment'),
                    'travel_history': intake_data.get('travel_history')
                },
                'social_determinants': {
                    'living_situation': intake_data.get('living_situation'),
                    'support_system': intake_data.get('support_system'),
                    'transportation_access': intake_data.get('transportation'),
                    'food_security': intake_data.get('food_security'),
                    'housing_stability': intake_data.get('housing_stability')
                },
                'insurance_financial': {
                    'insurance_provider': intake_data.get('insurance_provider'),
                    'policy_number': intake_data.get('policy_number'),
                    'financial_capability': intake_data.get('financial_capability'),
                    'preferred_cost_range': intake_data.get('cost_preference'),
                    'payment_method': intake_data.get('payment_method')
                },
                'mental_health': {
                    'mood_assessment': intake_data.get('mood_rating'),
                    'anxiety_level': intake_data.get('anxiety_level'),
                    'depression_screening': intake_data.get('depression_score'),
                    'mental_health_history': intake_data.get('mental_health_history'),
                    'current_stressors': intake_data.get('current_stressors', [])
                },
                'form_metadata': {
                    'completion_time': intake_data.get('completion_time'),
                    'form_language': intake_data.get('form_language', 'English'),
                    'assistance_needed': intake_data.get('assistance_needed', False),
                    'data_quality_score': self._calculate_data_quality(intake_data)
                },
                'created_at': datetime.utcnow(),
                'status': 'submitted',
                'ai_processing_status': 'pending'
            }
            
            result = self.intake_collection.insert_one(intake_form)
            intake_form['_id'] = str(result.inserted_id)
            
            logger.info(f"Intake form saved: {result.inserted_id}")
            return intake_form
            
        except Exception as e:
            logger.error(f"Intake form save failed: {e}")
            raise
    
    async def save_ai_assessment(self, user_id: str, intake_form_id: str, 
                               assessment_data: Dict, assessment_type: str = 'initial') -> Dict:
        """Save AI assessment results"""
        try:
            ai_report = {
                'user_id': ObjectId(user_id),
                'intake_form_id': ObjectId(intake_form_id),
                'assessment_type': assessment_type,
                'assessment_data': assessment_data,
                'model_info': {
                    'model_name': 'gemini-1.5-pro',
                    'model_version': '1.5',
                    'processing_time': assessment_data.get('processing_time'),
                    'confidence_metrics': assessment_data.get('confidence_metrics')
                },
                'clinical_summary': {
                    'primary_diagnosis': assessment_data.get('preliminary_assessment', {}).get('primary_diagnosis'),
                    'urgency_level': assessment_data.get('preliminary_assessment', {}).get('urgency_level'),
                    'confidence_score': assessment_data.get('preliminary_assessment', {}).get('confidence_score'),
                    'key_recommendations': assessment_data.get('follow_up_plan', {}).get('next_steps', [])
                },
                'created_at': datetime.utcnow(),
                'status': 'completed',
                'reviewed_by_human': False
            }
            
            result = self.reports_collection.insert_one(ai_report)
            ai_report['_id'] = str(result.inserted_id)
            
            # Update intake form status
            self.intake_collection.update_one(
                {'_id': ObjectId(intake_form_id)},
                {
                    '$set': {
                        'ai_processing_status': 'completed',
                        'ai_report_id': result.inserted_id,
                        'updated_at': datetime.utcnow()
                    }
                }
            )
            
            logger.info(f"AI assessment saved: {result.inserted_id}")
            return ai_report
            
        except Exception as e:
            logger.error(f"AI assessment save failed: {e}")
            raise
    
    def _calculate_data_quality(self, intake_data: Dict) -> float:
        """Calculate data quality score for intake form"""
        total_fields = 0
        completed_fields = 0
        
        # Define important fields and their weights
        important_fields = {
            'age': 1.0,
            'gender': 1.0,
            'symptoms': 2.0,
            'medical_history': 1.5,
            'current_medications': 1.5,
            'allergies_text': 1.0,
            'lifestyle': 1.0
        }
        
        for field, weight in important_fields.items():
            total_fields += weight
            value = intake_data.get(field)
            
            if value:
                if isinstance(value, list) and len(value) > 0:
                    completed_fields += weight
                elif isinstance(value, str) and value.strip():
                    completed_fields += weight
                elif isinstance(value, (int, float)) and value > 0:
                    completed_fields += weight
        
        return min(completed_fields / total_fields, 1.0) if total_fields > 0 else 0.0

# Patient model will be initialized with database manager