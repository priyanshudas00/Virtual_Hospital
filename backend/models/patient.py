from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from bson import ObjectId
import logging
import asyncio

logger = logging.getLogger(__name__)

class AdvancedPatientModel:
    """Advanced patient data management with comprehensive medical records"""
    
    def __init__(self, db_manager):
        self.db = db_manager
        self.users_collection = db_manager.get_collection('users')
        self.interactions_collection = db_manager.get_collection('interactions')
        self.medical_reports_collection = db_manager.get_collection('medical_reports')
        self.patient_timeline_collection = db_manager.get_collection('patient_timeline')
        self.appointments_collection = db_manager.get_collection('appointments')
    
    async def create_comprehensive_patient_profile(self, user_data: Dict) -> Dict:
        """Create comprehensive patient profile with medical history"""
        try:
            patient_profile = {
                'user_id': user_data['user_id'],
                'profile': {
                    'personal_info': {
                        'full_name': f"{user_data.get('first_name', '')} {user_data.get('last_name', '')}",
                        'age': user_data.get('age'),
                        'biological_sex': user_data.get('gender'),
                        'date_of_birth': user_data.get('date_of_birth'),
                        'phone': user_data.get('phone'),
                        'address': user_data.get('address'),
                        'emergency_contact': {
                            'name': user_data.get('emergency_contact_name'),
                            'phone': user_data.get('emergency_contact_phone'),
                            'relationship': user_data.get('emergency_contact_relationship')
                        }
                    },
                    'medical_history': {
                        'past_conditions': user_data.get('chronic_conditions', []),
                        'surgeries': self._format_surgeries(user_data.get('surgeries_text', '')),
                        'family_history': {
                            'diabetes': user_data.get('family_diabetes', False),
                            'heart_disease': user_data.get('family_heart_disease', False),
                            'cancer': user_data.get('family_cancer', False),
                            'hypertension': user_data.get('family_hypertension', False),
                            'other': user_data.get('family_other', [])
                        },
                        'immunization_status': user_data.get('immunizations', []),
                        'previous_hospitalizations': user_data.get('hospitalizations', [])
                    },
                    'medications': self._format_medications(user_data.get('current_medications', [])),
                    'allergies': self._format_allergies(user_data.get('allergies_text', '')),
                    'lifestyle': {
                        'smoking': {
                            'status': user_data.get('smoking', 'never'),
                            'packs_per_day': user_data.get('smoking_packs', 0),
                            'years_smoked': user_data.get('smoking_years', 0)
                        },
                        'alcohol': {
                            'frequency': user_data.get('alcohol', 'never'),
                            'drinks_per_week': user_data.get('alcohol_drinks', 0)
                        },
                        'exercise': {
                            'frequency': user_data.get('exercise', 'rarely'),
                            'type': user_data.get('exercise_type', []),
                            'duration_minutes': user_data.get('exercise_duration', 0)
                        },
                        'diet': {
                            'type': user_data.get('diet', 'mixed'),
                            'restrictions': user_data.get('diet_restrictions', [])
                        },
                        'sleep': {
                            'hours_per_night': user_data.get('sleep_hours', 8),
                            'quality': user_data.get('sleep_quality', 'good'),
                            'sleep_disorders': user_data.get('sleep_disorders', [])
                        },
                        'stress_level': user_data.get('stress_level', 5),
                        'occupation': user_data.get('occupation'),
                        'work_environment': user_data.get('work_environment')
                    },
                    'insurance': {
                        'provider': user_data.get('insurance_provider'),
                        'policy_number': user_data.get('insurance_number'),
                        'coverage_type': user_data.get('coverage_type'),
                        'financial_capability': user_data.get('financial_capability', 'medium')
                    },
                    'social_determinants': {
                        'living_situation': user_data.get('living_situation'),
                        'support_system': user_data.get('support_system'),
                        'transportation_access': user_data.get('transportation'),
                        'food_security': user_data.get('food_security'),
                        'housing_stability': user_data.get('housing_stability'),
                        'education_level': user_data.get('education_level'),
                        'employment_status': user_data.get('employment_status')
                    }
                },
                'privacy_settings': {
                    'data_sharing_consent': user_data.get('data_sharing_consent', False),
                    'research_participation': user_data.get('research_participation', False),
                    'doctor_access_level': user_data.get('doctor_access_level', 'limited')
                },
                'account_info': {
                    'created_at': datetime.utcnow(),
                    'last_updated': datetime.utcnow(),
                    'profile_completion': self._calculate_profile_completion(user_data),
                    'verification_status': 'pending',
                    'account_status': 'active'
                }
            }
            
            result = self.users_collection.insert_one(patient_profile)
            patient_profile['_id'] = str(result.inserted_id)
            
            # Create initial timeline entry
            await self._add_timeline_event(
                str(result.inserted_id),
                'profile_created',
                'Patient profile created',
                {'profile_completion': patient_profile['account_info']['profile_completion']}
            )
            
            logger.info(f"Comprehensive patient profile created: {result.inserted_id}")
            return patient_profile
            
        except Exception as e:
            logger.error(f"Patient profile creation failed: {e}")
            raise
    
    async def save_triage_interaction(self, user_id: str, interaction_data: Dict) -> Dict:
        """Save comprehensive triage interaction with AI assessment"""
        try:
            interaction_record = {
                'user_id': ObjectId(user_id),
                'session_id': interaction_data.get('session_id'),
                'started_at': datetime.utcnow(),
                'completed_at': interaction_data.get('completed_at'),
                'status': interaction_data.get('status', 'active'),
                'interaction_type': 'triage_assessment',
                
                'initial_complaint': {
                    'primary_symptom': interaction_data.get('primary_symptom'),
                    'user_description': interaction_data.get('user_description'),
                    'duration': interaction_data.get('duration'),
                    'severity': interaction_data.get('severity', 5),
                    'onset': interaction_data.get('onset', 'gradual')
                },
                
                'conversation_log': interaction_data.get('conversation_log', []),
                
                'ai_assessment': interaction_data.get('ai_assessment', {}),
                
                'doctor_report': {
                    'chief_complaint': interaction_data.get('chief_complaint'),
                    'history_present_illness': interaction_data.get('hpi'),
                    'relevant_pmh': interaction_data.get('relevant_pmh'),
                    'medications_allergies': interaction_data.get('medications_allergies'),
                    'differential_diagnosis': interaction_data.get('differential_diagnosis', []),
                    'red_flags': interaction_data.get('red_flags', []),
                    'clinical_impression': interaction_data.get('clinical_impression'),
                    'generated_at': datetime.utcnow()
                },
                
                'follow_up': {
                    'recommended_timeline': interaction_data.get('recommended_timeline'),
                    'monitoring_instructions': interaction_data.get('monitoring_instructions', []),
                    'return_if_worse': interaction_data.get('return_if_worse', []),
                    'specialist_referrals': interaction_data.get('specialist_referrals', [])
                },
                
                'metadata': {
                    'ai_model_version': 'gemini-1.5-pro',
                    'assessment_confidence': interaction_data.get('confidence_score', 0.0),
                    'data_quality_score': self._assess_interaction_quality(interaction_data),
                    'processing_time': interaction_data.get('processing_time', 0)
                }
            }
            
            result = self.interactions_collection.insert_one(interaction_record)
            interaction_record['_id'] = str(result.inserted_id)
            
            # Add to patient timeline
            await self._add_timeline_event(
                user_id,
                'triage_completed',
                f"AI Triage: {interaction_data.get('primary_diagnosis', 'Assessment completed')}",
                {
                    'interaction_id': str(result.inserted_id),
                    'urgency_level': interaction_data.get('urgency_level'),
                    'confidence': interaction_data.get('confidence_score')
                }
            )
            
            logger.info(f"Triage interaction saved: {result.inserted_id}")
            return interaction_record
            
        except Exception as e:
            logger.error(f"Triage interaction save failed: {e}")
            raise
    
    async def save_medical_report_analysis(self, user_id: str, report_data: Dict, 
                                         ai_analysis: Dict) -> Dict:
        """Save medical report with comprehensive AI analysis"""
        try:
            report_record = {
                'user_id': ObjectId(user_id),
                'upload_metadata': {
                    'original_filename': report_data.get('filename'),
                    'file_size': report_data.get('file_size'),
                    'mime_type': report_data.get('mime_type'),
                    'upload_timestamp': datetime.utcnow(),
                    'file_hash': report_data.get('file_hash'),
                    'storage_url': report_data.get('storage_url')
                },
                
                'report_classification': {
                    'type': report_data.get('type'),
                    'modality': report_data.get('modality'),
                    'body_part': report_data.get('body_part'),
                    'study_date': report_data.get('study_date'),
                    'ordering_physician': report_data.get('ordering_physician'),
                    'institution': report_data.get('institution'),
                    'clinical_indication': report_data.get('clinical_indication')
                },
                
                'ai_analysis': {
                    'image_findings': ai_analysis.get('image_analysis', {}),
                    'text_analysis': ai_analysis.get('report_analysis', {}),
                    'overall_assessment': ai_analysis.get('overall_impression'),
                    'confidence_score': ai_analysis.get('confidence_score', 0.0),
                    'processing_time': ai_analysis.get('processing_time', 0),
                    'model_version': ai_analysis.get('model_version', 'gemini-1.5-pro'),
                    
                    'flags': self._extract_medical_flags(ai_analysis),
                    
                    'recommendations': {
                        'immediate_actions': ai_analysis.get('immediate_actions', []),
                        'follow_up_studies': ai_analysis.get('follow_up_studies', []),
                        'specialist_referrals': ai_analysis.get('specialist_referrals', []),
                        'correlation_needed': ai_analysis.get('correlation_needed', [])
                    }
                },
                
                'radiologist_review': {
                    'reviewed': False,
                    'priority': self._determine_review_priority(ai_analysis),
                    'review_requested_at': datetime.utcnow(),
                    'estimated_review_time': self._estimate_review_time(ai_analysis)
                },
                
                'patient_communication': {
                    'simplified_explanation': ai_analysis.get('patient_explanation'),
                    'shared_with_patient': False,
                    'patient_questions': [],
                    'ai_responses': []
                },
                
                'integration_data': {
                    'linked_interactions': [],  # Link to related triage sessions
                    'previous_reports': [],     # Link to previous similar reports
                    'care_plan_impact': ai_analysis.get('care_plan_impact')
                }
            }
            
            result = self.medical_reports_collection.insert_one(report_record)
            report_record['_id'] = str(result.inserted_id)
            
            # Add to patient timeline
            await self._add_timeline_event(
                user_id,
                'report_analyzed',
                f"Medical Report: {report_data.get('type', 'Unknown')} analyzed",
                {
                    'report_id': str(result.inserted_id),
                    'report_type': report_data.get('type'),
                    'key_findings': ai_analysis.get('key_findings', [])[:3],  # Top 3 findings
                    'urgency': self._determine_review_priority(ai_analysis)
                }
            )
            
            logger.info(f"Medical report analysis saved: {result.inserted_id}")
            return report_record
            
        except Exception as e:
            logger.error(f"Medical report save failed: {e}")
            raise
    
    async def generate_patient_timeline(self, user_id: str, days_back: int = 90) -> List[Dict]:
        """Generate comprehensive patient medical timeline"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_back)
            
            # Get timeline events
            timeline_events = list(self.patient_timeline_collection.find({
                'user_id': ObjectId(user_id),
                'timestamp': {'$gte': cutoff_date}
            }).sort('timestamp', -1))
            
            # Get interactions
            interactions = list(self.interactions_collection.find({
                'user_id': ObjectId(user_id),
                'started_at': {'$gte': cutoff_date}
            }).sort('started_at', -1))
            
            # Get medical reports
            reports = list(self.medical_reports_collection.find({
                'user_id': ObjectId(user_id),
                'upload_metadata.upload_timestamp': {'$gte': cutoff_date}
            }).sort('upload_metadata.upload_timestamp', -1))
            
            # Combine and sort all events
            combined_timeline = []
            
            # Add timeline events
            for event in timeline_events:
                combined_timeline.append({
                    'id': str(event['_id']),
                    'type': 'timeline_event',
                    'event_type': event.get('event_type'),
                    'title': event.get('title'),
                    'description': event.get('description'),
                    'timestamp': event.get('timestamp'),
                    'metadata': event.get('metadata', {})
                })
            
            # Add interactions
            for interaction in interactions:
                primary_diagnosis = 'Unknown'
                if interaction.get('ai_assessment'):
                    conditions = interaction['ai_assessment'].get('likely_conditions', [])
                    if conditions:
                        primary_diagnosis = conditions[0].get('condition', 'Unknown')
                
                combined_timeline.append({
                    'id': str(interaction['_id']),
                    'type': 'triage_interaction',
                    'title': f"AI Triage: {primary_diagnosis}",
                    'description': interaction.get('initial_complaint', {}).get('user_description', ''),
                    'timestamp': interaction.get('started_at'),
                    'urgency': interaction.get('ai_assessment', {}).get('triage_level'),
                    'metadata': {
                        'session_id': interaction.get('session_id'),
                        'status': interaction.get('status')
                    }
                })
            
            # Add medical reports
            for report in reports:
                combined_timeline.append({
                    'id': str(report['_id']),
                    'type': 'medical_report',
                    'title': f"Medical Report: {report.get('report_classification', {}).get('type', 'Unknown')}",
                    'description': report.get('ai_analysis', {}).get('overall_assessment', ''),
                    'timestamp': report.get('upload_metadata', {}).get('upload_timestamp'),
                    'flags': report.get('ai_analysis', {}).get('flags', []),
                    'metadata': {
                        'filename': report.get('upload_metadata', {}).get('original_filename'),
                        'reviewed': report.get('radiologist_review', {}).get('reviewed', False)
                    }
                })
            
            # Sort by timestamp (most recent first)
            combined_timeline.sort(key=lambda x: x.get('timestamp', datetime.min), reverse=True)
            
            return combined_timeline
            
        except Exception as e:
            logger.error(f"Timeline generation failed: {e}")
            return []
    
    async def get_comprehensive_patient_data(self, user_id: str) -> Dict:
        """Get all patient data for comprehensive analysis"""
        try:
            # Get user profile
            user_profile = self.users_collection.find_one({'_id': ObjectId(user_id)})
            
            # Get recent interactions (last 10)
            recent_interactions = list(self.interactions_collection.find({
                'user_id': ObjectId(user_id)
            }).sort('started_at', -1).limit(10))
            
            # Get medical reports (last 20)
            medical_reports = list(self.medical_reports_collection.find({
                'user_id': ObjectId(user_id)
            }).sort('upload_metadata.upload_timestamp', -1).limit(20))
            
            # Get timeline
            timeline = await self.generate_patient_timeline(user_id, days_back=180)
            
            # Calculate health metrics
            health_metrics = self._calculate_comprehensive_health_metrics(
                user_profile, recent_interactions, medical_reports
            )
            
            return {
                'user_profile': user_profile,
                'recent_interactions': recent_interactions,
                'medical_reports': medical_reports,
                'timeline': timeline,
                'health_metrics': health_metrics,
                'summary_stats': {
                    'total_interactions': len(recent_interactions),
                    'total_reports': len(medical_reports),
                    'last_activity': timeline[0]['timestamp'] if timeline else None,
                    'profile_completion': user_profile.get('account_info', {}).get('profile_completion', 0) if user_profile else 0
                }
            }
            
        except Exception as e:
            logger.error(f"Comprehensive data retrieval failed: {e}")
            return {}
    
    async def _add_timeline_event(self, user_id: str, event_type: str, 
                                title: str, metadata: Dict = None) -> Dict:
        """Add event to patient timeline"""
        try:
            timeline_event = {
                'user_id': ObjectId(user_id),
                'event_type': event_type,
                'title': title,
                'description': metadata.get('description', '') if metadata else '',
                'timestamp': datetime.utcnow(),
                'metadata': metadata or {},
                'importance': self._determine_event_importance(event_type, metadata)
            }
            
            result = self.patient_timeline_collection.insert_one(timeline_event)
            timeline_event['_id'] = str(result.inserted_id)
            
            return timeline_event
            
        except Exception as e:
            logger.error(f"Timeline event creation failed: {e}")
            return {}
    
    def _format_medications(self, medications_list: List[str]) -> List[Dict]:
        """Format medications into structured format"""
        formatted_meds = []
        
        for med in medications_list:
            if isinstance(med, str):
                formatted_meds.append({
                    'name': med,
                    'dosage': 'As prescribed',
                    'frequency': 'As directed',
                    'start_date': 'Unknown',
                    'prescribing_doctor': 'Unknown',
                    'active': True,
                    'ai_info': f"Medication: {med} - Consult healthcare provider for specific information"
                })
            elif isinstance(med, dict):
                formatted_meds.append(med)
        
        return formatted_meds
    
    def _format_allergies(self, allergies_text: str) -> List[Dict]:
        """Format allergies into structured format"""
        if not allergies_text or allergies_text.lower() in ['none', 'no allergies', 'nkda']:
            return []
        
        allergies = []
        allergy_items = [item.strip() for item in allergies_text.split(',')]
        
        for item in allergy_items:
            allergies.append({
                'allergen': item,
                'reaction': 'Unknown',
                'severity': 'Unknown',
                'verified': False
            })
        
        return allergies
    
    def _format_surgeries(self, surgeries_text: str) -> List[Dict]:
        """Format surgical history into structured format"""
        if not surgeries_text or surgeries_text.lower() in ['none', 'no surgeries']:
            return []
        
        surgeries = []
        surgery_items = [item.strip() for item in surgeries_text.split(',')]
        
        for item in surgery_items:
            surgeries.append({
                'procedure': item,
                'date': 'Unknown',
                'hospital': 'Unknown',
                'complications': 'None reported',
                'outcome': 'Unknown'
            })
        
        return surgeries
    
    def _calculate_profile_completion(self, user_data: Dict) -> float:
        """Calculate profile completion percentage"""
        total_fields = 0
        completed_fields = 0
        
        # Essential fields with weights
        essential_fields = {
            'first_name': 1.0,
            'last_name': 1.0,
            'age': 1.0,
            'gender': 1.0,
            'phone': 0.8,
            'emergency_contact': 0.8,
            'chronic_conditions': 0.6,
            'current_medications': 0.6,
            'allergies_text': 0.4,
            'insurance_provider': 0.4
        }
        
        for field, weight in essential_fields.items():
            total_fields += weight
            value = user_data.get(field)
            
            if value:
                if isinstance(value, list) and len(value) > 0:
                    completed_fields += weight
                elif isinstance(value, str) and value.strip():
                    completed_fields += weight
                elif isinstance(value, (int, float)) and value > 0:
                    completed_fields += weight
        
        return min(completed_fields / total_fields, 1.0) if total_fields > 0 else 0.0
    
    def _calculate_comprehensive_health_metrics(self, user_profile: Dict, 
                                              interactions: List[Dict], 
                                              reports: List[Dict]) -> Dict:
        """Calculate comprehensive health metrics and risk scores"""
        metrics = {
            'overall_health_score': 75,  # Base score
            'risk_level': 'Low',
            'trend': 'Stable',
            'last_assessment_date': None,
            'improvement_areas': [],
            'positive_indicators': [],
            'risk_factors': [],
            'care_gaps': []
        }
        
        try:
            # Analyze recent interactions
            if interactions:
                latest_interaction = interactions[0]
                ai_assessment = latest_interaction.get('ai_assessment', {})
                
                # Extract urgency and conditions
                if ai_assessment.get('likely_conditions'):
                    primary_condition = ai_assessment['likely_conditions'][0]
                    triage_level = ai_assessment.get('triage_level', 'Self-Care Advice Possible')
                    
                    # Adjust health score based on triage level
                    if 'Emergency' in triage_level:
                        metrics['overall_health_score'] = 25
                        metrics['risk_level'] = 'Critical'
                    elif 'Within 24 Hours' in triage_level:
                        metrics['overall_health_score'] = 50
                        metrics['risk_level'] = 'High'
                    elif 'Self-Care' in triage_level:
                        metrics['overall_health_score'] = 80
                        metrics['risk_level'] = 'Low'
                
                metrics['last_assessment_date'] = latest_interaction.get('started_at')
            
            # Analyze medical reports for trends
            if reports:
                critical_findings = 0
                normal_findings = 0
                
                for report in reports:
                    flags = report.get('ai_analysis', {}).get('flags', [])
                    for flag in flags:
                        if flag.get('severity') in ['Critical', 'Warning']:
                            critical_findings += 1
                        else:
                            normal_findings += 1
                
                # Adjust score based on report findings
                if critical_findings > normal_findings:
                    metrics['overall_health_score'] -= 20
                    metrics['risk_level'] = 'High'
                elif normal_findings > critical_findings * 2:
                    metrics['overall_health_score'] += 10
            
            # Analyze lifestyle factors from profile
            if user_profile:
                lifestyle = user_profile.get('profile', {}).get('lifestyle', {})
                
                # Positive lifestyle factors
                if lifestyle.get('exercise', {}).get('frequency') in ['daily', 'regular']:
                    metrics['positive_indicators'].append('Regular exercise routine')
                    metrics['overall_health_score'] += 5
                
                if lifestyle.get('smoking', {}).get('status') == 'never':
                    metrics['positive_indicators'].append('Non-smoker')
                    metrics['overall_health_score'] += 5
                
                # Risk factors
                if lifestyle.get('smoking', {}).get('status') == 'current':
                    metrics['risk_factors'].append('Current smoker')
                    metrics['overall_health_score'] -= 15
                
                if lifestyle.get('alcohol', {}).get('frequency') == 'heavy':
                    metrics['risk_factors'].append('Heavy alcohol use')
                    metrics['overall_health_score'] -= 10
                
                # Sleep quality
                sleep_hours = lifestyle.get('sleep', {}).get('hours_per_night', 8)
                if sleep_hours < 6:
                    metrics['improvement_areas'].append('Sleep optimization needed')
                elif sleep_hours >= 7:
                    metrics['positive_indicators'].append('Adequate sleep duration')
            
            # Ensure score stays within bounds
            metrics['overall_health_score'] = max(0, min(100, metrics['overall_health_score']))
            
            return metrics
            
        except Exception as e:
            logger.error(f"Health metrics calculation failed: {e}")
            return metrics
    
    def _extract_medical_flags(self, ai_analysis: Dict) -> List[Dict]:
        """Extract and categorize medical flags from AI analysis"""
        flags = []
        
        # Extract from image analysis
        image_analysis = ai_analysis.get('image_analysis', {})
        if image_analysis.get('pathological_findings'):
            for finding in image_analysis['pathological_findings']:
                if finding.get('urgency') == 'immediate':
                    flags.append({
                        'severity': 'Critical',
                        'message': finding.get('finding'),
                        'location': finding.get('location'),
                        'recommendation': 'Immediate medical attention required',
                        'requires_immediate_attention': True
                    })
                elif finding.get('severity') in ['severe', 'moderate']:
                    flags.append({
                        'severity': 'Warning',
                        'message': finding.get('finding'),
                        'location': finding.get('location'),
                        'recommendation': 'Medical evaluation recommended',
                        'requires_immediate_attention': False
                    })
        
        # Extract from report analysis
        report_analysis = ai_analysis.get('report_analysis', {})
        if report_analysis.get('critical_values'):
            for critical_value in report_analysis['critical_values']:
                flags.append({
                    'severity': 'Critical',
                    'message': f"Critical lab value: {critical_value.get('parameter')}",
                    'location': 'Laboratory Results',
                    'recommendation': critical_value.get('immediate_action'),
                    'requires_immediate_attention': True
                })
        
        return flags
    
    def _determine_review_priority(self, ai_analysis: Dict) -> str:
        """Determine radiologist/physician review priority"""
        flags = self._extract_medical_flags(ai_analysis)
        
        # Check for critical flags
        if any(flag.get('severity') == 'Critical' for flag in flags):
            return 'stat'
        elif any(flag.get('severity') == 'Warning' for flag in flags):
            return 'urgent'
        elif ai_analysis.get('confidence_score', 1.0) < 0.7:
            return 'urgent'  # Low confidence requires expert review
        else:
            return 'routine'
    
    def _estimate_review_time(self, ai_analysis: Dict) -> str:
        """Estimate time for professional review"""
        priority = self._determine_review_priority(ai_analysis)
        
        time_estimates = {
            'stat': 'Within 1 hour',
            'urgent': 'Within 4 hours',
            'routine': 'Within 24 hours'
        }
        
        return time_estimates.get(priority, 'Within 24 hours')
    
    def _determine_event_importance(self, event_type: str, metadata: Dict = None) -> str:
        """Determine importance level of timeline events"""
        importance_mapping = {
            'profile_created': 'low',
            'triage_completed': 'medium',
            'report_analyzed': 'high',
            'emergency_detected': 'critical',
            'specialist_referral': 'high',
            'medication_change': 'medium',
            'appointment_scheduled': 'medium'
        }
        
        base_importance = importance_mapping.get(event_type, 'low')
        
        # Upgrade importance based on metadata
        if metadata:
            if metadata.get('urgency') in ['EMERGENCY', 'HIGH']:
                return 'critical'
            elif metadata.get('flags') and any(f.get('severity') == 'Critical' for f in metadata.get('flags', [])):
                return 'critical'
        
        return base_importance
    
    def _assess_interaction_quality(self, interaction_data: Dict) -> float:
        """Assess quality of triage interaction data"""
        quality_score = 0.0
        
        # Check conversation completeness
        conversation_log = interaction_data.get('conversation_log', [])
        if len(conversation_log) >= 5:
            quality_score += 0.4
        elif len(conversation_log) >= 3:
            quality_score += 0.2
        
        # Check symptom description quality
        description = interaction_data.get('user_description', '')
        if len(description) > 50:
            quality_score += 0.3
        elif len(description) > 20:
            quality_score += 0.2
        
        # Check assessment completeness
        if interaction_data.get('ai_assessment'):
            quality_score += 0.3
        
        return min(quality_score, 1.0)

# Global patient model will be initialized with database manager
patient_model = None

def initialize_patient_model(db_manager):
    """Initialize patient model with database manager"""
    global patient_model
    patient_model = AdvancedPatientModel(db_manager)
    return patient_model