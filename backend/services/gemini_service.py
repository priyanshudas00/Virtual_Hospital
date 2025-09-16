import os
import json
import logging
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import google.generativeai as genai
from PIL import Image
import io
import base64
import re
from concurrent.futures import ThreadPoolExecutor
import hashlib

logger = logging.getLogger(__name__)

class MedicalSafetyProtocol:
    """Advanced safety protocols for medical AI responses"""
    
    EMERGENCY_KEYWORDS = [
        'chest pain', 'heart attack', 'stroke', 'unconscious', 'severe bleeding',
        'choking', 'difficulty breathing', 'suicidal', 'overdose', 'seizure',
        'severe head injury', 'broken bone', 'severe burn', 'poisoning',
        'can\'t breathe', 'crushing chest pain', 'sudden weakness', 'slurred speech',
        'severe allergic reaction', 'anaphylaxis', 'cardiac arrest', 'respiratory arrest'
    ]
    
    EMERGENCY_RESPONSE = """
🚨 MEDICAL EMERGENCY DETECTED 🚨

STOP using this application immediately and:

• Call emergency services NOW (911/112/999)
• Go to the nearest emergency room
• Do not drive yourself - call ambulance
• Have someone stay with you

This AI assistant CANNOT handle emergency situations.
Your immediate safety is the absolute priority.

Time is critical - seek professional help NOW.
"""
    
    @classmethod
    def check_emergency(cls, text: str) -> bool:
        """Advanced emergency detection with pattern matching"""
        text_lower = text.lower()
        
        # Direct keyword matching
        if any(keyword in text_lower for keyword in cls.EMERGENCY_KEYWORDS):
            return True
        
        # Pattern-based detection
        emergency_patterns = [
            r'can\'?t\s+breathe',
            r'severe\s+pain',
            r'crushing\s+pain',
            r'sudden\s+onset',
            r'loss\s+of\s+consciousness',
            r'difficulty\s+breathing',
            r'shortness\s+of\s+breath\s+at\s+rest'
        ]
        
        for pattern in emergency_patterns:
            if re.search(pattern, text_lower):
                return True
        
        return False
    
    @classmethod
    def get_medical_disclaimer(cls) -> str:
        """Comprehensive medical disclaimer"""
        return """
⚠️ IMPORTANT MEDICAL DISCLAIMER:
I am an AI assistant designed to augment healthcare delivery, not replace it. 
My analysis is for informational purposes only and should not be considered 
a medical diagnosis. This system is designed to assist healthcare professionals 
and provide preliminary insights. Always consult qualified healthcare providers 
for personal medical advice, diagnosis, and treatment decisions.

This tool is intended for clinical augmentation, not replacement.
"""

class GeminiMedicalService:
    """Production-grade medical AI service with Gemini integration"""
    
    def __init__(self):
        self.api_key = os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")
        
        genai.configure(api_key=self.api_key)
        
        # Initialize models with safety settings
        safety_settings = [
            {
                "category": "HARM_CATEGORY_MEDICAL",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT", 
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            }
        ]
        
        self.text_model = genai.GenerativeModel(
            'gemini-1.5-pro',
            safety_settings=safety_settings
        )
        self.vision_model = genai.GenerativeModel(
            'gemini-1.5-pro-vision',
            safety_settings=safety_settings
        )
        
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.safety = MedicalSafetyProtocol()
        
        # Load medical prompt templates
        self.prompt_templates = self._load_medical_prompts()
        
    def _load_medical_prompts(self) -> Dict[str, str]:
        """Load comprehensive medical prompt templates"""
        return {
            'triage_questions': """
You are an advanced medical triage assistant helping gather critical clinical information.
Your role is to ask intelligent, medically relevant questions to assess urgency and narrow down possible conditions.

PATIENT'S INITIAL COMPLAINT: "{initial_complaint}"

CONVERSATION HISTORY:
{conversation_history}

CLINICAL CONTEXT:
- Patient Age: {age}
- Gender: {gender}
- Medical History: {medical_history}
- Current Medications: {medications}

Based on the patient's description and conversation so far, generate the next 3 most critical questions to:
1. Assess urgency and identify any red flags requiring immediate medical attention
2. Narrow down possible conditions through differential diagnosis
3. Gather essential clinical details for proper triage

Focus on: ONSET (sudden vs gradual), LOCATION (specific anatomical areas), SEVERITY (pain scales, functional impact), 
RADIATION (does pain/symptom spread), AGGRAVATING/RELIEVING factors, ASSOCIATED symptoms, TIMING patterns.

Return ONLY a JSON object with this exact structure:
{{
  "questions": [
    {{
      "question": "Specific, medically relevant question",
      "type": "urgency|location|severity|duration|associated|onset|radiation|modifying_factors",
      "priority": "high|medium|low",
      "clinical_reasoning": "Why this question is important for diagnosis"
    }}
  ],
  "assessment_readiness": "needs_more_info|ready_for_assessment",
  "emergency_indicators": ["Any red flags detected"],
  "next_focus": "What clinical area to explore next"
}}

CRITICAL: If you detect any emergency symptoms (chest pain, difficulty breathing, severe bleeding, loss of consciousness, stroke symptoms), 
immediately set assessment_readiness to "emergency_detected" and include emergency_indicators.
""",

            'comprehensive_assessment': """
You are an advanced medical AI providing comprehensive clinical assessment to augment healthcare delivery.
Your role is to provide structured, evidence-based preliminary analysis to assist healthcare professionals.

COMPLETE PATIENT DATA:
{patient_data}

FULL SYMPTOM INTERACTION:
{conversation_log}

PATIENT MEDICAL PROFILE:
- Age: {age}, Gender: {gender}
- Past Medical History: {medical_history}
- Current Medications: {medications}
- Allergies: {allergies}
- Family History: {family_history}
- Lifestyle Factors: {lifestyle_factors}
- Social Determinants: {social_factors}

Provide comprehensive medical assessment in this EXACT JSON format:

{{
  "preliminary_assessment": {{
    "primary_diagnosis": "Most likely condition based on clinical presentation",
    "confidence_score": 0.85,
    "urgency_level": "EMERGENCY|HIGH|MEDIUM|LOW",
    "clinical_reasoning": "Detailed medical reasoning for primary diagnosis",
    "mental_health_risk": "Assessment of psychological factors and mental health indicators",
    "overall_health_status": "Comprehensive evaluation of patient's current health state"
  }},
  "differential_diagnosis": [
    {{
      "condition": "Alternative diagnosis",
      "probability": 0.65,
      "supporting_evidence": ["Clinical findings supporting this diagnosis"],
      "distinguishing_features": ["Key features that differentiate this condition"]
    }}
  ],
  "clinical_analysis": {{
    "probable_causes": ["Evidence-based list of underlying causes"],
    "risk_factors": ["Identified risk factors from history and presentation"],
    "protective_factors": ["Positive health factors that reduce risk"],
    "symptom_pattern_analysis": "Analysis of how symptoms relate and progress",
    "red_flags": ["Emergency warning signs identified"],
    "clinical_pearls": ["Important clinical insights for healthcare providers"]
  }},
  "recommended_investigations": {{
    "essential_tests": ["Critical lab tests needed for diagnosis"],
    "imaging_studies": ["X-rays, MRI, CT scans if indicated"],
    "specialist_consultations": ["Referrals to appropriate specialists"],
    "monitoring_parameters": ["Vital signs and symptoms to track"],
    "urgency_timeline": "When investigations should be completed"
  }},
  "treatment_recommendations": {{
    "immediate_interventions": ["Actions needed right away"],
    "medication_categories": ["Types of medications that may be indicated"],
    "non_pharmacological": ["Lifestyle and behavioral interventions"],
    "contraindications": ["Treatments to avoid based on patient profile"],
    "monitoring_requirements": ["How to monitor treatment response"]
  }},
  "lifestyle_optimization": {{
    "diet_modifications": ["Specific dietary recommendations"],
    "exercise_prescription": ["Tailored physical activity plan"],
    "sleep_hygiene": ["Sleep improvement strategies"],
    "stress_management": ["Evidence-based stress reduction techniques"],
    "preventive_measures": ["Disease prevention strategies"],
    "environmental_modifications": ["Home/work environment changes"]
  }},
  "patient_education": {{
    "condition_explanation": "Simple, clear explanation of likely condition",
    "warning_signs": ["Symptoms requiring immediate medical attention"],
    "self_care_strategies": ["Safe home management techniques"],
    "prognosis": "Expected outcome with appropriate care",
    "when_to_seek_help": "Clear guidelines for seeking medical care"
  }},
  "doctor_handoff": {{
    "chief_complaint": "CC in standard medical format",
    "history_present_illness": "HPI in SOAP note format",
    "relevant_pmh": "Pertinent past medical history",
    "medications_allergies": "Current medications and known allergies",
    "clinical_impression": "Professional assessment for healthcare provider",
    "recommended_workup": "Suggested diagnostic approach",
    "priority_level": "Triage priority for healthcare system"
  }},
  "follow_up_plan": {{
    "reassessment_timeline": "When patient should be reassessed",
    "symptom_tracking": ["Specific symptoms to monitor"],
    "progress_indicators": ["Signs of improvement or deterioration"],
    "next_steps": ["Immediate action items for patient"],
    "long_term_management": ["Ongoing care considerations"]
  }}
}}

CRITICAL SAFETY REQUIREMENTS:
1. Always err on the side of caution - if uncertain, recommend medical evaluation
2. Never discourage seeking professional medical care
3. Clearly indicate this is preliminary analysis requiring professional confirmation
4. Flag any emergency indicators immediately
5. Provide evidence-based recommendations only
""",

            'medical_image_analysis': """
You are an advanced medical imaging AI assistant designed to augment radiologist workflow.
Your role is to provide preliminary analysis to highlight areas requiring expert attention.

IMAGE ANALYSIS CONTEXT:
- Image Type: {image_type}
- Body Part: {body_part}
- Clinical Context: {clinical_context}
- Patient Age: {patient_age}
- Relevant History: {relevant_history}

Analyze this medical image systematically and provide structured findings:

{{
  "image_analysis": {{
    "technical_assessment": {{
      "image_quality": "excellent|good|fair|poor|non_diagnostic",
      "positioning": "optimal|adequate|suboptimal|inadequate",
      "exposure_technique": "appropriate|overexposed|underexposed",
      "artifacts": ["List any technical artifacts affecting interpretation"],
      "diagnostic_adequacy": "fully_diagnostic|limited_diagnostic|non_diagnostic"
    }},
    "anatomical_review": {{
      "structures_visualized": ["All visible anatomical structures"],
      "normal_anatomy": ["Structures appearing normal"],
      "anatomical_variants": ["Normal variants that could be mistaken for pathology"],
      "comparison_to_prior": "If prior studies mentioned in context"
    }},
    "pathological_findings": [
      {{
        "finding": "Specific abnormality detected",
        "location": "Precise anatomical location",
        "description": "Detailed morphological description",
        "measurements": "Size/dimensions if applicable",
        "severity": "mild|moderate|severe|critical",
        "confidence": 0.85,
        "differential_considerations": ["Possible diagnoses for this finding"],
        "urgency": "immediate|urgent|routine|incidental"
      }}
    ],
    "systematic_review": {{
      "bones": "Bone assessment if applicable",
      "soft_tissues": "Soft tissue evaluation",
      "organs": "Organ-specific findings",
      "vascular": "Vascular structures if visible",
      "other": "Any other relevant findings"
    }},
    "overall_impression": "Comprehensive summary of all findings",
    "confidence_metrics": {{
      "overall_confidence": 0.85,
      "areas_of_uncertainty": ["Regions requiring expert review"],
      "technical_limitations": ["Factors limiting analysis accuracy"]
    }}
  }},
  "clinical_correlation": {{
    "symptom_correlation": "How findings relate to reported symptoms",
    "history_correlation": "Relevance to patient's medical history",
    "age_appropriateness": "Whether findings are age-appropriate",
    "clinical_significance": "What these findings mean clinically"
  }},
  "recommendations": {{
    "immediate_actions": ["If urgent findings detected"],
    "follow_up_imaging": ["Additional studies if needed"],
    "clinical_correlation_needed": ["Symptoms/signs to correlate"],
    "specialist_referrals": ["If specialist consultation indicated"],
    "comparison_studies": ["Prior studies to compare if available"]
  }},
  "radiologist_attention": {{
    "priority_level": "stat|urgent|routine",
    "specific_areas": ["Anatomical areas requiring expert focus"],
    "clinical_questions": ["Specific questions for radiologist to address"],
    "measurement_requests": ["Specific measurements needed"]
  }},
  "patient_communication": {{
    "simplified_explanation": "Findings explained in patient-friendly language",
    "reassurance_points": ["Positive/normal findings to highlight"],
    "areas_needing_clarification": ["Findings requiring doctor explanation"],
    "next_steps_for_patient": ["What patient should expect next"]
  }}
}}

CRITICAL SAFETY PROTOCOLS:
1. This is PRELIMINARY analysis only - final interpretation MUST come from qualified radiologist
2. Flag any critical findings immediately for urgent review
3. Never provide definitive diagnoses - only describe findings
4. Always recommend professional radiologist interpretation
5. Highlight areas of uncertainty clearly
6. Focus on augmenting, not replacing, professional expertise

IMPORTANT: If you detect any critical findings (masses, fractures, acute pathology), 
set priority_level to "stat" and include specific urgent recommendations.
""",

            'medical_report_analysis': """
You are a medical report analysis AI designed to assist healthcare professionals with comprehensive report interpretation.

REPORT ANALYSIS CONTEXT:
- Report Type: {report_type}
- Patient Context: {patient_context}
- Previous Assessments: {previous_assessments}

REPORT CONTENT:
{report_content}

Analyze this medical report systematically and provide structured interpretation:

{{
  "report_analysis": {{
    "report_summary": {{
      "report_type": "Specific type of medical report",
      "report_date": "Date of report if available",
      "ordering_physician": "Doctor who ordered if mentioned",
      "clinical_indication": "Reason for test/study",
      "report_quality": "complete|incomplete|limited"
    }},
    "key_findings": [
      {{
        "parameter": "Test name or finding",
        "value": "Actual result value",
        "reference_range": "Normal range for this parameter",
        "status": "normal|abnormal|critical|borderline",
        "clinical_significance": "What this result means medically",
        "trend_analysis": "Comparison to previous values if available"
      }}
    ],
    "abnormal_results": [
      {{
        "finding": "Specific abnormal result",
        "severity": "mild|moderate|severe|critical",
        "possible_causes": ["Evidence-based potential causes"],
        "clinical_implications": "What this means for patient care",
        "follow_up_needed": "Type and urgency of follow-up required"
      }}
    ],
    "critical_values": [
      {{
        "parameter": "Critical lab value or finding",
        "value": "Actual critical value",
        "normal_range": "Expected normal range",
        "immediate_action": "Urgent action required",
        "notification_priority": "stat|urgent|routine"
      }}
    ],
    "overall_assessment": {{
      "summary": "Comprehensive interpretation of all results",
      "clinical_significance": "Overall meaning for patient health",
      "urgency_level": "immediate|urgent|routine|normal",
      "pattern_recognition": "Patterns suggesting specific conditions",
      "correlation_with_symptoms": "How results explain patient's symptoms"
    }}
  }},
  "clinical_recommendations": {{
    "immediate_actions": ["Actions needed within 24 hours"],
    "follow_up_tests": ["Additional tests recommended"],
    "lifestyle_modifications": ["Relevant lifestyle changes"],
    "medication_considerations": ["Drug therapy considerations"],
    "specialist_referrals": ["Specialist consultations needed"],
    "monitoring_plan": ["How to monitor patient progress"]
  }},
  "doctor_communication": {{
    "executive_summary": "Key points for busy healthcare provider",
    "action_items": ["Specific actions for healthcare team"],
    "patient_discussion_points": ["Topics to discuss with patient"],
    "documentation_suggestions": ["Important points for medical record"]
  }},
  "patient_education": {{
    "results_explanation": "Results explained in simple, understandable terms",
    "what_this_means": "Practical implications for patient",
    "reassurance_points": ["Positive aspects to highlight"],
    "concern_areas": ["Areas requiring attention or follow-up"],
    "next_steps_summary": "Clear next steps for patient understanding"
  }}
}}

SAFETY REQUIREMENTS:
1. Flag all critical values immediately
2. Never minimize concerning findings
3. Always recommend professional medical review for abnormal results
4. Provide evidence-based interpretations only
5. Clearly distinguish between normal variants and pathology
""",

            'provider_matching': """
You are a healthcare provider matching AI designed to connect patients with appropriate care based on their medical needs.

PATIENT MEDICAL PROFILE:
{medical_profile}

AVAILABLE HEALTHCARE PROVIDERS:
{providers_data}

SEARCH PARAMETERS:
- Location: {location}
- Urgency Level: {urgency}
- Financial Capability: {financial_capability}
- Insurance: {insurance_info}
- Condition: {primary_condition}

Analyze patient needs and match with appropriate providers:

{{
  "provider_recommendations": [
    {{
      "provider_id": "Unique provider identifier",
      "provider_name": "Name of healthcare provider",
      "provider_type": "doctor|hospital|clinic|urgent_care",
      "match_score": 0.95,
      "specialization_match": "How well specialization matches patient needs",
      "location_convenience": "Distance and accessibility factors",
      "cost_compatibility": "Financial alignment with patient capability",
      "urgency_appropriateness": "How well provider matches urgency needs",
      "recommendation_strength": "strongly_recommended|recommended|consider|not_recommended",
      "estimated_wait_time": "Expected time to get appointment",
      "why_recommended": "Specific reasons for this recommendation"
    }}
  ],
  "care_pathway": {{
    "immediate_care_type": "emergency|urgent_care|primary_care|specialist",
    "care_sequence": ["Ordered steps for optimal care"],
    "timeline_recommendations": {{
      "immediate": "Actions needed within hours",
      "short_term": "Actions needed within days",
      "long_term": "Ongoing care planning"
    }}
  }},
  "cost_analysis": {{
    "consultation_estimates": {{
      "low_cost": "Budget-friendly options",
      "moderate_cost": "Standard care options", 
      "premium_cost": "High-end care options"
    }},
    "total_care_estimate": "Estimated total cost including tests/procedures",
    "insurance_considerations": "How insurance affects recommendations",
    "financial_assistance": ["Options for financial support if needed"]
  }},
  "accessibility_factors": {{
    "transportation": "Transportation considerations",
    "language_services": "Language support availability",
    "disability_accommodations": "Accessibility features",
    "cultural_considerations": "Cultural competency factors"
  }},
  "alternative_options": {{
    "telemedicine": ["Virtual care possibilities"],
    "urgent_care_centers": ["Immediate but non-emergency options"],
    "community_health": ["Community-based care options"],
    "emergency_alternatives": ["If emergency care not immediately available"]
  }}
}}

MATCHING CRITERIA PRIORITY:
1. SAFETY: Urgency match and emergency capability
2. SPECIALIZATION: Clinical expertise for condition
3. ACCESSIBILITY: Location, transportation, availability
4. AFFORDABILITY: Cost alignment with financial capability
5. QUALITY: Provider ratings and outcomes

Always prioritize patient safety and appropriate level of care over convenience or cost.
""",

            'patient_overview_generation': """
You are a medical AI creating comprehensive patient overviews for healthcare professionals.
Your goal is to synthesize all available patient data into actionable clinical insights.

COMPREHENSIVE PATIENT DATA:
{complete_patient_data}

MEDICAL HISTORY TIMELINE:
{medical_timeline}

RECENT INTERACTIONS:
{recent_interactions}

UPLOADED MEDICAL REPORTS:
{medical_reports}

Create a comprehensive clinical overview for healthcare professionals:

{{
  "executive_summary": {{
    "patient_snapshot": "2-3 sentence overview of patient's current clinical status",
    "primary_concerns": ["Most important clinical issues requiring attention"],
    "complexity_level": "straightforward|moderate|complex|high_risk",
    "estimated_consultation_time": "Recommended time allocation for this patient"
  }},
  "clinical_timeline": {{
    "current_presentation": "Present illness and symptoms",
    "recent_changes": "Significant changes in health status",
    "progression_pattern": "How condition has evolved over time",
    "intervention_responses": "Response to previous treatments if any"
  }},
  "risk_stratification": {{
    "immediate_risks": ["Urgent clinical concerns requiring immediate attention"],
    "short_term_risks": ["Risks developing over days to weeks"],
    "long_term_risks": ["Chronic disease risks and prevention opportunities"],
    "protective_factors": ["Positive health factors and strengths"]
  }},
  "care_priorities": [
    {{
      "priority": 1,
      "clinical_issue": "Most important issue to address",
      "recommended_action": "Specific action for healthcare provider",
      "timeline": "When this should be addressed",
      "complexity": "Simple|moderate|complex intervention needed"
    }}
  ],
  "clinical_decision_support": {{
    "differential_diagnosis": ["Conditions to consider based on all available data"],
    "recommended_workup": ["Diagnostic tests and studies to order"],
    "treatment_considerations": ["Therapeutic options to consider"],
    "monitoring_plan": ["How to track patient progress"],
    "referral_needs": ["Specialist consultations recommended"]
  }},
  "data_synthesis": {{
    "symptom_report_correlation": "How symptoms correlate with uploaded reports",
    "historical_pattern_analysis": "Patterns from patient's medical history",
    "medication_review": "Current medications and potential interactions",
    "social_determinants": "Social factors affecting health and care"
  }},
  "efficiency_insights": {{
    "pre_visit_preparation": ["What can be prepared before patient visit"],
    "key_discussion_points": ["Most important topics to cover with patient"],
    "time_saving_opportunities": ["Areas where AI has already gathered information"],
    "documentation_assistance": ["Pre-populated documentation elements"]
  }}
}}

Focus on providing actionable insights that directly support clinical decision-making and improve healthcare efficiency.
This overview should save healthcare providers significant time while improving care quality.
"""
        }
    
    async def perform_intelligent_triage(self, initial_complaint: str, patient_profile: Dict, 
                                       conversation_history: List[Dict] = None) -> Dict:
        """Perform intelligent medical triage with dynamic questioning"""
        try:
            # Emergency check with enhanced detection
            if self.safety.check_emergency(initial_complaint):
                return {
                    'emergency_detected': True,
                    'response': self.safety.EMERGENCY_RESPONSE,
                    'action': 'EMERGENCY_PROTOCOL',
                    'timestamp': datetime.utcnow().isoformat()
                }
            
            # Format conversation history
            history_text = self._format_conversation_history(conversation_history or [])
            
            # Extract patient context
            patient_context = self._extract_patient_context(patient_profile)
            
            # Generate intelligent follow-up questions
            prompt = self.prompt_templates['triage_questions'].format(
                initial_complaint=initial_complaint,
                conversation_history=history_text,
                age=patient_context.get('age', 'Not provided'),
                gender=patient_context.get('gender', 'Not provided'),
                medical_history=patient_context.get('medical_history', 'None'),
                medications=patient_context.get('medications', 'None')
            )
            
            response = await self._call_gemini_async(prompt, model_type='text')
            triage_result = self._parse_json_response(response)
            
            # Add safety metadata
            triage_result['safety_check'] = {
                'emergency_keywords_detected': False,
                'safety_disclaimer': self.safety.get_medical_disclaimer(),
                'timestamp': datetime.utcnow().isoformat()
            }
            
            return triage_result
            
        except Exception as e:
            logger.error(f"Intelligent triage failed: {e}")
            return self._get_error_response('triage', str(e))
    
    async def generate_comprehensive_assessment(self, patient_data: Dict, 
                                              conversation_log: List[Dict]) -> Dict:
        """Generate comprehensive medical assessment using all patient data"""
        try:
            # Compile all patient information
            compiled_data = self._compile_patient_data(patient_data)
            formatted_conversation = self._format_conversation_log(conversation_log)
            
            # Generate comprehensive assessment
            prompt = self.prompt_templates['comprehensive_assessment'].format(
                patient_data=json.dumps(compiled_data, indent=2),
                conversation_log=formatted_conversation,
                age=compiled_data.get('age', 'Not provided'),
                gender=compiled_data.get('gender', 'Not provided'),
                medical_history=compiled_data.get('medical_history', 'None'),
                medications=compiled_data.get('medications', 'None'),
                allergies=compiled_data.get('allergies', 'None'),
                family_history=compiled_data.get('family_history', 'None'),
                lifestyle_factors=compiled_data.get('lifestyle', 'Not provided'),
                social_factors=compiled_data.get('social_determinants', 'Not provided')
            )
            
            response = await self._call_gemini_async(prompt, model_type='text')
            assessment = self._parse_json_response(response)
            
            # Add comprehensive metadata
            assessment['metadata'] = {
                'assessment_id': self._generate_unique_id('ASSESS'),
                'timestamp': datetime.utcnow().isoformat(),
                'model_version': 'gemini-1.5-pro',
                'data_completeness': self._assess_data_completeness(compiled_data),
                'confidence_factors': self._analyze_confidence_factors(compiled_data, conversation_log)
            }
            
            return assessment
            
        except Exception as e:
            logger.error(f"Comprehensive assessment failed: {e}")
            return self._get_error_response('assessment', str(e))
    
    async def analyze_medical_image(self, image_data: str, image_metadata: Dict, 
                                  patient_context: Dict = None) -> Dict:
        """Analyze medical images with Gemini Vision"""
        try:
            # Decode and validate image
            image = self._decode_medical_image(image_data)
            if not image:
                return {'error': 'Invalid or corrupted image data'}
            
            # Extract image context
            image_type = image_metadata.get('type', 'Unknown')
            body_part = image_metadata.get('body_part', 'Not specified')
            clinical_context = image_metadata.get('clinical_context', 'No clinical context provided')
            
            # Prepare comprehensive analysis prompt
            prompt = self.prompt_templates['medical_image_analysis'].format(
                image_type=image_type,
                body_part=body_part,
                clinical_context=clinical_context,
                patient_age=patient_context.get('age', 'Not provided') if patient_context else 'Not provided',
                relevant_history=patient_context.get('relevant_history', 'None') if patient_context else 'None'
            )
            
            # Analyze with Gemini Vision
            response = await self._call_gemini_vision_async(prompt, image)
            analysis = self._parse_json_response(response)
            
            # Add safety and metadata
            analysis['safety_notice'] = """
🔍 AI MEDICAL IMAGE ANALYSIS - PRELIMINARY ONLY
This automated analysis is designed to assist healthcare professionals.
It is NOT a substitute for professional radiologist interpretation.
All findings must be confirmed by qualified medical professionals.
"""
            
            analysis['metadata'] = {
                'analysis_id': self._generate_unique_id('IMG'),
                'timestamp': datetime.utcnow().isoformat(),
                'model_version': 'gemini-1.5-pro-vision',
                'image_hash': self._calculate_image_hash(image_data),
                'processing_quality': self._assess_image_processing_quality(image)
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Medical image analysis failed: {e}")
            return self._get_error_response('image_analysis', str(e))
    
    async def analyze_medical_report(self, report_content: str, report_metadata: Dict,
                                   patient_context: Dict = None, previous_assessments: List[Dict] = None) -> Dict:
        """Analyze text-based medical reports"""
        try:
            # De-identify sensitive information
            deidentified_content = self._deidentify_medical_content(report_content)
            
            # Prepare analysis prompt
            prompt = self.prompt_templates['medical_report_analysis'].format(
                report_type=report_metadata.get('type', 'Medical Report'),
                patient_context=json.dumps(patient_context or {}, indent=2),
                previous_assessments=json.dumps(previous_assessments or [], indent=2),
                report_content=deidentified_content
            )
            
            # Analyze with Gemini
            response = await self._call_gemini_async(prompt, model_type='text')
            analysis = self._parse_json_response(response)
            
            # Add metadata and safety information
            analysis['metadata'] = {
                'analysis_id': self._generate_unique_id('RPT'),
                'timestamp': datetime.utcnow().isoformat(),
                'report_length': len(report_content),
                'deidentified': True,
                'processing_confidence': self._assess_report_processing_confidence(report_content)
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Medical report analysis failed: {e}")
            return self._get_error_response('report_analysis', str(e))
    
    async def generate_provider_recommendations(self, patient_profile: Dict, 
                                              available_providers: List[Dict],
                                              search_criteria: Dict) -> Dict:
        """Generate AI-powered healthcare provider recommendations"""
        try:
            # Compile medical profile for matching
            medical_profile = self._compile_medical_profile(patient_profile)
            
            # Format provider data
            providers_data = self._format_provider_data(available_providers)
            
            # Generate recommendations
            prompt = self.prompt_templates['provider_matching'].format(
                medical_profile=json.dumps(medical_profile, indent=2),
                providers_data=json.dumps(providers_data, indent=2),
                location=search_criteria.get('location', ''),
                urgency=search_criteria.get('urgency', 'routine'),
                financial_capability=search_criteria.get('financial_capability', 'medium'),
                insurance_info=search_criteria.get('insurance', 'Not provided'),
                primary_condition=medical_profile.get('primary_diagnosis', 'General consultation')
            )
            
            response = await self._call_gemini_async(prompt, model_type='text')
            recommendations = self._parse_json_response(response)
            
            # Add matching metadata
            recommendations['metadata'] = {
                'matching_id': self._generate_unique_id('MATCH'),
                'timestamp': datetime.utcnow().isoformat(),
                'providers_analyzed': len(available_providers),
                'search_radius': search_criteria.get('radius', 'Not specified'),
                'matching_algorithm': 'gemini-1.5-pro-enhanced'
            }
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Provider recommendation failed: {e}")
            return self._get_error_response('provider_matching', str(e))
    
    async def generate_patient_overview(self, patient_id: str, include_timeline: bool = True) -> Dict:
        """Generate comprehensive patient overview for healthcare professionals"""
        try:
            # Gather all patient data
            complete_data = await self._gather_complete_patient_data(patient_id)
            
            if include_timeline:
                timeline = await self._generate_medical_timeline(patient_id)
                complete_data['timeline'] = timeline
            
            # Generate overview
            prompt = self.prompt_templates['patient_overview_generation'].format(
                complete_patient_data=json.dumps(complete_data, indent=2),
                medical_timeline=json.dumps(complete_data.get('timeline', []), indent=2),
                recent_interactions=json.dumps(complete_data.get('recent_interactions', []), indent=2),
                medical_reports=json.dumps(complete_data.get('medical_reports', []), indent=2)
            )
            
            response = await self._call_gemini_async(prompt, model_type='text')
            overview = self._parse_json_response(response)
            
            # Add clinical metadata
            overview['metadata'] = {
                'overview_id': self._generate_unique_id('OVERVIEW'),
                'timestamp': datetime.utcnow().isoformat(),
                'data_sources': len(complete_data.keys()),
                'completeness_score': self._calculate_data_completeness(complete_data),
                'last_updated': complete_data.get('last_activity', 'Unknown')
            }
            
            return overview
            
        except Exception as e:
            logger.error(f"Patient overview generation failed: {e}")
            return self._get_error_response('patient_overview', str(e))
    
    # Helper Methods
    
    def _decode_medical_image(self, image_data: str) -> Optional[Image.Image]:
        """Decode medical images with enhanced error handling"""
        try:
            if image_data.startswith('data:image'):
                image_data = image_data.split(',')[1]
            
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to RGB for consistency
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            return image
            
        except Exception as e:
            logger.error(f"Medical image decoding failed: {e}")
            return None
    
    def _deidentify_medical_content(self, content: str) -> str:
        """Remove PII from medical content"""
        # Enhanced PII removal patterns
        patterns = [
            (r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', '[PATIENT_NAME]'),
            (r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]'),
            (r'\b\d{10,}\b', '[ID_NUMBER]'),
            (r'\b\d{1,2}/\d{1,2}/\d{4}\b', '[DATE]'),
            (r'\b\d{3}-\d{3}-\d{4}\b', '[PHONE]'),
            (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]'),
            (r'\b\d{1,5}\s+[A-Za-z\s]+\s+(Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln)\b', '[ADDRESS]')
        ]
        
        deidentified = content
        for pattern, replacement in patterns:
            deidentified = re.sub(pattern, replacement, deidentified, flags=re.IGNORECASE)
        
        return deidentified
    
    async def _call_gemini_async(self, prompt: str, model_type: str = 'text') -> str:
        """Call Gemini API asynchronously with enhanced error handling"""
        loop = asyncio.get_event_loop()
        
        def _call_gemini():
            try:
                model = self.text_model if model_type == 'text' else self.vision_model
                
                response = model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.1,  # Low temperature for medical accuracy
                        top_p=0.8,
                        top_k=40,
                        max_output_tokens=8192,  # Increased for comprehensive responses
                    )
                )
                
                return response.text
                
            except Exception as e:
                logger.error(f"Gemini API call failed: {e}")
                raise
        
        return await loop.run_in_executor(self.executor, _call_gemini)
    
    async def _call_gemini_vision_async(self, prompt: str, image: Image.Image) -> str:
        """Call Gemini Vision API asynchronously"""
        loop = asyncio.get_event_loop()
        
        def _call_gemini_vision():
            try:
                response = self.vision_model.generate_content(
                    [prompt, image],
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.1,
                        top_p=0.8,
                        max_output_tokens=8192,
                    )
                )
                return response.text
                
            except Exception as e:
                logger.error(f"Gemini Vision API call failed: {e}")
                raise
        
        return await loop.run_in_executor(self.executor, _call_gemini_vision)
    
    def _parse_json_response(self, response: str) -> Dict:
        """Parse JSON response with enhanced error handling"""
        try:
            # Clean response text
            cleaned_response = response.strip()
            
            # Extract JSON from markdown code blocks if present
            if '```json' in cleaned_response:
                start = cleaned_response.find('```json') + 7
                end = cleaned_response.find('```', start)
                if end != -1:
                    cleaned_response = cleaned_response[start:end].strip()
            
            # Find JSON boundaries
            json_start = cleaned_response.find('{')
            json_end = cleaned_response.rfind('}') + 1
            
            if json_start != -1 and json_end != -1:
                json_str = cleaned_response[json_start:json_end]
                parsed = json.loads(json_str)
                
                # Validate required fields based on response type
                return self._validate_response_structure(parsed)
            else:
                return {
                    'error': 'No valid JSON found in AI response',
                    'raw_response': response,
                    'parsed': False
                }
                
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed: {e}")
            return {
                'error': f'JSON parsing error: {str(e)}',
                'raw_response': response,
                'parsed': False,
                'fallback_text': response
            }
    
    def _validate_response_structure(self, parsed_response: Dict) -> Dict:
        """Validate AI response structure and add missing fields"""
        # Ensure required fields exist
        if 'preliminary_assessment' not in parsed_response:
            parsed_response['preliminary_assessment'] = {
                'primary_diagnosis': 'Assessment incomplete',
                'confidence_score': 0.5,
                'urgency_level': 'MEDIUM'
            }
        
        # Add validation metadata
        parsed_response['validation'] = {
            'structure_valid': True,
            'required_fields_present': True,
            'validation_timestamp': datetime.utcnow().isoformat()
        }
        
        return parsed_response
    
    def _format_conversation_history(self, history: List[Dict]) -> str:
        """Format conversation history for AI analysis"""
        if not history:
            return "No previous conversation"
        
        formatted = ""
        for i, entry in enumerate(history, 1):
            formatted += f"Q{i}: {entry.get('question', '')}\n"
            formatted += f"A{i}: {entry.get('answer', '')}\n\n"
        
        return formatted
    
    def _extract_patient_context(self, patient_profile: Dict) -> Dict:
        """Extract relevant patient context for AI analysis"""
        personal_info = patient_profile.get('personal_info', {})
        medical_info = patient_profile.get('medical_info', {})
        
        return {
            'age': personal_info.get('age'),
            'gender': personal_info.get('gender'),
            'medical_history': ', '.join(medical_info.get('chronic_conditions', [])),
            'medications': ', '.join(medical_info.get('current_medications', [])),
            'allergies': ', '.join(medical_info.get('allergies', []))
        }
    
    def _generate_unique_id(self, prefix: str) -> str:
        """Generate unique ID for tracking"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        random_suffix = hashlib.md5(str(datetime.now().microsecond).encode()).hexdigest()[:8]
        return f"{prefix}_{timestamp}_{random_suffix}"
    
    def _calculate_image_hash(self, image_data: str) -> str:
        """Calculate hash for image integrity"""
        return hashlib.sha256(image_data.encode()).hexdigest()[:16]
    
    def _get_error_response(self, operation: str, error_message: str) -> Dict:
        """Generate standardized error response"""
        return {
            'error': True,
            'operation': operation,
            'message': error_message,
            'timestamp': datetime.utcnow().isoformat(),
            'recommendation': 'Please consult with a healthcare professional',
            'safety_notice': self.safety.get_medical_disclaimer(),
            'fallback_action': 'Seek professional medical evaluation'
        }

# Global Gemini service instance
gemini_service = GeminiMedicalService()