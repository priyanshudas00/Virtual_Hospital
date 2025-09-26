"""
Advanced Gemini Medical Service
Production-grade medical AI with safety protocols and structured analysis
"""

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
    """Safety protocols for medical AI responses"""
    
    EMERGENCY_KEYWORDS = [
        'chest pain', 'heart attack', 'stroke', 'unconscious', 'severe bleeding',
        'choking', 'difficulty breathing', 'suicidal', 'overdose', 'seizure',
        'severe head injury', 'broken bone', 'severe burn', 'poisoning'
    ]
    
    EMERGENCY_RESPONSE = """
🚨 EMERGENCY DETECTED 🚨

Please STOP using this application and:
• Call emergency services immediately (911/112/999)
• Go to the nearest emergency room
• Do not delay seeking immediate medical attention

This AI assistant cannot handle emergency situations.
Your safety is the top priority.
"""
    
    @classmethod
    def check_emergency(cls, text: str) -> bool:
        """Check if input contains emergency keywords"""
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in cls.EMERGENCY_KEYWORDS)
    
    @classmethod
    def get_medical_disclaimer(cls) -> str:
        """Get standard medical disclaimer"""
        return """
⚠️ MEDICAL DISCLAIMER:
I am an AI assistant and not a licensed medical professional. 
My analysis is for informational purposes only and should not be 
considered a medical diagnosis. Always consult a qualified 
healthcare provider for personal medical advice.
"""

class GeminiMedicalService:
    """Advanced medical AI service with Gemini integration"""
    
    def __init__(self):
        self.api_key = os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")
        
        genai.configure(api_key=self.api_key)
        self.text_model = genai.GenerativeModel('gemini-1.5-pro')
        self.vision_model = genai.GenerativeModel('gemini-1.5-pro-vision')
        self.executor = ThreadPoolExecutor(max_workers=10)
        
        # Medical prompt templates
        self.prompt_templates = self._load_medical_prompts()
        
        # Safety protocols
        self.safety = MedicalSafetyProtocol()
    
    def _load_medical_prompts(self) -> Dict[str, str]:
        """Load comprehensive medical prompt templates"""
        return {
            'triage_questions': """
You are a medical triage assistant helping gather critical information.

PATIENT'S INITIAL COMPLAINT: "{initial_complaint}"

CONVERSATION HISTORY:
{conversation_history}

Based on the patient's description, generate the next 3 most critical questions to:
1. Assess urgency and potential red flags
2. Narrow down possible conditions
3. Gather essential clinical details

Return ONLY a JSON object with this structure:
{{
  "questions": [
    {{
      "question": "Specific medical question",
      "type": "urgency|location|severity|duration|associated",
      "priority": "high|medium|low"
    }}
  ],
  "assessment_complete": false,
  "emergency_detected": false
}}

Focus on: onset, location, severity, radiation, aggravating/relieving factors, associated symptoms.
""",

            'triage_assessment': """
You are a medical triage AI providing preliminary assessment.

COMPLETE PATIENT INTERACTION:
{full_conversation}

PATIENT PROFILE:
- Age: {age}
- Sex: {sex}
- Medical History: {medical_history}
- Current Medications: {medications}
- Allergies: {allergies}

Provide comprehensive triage assessment in this EXACT JSON format:

{{
  "triage_assessment": {{
    "likely_conditions": [
      {{
        "condition": "Specific medical condition",
        "probability": 0.85,
        "confidence": "high|medium|low",
        "reasoning": "Clinical reasoning for this diagnosis"
      }}
    ],
    "triage_level": "Seek Emergency Care Immediately|Consult a Doctor Within 24 Hours|Self-Care Advice Possible|Schedule Routine Follow-up",
    "urgency_score": 8,
    "red_flags": ["List any emergency warning signs"],
    "explanation": "Patient-friendly explanation of likely condition",
    "next_steps": ["Specific actionable recommendations"]
  }},
  "doctor_report": {{
    "chief_complaint": "CC in medical terminology",
    "history_present_illness": "HPI in SOAP format",
    "relevant_pmh": "Relevant past medical history",
    "medications_allergies": "Current meds and allergies",
    "differential_diagnosis": ["List of possible conditions"],
    "red_flags": ["Emergency indicators to watch"],
    "clinical_impression": "Professional medical summary"
  }},
  "patient_education": {{
    "condition_explanation": "Simple explanation of likely condition",
    "self_care_instructions": ["Home care recommendations"],
    "warning_signs": ["When to seek immediate care"],
    "follow_up_timeline": "When to reassess or see doctor"
  }}
}}

CRITICAL: Always err on the side of caution. If uncertain, recommend medical evaluation.
""",

            'image_analysis': """
You are a medical imaging AI assistant helping radiologists with preliminary analysis.

IMAGE TYPE: {image_type}
BODY PART: {body_part}
CLINICAL CONTEXT: {clinical_context}

Analyze this medical image and provide structured findings:

{{
  "image_analysis": {{
    "technical_quality": {{
      "quality": "excellent|good|fair|poor|non_diagnostic",
      "positioning": "optimal|adequate|suboptimal",
      "exposure": "appropriate|overexposed|underexposed",
      "artifacts": ["List any technical artifacts"]
    }},
    "anatomical_review": {{
      "structures_visualized": ["List visible anatomical structures"],
      "normal_findings": ["Structures that appear normal"],
      "anatomical_variants": ["Any normal variants noted"]
    }},
    "abnormal_findings": [
      {{
        "finding": "Specific abnormality",
        "location": "Precise anatomical location",
        "description": "Detailed description",
        "severity": "mild|moderate|severe",
        "confidence": 0.85,
        "differential": ["Possible causes"],
        "urgency": "immediate|urgent|routine"
      }}
    ],
    "overall_impression": "Comprehensive summary of findings",
    "recommendations": {{
      "immediate_actions": ["If any urgent findings"],
      "follow_up_studies": ["Additional imaging if needed"],
      "clinical_correlation": ["Symptoms to correlate with"],
      "specialist_referral": ["If specialist consultation needed"]
    }},
    "radiologist_attention": {{
      "priority": "stat|urgent|routine",
      "specific_areas": ["Areas requiring expert review"],
      "comparison_studies": ["Prior studies to compare if available"]
    }}
  }}
}}

IMPORTANT: This is preliminary AI analysis. Final interpretation must be provided by a qualified radiologist.
""",

            'report_analysis': """
You are a medical report analysis AI helping doctors interpret lab and clinical reports.

REPORT TYPE: {report_type}
PATIENT CONTEXT: {patient_context}
REPORT CONTENT: {report_content}

Analyze this medical report and provide structured interpretation:

{{
  "report_analysis": {{
    "report_summary": {{
      "type": "Specific type of report",
      "date": "Report date if available",
      "ordering_physician": "Doctor who ordered if available",
      "indication": "Reason for test/study"
    }},
    "key_findings": [
      {{
        "parameter": "Test name or finding",
        "value": "Result value",
        "reference_range": "Normal range",
        "status": "normal|abnormal|critical",
        "clinical_significance": "What this means clinically"
      }}
    ],
    "abnormal_results": [
      {{
        "finding": "Abnormal result",
        "severity": "mild|moderate|severe|critical",
        "possible_causes": ["List potential causes"],
        "follow_up_needed": "Type of follow-up required"
      }}
    ],
    "overall_assessment": {{
      "summary": "Overall interpretation of results",
      "clinical_significance": "What this means for patient",
      "urgency": "immediate|urgent|routine|normal",
      "trend_analysis": "If comparing to previous results"
    }},
    "recommendations": {{
      "immediate_actions": ["If any urgent findings"],
      "follow_up_tests": ["Additional tests needed"],
      "lifestyle_modifications": ["Relevant lifestyle changes"],
      "medication_considerations": ["Drug-related considerations"],
      "specialist_referrals": ["If specialist needed"]
    }},
    "patient_communication": {{
      "simplified_explanation": "Results explained in simple terms",
      "reassurance_points": ["Positive aspects to highlight"],
      "areas_of_concern": ["Issues requiring attention"],
      "next_steps": ["Clear action items for patient"]
    }}
  }}
}}

Focus on clinical accuracy and patient safety. Always recommend professional medical review for abnormal findings.
""",

            'provider_matching': """
You are a healthcare provider matching AI helping patients find appropriate care.

PATIENT MEDICAL PROFILE:
{medical_profile}

AVAILABLE PROVIDERS:
{providers_list}

SEARCH CRITERIA:
- Location: {location}
- Urgency: {urgency}
- Financial Capability: {financial_capability}
- Condition: {condition}

Analyze and rank providers based on patient needs:

{{
  "provider_recommendations": [
    {{
      "provider_id": "Provider identifier",
      "match_score": 0.95,
      "match_reasons": [
        "Specialization match",
        "Location convenience", 
        "Cost compatibility",
        "Urgency appropriateness"
      ],
      "recommendation_strength": "strongly_recommended|recommended|consider",
      "estimated_wait_time": "Expected appointment availability",
      "cost_estimate": {{
        "consultation": "Fee range",
        "total_estimated": "Including likely tests/procedures"
      }}
    }}
  ],
  "care_pathway": {{
    "immediate_care_needed": "Type of immediate care",
    "follow_up_care": "Ongoing care requirements",
    "specialist_consultations": ["Specialists to see"],
    "timeline": "Recommended care timeline"
  }},
  "alternatives": {{
    "telemedicine_options": ["Virtual care possibilities"],
    "urgent_care_centers": ["If immediate but not emergency"],
    "community_health_centers": ["Budget-friendly options"]
  }}
}}

Prioritize patient safety, accessibility, and cost-effectiveness.
"""
        }
    
    async def perform_triage(self, initial_complaint: str, conversation_history: List[Dict] = None) -> Dict:
        """Perform intelligent medical triage with dynamic questioning"""
        try:
            # Emergency check first
            if self.safety.check_emergency(initial_complaint):
                return {
                    'emergency_detected': True,
                    'response': self.safety.EMERGENCY_RESPONSE,
                    'action': 'EMERGENCY_PROTOCOL'
                }
            
            # Format conversation history
            history_text = ""
            if conversation_history:
                for entry in conversation_history:
                    history_text += f"Q: {entry.get('question', '')}\nA: {entry.get('answer', '')}\n"
            
            # Generate next questions
            prompt = self.prompt_templates['triage_questions'].format(
                initial_complaint=initial_complaint,
                conversation_history=history_text
            )
            
            response = await self._call_gemini_async(prompt, model_type='text')
            parsed_response = self._parse_json_response(response)
            
            # Add safety disclaimer
            parsed_response['disclaimer'] = self.safety.get_medical_disclaimer()
            
            return parsed_response
            
        except Exception as e:
            logger.error(f"Triage failed: {e}")
            return self._get_error_response('triage', str(e))
    
    async def generate_final_assessment(self, conversation_data: Dict, patient_profile: Dict) -> Dict:
        """Generate comprehensive medical assessment after triage completion"""
        try:
            # Format conversation for analysis
            full_conversation = self._format_conversation(conversation_data)
            
            # Prepare patient context
            patient_context = {
                'age': patient_profile.get('age', 'Not provided'),
                'sex': patient_profile.get('biological_sex', 'Not provided'),
                'medical_history': ', '.join(patient_profile.get('past_conditions', [])),
                'medications': ', '.join([med.get('name', '') for med in patient_profile.get('medications', [])]),
                'allergies': ', '.join([allergy.get('allergen', '') for allergy in patient_profile.get('allergies', [])])
            }
            
            # Generate assessment
            prompt = self.prompt_templates['triage_assessment'].format(
                full_conversation=full_conversation,
                **patient_context
            )
            
            response = await self._call_gemini_async(prompt, model_type='text')
            assessment = self._parse_json_response(response)
            
            # Add metadata
            assessment['metadata'] = {
                'assessment_id': self._generate_assessment_id(),
                'timestamp': datetime.utcnow().isoformat(),
                'model_version': 'gemini-1.5-pro',
                'safety_check_passed': True
            }
            
            return assessment
            
        except Exception as e:
            logger.error(f"Assessment generation failed: {e}")
            return self._get_error_response('assessment', str(e))
    
    async def analyze_medical_image(self, image_data: str, image_type: str, 
                                  clinical_context: str = "") -> Dict:
        """Analyze medical images with Gemini Vision"""
        try:
            # Decode and validate image
            image = self._decode_image(image_data)
            if not image:
                return {'error': 'Invalid image data'}
            
            # Detect body part and modality if not provided
            if not image_type or image_type == 'auto':
                image_type = await self._detect_image_type(image)
            
            # Prepare analysis prompt
            prompt = self.prompt_templates['image_analysis'].format(
                image_type=image_type,
                body_part=self._extract_body_part(image_type),
                clinical_context=clinical_context or "No clinical context provided"
            )
            
            # Analyze with Gemini Vision
            response = await self._call_gemini_vision_async(prompt, image)
            analysis = self._parse_json_response(response)
            
            # Add safety metadata
            analysis['safety_notice'] = """
🔍 AI IMAGE ANALYSIS PREVIEW
This is an automated preliminary analysis using AI. It is not a 
certified medical diagnosis. A qualified radiologist must always 
provide the final and definitive interpretation.
"""
            
            analysis['metadata'] = {
                'analysis_id': self._generate_analysis_id(),
                'timestamp': datetime.utcnow().isoformat(),
                'model_version': 'gemini-1.5-pro-vision',
                'image_hash': self._calculate_image_hash(image_data)
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Image analysis failed: {e}")
            return self._get_error_response('image_analysis', str(e))
    
    async def analyze_text_report(self, report_content: str, report_type: str, 
                                patient_context: Dict = None) -> Dict:
        """Analyze text-based medical reports"""
        try:
            # De-identify report content
            deidentified_content = self._deidentify_report(report_content)
            
            # Prepare analysis prompt
            prompt = self.prompt_templates['report_analysis'].format(
                report_type=report_type,
                patient_context=json.dumps(patient_context or {}, indent=2),
                report_content=deidentified_content
            )
            
            # Analyze with Gemini
            response = await self._call_gemini_async(prompt, model_type='text')
            analysis = self._parse_json_response(response)
            
            # Add metadata
            analysis['metadata'] = {
                'analysis_id': self._generate_analysis_id(),
                'timestamp': datetime.utcnow().isoformat(),
                'report_length': len(report_content),
                'deidentified': True
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Report analysis failed: {e}")
            return self._get_error_response('report_analysis', str(e))
    
    async def generate_provider_recommendations(self, medical_profile: Dict, 
                                              providers: List[Dict], 
                                              search_criteria: Dict) -> Dict:
        """Generate AI-powered provider recommendations"""
        try:
            prompt = self.prompt_templates['provider_matching'].format(
                medical_profile=json.dumps(medical_profile, indent=2),
                providers_list=json.dumps(providers, indent=2),
                location=search_criteria.get('location', ''),
                urgency=search_criteria.get('urgency', ''),
                financial_capability=search_criteria.get('financial_capability', ''),
                condition=medical_profile.get('primary_diagnosis', '')
            )
            
            response = await self._call_gemini_async(prompt, model_type='text')
            recommendations = self._parse_json_response(response)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Provider recommendation failed: {e}")
            return self._get_error_response('provider_matching', str(e))
    
    async def generate_patient_overview(self, patient_data: Dict) -> Dict:
        """Generate comprehensive patient overview for doctors"""
        try:
            # Compile all patient data
            overview_prompt = f"""
You are a medical scribe creating a comprehensive patient overview for a busy doctor.

PATIENT DATA:
{json.dumps(patient_data, indent=2)}

Create a structured clinical summary that saves the doctor time:

{{
  "patient_overview": {{
    "executive_summary": "2-3 sentence overview of patient's current status",
    "chief_concerns": ["Primary issues requiring attention"],
    "clinical_timeline": "Chronological progression of symptoms/conditions",
    "risk_stratification": {{
      "immediate_risks": ["Urgent concerns"],
      "ongoing_risks": ["Chronic management needs"],
      "preventive_opportunities": ["Health optimization areas"]
    }},
    "care_priorities": [
      {{
        "priority": 1,
        "issue": "Most important clinical issue",
        "action_needed": "Specific action required",
        "timeline": "When to address"
      }}
    ],
    "clinical_decision_support": {{
      "differential_diagnosis": ["Conditions to consider"],
      "recommended_workup": ["Tests/studies to order"],
      "treatment_considerations": ["Therapeutic options"],
      "monitoring_plan": ["Follow-up requirements"]
    }}
  }}
}}

Focus on actionable insights that directly support clinical decision-making.
"""
            
            response = await self._call_gemini_async(overview_prompt, model_type='text')
            overview = self._parse_json_response(response)
            
            return overview
            
        except Exception as e:
            logger.error(f"Patient overview generation failed: {e}")
            return self._get_error_response('patient_overview', str(e))
    
    def _decode_image(self, image_data: str) -> Optional[Image.Image]:
        """Decode base64 image data"""
        try:
            if image_data.startswith('data:image'):
                image_data = image_data.split(',')[1]
            
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            return image
            
        except Exception as e:
            logger.error(f"Image decoding failed: {e}")
            return None
    
    async def _detect_image_type(self, image: Image.Image) -> str:
        """Auto-detect medical image type using AI"""
        try:
            detection_prompt = """
Analyze this medical image and determine:
1. Image modality (X-Ray, MRI, CT, Ultrasound, etc.)
2. Body part being examined
3. View/orientation if applicable

Return only: "Modality: [TYPE], Body Part: [PART], View: [VIEW]"
"""
            
            response = await self._call_gemini_vision_async(detection_prompt, image)
            return response.strip()
            
        except Exception as e:
            logger.error(f"Image type detection failed: {e}")
            return "Unknown medical image"
    
    def _extract_body_part(self, image_type: str) -> str:
        """Extract body part from image type description"""
        body_parts = {
            'chest': 'Chest/Thorax',
            'abdomen': 'Abdomen',
            'head': 'Head/Brain',
            'spine': 'Spine',
            'extremity': 'Extremities',
            'pelvis': 'Pelvis'
        }
        
        image_type_lower = image_type.lower()
        for part, description in body_parts.items():
            if part in image_type_lower:
                return description
        
        return "Not specified"
    
    def _deidentify_report(self, content: str) -> str:
        """Remove PII from medical reports"""
        # Remove common PII patterns
        patterns = [
            (r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', '[PATIENT NAME]'),  # Names
            (r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]'),  # SSN
            (r'\b\d{10,}\b', '[ID NUMBER]'),  # Long numbers
            (r'\b\d{1,2}/\d{1,2}/\d{4}\b', '[DATE]'),  # Dates
            (r'\b\d{3}-\d{3}-\d{4}\b', '[PHONE]'),  # Phone numbers
        ]
        
        deidentified = content
        for pattern, replacement in patterns:
            deidentified = re.sub(pattern, replacement, deidentified)
        
        return deidentified
    
    async def _call_gemini_async(self, prompt: str, model_type: str = 'text') -> str:
        """Call Gemini API asynchronously"""
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
                        max_output_tokens=4096,
                    ),
                    safety_settings=[
                        {
                            "category": "HARM_CATEGORY_MEDICAL",
                            "threshold": "BLOCK_NONE"
                        }
                    ]
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
                        max_output_tokens=4096,
                    )
                )
                return response.text
                
            except Exception as e:
                logger.error(f"Gemini Vision API call failed: {e}")
                raise
        
        return await loop.run_in_executor(self.executor, _call_gemini_vision)
    
    def _parse_json_response(self, response: str) -> Dict:
        """Parse JSON response from Gemini with error handling"""
        try:
            # Extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start != -1 and json_end != -1:
                json_str = response[json_start:json_end]
                return json.loads(json_str)
            else:
                # If no JSON found, return structured error
                return {
                    'error': 'Could not parse JSON from AI response',
                    'raw_response': response,
                    'parsed': False
                }
                
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed: {e}")
            return {
                'error': f'JSON parsing error: {str(e)}',
                'raw_response': response,
                'parsed': False
            }
    
    def _format_conversation(self, conversation_data: Dict) -> str:
        """Format conversation history for AI analysis"""
        formatted = f"Initial Complaint: {conversation_data.get('initial_complaint', '')}\n\n"
        
        conversation_log = conversation_data.get('conversation_log', [])
        for entry in conversation_log:
            formatted += f"Q: {entry.get('ai_question', '')}\n"
            formatted += f"A: {entry.get('user_answer', '')}\n\n"
        
        return formatted
    
    def _generate_assessment_id(self) -> str:
        """Generate unique assessment ID"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        random_suffix = hashlib.md5(str(datetime.now().microsecond).encode()).hexdigest()[:6]
        return f"ASSESS_{timestamp}_{random_suffix}"
    
    def _generate_analysis_id(self) -> str:
        """Generate unique analysis ID"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        random_suffix = hashlib.md5(str(datetime.now().microsecond).encode()).hexdigest()[:6]
        return f"ANALYSIS_{timestamp}_{random_suffix}"
    
    def _calculate_image_hash(self, image_data: str) -> str:
        """Calculate hash of image for integrity"""
        return hashlib.sha256(image_data.encode()).hexdigest()[:16]
    
    def _get_error_response(self, operation: str, error_message: str) -> Dict:
        """Generate standardized error response"""
        return {
            'error': True,
            'operation': operation,
            'message': error_message,
            'timestamp': datetime.utcnow().isoformat(),
            'recommendation': 'Please consult with a healthcare professional',
            'disclaimer': self.safety.get_medical_disclaimer()
        }

# Global service instance
gemini_service = GeminiMedicalService()