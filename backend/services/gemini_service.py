import os
import json
import logging
import asyncio
from datetime import datetime
from typing import Dict, List, Any
import google.generativeai as genai
from PIL import Image
import io
import base64
import re

logger = logging.getLogger(__name__)

class MedicalSafetyProtocol:
    """Advanced safety protocols for medical AI"""
    
    EMERGENCY_KEYWORDS = [
        'chest pain', 'heart attack', 'stroke', 'unconscious', 'severe bleeding',
        'choking', 'difficulty breathing', 'suicidal', 'overdose', 'seizure',
        'severe head injury', 'broken bone', 'severe burn', 'poisoning',
        'can\'t breathe', 'crushing pain', 'sudden weakness', 'slurred speech'
    ]
    
    EMERGENCY_RESPONSE = """
🚨 MEDICAL EMERGENCY DETECTED 🚨

STOP using this application immediately and:
• Call emergency services NOW (911/112/999)
• Go to the nearest emergency room
• Do not drive yourself - call ambulance

This AI cannot handle emergencies. Seek professional help NOW.
"""
    
    @classmethod
    def check_emergency(cls, text: str) -> bool:
        """Check for emergency keywords"""
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in cls.EMERGENCY_KEYWORDS)
    
    @classmethod
    def get_disclaimer(cls) -> str:
        """Medical disclaimer"""
        return """
⚠️ MEDICAL DISCLAIMER:
I am an AI assistant, not a licensed medical professional. 
My analysis is for informational purposes only and should not 
be considered a medical diagnosis. Always consult qualified 
healthcare providers for personal medical advice.
"""

class GeminiMedicalService:
    """Production Gemini medical AI service"""
    
    def __init__(self):
        self.api_key = os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable required")
        
        genai.configure(api_key=self.api_key)
        self.text_model = genai.GenerativeModel('gemini-1.5-pro')
        self.vision_model = genai.GenerativeModel('gemini-1.5-pro-vision')
        self.safety = MedicalSafetyProtocol()
    
    async def analyze_symptoms(self, user_input: str, conversation_history: List[Dict] = None) -> Dict:
        """Intelligent symptom analysis with dynamic questioning"""
        try:
            # Emergency check
            if self.safety.check_emergency(user_input):
                return {
                    'emergency_detected': True,
                    'response': self.safety.EMERGENCY_RESPONSE
                }
            
            # Format conversation history
            history_text = ""
            if conversation_history:
                for entry in conversation_history:
                    history_text += f"Q: {entry.get('question', '')}\nA: {entry.get('answer', '')}\n"
            
            # Determine if we need more questions or can assess
            if not conversation_history or len(conversation_history) < 3:
                # Generate questions
                prompt = f"""
ROLE: You are a medical triage assistant gathering critical information.

PATIENT INPUT: "{user_input}"

CONVERSATION HISTORY:
{history_text}

TASK: Generate the next 3 most critical medical questions to:
1. Assess urgency and identify red flags
2. Understand symptom characteristics (onset, location, severity, radiation)
3. Gather associated symptoms and modifying factors

Return ONLY a JSON object:
{{
  "questions": [
    {{
      "question": "Specific medical question",
      "type": "urgency|location|severity|duration|associated",
      "priority": "high|medium|low"
    }}
  ],
  "assessment_ready": false
}}

Focus on: ONSET, LOCATION, SEVERITY, RADIATION, AGGRAVATING/RELIEVING factors.
"""
            else:
                # Generate assessment
                prompt = f"""
ROLE: Medical triage AI providing preliminary assessment.

COMPLETE CONVERSATION:
Initial: {user_input}
{history_text}

TASK: Provide comprehensive triage assessment in JSON format:

{{
  "assessment_ready": true,
  "preliminary_assessment": {{
    "likely_conditions": [
      {{
        "condition": "Specific condition name",
        "probability": "High|Medium|Low",
        "reasoning": "Clinical reasoning"
      }}
    ],
    "triage_level": "Emergency|Urgent - See Doctor within 24h|Self-Care Possible",
    "urgency_score": 8,
    "red_flags": ["Emergency warning signs if any"],
    "explanation": "Patient-friendly explanation"
  }},
  "next_steps": ["Specific recommendations"],
  "doctor_summary": "Concise summary for healthcare provider"
}}

CRITICAL: Always err on side of caution. If uncertain, recommend medical evaluation.
"""
            
            response = await self._call_gemini(prompt)
            return self._parse_json_response(response)
            
        except Exception as e:
            logger.error(f"Symptom analysis failed: {e}")
            return {'error': f'Analysis failed: {str(e)}'}
    
    async def analyze_text_report(self, report_text: str, report_type: str) -> Dict:
        """Analyze medical text reports"""
        try:
            # De-identify text
            deidentified_text = self._deidentify_text(report_text)
            
            prompt = f"""
ROLE: Medical report analysis expert.

REPORT TYPE: {report_type}
REPORT CONTENT:
{deidentified_text}

TASK: Analyze this medical report and provide structured findings:

{{
  "report_analysis": {{
    "summary": "3-4 sentence plain-English summary of key findings",
    "key_metrics": [
      {{
        "parameter": "Test name",
        "value": "Result value",
        "reference_range": "Normal range",
        "status": "normal|abnormal|critical"
      }}
    ],
    "abnormal_findings": [
      {{
        "finding": "Abnormal result",
        "severity": "mild|moderate|severe|critical",
        "clinical_significance": "What this means"
      }}
    ],
    "overall_impression": "Normal|Requires Review|Critical Findings Present",
    "recommendations": ["Next steps for patient/doctor"]
  }}
}}

Focus on clinical accuracy and patient safety.
"""
            
            response = await self._call_gemini(prompt)
            return self._parse_json_response(response)
            
        except Exception as e:
            logger.error(f"Text report analysis failed: {e}")
            return {'error': f'Analysis failed: {str(e)}'}
    
    async def analyze_medical_image(self, image_data: str, image_type: str, clinical_context: str = "") -> Dict:
        """Analyze medical images with Gemini Vision"""
        try:
            # Decode image
            image = self._decode_image(image_data)
            if not image:
                return {'error': 'Invalid image data'}
            
            prompt = f"""
ROLE: Medical imaging assistant to radiologist.

IMAGE TYPE: {image_type}
CLINICAL CONTEXT: {clinical_context}

TASK: Analyze this medical image systematically:

1. **Technical Quality**: Assess image quality and positioning
2. **Anatomical Review**: Describe visible structures
3. **Findings**: Identify any anomalies, irregularities, or notable features
4. **Location**: Specify precise anatomical locations
5. **Recommendations**: Areas requiring radiologist attention

Return JSON format:
{{
  "image_analysis": {{
    "technical_quality": "excellent|good|fair|poor",
    "anatomical_structures": ["List visible structures"],
    "findings": [
      {{
        "finding": "Specific observation",
        "location": "Precise anatomical location",
        "severity": "mild|moderate|severe",
        "confidence": "high|medium|low"
      }}
    ],
    "overall_impression": "Summary of key findings",
    "radiologist_attention": ["Areas requiring expert review"],
    "recommendations": ["Next steps"]
  }}
}}

CRITICAL: This is preliminary analysis. Final interpretation must be by qualified radiologist.
"""
            
            response = await self._call_gemini_vision(prompt, image)
            return self._parse_json_response(response)
            
        except Exception as e:
            logger.error(f"Image analysis failed: {e}")
            return {'error': f'Analysis failed: {str(e)}'}
    
    async def generate_doctor_report(self, patient_data: Dict, symptom_analysis: Dict, 
                                   report_analyses: List[Dict]) -> Dict:
        """Generate comprehensive doctor handoff report"""
        try:
            prompt = f"""
ROLE: Medical scribe creating patient summary for busy doctor.

PATIENT DATA:
{json.dumps(patient_data, indent=2)}

SYMPTOM ANALYSIS:
{json.dumps(symptom_analysis, indent=2)}

REPORT FINDINGS:
{json.dumps(report_analyses, indent=2)}

TASK: Create structured clinical summary in SOAP format:

{{
  "doctor_report": {{
    "chief_complaint": "CC in medical terminology",
    "history_present_illness": "HPI in SOAP format",
    "relevant_pmh": "Relevant past medical history",
    "medications_allergies": "Current meds and allergies",
    "objective_findings": "Key findings from reports",
    "assessment": "AI preliminary observations (NOT diagnosis)",
    "plan": "Suggested next steps for doctor",
    "red_flags": ["Emergency indicators if any"],
    "time_sensitive_items": ["Urgent items requiring attention"]
  }},
  "patient_education": {{
    "condition_explanation": "Simple explanation for patient",
    "self_care_instructions": ["Home care recommendations"],
    "warning_signs": ["When to seek immediate care"],
    "follow_up_timeline": "When to reassess"
  }}
}}

Use medical terminology for doctor section, simple language for patient section.
"""
            
            response = await self._call_gemini(prompt)
            return self._parse_json_response(response)
            
        except Exception as e:
            logger.error(f"Doctor report generation failed: {e}")
            return {'error': f'Report generation failed: {str(e)}'}
    
    def _decode_image(self, image_data: str) -> Image.Image:
        """Decode base64 image"""
        try:
            if image_data.startswith('data:image'):
                image_data = image_data.split(',')[1]
            
            image_bytes = base64.b64decode(image_data)
            return Image.open(io.BytesIO(image_bytes))
            
        except Exception as e:
            logger.error(f"Image decoding failed: {e}")
            return None
    
    def _deidentify_text(self, text: str) -> str:
        """Remove PII from medical text"""
        patterns = [
            (r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', '[PATIENT_NAME]'),
            (r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]'),
            (r'\b\d{1,2}/\d{1,2}/\d{4}\b', '[DATE]'),
            (r'\b\d{3}-\d{3}-\d{4}\b', '[PHONE]')
        ]
        
        deidentified = text
        for pattern, replacement in patterns:
            deidentified = re.sub(pattern, replacement, deidentified)
        
        return deidentified
    
    async def _call_gemini(self, prompt: str) -> str:
        """Call Gemini API"""
        try:
            response = self.text_model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    top_p=0.8,
                    max_output_tokens=4096,
                )
            )
            return response.text
        except Exception as e:
            logger.error(f"Gemini API call failed: {e}")
            raise
    
    async def _call_gemini_vision(self, prompt: str, image: Image.Image) -> str:
        """Call Gemini Vision API"""
        try:
            response = self.vision_model.generate_content(
                [prompt, image],
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=4096,
                )
            )
            return response.text
        except Exception as e:
            logger.error(f"Gemini Vision API call failed: {e}")
            raise
    
    def _parse_json_response(self, response: str) -> Dict:
        """Parse JSON from Gemini response"""
        try:
            # Extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start != -1 and json_end != -1:
                json_str = response[json_start:json_end]
                return json.loads(json_str)
            else:
                return {'error': 'No valid JSON in response', 'raw_response': response}
                
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed: {e}")
            return {'error': f'JSON parsing error: {str(e)}', 'raw_response': response}

# Global service instance
gemini_service = GeminiMedicalService()