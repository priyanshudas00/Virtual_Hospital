import os
import json
import logging
from datetime import datetime
import google.generativeai as genai
from typing import Dict, List, Any, Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

class GeminiMedicalService:
    def __init__(self):
        self.api_key = os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")
        
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-1.5-pro')
        self.executor = ThreadPoolExecutor(max_workers=5)
        
        # Medical prompt templates
        self.prompt_templates = self.load_prompt_templates()
        
    def load_prompt_templates(self) -> Dict[str, str]:
        """Load structured medical prompt templates"""
        return {
            'initial_assessment': """
You are an advanced AI medical assistant providing comprehensive health assessment. 
CRITICAL: This is preliminary analysis only - NOT a replacement for professional medical care.

PATIENT PROFILE:
- Age: {age} years
- Gender: {gender}
- Location: {location}
- Occupation: {occupation}

CURRENT SYMPTOMS:
{symptoms_description}
Duration: {symptom_duration}
Pain Level: {pain_level}/10
Severity: {urgency}

MEDICAL HISTORY:
- Past Conditions: {medical_history}
- Current Medications: {current_medications}
- Allergies: {allergies}
- Previous Surgeries: {surgeries}
- Family History: {family_history}

LIFESTYLE FACTORS:
- Sleep: {sleep_hours} hours/night
- Exercise: {exercise_frequency}
- Diet: {diet_type}
- Smoking: {smoking_status}
- Alcohol: {alcohol_consumption}
- Stress Level: {stress_level}/10

INSURANCE & FINANCIAL:
- Provider: {insurance_provider}
- Financial Capability: {financial_capability}

Please provide a comprehensive medical assessment in this EXACT JSON format:

{{
  "preliminary_assessment": {{
    "primary_diagnosis": "Most likely condition",
    "confidence_score": 0.85,
    "urgency_level": "LOW/MEDIUM/HIGH/EMERGENCY",
    "mental_health_risk": "Assessment of psychological factors",
    "overall_health_status": "General health evaluation"
  }},
  "clinical_analysis": {{
    "probable_causes": ["List of possible underlying causes"],
    "differential_diagnosis": ["Alternative conditions to consider"],
    "risk_factors": ["Identified risk factors"],
    "symptom_pattern": "Analysis of symptom relationships"
  }},
  "recommended_investigations": {{
    "essential_tests": ["Critical lab tests needed"],
    "imaging_studies": ["X-rays, MRI, CT scans if needed"],
    "specialist_consultations": ["Referrals to specialists"],
    "monitoring_parameters": ["Vital signs to track"]
  }},
  "lifestyle_recommendations": {{
    "diet_modifications": ["Specific dietary changes"],
    "exercise_plan": ["Tailored physical activity"],
    "sleep_hygiene": ["Sleep improvement strategies"],
    "stress_management": ["Stress reduction techniques"],
    "preventive_measures": ["Disease prevention strategies"]
  }},
  "medication_guidance": {{
    "otc_recommendations": ["Over-the-counter options"],
    "prescription_categories": ["Types of medications that may be needed"],
    "drug_interactions": ["Potential medication conflicts"],
    "dosage_considerations": ["General dosing guidelines"]
  }},
  "referral_recommendations": {{
    "specialist_needed": ["Required specialist consultations"],
    "urgency_timeline": "When to seek care (immediate/days/weeks)",
    "preparation_notes": ["How to prepare for appointments"]
  }},
  "patient_education": {{
    "condition_explanation": "Simple explanation of likely condition",
    "warning_signs": ["Symptoms requiring immediate attention"],
    "self_care_strategies": ["Home management techniques"],
    "prognosis": "Expected outcome with proper care"
  }},
  "follow_up_plan": {{
    "reassessment_timeline": "When to reassess (days/weeks)",
    "symptom_tracking": ["Symptoms to monitor"],
    "progress_indicators": ["Signs of improvement/worsening"],
    "next_steps": ["Immediate action items"]
  }}
}}

Ensure all recommendations are evidence-based, patient-appropriate, and emphasize the need for professional medical consultation.
""",
            
            'report_analysis': """
You are analyzing uploaded medical reports for a patient. Provide comprehensive interpretation.

PATIENT CONTEXT:
{patient_context}

PREVIOUS AI ASSESSMENT:
{previous_assessment}

UPLOADED REPORT CONTENT:
{report_content}

REPORT TYPE: {report_type}
UPLOAD DATE: {upload_date}

Please analyze this medical report and provide updated assessment in this EXACT JSON format:

{{
  "report_summary": {{
    "report_type": "Type of medical report",
    "key_findings": ["Major findings from the report"],
    "abnormal_values": ["Values outside normal ranges"],
    "critical_results": ["Results requiring immediate attention"],
    "normal_findings": ["Reassuring normal results"]
  }},
  "clinical_interpretation": {{
    "significance": "What these results mean for the patient",
    "correlation_with_symptoms": "How results relate to reported symptoms",
    "progression_analysis": "Changes from previous reports if applicable",
    "diagnostic_confirmation": "Conditions confirmed or ruled out"
  }},
  "updated_assessment": {{
    "revised_diagnosis": "Updated primary diagnosis",
    "confidence_change": "How confidence has changed",
    "new_risk_factors": ["Newly identified risks"],
    "resolved_concerns": ["Issues that are now resolved"]
  }},
  "action_plan": {{
    "immediate_actions": ["Steps needed right away"],
    "follow_up_tests": ["Additional tests required"],
    "lifestyle_updates": ["Modified lifestyle recommendations"],
    "medication_adjustments": ["Changes to medication guidance"]
  }},
  "specialist_referrals": {{
    "required_specialists": ["Specialists to consult"],
    "urgency_level": "How quickly to see specialists",
    "referral_reasons": ["Why each specialist is needed"]
  }},
  "patient_communication": {{
    "simple_explanation": "Results explained in simple terms",
    "reassurance_points": ["Positive aspects to highlight"],
    "concern_areas": ["Areas requiring attention"],
    "next_steps_summary": "Clear next steps for patient"
  }}
}}

Focus on actionable insights and clear communication while maintaining medical accuracy.
""",
            
            'provider_matching': """
Based on the patient's medical assessment, recommend appropriate healthcare providers.

PATIENT MEDICAL PROFILE:
{medical_profile}

CURRENT DIAGNOSIS:
{current_diagnosis}

URGENCY LEVEL: {urgency_level}
FINANCIAL CAPABILITY: {financial_capability}
LOCATION: {location}

AVAILABLE PROVIDERS:
{available_providers}

Please provide provider recommendations in this EXACT JSON format:

{{
  "recommended_providers": [
    {{
      "provider_id": "Provider identifier",
      "provider_name": "Name of doctor/hospital",
      "match_score": 0.95,
      "recommendation_reason": "Why this provider is recommended",
      "urgency_match": "How well they match urgency needs",
      "cost_match": "How well they match financial capability",
      "specialization_match": "Relevant specializations"
    }}
  ],
  "care_pathway": {{
    "immediate_care": "What type of immediate care is needed",
    "follow_up_care": "Ongoing care requirements",
    "specialist_care": "Specialist consultations needed"
  }},
  "cost_estimates": {{
    "consultation_range": "Expected consultation costs",
    "investigation_costs": "Estimated test costs",
    "treatment_costs": "Potential treatment expenses"
  }},
  "timeline_recommendations": {{
    "immediate": "Actions needed within 24 hours",
    "short_term": "Actions needed within 1 week",
    "long_term": "Ongoing care plan"
  }}
}}

Prioritize patient safety, accessibility, and cost-effectiveness in recommendations.
"""
        }
    
    async def get_initial_assessment(self, patient_data: Dict) -> Dict:
        """Get comprehensive initial medical assessment from Gemini"""
        try:
            # Format the prompt with patient data
            prompt = self.prompt_templates['initial_assessment'].format(
                age=patient_data.get('age', 'Not provided'),
                gender=patient_data.get('gender', 'Not provided'),
                location=patient_data.get('location', 'Not provided'),
                occupation=patient_data.get('occupation', 'Not provided'),
                symptoms_description=self._format_symptoms(patient_data.get('symptoms', [])),
                symptom_duration=patient_data.get('symptom_duration', 'Not specified'),
                pain_level=patient_data.get('pain_level', 0),
                urgency=patient_data.get('urgency', 'Not specified'),
                medical_history=self._format_list(patient_data.get('medical_history', [])),
                current_medications=self._format_list(patient_data.get('current_medications', [])),
                allergies=self._format_list(patient_data.get('allergies', [])),
                surgeries=patient_data.get('surgeries_text', 'None reported'),
                family_history=patient_data.get('family_history_text', 'None reported'),
                sleep_hours=patient_data.get('sleep_hours', 'Not provided'),
                exercise_frequency=patient_data.get('exercise', 'Not provided'),
                diet_type=patient_data.get('diet', 'Not provided'),
                smoking_status=patient_data.get('smoking', 'Not provided'),
                alcohol_consumption=patient_data.get('alcohol', 'Not provided'),
                stress_level=patient_data.get('stress_level', 'Not provided'),
                insurance_provider=patient_data.get('insurance_provider', 'Not provided'),
                financial_capability=patient_data.get('financial_capability', 'Not provided')
            )
            
            # Get response from Gemini
            response = await self._call_gemini_async(prompt)
            
            # Parse JSON response
            assessment = self._parse_json_response(response)
            
            # Add metadata
            assessment['metadata'] = {
                'assessment_id': f"ASSESS_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'timestamp': datetime.utcnow().isoformat(),
                'model_version': 'gemini-1.5-pro',
                'patient_id': patient_data.get('user_id')
            }
            
            return assessment
            
        except Exception as e:
            logger.error(f"Initial assessment failed: {e}")
            return self._get_error_response('initial_assessment', str(e))
    
    async def analyze_medical_report(self, report_data: Dict, patient_context: Dict, previous_assessment: Dict = None) -> Dict:
        """Analyze uploaded medical reports with Gemini"""
        try:
            prompt = self.prompt_templates['report_analysis'].format(
                patient_context=json.dumps(patient_context, indent=2),
                previous_assessment=json.dumps(previous_assessment, indent=2) if previous_assessment else "No previous assessment",
                report_content=report_data.get('extracted_text', ''),
                report_type=report_data.get('report_type', 'Unknown'),
                upload_date=report_data.get('upload_date', datetime.now().isoformat())
            )
            
            response = await self._call_gemini_async(prompt)
            analysis = self._parse_json_response(response)
            
            # Add metadata
            analysis['metadata'] = {
                'analysis_id': f"REPORT_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'timestamp': datetime.utcnow().isoformat(),
                'report_id': report_data.get('upload_id'),
                'model_version': 'gemini-1.5-pro'
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Report analysis failed: {e}")
            return self._get_error_response('report_analysis', str(e))
    
    async def get_provider_recommendations(self, medical_profile: Dict, available_providers: List[Dict]) -> Dict:
        """Get healthcare provider recommendations from Gemini"""
        try:
            prompt = self.prompt_templates['provider_matching'].format(
                medical_profile=json.dumps(medical_profile, indent=2),
                current_diagnosis=medical_profile.get('primary_diagnosis', 'Not specified'),
                urgency_level=medical_profile.get('urgency_level', 'MEDIUM'),
                financial_capability=medical_profile.get('financial_capability', 'medium'),
                location=medical_profile.get('location', 'Not specified'),
                available_providers=json.dumps(available_providers, indent=2)
            )
            
            response = await self._call_gemini_async(prompt)
            recommendations = self._parse_json_response(response)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Provider recommendation failed: {e}")
            return self._get_error_response('provider_matching', str(e))
    
    async def _call_gemini_async(self, prompt: str) -> str:
        """Call Gemini API asynchronously"""
        loop = asyncio.get_event_loop()
        
        def _call_gemini():
            try:
                response = self.model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.1,  # Low temperature for medical accuracy
                        top_p=0.8,
                        top_k=40,
                        max_output_tokens=4096,
                    )
                )
                return response.text
            except Exception as e:
                logger.error(f"Gemini API call failed: {e}")
                raise
        
        return await loop.run_in_executor(self.executor, _call_gemini)
    
    def _format_symptoms(self, symptoms: List[str]) -> str:
        """Format symptoms list for prompt"""
        if not symptoms:
            return "No specific symptoms reported"
        return "\n".join([f"• {symptom}" for symptom in symptoms])
    
    def _format_list(self, items: List[str]) -> str:
        """Format list items for prompt"""
        if not items:
            return "None reported"
        return ", ".join(items)
    
    def _parse_json_response(self, response: str) -> Dict:
        """Parse JSON response from Gemini"""
        try:
            # Extract JSON from response (handle markdown formatting)
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start != -1 and json_end != -1:
                json_str = response[json_start:json_end]
                return json.loads(json_str)
            else:
                # If no JSON found, create structured response
                return {
                    'raw_response': response,
                    'parsed': False,
                    'error': 'Could not parse JSON from response'
                }
                
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed: {e}")
            return {
                'raw_response': response,
                'parsed': False,
                'error': f'JSON parsing error: {str(e)}'
            }
    
    def _get_error_response(self, assessment_type: str, error_message: str) -> Dict:
        """Generate error response structure"""
        return {
            'error': True,
            'assessment_type': assessment_type,
            'error_message': error_message,
            'timestamp': datetime.utcnow().isoformat(),
            'fallback_recommendations': [
                'Please consult with a healthcare professional',
                'Provide detailed symptom description to medical provider',
                'Seek immediate care if symptoms are severe'
            ]
        }

# Global Gemini service instance
gemini_service = GeminiMedicalService()