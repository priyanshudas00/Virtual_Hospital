import os
import cv2
import numpy as np
from PIL import Image as PILImage, ImageEnhance, ImageFilter
import base64
import io
import logging
from typing import Dict, Optional
from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.media import Image as AgnoImage

logger = logging.getLogger(__name__)

class AdvancedImagingAnalyzer:
    """Advanced medical imaging analysis with Agno AI"""
    
    def __init__(self):
        # Initialize Agno AI agent for enhanced image analysis
        self.agent = Agent(
            model=Gemini(api_key=os.getenv('GEMINI_API_KEY')),
            tools=[DuckDuckGoTools()],
            show_tool_calls=True,
            markdown=True
        )
        
        # Medical imaging configurations
        self.imaging_types = {
            'xray': {
                'preprocessing': 'enhance_contrast',
                'analysis_focus': 'bone_structures_lung_fields',
                'common_findings': ['fractures', 'pneumonia', 'effusions']
            },
            'mri': {
                'preprocessing': 'noise_reduction',
                'analysis_focus': 'soft_tissue_contrast',
                'common_findings': ['tumors', 'inflammation', 'structural_abnormalities']
            },
            'ct': {
                'preprocessing': 'window_leveling',
                'analysis_focus': 'cross_sectional_anatomy',
                'common_findings': ['masses', 'bleeding', 'organ_pathology']
            },
            'ultrasound': {
                'preprocessing': 'speckle_reduction',
                'analysis_focus': 'real_time_structures',
                'common_findings': ['cysts', 'stones', 'fluid_collections']
            }
        }
    
    async def analyze_medical_image(self, image_data: str, image_type: str, 
                                  clinical_context: str = "") -> Dict:
        """Advanced medical image analysis using Agno AI"""
        try:
            # Preprocess image
            processed_image = self._preprocess_medical_image(image_data, image_type)
            if not processed_image:
                return {'error': 'Image preprocessing failed'}
            
            # Convert to Agno Image format
            agno_image = AgnoImage.from_pil(processed_image)
            
            # Create specialized medical imaging prompt
            imaging_config = self.imaging_types.get(image_type.lower(), self.imaging_types['xray'])
            
            prompt = f"""
You are an advanced medical imaging AI assistant specializing in {image_type.upper()} analysis.

CLINICAL CONTEXT: {clinical_context}
IMAGE TYPE: {image_type.upper()}
ANALYSIS FOCUS: {imaging_config['analysis_focus']}

Perform systematic medical image analysis:

1. **Technical Assessment**:
   - Image quality and positioning
   - Exposure and contrast adequacy
   - Artifacts or technical limitations

2. **Anatomical Review**:
   - Identify all visible anatomical structures
   - Assess normal anatomy
   - Note any anatomical variants

3. **Pathological Findings**:
   - Detect any abnormalities or lesions
   - Specify precise anatomical locations
   - Assess severity and characteristics
   - Provide differential considerations

4. **Clinical Correlation**:
   - Relate findings to clinical context
   - Suggest additional imaging if needed
   - Recommend specialist consultation if indicated

5. **Radiologist Priorities**:
   - Highlight areas requiring urgent attention
   - Flag critical findings
   - Suggest comparison with prior studies

Provide detailed, systematic analysis while emphasizing this is preliminary assessment requiring radiologist confirmation.

Common findings to look for in {image_type}: {', '.join(imaging_config['common_findings'])}
"""
            
            # Analyze with Agno AI agent
            response = await self.agent.run(prompt, images=[agno_image])
            
            # Extract structured findings
            structured_analysis = self._extract_structured_findings(response.content, image_type)
            
            # Add metadata
            structured_analysis['metadata'] = {
                'analysis_timestamp': datetime.now().isoformat(),
                'image_type': image_type,
                'preprocessing_applied': imaging_config['preprocessing'],
                'ai_model': 'agno-gemini-vision',
                'clinical_context': clinical_context
            }
            
            return structured_analysis
            
        except Exception as e:
            logger.error(f"Advanced image analysis failed: {e}")
            return {'error': f'Image analysis failed: {str(e)}'}
    
    def _preprocess_medical_image(self, image_data: str, image_type: str) -> Optional[PILImage.Image]:
        """Advanced medical image preprocessing"""
        try:
            # Decode image
            if image_data.startswith('data:image'):
                image_data = image_data.split(',')[1]
            
            image_bytes = base64.b64decode(image_data)
            image = PILImage.open(io.BytesIO(image_bytes))
            
            # Convert to RGB
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Apply type-specific preprocessing
            imaging_config = self.imaging_types.get(image_type.lower(), self.imaging_types['xray'])
            preprocessing = imaging_config['preprocessing']
            
            if preprocessing == 'enhance_contrast':
                # For X-rays: enhance contrast and reduce noise
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(1.5)
                image = image.filter(ImageFilter.MedianFilter(size=3))
                
            elif preprocessing == 'noise_reduction':
                # For MRI: reduce noise while preserving edges
                image_array = np.array(image)
                denoised = cv2.bilateralFilter(image_array, 9, 75, 75)
                image = PILImage.fromarray(denoised)
                
            elif preprocessing == 'window_leveling':
                # For CT: optimize window/level for soft tissue
                image_array = np.array(image)
                # Apply histogram equalization
                for i in range(3):  # RGB channels
                    image_array[:,:,i] = cv2.equalizeHist(image_array[:,:,i])
                image = PILImage.fromarray(image_array)
                
            elif preprocessing == 'speckle_reduction':
                # For Ultrasound: reduce speckle noise
                image_array = np.array(image)
                denoised = cv2.fastNlMeansDenoisingColored(image_array, None, 10, 10, 7, 21)
                image = PILImage.fromarray(denoised)
            
            return image
            
        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            return None
    
    def _extract_structured_findings(self, ai_response: str, image_type: str) -> Dict:
        """Extract structured findings from AI response"""
        try:
            # Parse the AI response and structure it
            findings = {
                'image_analysis': {
                    'technical_quality': 'good',
                    'anatomical_structures': [],
                    'findings': [],
                    'overall_impression': '',
                    'radiologist_attention': [],
                    'recommendations': []
                },
                'clinical_significance': '',
                'urgency_level': 'routine',
                'confidence_score': 0.8
            }
            
            # Extract key information from response
            response_lower = ai_response.lower()
            
            # Determine urgency based on keywords
            if any(word in response_lower for word in ['critical', 'urgent', 'immediate', 'emergency']):
                findings['urgency_level'] = 'urgent'
            elif any(word in response_lower for word in ['abnormal', 'concerning', 'suspicious']):
                findings['urgency_level'] = 'moderate'
            
            # Extract findings
            if 'finding' in response_lower or 'abnormal' in response_lower:
                findings['image_analysis']['findings'].append({
                    'finding': 'Abnormality detected - see full analysis',
                    'location': 'Multiple areas',
                    'severity': 'moderate',
                    'confidence': 'medium'
                })
            
            findings['image_analysis']['overall_impression'] = ai_response[:200] + "..."
            findings['clinical_significance'] = f"AI analysis of {image_type} completed. Radiologist review recommended."
            
            return findings
            
        except Exception as e:
            logger.error(f"Findings extraction failed: {e}")
            return {
                'image_analysis': {
                    'overall_impression': ai_response,
                    'findings': [],
                    'recommendations': ['Radiologist review required']
                },
                'error': 'Structured extraction failed'
            }

# Global imaging analyzer
imaging_analyzer = AdvancedImagingAnalyzer()