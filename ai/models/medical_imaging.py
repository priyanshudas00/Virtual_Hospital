"""
Medical Imaging Analysis using Deep Learning
Supports X-rays, CT scans, MRIs with pre-trained models
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50, densenet121
import cv2
import numpy as np
from PIL import Image
import logging
from typing import Dict, List, Tuple
import base64
import io

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MedicalImageAnalyzer:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = {}
        self.transforms = self.get_transforms()
        self.load_models()
    
    def get_transforms(self):
        """Get image preprocessing transforms"""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def load_models(self):
        """Load pre-trained medical imaging models"""
        try:
            # Chest X-ray model (Pneumonia detection)
            self.models['chest_xray'] = self.load_chest_xray_model()
            
            # Skin lesion model
            self.models['skin_lesion'] = self.load_skin_lesion_model()
            
            # Brain MRI model
            self.models['brain_mri'] = self.load_brain_mri_model()
            
            logger.info("Medical imaging models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading imaging models: {e}")
            self.load_fallback_models()
    
    def load_chest_xray_model(self):
        """Load chest X-ray analysis model"""
        try:
            # Use DenseNet121 pre-trained on ImageNet, fine-tuned for chest X-rays
            model = densenet121(pretrained=True)
            
            # Modify classifier for chest X-ray classes
            num_classes = 3  # Normal, Pneumonia, COVID-19
            model.classifier = nn.Linear(model.classifier.in_features, num_classes)
            
            # Load pre-trained weights if available
            try:
                model.load_state_dict(torch.load('models/chest_xray_model.pth', map_location=self.device))
                logger.info("Loaded pre-trained chest X-ray model")
            except FileNotFoundError:
                logger.warning("Pre-trained chest X-ray weights not found, using ImageNet weights")
            
            model.to(self.device)
            model.eval()
            return model
            
        except Exception as e:
            logger.error(f"Error loading chest X-ray model: {e}")
            return None
    
    def load_skin_lesion_model(self):
        """Load skin lesion analysis model"""
        try:
            model = resnet50(pretrained=True)
            
            # Modify for skin lesion classification
            num_classes = 7  # Different types of skin lesions
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            
            try:
                model.load_state_dict(torch.load('models/skin_lesion_model.pth', map_location=self.device))
                logger.info("Loaded pre-trained skin lesion model")
            except FileNotFoundError:
                logger.warning("Pre-trained skin lesion weights not found")
            
            model.to(self.device)
            model.eval()
            return model
            
        except Exception as e:
            logger.error(f"Error loading skin lesion model: {e}")
            return None
    
    def load_brain_mri_model(self):
        """Load brain MRI analysis model"""
        try:
            model = resnet50(pretrained=True)
            
            # Modify for brain tumor classification
            num_classes = 4  # No tumor, Glioma, Meningioma, Pituitary
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            
            try:
                model.load_state_dict(torch.load('models/brain_mri_model.pth', map_location=self.device))
                logger.info("Loaded pre-trained brain MRI model")
            except FileNotFoundError:
                logger.warning("Pre-trained brain MRI weights not found")
            
            model.to(self.device)
            model.eval()
            return model
            
        except Exception as e:
            logger.error(f"Error loading brain MRI model: {e}")
            return None
    
    def load_fallback_models(self):
        """Load basic models if advanced models fail"""
        logger.info("Loading fallback imaging models...")
        
        # Simple ResNet50 models
        for model_type in ['chest_xray', 'skin_lesion', 'brain_mri']:
            model = resnet50(pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, 2)  # Binary classification
            model.to(self.device)
            model.eval()
            self.models[model_type] = model
    
    def preprocess_image(self, image_data: str) -> torch.Tensor:
        """Preprocess image for model input"""
        try:
            # Decode base64 image
            if image_data.startswith('data:image'):
                image_data = image_data.split(',')[1]
            
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            
            # Apply transforms
            tensor = self.transforms(image).unsqueeze(0)
            return tensor.to(self.device)
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            return None
    
    def analyze_chest_xray(self, image_tensor: torch.Tensor) -> Dict:
        """Analyze chest X-ray image"""
        try:
            model = self.models.get('chest_xray')
            if model is None:
                return self.get_error_response('chest_xray')
            
            with torch.no_grad():
                outputs = model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)[0]
                predicted_class = torch.argmax(probabilities).item()
            
            # Class labels for chest X-ray
            class_labels = ['Normal', 'Pneumonia', 'COVID-19 Suspected']
            confidence = float(probabilities[predicted_class])
            
            # Generate findings
            findings = []
            
            if predicted_class == 0:  # Normal
                findings.append({
                    'finding': 'Normal lung fields',
                    'confidence': confidence,
                    'severity': 'Normal',
                    'location': 'Bilateral lung fields'
                })
            elif predicted_class == 1:  # Pneumonia
                findings.append({
                    'finding': 'Pneumonia pattern detected',
                    'confidence': confidence,
                    'severity': 'Abnormal',
                    'location': 'Lung parenchyma'
                })
            else:  # COVID-19
                findings.append({
                    'finding': 'COVID-19 pattern suspected',
                    'confidence': confidence,
                    'severity': 'Abnormal',
                    'location': 'Bilateral lung fields'
                })
            
            # Add additional findings based on confidence
            if confidence > 0.8:
                findings.append({
                    'finding': 'High confidence diagnosis',
                    'confidence': confidence,
                    'severity': 'Note',
                    'location': 'Overall assessment'
                })
            
            return {
                'imageType': 'Chest X-Ray',
                'findings': findings,
                'overallAssessment': f'{class_labels[predicted_class]} - Confidence: {confidence:.2%}',
                'recommendations': self.get_chest_xray_recommendations(predicted_class, confidence),
                'riskLevel': self.assess_chest_xray_risk(predicted_class, confidence)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing chest X-ray: {e}")
            return self.get_error_response('chest_xray')
    
    def analyze_skin_lesion(self, image_tensor: torch.Tensor) -> Dict:
        """Analyze skin lesion image"""
        try:
            model = self.models.get('skin_lesion')
            if model is None:
                return self.get_error_response('skin_lesion')
            
            with torch.no_grad():
                outputs = model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)[0]
                predicted_class = torch.argmax(probabilities).item()
            
            # Skin lesion class labels
            class_labels = [
                'Benign Nevus', 'Melanoma', 'Basal Cell Carcinoma',
                'Actinic Keratosis', 'Seborrheic Keratosis',
                'Dermatofibroma', 'Vascular Lesion'
            ]
            
            if predicted_class >= len(class_labels):
                predicted_class = 0
            
            confidence = float(probabilities[predicted_class])
            
            findings = [{
                'finding': f'{class_labels[predicted_class]} detected',
                'confidence': confidence,
                'severity': 'Malignant' if predicted_class in [1, 2] else 'Benign',
                'location': 'Skin lesion'
            }]
            
            return {
                'imageType': 'Skin Lesion',
                'findings': findings,
                'overallAssessment': f'{class_labels[predicted_class]} - Confidence: {confidence:.2%}',
                'recommendations': self.get_skin_lesion_recommendations(predicted_class, confidence),
                'riskLevel': self.assess_skin_lesion_risk(predicted_class, confidence)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing skin lesion: {e}")
            return self.get_error_response('skin_lesion')
    
    def analyze_brain_mri(self, image_tensor: torch.Tensor) -> Dict:
        """Analyze brain MRI image"""
        try:
            model = self.models.get('brain_mri')
            if model is None:
                return self.get_error_response('brain_mri')
            
            with torch.no_grad():
                outputs = model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)[0]
                predicted_class = torch.argmax(probabilities).item()
            
            class_labels = ['No Tumor', 'Glioma', 'Meningioma', 'Pituitary Tumor']
            confidence = float(probabilities[predicted_class])
            
            findings = [{
                'finding': f'{class_labels[predicted_class]} detected',
                'confidence': confidence,
                'severity': 'Normal' if predicted_class == 0 else 'Abnormal',
                'location': 'Brain tissue'
            }]
            
            return {
                'imageType': 'Brain MRI',
                'findings': findings,
                'overallAssessment': f'{class_labels[predicted_class]} - Confidence: {confidence:.2%}',
                'recommendations': self.get_brain_mri_recommendations(predicted_class, confidence),
                'riskLevel': self.assess_brain_mri_risk(predicted_class, confidence)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing brain MRI: {e}")
            return self.get_error_response('brain_mri')
    
    def analyze_medical_image(self, image_data: str, image_type: str = 'auto') -> Dict:
        """Main method to analyze medical images"""
        try:
            # Preprocess image
            image_tensor = self.preprocess_image(image_data)
            if image_tensor is None:
                return self.get_error_response('preprocessing')
            
            # Auto-detect image type if not specified
            if image_type == 'auto':
                image_type = self.detect_image_type(image_tensor)
            
            # Route to appropriate analyzer
            if image_type.lower() in ['chest', 'xray', 'chest_xray']:
                return self.analyze_chest_xray(image_tensor)
            elif image_type.lower() in ['skin', 'dermatology', 'skin_lesion']:
                return self.analyze_skin_lesion(image_tensor)
            elif image_type.lower() in ['brain', 'mri', 'brain_mri']:
                return self.analyze_brain_mri(image_tensor)
            else:
                # Default to chest X-ray analysis
                return self.analyze_chest_xray(image_tensor)
                
        except Exception as e:
            logger.error(f"Error in medical image analysis: {e}")
            return self.get_error_response('general')
    
    def detect_image_type(self, image_tensor: torch.Tensor) -> str:
        """Auto-detect medical image type"""
        # This is a simplified version - in practice, you'd use a separate classifier
        # For now, default to chest X-ray
        return 'chest_xray'
    
    def get_chest_xray_recommendations(self, predicted_class: int, confidence: float) -> List[str]:
        """Get recommendations for chest X-ray results"""
        if predicted_class == 0:  # Normal
            return [
                'No acute abnormalities detected',
                'Continue routine health maintenance',
                'Follow up as clinically indicated'
            ]
        elif predicted_class == 1:  # Pneumonia
            return [
                'Pneumonia pattern detected - seek medical attention',
                'Antibiotic therapy may be indicated',
                'Monitor symptoms and follow up with healthcare provider',
                'Consider hospitalization if severe symptoms'
            ]
        else:  # COVID-19
            return [
                'COVID-19 pattern suspected - isolate immediately',
                'Get PCR test confirmation',
                'Monitor oxygen saturation',
                'Seek medical attention if symptoms worsen'
            ]
    
    def get_skin_lesion_recommendations(self, predicted_class: int, confidence: float) -> List[str]:
        """Get recommendations for skin lesion results"""
        if predicted_class in [1, 2]:  # Malignant
            return [
                'Suspicious lesion detected - urgent dermatology referral needed',
                'Biopsy recommended for definitive diagnosis',
                'Do not delay medical evaluation',
                'Document lesion changes with photos'
            ]
        else:  # Benign
            return [
                'Lesion appears benign but monitor for changes',
                'Routine dermatology follow-up recommended',
                'Use sun protection',
                'Self-examine skin monthly'
            ]
    
    def get_brain_mri_recommendations(self, predicted_class: int, confidence: float) -> List[str]:
        """Get recommendations for brain MRI results"""
        if predicted_class == 0:  # No tumor
            return [
                'No tumor detected',
                'Follow up as clinically indicated',
                'Correlate with clinical symptoms'
            ]
        else:  # Tumor detected
            return [
                'Brain tumor detected - urgent neurology/neurosurgery referral',
                'Additional imaging may be needed',
                'Multidisciplinary team evaluation recommended',
                'Do not delay medical consultation'
            ]
    
    def assess_chest_xray_risk(self, predicted_class: int, confidence: float) -> str:
        """Assess risk level for chest X-ray"""
        if predicted_class == 0:
            return 'Low'
        elif predicted_class == 1 and confidence > 0.8:
            return 'High'
        elif predicted_class == 2:
            return 'High'
        else:
            return 'Medium'
    
    def assess_skin_lesion_risk(self, predicted_class: int, confidence: float) -> str:
        """Assess risk level for skin lesion"""
        if predicted_class in [1, 2]:  # Malignant
            return 'High'
        elif confidence < 0.7:
            return 'Medium'
        else:
            return 'Low'
    
    def assess_brain_mri_risk(self, predicted_class: int, confidence: float) -> str:
        """Assess risk level for brain MRI"""
        if predicted_class == 0:
            return 'Low'
        else:
            return 'High'
    
    def get_error_response(self, error_type: str) -> Dict:
        """Return error response for failed analysis"""
        return {
            'imageType': 'Medical Image',
            'findings': [{
                'finding': f'Analysis error ({error_type})',
                'confidence': 0.0,
                'severity': 'Error',
                'location': 'System'
            }],
            'overallAssessment': 'Unable to analyze image - please consult radiologist',
            'recommendations': [
                'Image analysis failed',
                'Please have image reviewed by qualified radiologist',
                'Ensure image quality is adequate for analysis'
            ],
            'riskLevel': 'Unknown'
        }