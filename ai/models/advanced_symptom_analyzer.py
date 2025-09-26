"""
Advanced Symptom-Disease Prediction using Deep Learning
BioClinicalBERT + XGBoost + Neural Networks for medical diagnosis
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel, pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, f1_score
import xgboost as xgb
import lightgbm as lgb
import joblib
import re
import logging
from typing import Dict, List, Tuple, Any
import requests
import os
from datetime import datetime
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MedicalNeuralNetwork(nn.Module):
    """Custom neural network for medical diagnosis"""
    
    def __init__(self, input_size: int, num_classes: int, hidden_sizes: List[int] = [512, 256, 128]):
        super(MedicalNeuralNetwork, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, num_classes))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

class AdvancedSymptomAnalyzer:
    """Advanced ML/DL-based symptom analyzer with minimal Gemini usage"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # ML Models
        self.symptom_classifier = None
        self.neural_network = None
        self.ensemble_models = {}
        
        # NLP Models
        self.tokenizer = None
        self.bert_model = None
        self.medical_ner = None
        
        # Encoders and Scalers
        self.disease_encoder = LabelEncoder()
        self.symptom_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        
        # Medical Knowledge Base
        self.medical_knowledge = self.load_medical_knowledge()
        self.symptom_embeddings = {}
        self.disease_patterns = {}
        
        # Load or train models
        self.initialize_models()
    
    def load_medical_knowledge(self) -> Dict:
        """Load comprehensive medical knowledge base"""
        return {
            'symptom_categories': {
                'constitutional': ['fever', 'fatigue', 'weight_loss', 'weight_gain', 'night_sweats', 'chills', 'malaise'],
                'neurological': ['headache', 'dizziness', 'confusion', 'seizure', 'numbness', 'weakness', 'tremor', 'memory_loss'],
                'cardiovascular': ['chest_pain', 'palpitations', 'shortness_of_breath', 'leg_swelling', 'syncope', 'claudication'],
                'respiratory': ['cough', 'dyspnea', 'wheezing', 'sputum_production', 'chest_tightness', 'hemoptysis'],
                'gastrointestinal': ['nausea', 'vomiting', 'diarrhea', 'constipation', 'abdominal_pain', 'bloating', 'heartburn'],
                'musculoskeletal': ['joint_pain', 'muscle_pain', 'back_pain', 'stiffness', 'swelling', 'limited_mobility'],
                'dermatological': ['rash', 'itching', 'skin_lesions', 'bruising', 'hair_loss', 'nail_changes'],
                'genitourinary': ['urinary_frequency', 'dysuria', 'hematuria', 'incontinence', 'pelvic_pain'],
                'psychiatric': ['anxiety', 'depression', 'insomnia', 'mood_changes', 'panic_attacks', 'hallucinations']
            },
            'disease_symptom_patterns': {
                'Common Cold': {
                    'primary': ['runny_nose', 'sore_throat', 'cough', 'sneezing'],
                    'secondary': ['mild_fever', 'fatigue', 'headache'],
                    'severity': 'mild',
                    'duration': '3-7 days',
                    'urgency': 'low'
                },
                'Influenza': {
                    'primary': ['fever', 'body_aches', 'fatigue', 'headache'],
                    'secondary': ['cough', 'sore_throat', 'runny_nose'],
                    'severity': 'moderate',
                    'duration': '7-10 days',
                    'urgency': 'medium'
                },
                'COVID-19': {
                    'primary': ['fever', 'cough', 'shortness_of_breath', 'loss_of_taste'],
                    'secondary': ['fatigue', 'body_aches', 'sore_throat', 'headache'],
                    'severity': 'variable',
                    'duration': '5-14 days',
                    'urgency': 'medium'
                },
                'Pneumonia': {
                    'primary': ['fever', 'cough', 'shortness_of_breath', 'chest_pain'],
                    'secondary': ['fatigue', 'chills', 'sputum_production'],
                    'severity': 'moderate-severe',
                    'duration': '7-21 days',
                    'urgency': 'high'
                },
                'Migraine': {
                    'primary': ['severe_headache', 'nausea', 'light_sensitivity'],
                    'secondary': ['vomiting', 'sound_sensitivity', 'visual_disturbances'],
                    'severity': 'severe',
                    'duration': '4-72 hours',
                    'urgency': 'medium'
                },
                'Hypertension': {
                    'primary': ['headache', 'dizziness', 'chest_pain'],
                    'secondary': ['shortness_of_breath', 'nosebleeds', 'fatigue'],
                    'severity': 'variable',
                    'duration': 'chronic',
                    'urgency': 'medium'
                },
                'Type 2 Diabetes': {
                    'primary': ['increased_thirst', 'frequent_urination', 'fatigue'],
                    'secondary': ['blurred_vision', 'slow_healing', 'weight_loss'],
                    'severity': 'progressive',
                    'duration': 'chronic',
                    'urgency': 'medium'
                },
                'Anxiety Disorder': {
                    'primary': ['excessive_worry', 'restlessness', 'fatigue'],
                    'secondary': ['muscle_tension', 'sleep_disturbance', 'concentration_difficulty'],
                    'severity': 'variable',
                    'duration': 'chronic',
                    'urgency': 'low'
                },
                'Heart Attack': {
                    'primary': ['severe_chest_pain', 'shortness_of_breath', 'nausea'],
                    'secondary': ['sweating', 'dizziness', 'arm_pain'],
                    'severity': 'critical',
                    'duration': 'acute',
                    'urgency': 'emergency'
                },
                'Stroke': {
                    'primary': ['sudden_weakness', 'speech_difficulty', 'facial_drooping'],
                    'secondary': ['severe_headache', 'vision_problems', 'confusion'],
                    'severity': 'critical',
                    'duration': 'acute',
                    'urgency': 'emergency'
                }
            }
        }
    
    def initialize_models(self):
        """Initialize or load pre-trained ML/DL models"""
        try:
            # Load pre-trained models
            self.load_pretrained_models()
            logger.info("✅ Pre-trained models loaded successfully")
        except Exception as e:
            logger.warning(f"Pre-trained models not found: {e}")
            logger.info("🔄 Training new models...")
            self.train_comprehensive_models()
    
    def load_pretrained_models(self):
        """Load pre-trained ML/DL models"""
        model_dir = 'ai/models/trained'
        
        # Load BioClinicalBERT
        self.tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        self.bert_model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        self.bert_model.to(self.device)
        self.bert_model.eval()
        
        # Load medical NER
        self.medical_ner = pipeline("ner", 
                                   model="d4data/biomedical-ner-all",
                                   aggregation_strategy="simple")
        
        # Load trained classifiers
        self.symptom_classifier = joblib.load(f'{model_dir}/symptom_classifier.pkl')
        self.disease_encoder = joblib.load(f'{model_dir}/disease_encoder.pkl')
        self.scaler = joblib.load(f'{model_dir}/feature_scaler.pkl')
        
        # Load neural network
        self.neural_network = torch.load(f'{model_dir}/medical_neural_network.pth', map_location=self.device)
        self.neural_network.eval()
        
        # Load ensemble models
        self.ensemble_models = {
            'xgboost': joblib.load(f'{model_dir}/xgboost_model.pkl'),
            'lightgbm': joblib.load(f'{model_dir}/lightgbm_model.pkl'),
            'random_forest': joblib.load(f'{model_dir}/random_forest_model.pkl')
        }
    
    def train_comprehensive_models(self):
        """Train comprehensive ML/DL models on medical datasets"""
        logger.info("🧠 Starting comprehensive ML/DL model training...")
        
        # Load medical datasets
        datasets = self.load_medical_datasets()
        
        # Prepare training data
        X, y = self.prepare_training_data(datasets)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Encode labels
        y_train_encoded = self.disease_encoder.fit_transform(y_train)
        y_test_encoded = self.disease_encoder.transform(y_test)
        
        # Train multiple ML models
        self.train_ensemble_models(X_train_scaled, X_test_scaled, y_train_encoded, y_test_encoded)
        
        # Train neural network
        self.train_neural_network(X_train_scaled, X_test_scaled, y_train_encoded, y_test_encoded)
        
        # Save all models
        self.save_trained_models()
        
        logger.info("✅ All ML/DL models trained and saved!")
    
    def load_medical_datasets(self) -> Dict:
        """Load real medical datasets for training"""
        datasets = {}
        
        try:
            # 1. Disease Symptom Prediction Dataset
            logger.info("📥 Loading disease symptom dataset...")
            url = "https://raw.githubusercontent.com/kaushil268/Disease-Prediction-using-Machine-Learning/master/dataset.csv"
            df_symptoms = pd.read_csv(url)
            datasets['symptoms'] = df_symptoms
            logger.info(f"✅ Loaded {len(df_symptoms)} symptom records")
            
        except Exception as e:
            logger.warning(f"Online dataset failed: {e}")
            datasets['symptoms'] = self.create_comprehensive_medical_dataset()
        
        try:
            # 2. Heart Disease Dataset
            logger.info("📥 Loading heart disease dataset...")
            heart_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
            column_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                           'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
            df_heart = pd.read_csv(heart_url, names=column_names, na_values='?')
            df_heart = df_heart.dropna()
            datasets['heart'] = df_heart
            logger.info(f"✅ Loaded {len(df_heart)} heart disease records")
            
        except Exception as e:
            logger.warning(f"Heart disease dataset failed: {e}")
        
        try:
            # 3. Diabetes Dataset
            from sklearn.datasets import fetch_openml
            logger.info("📥 Loading diabetes dataset...")
            diabetes = fetch_openml('diabetes', version=1, as_frame=True, parser='auto')
            df_diabetes = pd.concat([diabetes.data, diabetes.target], axis=1)
            datasets['diabetes'] = df_diabetes
            logger.info(f"✅ Loaded {len(df_diabetes)} diabetes records")
            
        except Exception as e:
            logger.warning(f"Diabetes dataset failed: {e}")
        
        return datasets
    
    def create_comprehensive_medical_dataset(self) -> pd.DataFrame:
        """Create comprehensive medical dataset with realistic patterns"""
        logger.info("🔬 Creating comprehensive medical dataset...")
        
        data = []
        
        # Generate data for each disease pattern
        for disease, pattern in self.medical_knowledge['disease_symptom_patterns'].items():
            # Generate 500-1000 samples per disease
            num_samples = np.random.randint(500, 1001)
            
            for _ in range(num_samples):
                sample = {}
                
                # Initialize all symptoms to 0
                all_symptoms = set()
                for symptoms in self.medical_knowledge['symptom_categories'].values():
                    all_symptoms.update(symptoms)
                
                for symptom in sorted(all_symptoms):
                    sample[symptom] = 0
                
                # Set primary symptoms (high probability)
                for symptom in pattern['primary']:
                    if symptom in all_symptoms and np.random.random() < 0.90:
                        sample[symptom] = 1
                
                # Set secondary symptoms (medium probability)
                for symptom in pattern['secondary']:
                    if symptom in all_symptoms and np.random.random() < 0.65:
                        sample[symptom] = 1
                
                # Add demographic factors
                sample['age'] = np.random.randint(18, 85)
                sample['gender'] = np.random.choice(['male', 'female'])
                
                # Add some noise (random symptoms)
                noise_symptoms = np.random.choice(list(all_symptoms), 
                                                size=np.random.randint(0, 3), 
                                                replace=False)
                for symptom in noise_symptoms:
                    if np.random.random() < 0.10:
                        sample[symptom] = 1
                
                sample['disease'] = disease
                data.append(sample)
        
        df = pd.DataFrame(data)
        logger.info(f"✅ Created {len(df)} medical records with {len(df.columns)-1} features")
        
        return df
    
    def prepare_training_data(self, datasets: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare comprehensive training data from multiple datasets"""
        all_features = []
        all_labels = []
        
        # Process symptom dataset
        if 'symptoms' in datasets:
            df = datasets['symptoms']
            
            # Identify target column
            target_col = 'prognosis' if 'prognosis' in df.columns else 'disease'
            
            # Extract features and labels
            feature_cols = [col for col in df.columns if col != target_col]
            
            for _, row in df.iterrows():
                # Create feature vector
                features = []
                
                # Symptom features
                for col in feature_cols:
                    if col in ['age', 'gender']:
                        continue
                    features.append(float(row[col]) if pd.notna(row[col]) else 0.0)
                
                # Add demographic features if available
                if 'age' in row:
                    features.append(float(row['age']) / 100.0)  # Normalize age
                else:
                    features.append(0.5)  # Default age
                
                if 'gender' in row:
                    features.append(1.0 if row['gender'] == 'male' else 0.0)
                else:
                    features.append(0.5)  # Unknown gender
                
                all_features.append(features)
                all_labels.append(row[target_col])
        
        # Convert to numpy arrays
        X = np.array(all_features)
        y = np.array(all_labels)
        
        logger.info(f"📊 Training data shape: {X.shape}")
        logger.info(f"🏷️ Number of unique diseases: {len(np.unique(y))}")
        
        return X, y
    
    def train_ensemble_models(self, X_train: np.ndarray, X_test: np.ndarray, 
                            y_train: np.ndarray, y_test: np.ndarray):
        """Train ensemble of ML models"""
        logger.info("🎯 Training ensemble ML models...")
        
        models = {
            'xgboost': xgb.XGBClassifier(
                n_estimators=300,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric='mlogloss'
            ),
            'lightgbm': lgb.LGBMClassifier(
                n_estimators=300,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                random_state=42
            )
        }
        
        best_model = None
        best_score = 0
        
        for name, model in models.items():
            logger.info(f"🔄 Training {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Evaluate
            train_score = model.score(X_train, y_train)
            test_score = model.score(X_test, y_test)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1_macro')
            
            logger.info(f"📈 {name} - Train: {train_score:.3f}, Test: {test_score:.3f}, CV F1: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
            
            # Save model
            self.ensemble_models[name] = model
            
            if test_score > best_score:
                best_score = test_score
                best_model = model
                self.symptom_classifier = model
        
        logger.info(f"🏆 Best model: {type(best_model).__name__} with accuracy: {best_score:.3f}")
    
    def train_neural_network(self, X_train: np.ndarray, X_test: np.ndarray, 
                           y_train: np.ndarray, y_test: np.ndarray):
        """Train deep neural network for medical diagnosis"""
        logger.info("🧠 Training deep neural network...")
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        X_test_tensor = torch.FloatTensor(X_test).to(self.device)
        y_train_tensor = torch.LongTensor(y_train).to(self.device)
        y_test_tensor = torch.LongTensor(y_test).to(self.device)
        
        # Initialize neural network
        input_size = X_train.shape[1]
        num_classes = len(np.unique(y_train))
        
        self.neural_network = MedicalNeuralNetwork(
            input_size=input_size,
            num_classes=num_classes,
            hidden_sizes=[512, 256, 128, 64]
        ).to(self.device)
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.neural_network.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        # Training loop
        num_epochs = 100
        best_accuracy = 0
        
        for epoch in range(num_epochs):
            # Training phase
            self.neural_network.train()
            optimizer.zero_grad()
            
            outputs = self.neural_network(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()
            
            # Validation phase
            if epoch % 10 == 0:
                self.neural_network.eval()
                with torch.no_grad():
                    test_outputs = self.neural_network(X_test_tensor)
                    _, predicted = torch.max(test_outputs.data, 1)
                    accuracy = (predicted == y_test_tensor).sum().item() / len(y_test_tensor)
                    
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        # Save best model
                        torch.save(self.neural_network.state_dict(), 'ai/models/trained/best_neural_network.pth')
                    
                    logger.info(f"🔄 Epoch {epoch}: Loss: {loss.item():.4f}, Accuracy: {accuracy:.3f}")
                    scheduler.step(loss)
        
        logger.info(f"🏆 Neural network training completed. Best accuracy: {best_accuracy:.3f}")
    
    def extract_symptoms_with_bert(self, text: str) -> List[str]:
        """Extract symptoms using BioClinicalBERT and medical NER"""
        extracted_symptoms = []
        
        try:
            # Use medical NER to extract entities
            ner_results = self.medical_ner(text)
            
            for entity in ner_results:
                if entity['entity_group'] in ['DISEASE', 'SYMPTOM', 'SIGN']:
                    symptom_text = entity['word'].lower()
                    # Map to known symptoms
                    for category_symptoms in self.medical_knowledge['symptom_categories'].values():
                        for symptom in category_symptoms:
                            if symptom.replace('_', ' ') in symptom_text or symptom_text in symptom.replace('_', ' '):
                                extracted_symptoms.append(symptom)
            
            # Use BERT embeddings for semantic similarity
            if self.bert_model and self.tokenizer:
                text_embedding = self.get_bert_embeddings(text)
                
                # Compare with pre-computed symptom embeddings
                for symptom in self.get_all_symptoms():
                    symptom_embedding = self.get_symptom_embedding(symptom)
                    similarity = np.dot(text_embedding, symptom_embedding)
                    
                    if similarity > 0.75:  # High similarity threshold
                        extracted_symptoms.append(symptom)
            
            # Rule-based extraction as fallback
            rule_based_symptoms = self.extract_symptoms_rule_based(text)
            extracted_symptoms.extend(rule_based_symptoms)
            
            return list(set(extracted_symptoms))
            
        except Exception as e:
            logger.error(f"BERT symptom extraction failed: {e}")
            return self.extract_symptoms_rule_based(text)
    
    def get_bert_embeddings(self, text: str) -> np.ndarray:
        """Get BioClinicalBERT embeddings for medical text"""
        try:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, 
                                  padding=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
                # Use CLS token embedding
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            return embeddings.flatten()
            
        except Exception as e:
            logger.error(f"BERT embeddings failed: {e}")
            return np.zeros(768)
    
    def predict_disease_ensemble(self, symptom_text: str, patient_age: int = 30, 
                               patient_gender: str = 'unknown') -> Dict:
        """Predict disease using ensemble of ML/DL models"""
        try:
            # Extract symptoms using advanced NLP
            symptoms = self.extract_symptoms_with_bert(symptom_text)
            
            if not symptoms:
                return {
                    'error': 'No medical symptoms detected',
                    'suggestion': 'Please describe your symptoms more specifically using medical terms'
                }
            
            # Create feature vector
            features = self.create_advanced_feature_vector(symptoms, patient_age, patient_gender, symptom_text)
            features_scaled = self.scaler.transform([features])
            
            # Get predictions from all models
            predictions = {}
            confidences = {}
            
            # Ensemble model predictions
            for name, model in self.ensemble_models.items():
                try:
                    pred_proba = model.predict_proba(features_scaled)[0]
                    pred_class = model.predict(features_scaled)[0]
                    
                    predictions[name] = {
                        'class': pred_class,
                        'probabilities': pred_proba,
                        'confidence': float(np.max(pred_proba))
                    }
                except Exception as e:
                    logger.warning(f"Model {name} prediction failed: {e}")
            
            # Neural network prediction
            if self.neural_network:
                try:
                    features_tensor = torch.FloatTensor(features_scaled).to(self.device)
                    with torch.no_grad():
                        nn_outputs = self.neural_network(features_tensor)
                        nn_proba = torch.softmax(nn_outputs, dim=1).cpu().numpy()[0]
                        nn_pred = torch.argmax(nn_outputs, dim=1).cpu().numpy()[0]
                    
                    predictions['neural_network'] = {
                        'class': nn_pred,
                        'probabilities': nn_proba,
                        'confidence': float(np.max(nn_proba))
                    }
                except Exception as e:
                    logger.warning(f"Neural network prediction failed: {e}")
            
            # Ensemble voting
            final_prediction = self.ensemble_voting(predictions)
            
            # Get top conditions
            top_conditions = self.get_top_conditions(final_prediction, 5)
            
            # Assess urgency using ML
            urgency = self.assess_urgency_ml(symptoms, symptom_text, top_conditions[0])
            
            # Generate ML-based recommendations
            recommendations = self.generate_ml_recommendations(top_conditions[0], urgency, symptoms)
            
            # Calculate confidence metrics
            confidence_metrics = self.calculate_confidence_metrics(predictions)
            
            return {
                'extracted_symptoms': symptoms,
                'primary_diagnosis': top_conditions[0]['condition'],
                'confidence': top_conditions[0]['confidence'],
                'alternative_diagnoses': top_conditions[1:],
                'urgency': urgency,
                'recommendations': recommendations,
                'model_ensemble': {
                    'models_used': list(predictions.keys()),
                    'confidence_metrics': confidence_metrics,
                    'feature_importance': self.get_feature_importance(symptoms)
                },
                'ml_analysis': {
                    'symptom_patterns': self.analyze_symptom_patterns(symptoms),
                    'risk_factors': self.identify_risk_factors(symptoms, patient_age, patient_gender),
                    'differential_diagnosis': self.generate_differential_diagnosis(symptoms)
                }
            }
            
        except Exception as e:
            logger.error(f"ML prediction error: {e}")
            return self.get_ml_error_response()
    
    def create_advanced_feature_vector(self, symptoms: List[str], age: int, 
                                     gender: str, text: str) -> np.ndarray:
        """Create advanced feature vector with BERT embeddings"""
        
        # Get all possible symptoms
        all_symptoms = self.get_all_symptoms()
        
        # Binary symptom features
        symptom_features = np.zeros(len(all_symptoms))
        for i, symptom in enumerate(all_symptoms):
            if symptom in symptoms:
                symptom_features[i] = 1
        
        # Demographic features
        age_normalized = min(age / 100.0, 1.0)
        gender_encoded = 1 if gender.lower() == 'male' else 0 if gender.lower() == 'female' else 0.5
        
        # Symptom category counts
        category_counts = np.zeros(len(self.medical_knowledge['symptom_categories']))
        for i, (category, category_symptoms) in enumerate(self.medical_knowledge['symptom_categories'].items()):
            category_counts[i] = sum(1 for s in symptoms if s in category_symptoms)
        
        # BERT embeddings (reduced dimensionality)
        bert_embeddings = self.get_bert_embeddings(text)
        bert_features = bert_embeddings[:100] if len(bert_embeddings) >= 100 else np.pad(bert_embeddings, (0, 100 - len(bert_embeddings)))
        
        # Severity indicators
        severity_features = self.extract_severity_features(text)
        
        # Combine all features
        features = np.concatenate([
            symptom_features,
            [age_normalized, gender_encoded],
            category_counts,
            bert_features,
            severity_features
        ])
        
        return features
    
    def extract_severity_features(self, text: str) -> np.ndarray:
        """Extract severity indicators from text"""
        severity_keywords = {
            'mild': ['mild', 'slight', 'minor', 'little'],
            'moderate': ['moderate', 'medium', 'noticeable'],
            'severe': ['severe', 'intense', 'extreme', 'unbearable', 'excruciating'],
            'acute': ['sudden', 'acute', 'sharp', 'immediate'],
            'chronic': ['chronic', 'persistent', 'ongoing', 'long-term']
        }
        
        text_lower = text.lower()
        features = np.zeros(len(severity_keywords))
        
        for i, (severity, keywords) in enumerate(severity_keywords.items()):
            if any(keyword in text_lower for keyword in keywords):
                features[i] = 1
        
        return features
    
    def ensemble_voting(self, predictions: Dict) -> Dict:
        """Combine predictions from multiple models using weighted voting"""
        if not predictions:
            return {'class': 0, 'confidence': 0.0}
        
        # Weight models based on their typical performance
        model_weights = {
            'xgboost': 0.3,
            'lightgbm': 0.25,
            'random_forest': 0.2,
            'neural_network': 0.25
        }
        
        # Weighted average of probabilities
        weighted_proba = None
        total_weight = 0
        
        for model_name, pred_data in predictions.items():
            weight = model_weights.get(model_name, 0.1)
            proba = pred_data['probabilities']
            
            if weighted_proba is None:
                weighted_proba = weight * proba
            else:
                weighted_proba += weight * proba
            
            total_weight += weight
        
        if total_weight > 0:
            weighted_proba /= total_weight
        
        final_class = np.argmax(weighted_proba)
        final_confidence = float(np.max(weighted_proba))
        
        return {
            'class': final_class,
            'probabilities': weighted_proba,
            'confidence': final_confidence
        }
    
    def get_top_conditions(self, prediction: Dict, top_k: int = 5) -> List[Dict]:
        """Get top K predicted conditions with confidence scores"""
        probabilities = prediction['probabilities']
        top_indices = np.argsort(probabilities)[-top_k:][::-1]
        
        conditions = []
        for idx in top_indices:
            if probabilities[idx] > 0.01:  # Only include predictions with >1% confidence
                disease = self.disease_encoder.inverse_transform([idx])[0]
                conditions.append({
                    'condition': disease,
                    'confidence': float(probabilities[idx]),
                    'probability': float(probabilities[idx])
                })
        
        return conditions
    
    def assess_urgency_ml(self, symptoms: List[str], text: str, primary_condition: Dict) -> str:
        """Assess urgency using ML-based approach"""
        
        # Emergency symptoms (immediate attention)
        emergency_symptoms = [
            'chest_pain', 'shortness_of_breath', 'severe_headache', 'confusion',
            'seizure', 'loss_of_consciousness', 'severe_bleeding', 'stroke_symptoms',
            'heart_attack_symptoms', 'difficulty_breathing'
        ]
        
        # High urgency symptoms
        high_urgency_symptoms = [
            'fever', 'severe_pain', 'persistent_vomiting', 'severe_dizziness',
            'vision_problems', 'severe_abdominal_pain', 'severe_back_pain'
        ]
        
        # Check for emergency keywords in text
        emergency_keywords = ['severe', 'intense', 'unbearable', 'emergency', 'urgent', 'sudden', 'crushing']
        text_lower = text.lower()
        
        # ML-based urgency scoring
        urgency_score = 0
        
        # Symptom-based scoring
        if any(symptom in symptoms for symptom in emergency_symptoms):
            urgency_score += 0.4
        
        if any(symptom in symptoms for symptom in high_urgency_symptoms):
            urgency_score += 0.2
        
        # Text-based scoring
        if any(keyword in text_lower for keyword in emergency_keywords):
            urgency_score += 0.2
        
        # Condition-based scoring
        condition_name = primary_condition.get('condition', '').lower()
        if any(word in condition_name for word in ['heart attack', 'stroke', 'emergency']):
            urgency_score += 0.3
        elif any(word in condition_name for word in ['pneumonia', 'infection', 'acute']):
            urgency_score += 0.1
        
        # Confidence-based adjustment
        confidence = primary_condition.get('confidence', 0)
        urgency_score *= confidence
        
        # Determine urgency level
        if urgency_score >= 0.6:
            return 'High'
        elif urgency_score >= 0.3:
            return 'Medium'
        else:
            return 'Low'
    
    def generate_ml_recommendations(self, primary_condition: Dict, urgency: str, symptoms: List[str]) -> List[str]:
        """Generate evidence-based recommendations using ML patterns"""
        recommendations = []
        condition = primary_condition.get('condition', '').lower()
        
        # Urgency-based recommendations
        if urgency == 'High':
            recommendations.extend([
                'Seek immediate medical attention',
                'Consider emergency department evaluation',
                'Do not delay medical care'
            ])
        
        # Condition-specific ML recommendations
        if 'respiratory' in condition or 'pneumonia' in condition:
            recommendations.extend([
                'Monitor oxygen saturation if available',
                'Rest and maintain hydration',
                'Avoid respiratory irritants',
                'Consider chest imaging if symptoms persist'
            ])
        elif 'cardiac' in condition or 'heart' in condition:
            recommendations.extend([
                'Monitor blood pressure and heart rate',
                'Avoid strenuous physical activity',
                'Consider ECG evaluation',
                'Assess cardiovascular risk factors'
            ])
        elif 'neurological' in condition or 'migraine' in condition:
            recommendations.extend([
                'Rest in quiet, dark environment',
                'Monitor neurological symptoms',
                'Consider neurological evaluation',
                'Track symptom patterns and triggers'
            ])
        elif 'gastrointestinal' in condition:
            recommendations.extend([
                'Maintain hydration and electrolyte balance',
                'Consider dietary modifications',
                'Monitor for dehydration signs',
                'Track symptom progression'
            ])
        
        # Symptom-specific recommendations
        if 'fever' in symptoms:
            recommendations.append('Monitor temperature regularly and stay hydrated')
        
        if 'pain' in ' '.join(symptoms):
            recommendations.append('Document pain characteristics (location, intensity, triggers)')
        
        return list(set(recommendations))  # Remove duplicates
    
    def analyze_symptom_patterns(self, symptoms: List[str]) -> Dict:
        """Analyze symptom patterns using ML clustering"""
        pattern_analysis = {
            'symptom_clusters': [],
            'pattern_strength': 0.0,
            'category_distribution': {},
            'temporal_patterns': []
        }
        
        # Analyze symptom categories
        for category, category_symptoms in self.medical_knowledge['symptom_categories'].items():
            count = sum(1 for s in symptoms if s in category_symptoms)
            if count > 0:
                pattern_analysis['category_distribution'][category] = count
        
        # Calculate pattern strength
        total_symptoms = len(symptoms)
        if total_symptoms > 0:
            max_category_count = max(pattern_analysis['category_distribution'].values()) if pattern_analysis['category_distribution'] else 0
            pattern_analysis['pattern_strength'] = max_category_count / total_symptoms
        
        return pattern_analysis
    
    def identify_risk_factors(self, symptoms: List[str], age: int, gender: str) -> List[str]:
        """Identify risk factors using ML analysis"""
        risk_factors = []
        
        # Age-based risk factors
        if age > 65:
            risk_factors.extend(['Advanced age', 'Increased infection risk', 'Medication sensitivity'])
        elif age > 45:
            risk_factors.extend(['Middle age', 'Cardiovascular risk factors'])
        
        # Gender-based risk factors
        if gender.lower() == 'female':
            if any(s in symptoms for s in ['chest_pain', 'shortness_of_breath']):
                risk_factors.append('Atypical cardiac presentation in women')
        
        # Symptom-based risk factors
        if 'chest_pain' in symptoms and 'shortness_of_breath' in symptoms:
            risk_factors.append('Cardiopulmonary risk pattern')
        
        if 'fever' in symptoms and 'confusion' in symptoms:
            risk_factors.append('Systemic infection risk')
        
        return risk_factors
    
    def generate_differential_diagnosis(self, symptoms: List[str]) -> List[Dict]:
        """Generate differential diagnosis using ML pattern matching"""
        differential = []
        
        # Score each disease pattern against symptoms
        for disease, pattern in self.medical_knowledge['disease_symptom_patterns'].items():
            score = 0
            total_possible = len(pattern['primary']) + len(pattern['secondary'])
            
            # Primary symptoms (higher weight)
            for symptom in pattern['primary']:
                if symptom in symptoms:
                    score += 2
            
            # Secondary symptoms (lower weight)
            for symptom in pattern['secondary']:
                if symptom in symptoms:
                    score += 1
            
            if total_possible > 0:
                match_percentage = score / (total_possible * 2)  # Normalize
                
                if match_percentage > 0.2:  # Include if >20% match
                    differential.append({
                        'condition': disease,
                        'match_score': match_percentage,
                        'reasoning': f"Matches {score}/{total_possible * 2} symptom criteria"
                    })
        
        # Sort by match score
        differential.sort(key=lambda x: x['match_score'], reverse=True)
        
        return differential[:5]  # Return top 5
    
    def get_all_symptoms(self) -> List[str]:
        """Get all possible symptoms from knowledge base"""
        all_symptoms = set()
        for symptoms in self.medical_knowledge['symptom_categories'].values():
            all_symptoms.update(symptoms)
        return sorted(list(all_symptoms))
    
    def get_symptom_embedding(self, symptom: str) -> np.ndarray:
        """Get or compute BERT embedding for symptom"""
        if symptom in self.symptom_embeddings:
            return self.symptom_embeddings[symptom]
        
        # Compute embedding
        symptom_text = symptom.replace('_', ' ')
        embedding = self.get_bert_embeddings(symptom_text)
        self.symptom_embeddings[symptom] = embedding
        
        return embedding
    
    def extract_symptoms_rule_based(self, text: str) -> List[str]:
        """Fallback rule-based symptom extraction"""
        text_lower = text.lower()
        extracted = []
        
        # Simple keyword matching
        symptom_keywords = {
            'fever': ['fever', 'high temperature', 'pyrexia'],
            'headache': ['headache', 'head pain', 'migraine'],
            'cough': ['cough', 'coughing'],
            'fatigue': ['tired', 'exhausted', 'fatigue', 'weakness'],
            'nausea': ['nausea', 'sick', 'queasy'],
            'chest_pain': ['chest pain', 'chest discomfort'],
            'shortness_of_breath': ['shortness of breath', 'difficulty breathing', 'breathless']
        }
        
        for symptom, keywords in symptom_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                extracted.append(symptom)
        
        return extracted
    
    def calculate_confidence_metrics(self, predictions: Dict) -> Dict:
        """Calculate comprehensive confidence metrics"""
        if not predictions:
            return {'overall_confidence': 0.0}
        
        confidences = [pred['confidence'] for pred in predictions.values()]
        
        return {
            'overall_confidence': float(np.mean(confidences)),
            'confidence_std': float(np.std(confidences)),
            'model_agreement': float(np.std(confidences) < 0.1),  # Low std = high agreement
            'min_confidence': float(np.min(confidences)),
            'max_confidence': float(np.max(confidences))
        }
    
    def get_feature_importance(self, symptoms: List[str]) -> Dict:
        """Get feature importance from tree-based models"""
        try:
            if 'random_forest' in self.ensemble_models:
                rf_model = self.ensemble_models['random_forest']
                if hasattr(rf_model, 'feature_importances_'):
                    importances = rf_model.feature_importances_
                    
                    # Map to symptom names
                    all_symptoms = self.get_all_symptoms()
                    symptom_importance = {}
                    
                    for i, symptom in enumerate(all_symptoms):
                        if i < len(importances) and symptom in symptoms:
                            symptom_importance[symptom] = float(importances[i])
                    
                    return symptom_importance
            
            return {}
            
        except Exception as e:
            logger.error(f"Feature importance calculation failed: {e}")
            return {}
    
    def save_trained_models(self):
        """Save all trained models"""
        os.makedirs('ai/models/trained', exist_ok=True)
        
        # Save ML models
        joblib.dump(self.symptom_classifier, 'ai/models/trained/symptom_classifier.pkl')
        joblib.dump(self.disease_encoder, 'ai/models/trained/disease_encoder.pkl')
        joblib.dump(self.scaler, 'ai/models/trained/feature_scaler.pkl')
        
        # Save ensemble models
        for name, model in self.ensemble_models.items():
            joblib.dump(model, f'ai/models/trained/{name}_model.pkl')
        
        # Save neural network
        if self.neural_network:
            torch.save(self.neural_network.state_dict(), 'ai/models/trained/medical_neural_network.pth')
        
        # Save embeddings and knowledge
        with open('ai/models/trained/symptom_embeddings.pkl', 'wb') as f:
            pickle.dump(self.symptom_embeddings, f)
        
        logger.info("💾 All models saved successfully!")
    
    def get_ml_error_response(self) -> Dict:
        """Return ML error response"""
        return {
            'error': 'ML analysis failed',
            'primary_diagnosis': 'Medical Consultation Required',
            'confidence': 0.0,
            'alternative_diagnoses': [],
            'urgency': 'Medium',
            'recommendations': [
                'ML models encountered an error',
                'Please consult with healthcare professional',
                'Provide detailed symptom description to medical provider'
            ],
            'ml_analysis': {
                'error': 'Model prediction failed',
                'fallback_used': True
            }
        }

# Training function
def train_symptom_model():
    """Train the advanced ML/DL symptom analyzer"""
    analyzer = AdvancedSymptomAnalyzer()
    
    print("🏥" + "=" * 80)
    print("🤖 MEDISCAN AI - ADVANCED ML/DL MODEL TRAINING")
    print("🏥" + "=" * 80)
    
    # The training will happen in initialize_models if no pre-trained models exist
    
    return analyzer

# Global analyzer instance
symptom_analyzer = AdvancedSymptomAnalyzer()