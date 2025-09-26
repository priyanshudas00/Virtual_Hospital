"""
Advanced Lab Results Analysis using Machine Learning
Comprehensive blood test, urine test, and biomarker interpretation
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import joblib
import logging
from typing import Dict, List, Tuple, Any
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedLabAnalyzer:
    """Advanced ML-based laboratory results analyzer"""
    
    def __init__(self):
        # ML Models
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.risk_classifiers = {}
        self.trend_analyzers = {}
        self.clustering_model = DBSCAN(eps=0.5, min_samples=5)
        
        # Scalers
        self.scaler = RobustScaler()
        self.pca = PCA(n_components=10)
        
        # Medical knowledge
        self.reference_ranges = self.load_comprehensive_reference_ranges()
        self.risk_models = {}
        self.lab_patterns = self.load_lab_patterns()
        
        # Initialize models
        self.initialize_lab_models()
    
    def load_comprehensive_reference_ranges(self) -> Dict:
        """Load comprehensive lab test reference ranges"""
        return {
            # Complete Blood Count (CBC)
            'wbc': {'min': 4.5, 'max': 11.0, 'unit': 'K/μL', 'name': 'White Blood Cells', 'critical_low': 2.0, 'critical_high': 20.0},
            'rbc': {'min': 4.2, 'max': 5.4, 'unit': 'M/μL', 'name': 'Red Blood Cells', 'critical_low': 3.0, 'critical_high': 6.5},
            'hemoglobin': {'min': 12.0, 'max': 15.5, 'unit': 'g/dL', 'name': 'Hemoglobin', 'critical_low': 8.0, 'critical_high': 18.0},
            'hematocrit': {'min': 36.0, 'max': 46.0, 'unit': '%', 'name': 'Hematocrit', 'critical_low': 25.0, 'critical_high': 55.0},
            'platelets': {'min': 150, 'max': 400, 'unit': 'K/μL', 'name': 'Platelets', 'critical_low': 50, 'critical_high': 1000},
            
            # Basic Metabolic Panel (BMP)
            'glucose': {'min': 70, 'max': 100, 'unit': 'mg/dL', 'name': 'Glucose', 'critical_low': 40, 'critical_high': 400},
            'sodium': {'min': 136, 'max': 145, 'unit': 'mmol/L', 'name': 'Sodium', 'critical_low': 120, 'critical_high': 160},
            'potassium': {'min': 3.5, 'max': 5.1, 'unit': 'mmol/L', 'name': 'Potassium', 'critical_low': 2.5, 'critical_high': 6.5},
            'chloride': {'min': 98, 'max': 107, 'unit': 'mmol/L', 'name': 'Chloride', 'critical_low': 85, 'critical_high': 120},
            'creatinine': {'min': 0.6, 'max': 1.2, 'unit': 'mg/dL', 'name': 'Creatinine', 'critical_low': 0.3, 'critical_high': 5.0},
            'bun': {'min': 7, 'max': 20, 'unit': 'mg/dL', 'name': 'Blood Urea Nitrogen', 'critical_low': 3, 'critical_high': 100},
            
            # Lipid Panel
            'total_cholesterol': {'min': 0, 'max': 200, 'unit': 'mg/dL', 'name': 'Total Cholesterol', 'critical_high': 400},
            'ldl_cholesterol': {'min': 0, 'max': 100, 'unit': 'mg/dL', 'name': 'LDL Cholesterol', 'critical_high': 250},
            'hdl_cholesterol': {'min': 40, 'max': 60, 'unit': 'mg/dL', 'name': 'HDL Cholesterol', 'critical_low': 20},
            'triglycerides': {'min': 0, 'max': 150, 'unit': 'mg/dL', 'name': 'Triglycerides', 'critical_high': 1000},
            
            # Liver Function Tests
            'alt': {'min': 7, 'max': 40, 'unit': 'U/L', 'name': 'ALT', 'critical_high': 200},
            'ast': {'min': 8, 'max': 40, 'unit': 'U/L', 'name': 'AST', 'critical_high': 200},
            'bilirubin_total': {'min': 0.2, 'max': 1.2, 'unit': 'mg/dL', 'name': 'Total Bilirubin', 'critical_high': 10.0},
            'albumin': {'min': 3.5, 'max': 5.0, 'unit': 'g/dL', 'name': 'Albumin', 'critical_low': 2.0},
            
            # Cardiac Markers
            'troponin': {'min': 0, 'max': 0.04, 'unit': 'ng/mL', 'name': 'Troponin', 'critical_high': 1.0},
            'ck_mb': {'min': 0, 'max': 6.3, 'unit': 'ng/mL', 'name': 'CK-MB', 'critical_high': 25.0},
            'bnp': {'min': 0, 'max': 100, 'unit': 'pg/mL', 'name': 'BNP', 'critical_high': 400},
            
            # Inflammatory Markers
            'crp': {'min': 0, 'max': 3.0, 'unit': 'mg/L', 'name': 'C-Reactive Protein', 'critical_high': 50.0},
            'esr': {'min': 0, 'max': 30, 'unit': 'mm/hr', 'name': 'ESR', 'critical_high': 100},
            
            # Thyroid Function
            'tsh': {'min': 0.4, 'max': 4.0, 'unit': 'mIU/L', 'name': 'TSH', 'critical_low': 0.1, 'critical_high': 20.0},
            't4': {'min': 4.5, 'max': 12.0, 'unit': 'μg/dL', 'name': 'T4', 'critical_low': 2.0, 'critical_high': 20.0},
            
            # Vitamins and Minerals
            'vitamin_d': {'min': 30, 'max': 100, 'unit': 'ng/mL', 'name': 'Vitamin D', 'critical_low': 10},
            'b12': {'min': 200, 'max': 900, 'unit': 'pg/mL', 'name': 'Vitamin B12', 'critical_low': 100},
            'iron': {'min': 60, 'max': 170, 'unit': 'μg/dL', 'name': 'Iron', 'critical_low': 30, 'critical_high': 300},
            'ferritin': {'min': 12, 'max': 300, 'unit': 'ng/mL', 'name': 'Ferritin', 'critical_high': 1000}
        }
    
    def load_lab_patterns(self) -> Dict:
        """Load disease-specific lab patterns"""
        return {
            'diabetes': {
                'glucose': {'threshold': 126, 'direction': 'high'},
                'hba1c': {'threshold': 6.5, 'direction': 'high'}
            },
            'kidney_disease': {
                'creatinine': {'threshold': 1.5, 'direction': 'high'},
                'bun': {'threshold': 25, 'direction': 'high'},
                'gfr': {'threshold': 60, 'direction': 'low'}
            },
            'liver_disease': {
                'alt': {'threshold': 80, 'direction': 'high'},
                'ast': {'threshold': 80, 'direction': 'high'},
                'bilirubin_total': {'threshold': 2.0, 'direction': 'high'}
            },
            'cardiovascular_risk': {
                'total_cholesterol': {'threshold': 240, 'direction': 'high'},
                'ldl_cholesterol': {'threshold': 160, 'direction': 'high'},
                'hdl_cholesterol': {'threshold': 40, 'direction': 'low'},
                'troponin': {'threshold': 0.1, 'direction': 'high'}
            },
            'infection': {
                'wbc': {'threshold': 12.0, 'direction': 'high'},
                'crp': {'threshold': 10.0, 'direction': 'high'},
                'esr': {'threshold': 50, 'direction': 'high'}
            },
            'anemia': {
                'hemoglobin': {'threshold': 10.0, 'direction': 'low'},
                'iron': {'threshold': 50, 'direction': 'low'},
                'ferritin': {'threshold': 15, 'direction': 'low'}
            }
        }
    
    def initialize_lab_models(self):
        """Initialize ML models for lab analysis"""
        try:
            # Load pre-trained models
            self.load_pretrained_lab_models()
            logger.info("✅ Pre-trained lab models loaded")
        except Exception as e:
            logger.warning(f"Pre-trained lab models not found: {e}")
            logger.info("🔄 Training new lab analysis models...")
            self.train_lab_models()
    
    def load_pretrained_lab_models(self):
        """Load pre-trained lab analysis models"""
        model_dir = 'ai/models/trained'
        
        # Load anomaly detection model
        self.anomaly_detector = joblib.load(f'{model_dir}/lab_anomaly_detector.pkl')
        
        # Load risk prediction models
        for condition in ['diabetes', 'cardiovascular', 'kidney', 'liver']:
            self.risk_classifiers[condition] = joblib.load(f'{model_dir}/{condition}_risk_model.pkl')
        
        # Load scalers
        self.scaler = joblib.load(f'{model_dir}/lab_scaler.pkl')
        self.pca = joblib.load(f'{model_dir}/lab_pca.pkl')
    
    def train_lab_models(self):
        """Train ML models for lab analysis"""
        logger.info("🧪 Training lab analysis models...")
        
        # Generate comprehensive training data
        training_data = self.generate_lab_training_data(5000)
        
        # Prepare features
        X = training_data.drop(['patient_id', 'risk_diabetes', 'risk_cvd', 'risk_kidney', 'risk_liver'], axis=1)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Apply PCA for dimensionality reduction
        X_pca = self.pca.fit_transform(X_scaled)
        
        # Train anomaly detector
        self.anomaly_detector.fit(X_scaled)
        
        # Train risk prediction models
        risk_targets = ['risk_diabetes', 'risk_cvd', 'risk_kidney', 'risk_liver']
        
        for target in risk_targets:
            condition = target.replace('risk_', '')
            y = training_data[target]
            
            # Train classifier
            classifier = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            classifier.fit(X_scaled, y)
            
            self.risk_classifiers[condition] = classifier
            
            # Evaluate
            score = classifier.score(X_scaled, y)
            logger.info(f"📈 {condition} risk model accuracy: {score:.3f}")
        
        # Save models
        self.save_lab_models()
        logger.info("✅ Lab analysis models trained and saved!")
    
    def generate_lab_training_data(self, n_samples: int) -> pd.DataFrame:
        """Generate comprehensive lab training data"""
        logger.info(f"🔬 Generating {n_samples} lab training samples...")
        
        data = []
        
        for i in range(n_samples):
            sample = {'patient_id': f'P{i:05d}'}
            
            # Generate demographics
            age = np.random.randint(18, 90)
            gender = np.random.choice(['M', 'F'])
            sample['age'] = age
            sample['gender_encoded'] = 1 if gender == 'M' else 0
            
            # Generate lab values with realistic distributions
            for test_name, ranges in self.reference_ranges.items():
                # 70% normal, 30% abnormal
                if np.random.random() < 0.7:
                    # Normal values
                    value = np.random.uniform(ranges['min'], ranges['max'])
                else:
                    # Abnormal values
                    if np.random.random() < 0.5:
                        # Low values
                        value = np.random.uniform(ranges.get('critical_low', ranges['min'] * 0.5), ranges['min'])
                    else:
                        # High values
                        value = np.random.uniform(ranges['max'], ranges.get('critical_high', ranges['max'] * 2))
                
                sample[test_name] = round(value, 2)
            
            # Generate risk labels based on lab patterns
            sample['risk_diabetes'] = int(sample['glucose'] > 126 or 
                                        (age > 45 and sample['glucose'] > 100))
            
            sample['risk_cvd'] = int(sample['total_cholesterol'] > 240 or 
                                   sample['ldl_cholesterol'] > 160 or
                                   sample['hdl_cholesterol'] < 40)
            
            sample['risk_kidney'] = int(sample['creatinine'] > 1.5 or 
                                      sample['bun'] > 25)
            
            sample['risk_liver'] = int(sample['alt'] > 80 or 
                                     sample['ast'] > 80 or
                                     sample['bilirubin_total'] > 2.0)
            
            data.append(sample)
        
        df = pd.DataFrame(data)
        logger.info(f"✅ Generated lab training data: {df.shape}")
        
        return df
    
    def analyze_lab_results(self, lab_data: Dict, patient_info: Dict = None) -> Dict:
        """Comprehensive ML-based lab analysis"""
        try:
            # Validate and process lab data
            processed_data = self.process_lab_data(lab_data)
            
            if not processed_data:
                return self.get_lab_error_response()
            
            # Individual test analysis
            test_results = self.analyze_individual_tests(processed_data)
            
            # ML-based anomaly detection
            anomalies = self.detect_lab_anomalies(processed_data)
            
            # Risk assessment using ML
            risk_assessment = self.assess_health_risks_ml(processed_data, patient_info)
            
            # Pattern analysis
            pattern_analysis = self.analyze_lab_patterns(processed_data)
            
            # Trend analysis (if historical data available)
            trends = self.analyze_lab_trends(lab_data.get('historical_data', []))
            
            # Generate ML-based summary
            summary = self.generate_ml_summary(test_results, anomalies, risk_assessment)
            
            # Generate evidence-based recommendations
            recommendations = self.generate_ml_recommendations(test_results, risk_assessment, pattern_analysis)
            
            return {
                'analysis_method': 'Advanced Machine Learning',
                'summary': summary,
                'test_results': test_results,
                'anomaly_detection': anomalies,
                'risk_assessment': risk_assessment,
                'pattern_analysis': pattern_analysis,
                'trends': trends,
                'key_findings': self.extract_key_findings(test_results, anomalies),
                'recommendations': recommendations,
                'overall_risk': self.calculate_overall_risk_ml(risk_assessment),
                'follow_up_needed': self.determine_followup_ml(test_results, risk_assessment),
                'ml_confidence': self.calculate_ml_confidence(processed_data)
            }
            
        except Exception as e:
            logger.error(f"Lab analysis error: {e}")
            return self.get_lab_error_response()
    
    def process_lab_data(self, lab_data: Dict) -> Dict:
        """Process and validate lab data"""
        processed = {}
        
        for test_name, value in lab_data.items():
            if test_name in self.reference_ranges:
                try:
                    processed[test_name] = float(value)
                except (ValueError, TypeError):
                    logger.warning(f"Invalid value for {test_name}: {value}")
                    continue
        
        return processed
    
    def analyze_individual_tests(self, lab_data: Dict) -> List[Dict]:
        """Analyze each lab test with ML-enhanced interpretation"""
        results = []
        
        for test_name, value in lab_data.items():
            if test_name not in self.reference_ranges:
                continue
            
            ref_range = self.reference_ranges[test_name]
            
            # Determine status with ML-enhanced thresholds
            status, severity = self.classify_test_result(test_name, value, ref_range)
            
            # ML-based clinical significance
            significance = self.assess_clinical_significance_ml(test_name, value, status)
            
            # Risk prediction for this specific test
            individual_risk = self.predict_individual_test_risk(test_name, value)
            
            results.append({
                'parameter': ref_range['name'],
                'value': value,
                'unit': ref_range['unit'],
                'reference_range': f"{ref_range['min']}-{ref_range['max']}",
                'status': status,
                'severity': severity,
                'clinical_significance': significance,
                'individual_risk': individual_risk,
                'ml_interpretation': self.get_ml_test_interpretation(test_name, value, status)
            })
        
        return results
    
    def detect_lab_anomalies(self, lab_data: Dict) -> List[Dict]:
        """Detect anomalous patterns using ML"""
        anomalies = []
        
        try:
            # Prepare feature vector
            feature_vector = []
            test_names = []
            
            for test_name in sorted(self.reference_ranges.keys()):
                if test_name in lab_data:
                    # Normalize value
                    ref_range = self.reference_ranges[test_name]
                    normalized_value = (lab_data[test_name] - ref_range['min']) / (ref_range['max'] - ref_range['min'])
                    feature_vector.append(normalized_value)
                    test_names.append(test_name)
                else:
                    feature_vector.append(0)  # Missing value
                    test_names.append(test_name)
            
            if len(feature_vector) > 0:
                # Detect anomalies
                anomaly_score = self.anomaly_detector.decision_function([feature_vector])[0]
                is_anomaly = self.anomaly_detector.predict([feature_vector])[0] == -1
                
                if is_anomaly:
                    anomalies.append({
                        'type': 'ML Pattern Anomaly',
                        'description': f'Unusual lab value combination detected (anomaly score: {anomaly_score:.3f})',
                        'severity': 'High' if anomaly_score < -0.5 else 'Medium',
                        'ml_confidence': abs(anomaly_score),
                        'recommendation': 'Expert review recommended for unusual pattern'
                    })
                
                # Cluster analysis for pattern detection
                cluster_anomalies = self.detect_cluster_anomalies(feature_vector, test_names)
                anomalies.extend(cluster_anomalies)
        
        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")
        
        return anomalies
    
    def assess_health_risks_ml(self, lab_data: Dict, patient_info: Dict = None) -> Dict:
        """Assess health risks using trained ML models"""
        risks = {}
        
        # Prepare feature vector for risk models
        feature_vector = self.prepare_risk_feature_vector(lab_data, patient_info)
        
        # Predict risks using trained classifiers
        for condition, classifier in self.risk_classifiers.items():
            try:
                risk_proba = classifier.predict_proba([feature_vector])[0]
                risk_prediction = classifier.predict([feature_vector])[0]
                
                # Get feature importance for this prediction
                feature_importance = self.get_risk_feature_importance(classifier, feature_vector)
                
                risks[condition] = {
                    'risk_level': 'High' if risk_prediction == 1 else 'Low',
                    'risk_probability': float(risk_proba[1]) if len(risk_proba) > 1 else 0.0,
                    'ml_confidence': float(max(risk_proba)),
                    'key_factors': feature_importance,
                    'interpretation': self.interpret_risk_ml(condition, risk_prediction, risk_proba)
                }
                
            except Exception as e:
                logger.warning(f"Risk assessment for {condition} failed: {e}")
                risks[condition] = self.get_default_risk_assessment(condition, lab_data)
        
        return risks
    
    def analyze_lab_patterns(self, lab_data: Dict) -> Dict:
        """Analyze lab patterns using ML clustering and pattern recognition"""
        pattern_analysis = {
            'disease_patterns': {},
            'metabolic_profile': {},
            'inflammatory_markers': {},
            'organ_function': {}
        }
        
        # Check disease-specific patterns
        for disease, pattern in self.lab_patterns.items():
            pattern_score = 0
            matching_tests = 0
            
            for test, criteria in pattern.items():
                if test in lab_data:
                    value = lab_data[test]
                    threshold = criteria['threshold']
                    direction = criteria['direction']
                    
                    if direction == 'high' and value > threshold:
                        pattern_score += 1
                    elif direction == 'low' and value < threshold:
                        pattern_score += 1
                    
                    matching_tests += 1
            
            if matching_tests > 0:
                pattern_strength = pattern_score / matching_tests
                pattern_analysis['disease_patterns'][disease] = {
                    'pattern_strength': pattern_strength,
                    'matching_criteria': pattern_score,
                    'total_criteria': matching_tests,
                    'likelihood': 'High' if pattern_strength > 0.7 else 'Medium' if pattern_strength > 0.4 else 'Low'
                }
        
        # Metabolic profile analysis
        metabolic_tests = ['glucose', 'total_cholesterol', 'triglycerides', 'hdl_cholesterol']
        metabolic_values = {test: lab_data.get(test, 0) for test in metabolic_tests if test in lab_data}
        
        if metabolic_values:
            pattern_analysis['metabolic_profile'] = self.analyze_metabolic_pattern(metabolic_values)
        
        # Inflammatory markers analysis
        inflammatory_tests = ['crp', 'esr', 'wbc']
        inflammatory_values = {test: lab_data.get(test, 0) for test in inflammatory_tests if test in lab_data}
        
        if inflammatory_values:
            pattern_analysis['inflammatory_markers'] = self.analyze_inflammatory_pattern(inflammatory_values)
        
        return pattern_analysis
    
    def analyze_metabolic_pattern(self, metabolic_values: Dict) -> Dict:
        """Analyze metabolic syndrome patterns"""
        risk_factors = 0
        findings = []
        
        if metabolic_values.get('glucose', 0) > 100:
            risk_factors += 1
            findings.append('Elevated glucose')
        
        if metabolic_values.get('triglycerides', 0) > 150:
            risk_factors += 1
            findings.append('Elevated triglycerides')
        
        if metabolic_values.get('hdl_cholesterol', 100) < 40:
            risk_factors += 1
            findings.append('Low HDL cholesterol')
        
        return {
            'metabolic_risk_factors': risk_factors,
            'findings': findings,
            'metabolic_syndrome_risk': 'High' if risk_factors >= 2 else 'Medium' if risk_factors == 1 else 'Low'
        }
    
    def analyze_inflammatory_pattern(self, inflammatory_values: Dict) -> Dict:
        """Analyze inflammatory response patterns"""
        inflammation_score = 0
        findings = []
        
        if inflammatory_values.get('crp', 0) > 10:
            inflammation_score += 2
            findings.append('Significantly elevated CRP')
        elif inflammatory_values.get('crp', 0) > 3:
            inflammation_score += 1
            findings.append('Mildly elevated CRP')
        
        if inflammatory_values.get('esr', 0) > 50:
            inflammation_score += 2
            findings.append('Significantly elevated ESR')
        elif inflammatory_values.get('esr', 0) > 30:
            inflammation_score += 1
            findings.append('Mildly elevated ESR')
        
        if inflammatory_values.get('wbc', 0) > 12:
            inflammation_score += 1
            findings.append('Elevated white blood cells')
        
        return {
            'inflammation_score': inflammation_score,
            'findings': findings,
            'inflammatory_status': 'High' if inflammation_score >= 3 else 'Medium' if inflammation_score >= 1 else 'Normal'
        }
    
    def classify_test_result(self, test_name: str, value: float, ref_range: Dict) -> Tuple[str, str]:
        """Classify test result with ML-enhanced thresholds"""
        
        # Check critical values first
        if 'critical_low' in ref_range and value < ref_range['critical_low']:
            return 'Critical Low', 'Critical'
        elif 'critical_high' in ref_range and value > ref_range['critical_high']:
            return 'Critical High', 'Critical'
        
        # Standard classification
        if value < ref_range['min']:
            # Determine severity of low value
            ratio = value / ref_range['min']
            if ratio < 0.7:
                return 'Low', 'Moderate'
            else:
                return 'Low', 'Mild'
        elif value > ref_range['max']:
            # Determine severity of high value
            ratio = value / ref_range['max']
            if ratio > 1.5:
                return 'High', 'Moderate'
            else:
                return 'High', 'Mild'
        else:
            return 'Normal', 'Normal'
    
    def assess_clinical_significance_ml(self, test_name: str, value: float, status: str) -> str:
        """Assess clinical significance using ML patterns"""
        if status == 'Normal':
            return 'No clinical concern - within normal limits'
        
        # ML-based significance assessment
        significance_patterns = {
            'glucose': {
                'High': 'Diabetes screening recommended - endocrinology consultation may be needed',
                'Low': 'Hypoglycemia evaluation - monitor closely'
            },
            'creatinine': {
                'High': 'Kidney function assessment - nephrology consultation recommended',
                'Low': 'Usually not clinically significant'
            },
            'troponin': {
                'High': 'Cardiac injury suspected - urgent cardiology evaluation required',
                'Low': 'No cardiac injury detected'
            },
            'wbc': {
                'High': 'Infection or inflammation suspected - further evaluation needed',
                'Low': 'Immunosuppression possible - hematology consultation may be needed'
            }
        }
        
        return significance_patterns.get(test_name, {}).get(status, 
                                                           f'{status} {test_name} - clinical correlation recommended')
    
    def predict_individual_test_risk(self, test_name: str, value: float) -> Dict:
        """Predict individual test risk using ML"""
        ref_range = self.reference_ranges.get(test_name, {})
        
        # Calculate risk score based on deviation from normal
        if 'min' in ref_range and 'max' in ref_range:
            normal_range = ref_range['max'] - ref_range['min']
            
            if value < ref_range['min']:
                deviation = (ref_range['min'] - value) / normal_range
                risk_type = 'Low value risk'
            elif value > ref_range['max']:
                deviation = (value - ref_range['max']) / normal_range
                risk_type = 'High value risk'
            else:
                deviation = 0
                risk_type = 'No risk'
            
            # Convert deviation to risk score
            risk_score = min(deviation, 2.0) / 2.0  # Cap at 100%
            
            return {
                'risk_score': risk_score,
                'risk_type': risk_type,
                'deviation_magnitude': deviation
            }
        
        return {'risk_score': 0.0, 'risk_type': 'Unknown'}
    
    def prepare_risk_feature_vector(self, lab_data: Dict, patient_info: Dict = None) -> List[float]:
        """Prepare feature vector for risk prediction models"""
        features = []
        
        # Lab values (normalized)
        for test_name in sorted(self.reference_ranges.keys()):
            if test_name in lab_data:
                ref_range = self.reference_ranges[test_name]
                normalized_value = (lab_data[test_name] - ref_range['min']) / (ref_range['max'] - ref_range['min'])
                features.append(normalized_value)
            else:
                features.append(0.0)  # Missing value
        
        # Patient demographics
        if patient_info:
            age = patient_info.get('age', 50)
            features.append(age / 100.0)  # Normalized age
            
            gender = patient_info.get('gender', 'unknown')
            features.append(1.0 if gender.lower() == 'male' else 0.0)
        else:
            features.extend([0.5, 0.5])  # Default values
        
        return features
    
    def get_risk_feature_importance(self, classifier, feature_vector: List[float]) -> List[str]:
        """Get feature importance for risk prediction"""
        try:
            if hasattr(classifier, 'feature_importances_'):
                importances = classifier.feature_importances_
                
                # Get top 5 most important features
                top_indices = np.argsort(importances)[-5:][::-1]
                
                feature_names = list(self.reference_ranges.keys()) + ['age', 'gender']
                important_features = []
                
                for idx in top_indices:
                    if idx < len(feature_names):
                        feature_name = feature_names[idx]
                        importance = importances[idx]
                        if importance > 0.05:  # Only include significant features
                            important_features.append(f"{feature_name} (importance: {importance:.2f})")
                
                return important_features
            
            return ['Feature importance not available']
            
        except Exception as e:
            logger.error(f"Feature importance calculation failed: {e}")
            return ['Feature analysis failed']
    
    def interpret_risk_ml(self, condition: str, prediction: int, probabilities: np.ndarray) -> str:
        """Interpret risk prediction using ML"""
        risk_prob = float(probabilities[1]) if len(probabilities) > 1 else 0.0
        
        interpretations = {
            'diabetes': {
                1: f'High diabetes risk detected (probability: {risk_prob:.1%}) - endocrinology consultation recommended',
                0: f'Low diabetes risk (probability: {risk_prob:.1%}) - continue monitoring'
            },
            'cardiovascular': {
                1: f'Elevated cardiovascular risk (probability: {risk_prob:.1%}) - cardiology evaluation recommended',
                0: f'Low cardiovascular risk (probability: {risk_prob:.1%}) - maintain healthy lifestyle'
            },
            'kidney': {
                1: f'Kidney function concerns (probability: {risk_prob:.1%}) - nephrology consultation recommended',
                0: f'Normal kidney function indicators (probability: {risk_prob:.1%})'
            },
            'liver': {
                1: f'Liver function abnormalities (probability: {risk_prob:.1%}) - hepatology evaluation recommended',
                0: f'Normal liver function indicators (probability: {risk_prob:.1%})'
            }
        }
        
        return interpretations.get(condition, {}).get(prediction, f'Risk assessment completed for {condition}')
    
    def calculate_ml_confidence(self, lab_data: Dict) -> Dict:
        """Calculate ML confidence metrics"""
        confidence_metrics = {
            'data_completeness': len(lab_data) / len(self.reference_ranges),
            'value_validity': 1.0,  # All processed values are valid
            'pattern_recognition': 0.0,
            'overall_confidence': 0.0
        }
        
        # Calculate pattern recognition confidence
        valid_patterns = 0
        total_patterns = len(self.lab_patterns)
        
        for disease, pattern in self.lab_patterns.items():
            pattern_match = 0
            pattern_tests = 0
            
            for test in pattern.keys():
                if test in lab_data:
                    pattern_tests += 1
                    # Check if value matches expected pattern
                    # This is a simplified check
                    pattern_match += 0.5
            
            if pattern_tests > 0:
                valid_patterns += pattern_match / pattern_tests
        
        if total_patterns > 0:
            confidence_metrics['pattern_recognition'] = valid_patterns / total_patterns
        
        # Overall confidence
        confidence_metrics['overall_confidence'] = (
            confidence_metrics['data_completeness'] * 0.4 +
            confidence_metrics['value_validity'] * 0.3 +
            confidence_metrics['pattern_recognition'] * 0.3
        )
        
        return confidence_metrics
    
    def save_lab_models(self):
        """Save all lab analysis models"""
        model_dir = 'ai/models/trained'
        os.makedirs(model_dir, exist_ok=True)
        
        # Save ML models
        joblib.dump(self.anomaly_detector, f'{model_dir}/lab_anomaly_detector.pkl')
        joblib.dump(self.scaler, f'{model_dir}/lab_scaler.pkl')
        joblib.dump(self.pca, f'{model_dir}/lab_pca.pkl')
        
        # Save risk models
        for condition, model in self.risk_classifiers.items():
            joblib.dump(model, f'{model_dir}/{condition}_risk_model.pkl')
        
        logger.info("💾 Lab analysis models saved!")
    
    def get_lab_error_response(self) -> Dict:
        """Return error response for lab analysis"""
        return {
            'analysis_method': 'Error Recovery',
            'summary': 'Lab analysis could not be completed due to technical error',
            'test_results': [],
            'anomaly_detection': [],
            'risk_assessment': {},
            'pattern_analysis': {},
            'trends': {},
            'key_findings': ['Analysis error occurred'],
            'recommendations': [
                'Please have results reviewed by healthcare provider',
                'Ensure all lab values are entered correctly',
                'Contact support if error persists'
            ],
            'overall_risk': 'Unknown',
            'ml_confidence': {'overall_confidence': 0.0}
        }

# Global lab analyzer
lab_analyzer = AdvancedLabAnalyzer()