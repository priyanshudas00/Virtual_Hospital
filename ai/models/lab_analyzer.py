"""
Lab Results Analysis using Machine Learning
Analyzes blood tests, urine tests, and other lab results
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import logging
from typing import Dict, List, Tuple, Any
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LabAnalyzer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.trend_analyzer = RandomForestRegressor(n_estimators=100, random_state=42)
        self.reference_ranges = self.load_reference_ranges()
        self.risk_models = {}
        self.load_models()
    
    def load_reference_ranges(self) -> Dict:
        """Load normal reference ranges for lab tests"""
        return {
            # Complete Blood Count (CBC)
            'wbc': {'min': 4.5, 'max': 11.0, 'unit': 'K/μL', 'name': 'White Blood Cells'},
            'rbc': {'min': 4.2, 'max': 5.4, 'unit': 'M/μL', 'name': 'Red Blood Cells'},
            'hemoglobin': {'min': 12.0, 'max': 15.5, 'unit': 'g/dL', 'name': 'Hemoglobin'},
            'hematocrit': {'min': 36.0, 'max': 46.0, 'unit': '%', 'name': 'Hematocrit'},
            'platelets': {'min': 150, 'max': 400, 'unit': 'K/μL', 'name': 'Platelets'},
            
            # Basic Metabolic Panel (BMP)
            'glucose': {'min': 70, 'max': 100, 'unit': 'mg/dL', 'name': 'Glucose'},
            'sodium': {'min': 136, 'max': 145, 'unit': 'mmol/L', 'name': 'Sodium'},
            'potassium': {'min': 3.5, 'max': 5.1, 'unit': 'mmol/L', 'name': 'Potassium'},
            'chloride': {'min': 98, 'max': 107, 'unit': 'mmol/L', 'name': 'Chloride'},
            'creatinine': {'min': 0.6, 'max': 1.2, 'unit': 'mg/dL', 'name': 'Creatinine'},
            'bun': {'min': 7, 'max': 20, 'unit': 'mg/dL', 'name': 'Blood Urea Nitrogen'},
            
            # Lipid Panel
            'total_cholesterol': {'min': 0, 'max': 200, 'unit': 'mg/dL', 'name': 'Total Cholesterol'},
            'ldl_cholesterol': {'min': 0, 'max': 100, 'unit': 'mg/dL', 'name': 'LDL Cholesterol'},
            'hdl_cholesterol': {'min': 40, 'max': 60, 'unit': 'mg/dL', 'name': 'HDL Cholesterol'},
            'triglycerides': {'min': 0, 'max': 150, 'unit': 'mg/dL', 'name': 'Triglycerides'},
            
            # Liver Function Tests
            'alt': {'min': 7, 'max': 40, 'unit': 'U/L', 'name': 'ALT'},
            'ast': {'min': 8, 'max': 40, 'unit': 'U/L', 'name': 'AST'},
            'bilirubin_total': {'min': 0.2, 'max': 1.2, 'unit': 'mg/dL', 'name': 'Total Bilirubin'},
            'albumin': {'min': 3.5, 'max': 5.0, 'unit': 'g/dL', 'name': 'Albumin'},
            
            # Thyroid Function
            'tsh': {'min': 0.4, 'max': 4.0, 'unit': 'mIU/L', 'name': 'TSH'},
            't4': {'min': 4.5, 'max': 12.0, 'unit': 'μg/dL', 'name': 'T4'},
            't3': {'min': 80, 'max': 200, 'unit': 'ng/dL', 'name': 'T3'},
            
            # Cardiac Markers
            'troponin': {'min': 0, 'max': 0.04, 'unit': 'ng/mL', 'name': 'Troponin'},
            'ck_mb': {'min': 0, 'max': 6.3, 'unit': 'ng/mL', 'name': 'CK-MB'},
            
            # Inflammatory Markers
            'crp': {'min': 0, 'max': 3.0, 'unit': 'mg/L', 'name': 'C-Reactive Protein'},
            'esr': {'min': 0, 'max': 30, 'unit': 'mm/hr', 'name': 'ESR'},
            
            # Vitamins and Minerals
            'vitamin_d': {'min': 30, 'max': 100, 'unit': 'ng/mL', 'name': 'Vitamin D'},
            'b12': {'min': 200, 'max': 900, 'unit': 'pg/mL', 'name': 'Vitamin B12'},
            'iron': {'min': 60, 'max': 170, 'unit': 'μg/dL', 'name': 'Iron'},
            'ferritin': {'min': 12, 'max': 300, 'unit': 'ng/mL', 'name': 'Ferritin'}
        }
    
    def load_models(self):
        """Load pre-trained models for lab analysis"""
        try:
            # Load anomaly detection model
            self.anomaly_detector = joblib.load('models/lab_anomaly_detector.pkl')
            
            # Load risk prediction models
            self.risk_models['diabetes'] = joblib.load('models/diabetes_risk_model.pkl')
            self.risk_models['cardiovascular'] = joblib.load('models/cvd_risk_model.pkl')
            self.risk_models['kidney'] = joblib.load('models/kidney_risk_model.pkl')
            
            logger.info("Lab analysis models loaded successfully")
            
        except FileNotFoundError:
            logger.warning("Pre-trained models not found, using default models")
            self.train_default_models()
    
    def train_default_models(self):
        """Train default models if pre-trained ones are not available"""
        logger.info("Training default lab analysis models...")
        
        # Generate synthetic training data
        synthetic_data = self.generate_synthetic_lab_data(1000)
        
        # Train anomaly detector
        self.anomaly_detector.fit(synthetic_data)
        
        # Train risk models (simplified)
        self.risk_models['diabetes'] = RandomForestRegressor(n_estimators=50, random_state=42)
        self.risk_models['cardiovascular'] = RandomForestRegressor(n_estimators=50, random_state=42)
        self.risk_models['kidney'] = RandomForestRegressor(n_estimators=50, random_state=42)
        
        # Fit with synthetic data
        y_diabetes = np.random.random(len(synthetic_data))
        y_cvd = np.random.random(len(synthetic_data))
        y_kidney = np.random.random(len(synthetic_data))
        
        self.risk_models['diabetes'].fit(synthetic_data, y_diabetes)
        self.risk_models['cardiovascular'].fit(synthetic_data, y_cvd)
        self.risk_models['kidney'].fit(synthetic_data, y_kidney)
    
    def generate_synthetic_lab_data(self, n_samples: int) -> np.ndarray:
        """Generate synthetic lab data for training"""
        data = []
        
        for _ in range(n_samples):
            sample = []
            for test_name, ranges in self.reference_ranges.items():
                # Generate mostly normal values with some outliers
                if np.random.random() < 0.8:  # 80% normal
                    value = np.random.uniform(ranges['min'], ranges['max'])
                else:  # 20% abnormal
                    if np.random.random() < 0.5:
                        value = np.random.uniform(ranges['min'] * 0.5, ranges['min'])  # Low
                    else:
                        value = np.random.uniform(ranges['max'], ranges['max'] * 1.5)  # High
                
                sample.append(value)
            
            data.append(sample)
        
        return np.array(data)
    
    def analyze_lab_results(self, lab_data: Dict, patient_info: Dict = None) -> Dict:
        """
        Analyze lab results and provide comprehensive interpretation
        """
        try:
            # Validate and process lab data
            processed_data = self.process_lab_data(lab_data)
            
            # Individual test analysis
            test_results = self.analyze_individual_tests(processed_data)
            
            # Anomaly detection
            anomalies = self.detect_anomalies(processed_data)
            
            # Risk assessment
            risk_assessment = self.assess_health_risks(processed_data, patient_info)
            
            # Trend analysis (if historical data available)
            trends = self.analyze_trends(lab_data.get('historical_data', []))
            
            # Generate summary and recommendations
            summary = self.generate_summary(test_results, anomalies, risk_assessment)
            recommendations = self.generate_recommendations(test_results, risk_assessment)
            
            return {
                'summary': summary,
                'testResults': test_results,
                'anomalies': anomalies,
                'riskAssessment': risk_assessment,
                'trends': trends,
                'keyFindings': self.extract_key_findings(test_results, anomalies),
                'recommendations': recommendations,
                'overallRisk': self.calculate_overall_risk(risk_assessment),
                'followUpNeeded': self.determine_followup_needs(test_results, risk_assessment)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing lab results: {e}")
            return self.get_error_response()
    
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
        """Analyze each lab test individually"""
        results = []
        
        for test_name, value in lab_data.items():
            if test_name not in self.reference_ranges:
                continue
                
            ref_range = self.reference_ranges[test_name]
            
            # Determine status
            if value < ref_range['min']:
                status = 'Low'
                severity = self.calculate_severity(value, ref_range['min'], 'low')
            elif value > ref_range['max']:
                status = 'High'
                severity = self.calculate_severity(value, ref_range['max'], 'high')
            else:
                status = 'Normal'
                severity = 'Normal'
            
            # Clinical significance
            significance = self.assess_clinical_significance(test_name, value, status)
            
            results.append({
                'parameter': ref_range['name'],
                'value': value,
                'unit': ref_range['unit'],
                'referenceRange': f"{ref_range['min']}-{ref_range['max']}",
                'status': status,
                'severity': severity,
                'clinicalSignificance': significance,
                'interpretation': self.get_test_interpretation(test_name, value, status)
            })
        
        return results
    
    def calculate_severity(self, value: float, threshold: float, direction: str) -> str:
        """Calculate severity of abnormal values"""
        if direction == 'low':
            ratio = value / threshold
            if ratio < 0.5:
                return 'Critically Low'
            elif ratio < 0.8:
                return 'Moderately Low'
            else:
                return 'Mildly Low'
        else:  # high
            ratio = value / threshold
            if ratio > 2.0:
                return 'Critically High'
            elif ratio > 1.5:
                return 'Moderately High'
            else:
                return 'Mildly High'
    
    def assess_clinical_significance(self, test_name: str, value: float, status: str) -> str:
        """Assess clinical significance of test results"""
        if status == 'Normal':
            return 'No clinical concern'
        
        # Test-specific clinical significance
        significance_map = {
            'glucose': {
                'High': 'Possible diabetes or prediabetes - requires follow-up',
                'Low': 'Hypoglycemia - may require immediate attention'
            },
            'creatinine': {
                'High': 'Possible kidney dysfunction - nephrology consultation may be needed',
                'Low': 'Usually not clinically significant'
            },
            'troponin': {
                'High': 'Possible myocardial infarction - urgent cardiology evaluation needed',
                'Low': 'Normal - no cardiac injury detected'
            },
            'hemoglobin': {
                'Low': 'Anemia - investigate underlying cause',
                'High': 'Polycythemia - may require hematology evaluation'
            },
            'wbc': {
                'High': 'Possible infection or inflammation',
                'Low': 'Possible immunosuppression or bone marrow issue'
            }
        }
        
        return significance_map.get(test_name, {}).get(status, 'Abnormal value - clinical correlation recommended')
    
    def get_test_interpretation(self, test_name: str, value: float, status: str) -> str:
        """Get detailed interpretation for specific tests"""
        interpretations = {
            'glucose': {
                'Normal': 'Normal glucose metabolism',
                'High': f'Elevated glucose ({value} mg/dL) suggests diabetes or prediabetes',
                'Low': f'Low glucose ({value} mg/dL) may indicate hypoglycemia'
            },
            'total_cholesterol': {
                'Normal': 'Cholesterol within healthy range',
                'High': f'Elevated cholesterol ({value} mg/dL) increases cardiovascular risk',
                'Low': 'Low cholesterol - generally not concerning'
            },
            'creatinine': {
                'Normal': 'Normal kidney function',
                'High': f'Elevated creatinine ({value} mg/dL) suggests reduced kidney function',
                'Low': 'Low creatinine - usually not clinically significant'
            }
        }
        
        return interpretations.get(test_name, {}).get(status, f'{status} {test_name} level')
    
    def detect_anomalies(self, lab_data: Dict) -> List[Dict]:
        """Detect anomalous patterns in lab results"""
        anomalies = []
        
        try:
            # Prepare data for anomaly detection
            feature_vector = []
            test_names = []
            
            for test_name in sorted(self.reference_ranges.keys()):
                if test_name in lab_data:
                    feature_vector.append(lab_data[test_name])
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
                        'type': 'Pattern Anomaly',
                        'description': 'Unusual combination of lab values detected',
                        'severity': 'Medium' if anomaly_score > -0.5 else 'High',
                        'recommendation': 'Review results with healthcare provider'
                    })
        
        except Exception as e:
            logger.error(f"Error in anomaly detection: {e}")
        
        return anomalies
    
    def assess_health_risks(self, lab_data: Dict, patient_info: Dict = None) -> Dict:
        """Assess various health risks based on lab results"""
        risks = {}
        
        # Diabetes risk
        risks['diabetes'] = self.assess_diabetes_risk(lab_data, patient_info)
        
        # Cardiovascular risk
        risks['cardiovascular'] = self.assess_cardiovascular_risk(lab_data, patient_info)
        
        # Kidney disease risk
        risks['kidney'] = self.assess_kidney_risk(lab_data, patient_info)
        
        # Liver disease risk
        risks['liver'] = self.assess_liver_risk(lab_data, patient_info)
        
        return risks
    
    def assess_diabetes_risk(self, lab_data: Dict, patient_info: Dict = None) -> Dict:
        """Assess diabetes risk"""
        glucose = lab_data.get('glucose', 90)
        
        if glucose >= 126:
            risk_level = 'High'
            risk_score = 0.9
            interpretation = 'Diabetes likely - confirmatory testing needed'
        elif glucose >= 100:
            risk_level = 'Medium'
            risk_score = 0.6
            interpretation = 'Prediabetes - lifestyle modifications recommended'
        else:
            risk_level = 'Low'
            risk_score = 0.1
            interpretation = 'Normal glucose metabolism'
        
        return {
            'riskLevel': risk_level,
            'riskScore': risk_score,
            'interpretation': interpretation,
            'keyFactors': ['Fasting glucose level']
        }
    
    def assess_cardiovascular_risk(self, lab_data: Dict, patient_info: Dict = None) -> Dict:
        """Assess cardiovascular disease risk"""
        total_chol = lab_data.get('total_cholesterol', 180)
        ldl_chol = lab_data.get('ldl_cholesterol', 100)
        hdl_chol = lab_data.get('hdl_cholesterol', 50)
        
        risk_factors = 0
        key_factors = []
        
        if total_chol > 240:
            risk_factors += 2
            key_factors.append('High total cholesterol')
        elif total_chol > 200:
            risk_factors += 1
            key_factors.append('Borderline high cholesterol')
        
        if ldl_chol > 160:
            risk_factors += 2
            key_factors.append('High LDL cholesterol')
        elif ldl_chol > 130:
            risk_factors += 1
            key_factors.append('Borderline high LDL')
        
        if hdl_chol < 40:
            risk_factors += 1
            key_factors.append('Low HDL cholesterol')
        
        if risk_factors >= 3:
            risk_level = 'High'
            risk_score = 0.8
            interpretation = 'High cardiovascular risk - aggressive management needed'
        elif risk_factors >= 1:
            risk_level = 'Medium'
            risk_score = 0.5
            interpretation = 'Moderate cardiovascular risk - lifestyle modifications recommended'
        else:
            risk_level = 'Low'
            risk_score = 0.2
            interpretation = 'Low cardiovascular risk'
        
        return {
            'riskLevel': risk_level,
            'riskScore': risk_score,
            'interpretation': interpretation,
            'keyFactors': key_factors if key_factors else ['Normal lipid profile']
        }
    
    def assess_kidney_risk(self, lab_data: Dict, patient_info: Dict = None) -> Dict:
        """Assess kidney disease risk"""
        creatinine = lab_data.get('creatinine', 1.0)
        bun = lab_data.get('bun', 15)
        
        if creatinine > 1.5:
            risk_level = 'High'
            risk_score = 0.8
            interpretation = 'Possible kidney dysfunction - nephrology consultation recommended'
        elif creatinine > 1.2:
            risk_level = 'Medium'
            risk_score = 0.5
            interpretation = 'Mildly elevated creatinine - monitor kidney function'
        else:
            risk_level = 'Low'
            risk_score = 0.1
            interpretation = 'Normal kidney function'
        
        return {
            'riskLevel': risk_level,
            'riskScore': risk_score,
            'interpretation': interpretation,
            'keyFactors': ['Serum creatinine', 'BUN levels']
        }
    
    def assess_liver_risk(self, lab_data: Dict, patient_info: Dict = None) -> Dict:
        """Assess liver disease risk"""
        alt = lab_data.get('alt', 25)
        ast = lab_data.get('ast', 25)
        bilirubin = lab_data.get('bilirubin_total', 0.8)
        
        risk_factors = 0
        key_factors = []
        
        if alt > 80:
            risk_factors += 2
            key_factors.append('Significantly elevated ALT')
        elif alt > 40:
            risk_factors += 1
            key_factors.append('Mildly elevated ALT')
        
        if ast > 80:
            risk_factors += 2
            key_factors.append('Significantly elevated AST')
        elif ast > 40:
            risk_factors += 1
            key_factors.append('Mildly elevated AST')
        
        if bilirubin > 2.0:
            risk_factors += 2
            key_factors.append('Elevated bilirubin')
        
        if risk_factors >= 3:
            risk_level = 'High'
            risk_score = 0.8
            interpretation = 'Possible liver dysfunction - hepatology consultation recommended'
        elif risk_factors >= 1:
            risk_level = 'Medium'
            risk_score = 0.4
            interpretation = 'Mild liver enzyme elevation - monitor and investigate cause'
        else:
            risk_level = 'Low'
            risk_score = 0.1
            interpretation = 'Normal liver function'
        
        return {
            'riskLevel': risk_level,
            'riskScore': risk_score,
            'interpretation': interpretation,
            'keyFactors': key_factors if key_factors else ['Normal liver enzymes']
        }
    
    def analyze_trends(self, historical_data: List[Dict]) -> Dict:
        """Analyze trends in lab results over time"""
        if not historical_data or len(historical_data) < 2:
            return {'message': 'Insufficient historical data for trend analysis'}
        
        trends = {}
        
        # Analyze trends for key parameters
        key_params = ['glucose', 'total_cholesterol', 'creatinine', 'hemoglobin']
        
        for param in key_params:
            values = []
            dates = []
            
            for record in historical_data:
                if param in record and 'date' in record:
                    values.append(record[param])
                    dates.append(record['date'])
            
            if len(values) >= 2:
                # Calculate trend
                trend_direction = 'stable'
                if values[-1] > values[0] * 1.1:
                    trend_direction = 'increasing'
                elif values[-1] < values[0] * 0.9:
                    trend_direction = 'decreasing'
                
                trends[param] = {
                    'direction': trend_direction,
                    'currentValue': values[-1],
                    'previousValue': values[0],
                    'percentChange': ((values[-1] - values[0]) / values[0]) * 100,
                    'clinicalSignificance': self.assess_trend_significance(param, trend_direction, values)
                }
        
        return trends
    
    def assess_trend_significance(self, param: str, direction: str, values: List[float]) -> str:
        """Assess clinical significance of trends"""
        if direction == 'stable':
            return 'Stable values - continue current management'
        
        significance_map = {
            'glucose': {
                'increasing': 'Rising glucose trend - diabetes risk increasing',
                'decreasing': 'Improving glucose control'
            },
            'creatinine': {
                'increasing': 'Declining kidney function - nephrology follow-up needed',
                'decreasing': 'Improving kidney function'
            },
            'total_cholesterol': {
                'increasing': 'Worsening lipid profile - review medications and lifestyle',
                'decreasing': 'Improving cholesterol levels'
            }
        }
        
        return significance_map.get(param, {}).get(direction, f'{direction.capitalize()} trend noted')
    
    def generate_summary(self, test_results: List[Dict], anomalies: List[Dict], risk_assessment: Dict) -> str:
        """Generate overall summary of lab results"""
        abnormal_count = sum(1 for result in test_results if result['status'] != 'Normal')
        total_tests = len(test_results)
        
        if abnormal_count == 0:
            summary = "All lab values are within normal ranges. "
        else:
            summary = f"{abnormal_count} out of {total_tests} lab values are outside normal ranges. "
        
        # Add risk assessment summary
        high_risks = [risk for risk, data in risk_assessment.items() if data['riskLevel'] == 'High']
        if high_risks:
            summary += f"High risk identified for: {', '.join(high_risks)}. "
        
        # Add anomaly information
        if anomalies:
            summary += f"{len(anomalies)} unusual patterns detected. "
        
        summary += "Detailed analysis and recommendations provided below."
        
        return summary
    
    def extract_key_findings(self, test_results: List[Dict], anomalies: List[Dict]) -> List[str]:
        """Extract key findings from lab analysis"""
        findings = []
        
        # Critical abnormalities
        critical_tests = [result for result in test_results if 'Critical' in result.get('severity', '')]
        for test in critical_tests:
            findings.append(f"Critical: {test['parameter']} is {test['severity'].lower()}")
        
        # Significant abnormalities
        significant_tests = [result for result in test_results if result['status'] != 'Normal' and 'Critical' not in result.get('severity', '')]
        for test in significant_tests[:3]:  # Top 3
            findings.append(f"{test['parameter']}: {test['status']} ({test['value']} {test['unit']})")
        
        # Anomalies
        for anomaly in anomalies:
            findings.append(f"Pattern: {anomaly['description']}")
        
        return findings[:5]  # Return top 5 findings
    
    def generate_recommendations(self, test_results: List[Dict], risk_assessment: Dict) -> List[str]:
        """Generate recommendations based on lab results"""
        recommendations = []
        
        # Critical value recommendations
        critical_tests = [result for result in test_results if 'Critical' in result.get('severity', '')]
        if critical_tests:
            recommendations.append("Immediate medical attention required for critical lab values")
        
        # Risk-based recommendations
        for risk_type, risk_data in risk_assessment.items():
            if risk_data['riskLevel'] == 'High':
                if risk_type == 'diabetes':
                    recommendations.append("Endocrinology consultation for diabetes management")
                elif risk_type == 'cardiovascular':
                    recommendations.append("Cardiology evaluation and aggressive lipid management")
                elif risk_type == 'kidney':
                    recommendations.append("Nephrology consultation for kidney function assessment")
                elif risk_type == 'liver':
                    recommendations.append("Hepatology evaluation for liver function")
        
        # General recommendations
        abnormal_tests = [result for result in test_results if result['status'] != 'Normal']
        if abnormal_tests:
            recommendations.append("Repeat abnormal tests in 4-6 weeks to confirm results")
            recommendations.append("Discuss results with healthcare provider")
        
        # Lifestyle recommendations
        if any('cholesterol' in result['parameter'].lower() for result in abnormal_tests):
            recommendations.append("Consider dietary modifications and exercise for cholesterol management")
        
        if any('glucose' in result['parameter'].lower() for result in abnormal_tests):
            recommendations.append("Monitor blood glucose and consider dietary changes")
        
        return recommendations[:6]  # Return top 6 recommendations
    
    def calculate_overall_risk(self, risk_assessment: Dict) -> str:
        """Calculate overall health risk level"""
        high_risks = sum(1 for risk_data in risk_assessment.values() if risk_data['riskLevel'] == 'High')
        medium_risks = sum(1 for risk_data in risk_assessment.values() if risk_data['riskLevel'] == 'Medium')
        
        if high_risks >= 2:
            return 'High'
        elif high_risks >= 1 or medium_risks >= 2:
            return 'Medium'
        else:
            return 'Low'
    
    def determine_followup_needs(self, test_results: List[Dict], risk_assessment: Dict) -> Dict:
        """Determine follow-up needs based on results"""
        critical_count = sum(1 for result in test_results if 'Critical' in result.get('severity', ''))
        high_risk_count = sum(1 for risk_data in risk_assessment.values() if risk_data['riskLevel'] == 'High')
        
        if critical_count > 0:
            urgency = 'Immediate'
            timeframe = 'Within 24 hours'
        elif high_risk_count > 0:
            urgency = 'Urgent'
            timeframe = 'Within 1 week'
        else:
            urgency = 'Routine'
            timeframe = 'Within 1 month'
        
        return {
            'urgency': urgency,
            'timeframe': timeframe,
            'specialistNeeded': high_risk_count > 0 or critical_count > 0
        }
    
    def get_error_response(self) -> Dict:
        """Return error response when analysis fails"""
        return {
            'summary': 'Lab analysis could not be completed due to technical error',
            'testResults': [],
            'anomalies': [],
            'riskAssessment': {},
            'trends': {},
            'keyFindings': ['Analysis error occurred'],
            'recommendations': [
                'Please have results reviewed by healthcare provider',
                'Ensure all lab values are entered correctly',
                'Contact support if error persists'
            ],
            'overallRisk': 'Unknown',
            'followUpNeeded': {
                'urgency': 'Routine',
                'timeframe': 'As clinically indicated',
                'specialistNeeded': False
            }
        }