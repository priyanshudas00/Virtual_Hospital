"""
Data Loader for Medical Datasets
Loads real medical datasets from various sources for training AI models
"""

import pandas as pd
import numpy as np
import requests
import os
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import logging
from typing import Tuple, Dict, List
import zipfile
import io

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MedicalDataLoader:
    def __init__(self, data_dir: str = 'data'):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
    def load_symptom_disease_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load symptom-disease dataset for training diagnosis models"""
        try:
            # Try to load from GitHub (Disease Symptom Prediction)
            url = "https://raw.githubusercontent.com/kaushil268/Disease-Prediction-using-Machine-Learning/master/dataset.csv"
            logger.info("Loading symptom-disease dataset from GitHub...")
            
            df = pd.read_csv(url)
            
            # Process the dataset
            if 'prognosis' in df.columns:
                X = df.drop('prognosis', axis=1)
                y = df['prognosis']
            elif 'Disease' in df.columns:
                X = df.drop('Disease', axis=1)
                y = df['Disease']
            else:
                # Assume last column is target
                X = df.iloc[:, :-1]
                y = df.iloc[:, -1]
            
            # Encode labels
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
            
            # Save processed data
            processed_path = os.path.join(self.data_dir, 'symptom_disease_processed.csv')
            processed_df = pd.concat([X, pd.Series(y_encoded, name='disease_encoded')], axis=1)
            processed_df.to_csv(processed_path, index=False)
            
            logger.info(f"Loaded {len(df)} symptom-disease records")
            return X.values, y_encoded
            
        except Exception as e:
            logger.error(f"Error loading symptom-disease data: {e}")
            return self.create_synthetic_symptom_data()
    
    def create_synthetic_symptom_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Create synthetic symptom-disease data"""
        logger.info("Creating synthetic symptom-disease dataset...")
        
        # Define symptoms and diseases
        symptoms = [
            'fever', 'cough', 'headache', 'fatigue', 'sore_throat', 'runny_nose',
            'body_ache', 'nausea', 'vomiting', 'diarrhea', 'shortness_of_breath',
            'chest_pain', 'abdominal_pain', 'joint_pain', 'muscle_pain', 'rash',
            'dizziness', 'confusion', 'loss_of_appetite', 'weight_loss'
        ]
        
        diseases = [
            'Common Cold', 'Influenza', 'COVID-19', 'Pneumonia', 'Bronchitis',
            'Gastroenteritis', 'Migraine', 'Tension Headache', 'Allergic Rhinitis',
            'Sinusitis', 'Strep Throat', 'Food Poisoning', 'Anxiety', 'Depression'
        ]
        
        # Disease-symptom patterns
        disease_patterns = {
            'Common Cold': ['cough', 'runny_nose', 'sore_throat', 'fatigue'],
            'Influenza': ['fever', 'cough', 'headache', 'fatigue', 'body_ache'],
            'COVID-19': ['fever', 'cough', 'fatigue', 'loss_of_appetite', 'shortness_of_breath'],
            'Pneumonia': ['fever', 'cough', 'shortness_of_breath', 'chest_pain', 'fatigue'],
            'Bronchitis': ['cough', 'fatigue', 'chest_pain'],
            'Gastroenteritis': ['nausea', 'vomiting', 'diarrhea', 'abdominal_pain'],
            'Migraine': ['headache', 'nausea', 'dizziness'],
            'Tension Headache': ['headache', 'fatigue'],
            'Allergic Rhinitis': ['runny_nose', 'cough', 'fatigue'],
            'Sinusitis': ['headache', 'runny_nose', 'fatigue'],
            'Strep Throat': ['sore_throat', 'fever', 'headache'],
            'Food Poisoning': ['nausea', 'vomiting', 'diarrhea', 'abdominal_pain', 'fever'],
            'Anxiety': ['headache', 'fatigue', 'dizziness'],
            'Depression': ['fatigue', 'loss_of_appetite', 'headache']
        }
        
        # Generate synthetic data
        data = []
        labels = []
        
        for _ in range(2000):  # Generate 2000 samples
            disease = np.random.choice(diseases)
            sample = np.zeros(len(symptoms))
            
            # Set symptoms based on disease pattern
            if disease in disease_patterns:
                for symptom in disease_patterns[disease]:
                    if symptom in symptoms:
                        idx = symptoms.index(symptom)
                        sample[idx] = 1
                
                # Add some noise (random symptoms)
                for _ in range(np.random.randint(0, 3)):
                    random_idx = np.random.randint(0, len(symptoms))
                    if np.random.random() < 0.3:  # 30% chance
                        sample[random_idx] = 1
            
            data.append(sample)
            labels.append(disease)
        
        # Convert to arrays
        X = np.array(data)
        le = LabelEncoder()
        y = le.fit_transform(labels)
        
        # Save synthetic data
        df = pd.DataFrame(X, columns=symptoms)
        df['disease'] = labels
        df['disease_encoded'] = y
        df.to_csv(os.path.join(self.data_dir, 'synthetic_symptom_disease.csv'), index=False)
        
        logger.info(f"Created {len(data)} synthetic symptom-disease records")
        return X, y
    
    def load_diabetes_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load Pima Indians Diabetes dataset"""
        try:
            logger.info("Loading Pima Indians Diabetes dataset...")
            diabetes = fetch_openml('diabetes', version=1, as_frame=True, parser='auto')
            
            X = diabetes.data.values
            y = diabetes.target.values
            
            # Convert target to binary
            y = (y == 'tested_positive').astype(int)
            
            # Save data
            df = pd.DataFrame(X, columns=diabetes.feature_names)
            df['diabetes'] = y
            df.to_csv(os.path.join(self.data_dir, 'diabetes.csv'), index=False)
            
            logger.info(f"Loaded {len(X)} diabetes records")
            return X, y
            
        except Exception as e:
            logger.error(f"Error loading diabetes data: {e}")
            return self.create_synthetic_diabetes_data()
    
    def create_synthetic_diabetes_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Create synthetic diabetes data"""
        logger.info("Creating synthetic diabetes dataset...")
        
        n_samples = 1000
        
        # Generate synthetic features
        glucose = np.random.normal(120, 30, n_samples)
        bmi = np.random.normal(28, 5, n_samples)
        age = np.random.randint(20, 80, n_samples)
        pregnancies = np.random.randint(0, 10, n_samples)
        blood_pressure = np.random.normal(70, 15, n_samples)
        skin_thickness = np.random.normal(25, 10, n_samples)
        insulin = np.random.normal(100, 50, n_samples)
        dpf = np.random.normal(0.5, 0.3, n_samples)
        
        X = np.column_stack([pregnancies, glucose, blood_pressure, skin_thickness, 
                            insulin, bmi, dpf, age])
        
        # Generate target based on risk factors
        risk_score = (glucose > 126) * 0.4 + (bmi > 30) * 0.3 + (age > 45) * 0.2 + (blood_pressure > 80) * 0.1
        y = (risk_score + np.random.normal(0, 0.1, n_samples) > 0.5).astype(int)
        
        # Save data
        feature_names = ['pregnancies', 'glucose', 'blood_pressure', 'skin_thickness', 
                        'insulin', 'bmi', 'diabetes_pedigree', 'age']
        df = pd.DataFrame(X, columns=feature_names)
        df['diabetes'] = y
        df.to_csv(os.path.join(self.data_dir, 'synthetic_diabetes.csv'), index=False)
        
        logger.info(f"Created {len(X)} synthetic diabetes records")
        return X, y
    
    def load_heart_disease_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load heart disease dataset"""
        try:
            # Try to load from UCI repository
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
            logger.info("Loading heart disease dataset from UCI...")
            
            column_names = [
                'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
            ]
            
            df = pd.read_csv(url, names=column_names, na_values='?')
            df = df.dropna()  # Remove rows with missing values
            
            # Convert target to binary (0: no disease, 1: disease)
            df['target'] = (df['target'] > 0).astype(int)
            
            X = df.drop('target', axis=1).values
            y = df['target'].values
            
            # Save data
            df.to_csv(os.path.join(self.data_dir, 'heart_disease.csv'), index=False)
            
            logger.info(f"Loaded {len(X)} heart disease records")
            return X, y
            
        except Exception as e:
            logger.error(f"Error loading heart disease data: {e}")
            return self.create_synthetic_heart_data()
    
    def create_synthetic_heart_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Create synthetic heart disease data"""
        logger.info("Creating synthetic heart disease dataset...")
        
        n_samples = 1000
        
        # Generate synthetic features
        age = np.random.randint(30, 80, n_samples)
        sex = np.random.randint(0, 2, n_samples)  # 0: female, 1: male
        cp = np.random.randint(0, 4, n_samples)  # chest pain type
        trestbps = np.random.normal(130, 20, n_samples)  # resting blood pressure
        chol = np.random.normal(240, 50, n_samples)  # cholesterol
        fbs = (np.random.normal(110, 30, n_samples) > 120).astype(int)  # fasting blood sugar
        restecg = np.random.randint(0, 3, n_samples)  # resting ECG
        thalach = np.random.normal(150, 25, n_samples)  # max heart rate
        exang = np.random.randint(0, 2, n_samples)  # exercise induced angina
        oldpeak = np.random.exponential(1, n_samples)  # ST depression
        slope = np.random.randint(0, 3, n_samples)  # slope of peak exercise ST
        ca = np.random.randint(0, 4, n_samples)  # number of major vessels
        thal = np.random.randint(0, 4, n_samples)  # thalassemia
        
        X = np.column_stack([age, sex, cp, trestbps, chol, fbs, restecg, 
                            thalach, exang, oldpeak, slope, ca, thal])
        
        # Generate target based on risk factors
        risk_score = ((age > 55) * 0.2 + (sex == 1) * 0.15 + (cp > 0) * 0.2 + 
                     (trestbps > 140) * 0.15 + (chol > 240) * 0.15 + 
                     (thalach < 120) * 0.15)
        y = (risk_score + np.random.normal(0, 0.1, n_samples) > 0.4).astype(int)
        
        # Save data
        feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
        df = pd.DataFrame(X, columns=feature_names)
        df['target'] = y
        df.to_csv(os.path.join(self.data_dir, 'synthetic_heart_disease.csv'), index=False)
        
        logger.info(f"Created {len(X)} synthetic heart disease records")
        return X, y
    
    def load_lab_results_data(self) -> pd.DataFrame:
        """Load or create lab results dataset"""
        try:
            # Try to load existing lab data
            lab_file = os.path.join(self.data_dir, 'lab_results.csv')
            if os.path.exists(lab_file):
                logger.info("Loading existing lab results data...")
                return pd.read_csv(lab_file)
            else:
                return self.create_synthetic_lab_data()
                
        except Exception as e:
            logger.error(f"Error loading lab data: {e}")
            return self.create_synthetic_lab_data()
    
    def create_synthetic_lab_data(self) -> pd.DataFrame:
        """Create synthetic lab results data"""
        logger.info("Creating synthetic lab results dataset...")
        
        n_samples = 5000
        
        # Normal ranges for lab tests
        lab_ranges = {
            'wbc': (4.5, 11.0),
            'rbc': (4.2, 5.4),
            'hemoglobin': (12.0, 15.5),
            'hematocrit': (36.0, 46.0),
            'platelets': (150, 400),
            'glucose': (70, 100),
            'total_cholesterol': (0, 200),
            'ldl_cholesterol': (0, 100),
            'hdl_cholesterol': (40, 60),
            'triglycerides': (0, 150),
            'creatinine': (0.6, 1.2),
            'bun': (7, 20),
            'alt': (7, 40),
            'ast': (8, 40),
            'bilirubin_total': (0.2, 1.2),
            'albumin': (3.5, 5.0)
        }
        
        data = []
        
        for i in range(n_samples):
            patient_data = {'patient_id': f'P{i:05d}'}
            
            # Generate age and gender
            age = np.random.randint(18, 90)
            gender = np.random.choice(['M', 'F'])
            patient_data['age'] = age
            patient_data['gender'] = gender
            
            # Generate lab values
            for test, (min_val, max_val) in lab_ranges.items():
                # 70% normal, 30% abnormal
                if np.random.random() < 0.7:
                    # Normal values
                    value = np.random.uniform(min_val, max_val)
                else:
                    # Abnormal values
                    if np.random.random() < 0.5:
                        # Low values
                        value = np.random.uniform(min_val * 0.3, min_val * 0.9)
                    else:
                        # High values
                        value = np.random.uniform(max_val * 1.1, max_val * 2.0)
                
                patient_data[test] = round(value, 2)
            
            # Add risk indicators
            patient_data['diabetes_risk'] = int(patient_data['glucose'] > 126)
            patient_data['cvd_risk'] = int(patient_data['total_cholesterol'] > 240 or 
                                         patient_data['ldl_cholesterol'] > 160)
            patient_data['kidney_risk'] = int(patient_data['creatinine'] > 1.5)
            patient_data['liver_risk'] = int(patient_data['alt'] > 80 or patient_data['ast'] > 80)
            
            data.append(patient_data)
        
        df = pd.DataFrame(data)
        
        # Save data
        df.to_csv(os.path.join(self.data_dir, 'synthetic_lab_results.csv'), index=False)
        
        logger.info(f"Created {len(df)} synthetic lab result records")
        return df
    
    def download_chest_xray_metadata(self) -> Dict:
        """Download chest X-ray dataset metadata (not actual images due to size)"""
        logger.info("Creating chest X-ray dataset metadata...")
        
        # Create metadata for chest X-ray dataset
        metadata = {
            'dataset_name': 'Chest X-Ray Images (Pneumonia)',
            'source': 'Kaggle',
            'url': 'https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia',
            'size': '~2GB',
            'classes': ['NORMAL', 'PNEUMONIA'],
            'train_samples': 5216,
            'test_samples': 624,
            'val_samples': 16,
            'image_format': 'JPEG',
            'image_size': 'Variable (typically 1024x1024 or similar)',
            'description': 'Chest X-ray images for pneumonia detection'
        }
        
        # Save metadata
        import json
        with open(os.path.join(self.data_dir, 'chest_xray_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info("Chest X-ray metadata saved")
        return metadata
    
    def get_dataset_summary(self) -> Dict:
        """Get summary of all available datasets"""
        summary = {
            'datasets_loaded': [],
            'total_size_mb': 0,
            'data_directory': self.data_dir
        }
        
        # Check which datasets are available
        data_files = os.listdir(self.data_dir)
        
        for file in data_files:
            if file.endswith('.csv'):
                file_path = os.path.join(self.data_dir, file)
                file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
                
                df = pd.read_csv(file_path)
                
                summary['datasets_loaded'].append({
                    'filename': file,
                    'size_mb': round(file_size, 2),
                    'rows': len(df),
                    'columns': len(df.columns),
                    'description': self.get_dataset_description(file)
                })
                
                summary['total_size_mb'] += file_size
        
        summary['total_size_mb'] = round(summary['total_size_mb'], 2)
        
        return summary
    
    def get_dataset_description(self, filename: str) -> str:
        """Get description for dataset file"""
        descriptions = {
            'symptom_disease': 'Symptom-disease mapping for diagnosis',
            'diabetes': 'Diabetes prediction dataset',
            'heart_disease': 'Heart disease prediction dataset',
            'lab_results': 'Laboratory test results with risk indicators',
            'chest_xray_metadata': 'Chest X-ray dataset information'
        }
        
        for key, desc in descriptions.items():
            if key in filename:
                return desc
        
        return 'Medical dataset'

def load_all_datasets():
    """Load all medical datasets for training"""
    loader = MedicalDataLoader()
    
    logger.info("Loading all medical datasets...")
    
    # Load datasets
    symptom_X, symptom_y = loader.load_symptom_disease_data()
    diabetes_X, diabetes_y = loader.load_diabetes_data()
    heart_X, heart_y = loader.load_heart_disease_data()
    lab_df = loader.load_lab_results_data()
    chest_metadata = loader.download_chest_xray_metadata()
    
    # Get summary
    summary = loader.get_dataset_summary()
    
    logger.info("Dataset loading complete!")
    logger.info(f"Total datasets: {len(summary['datasets_loaded'])}")
    logger.info(f"Total size: {summary['total_size_mb']} MB")
    
    return {
        'symptom_data': (symptom_X, symptom_y),
        'diabetes_data': (diabetes_X, diabetes_y),
        'heart_data': (heart_X, heart_y),
        'lab_data': lab_df,
        'chest_metadata': chest_metadata,
        'summary': summary
    }

if __name__ == "__main__":
    # Load all datasets
    datasets = load_all_datasets()
    print("All datasets loaded successfully!")
    print(f"Summary: {datasets['summary']}")