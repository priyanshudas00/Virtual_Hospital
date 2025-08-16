"""
Data preparation script for AI medical models
Downloads and prepares medical datasets for training
"""

import pandas as pd
import numpy as np
import requests
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class MedicalDataLoader:
    def __init__(self):
        self.data_dir = '../data'
        os.makedirs(self.data_dir, exist_ok=True)
    
    def load_symptom_disease_data(self):
        """Load symptom-disease dataset from GitHub"""
        try:
            # Dataset from Kaggle/GitHub (Disease Symptom Prediction)
            url = "https://raw.githubusercontent.com/kaushil268/Disease-Prediction-using-Machine-Learning/master/dataset.csv"
            
            print("📥 Downloading symptom-disease dataset...")
            df = pd.read_csv(url)
            
            # Save locally
            df.to_csv(f"{self.data_dir}/symptom_disease.csv", index=False)
            print(f"✅ Saved {len(df)} records to symptom_disease.csv")
            
            return df
            
        except Exception as e:
            print(f"❌ Error loading symptom data: {e}")
            # Return mock data
            return self.create_mock_symptom_data()
    
    def create_mock_symptom_data(self):
        """Create mock symptom-disease data"""
        print("🔄 Creating mock symptom dataset...")
        
        symptoms = ['fever', 'cough', 'headache', 'fatigue', 'sore_throat', 
                   'runny_nose', 'body_ache', 'nausea', 'vomiting', 'diarrhea']
        
        diseases = ['Common Cold', 'Flu', 'COVID-19', 'Allergies', 'Migraine', 
                   'Food Poisoning', 'Bronchitis', 'Pneumonia']
        
        # Generate synthetic data
        data = []
        for _ in range(1000):
            record = {}
            disease = np.random.choice(diseases)
            
            # Set symptoms based on disease patterns
            if disease == 'Common Cold':
                record = {'fever': 0, 'cough': 1, 'headache': 0, 'fatigue': 1, 
                         'sore_throat': 1, 'runny_nose': 1}
            elif disease == 'Flu':
                record = {'fever': 1, 'cough': 1, 'headache': 1, 'fatigue': 1, 
                         'sore_throat': 1, 'body_ache': 1}
            elif disease == 'COVID-19':
                record = {'fever': 1, 'cough': 1, 'fatigue': 1, 'sore_throat': 1}
            # Add more patterns...
            
            record['disease'] = disease
            data.append(record)
        
        df = pd.DataFrame(data).fillna(0)
        df.to_csv(f"{self.data_dir}/mock_symptom_disease.csv", index=False)
        return df
    
    def load_diabetes_data(self):
        """Load Pima Indians Diabetes dataset"""
        try:
            from sklearn.datasets import fetch_openml
            
            print("📥 Loading Pima Indians Diabetes dataset...")
            diabetes = fetch_openml('diabetes', version=1, as_frame=True, parser='auto')
            
            df = pd.concat([diabetes.data, diabetes.target], axis=1)
            df.to_csv(f"{self.data_dir}/diabetes.csv", index=False)
            print(f"✅ Saved {len(df)} diabetes records")
            
            return df
            
        except Exception as e:
            print(f"❌ Error loading diabetes data: {e}")
            return None
    
    def prepare_chest_xray_paths(self):
        """Prepare chest X-ray dataset paths (Kaggle)"""
        # This would normally download from Kaggle API
        # For now, just create the directory structure
        
        xray_dir = f"{self.data_dir}/chest_xray"
        os.makedirs(f"{xray_dir}/train/normal", exist_ok=True)
        os.makedirs(f"{xray_dir}/train/pneumonia", exist_ok=True)
        os.makedirs(f"{xray_dir}/test/normal", exist_ok=True)
        os.makedirs(f"{xray_dir}/test/pneumonia", exist_ok=True)
        
        print(f"📁 Created chest X-ray directory structure at {xray_dir}")
        print("💡 To use real data, download from: https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia")
        
        return xray_dir
    
    def create_lab_results_data(self):
        """Create mock lab results dataset"""
        print("🧪 Creating mock lab results dataset...")
        
        # Normal ranges for common lab tests
        lab_tests = {
            'wbc': (4.5, 11.0),      # White blood cells (K/μL)
            'rbc': (4.2, 5.4),       # Red blood cells (M/μL)
            'hemoglobin': (12.0, 15.5),  # g/dL
            'hematocrit': (36.0, 46.0),  # %
            'platelets': (150, 400),      # K/μL
            'glucose': (70, 100),         # mg/dL
            'cholesterol': (0, 200),      # mg/dL
            'ldl': (0, 100),             # mg/dL
            'hdl': (40, 60),             # mg/dL
            'triglycerides': (0, 150),   # mg/dL
            'creatinine': (0.6, 1.2),    # mg/dL
            'bun': (7, 20),              # mg/dL
        }
        
        data = []
        for i in range(1000):
            record = {'patient_id': f'P{i:04d}'}
            
            for test, (min_val, max_val) in lab_tests.items():
                # 80% normal, 20% abnormal
                if np.random.random() < 0.8:
                    value = np.random.uniform(min_val, max_val)
                else:
                    # Abnormal values
                    if np.random.random() < 0.5:
                        value = np.random.uniform(min_val * 0.5, min_val)  # Low
                    else:
                        value = np.random.uniform(max_val, max_val * 1.5)  # High
                
                record[test] = round(value, 2)
            
            # Add risk assessment
            if record['cholesterol'] > 200 or record['glucose'] > 100:
                record['risk_level'] = 'High'
            elif record['cholesterol'] > 180 or record['glucose'] > 90:
                record['risk_level'] = 'Medium'
            else:
                record['risk_level'] = 'Low'
            
            data.append(record)
        
        df = pd.DataFrame(data)
        df.to_csv(f"{self.data_dir}/lab_results.csv", index=False)
        print(f"✅ Created {len(df)} lab result records")
        
        return df
    
    def download_covid_data(self):
        """Download COVID-19 dataset (subset)"""
        try:
            # COVID-19 symptom dataset
            url = "https://raw.githubusercontent.com/nshomron/covidpred/master/data/covid_dataset.csv"
            
            print("📥 Downloading COVID-19 dataset...")
            df = pd.read_csv(url)
            
            # Take subset to keep under size limit
            df_subset = df.sample(n=min(1000, len(df)), random_state=42)
            df_subset.to_csv(f"{self.data_dir}/covid_symptoms.csv", index=False)
            print(f"✅ Saved {len(df_subset)} COVID-19 records")
            
            return df_subset
            
        except Exception as e:
            print(f"❌ Error loading COVID data: {e}")
            return None

def main():
    """Main data preparation pipeline"""
    print("🏥 AI Hospital Data Preparation")
    print("=" * 50)
    
    loader = MedicalDataLoader()
    
    # Load all datasets
    print("\n1. Loading Symptom-Disease Data...")
    symptom_data = loader.load_symptom_disease_data()
    
    print("\n2. Loading Diabetes Data...")
    diabetes_data = loader.load_diabetes_data()
    
    print("\n3. Preparing Chest X-ray Structure...")
    xray_dir = loader.prepare_chest_xray_paths()
    
    print("\n4. Creating Lab Results Data...")
    lab_data = loader.create_lab_results_data()
    
    print("\n5. Downloading COVID-19 Data...")
    covid_data = loader.download_covid_data()
    
    print("\n" + "=" * 50)
    print("✅ Data preparation complete!")
    print(f"📊 Datasets saved in: {loader.data_dir}")
    print("\n📋 Available datasets:")
    
    for file in os.listdir(loader.data_dir):
        if file.endswith('.csv'):
            df = pd.read_csv(f"{loader.data_dir}/{file}")
            print(f"  • {file}: {len(df)} records, {len(df.columns)} columns")
    
    print("\n🚀 Ready for model training!")

if __name__ == "__main__":
    main()