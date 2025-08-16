import React, { useState } from 'react';
import { motion } from 'framer-motion';
import {
  Heart,
  Pill,
  Clock,
  AlertCircle,
  CheckCircle,
  Calendar,
  User,
  Weight,
  Zap,
  Shield
} from 'lucide-react';

export const TreatmentPage: React.FC = () => {
  const [selectedCondition, setSelectedCondition] = useState('hypertension');

  const patientProfile = {
    age: 45,
    weight: 75,
    height: 170,
    allergies: ['Penicillin'],
    currentMedications: ['Vitamin D3', 'Multivitamin'],
    medicalHistory: ['Type 2 Diabetes', 'Hypertension']
  };

  const treatmentPlans = {
    hypertension: {
      condition: 'Essential Hypertension',
      severity: 'Stage 1',
      primaryTreatment: {
        medication: 'Lisinopril',
        dosage: '10mg once daily',
        duration: 'Long-term',
        instructions: 'Take in the morning with or without food'
      },
      alternativeTreatments: [
        {
          medication: 'Amlodipine',
          dosage: '5mg once daily',
          reason: 'ACE inhibitor intolerance'
        },
        {
          medication: 'Hydrochlorothiazide',
          dosage: '25mg once daily',
          reason: 'Combination therapy if needed'
        }
      ],
      lifestyle: [
        'Reduce sodium intake to less than 2,300mg daily',
        'Exercise 30 minutes, 5 days per week',
        'Maintain healthy weight (BMI 18.5-24.9)',
        'Limit alcohol consumption',
        'Manage stress through relaxation techniques'
      ],
      monitoring: [
        'Check blood pressure weekly at home',
        'Monthly follow-up for first 3 months',
        'Lab tests every 6 months',
        'Annual eye and kidney function tests'
      ],
      drugInteractions: [
        'NSAIDs may reduce effectiveness',
        'Potassium supplements require monitoring',
        'Lithium levels may increase'
      ]
    },
    diabetes: {
      condition: 'Type 2 Diabetes Mellitus',
      severity: 'Well-controlled',
      primaryTreatment: {
        medication: 'Metformin',
        dosage: '1000mg twice daily',
        duration: 'Long-term',
        instructions: 'Take with meals to reduce GI upset'
      },
      alternativeTreatments: [
        {
          medication: 'Glipizide',
          dosage: '5mg before breakfast',
          reason: 'If metformin contraindicated'
        },
        {
          medication: 'Sitagliptin',
          dosage: '100mg once daily',
          reason: 'Add-on therapy if HbA1c >7%'
        }
      ],
      lifestyle: [
        'Follow diabetic diet plan',
        'Regular blood glucose monitoring',
        'Carbohydrate counting',
        'Regular physical activity',
        'Weight management'
      ],
      monitoring: [
        'Daily blood glucose checks',
        'HbA1c every 3 months',
        'Annual eye examination',
        'Quarterly clinical visits',
        'Annual foot examination'
      ],
      drugInteractions: [
        'Alcohol may cause hypoglycemia',
        'Corticosteroids may increase blood sugar',
        'Beta-blockers may mask hypoglycemia'
      ]
    }
  };

  const currentPlan = treatmentPlans[selectedCondition as keyof typeof treatmentPlans];

  const sideEffects = {
    common: ['Dizziness', 'Fatigue', 'Dry cough'],
    serious: ['Angioedema', 'Hyperkalemia', 'Kidney problems'],
    whenToCall: [
      'Severe dizziness or fainting',
      'Difficulty breathing or swallowing',
      'Severe fatigue or weakness',
      'Unusual swelling of face or throat'
    ]
  };

  return (
    <div className="min-h-screen py-8 px-4 sm:px-6 lg:px-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-12"
        >
          <div className="bg-gradient-to-r from-red-500 to-pink-600 w-16 h-16 rounded-2xl flex items-center justify-center mx-auto mb-4">
            <Heart className="w-8 h-8 text-white" />
          </div>
          <h1 className="text-3xl font-bold text-gray-900 mb-2">AI Treatment Planner</h1>
          <p className="text-gray-600">Personalized treatment recommendations based on your medical profile</p>
        </motion.div>

        <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
          {/* Patient Profile */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            className="lg:col-span-1"
          >
            <div className="bg-white rounded-2xl shadow-lg p-6 mb-6">
              <h2 className="text-lg font-bold text-gray-900 mb-4 flex items-center space-x-2">
                <User className="w-5 h-5" />
                <span>Patient Profile</span>
              </h2>
              <div className="space-y-3">
                <div className="flex justify-between">
                  <span className="text-gray-600">Age:</span>
                  <span className="font-medium">{patientProfile.age} years</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Weight:</span>
                  <span className="font-medium">{patientProfile.weight} kg</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Height:</span>
                  <span className="font-medium">{patientProfile.height} cm</span>
                </div>
                <div>
                  <span className="text-gray-600">Allergies:</span>
                  <div className="mt-1">
                    {patientProfile.allergies.map((allergy, index) => (
                      <span key={index} className="inline-block bg-red-100 text-red-800 text-xs px-2 py-1 rounded-full mr-1">
                        {allergy}
                      </span>
                    ))}
                  </div>
                </div>
                <div>
                  <span className="text-gray-600">Current Medications:</span>
                  <div className="mt-1">
                    {patientProfile.currentMedications.map((med, index) => (
                      <div key={index} className="text-sm text-gray-700">{med}</div>
                    ))}
                  </div>
                </div>
              </div>
            </div>

            {/* Condition Selection */}
            <div className="bg-white rounded-2xl shadow-lg p-6">
              <h3 className="text-lg font-bold text-gray-900 mb-4">Select Condition</h3>
              <div className="space-y-2">
                {Object.entries(treatmentPlans).map(([key, plan]) => (
                  <button
                    key={key}
                    onClick={() => setSelectedCondition(key)}
                    className={`w-full text-left p-3 rounded-xl transition-all duration-200 ${
                      selectedCondition === key
                        ? 'bg-red-50 border-2 border-red-200 text-red-900'
                        : 'hover:bg-gray-50 border-2 border-transparent'
                    }`}
                  >
                    <div className="font-medium">{plan.condition}</div>
                    <div className="text-sm text-gray-600">Severity: {plan.severity}</div>
                  </button>
                ))}
              </div>
            </div>
          </motion.div>

          {/* Treatment Plan */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="lg:col-span-3 space-y-6"
          >
            {/* Primary Treatment */}
            <div className="bg-white rounded-2xl shadow-lg p-8">
              <h2 className="text-2xl font-bold text-gray-900 mb-6 flex items-center space-x-2">
                <Pill className="w-6 h-6 text-red-600" />
                <span>Primary Treatment: {currentPlan.condition}</span>
              </h2>

              <div className="bg-gradient-to-r from-red-50 to-pink-50 rounded-xl p-6 border border-red-100">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <h3 className="font-bold text-gray-900 mb-2">Medication</h3>
                    <p className="text-xl font-bold text-red-600 mb-1">{currentPlan.primaryTreatment.medication}</p>
                    <p className="text-gray-700 mb-2">{currentPlan.primaryTreatment.dosage}</p>
                    <p className="text-sm text-gray-600">{currentPlan.primaryTreatment.instructions}</p>
                  </div>
                  <div>
                    <h3 className="font-bold text-gray-900 mb-2">Duration</h3>
                    <div className="flex items-center space-x-2 mb-4">
                      <Clock className="w-5 h-5 text-gray-600" />
                      <span className="text-gray-700">{currentPlan.primaryTreatment.duration}</span>
                    </div>
                    <div className="flex items-center space-x-2">
                      <Shield className="w-5 h-5 text-green-600" />
                      <span className="text-sm text-green-700">FDA Approved</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Alternative Treatments */}
            <div className="bg-white rounded-2xl shadow-lg p-8">
              <h3 className="text-xl font-bold text-gray-900 mb-6">Alternative Treatment Options</h3>
              <div className="space-y-4">
                {currentPlan.alternativeTreatments.map((treatment, index) => (
                  <div key={index} className="border border-gray-200 rounded-xl p-4">
                    <div className="flex justify-between items-start mb-2">
                      <div>
                        <h4 className="font-bold text-gray-900">{treatment.medication}</h4>
                        <p className="text-gray-700">{treatment.dosage}</p>
                      </div>
                      <span className="bg-blue-100 text-blue-800 text-xs px-2 py-1 rounded-full">
                        Alternative
                      </span>
                    </div>
                    <p className="text-sm text-gray-600">
                      <strong>Use case:</strong> {treatment.reason}
                    </p>
                  </div>
                ))}
              </div>
            </div>

            {/* Lifestyle Modifications & Monitoring */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Lifestyle */}
              <div className="bg-white rounded-2xl shadow-lg p-6">
                <h3 className="text-xl font-bold text-gray-900 mb-4 flex items-center space-x-2">
                  <Zap className="w-5 h-5 text-orange-600" />
                  <span>Lifestyle Modifications</span>
                </h3>
                <div className="space-y-3">
                  {currentPlan.lifestyle.map((item, index) => (
                    <div key={index} className="flex items-start space-x-3">
                      <div className="bg-orange-100 rounded-full p-1 mt-1">
                        <div className="w-2 h-2 bg-orange-600 rounded-full"></div>
                      </div>
                      <p className="text-gray-700 text-sm">{item}</p>
                    </div>
                  ))}
                </div>
              </div>

              {/* Monitoring */}
              <div className="bg-white rounded-2xl shadow-lg p-6">
                <h3 className="text-xl font-bold text-gray-900 mb-4 flex items-center space-x-2">
                  <Calendar className="w-5 h-5 text-blue-600" />
                  <span>Monitoring Plan</span>
                </h3>
                <div className="space-y-3">
                  {currentPlan.monitoring.map((item, index) => (
                    <div key={index} className="flex items-start space-x-3">
                      <div className="bg-blue-100 rounded-full p-1 mt-1">
                        <div className="w-2 h-2 bg-blue-600 rounded-full"></div>
                      </div>
                      <p className="text-gray-700 text-sm">{item}</p>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Side Effects & Drug Interactions */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Side Effects */}
              <div className="bg-white rounded-2xl shadow-lg p-6">
                <h3 className="text-xl font-bold text-gray-900 mb-4 flex items-center space-x-2">
                  <AlertCircle className="w-5 h-5 text-yellow-600" />
                  <span>Potential Side Effects</span>
                </h3>
                <div className="space-y-4">
                  <div>
                    <h4 className="font-medium text-gray-900 mb-2">Common (&gt;10%)</h4>
                    <div className="flex flex-wrap gap-2">
                      {sideEffects.common.map((effect, index) => (
                        <span key={index} className="bg-yellow-100 text-yellow-800 text-xs px-2 py-1 rounded-full">
                          {effect}
                        </span>
                      ))}
                    </div>
                  </div>
                  <div>
                    <h4 className="font-medium text-gray-900 mb-2">Serious (Rare)</h4>
                    <div className="flex flex-wrap gap-2">
                      {sideEffects.serious.map((effect, index) => (
                        <span key={index} className="bg-red-100 text-red-800 text-xs px-2 py-1 rounded-full">
                          {effect}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>
              </div>

              {/* Drug Interactions */}
              <div className="bg-white rounded-2xl shadow-lg p-6">
                <h3 className="text-xl font-bold text-gray-900 mb-4 flex items-center space-x-2">
                  <Shield className="w-5 h-5 text-purple-600" />
                  <span>Drug Interactions</span>
                </h3>
                <div className="space-y-3">
                  {currentPlan.drugInteractions.map((interaction, index) => (
                    <div key={index} className="p-3 bg-purple-50 rounded-lg border border-purple-200">
                      <p className="text-sm text-purple-800">{interaction}</p>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Emergency Contact */}
            <div className="bg-red-50 border border-red-200 rounded-2xl p-6">
              <h3 className="text-lg font-bold text-red-900 mb-4 flex items-center space-x-2">
                <AlertCircle className="w-5 h-5" />
                <span>When to Contact Healthcare Provider</span>
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {sideEffects.whenToCall.map((symptom, index) => (
                  <div key={index} className="flex items-start space-x-2">
                    <AlertCircle className="w-4 h-4 text-red-600 mt-0.5" />
                    <p className="text-sm text-red-800">{symptom}</p>
                  </div>
                ))}
              </div>
            </div>
          </motion.div>
        </div>

        {/* Disclaimer */}
        <div className="mt-8 p-4 bg-gray-50 rounded-xl">
          <p className="text-sm text-gray-600 text-center">
            <strong>Medical Disclaimer:</strong> AI-generated treatment plans are for informational purposes only. 
            Always consult with qualified healthcare professionals before starting, stopping, or changing any medication regimen.
          </p>
        </div>
      </div>
    </div>
  );
};