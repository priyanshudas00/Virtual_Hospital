import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { useForm } from 'react-hook-form';
import { yupResolver } from '@hookform/resolvers/yup';
import * as yup from 'yup';
import { useAuth } from '../contexts/AuthContext';
import {
  User,
  MapPin,
  Heart,
  Pill,
  AlertTriangle,
  Activity,
  Coffee,
  Moon,
  DollarSign,
  FileText,
  Brain,
  Save,
  Send,
  CheckCircle
} from 'lucide-react';

const intakeSchema = yup.object({
  // Basic Information
  age: yup.number().required('Age is required').min(1).max(120),
  gender: yup.string().required('Gender is required'),
  location: yup.string().required('Location is required'),
  occupation: yup.string().required('Occupation is required'),
  
  // Symptoms
  symptoms: yup.array().min(1, 'Please describe at least one symptom'),
  symptomDuration: yup.string().required('Symptom duration is required'),
  painLevel: yup.number().min(0).max(10),
  
  // Medical History
  medicalHistory: yup.array(),
  currentMedications: yup.array(),
  allergies: yup.array(),
  surgeries: yup.array(),
  familyHistory: yup.array(),
  
  // Lifestyle
  sleepHours: yup.number().min(0).max(24),
  exercise: yup.string().required('Exercise information is required'),
  diet: yup.string().required('Diet information is required'),
  smoking: yup.string().required('Smoking status is required'),
  alcohol: yup.string().required('Alcohol consumption is required'),
  stressLevel: yup.number().min(1).max(10),
  
  // Insurance & Financial
  insuranceProvider: yup.string(),
  policyNumber: yup.string(),
  financialCapability: yup.string().required('Financial capability is required'),
});

export const IntakeFormPage: React.FC = () => {
  const { user } = useAuth();
  const [currentStep, setCurrentStep] = useState(1);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [aiAssessment, setAiAssessment] = useState<any>(null);
  const [formData, setFormData] = useState<any>({});

  const form = useForm({
    resolver: yupResolver(intakeSchema),
    defaultValues: {
      symptoms: [],
      medicalHistory: [],
      currentMedications: [],
      allergies: [],
      surgeries: [],
      familyHistory: [],
      painLevel: 0,
      stressLevel: 5,
      sleepHours: 8,
    },
  });

  const steps = [
    { id: 1, title: 'Basic Information', icon: User },
    { id: 2, title: 'Current Symptoms', icon: Heart },
    { id: 3, title: 'Medical History', icon: FileText },
    { id: 4, title: 'Lifestyle Factors', icon: Activity },
    { id: 5, title: 'Insurance & Financial', icon: DollarSign },
    { id: 6, title: 'Review & Submit', icon: CheckCircle }
  ];

  const commonSymptoms = [
    'Fever', 'Headache', 'Cough', 'Fatigue', 'Nausea', 'Vomiting',
    'Diarrhea', 'Chest Pain', 'Shortness of Breath', 'Dizziness',
    'Sore Throat', 'Runny Nose', 'Body Aches', 'Joint Pain',
    'Abdominal Pain', 'Back Pain', 'Skin Rash', 'Anxiety',
    'Depression', 'Sleep Problems', 'Weight Loss', 'Weight Gain'
  ];

  const commonConditions = [
    'Diabetes', 'Hypertension', 'Heart Disease', 'Asthma', 'Arthritis',
    'Depression', 'Anxiety', 'Thyroid Disorders', 'Kidney Disease',
    'Liver Disease', 'Cancer', 'Stroke', 'COPD', 'Osteoporosis'
  ];

  const commonMedications = [
    'Aspirin', 'Ibuprofen', 'Acetaminophen', 'Metformin', 'Lisinopril',
    'Atorvastatin', 'Omeprazole', 'Levothyroxine', 'Amlodipine',
    'Metoprolol', 'Losartan', 'Simvastatin', 'Prednisone', 'Insulin'
  ];

  const handleNext = () => {
    if (currentStep < steps.length) {
      setCurrentStep(currentStep + 1);
    }
  };

  const handlePrevious = () => {
    if (currentStep > 1) {
      setCurrentStep(currentStep - 1);
    }
  };

  const handleSubmit = async (data: any) => {
    setIsSubmitting(true);
    
    try {
      const response = await fetch(`${import.meta.env.VITE_API_URL}/api/intake-form`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        },
        body: JSON.stringify(data)
      });

      const result = await response.json();

      if (response.ok) {
        setAiAssessment(result.assessment);
        setFormData(data);
        setCurrentStep(7); // Show results
      } else {
        throw new Error(result.error || 'Failed to submit form');
      }
    } catch (error) {
      console.error('Form submission error:', error);
      alert('Failed to submit form. Please try again.');
    } finally {
      setIsSubmitting(false);
    }
  };

  const addToArray = (fieldName: string, value: string) => {
    const currentValues = form.getValues(fieldName) || [];
    if (!currentValues.includes(value)) {
      form.setValue(fieldName, [...currentValues, value]);
    }
  };

  const removeFromArray = (fieldName: string, value: string) => {
    const currentValues = form.getValues(fieldName) || [];
    form.setValue(fieldName, currentValues.filter((item: string) => item !== value));
  };

  if (!user) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <AlertTriangle className="w-16 h-16 text-red-500 mx-auto mb-4" />
          <h2 className="text-2xl font-bold text-gray-900 mb-2">Authentication Required</h2>
          <p className="text-gray-600">Please sign in to access the intake form.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen py-8 px-4 sm:px-6 lg:px-8 bg-gray-50">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-8"
        >
          <h1 className="text-3xl font-bold text-gray-900 mb-2">Medical Intake Form</h1>
          <p className="text-gray-600">Complete assessment for AI-powered medical analysis</p>
        </motion.div>

        {/* Progress Steps */}
        <div className="mb-8">
          <div className="flex items-center justify-between">
            {steps.map((step, index) => {
              const Icon = step.icon;
              return (
                <div key={step.id} className="flex items-center">
                  <div className={`flex items-center justify-center w-10 h-10 rounded-full ${
                    currentStep >= step.id 
                      ? 'bg-blue-600 text-white' 
                      : 'bg-gray-200 text-gray-600'
                  }`}>
                    {currentStep > step.id ? (
                      <CheckCircle className="w-5 h-5" />
                    ) : (
                      <Icon className="w-5 h-5" />
                    )}
                  </div>
                  {index < steps.length - 1 && (
                    <div className={`w-full h-1 mx-2 ${
                      currentStep > step.id ? 'bg-blue-600' : 'bg-gray-200'
                    }`} />
                  )}
                </div>
              );
            })}
          </div>
          <div className="flex justify-between mt-2">
            {steps.map((step) => (
              <div key={step.id} className="text-xs text-gray-600 text-center">
                {step.title}
              </div>
            ))}
          </div>
        </div>

        <form onSubmit={form.handleSubmit(handleSubmit)}>
          {/* Step 1: Basic Information */}
          {currentStep === 1 && (
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              className="bg-white rounded-2xl shadow-lg p-8"
            >
              <h2 className="text-2xl font-bold text-gray-900 mb-6">Basic Information</h2>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Age</label>
                  <input
                    type="number"
                    {...form.register('age')}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                    placeholder="Enter your age"
                  />
                  {form.formState.errors.age && (
                    <p className="text-red-600 text-sm mt-1">{form.formState.errors.age.message}</p>
                  )}
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Gender</label>
                  <select
                    {...form.register('gender')}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  >
                    <option value="">Select gender</option>
                    <option value="male">Male</option>
                    <option value="female">Female</option>
                    <option value="other">Other</option>
                    <option value="prefer_not_to_say">Prefer not to say</option>
                  </select>
                  {form.formState.errors.gender && (
                    <p className="text-red-600 text-sm mt-1">{form.formState.errors.gender.message}</p>
                  )}
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Location</label>
                  <input
                    type="text"
                    {...form.register('location')}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                    placeholder="City, State, Country"
                  />
                  {form.formState.errors.location && (
                    <p className="text-red-600 text-sm mt-1">{form.formState.errors.location.message}</p>
                  )}
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Occupation</label>
                  <input
                    type="text"
                    {...form.register('occupation')}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                    placeholder="Your occupation"
                  />
                  {form.formState.errors.occupation && (
                    <p className="text-red-600 text-sm mt-1">{form.formState.errors.occupation.message}</p>
                  )}
                </div>
              </div>

              <div className="flex justify-end mt-8">
                <button
                  type="button"
                  onClick={handleNext}
                  className="bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700 transition-all duration-200"
                >
                  Next
                </button>
              </div>
            </motion.div>
          )}

          {/* Step 2: Current Symptoms */}
          {currentStep === 2 && (
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              className="bg-white rounded-2xl shadow-lg p-8"
            >
              <h2 className="text-2xl font-bold text-gray-900 mb-6">Current Symptoms</h2>
              
              <div className="space-y-6">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-3">
                    Select your current symptoms:
                  </label>
                  <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                    {commonSymptoms.map((symptom) => (
                      <label key={symptom} className="flex items-center p-3 border rounded-lg cursor-pointer hover:bg-blue-50 transition-all duration-200">
                        <input
                          type="checkbox"
                          onChange={(e) => {
                            if (e.target.checked) {
                              addToArray('symptoms', symptom);
                            } else {
                              removeFromArray('symptoms', symptom);
                            }
                          }}
                          className="mr-2"
                        />
                        <span className="text-sm">{symptom}</span>
                      </label>
                    ))}
                  </div>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Symptom Duration
                  </label>
                  <select
                    {...form.register('symptomDuration')}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  >
                    <option value="">Select duration</option>
                    <option value="less_than_24_hours">Less than 24 hours</option>
                    <option value="1_3_days">1-3 days</option>
                    <option value="4_7_days">4-7 days</option>
                    <option value="1_2_weeks">1-2 weeks</option>
                    <option value="2_4_weeks">2-4 weeks</option>
                    <option value="more_than_month">More than a month</option>
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Pain Level (0-10)
                  </label>
                  <input
                    type="range"
                    min="0"
                    max="10"
                    {...form.register('painLevel')}
                    className="w-full"
                  />
                  <div className="flex justify-between text-xs text-gray-500 mt-1">
                    <span>No Pain</span>
                    <span>Severe Pain</span>
                  </div>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Additional Symptom Details
                  </label>
                  <textarea
                    {...form.register('symptomDetails')}
                    rows={4}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                    placeholder="Describe your symptoms in detail..."
                  />
                </div>
              </div>

              <div className="flex justify-between mt-8">
                <button
                  type="button"
                  onClick={handlePrevious}
                  className="bg-gray-200 text-gray-800 px-6 py-2 rounded-lg hover:bg-gray-300 transition-all duration-200"
                >
                  Previous
                </button>
                <button
                  type="button"
                  onClick={handleNext}
                  className="bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700 transition-all duration-200"
                >
                  Next
                </button>
              </div>
            </motion.div>
          )}

          {/* Step 3: Medical History */}
          {currentStep === 3 && (
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              className="bg-white rounded-2xl shadow-lg p-8"
            >
              <h2 className="text-2xl font-bold text-gray-900 mb-6">Medical History</h2>
              
              <div className="space-y-6">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-3">
                    Past Medical Conditions:
                  </label>
                  <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                    {commonConditions.map((condition) => (
                      <label key={condition} className="flex items-center p-3 border rounded-lg cursor-pointer hover:bg-blue-50 transition-all duration-200">
                        <input
                          type="checkbox"
                          onChange={(e) => {
                            if (e.target.checked) {
                              addToArray('medicalHistory', condition);
                            } else {
                              removeFromArray('medicalHistory', condition);
                            }
                          }}
                          className="mr-2"
                        />
                        <span className="text-sm">{condition}</span>
                      </label>
                    ))}
                  </div>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-3">
                    Current Medications:
                  </label>
                  <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                    {commonMedications.map((medication) => (
                      <label key={medication} className="flex items-center p-3 border rounded-lg cursor-pointer hover:bg-blue-50 transition-all duration-200">
                        <input
                          type="checkbox"
                          onChange={(e) => {
                            if (e.target.checked) {
                              addToArray('currentMedications', medication);
                            } else {
                              removeFromArray('currentMedications', medication);
                            }
                          }}
                          className="mr-2"
                        />
                        <span className="text-sm">{medication}</span>
                      </label>
                    ))}
                  </div>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Allergies
                  </label>
                  <textarea
                    {...form.register('allergiesText')}
                    rows={3}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                    placeholder="List any known allergies (medications, foods, environmental)..."
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Previous Surgeries
                  </label>
                  <textarea
                    {...form.register('surgeriesText')}
                    rows={3}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                    placeholder="List any previous surgeries with dates..."
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Family Medical History
                  </label>
                  <textarea
                    {...form.register('familyHistoryText')}
                    rows={3}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                    placeholder="Family history of medical conditions..."
                  />
                </div>
              </div>

              <div className="flex justify-between mt-8">
                <button
                  type="button"
                  onClick={handlePrevious}
                  className="bg-gray-200 text-gray-800 px-6 py-2 rounded-lg hover:bg-gray-300 transition-all duration-200"
                >
                  Previous
                </button>
                <button
                  type="button"
                  onClick={handleNext}
                  className="bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700 transition-all duration-200"
                >
                  Next
                </button>
              </div>
            </motion.div>
          )}

          {/* Step 4: Lifestyle Factors */}
          {currentStep === 4 && (
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              className="bg-white rounded-2xl shadow-lg p-8"
            >
              <h2 className="text-2xl font-bold text-gray-900 mb-6">Lifestyle Factors</h2>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Sleep Hours per Night
                  </label>
                  <input
                    type="number"
                    min="0"
                    max="24"
                    {...form.register('sleepHours')}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Exercise Frequency
                  </label>
                  <select
                    {...form.register('exercise')}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  >
                    <option value="">Select frequency</option>
                    <option value="none">No exercise</option>
                    <option value="rarely">Rarely (less than once a week)</option>
                    <option value="sometimes">Sometimes (1-2 times a week)</option>
                    <option value="regularly">Regularly (3-4 times a week)</option>
                    <option value="daily">Daily</option>
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Diet Type
                  </label>
                  <select
                    {...form.register('diet')}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  >
                    <option value="">Select diet type</option>
                    <option value="balanced">Balanced diet</option>
                    <option value="vegetarian">Vegetarian</option>
                    <option value="vegan">Vegan</option>
                    <option value="keto">Ketogenic</option>
                    <option value="mediterranean">Mediterranean</option>
                    <option value="fast_food">Mostly fast food</option>
                    <option value="other">Other</option>
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Smoking Status
                  </label>
                  <select
                    {...form.register('smoking')}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  >
                    <option value="">Select status</option>
                    <option value="never">Never smoked</option>
                    <option value="former">Former smoker</option>
                    <option value="current">Current smoker</option>
                    <option value="occasional">Occasional smoker</option>
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Alcohol Consumption
                  </label>
                  <select
                    {...form.register('alcohol')}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  >
                    <option value="">Select frequency</option>
                    <option value="never">Never</option>
                    <option value="rarely">Rarely</option>
                    <option value="socially">Socially</option>
                    <option value="weekly">Weekly</option>
                    <option value="daily">Daily</option>
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Stress Level (1-10)
                  </label>
                  <input
                    type="range"
                    min="1"
                    max="10"
                    {...form.register('stressLevel')}
                    className="w-full"
                  />
                  <div className="flex justify-between text-xs text-gray-500 mt-1">
                    <span>Low Stress</span>
                    <span>High Stress</span>
                  </div>
                </div>
              </div>

              <div className="flex justify-between mt-8">
                <button
                  type="button"
                  onClick={handlePrevious}
                  className="bg-gray-200 text-gray-800 px-6 py-2 rounded-lg hover:bg-gray-300 transition-all duration-200"
                >
                  Previous
                </button>
                <button
                  type="button"
                  onClick={handleNext}
                  className="bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700 transition-all duration-200"
                >
                  Next
                </button>
              </div>
            </motion.div>
          )}

          {/* Step 5: Insurance & Financial */}
          {currentStep === 5 && (
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              className="bg-white rounded-2xl shadow-lg p-8"
            >
              <h2 className="text-2xl font-bold text-gray-900 mb-6">Insurance & Financial Information</h2>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Insurance Provider
                  </label>
                  <input
                    type="text"
                    {...form.register('insuranceProvider')}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                    placeholder="Insurance company name"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Policy Number
                  </label>
                  <input
                    type="text"
                    {...form.register('policyNumber')}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                    placeholder="Policy number"
                  />
                </div>

                <div className="md:col-span-2">
                  <label className="block text-sm font-medium text-gray-700 mb-3">
                    Financial Capability for Healthcare
                  </label>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                    <label className="flex items-center p-4 border rounded-xl cursor-pointer hover:bg-green-50 transition-all duration-200">
                      <input
                        type="radio"
                        value="low"
                        {...form.register('financialCapability')}
                        className="mr-3"
                      />
                      <div>
                        <div className="font-medium text-green-700">Budget-Friendly</div>
                        <div className="text-xs text-gray-500">Government hospitals, basic care</div>
                      </div>
                    </label>
                    <label className="flex items-center p-4 border rounded-xl cursor-pointer hover:bg-blue-50 transition-all duration-200">
                      <input
                        type="radio"
                        value="medium"
                        {...form.register('financialCapability')}
                        className="mr-3"
                      />
                      <div>
                        <div className="font-medium text-blue-700">Moderate</div>
                        <div className="text-xs text-gray-500">Private clinics, standard care</div>
                      </div>
                    </label>
                    <label className="flex items-center p-4 border rounded-xl cursor-pointer hover:bg-purple-50 transition-all duration-200">
                      <input
                        type="radio"
                        value="high"
                        {...form.register('financialCapability')}
                        className="mr-3"
                      />
                      <div>
                        <div className="font-medium text-purple-700">Premium</div>
                        <div className="text-xs text-gray-500">Top hospitals, specialized care</div>
                      </div>
                    </label>
                  </div>
                </div>
              </div>

              <div className="flex justify-between mt-8">
                <button
                  type="button"
                  onClick={handlePrevious}
                  className="bg-gray-200 text-gray-800 px-6 py-2 rounded-lg hover:bg-gray-300 transition-all duration-200"
                >
                  Previous
                </button>
                <button
                  type="button"
                  onClick={handleNext}
                  className="bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700 transition-all duration-200"
                >
                  Review
                </button>
              </div>
            </motion.div>
          )}

          {/* Step 6: Review & Submit */}
          {currentStep === 6 && (
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              className="bg-white rounded-2xl shadow-lg p-8"
            >
              <h2 className="text-2xl font-bold text-gray-900 mb-6">Review & Submit</h2>
              
              <div className="space-y-6">
                <div className="bg-gray-50 rounded-xl p-6">
                  <h3 className="font-bold text-gray-900 mb-3">Form Summary</h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
                    <div>
                      <span className="font-medium">Age:</span> {form.watch('age')}
                    </div>
                    <div>
                      <span className="font-medium">Gender:</span> {form.watch('gender')}
                    </div>
                    <div>
                      <span className="font-medium">Location:</span> {form.watch('location')}
                    </div>
                    <div>
                      <span className="font-medium">Symptoms:</span> {form.watch('symptoms')?.length || 0} selected
                    </div>
                  </div>
                </div>

                <div className="bg-blue-50 border border-blue-200 rounded-xl p-6">
                  <div className="flex items-start space-x-3">
                    <Brain className="w-6 h-6 text-blue-600 mt-1" />
                    <div>
                      <h4 className="font-bold text-blue-900 mb-2">AI Medical Analysis</h4>
                      <p className="text-blue-800 text-sm">
                        Our advanced AI will analyze your information and provide:
                      </p>
                      <ul className="text-blue-700 text-sm mt-2 space-y-1">
                        <li>• Preliminary health assessment</li>
                        <li>• Possible conditions and risk factors</li>
                        <li>• Recommended tests and investigations</li>
                        <li>• Lifestyle and dietary suggestions</li>
                        <li>• Medication guidance (general)</li>
                        <li>• Specialist referral recommendations</li>
                      </ul>
                    </div>
                  </div>
                </div>

                <div className="bg-yellow-50 border border-yellow-200 rounded-xl p-6">
                  <div className="flex items-start space-x-3">
                    <AlertTriangle className="w-6 h-6 text-yellow-600 mt-1" />
                    <div>
                      <h4 className="font-bold text-yellow-900 mb-2">Important Disclaimer</h4>
                      <p className="text-yellow-800 text-sm">
                        This AI assessment is for informational purposes only and does not replace 
                        professional medical advice, diagnosis, or treatment. Always consult with 
                        qualified healthcare professionals for medical decisions.
                      </p>
                    </div>
                  </div>
                </div>
              </div>

              <div className="flex justify-between mt-8">
                <button
                  type="button"
                  onClick={handlePrevious}
                  className="bg-gray-200 text-gray-800 px-6 py-2 rounded-lg hover:bg-gray-300 transition-all duration-200"
                >
                  Previous
                </button>
                <button
                  type="submit"
                  disabled={isSubmitting}
                  className="bg-gradient-to-r from-blue-600 to-green-600 text-white px-8 py-3 rounded-lg font-semibold disabled:opacity-50 disabled:cursor-not-allowed hover:shadow-lg transition-all duration-200 flex items-center space-x-2"
                >
                  {isSubmitting ? (
                    <>
                      <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
                      <span>Analyzing...</span>
                    </>
                  ) : (
                    <>
                      <Send className="w-5 h-5" />
                      <span>Submit for AI Analysis</span>
                    </>
                  )}
                </button>
              </div>
            </motion.div>
          )}

          {/* Step 7: AI Assessment Results */}
          {currentStep === 7 && aiAssessment && (
            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              className="bg-white rounded-2xl shadow-lg p-8"
            >
              <div className="text-center mb-8">
                <div className="bg-green-100 w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4">
                  <CheckCircle className="w-8 h-8 text-green-600" />
                </div>
                <h2 className="text-2xl font-bold text-gray-900 mb-2">AI Medical Assessment Complete</h2>
                <p className="text-gray-600">Your comprehensive health analysis is ready</p>
              </div>

              <div className="bg-gray-50 rounded-xl p-6 mb-6">
                <h3 className="font-bold text-gray-900 mb-4">AI Assessment Report</h3>
                <div className="prose max-w-none">
                  <pre className="whitespace-pre-wrap text-sm text-gray-700 font-sans">
                    {aiAssessment}
                  </pre>
                </div>
              </div>

              <div className="flex space-x-4">
                <button
                  onClick={() => window.location.href = '/dashboard'}
                  className="flex-1 bg-blue-600 text-white py-3 px-4 rounded-xl font-semibold hover:bg-blue-700 transition-all duration-200"
                >
                  Go to Dashboard
                </button>
                <button
                  onClick={() => window.location.href = '/find-healthcare'}
                  className="flex-1 bg-green-600 text-white py-3 px-4 rounded-xl font-semibold hover:bg-green-700 transition-all duration-200"
                >
                  Find Healthcare Providers
                </button>
              </div>
            </motion.div>
          )}
        </form>
      </div>
    </div>
  );
};