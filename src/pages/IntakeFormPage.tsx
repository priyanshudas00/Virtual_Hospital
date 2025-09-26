import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useForm } from 'react-hook-form';
import { yupResolver } from '@hookform/resolvers/yup';
import * as yup from 'yup';
import { useAuth } from '../contexts/AuthContext';
import { DiagnosisModal } from '../components/DiagnosisModal';
import {
  User,
  Heart,
  Activity,
  Pill,
  Home,
  DollarSign,
  Brain,
  CheckCircle,
  AlertTriangle,
  ArrowRight,
  ArrowLeft,
  Save,
  Stethoscope,
  Clock,
  MapPin,
  Phone
} from 'lucide-react';

// Comprehensive validation schema
const intakeSchema = yup.object({
  // Basic Information
  age: yup.number().min(1).max(120).required('Age is required'),
  gender: yup.string().required('Gender is required'),
  location: yup.string().required('Location is required'),
  occupation: yup.string().required('Occupation is required'),
  marital_status: yup.string().required('Marital status is required'),
  
  // Primary Symptoms
  primary_symptoms: yup.array().min(1, 'Please select at least one symptom'),
  symptom_duration: yup.string().required('Symptom duration is required'),
  pain_level: yup.number().min(0).max(10),
  symptom_details: yup.string().min(10, 'Please provide more details about your symptoms'),
  
  // Medical History
  chronic_conditions: yup.array(),
  current_medications: yup.array(),
  allergies_text: yup.string(),
  surgeries_text: yup.string(),
  family_history_text: yup.string(),
  
  // Lifestyle
  sleep_hours: yup.number().min(1).max(24),
  exercise: yup.string().required('Exercise frequency is required'),
  diet: yup.string().required('Diet type is required'),
  smoking: yup.string().required('Smoking status is required'),
  alcohol: yup.string().required('Alcohol consumption is required'),
  stress_level: yup.number().min(1).max(10),
  
  // Insurance
  insurance_provider: yup.string(),
  financial_capability: yup.string().required('Financial capability is required')
});

export const IntakeFormPage: React.FC = () => {
  const { user } = useAuth();
  const [currentStep, setCurrentStep] = useState(1);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [diagnosisData, setDiagnosisData] = useState<any>(null);
  const [showDiagnosisModal, setShowDiagnosisModal] = useState(false);
  const [formProgress, setFormProgress] = useState(0);

  const form = useForm({
    resolver: yupResolver(intakeSchema),
    mode: 'onChange',
    defaultValues: {
      pain_level: 5,
      stress_level: 5,
      sleep_hours: 8,
      primary_symptoms: [],
      chronic_conditions: [],
      current_medications: []
    }
  });

  const totalSteps = 7;

  useEffect(() => {
    setFormProgress((currentStep / totalSteps) * 100);
  }, [currentStep]);

  // Auto-save functionality
  useEffect(() => {
    const subscription = form.watch((data) => {
      localStorage.setItem('intake_form_draft', JSON.stringify(data));
    });
    return () => subscription.unsubscribe();
  }, [form]);

  // Load saved draft on mount
  useEffect(() => {
    const savedDraft = localStorage.getItem('intake_form_draft');
    if (savedDraft) {
      try {
        const draftData = JSON.parse(savedDraft);
        Object.keys(draftData).forEach(key => {
          if (draftData[key] !== undefined) {
            form.setValue(key as any, draftData[key]);
          }
        });
      } catch (error) {
        console.error('Failed to load draft:', error);
      }
    }
  }, [form]);

  const symptomCategories = {
    'Constitutional': ['Fever', 'Fatigue', 'Weight Loss', 'Weight Gain', 'Night Sweats', 'Chills'],
    'Neurological': ['Headache', 'Dizziness', 'Confusion', 'Memory Loss', 'Seizures', 'Numbness', 'Weakness'],
    'Cardiovascular': ['Chest Pain', 'Palpitations', 'Shortness of Breath', 'Leg Swelling', 'Fainting'],
    'Respiratory': ['Cough', 'Shortness of Breath', 'Wheezing', 'Sputum Production', 'Chest Tightness'],
    'Gastrointestinal': ['Nausea', 'Vomiting', 'Diarrhea', 'Constipation', 'Abdominal Pain', 'Loss of Appetite'],
    'Musculoskeletal': ['Joint Pain', 'Muscle Pain', 'Back Pain', 'Stiffness', 'Swelling'],
    'Dermatological': ['Rash', 'Itching', 'Skin Lesions', 'Bruising', 'Hair Loss'],
    'Genitourinary': ['Urinary Frequency', 'Burning Urination', 'Blood in Urine', 'Pelvic Pain'],
    'Psychiatric': ['Anxiety', 'Depression', 'Insomnia', 'Mood Changes', 'Panic Attacks']
  };

  const chronicConditions = [
    'Diabetes', 'Hypertension', 'Heart Disease', 'Asthma', 'COPD', 'Arthritis',
    'Depression', 'Anxiety', 'Thyroid Disease', 'Kidney Disease', 'Liver Disease',
    'Cancer', 'Stroke', 'Epilepsy', 'Migraine', 'Osteoporosis'
  ];

  const handleNext = async () => {
    const fieldsToValidate = getFieldsForStep(currentStep);
    const isValid = await form.trigger(fieldsToValidate);
    
    if (isValid) {
      setCurrentStep(prev => Math.min(prev + 1, totalSteps));
    }
  };

  const handlePrevious = () => {
    setCurrentStep(prev => Math.max(prev - 1, 1));
  };

  const getFieldsForStep = (step: number): string[] => {
    switch (step) {
      case 1: return ['age', 'gender', 'location', 'occupation', 'marital_status'];
      case 2: return ['primary_symptoms', 'symptom_duration', 'pain_level', 'symptom_details'];
      case 3: return ['chronic_conditions', 'current_medications', 'allergies_text'];
      case 4: return ['surgeries_text', 'family_history_text'];
      case 5: return ['sleep_hours', 'exercise', 'diet', 'smoking', 'alcohol', 'stress_level'];
      case 6: return ['insurance_provider', 'financial_capability'];
      case 7: return [];
      default: return [];
    }
  };

  const handleSubmit = async (data: any) => {
    setIsSubmitting(true);
    
    try {
      // Prepare comprehensive data for AI analysis
      const comprehensiveData = {
        ...data,
        user_id: user?.id,
        submission_timestamp: new Date().toISOString(),
        form_version: '2.0',
        completion_time: Date.now() - (parseInt(localStorage.getItem('form_start_time') || '0')),
        
        // Enhanced data structure
        basic_info: {
          age: data.age,
          gender: data.gender,
          location: data.location,
          occupation: data.occupation,
          marital_status: data.marital_status,
          education_level: data.education_level
        },
        
        symptoms: {
          primary_symptoms: data.primary_symptoms,
          symptom_duration: data.symptom_duration,
          pain_level: data.pain_level,
          symptom_details: data.symptom_details,
          onset_pattern: data.onset_pattern,
          aggravating_factors: data.aggravating_factors || [],
          relieving_factors: data.relieving_factors || []
        },
        
        medical_history: {
          chronic_conditions: data.chronic_conditions,
          current_medications: data.current_medications,
          allergies: data.allergies_text,
          surgeries: data.surgeries_text,
          family_history: data.family_history_text,
          immunizations: data.immunizations || [],
          hospitalizations: data.hospitalizations || []
        },
        
        lifestyle_factors: {
          sleep_hours: data.sleep_hours,
          sleep_quality: data.sleep_quality,
          exercise: data.exercise,
          exercise_type: data.exercise_type,
          diet: data.diet,
          smoking: data.smoking,
          alcohol: data.alcohol,
          stress_level: data.stress_level,
          work_environment: data.work_environment
        },
        
        social_determinants: {
          living_situation: data.living_situation,
          support_system: data.support_system,
          transportation: data.transportation,
          food_security: data.food_security,
          housing_stability: data.housing_stability
        },
        
        insurance_financial: {
          insurance_provider: data.insurance_provider,
          financial_capability: data.financial_capability,
          cost_preference: data.cost_preference
        }
      };

      // Submit to backend for AI analysis
      const response = await fetch(`${import.meta.env.VITE_API_URL}/api/intake-form`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        },
        body: JSON.stringify(comprehensiveData)
      });

      const result = await response.json();

      if (response.ok) {
        // Clear draft
        localStorage.removeItem('intake_form_draft');
        
        // Show diagnosis modal with AI results
        setDiagnosisData(result.assessment_data);
        setShowDiagnosisModal(true);
      } else {
        throw new Error(result.error || 'Submission failed');
      }
    } catch (error) {
      console.error('Form submission failed:', error);
      alert('Failed to submit form. Please try again.');
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleSaveToDashboard = async () => {
    // The report is already saved by the backend
    // Just redirect to diagnosis page
    window.location.href = '/diagnosis';
  };

  const handleBookAppointment = () => {
    window.location.href = '/book-appointment';
  };

  const steps = [
    { number: 1, title: 'Basic Information', icon: User, color: 'bg-blue-500' },
    { number: 2, title: 'Current Symptoms', icon: Stethoscope, color: 'bg-red-500' },
    { number: 3, title: 'Medical History', icon: Heart, color: 'bg-purple-500' },
    { number: 4, title: 'Family & Surgery History', icon: Activity, color: 'bg-green-500' },
    { number: 5, title: 'Lifestyle Factors', icon: Brain, color: 'bg-orange-500' },
    { number: 6, title: 'Insurance & Financial', icon: DollarSign, color: 'bg-indigo-500' },
    { number: 7, title: 'Review & Submit', icon: CheckCircle, color: 'bg-teal-500' }
  ];

  if (!user) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-50">
        <div className="text-center">
          <AlertTriangle className="w-16 h-16 text-red-500 mx-auto mb-4" />
          <h2 className="text-2xl font-bold text-gray-900 mb-2">Authentication Required</h2>
          <p className="text-gray-600">Please sign in to access the medical intake form.</p>
        </div>
      </div>
    );
  }

  return (
    <>
      <div className="min-h-screen py-8 px-4 sm:px-6 lg:px-8 bg-gradient-to-br from-blue-50 via-white to-green-50">
        <div className="max-w-4xl mx-auto">
          {/* Header */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-center mb-8"
          >
            <div className="bg-gradient-to-r from-blue-600 to-purple-600 w-20 h-20 rounded-3xl flex items-center justify-center mx-auto mb-6">
              <Stethoscope className="w-10 h-10 text-white" />
            </div>
            <h1 className="text-4xl font-bold text-gray-900 mb-3">Medical Intake Assessment</h1>
            <p className="text-xl text-gray-600">Comprehensive health evaluation powered by AI</p>
          </motion.div>

          {/* Progress Indicator */}
          <div className="mb-8">
            <div className="bg-white rounded-2xl shadow-lg p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-bold text-gray-900">Assessment Progress</h3>
                <span className="text-sm text-gray-600">{Math.round(formProgress)}% Complete</span>
              </div>
              
              <div className="bg-gray-200 rounded-full h-3 mb-4">
                <div 
                  className="bg-gradient-to-r from-blue-600 to-purple-600 h-3 rounded-full transition-all duration-500"
                  style={{ width: `${formProgress}%` }}
                />
              </div>
              
              <div className="flex items-center justify-between">
                {steps.map((step) => {
                  const Icon = step.icon;
                  const isActive = currentStep === step.number;
                  const isCompleted = currentStep > step.number;
                  
                  return (
                    <div key={step.number} className="flex flex-col items-center">
                      <div className={`w-12 h-12 rounded-full flex items-center justify-center text-white font-bold mb-2 transition-all duration-300 ${
                        isCompleted ? 'bg-green-500' : isActive ? step.color : 'bg-gray-300'
                      }`}>
                        {isCompleted ? <CheckCircle className="w-6 h-6" /> : <Icon className="w-6 h-6" />}
                      </div>
                      <span className={`text-xs text-center font-medium ${
                        isActive ? 'text-gray-900' : 'text-gray-500'
                      }`}>
                        {step.title}
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          {/* Form Steps */}
          <form onSubmit={form.handleSubmit(handleSubmit)}>
            <AnimatePresence mode="wait">
              <motion.div
                key={currentStep}
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -20 }}
                transition={{ duration: 0.3 }}
                className="bg-white rounded-2xl shadow-xl p-8"
              >
                {/* Step 1: Basic Information */}
                {currentStep === 1 && (
                  <div className="space-y-6">
                    <div className="text-center mb-8">
                      <User className="w-16 h-16 text-blue-600 mx-auto mb-4" />
                      <h2 className="text-2xl font-bold text-gray-900">Basic Information</h2>
                      <p className="text-gray-600">Let's start with some basic details about you</p>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">Age *</label>
                        <input
                          type="number"
                          {...form.register('age')}
                          className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                          placeholder="Enter your age"
                        />
                        {form.formState.errors.age && (
                          <p className="text-red-600 text-sm mt-1">{form.formState.errors.age.message}</p>
                        )}
                      </div>

                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">Gender *</label>
                        <select
                          {...form.register('gender')}
                          className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
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
                        <label className="block text-sm font-medium text-gray-700 mb-2">Location *</label>
                        <div className="relative">
                          <MapPin className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
                          <input
                            type="text"
                            {...form.register('location')}
                            className="w-full pl-10 pr-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                            placeholder="City, State/Country"
                          />
                        </div>
                        {form.formState.errors.location && (
                          <p className="text-red-600 text-sm mt-1">{form.formState.errors.location.message}</p>
                        )}
                      </div>

                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">Occupation *</label>
                        <input
                          type="text"
                          {...form.register('occupation')}
                          className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                          placeholder="Your profession"
                        />
                        {form.formState.errors.occupation && (
                          <p className="text-red-600 text-sm mt-1">{form.formState.errors.occupation.message}</p>
                        )}
                      </div>

                      <div className="md:col-span-2">
                        <label className="block text-sm font-medium text-gray-700 mb-2">Marital Status *</label>
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                          {['Single', 'Married', 'Divorced', 'Widowed'].map((status) => (
                            <label key={status} className="flex items-center p-3 border rounded-xl cursor-pointer hover:bg-blue-50 transition-all duration-200">
                              <input
                                type="radio"
                                value={status.toLowerCase()}
                                {...form.register('marital_status')}
                                className="mr-2"
                              />
                              <span className="font-medium">{status}</span>
                            </label>
                          ))}
                        </div>
                        {form.formState.errors.marital_status && (
                          <p className="text-red-600 text-sm mt-1">{form.formState.errors.marital_status.message}</p>
                        )}
                      </div>
                    </div>
                  </div>
                )}

                {/* Step 2: Current Symptoms */}
                {currentStep === 2 && (
                  <div className="space-y-6">
                    <div className="text-center mb-8">
                      <Stethoscope className="w-16 h-16 text-red-600 mx-auto mb-4" />
                      <h2 className="text-2xl font-bold text-gray-900">Current Symptoms</h2>
                      <p className="text-gray-600">Tell us about what you're experiencing</p>
                    </div>

                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-4">
                        Primary Symptoms * (Select all that apply)
                      </label>
                      <div className="space-y-4">
                        {Object.entries(symptomCategories).map(([category, symptoms]) => (
                          <div key={category} className="border border-gray-200 rounded-xl p-4">
                            <h4 className="font-bold text-gray-900 mb-3">{category}</h4>
                            <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
                              {symptoms.map((symptom) => (
                                <label key={symptom} className="flex items-center p-2 hover:bg-gray-50 rounded-lg cursor-pointer">
                                  <input
                                    type="checkbox"
                                    value={symptom}
                                    {...form.register('primary_symptoms')}
                                    className="mr-2 rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                                  />
                                  <span className="text-sm">{symptom}</span>
                                </label>
                              ))}
                            </div>
                          </div>
                        ))}
                      </div>
                      {form.formState.errors.primary_symptoms && (
                        <p className="text-red-600 text-sm mt-1">{form.formState.errors.primary_symptoms.message}</p>
                      )}
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">Symptom Duration *</label>
                        <select
                          {...form.register('symptom_duration')}
                          className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                        >
                          <option value="">Select duration</option>
                          <option value="less_than_1_day">Less than 1 day</option>
                          <option value="1_3_days">1-3 days</option>
                          <option value="4_7_days">4-7 days</option>
                          <option value="1_2_weeks">1-2 weeks</option>
                          <option value="2_4_weeks">2-4 weeks</option>
                          <option value="1_3_months">1-3 months</option>
                          <option value="more_than_3_months">More than 3 months</option>
                        </select>
                        {form.formState.errors.symptom_duration && (
                          <p className="text-red-600 text-sm mt-1">{form.formState.errors.symptom_duration.message}</p>
                        )}
                      </div>

                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">
                          Pain Level (0 = No pain, 10 = Severe pain)
                        </label>
                        <div className="space-y-2">
                          <input
                            type="range"
                            min="0"
                            max="10"
                            {...form.register('pain_level')}
                            className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                          />
                          <div className="flex justify-between text-sm text-gray-600">
                            <span>No Pain (0)</span>
                            <span className="font-bold text-lg text-red-600">
                              {form.watch('pain_level') || 0}
                            </span>
                            <span>Severe (10)</span>
                          </div>
                        </div>
                      </div>
                    </div>

                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        Detailed Symptom Description *
                      </label>
                      <textarea
                        {...form.register('symptom_details')}
                        rows={4}
                        className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                        placeholder="Please describe your symptoms in detail - when they started, how they feel, what makes them better or worse, etc."
                      />
                      {form.formState.errors.symptom_details && (
                        <p className="text-red-600 text-sm mt-1">{form.formState.errors.symptom_details.message}</p>
                      )}
                    </div>
                  </div>
                )}

                {/* Step 3: Medical History */}
                {currentStep === 3 && (
                  <div className="space-y-6">
                    <div className="text-center mb-8">
                      <Heart className="w-16 h-16 text-purple-600 mx-auto mb-4" />
                      <h2 className="text-2xl font-bold text-gray-900">Medical History</h2>
                      <p className="text-gray-600">Your past and current medical information</p>
                    </div>

                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-4">
                        Chronic Conditions (Select all that apply)
                      </label>
                      <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                        {chronicConditions.map((condition) => (
                          <label key={condition} className="flex items-center p-3 border rounded-xl cursor-pointer hover:bg-purple-50 transition-all duration-200">
                            <input
                              type="checkbox"
                              value={condition}
                              {...form.register('chronic_conditions')}
                              className="mr-2 rounded border-gray-300 text-purple-600 focus:ring-purple-500"
                            />
                            <span className="text-sm font-medium">{condition}</span>
                          </label>
                        ))}
                      </div>
                    </div>

                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        Current Medications
                      </label>
                      <div className="space-y-3">
                        <input
                          type="text"
                          placeholder="Add medication name"
                          className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-purple-500 focus:border-purple-500"
                          onKeyPress={(e) => {
                            if (e.key === 'Enter') {
                              e.preventDefault();
                              const value = (e.target as HTMLInputElement).value;
                              if (value.trim()) {
                                const currentMeds = form.getValues('current_medications') || [];
                                form.setValue('current_medications', [...currentMeds, value.trim()]);
                                (e.target as HTMLInputElement).value = '';
                              }
                            }
                          }}
                        />
                        <div className="space-y-2">
                          {(form.watch('current_medications') || []).map((med: string, index: number) => (
                            <div key={index} className="flex items-center justify-between p-3 bg-purple-50 rounded-xl">
                              <span className="text-purple-800">{med}</span>
                              <button
                                type="button"
                                onClick={() => {
                                  const meds = form.getValues('current_medications') || [];
                                  form.setValue('current_medications', meds.filter((_, i) => i !== index));
                                }}
                                className="text-red-600 hover:text-red-800"
                              >
                                Remove
                              </button>
                            </div>
                          ))}
                        </div>
                      </div>
                    </div>

                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        Allergies (medications, foods, environmental)
                      </label>
                      <textarea
                        {...form.register('allergies_text')}
                        rows={3}
                        className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-purple-500 focus:border-purple-500"
                        placeholder="List any known allergies and reactions (e.g., Penicillin - rash, Peanuts - swelling)"
                      />
                    </div>
                  </div>
                )}

                {/* Step 4: Family & Surgery History */}
                {currentStep === 4 && (
                  <div className="space-y-6">
                    <div className="text-center mb-8">
                      <Activity className="w-16 h-16 text-green-600 mx-auto mb-4" />
                      <h2 className="text-2xl font-bold text-gray-900">Family & Surgery History</h2>
                      <p className="text-gray-600">Important family medical history and past surgeries</p>
                    </div>

                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        Previous Surgeries
                      </label>
                      <textarea
                        {...form.register('surgeries_text')}
                        rows={4}
                        className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-green-500 focus:border-green-500"
                        placeholder="List any surgeries with approximate dates (e.g., Appendectomy - 2020, Knee surgery - 2018)"
                      />
                    </div>

                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        Family Medical History
                      </label>
                      <textarea
                        {...form.register('family_history_text')}
                        rows={4}
                        className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-green-500 focus:border-green-500"
                        placeholder="Family history of diabetes, heart disease, cancer, etc. Include relationship (e.g., Father - diabetes, Mother - hypertension)"
                      />
                    </div>
                  </div>
                )}

                {/* Step 5: Lifestyle Factors */}
                {currentStep === 5 && (
                  <div className="space-y-6">
                    <div className="text-center mb-8">
                      <Brain className="w-16 h-16 text-orange-600 mx-auto mb-4" />
                      <h2 className="text-2xl font-bold text-gray-900">Lifestyle Factors</h2>
                      <p className="text-gray-600">Your daily habits and lifestyle choices</p>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">
                          Sleep Hours per Night *
                        </label>
                        <input
                          type="number"
                          min="1"
                          max="24"
                          step="0.5"
                          {...form.register('sleep_hours')}
                          className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-orange-500 focus:border-orange-500"
                        />
                      </div>

                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">Exercise Frequency *</label>
                        <select
                          {...form.register('exercise')}
                          className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-orange-500 focus:border-orange-500"
                        >
                          <option value="">Select frequency</option>
                          <option value="daily">Daily</option>
                          <option value="4-6_times_week">4-6 times per week</option>
                          <option value="2-3_times_week">2-3 times per week</option>
                          <option value="once_week">Once per week</option>
                          <option value="rarely">Rarely</option>
                          <option value="never">Never</option>
                        </select>
                      </div>

                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">Diet Type *</label>
                        <select
                          {...form.register('diet')}
                          className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-orange-500 focus:border-orange-500"
                        >
                          <option value="">Select diet type</option>
                          <option value="balanced">Balanced/Mixed</option>
                          <option value="vegetarian">Vegetarian</option>
                          <option value="vegan">Vegan</option>
                          <option value="keto">Ketogenic</option>
                          <option value="mediterranean">Mediterranean</option>
                          <option value="low_carb">Low Carb</option>
                          <option value="other">Other</option>
                        </select>
                      </div>

                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">Smoking Status *</label>
                        <select
                          {...form.register('smoking')}
                          className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-orange-500 focus:border-orange-500"
                        >
                          <option value="">Select status</option>
                          <option value="never">Never smoked</option>
                          <option value="former">Former smoker</option>
                          <option value="current">Current smoker</option>
                          <option value="occasional">Occasional smoker</option>
                        </select>
                      </div>

                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">Alcohol Consumption *</label>
                        <select
                          {...form.register('alcohol')}
                          className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-orange-500 focus:border-orange-500"
                        >
                          <option value="">Select frequency</option>
                          <option value="never">Never</option>
                          <option value="rarely">Rarely</option>
                          <option value="weekly">Weekly</option>
                          <option value="daily">Daily</option>
                          <option value="heavy">Heavy use</option>
                        </select>
                      </div>

                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">
                          Stress Level (1 = Very low, 10 = Very high) *
                        </label>
                        <div className="space-y-2">
                          <input
                            type="range"
                            min="1"
                            max="10"
                            {...form.register('stress_level')}
                            className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                          />
                          <div className="flex justify-between text-sm text-gray-600">
                            <span>Very Low (1)</span>
                            <span className="font-bold text-lg text-orange-600">
                              {form.watch('stress_level') || 5}
                            </span>
                            <span>Very High (10)</span>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                )}

                {/* Step 6: Insurance & Financial */}
                {currentStep === 6 && (
                  <div className="space-y-6">
                    <div className="text-center mb-8">
                      <DollarSign className="w-16 h-16 text-indigo-600 mx-auto mb-4" />
                      <h2 className="text-2xl font-bold text-gray-900">Insurance & Financial Information</h2>
                      <p className="text-gray-600">Help us find the right care options for your budget</p>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">Insurance Provider</label>
                        <input
                          type="text"
                          {...form.register('insurance_provider')}
                          className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
                          placeholder="e.g., Blue Cross Blue Shield, Aetna"
                        />
                      </div>

                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">Financial Capability *</label>
                        <select
                          {...form.register('financial_capability')}
                          className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
                        >
                          <option value="">Select capability</option>
                          <option value="low">Budget-conscious (Under $100)</option>
                          <option value="medium">Moderate budget ($100-$500)</option>
                          <option value="high">Flexible budget ($500+)</option>
                        </select>
                        {form.formState.errors.financial_capability && (
                          <p className="text-red-600 text-sm mt-1">{form.formState.errors.financial_capability.message}</p>
                        )}
                      </div>
                    </div>
                  </div>
                )}

                {/* Step 7: Review & Submit */}
                {currentStep === 7 && (
                  <div className="space-y-6">
                    <div className="text-center mb-8">
                      <CheckCircle className="w-16 h-16 text-teal-600 mx-auto mb-4" />
                      <h2 className="text-2xl font-bold text-gray-900">Review & Submit</h2>
                      <p className="text-gray-600">Review your information before AI analysis</p>
                    </div>

                    <div className="bg-gradient-to-r from-blue-50 to-purple-50 rounded-2xl p-6 border border-blue-200">
                      <h3 className="text-xl font-bold text-gray-900 mb-4">What happens next?</h3>
                      <div className="space-y-3">
                        <div className="flex items-center space-x-3">
                          <div className="bg-blue-600 text-white rounded-full w-8 h-8 flex items-center justify-center text-sm font-bold">1</div>
                          <span className="text-gray-700">Your data will be analyzed by advanced AI (Gemini)</span>
                        </div>
                        <div className="flex items-center space-x-3">
                          <div className="bg-purple-600 text-white rounded-full w-8 h-8 flex items-center justify-center text-sm font-bold">2</div>
                          <span className="text-gray-700">You'll receive a comprehensive medical assessment</span>
                        </div>
                        <div className="flex items-center space-x-3">
                          <div className="bg-green-600 text-white rounded-full w-8 h-8 flex items-center justify-center text-sm font-bold">3</div>
                          <span className="text-gray-700">Results will be saved to your dashboard for future reference</span>
                        </div>
                      </div>
                    </div>

                    <div className="bg-yellow-50 border border-yellow-200 rounded-xl p-6">
                      <div className="flex items-start space-x-3">
                        <AlertTriangle className="w-6 h-6 text-yellow-600 mt-1" />
                        <div>
                          <h4 className="font-bold text-yellow-900 mb-2">Important Medical Disclaimer</h4>
                          <p className="text-yellow-800 text-sm leading-relaxed">
                            This AI assessment is designed to augment healthcare delivery, not replace it. 
                            The analysis provides preliminary insights to assist healthcare professionals. 
                            Always consult qualified healthcare providers for personal medical advice, 
                            diagnosis, and treatment decisions. In case of emergency, call 911 immediately.
                          </p>
                        </div>
                      </div>
                    </div>

                    <button
                      type="submit"
                      disabled={isSubmitting}
                      className="w-full bg-gradient-to-r from-blue-600 via-purple-600 to-green-600 text-white py-4 px-6 rounded-xl font-bold text-lg disabled:opacity-50 disabled:cursor-not-allowed hover:shadow-lg transition-all duration-200 flex items-center justify-center space-x-3"
                    >
                      {isSubmitting ? (
                        <>
                          <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-white"></div>
                          <span>AI Analyzing Your Health Data...</span>
                          <Brain className="w-6 h-6" />
                        </>
                      ) : (
                        <>
                          <Stethoscope className="w-6 h-6" />
                          <span>Submit for AI Medical Analysis</span>
                          <Brain className="w-6 h-6" />
                        </>
                      )}
                    </button>
                  </div>
                )}
              </motion.div>
            </AnimatePresence>

            {/* Navigation */}
            {currentStep < 7 && (
              <div className="flex justify-between mt-6">
                <button
                  type="button"
                  onClick={handlePrevious}
                  disabled={currentStep === 1}
                  className="flex items-center space-x-2 px-6 py-3 bg-gray-200 text-gray-700 rounded-xl font-semibold disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-300 transition-all duration-200"
                >
                  <ArrowLeft className="w-5 h-5" />
                  <span>Previous</span>
                </button>

                <button
                  type="button"
                  onClick={handleNext}
                  className="flex items-center space-x-2 px-6 py-3 bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-xl font-semibold hover:shadow-lg transition-all duration-200"
                >
                  <span>Next</span>
                  <ArrowRight className="w-5 h-5" />
                </button>
              </div>
            )}
          </form>
        </div>
      </div>

      {/* Diagnosis Modal */}
      <DiagnosisModal
        isOpen={showDiagnosisModal}
        onClose={() => setShowDiagnosisModal(false)}
        diagnosisData={diagnosisData}
        patientData={form.getValues()}
        onSaveToDashboard={handleSaveToDashboard}
        onBookAppointment={handleBookAppointment}
      />
    </>
  );
};