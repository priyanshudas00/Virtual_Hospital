import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useForm } from 'react-hook-form';
import { yupResolver } from '@hookform/resolvers/yup';
import * as yup from 'yup';
import { useAuth } from '../contexts/AuthContext';
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
  Phone,
  X,
  Zap
} from 'lucide-react';

// Enhanced validation schema
const intakeSchema = yup.object({
  // Basic Information
  age: yup.number().min(1).max(120).required('Age is required'),
  gender: yup.string().required('Gender is required'),
  location: yup.string().required('Location is required'),
  occupation: yup.string().required('Occupation is required'),
  
  // Primary Symptoms
  primary_symptoms: yup.array().min(1, 'Please select at least one symptom'),
  symptom_duration: yup.string().required('Symptom duration is required'),
  pain_level: yup.number().min(0).max(10),
  symptom_details: yup.string().min(10, 'Please provide more details about your symptoms'),
  
  // Medical History
  chronic_conditions: yup.array(),
  current_medications: yup.array(),
  allergies_text: yup.string(),
  
  // Lifestyle
  sleep_hours: yup.number().min(1).max(24),
  exercise: yup.string().required('Exercise frequency is required'),
  smoking: yup.string().required('Smoking status is required'),
  alcohol: yup.string().required('Alcohol consumption is required'),
  stress_level: yup.number().min(1).max(10),
  
  // Financial
  financial_capability: yup.string().required('Financial capability is required')
});

interface IntakeFormModalProps {
  isOpen: boolean;
  onClose: () => void;
  onAssessmentComplete: (assessment: any) => void;
}

export const IntakeFormModal: React.FC<IntakeFormModalProps> = ({
  isOpen,
  onClose,
  onAssessmentComplete
}) => {
  const { user, updateProfile } = useAuth();
  const [currentStep, setCurrentStep] = useState(1);
  const [isSubmitting, setIsSubmitting] = useState(false);
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

  const totalSteps = 6;

  useEffect(() => {
    setFormProgress((currentStep / totalSteps) * 100);
  }, [currentStep]);

  const symptomCategories = {
    'Constitutional': ['Fever', 'Fatigue', 'Weight Loss', 'Weight Gain', 'Night Sweats', 'Chills'],
    'Neurological': ['Headache', 'Dizziness', 'Confusion', 'Memory Loss', 'Seizures', 'Numbness', 'Weakness'],
    'Cardiovascular': ['Chest Pain', 'Palpitations', 'Shortness of Breath', 'Leg Swelling', 'Fainting'],
    'Respiratory': ['Cough', 'Shortness of Breath', 'Wheezing', 'Sputum Production', 'Chest Tightness'],
    'Gastrointestinal': ['Nausea', 'Vomiting', 'Diarrhea', 'Constipation', 'Abdominal Pain', 'Loss of Appetite'],
    'Musculoskeletal': ['Joint Pain', 'Muscle Pain', 'Back Pain', 'Stiffness', 'Swelling'],
    'Dermatological': ['Rash', 'Itching', 'Skin Lesions', 'Bruising', 'Hair Loss'],
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
      case 1: return ['age', 'gender', 'location', 'occupation'];
      case 2: return ['primary_symptoms', 'symptom_duration', 'pain_level', 'symptom_details'];
      case 3: return ['chronic_conditions', 'current_medications', 'allergies_text'];
      case 4: return ['sleep_hours', 'exercise', 'smoking', 'alcohol', 'stress_level'];
      case 5: return ['financial_capability'];
      case 6: return [];
      default: return [];
    }
  };

  const handleSubmit = async (data: any) => {
    setIsSubmitting(true);
    
    try {
      // Update user profile with basic info
      await updateProfile({
        firstName: user?.firstName,
        lastName: user?.lastName,
        age: data.age,
        gender: data.gender,
        location: data.location,
        occupation: data.occupation
      });

      // Submit for ML/AI analysis
      const response = await fetch(`${import.meta.env.VITE_API_URL}/api/intake-form`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        },
        body: JSON.stringify({
          ...data,
          user_id: user?.id,
          submission_timestamp: new Date().toISOString()
        })
      });

      const result = await response.json();

      if (response.ok) {
        onAssessmentComplete(result.assessment_data);
        onClose();
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

  const steps = [
    { number: 1, title: 'Basic Info', icon: User, color: 'bg-blue-500' },
    { number: 2, title: 'Symptoms', icon: Stethoscope, color: 'bg-red-500' },
    { number: 3, title: 'Medical History', icon: Heart, color: 'bg-purple-500' },
    { number: 4, title: 'Lifestyle', icon: Activity, color: 'bg-green-500' },
    { number: 5, title: 'Financial', icon: DollarSign, color: 'bg-orange-500' },
    { number: 6, title: 'Review', icon: CheckCircle, color: 'bg-teal-500' }
  ];

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center p-4 z-50">
      <motion.div
        initial={{ opacity: 0, scale: 0.9, y: 20 }}
        animate={{ opacity: 1, scale: 1, y: 0 }}
        exit={{ opacity: 0, scale: 0.9, y: 20 }}
        className="bg-white rounded-3xl shadow-2xl max-w-4xl w-full max-h-[90vh] overflow-hidden"
      >
        {/* Enhanced Header */}
        <div className="bg-gradient-to-r from-blue-600 via-purple-600 to-pink-600 p-6 text-white relative overflow-hidden">
          <div className="absolute inset-0 bg-white/10 backdrop-blur-sm"></div>
          <div className="relative flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <div className="bg-white/20 p-3 rounded-2xl backdrop-blur-sm">
                <Brain className="w-8 h-8" />
              </div>
              <div>
                <h1 className="text-2xl font-bold">AI Medical Assessment</h1>
                <p className="text-blue-100">Comprehensive health evaluation</p>
              </div>
            </div>
            <button
              onClick={onClose}
              className="p-2 bg-white/20 rounded-xl hover:bg-white/30 transition-all duration-200 backdrop-blur-sm"
            >
              <X className="w-6 h-6" />
            </button>
          </div>
          
          {/* Enhanced Progress Bar */}
          <div className="relative mt-6">
            <div className="bg-white/20 rounded-full h-3 backdrop-blur-sm">
              <div 
                className="bg-white h-3 rounded-full transition-all duration-500 shadow-lg"
                style={{ width: `${formProgress}%` }}
              />
            </div>
            <div className="flex justify-between mt-3">
              {steps.map((step) => {
                const Icon = step.icon;
                const isActive = currentStep === step.number;
                const isCompleted = currentStep > step.number;
                
                return (
                  <div key={step.number} className="flex flex-col items-center">
                    <div className={`w-10 h-10 rounded-full flex items-center justify-center text-white font-bold transition-all duration-300 ${
                      isCompleted ? 'bg-white/30 scale-110' : isActive ? 'bg-white/40 scale-125 shadow-lg' : 'bg-white/10'
                    }`}>
                      {isCompleted ? <CheckCircle className="w-5 h-5" /> : <Icon className="w-5 h-5" />}
                    </div>
                    <span className={`text-xs mt-1 font-medium ${isActive ? 'text-white' : 'text-blue-100'}`}>
                      {step.title}
                    </span>
                  </div>
                );
              })}
            </div>
          </div>
        </div>

        {/* Form Content */}
        <div className="p-8 overflow-y-auto max-h-[calc(90vh-200px)]">
          <form onSubmit={form.handleSubmit(handleSubmit)}>
            <AnimatePresence mode="wait">
              <motion.div
                key={currentStep}
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -20 }}
                transition={{ duration: 0.3 }}
              >
                {/* Step 1: Basic Information */}
                {currentStep === 1 && (
                  <div className="space-y-6">
                    <div className="text-center mb-8">
                      <div className="bg-gradient-to-r from-blue-100 to-purple-100 w-20 h-20 rounded-3xl flex items-center justify-center mx-auto mb-4">
                        <User className="w-10 h-10 text-blue-600" />
                      </div>
                      <h2 className="text-3xl font-bold text-gray-900 mb-2">Basic Information</h2>
                      <p className="text-gray-600">Let's start with some essential details</p>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      <div className="space-y-2">
                        <label className="block text-sm font-bold text-gray-700">Age *</label>
                        <input
                          type="number"
                          {...form.register('age')}
                          className="w-full px-4 py-3 border-2 border-gray-200 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all duration-200"
                          placeholder="Enter your age"
                        />
                        {form.formState.errors.age && (
                          <p className="text-red-600 text-sm">{form.formState.errors.age.message}</p>
                        )}
                      </div>

                      <div className="space-y-2">
                        <label className="block text-sm font-bold text-gray-700">Gender *</label>
                        <select
                          {...form.register('gender')}
                          className="w-full px-4 py-3 border-2 border-gray-200 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all duration-200"
                        >
                          <option value="">Select gender</option>
                          <option value="male">Male</option>
                          <option value="female">Female</option>
                          <option value="other">Other</option>
                        </select>
                        {form.formState.errors.gender && (
                          <p className="text-red-600 text-sm">{form.formState.errors.gender.message}</p>
                        )}
                      </div>

                      <div className="space-y-2">
                        <label className="block text-sm font-bold text-gray-700">Location *</label>
                        <div className="relative">
                          <MapPin className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
                          <input
                            type="text"
                            {...form.register('location')}
                            className="w-full pl-10 pr-4 py-3 border-2 border-gray-200 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all duration-200"
                            placeholder="City, State/Country"
                          />
                        </div>
                        {form.formState.errors.location && (
                          <p className="text-red-600 text-sm">{form.formState.errors.location.message}</p>
                        )}
                      </div>

                      <div className="space-y-2">
                        <label className="block text-sm font-bold text-gray-700">Occupation *</label>
                        <input
                          type="text"
                          {...form.register('occupation')}
                          className="w-full px-4 py-3 border-2 border-gray-200 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all duration-200"
                          placeholder="Your profession"
                        />
                        {form.formState.errors.occupation && (
                          <p className="text-red-600 text-sm">{form.formState.errors.occupation.message}</p>
                        )}
                      </div>
                    </div>
                  </div>
                )}

                {/* Step 2: Current Symptoms */}
                {currentStep === 2 && (
                  <div className="space-y-6">
                    <div className="text-center mb-8">
                      <div className="bg-gradient-to-r from-red-100 to-pink-100 w-20 h-20 rounded-3xl flex items-center justify-center mx-auto mb-4">
                        <Stethoscope className="w-10 h-10 text-red-600" />
                      </div>
                      <h2 className="text-3xl font-bold text-gray-900 mb-2">Current Symptoms</h2>
                      <p className="text-gray-600">Tell us about what you're experiencing</p>
                    </div>

                    <div>
                      <label className="block text-sm font-bold text-gray-700 mb-4">
                        Primary Symptoms * (Select all that apply)
                      </label>
                      <div className="space-y-4">
                        {Object.entries(symptomCategories).map(([category, symptoms]) => (
                          <div key={category} className="border-2 border-gray-200 rounded-2xl p-6 hover:border-blue-300 transition-all duration-200">
                            <h4 className="font-bold text-gray-900 mb-4 text-lg">{category}</h4>
                            <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                              {symptoms.map((symptom) => (
                                <label key={symptom} className="flex items-center p-3 hover:bg-blue-50 rounded-xl cursor-pointer transition-all duration-200 border border-transparent hover:border-blue-200">
                                  <input
                                    type="checkbox"
                                    value={symptom}
                                    {...form.register('primary_symptoms')}
                                    className="mr-3 rounded border-gray-300 text-blue-600 focus:ring-blue-500 w-4 h-4"
                                  />
                                  <span className="text-sm font-medium">{symptom}</span>
                                </label>
                              ))}
                            </div>
                          </div>
                        ))}
                      </div>
                      {form.formState.errors.primary_symptoms && (
                        <p className="text-red-600 text-sm mt-2">{form.formState.errors.primary_symptoms.message}</p>
                      )}
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      <div className="space-y-2">
                        <label className="block text-sm font-bold text-gray-700">Symptom Duration *</label>
                        <select
                          {...form.register('symptom_duration')}
                          className="w-full px-4 py-3 border-2 border-gray-200 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all duration-200"
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
                          <p className="text-red-600 text-sm">{form.formState.errors.symptom_duration.message}</p>
                        )}
                      </div>

                      <div className="space-y-2">
                        <label className="block text-sm font-bold text-gray-700">
                          Pain Level (0 = No pain, 10 = Severe pain)
                        </label>
                        <div className="space-y-3">
                          <input
                            type="range"
                            min="0"
                            max="10"
                            {...form.register('pain_level')}
                            className="w-full h-3 bg-gradient-to-r from-green-200 via-yellow-200 to-red-200 rounded-lg appearance-none cursor-pointer"
                          />
                          <div className="flex justify-between text-sm text-gray-600">
                            <span>No Pain (0)</span>
                            <span className="font-bold text-2xl text-red-600 bg-red-50 px-3 py-1 rounded-xl">
                              {form.watch('pain_level') || 0}
                            </span>
                            <span>Severe (10)</span>
                          </div>
                        </div>
                      </div>
                    </div>

                    <div className="space-y-2">
                      <label className="block text-sm font-bold text-gray-700">
                        Detailed Symptom Description *
                      </label>
                      <textarea
                        {...form.register('symptom_details')}
                        rows={4}
                        className="w-full px-4 py-3 border-2 border-gray-200 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all duration-200"
                        placeholder="Please describe your symptoms in detail - when they started, how they feel, what makes them better or worse, etc."
                      />
                      {form.formState.errors.symptom_details && (
                        <p className="text-red-600 text-sm">{form.formState.errors.symptom_details.message}</p>
                      )}
                    </div>
                  </div>
                )}

                {/* Step 3: Medical History */}
                {currentStep === 3 && (
                  <div className="space-y-6">
                    <div className="text-center mb-8">
                      <div className="bg-gradient-to-r from-purple-100 to-pink-100 w-20 h-20 rounded-3xl flex items-center justify-center mx-auto mb-4">
                        <Heart className="w-10 h-10 text-purple-600" />
                      </div>
                      <h2 className="text-3xl font-bold text-gray-900 mb-2">Medical History</h2>
                      <p className="text-gray-600">Your past and current medical information</p>
                    </div>

                    <div>
                      <label className="block text-sm font-bold text-gray-700 mb-4">
                        Chronic Conditions (Select all that apply)
                      </label>
                      <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                        {chronicConditions.map((condition) => (
                          <label key={condition} className="flex items-center p-4 border-2 border-gray-200 rounded-xl cursor-pointer hover:bg-purple-50 hover:border-purple-300 transition-all duration-200">
                            <input
                              type="checkbox"
                              value={condition}
                              {...form.register('chronic_conditions')}
                              className="mr-3 rounded border-gray-300 text-purple-600 focus:ring-purple-500 w-4 h-4"
                            />
                            <span className="text-sm font-medium">{condition}</span>
                          </label>
                        ))}
                      </div>
                    </div>

                    <div className="space-y-2">
                      <label className="block text-sm font-bold text-gray-700">
                        Current Medications
                      </label>
                      <div className="space-y-3">
                        <input
                          type="text"
                          placeholder="Add medication name and press Enter"
                          className="w-full px-4 py-3 border-2 border-gray-200 rounded-xl focus:ring-2 focus:ring-purple-500 focus:border-purple-500 transition-all duration-200"
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
                            <div key={index} className="flex items-center justify-between p-3 bg-purple-50 rounded-xl border border-purple-200">
                              <span className="text-purple-800 font-medium">{med}</span>
                              <button
                                type="button"
                                onClick={() => {
                                  const meds = form.getValues('current_medications') || [];
                                  form.setValue('current_medications', meds.filter((_, i) => i !== index));
                                }}
                                className="text-red-600 hover:text-red-800 font-bold"
                              >
                                ✕
                              </button>
                            </div>
                          ))}
                        </div>
                      </div>
                    </div>

                    <div className="space-y-2">
                      <label className="block text-sm font-bold text-gray-700">
                        Allergies (medications, foods, environmental)
                      </label>
                      <textarea
                        {...form.register('allergies_text')}
                        rows={3}
                        className="w-full px-4 py-3 border-2 border-gray-200 rounded-xl focus:ring-2 focus:ring-purple-500 focus:border-purple-500 transition-all duration-200"
                        placeholder="List any known allergies and reactions (e.g., Penicillin - rash, Peanuts - swelling)"
                      />
                    </div>
                  </div>
                )}

                {/* Step 4: Lifestyle Factors */}
                {currentStep === 4 && (
                  <div className="space-y-6">
                    <div className="text-center mb-8">
                      <div className="bg-gradient-to-r from-green-100 to-teal-100 w-20 h-20 rounded-3xl flex items-center justify-center mx-auto mb-4">
                        <Activity className="w-10 h-10 text-green-600" />
                      </div>
                      <h2 className="text-3xl font-bold text-gray-900 mb-2">Lifestyle Factors</h2>
                      <p className="text-gray-600">Your daily habits and lifestyle choices</p>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      <div className="space-y-2">
                        <label className="block text-sm font-bold text-gray-700">Sleep Hours per Night *</label>
                        <input
                          type="number"
                          min="1"
                          max="24"
                          step="0.5"
                          {...form.register('sleep_hours')}
                          className="w-full px-4 py-3 border-2 border-gray-200 rounded-xl focus:ring-2 focus:ring-green-500 focus:border-green-500 transition-all duration-200"
                        />
                      </div>

                      <div className="space-y-2">
                        <label className="block text-sm font-bold text-gray-700">Exercise Frequency *</label>
                        <select
                          {...form.register('exercise')}
                          className="w-full px-4 py-3 border-2 border-gray-200 rounded-xl focus:ring-2 focus:ring-green-500 focus:border-green-500 transition-all duration-200"
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

                      <div className="space-y-2">
                        <label className="block text-sm font-bold text-gray-700">Smoking Status *</label>
                        <select
                          {...form.register('smoking')}
                          className="w-full px-4 py-3 border-2 border-gray-200 rounded-xl focus:ring-2 focus:ring-green-500 focus:border-green-500 transition-all duration-200"
                        >
                          <option value="">Select status</option>
                          <option value="never">Never smoked</option>
                          <option value="former">Former smoker</option>
                          <option value="current">Current smoker</option>
                        </select>
                      </div>

                      <div className="space-y-2">
                        <label className="block text-sm font-bold text-gray-700">Alcohol Consumption *</label>
                        <select
                          {...form.register('alcohol')}
                          className="w-full px-4 py-3 border-2 border-gray-200 rounded-xl focus:ring-2 focus:ring-green-500 focus:border-green-500 transition-all duration-200"
                        >
                          <option value="">Select frequency</option>
                          <option value="never">Never</option>
                          <option value="rarely">Rarely</option>
                          <option value="weekly">Weekly</option>
                          <option value="daily">Daily</option>
                        </select>
                      </div>
                    </div>

                    <div className="space-y-2">
                      <label className="block text-sm font-bold text-gray-700">
                        Stress Level (1 = Very low, 10 = Very high) *
                      </label>
                      <div className="space-y-3">
                        <input
                          type="range"
                          min="1"
                          max="10"
                          {...form.register('stress_level')}
                          className="w-full h-3 bg-gradient-to-r from-green-200 via-yellow-200 to-red-200 rounded-lg appearance-none cursor-pointer"
                        />
                        <div className="flex justify-between text-sm text-gray-600">
                          <span>Very Low (1)</span>
                          <span className="font-bold text-2xl text-orange-600 bg-orange-50 px-3 py-1 rounded-xl">
                            {form.watch('stress_level') || 5}
                          </span>
                          <span>Very High (10)</span>
                        </div>
                      </div>
                    </div>
                  </div>
                )}

                {/* Step 5: Financial Information */}
                {currentStep === 5 && (
                  <div className="space-y-6">
                    <div className="text-center mb-8">
                      <div className="bg-gradient-to-r from-orange-100 to-yellow-100 w-20 h-20 rounded-3xl flex items-center justify-center mx-auto mb-4">
                        <DollarSign className="w-10 h-10 text-orange-600" />
                      </div>
                      <h2 className="text-3xl font-bold text-gray-900 mb-2">Financial Information</h2>
                      <p className="text-gray-600">Help us find the right care options for your budget</p>
                    </div>

                    <div className="space-y-2">
                      <label className="block text-sm font-bold text-gray-700">Financial Capability *</label>
                      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                        {[
                          { value: 'low', label: 'Budget-conscious', desc: 'Under $100', color: 'border-green-300 hover:bg-green-50' },
                          { value: 'medium', label: 'Moderate budget', desc: '$100-$500', color: 'border-blue-300 hover:bg-blue-50' },
                          { value: 'high', label: 'Flexible budget', desc: '$500+', color: 'border-purple-300 hover:bg-purple-50' }
                        ].map((option) => (
                          <label key={option.value} className={`flex flex-col p-6 border-2 rounded-2xl cursor-pointer transition-all duration-200 ${option.color}`}>
                            <input
                              type="radio"
                              value={option.value}
                              {...form.register('financial_capability')}
                              className="sr-only"
                            />
                            <div className="text-center">
                              <div className="font-bold text-lg text-gray-900 mb-1">{option.label}</div>
                              <div className="text-sm text-gray-600">{option.desc}</div>
                            </div>
                          </label>
                        ))}
                      </div>
                      {form.formState.errors.financial_capability && (
                        <p className="text-red-600 text-sm">{form.formState.errors.financial_capability.message}</p>
                      )}
                    </div>
                  </div>
                )}

                {/* Step 6: Review & Submit */}
                {currentStep === 6 && (
                  <div className="space-y-6">
                    <div className="text-center mb-8">
                      <div className="bg-gradient-to-r from-teal-100 to-green-100 w-20 h-20 rounded-3xl flex items-center justify-center mx-auto mb-4">
                        <CheckCircle className="w-10 h-10 text-teal-600" />
                      </div>
                      <h2 className="text-3xl font-bold text-gray-900 mb-2">Review & Submit</h2>
                      <p className="text-gray-600">Ready for advanced ML/AI analysis</p>
                    </div>

                    <div className="bg-gradient-to-r from-blue-50 via-purple-50 to-pink-50 rounded-2xl p-8 border-2 border-blue-200">
                      <h3 className="text-2xl font-bold text-gray-900 mb-6 text-center">🤖 Advanced AI Analysis Pipeline</h3>
                      <div className="space-y-4">
                        <div className="flex items-center space-x-4 p-4 bg-white/70 rounded-xl">
                          <div className="bg-blue-600 text-white rounded-full w-10 h-10 flex items-center justify-center text-lg font-bold">1</div>
                          <div>
                            <div className="font-bold text-gray-900">BioClinicalBERT NLP Analysis</div>
                            <div className="text-gray-600 text-sm">Advanced medical text understanding and symptom extraction</div>
                          </div>
                        </div>
                        <div className="flex items-center space-x-4 p-4 bg-white/70 rounded-xl">
                          <div className="bg-purple-600 text-white rounded-full w-10 h-10 flex items-center justify-center text-lg font-bold">2</div>
                          <div>
                            <div className="font-bold text-gray-900">Ensemble ML Models</div>
                            <div className="text-gray-600 text-sm">XGBoost + LightGBM + Random Forest + Neural Networks</div>
                          </div>
                        </div>
                        <div className="flex items-center space-x-4 p-4 bg-white/70 rounded-xl">
                          <div className="bg-green-600 text-white rounded-full w-10 h-10 flex items-center justify-center text-lg font-bold">3</div>
                          <div>
                            <div className="font-bold text-gray-900">Risk Assessment & Pattern Analysis</div>
                            <div className="text-gray-600 text-sm">Advanced anomaly detection and health risk prediction</div>
                          </div>
                        </div>
                        <div className="flex items-center space-x-4 p-4 bg-white/70 rounded-xl">
                          <div className="bg-pink-600 text-white rounded-full w-10 h-10 flex items-center justify-center text-lg font-bold">4</div>
                          <div>
                            <div className="font-bold text-gray-900">Gemini AI Validation (Minimal)</div>
                            <div className="text-gray-600 text-sm">Final validation and patient-friendly explanation</div>
                          </div>
                        </div>
                      </div>
                    </div>

                    <div className="bg-yellow-50 border-2 border-yellow-200 rounded-2xl p-6">
                      <div className="flex items-start space-x-4">
                        <AlertTriangle className="w-8 h-8 text-yellow-600 mt-1" />
                        <div>
                          <h4 className="font-bold text-yellow-900 mb-2 text-lg">Important Medical Disclaimer</h4>
                          <p className="text-yellow-800 leading-relaxed">
                            This advanced AI assessment uses state-of-the-art machine learning models trained on medical data. 
                            It provides preliminary insights to assist healthcare professionals, not replace them. 
                            Always consult qualified healthcare providers for final diagnosis and treatment decisions.
                          </p>
                        </div>
                      </div>
                    </div>

                    <button
                      type="submit"
                      disabled={isSubmitting}
                      className="w-full bg-gradient-to-r from-blue-600 via-purple-600 to-pink-600 text-white py-6 px-8 rounded-2xl font-bold text-xl disabled:opacity-50 disabled:cursor-not-allowed hover:shadow-2xl transition-all duration-300 flex items-center justify-center space-x-4 transform hover:scale-105"
                    >
                      {isSubmitting ? (
                        <>
                          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-white"></div>
                          <span>🧠 Advanced ML/AI Analysis in Progress...</span>
                          <Brain className="w-8 h-8" />
                        </>
                      ) : (
                        <>
                          <Zap className="w-8 h-8" />
                          <span>🚀 Start Advanced ML/AI Analysis</span>
                          <Brain className="w-8 h-8" />
                        </>
                      )}
                    </button>
                  </div>
                )}
              </motion.div>
            </AnimatePresence>

            {/* Navigation */}
            {currentStep < 6 && (
              <div className="flex justify-between mt-8">
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
      </motion.div>
    </div>
  );
};