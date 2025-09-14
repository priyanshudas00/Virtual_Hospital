import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { useForm } from 'react-hook-form';
import { yupResolver } from '@hookform/resolvers/yup';
import * as yup from 'yup';
import { useAuth } from '../contexts/AuthContext';
import { DiagnosisModal } from '../components/DiagnosisModal';
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
  CheckCircle,
  Clock,
  Stethoscope,
  Shield,
  Home,
  Briefcase
} from 'lucide-react';

const intakeSchema = yup.object({
  // Basic Information
  age: yup.number().required('Age is required').min(1).max(120),
  gender: yup.string().required('Gender is required'),
  location: yup.string().required('Location is required'),
  occupation: yup.string().required('Occupation is required'),
  marital_status: yup.string().required('Marital status is required'),
  education_level: yup.string().required('Education level is required'),
  
  // Symptoms
  symptoms: yup.array().min(1, 'Please describe at least one symptom'),
  symptom_duration: yup.string().required('Symptom duration is required'),
  pain_level: yup.number().min(0).max(10),
  onset_pattern: yup.string().required('Onset pattern is required'),
  
  // Medical History
  medical_history: yup.array(),
  current_medications: yup.array(),
  allergies_text: yup.string(),
  surgeries_text: yup.string(),
  family_history_text: yup.string(),
  
  // Lifestyle
  sleep_hours: yup.number().min(0).max(24),
  sleep_quality: yup.string().required('Sleep quality is required'),
  exercise: yup.string().required('Exercise information is required'),
  diet: yup.string().required('Diet information is required'),
  smoking: yup.string().required('Smoking status is required'),
  alcohol: yup.string().required('Alcohol consumption is required'),
  stress_level: yup.number().min(1).max(10),
  
  // Social & Environmental
  living_situation: yup.string().required('Living situation is required'),
  support_system: yup.string().required('Support system is required'),
  transportation: yup.string().required('Transportation access is required'),
  
  // Insurance & Financial
  insurance_provider: yup.string(),
  policy_number: yup.string(),
  financial_capability: yup.string().required('Financial capability is required'),
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
    defaultValues: {
      symptoms: [],
      medical_history: [],
      current_medications: [],
      allergies: [],
      surgeries: [],
      family_history: [],
      pain_level: 0,
      stress_level: 5,
      sleep_hours: 8,
      mood_rating: 5,
      anxiety_level: 3,
    },
  });

  const steps = [
    { id: 1, title: 'Personal Details', icon: User, fields: ['age', 'gender', 'location', 'occupation', 'marital_status', 'education_level'] },
    { id: 2, title: 'Current Symptoms', icon: Heart, fields: ['symptoms', 'symptom_duration', 'pain_level', 'onset_pattern'] },
    { id: 3, title: 'Medical History', icon: FileText, fields: ['medical_history', 'current_medications', 'allergies_text', 'surgeries_text'] },
    { id: 4, title: 'Lifestyle & Habits', icon: Activity, fields: ['sleep_hours', 'exercise', 'diet', 'smoking', 'alcohol', 'stress_level'] },
    { id: 5, title: 'Social & Mental Health', icon: Brain, fields: ['living_situation', 'support_system', 'mood_rating', 'anxiety_level'] },
    { id: 6, title: 'Insurance & Financial', icon: DollarSign, fields: ['insurance_provider', 'financial_capability'] },
    { id: 7, title: 'Review & Submit', icon: CheckCircle, fields: [] }
  ];

  // Enhanced symptom categories
  const symptomCategories = {
    'Constitutional': ['Fever', 'Fatigue', 'Weight Loss', 'Weight Gain', 'Night Sweats', 'Chills', 'Loss of Appetite'],
    'Neurological': ['Headache', 'Dizziness', 'Confusion', 'Memory Loss', 'Seizures', 'Numbness', 'Weakness', 'Tremors'],
    'Cardiovascular': ['Chest Pain', 'Palpitations', 'Shortness of Breath', 'Leg Swelling', 'Irregular Heartbeat'],
    'Respiratory': ['Cough', 'Shortness of Breath', 'Wheezing', 'Sputum Production', 'Chest Tightness'],
    'Gastrointestinal': ['Nausea', 'Vomiting', 'Diarrhea', 'Constipation', 'Abdominal Pain', 'Heartburn', 'Bloating'],
    'Musculoskeletal': ['Joint Pain', 'Muscle Pain', 'Back Pain', 'Neck Pain', 'Stiffness', 'Swelling'],
    'Dermatological': ['Skin Rash', 'Itching', 'Skin Lesions', 'Bruising', 'Hair Loss', 'Nail Changes'],
    'Genitourinary': ['Urinary Frequency', 'Burning Urination', 'Blood in Urine', 'Pelvic Pain'],
    'Psychiatric': ['Anxiety', 'Depression', 'Insomnia', 'Mood Changes', 'Panic Attacks', 'Irritability']
  };

  const commonConditions = [
    'Diabetes Type 1', 'Diabetes Type 2', 'Hypertension', 'Heart Disease', 'Asthma', 'COPD',
    'Arthritis', 'Osteoporosis', 'Depression', 'Anxiety Disorder', 'Thyroid Disorders',
    'Kidney Disease', 'Liver Disease', 'Cancer History', 'Stroke', 'Migraine',
    'Epilepsy', 'Allergic Rhinitis', 'Eczema', 'Psoriasis', 'Inflammatory Bowel Disease'
  ];

  const commonMedications = [
    'Aspirin', 'Ibuprofen', 'Acetaminophen', 'Metformin', 'Lisinopril', 'Atorvastatin',
    'Omeprazole', 'Levothyroxine', 'Amlodipine', 'Metoprolol', 'Losartan', 'Simvastatin',
    'Prednisone', 'Insulin', 'Albuterol', 'Warfarin', 'Clopidogrel', 'Furosemide'
  ];

  // Calculate form progress
  React.useEffect(() => {
    const watchedValues = form.watch();
    const totalFields = Object.keys(intakeSchema.fields).length;
    const completedFields = Object.values(watchedValues).filter(value => {
      if (Array.isArray(value)) return value.length > 0;
      if (typeof value === 'string') return value.trim() !== '';
      if (typeof value === 'number') return value > 0;
      return value !== null && value !== undefined;
    }).length;
    
    setFormProgress((completedFields / totalFields) * 100);
  }, [form.watch()]);

  const handleNext = async () => {
    const currentStepFields = steps[currentStep - 1].fields;
    const isValid = await form.trigger(currentStepFields);
    
    if (isValid && currentStep < steps.length) {
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
      // Add completion metadata
      const submissionData = {
        ...data,
        completion_time: new Date().toISOString(),
        form_version: '2.0',
        user_agent: navigator.userAgent,
        completion_percentage: formProgress
      };

      const response = await fetch(`${import.meta.env.VITE_API_URL}/api/intake-form`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        },
        body: JSON.stringify(submissionData)
      });

      const result = await response.json();

      if (response.ok && result.assessment_data) {
        setDiagnosisData(result.assessment_data);
        setShowDiagnosisModal(true);
      } else {
        throw new Error(result.error || 'Failed to get AI assessment');
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

  const handleSaveToDashboard = async () => {
    // Navigate to diagnosis page with the data
    window.location.href = `/diagnosis?report_id=${diagnosisData?.metadata?.assessment_id}`;
  };

  const handleBookAppointment = () => {
    window.location.href = '/book-appointment';
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
    <>
      <div className="min-h-screen py-8 px-4 sm:px-6 lg:px-8 bg-gradient-to-br from-blue-50 via-white to-green-50">
        <div className="max-w-5xl mx-auto">
          {/* Header */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-center mb-8"
          >
            <div className="bg-gradient-to-r from-blue-600 to-purple-600 w-20 h-20 rounded-3xl flex items-center justify-center mx-auto mb-6">
              <Stethoscope className="w-10 h-10 text-white" />
            </div>
            <h1 className="text-4xl font-bold text-gray-900 mb-3">Comprehensive Medical Assessment</h1>
            <p className="text-xl text-gray-600 mb-4">Complete evaluation for AI-powered medical analysis</p>
            
            {/* Progress Bar */}
            <div className="max-w-md mx-auto">
              <div className="flex justify-between text-sm text-gray-600 mb-2">
                <span>Progress</span>
                <span>{Math.round(formProgress)}% Complete</span>
              </div>
              <div className="bg-gray-200 rounded-full h-3">
                <motion.div
                  className="bg-gradient-to-r from-blue-600 to-purple-600 h-3 rounded-full"
                  initial={{ width: 0 }}
                  animate={{ width: `${formProgress}%` }}
                  transition={{ duration: 0.5 }}
                />
              </div>
            </div>
          </motion.div>

          {/* Progress Steps */}
          <div className="mb-8">
            <div className="flex items-center justify-between max-w-4xl mx-auto">
              {steps.map((step, index) => {
                const Icon = step.icon;
                return (
                  <div key={step.id} className="flex items-center">
                    <div className={`flex items-center justify-center w-12 h-12 rounded-full transition-all duration-300 ${
                      currentStep >= step.id 
                        ? 'bg-gradient-to-r from-blue-600 to-purple-600 text-white shadow-lg' 
                        : 'bg-gray-200 text-gray-600'
                    }`}>
                      {currentStep > step.id ? (
                        <CheckCircle className="w-6 h-6" />
                      ) : (
                        <Icon className="w-6 h-6" />
                      )}
                    </div>
                    {index < steps.length - 1 && (
                      <div className={`w-full h-1 mx-3 transition-all duration-300 ${
                        currentStep > step.id ? 'bg-gradient-to-r from-blue-600 to-purple-600' : 'bg-gray-200'
                      }`} />
                    )}
                  </div>
                );
              })}
            </div>
            <div className="flex justify-between mt-3 max-w-4xl mx-auto">
              {steps.map((step) => (
                <div key={step.id} className="text-xs text-gray-600 text-center flex-1">
                  {step.title}
                </div>
              ))}
            </div>
          </div>

          <form onSubmit={form.handleSubmit(handleSubmit)}>
            {/* Step 1: Personal Details */}
            {currentStep === 1 && (
              <motion.div
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                className="bg-white rounded-3xl shadow-xl p-8"
              >
                <div className="flex items-center space-x-3 mb-8">
                  <User className="w-8 h-8 text-blue-600" />
                  <h2 className="text-3xl font-bold text-gray-900">Personal Information</h2>
                </div>
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <label className="block text-sm font-semibold text-gray-700 mb-2">Age *</label>
                    <input
                      type="number"
                      {...form.register('age')}
                      className="w-full px-4 py-3 border-2 border-gray-200 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all duration-200"
                      placeholder="Enter your age"
                    />
                    {form.formState.errors.age && (
                      <p className="text-red-600 text-sm mt-1">{form.formState.errors.age.message}</p>
                    )}
                  </div>

                  <div>
                    <label className="block text-sm font-semibold text-gray-700 mb-2">Gender *</label>
                    <select
                      {...form.register('gender')}
                      className="w-full px-4 py-3 border-2 border-gray-200 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all duration-200"
                    >
                      <option value="">Select gender</option>
                      <option value="male">Male</option>
                      <option value="female">Female</option>
                      <option value="non_binary">Non-binary</option>
                      <option value="prefer_not_to_say">Prefer not to say</option>
                    </select>
                    {form.formState.errors.gender && (
                      <p className="text-red-600 text-sm mt-1">{form.formState.errors.gender.message}</p>
                    )}
                  </div>

                  <div>
                    <label className="block text-sm font-semibold text-gray-700 mb-2">Location *</label>
                    <div className="relative">
                      <MapPin className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
                      <input
                        type="text"
                        {...form.register('location')}
                        className="w-full pl-10 pr-4 py-3 border-2 border-gray-200 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all duration-200"
                        placeholder="City, State, Country"
                      />
                    </div>
                    {form.formState.errors.location && (
                      <p className="text-red-600 text-sm mt-1">{form.formState.errors.location.message}</p>
                    )}
                  </div>

                  <div>
                    <label className="block text-sm font-semibold text-gray-700 mb-2">Occupation *</label>
                    <div className="relative">
                      <Briefcase className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
                      <input
                        type="text"
                        {...form.register('occupation')}
                        className="w-full pl-10 pr-4 py-3 border-2 border-gray-200 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all duration-200"
                        placeholder="Your occupation"
                      />
                    </div>
                    {form.formState.errors.occupation && (
                      <p className="text-red-600 text-sm mt-1">{form.formState.errors.occupation.message}</p>
                    )}
                  </div>

                  <div>
                    <label className="block text-sm font-semibold text-gray-700 mb-2">Marital Status *</label>
                    <select
                      {...form.register('marital_status')}
                      className="w-full px-4 py-3 border-2 border-gray-200 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all duration-200"
                    >
                      <option value="">Select status</option>
                      <option value="single">Single</option>
                      <option value="married">Married</option>
                      <option value="divorced">Divorced</option>
                      <option value="widowed">Widowed</option>
                      <option value="separated">Separated</option>
                    </select>
                    {form.formState.errors.marital_status && (
                      <p className="text-red-600 text-sm mt-1">{form.formState.errors.marital_status.message}</p>
                    )}
                  </div>

                  <div>
                    <label className="block text-sm font-semibold text-gray-700 mb-2">Education Level *</label>
                    <select
                      {...form.register('education_level')}
                      className="w-full px-4 py-3 border-2 border-gray-200 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all duration-200"
                    >
                      <option value="">Select education level</option>
                      <option value="primary">Primary School</option>
                      <option value="secondary">High School</option>
                      <option value="undergraduate">Bachelor's Degree</option>
                      <option value="graduate">Master's Degree</option>
                      <option value="postgraduate">PhD/Professional Degree</option>
                    </select>
                    {form.formState.errors.education_level && (
                      <p className="text-red-600 text-sm mt-1">{form.formState.errors.education_level.message}</p>
                    )}
                  </div>
                </div>

                <div className="flex justify-end mt-8">
                  <button
                    type="button"
                    onClick={handleNext}
                    className="bg-gradient-to-r from-blue-600 to-purple-600 text-white px-8 py-3 rounded-xl font-semibold hover:shadow-lg transition-all duration-200 flex items-center space-x-2"
                  >
                    <span>Next Step</span>
                    <CheckCircle className="w-5 h-5" />
                  </button>
                </div>
              </motion.div>
            )}

            {/* Step 2: Current Symptoms */}
            {currentStep === 2 && (
              <motion.div
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                className="bg-white rounded-3xl shadow-xl p-8"
              >
                <div className="flex items-center space-x-3 mb-8">
                  <Heart className="w-8 h-8 text-red-600" />
                  <h2 className="text-3xl font-bold text-gray-900">Current Symptoms</h2>
                </div>
                
                <div className="space-y-8">
                  {/* Symptom Categories */}
                  <div>
                    <label className="block text-lg font-semibold text-gray-700 mb-4">
                      Select your current symptoms: *
                    </label>
                    
                    {Object.entries(symptomCategories).map(([category, symptoms]) => (
                      <div key={category} className="mb-6">
                        <h4 className="font-semibold text-gray-800 mb-3 text-lg">{category}</h4>
                        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3">
                          {symptoms.map((symptom) => (
                            <label key={symptom} className="flex items-center p-3 border-2 border-gray-200 rounded-xl cursor-pointer hover:bg-blue-50 hover:border-blue-300 transition-all duration-200">
                              <input
                                type="checkbox"
                                onChange={(e) => {
                                  if (e.target.checked) {
                                    addToArray('symptoms', symptom);
                                  } else {
                                    removeFromArray('symptoms', symptom);
                                  }
                                }}
                                className="mr-3 w-4 h-4 text-blue-600 rounded focus:ring-blue-500"
                              />
                              <span className="text-sm font-medium">{symptom}</span>
                            </label>
                          ))}
                        </div>
                      </div>
                    ))}
                    {form.formState.errors.symptoms && (
                      <p className="text-red-600 text-sm mt-1">{form.formState.errors.symptoms.message}</p>
                    )}
                  </div>

                  {/* Symptom Details */}
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div>
                      <label className="block text-sm font-semibold text-gray-700 mb-2">
                        Symptom Duration *
                      </label>
                      <select
                        {...form.register('symptom_duration')}
                        className="w-full px-4 py-3 border-2 border-gray-200 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all duration-200"
                      >
                        <option value="">Select duration</option>
                        <option value="less_than_24_hours">Less than 24 hours</option>
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
                      <label className="block text-sm font-semibold text-gray-700 mb-2">
                        Symptom Onset Pattern *
                      </label>
                      <select
                        {...form.register('onset_pattern')}
                        className="w-full px-4 py-3 border-2 border-gray-200 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all duration-200"
                      >
                        <option value="">Select pattern</option>
                        <option value="sudden">Sudden onset</option>
                        <option value="gradual">Gradual onset</option>
                        <option value="intermittent">Comes and goes</option>
                        <option value="constant">Constant/persistent</option>
                        <option value="worsening">Getting worse</option>
                        <option value="improving">Getting better</option>
                      </select>
                      {form.formState.errors.onset_pattern && (
                        <p className="text-red-600 text-sm mt-1">{form.formState.errors.onset_pattern.message}</p>
                      )}
                    </div>
                  </div>

                  {/* Pain Level */}
                  <div>
                    <label className="block text-sm font-semibold text-gray-700 mb-3">
                      Pain Level (0-10)
                    </label>
                    <div className="relative">
                      <input
                        type="range"
                        min="0"
                        max="10"
                        {...form.register('pain_level')}
                        className="w-full h-3 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                      />
                      <div className="flex justify-between text-sm text-gray-500 mt-2">
                        <span>No Pain</span>
                        <span className="font-bold text-lg text-gray-900">{form.watch('pain_level')}/10</span>
                        <span>Severe Pain</span>
                      </div>
                    </div>
                  </div>

                  {/* Detailed Description */}
                  <div>
                    <label className="block text-sm font-semibold text-gray-700 mb-2">
                      Detailed Symptom Description
                    </label>
                    <textarea
                      {...form.register('symptom_details')}
                      rows={4}
                      className="w-full px-4 py-3 border-2 border-gray-200 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all duration-200"
                      placeholder="Describe your symptoms in detail, including when they occur, what makes them better or worse, and any patterns you've noticed..."
                    />
                  </div>
                </div>

                <div className="flex justify-between mt-8">
                  <button
                    type="button"
                    onClick={handlePrevious}
                    className="bg-gray-200 text-gray-800 px-6 py-3 rounded-xl font-semibold hover:bg-gray-300 transition-all duration-200"
                  >
                    Previous
                  </button>
                  <button
                    type="button"
                    onClick={handleNext}
                    className="bg-gradient-to-r from-blue-600 to-purple-600 text-white px-8 py-3 rounded-xl font-semibold hover:shadow-lg transition-all duration-200"
                  >
                    Next Step
                  </button>
                </div>
              </motion.div>
            )}

            {/* Step 7: Review & Submit */}
            {currentStep === 7 && (
              <motion.div
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                className="bg-white rounded-3xl shadow-xl p-8"
              >
                <div className="flex items-center space-x-3 mb-8">
                  <CheckCircle className="w-8 h-8 text-green-600" />
                  <h2 className="text-3xl font-bold text-gray-900">Review & Submit</h2>
                </div>
                
                <div className="space-y-6">
                  {/* Form Summary */}
                  <div className="bg-gradient-to-r from-blue-50 to-purple-50 rounded-2xl p-6 border border-blue-200">
                    <h3 className="font-bold text-gray-900 mb-4 text-xl">Assessment Summary</h3>
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                      <div className="text-center">
                        <div className="text-3xl font-bold text-blue-600">{form.watch('symptoms')?.length || 0}</div>
                        <div className="text-sm text-gray-600">Symptoms Reported</div>
                      </div>
                      <div className="text-center">
                        <div className="text-3xl font-bold text-purple-600">{Math.round(formProgress)}%</div>
                        <div className="text-sm text-gray-600">Form Completion</div>
                      </div>
                      <div className="text-center">
                        <div className="text-3xl font-bold text-green-600">{form.watch('pain_level') || 0}/10</div>
                        <div className="text-sm text-gray-600">Pain Level</div>
                      </div>
                    </div>
                  </div>

                  {/* AI Analysis Preview */}
                  <div className="bg-gradient-to-r from-purple-50 to-pink-50 rounded-2xl p-6 border border-purple-200">
                    <div className="flex items-start space-x-4">
                      <Brain className="w-8 h-8 text-purple-600 mt-1" />
                      <div>
                        <h4 className="font-bold text-purple-900 mb-3 text-xl">AI Medical Analysis</h4>
                        <p className="text-purple-800 mb-4">
                          Our advanced Gemini AI will analyze your comprehensive health information and provide:
                        </p>
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                          <ul className="text-purple-700 space-y-2">
                            <li className="flex items-center space-x-2">
                              <CheckCircle className="w-4 h-4" />
                              <span>Preliminary health assessment</span>
                            </li>
                            <li className="flex items-center space-x-2">
                              <CheckCircle className="w-4 h-4" />
                              <span>Possible conditions & risk factors</span>
                            </li>
                            <li className="flex items-center space-x-2">
                              <CheckCircle className="w-4 h-4" />
                              <span>Recommended tests & investigations</span>
                            </li>
                            <li className="flex items-center space-x-2">
                              <CheckCircle className="w-4 h-4" />
                              <span>Personalized lifestyle suggestions</span>
                            </li>
                          </ul>
                          <ul className="text-purple-700 space-y-2">
                            <li className="flex items-center space-x-2">
                              <CheckCircle className="w-4 h-4" />
                              <span>Medication guidance (general)</span>
                            </li>
                            <li className="flex items-center space-x-2">
                              <CheckCircle className="w-4 h-4" />
                              <span>Specialist referral recommendations</span>
                            </li>
                            <li className="flex items-center space-x-2">
                              <CheckCircle className="w-4 h-4" />
                              <span>Patient education & self-care</span>
                            </li>
                            <li className="flex items-center space-x-2">
                              <CheckCircle className="w-4 h-4" />
                              <span>Follow-up timeline & monitoring</span>
                            </li>
                          </ul>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Medical Disclaimer */}
                  <div className="bg-yellow-50 border-2 border-yellow-200 rounded-2xl p-6">
                    <div className="flex items-start space-x-3">
                      <Shield className="w-6 h-6 text-yellow-600 mt-1" />
                      <div>
                        <h4 className="font-bold text-yellow-900 mb-2">Important Medical Disclaimer</h4>
                        <p className="text-yellow-800 text-sm leading-relaxed">
                          This AI assessment is for informational and educational purposes only and does not replace 
                          professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare 
                          professionals for medical decisions. In case of emergency, call your local emergency services immediately.
                        </p>
                      </div>
                    </div>
                  </div>
                </div>

                <div className="flex justify-between mt-8">
                  <button
                    type="button"
                    onClick={handlePrevious}
                    className="bg-gray-200 text-gray-800 px-6 py-3 rounded-xl font-semibold hover:bg-gray-300 transition-all duration-200"
                  >
                    Previous
                  </button>
                  <button
                    type="submit"
                    disabled={isSubmitting}
                    className="bg-gradient-to-r from-green-600 to-blue-600 text-white px-12 py-4 rounded-xl font-bold text-lg disabled:opacity-50 disabled:cursor-not-allowed hover:shadow-xl transition-all duration-200 flex items-center space-x-3"
                  >
                    {isSubmitting ? (
                      <>
                        <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-white"></div>
                        <span>AI Analyzing...</span>
                        <Brain className="w-6 h-6" />
                      </>
                    ) : (
                      <>
                        <Send className="w-6 h-6" />
                        <span>Get AI Medical Assessment</span>
                        <Brain className="w-6 h-6" />
                      </>
                    )}
                  </button>
                </div>
              </motion.div>
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