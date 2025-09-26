import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { useForm } from 'react-hook-form';
import { yupResolver } from '@hookform/resolvers/yup';
import * as yup from 'yup';
import { useAuth } from '../contexts/AuthContext';
import { 
  Calendar, 
  Clock, 
  CreditCard, 
  User, 
  FileText, 
  AlertCircle,
  CheckCircle,
  Stethoscope,
  Brain,
  Activity,
  Heart,
  Zap
} from 'lucide-react';
import { format, addDays, addHours } from 'date-fns';

const appointmentSchema = yup.object({
  type: yup.string().required('Appointment type is required'),
  preferredDate: yup.string().required('Preferred date is required'),
  preferredTime: yup.string().required('Preferred time is required'),
  symptoms: yup.string().when('type', {
    is: (val: string) => ['consultation', 'diagnosis', 'emergency'].includes(val),
    then: (schema) => schema.required('Please describe your symptoms'),
    otherwise: (schema) => schema,
  }),
  urgency: yup.string().required('Urgency level is required'),
  notes: yup.string(),
});

interface AppointmentType {
  id: string;
  name: string;
  description: string;
  duration: number;
  cost: number;
  icon: React.ComponentType<any>;
  color: string;
}

const appointmentTypes: AppointmentType[] = [
  {
    id: 'consultation',
    name: 'AI Consultation',
    description: 'General health consultation with AI analysis',
    duration: 30,
    cost: 49.99,
    icon: Stethoscope,
    color: 'bg-blue-500'
  },
  {
    id: 'diagnosis',
    name: 'AI Diagnosis',
    description: 'Comprehensive symptom analysis and diagnosis',
    duration: 45,
    cost: 79.99,
    icon: Brain,
    color: 'bg-purple-500'
  },
  {
    id: 'imaging',
    name: 'Medical Imaging',
    description: 'X-ray, MRI, or CT scan analysis',
    duration: 60,
    cost: 129.99,
    icon: Activity,
    color: 'bg-green-500'
  },
  {
    id: 'lab_test',
    name: 'Lab Test Analysis',
    description: 'Blood work and lab result interpretation',
    duration: 30,
    cost: 39.99,
    icon: Heart,
    color: 'bg-red-500'
  },
  {
    id: 'emergency',
    name: 'Emergency Assessment',
    description: 'Urgent medical evaluation and triage',
    duration: 15,
    cost: 199.99,
    icon: Zap,
    color: 'bg-orange-500'
  }
];

export const AppointmentBookingPage: React.FC = () => {
  const { user } = useAuth();
  const [selectedType, setSelectedType] = useState<AppointmentType | null>(null);
  const [currentStep, setCurrentStep] = useState(1);
  const [loading, setLoading] = useState(false);

  const form = useForm({
    resolver: yupResolver(appointmentSchema),
    defaultValues: {
      urgency: 'medium',
    },
  });

  const generateTimeSlots = () => {
    const slots = [];
    for (let hour = 8; hour <= 18; hour++) {
      for (let minute = 0; minute < 60; minute += 30) {
        const time = `${hour.toString().padStart(2, '0')}:${minute.toString().padStart(2, '0')}`;
        slots.push(time);
      }
    }
    return slots;
  };

  const generateAvailableDates = () => {
    const dates = [];
    for (let i = 0; i < 14; i++) {
      dates.push(addDays(new Date(), i));
    }
    return dates;
  };

  const handleTypeSelection = (type: AppointmentType) => {
    setSelectedType(type);
    form.setValue('type', type.id);
    setCurrentStep(2);
  };

  const handleScheduling = (data: any) => {
    setCurrentStep(3);
  };

  const handlePayment = async () => {
    setLoading(true);
    try {
      // Create appointment and process payment
      const appointmentData = {
        ...form.getValues(),
        patientId: user?.id,
        cost: selectedType?.cost,
        status: 'scheduled',
      };

      // Here you would integrate with your payment processor
      console.log('Processing payment for appointment:', appointmentData);
      
      // Simulate payment processing
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      setCurrentStep(4);
    } catch (error) {
      console.error('Payment failed:', error);
    } finally {
      setLoading(false);
    }
  };

  if (!user) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <AlertCircle className="w-16 h-16 text-red-500 mx-auto mb-4" />
          <h2 className="text-2xl font-bold text-gray-900 mb-2">Authentication Required</h2>
          <p className="text-gray-600">Please sign in to book an appointment.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen py-8 px-4 sm:px-6 lg:px-8">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-12"
        >
          <h1 className="text-3xl font-bold text-gray-900 mb-2">Book Your Appointment</h1>
          <p className="text-gray-600">Schedule your AI-powered medical consultation</p>
        </motion.div>

        {/* Progress Steps */}
        <div className="mb-8">
          <div className="flex items-center justify-center space-x-4">
            {[1, 2, 3, 4].map((step) => (
              <div key={step} className="flex items-center">
                <div className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-medium ${
                  currentStep >= step 
                    ? 'bg-blue-600 text-white' 
                    : 'bg-gray-200 text-gray-600'
                }`}>
                  {currentStep > step ? <CheckCircle className="w-4 h-4" /> : step}
                </div>
                {step < 4 && (
                  <div className={`w-16 h-1 mx-2 ${
                    currentStep > step ? 'bg-blue-600' : 'bg-gray-200'
                  }`} />
                )}
              </div>
            ))}
          </div>
          <div className="flex justify-center mt-2">
            <div className="text-sm text-gray-600">
              {currentStep === 1 && 'Select Service'}
              {currentStep === 2 && 'Schedule Appointment'}
              {currentStep === 3 && 'Payment'}
              {currentStep === 4 && 'Confirmation'}
            </div>
          </div>
        </div>

        {/* Step 1: Service Selection */}
        {currentStep === 1 && (
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6"
          >
            {appointmentTypes.map((type) => {
              const Icon = type.icon;
              return (
                <div
                  key={type.id}
                  onClick={() => handleTypeSelection(type)}
                  className="bg-white rounded-2xl shadow-lg p-6 cursor-pointer hover:shadow-xl transition-all duration-300 border-2 border-transparent hover:border-blue-200"
                >
                  <div className={`${type.color} w-12 h-12 rounded-xl flex items-center justify-center mb-4`}>
                    <Icon className="w-6 h-6 text-white" />
                  </div>
                  <h3 className="text-xl font-bold text-gray-900 mb-2">{type.name}</h3>
                  <p className="text-gray-600 mb-4">{type.description}</p>
                  <div className="flex justify-between items-center">
                    <span className="text-2xl font-bold text-green-600">${type.cost}</span>
                    <span className="text-sm text-gray-500">{type.duration} min</span>
                  </div>
                </div>
              );
            })}
          </motion.div>
        )}

        {/* Step 2: Scheduling */}
        {currentStep === 2 && selectedType && (
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            className="bg-white rounded-2xl shadow-lg p-8"
          >
            <div className="mb-6">
              <h2 className="text-2xl font-bold text-gray-900 mb-2">Schedule Your {selectedType.name}</h2>
              <p className="text-gray-600">Select your preferred date and time</p>
            </div>

            <form onSubmit={form.handleSubmit(handleScheduling)} className="space-y-6">
              {/* Date Selection */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-3">
                  Preferred Date
                </label>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                  {generateAvailableDates().slice(0, 8).map((date) => (
                    <label key={date.toISOString()} className="flex items-center p-3 border rounded-xl cursor-pointer hover:bg-blue-50 transition-all duration-200">
                      <input
                        type="radio"
                        value={format(date, 'yyyy-MM-dd')}
                        {...form.register('preferredDate')}
                        className="mr-2"
                      />
                      <div className="text-center">
                        <div className="text-sm font-medium">{format(date, 'MMM dd')}</div>
                        <div className="text-xs text-gray-500">{format(date, 'EEE')}</div>
                      </div>
                    </label>
                  ))}
                </div>
                {form.formState.errors.preferredDate && (
                  <p className="text-red-600 text-sm mt-1">{form.formState.errors.preferredDate.message}</p>
                )}
              </div>

              {/* Time Selection */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-3">
                  Preferred Time
                </label>
                <div className="grid grid-cols-3 md:grid-cols-6 gap-3">
                  {generateTimeSlots().slice(0, 12).map((time) => (
                    <label key={time} className="flex items-center justify-center p-2 border rounded-lg cursor-pointer hover:bg-blue-50 transition-all duration-200">
                      <input
                        type="radio"
                        value={time}
                        {...form.register('preferredTime')}
                        className="sr-only"
                      />
                      <span className="text-sm font-medium">{time}</span>
                    </label>
                  ))}
                </div>
                {form.formState.errors.preferredTime && (
                  <p className="text-red-600 text-sm mt-1">{form.formState.errors.preferredTime.message}</p>
                )}
              </div>

              {/* Symptoms (if applicable) */}
              {['consultation', 'diagnosis', 'emergency'].includes(selectedType.id) && (
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Describe Your Symptoms
                  </label>
                  <textarea
                    {...form.register('symptoms')}
                    rows={4}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                    placeholder="Please describe your symptoms in detail..."
                  />
                  {form.formState.errors.symptoms && (
                    <p className="text-red-600 text-sm mt-1">{form.formState.errors.symptoms.message}</p>
                  )}
                </div>
              )}

              {/* Urgency Level */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-3">
                  Urgency Level
                </label>
                <div className="grid grid-cols-3 gap-3">
                  <label className="flex items-center p-3 border rounded-xl cursor-pointer hover:bg-green-50 transition-all duration-200">
                    <input
                      type="radio"
                      value="low"
                      {...form.register('urgency')}
                      className="mr-2"
                    />
                    <div>
                      <div className="font-medium text-green-700">Low</div>
                      <div className="text-xs text-gray-500">Routine care</div>
                    </div>
                  </label>
                  <label className="flex items-center p-3 border rounded-xl cursor-pointer hover:bg-yellow-50 transition-all duration-200">
                    <input
                      type="radio"
                      value="medium"
                      {...form.register('urgency')}
                      className="mr-2"
                    />
                    <div>
                      <div className="font-medium text-yellow-700">Medium</div>
                      <div className="text-xs text-gray-500">Within days</div>
                    </div>
                  </label>
                  <label className="flex items-center p-3 border rounded-xl cursor-pointer hover:bg-red-50 transition-all duration-200">
                    <input
                      type="radio"
                      value="high"
                      {...form.register('urgency')}
                      className="mr-2"
                    />
                    <div>
                      <div className="font-medium text-red-700">High</div>
                      <div className="text-xs text-gray-500">Urgent care</div>
                    </div>
                  </label>
                </div>
              </div>

              {/* Additional Notes */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Additional Notes (Optional)
                </label>
                <textarea
                  {...form.register('notes')}
                  rows={3}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  placeholder="Any additional information you'd like to share..."
                />
              </div>

              <div className="flex space-x-4">
                <button
                  type="button"
                  onClick={() => setCurrentStep(1)}
                  className="flex-1 bg-gray-200 text-gray-800 py-3 px-4 rounded-xl font-semibold hover:bg-gray-300 transition-all duration-200"
                >
                  Back
                </button>
                <button
                  type="submit"
                  className="flex-1 bg-gradient-to-r from-blue-600 to-green-600 text-white py-3 px-4 rounded-xl font-semibold hover:shadow-lg transition-all duration-200"
                >
                  Continue to Payment
                </button>
              </div>
            </form>
          </motion.div>
        )}

        {/* Step 3: Payment */}
        {currentStep === 3 && selectedType && (
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            className="bg-white rounded-2xl shadow-lg p-8"
          >
            <div className="mb-6">
              <h2 className="text-2xl font-bold text-gray-900 mb-2">Payment Information</h2>
              <p className="text-gray-600">Secure payment processing</p>
            </div>

            {/* Order Summary */}
            <div className="bg-gray-50 rounded-xl p-6 mb-6">
              <h3 className="text-lg font-bold text-gray-900 mb-4">Order Summary</h3>
              <div className="space-y-3">
                <div className="flex justify-between">
                  <span className="text-gray-600">Service</span>
                  <span className="font-medium">{selectedType.name}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Duration</span>
                  <span className="font-medium">{selectedType.duration} minutes</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Date & Time</span>
                  <span className="font-medium">
                    {form.getValues('preferredDate')} at {form.getValues('preferredTime')}
                  </span>
                </div>
                <hr />
                <div className="flex justify-between text-lg font-bold">
                  <span>Total</span>
                  <span className="text-green-600">${selectedType.cost}</span>
                </div>
              </div>
            </div>

            {/* Payment Form */}
            <div className="space-y-6">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Card Number
                </label>
                <input
                  type="text"
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  placeholder="1234 5678 9012 3456"
                />
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Expiry Date
                  </label>
                  <input
                    type="text"
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                    placeholder="MM/YY"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    CVV
                  </label>
                  <input
                    type="text"
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                    placeholder="123"
                  />
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Cardholder Name
                </label>
                <input
                  type="text"
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  placeholder="John Doe"
                />
              </div>

              <div className="flex space-x-4">
                <button
                  type="button"
                  onClick={() => setCurrentStep(2)}
                  className="flex-1 bg-gray-200 text-gray-800 py-3 px-4 rounded-xl font-semibold hover:bg-gray-300 transition-all duration-200"
                >
                  Back
                </button>
                <button
                  onClick={handlePayment}
                  disabled={loading}
                  className="flex-1 bg-gradient-to-r from-blue-600 to-green-600 text-white py-3 px-4 rounded-xl font-semibold disabled:opacity-50 disabled:cursor-not-allowed hover:shadow-lg transition-all duration-200 flex items-center justify-center space-x-2"
                >
                  {loading ? (
                    <>
                      <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
                      <span>Processing...</span>
                    </>
                  ) : (
                    <>
                      <CreditCard className="w-5 h-5" />
                      <span>Pay ${selectedType.cost}</span>
                    </>
                  )}
                </button>
              </div>
            </div>
          </motion.div>
        )}

        {/* Step 4: Confirmation */}
        {currentStep === 4 && selectedType && (
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="bg-white rounded-2xl shadow-lg p-8 text-center"
          >
            <div className="bg-green-100 w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-6">
              <CheckCircle className="w-8 h-8 text-green-600" />
            </div>
            <h2 className="text-2xl font-bold text-gray-900 mb-2">Appointment Confirmed!</h2>
            <p className="text-gray-600 mb-6">
              Your {selectedType.name} has been successfully scheduled.
            </p>
            
            <div className="bg-gray-50 rounded-xl p-6 mb-6">
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span className="text-gray-600">Appointment ID</span>
                  <span className="font-medium">#APT-{Date.now()}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Service</span>
                  <span className="font-medium">{selectedType.name}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Date & Time</span>
                  <span className="font-medium">
                    {form.getValues('preferredDate')} at {form.getValues('preferredTime')}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Amount Paid</span>
                  <span className="font-medium text-green-600">${selectedType.cost}</span>
                </div>
              </div>
            </div>

            <div className="space-y-4">
              <p className="text-sm text-gray-600">
                You will receive a confirmation email with appointment details and preparation instructions.
              </p>
              <button
                onClick={() => window.location.href = '/dashboard'}
                className="w-full bg-gradient-to-r from-blue-600 to-green-600 text-white py-3 px-4 rounded-xl font-semibold hover:shadow-lg transition-all duration-200"
              >
                Go to Dashboard
              </button>
            </div>
          </motion.div>
        )}
      </div>
    </div>
  );
};