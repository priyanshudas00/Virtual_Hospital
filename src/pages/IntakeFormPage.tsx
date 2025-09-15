import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { useAuth } from '../contexts/AuthContext';
import { TriageChat } from '../components/TriageChat';
import {
  Stethoscope,
  Brain,
  MessageSquare,
  Activity,
  AlertTriangle
} from 'lucide-react';
import { DiagnosisModal } from '../components/DiagnosisModal';

export const IntakeFormPage: React.FC = () => {
  const { user } = useAuth();
  const [assessmentMode, setAssessmentMode] = useState<'chat' | 'form'>('chat');
  const [diagnosisData, setDiagnosisData] = useState<any>(null);
  const [showDiagnosisModal, setShowDiagnosisModal] = useState(false);

  const handleSaveToDashboard = async () => {
    window.location.href = `/diagnosis`;
  };

  const handleBookAppointment = () => {
    window.location.href = '/book-appointment';
  };

  const handleAssessmentComplete = (assessment: any) => {
    setDiagnosisData(assessment);
    setShowDiagnosisModal(true);
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
        <div className="max-w-6xl mx-auto">
          {/* Header */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-center mb-8"
          >
            <div className="bg-gradient-to-r from-blue-600 to-purple-600 w-24 h-24 rounded-3xl flex items-center justify-center mx-auto mb-6">
              <Stethoscope className="w-10 h-10 text-white" />
            </div>
            <h1 className="text-4xl font-bold text-gray-900 mb-3">AI Medical Triage</h1>
            <p className="text-xl text-gray-600 mb-6">Intelligent symptom assessment with dynamic questioning</p>
          </motion.div>

          {/* Assessment Mode Selection */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="max-w-4xl mx-auto mb-8"
          >
            <div className="bg-white rounded-2xl shadow-xl p-6">
              <h2 className="text-2xl font-bold text-gray-900 mb-6 text-center">Choose Your Assessment Method</h2>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <button
                  onClick={() => setAssessmentMode('chat')}
                  className={`p-8 rounded-2xl border-2 transition-all duration-300 ${
                    assessmentMode === 'chat'
                      ? 'border-blue-500 bg-blue-50 shadow-lg'
                      : 'border-gray-200 hover:border-blue-300 hover:bg-blue-50'
                  }`}
                >
                  <div className="text-center">
                    <div className="bg-gradient-to-r from-blue-600 to-purple-600 w-16 h-16 rounded-2xl flex items-center justify-center mx-auto mb-4">
                      <MessageSquare className="w-8 h-8 text-white" />
                    </div>
                    <h3 className="text-xl font-bold text-gray-900 mb-3">AI Chat Triage</h3>
                    <p className="text-gray-600 mb-4">
                      Interactive conversation with AI that asks intelligent follow-up questions 
                      based on your responses. More natural and adaptive.
                    </p>
                    <div className="flex items-center justify-center space-x-2 text-sm text-blue-600">
                      <Brain className="w-4 h-4" />
                      <span>Recommended for most users</span>
                    </div>
                  </div>
                </button>

                <button
                  onClick={() => setAssessmentMode('form')}
                  className={`p-8 rounded-2xl border-2 transition-all duration-300 ${
                    assessmentMode === 'form'
                      ? 'border-green-500 bg-green-50 shadow-lg'
                      : 'border-gray-200 hover:border-green-300 hover:bg-green-50'
                  }`}
                >
                  <div className="text-center">
                    <div className="bg-gradient-to-r from-green-600 to-blue-600 w-16 h-16 rounded-2xl flex items-center justify-center mx-auto mb-4">
                      <Activity className="w-8 h-8 text-white" />
                    </div>
                    <h3 className="text-xl font-bold text-gray-900 mb-3">Comprehensive Form</h3>
                    <p className="text-gray-600 mb-4">
                      Detailed structured form covering all aspects of your health. 
                      More thorough but takes longer to complete.
                    </p>
                    <div className="flex items-center justify-center space-x-2 text-sm text-green-600">
                      <Activity className="w-4 h-4" />
                      <span>For detailed assessment</span>
                    </div>
                  </div>
                </button>
              </div>
            </div>
          </motion.div>

          {/* Assessment Interface */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="max-w-4xl mx-auto"
          >
            {assessmentMode === 'chat' ? (
              <TriageChat 
                onAssessmentComplete={handleAssessmentComplete}
                initialComplaint="I need medical assistance"
              />
            ) : (
              <div className="bg-white rounded-2xl shadow-xl p-8 text-center">
                <Activity className="w-16 h-16 text-gray-300 mx-auto mb-4" />
                <h3 className="text-xl font-bold text-gray-900 mb-2">Comprehensive Form</h3>
                <p className="text-gray-600 mb-6">
                  The detailed form is being enhanced. For now, please use the AI Chat Triage 
                  for the most advanced assessment experience.
                </p>
                <button
                  onClick={() => setAssessmentMode('chat')}
                  className="bg-gradient-to-r from-blue-600 to-purple-600 text-white px-6 py-3 rounded-xl font-semibold hover:shadow-lg transition-all duration-200"
                >
                  Switch to AI Chat
                </button>
              </div>
            )}
          </motion.div>

          {/* Safety Notice */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
            className="max-w-4xl mx-auto mt-8"
          >
            <div className="bg-red-50 border border-red-200 rounded-xl p-6">
              <div className="flex items-start space-x-3">
                <AlertTriangle className="w-6 h-6 text-red-600 mt-1" />
                <div>
                  <h4 className="font-bold text-red-900 mb-2">Emergency Protocol</h4>
                  <p className="text-red-800 text-sm leading-relaxed">
                    If you are experiencing a medical emergency (chest pain, difficulty breathing, 
                    severe bleeding, loss of consciousness), please call emergency services immediately (911/112/999) 
                    or go to the nearest emergency room. Do not use this application for emergency situations.
                  </p>
                </div>
              </div>
            </div>
          </motion.div>
        </div>
      </div>

      {/* Diagnosis Modal */}
      <DiagnosisModal
        isOpen={showDiagnosisModal}
        onClose={() => setShowDiagnosisModal(false)}
        diagnosisData={diagnosisData}
        patientData={{}}
        onSaveToDashboard={handleSaveToDashboard}
        onBookAppointment={handleBookAppointment}
      />
    </>
  );
};