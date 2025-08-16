import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { 
  Search, 
  Send, 
  Mic, 
  Brain, 
  AlertCircle, 
  CheckCircle, 
  Clock,
  TrendingUp,
  FileText
} from 'lucide-react';
import axios from 'axios';

export const DiagnosisPage: React.FC = () => {
  const [symptoms, setSymptoms] = useState('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [diagnosis, setDiagnosis] = useState<any>(null);
  const [isListening, setIsListening] = useState(false);

  const commonSymptoms = [
    'Headache and fever',
    'Chest pain and shortness of breath',
    'Abdominal pain and nausea',
    'Persistent cough',
    'Joint pain and stiffness',
    'Skin rash and itching'
  ];

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!symptoms.trim()) return;

    setIsAnalyzing(true);
    
    // Simulate API call to AI diagnosis service
    try {
      // In production, this would call your AI service
      setTimeout(() => {
        const mockDiagnosis = {
          primaryDiagnosis: 'Viral Upper Respiratory Infection',
          confidence: 0.87,
          alternativeDiagnoses: [
            { condition: 'Common Cold', probability: 0.72 },
            { condition: 'Allergic Rhinitis', probability: 0.45 },
            { condition: 'Sinusitis', probability: 0.38 }
          ],
          urgency: 'Low',
          recommendedActions: [
            'Rest and stay hydrated',
            'Take over-the-counter pain relievers',
            'Monitor symptoms for 3-5 days',
            'Consult doctor if symptoms worsen'
          ],
          followUp: 'Schedule follow-up if symptoms persist beyond 7 days'
        };
        
        setDiagnosis(mockDiagnosis);
        setIsAnalyzing(false);
      }, 3000);
    } catch (error) {
      console.error('Diagnosis error:', error);
      setIsAnalyzing(false);
    }
  };

  const handleVoiceInput = () => {
    if ('webkitSpeechRecognition' in window) {
      const recognition = new (window as any).webkitSpeechRecognition();
      recognition.continuous = false;
      recognition.interimResults = false;
      recognition.lang = 'en-US';

      recognition.onstart = () => setIsListening(true);
      recognition.onend = () => setIsListening(false);
      recognition.onresult = (event: any) => {
        const transcript = event.results[0][0].transcript;
        setSymptoms(prev => prev + (prev ? ' ' : '') + transcript);
      };

      recognition.start();
    }
  };

  return (
    <div className="min-h-screen py-8 px-4 sm:px-6 lg:px-8">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-12"
        >
          <div className="bg-gradient-to-r from-blue-600 to-purple-600 w-16 h-16 rounded-2xl flex items-center justify-center mx-auto mb-4">
            <Brain className="w-8 h-8 text-white" />
          </div>
          <h1 className="text-3xl font-bold text-gray-900 mb-2">AI Diagnosis Assistant</h1>
          <p className="text-gray-600">Describe your symptoms and get instant AI-powered medical insights</p>
        </motion.div>

        {/* Symptom Input Form */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="bg-white rounded-2xl shadow-lg p-8 mb-8"
        >
          <form onSubmit={handleSubmit} className="space-y-6">
            <div>
              <label htmlFor="symptoms" className="block text-sm font-medium text-gray-700 mb-2">
                Describe Your Symptoms
              </label>
              <div className="relative">
                <textarea
                  id="symptoms"
                  value={symptoms}
                  onChange={(e) => setSymptoms(e.target.value)}
                  placeholder="Please describe your symptoms in detail (e.g., headache for 2 days, fever, body aches...)"
                  className="w-full h-32 px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 resize-none"
                  disabled={isAnalyzing}
                />
                <button
                  type="button"
                  onClick={handleVoiceInput}
                  className={`absolute right-3 top-3 p-2 rounded-lg transition-all duration-200 ${
                    isListening 
                      ? 'bg-red-100 text-red-600' 
                      : 'bg-gray-100 text-gray-600 hover:bg-blue-100 hover:text-blue-600'
                  }`}
                >
                  <Mic className="w-5 h-5" />
                </button>
              </div>
            </div>

            {/* Common Symptoms */}
            <div>
              <p className="text-sm font-medium text-gray-700 mb-3">Common Symptoms:</p>
              <div className="flex flex-wrap gap-2">
                {commonSymptoms.map((symptom, index) => (
                  <button
                    key={index}
                    type="button"
                    onClick={() => setSymptoms(symptom)}
                    className="px-3 py-1 text-sm bg-blue-50 text-blue-700 rounded-full hover:bg-blue-100 transition-all duration-200"
                    disabled={isAnalyzing}
                  >
                    {symptom}
                  </button>
                ))}
              </div>
            </div>

            <button
              type="submit"
              disabled={!symptoms.trim() || isAnalyzing}
              className="w-full bg-gradient-to-r from-blue-600 to-purple-600 text-white py-4 px-6 rounded-xl font-semibold text-lg disabled:opacity-50 disabled:cursor-not-allowed hover:shadow-lg transition-all duration-200 flex items-center justify-center space-x-2"
            >
              {isAnalyzing ? (
                <>
                  <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
                  <span>Analyzing Symptoms...</span>
                </>
              ) : (
                <>
                  <Send className="w-5 h-5" />
                  <span>Get AI Diagnosis</span>
                </>
              )}
            </button>
          </form>
        </motion.div>

        {/* Analysis Results */}
        {diagnosis && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="space-y-6"
          >
            {/* Primary Diagnosis */}
            <div className="bg-white rounded-2xl shadow-lg p-8">
              <div className="flex items-start space-x-4 mb-6">
                <div className="bg-green-100 p-3 rounded-xl">
                  <CheckCircle className="w-6 h-6 text-green-600" />
                </div>
                <div className="flex-1">
                  <h2 className="text-xl font-bold text-gray-900 mb-2">Primary Diagnosis</h2>
                  <p className="text-2xl font-bold text-green-600 mb-2">{diagnosis.primaryDiagnosis}</p>
                  <div className="flex items-center space-x-2">
                    <span className="text-sm text-gray-600">Confidence Level:</span>
                    <div className="bg-gray-200 rounded-full h-2 w-32">
                      <div 
                        className="bg-green-500 h-2 rounded-full transition-all duration-500"
                        style={{ width: `${diagnosis.confidence * 100}%` }}
                      ></div>
                    </div>
                    <span className="text-sm font-medium">{Math.round(diagnosis.confidence * 100)}%</span>
                  </div>
                </div>
              </div>

              {/* Urgency Level */}
              <div className={`p-4 rounded-xl mb-6 ${
                diagnosis.urgency === 'High' ? 'bg-red-50 border border-red-200' :
                diagnosis.urgency === 'Medium' ? 'bg-yellow-50 border border-yellow-200' :
                'bg-blue-50 border border-blue-200'
              }`}>
                <div className="flex items-center space-x-2">
                  {diagnosis.urgency === 'High' && <AlertCircle className="w-5 h-5 text-red-600" />}
                  {diagnosis.urgency === 'Medium' && <Clock className="w-5 h-5 text-yellow-600" />}
                  {diagnosis.urgency === 'Low' && <CheckCircle className="w-5 h-5 text-blue-600" />}
                  <span className={`font-semibold ${
                    diagnosis.urgency === 'High' ? 'text-red-600' :
                    diagnosis.urgency === 'Medium' ? 'text-yellow-600' :
                    'text-blue-600'
                  }`}>
                    {diagnosis.urgency} Urgency
                  </span>
                </div>
              </div>
            </div>

            {/* Alternative Diagnoses */}
            <div className="bg-white rounded-2xl shadow-lg p-8">
              <h3 className="text-xl font-bold text-gray-900 mb-6 flex items-center space-x-2">
                <TrendingUp className="w-5 h-5" />
                <span>Alternative Possibilities</span>
              </h3>
              <div className="space-y-4">
                {diagnosis.alternativeDiagnoses.map((alt: any, index: number) => (
                  <div key={index} className="flex items-center justify-between p-4 bg-gray-50 rounded-xl">
                    <span className="font-medium text-gray-900">{alt.condition}</span>
                    <div className="flex items-center space-x-2">
                      <div className="bg-gray-200 rounded-full h-2 w-24">
                        <div 
                          className="bg-blue-500 h-2 rounded-full"
                          style={{ width: `${alt.probability * 100}%` }}
                        ></div>
                      </div>
                      <span className="text-sm font-medium">{Math.round(alt.probability * 100)}%</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Recommended Actions */}
            <div className="bg-white rounded-2xl shadow-lg p-8">
              <h3 className="text-xl font-bold text-gray-900 mb-6 flex items-center space-x-2">
                <FileText className="w-5 h-5" />
                <span>Recommended Actions</span>
              </h3>
              <div className="space-y-3">
                {diagnosis.recommendedActions.map((action: string, index: number) => (
                  <div key={index} className="flex items-start space-x-3">
                    <div className="bg-blue-100 rounded-full p-1 mt-1">
                      <div className="w-2 h-2 bg-blue-600 rounded-full"></div>
                    </div>
                    <span className="text-gray-700">{action}</span>
                  </div>
                ))}
              </div>
              
              <div className="mt-6 p-4 bg-yellow-50 border border-yellow-200 rounded-xl">
                <p className="text-sm text-yellow-800">
                  <strong>Follow-up:</strong> {diagnosis.followUp}
                </p>
              </div>
            </div>
          </motion.div>
        )}

        {/* Disclaimer */}
        <div className="mt-8 p-4 bg-gray-50 rounded-xl">
          <p className="text-sm text-gray-600 text-center">
            <strong>Medical Disclaimer:</strong> This AI diagnosis is for informational purposes only and should not replace professional medical advice. 
            Please consult with a healthcare provider for accurate diagnosis and treatment.
          </p>
        </div>
      </div>
    </div>
  );
};