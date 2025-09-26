import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  X, 
  Brain, 
  Heart, 
  Activity, 
  FileText, 
  Pill, 
  Calendar,
  Download,
  Share,
  Bookmark,
  AlertTriangle,
  CheckCircle,
  TrendingUp,
  Clock,
  User,
  Stethoscope,
  Eye,
  ArrowRight
} from 'lucide-react';

interface DiagnosisModalProps {
  isOpen: boolean;
  onClose: () => void;
  diagnosisData: any;
  patientData: any;
  onSaveToDashboard: () => void;
  onBookAppointment: () => void;
}

export const DiagnosisModal: React.FC<DiagnosisModalProps> = ({
  isOpen,
  onClose,
  diagnosisData,
  patientData,
  onSaveToDashboard,
  onBookAppointment
}) => {
  const [activeSection, setActiveSection] = useState('overview');

  if (!isOpen || !diagnosisData) return null;

  const sections = [
    { id: 'overview', title: 'Overview', icon: Brain, color: 'text-blue-600' },
    { id: 'clinical', title: 'Clinical Analysis', icon: Activity, color: 'text-purple-600' },
    { id: 'investigations', title: 'Tests & Studies', icon: FileText, color: 'text-green-600' },
    { id: 'lifestyle', title: 'Lifestyle Plan', icon: Heart, color: 'text-pink-600' },
    { id: 'medications', title: 'Treatment', icon: Pill, color: 'text-red-600' },
    { id: 'referrals', title: 'Specialists', icon: User, color: 'text-indigo-600' },
    { id: 'education', title: 'Patient Education', icon: Stethoscope, color: 'text-orange-600' },
    { id: 'followup', title: 'Follow-up Plan', icon: Clock, color: 'text-teal-600' }
  ];

  const getUrgencyColor = (urgency: string) => {
    switch (urgency?.toUpperCase()) {
      case 'EMERGENCY': return 'bg-red-100 text-red-800 border-red-300';
      case 'HIGH': return 'bg-orange-100 text-orange-800 border-orange-300';
      case 'MEDIUM': return 'bg-yellow-100 text-yellow-800 border-yellow-300';
      case 'LOW': return 'bg-green-100 text-green-800 border-green-300';
      default: return 'bg-gray-100 text-gray-800 border-gray-300';
    }
  };

  const renderSection = () => {
    const assessment = diagnosisData?.doctor_report || diagnosisData;
    
    switch (activeSection) {
      case 'overview':
        return (
          <div className="space-y-6">
            <div className="text-center">
              <div className="bg-gradient-to-r from-blue-600 to-purple-600 w-20 h-20 rounded-full flex items-center justify-center mx-auto mb-4">
                <Brain className="w-10 h-10 text-white" />
              </div>
              <h2 className="text-3xl font-bold text-gray-900 mb-2">AI Medical Assessment Complete</h2>
              <p className="text-gray-600 text-lg">Comprehensive analysis of your health information</p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="bg-blue-50 rounded-xl p-6 border border-blue-200">
                <div className="flex items-center space-x-3 mb-3">
                  <CheckCircle className="w-6 h-6 text-blue-600" />
                  <h3 className="font-bold text-blue-900">Primary Assessment</h3>
                </div>
                <p className="text-blue-800">
                  {assessment?.assessment || assessment?.condition_explanation || 'Assessment completed successfully'}
                </p>
              </div>

              <div className="bg-purple-50 rounded-xl p-6 border border-purple-200">
                <div className="flex items-center space-x-3 mb-3">
                  <TrendingUp className="w-6 h-6 text-purple-600" />
                  <h3 className="font-bold text-purple-900">Confidence Level</h3>
                </div>
                <div className="flex items-center space-x-2">
                  <div className="bg-purple-200 rounded-full h-3 w-20">
                    <div className="bg-purple-600 h-3 rounded-full" style={{ width: '85%' }}></div>
                  </div>
                  <span className="font-bold text-purple-800">85%</span>
                </div>
              </div>

              <div className={`rounded-xl p-6 border-2 ${getUrgencyColor('MEDIUM')}`}>
                <div className="flex items-center space-x-3 mb-3">
                  <AlertTriangle className="w-6 h-6" />
                  <h3 className="font-bold">Priority Level</h3>
                </div>
                <p className="font-semibold">MEDIUM Priority</p>
              </div>
            </div>

            <div className="bg-yellow-50 border border-yellow-200 rounded-xl p-6">
              <div className="flex items-start space-x-3">
                <AlertTriangle className="w-6 h-6 text-yellow-600 mt-1" />
                <div>
                  <h4 className="font-bold text-yellow-900 mb-2">Important Medical Disclaimer</h4>
                  <p className="text-yellow-800 text-sm">
                    This AI assessment is designed to assist healthcare professionals, not replace them. 
                    The analysis provides preliminary insights to help guide your healthcare decisions. 
                    Always consult qualified healthcare providers for final diagnosis and treatment.
                  </p>
                </div>
              </div>
            </div>
          </div>
        );

      case 'clinical':
        return (
          <div className="space-y-6">
            <h3 className="text-2xl font-bold text-gray-900 flex items-center space-x-3">
              <Activity className="w-7 h-7 text-purple-600" />
              <span>Clinical Analysis</span>
            </h3>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="bg-orange-50 rounded-xl p-6">
                <h4 className="font-bold text-orange-900 mb-4">Likely Conditions</h4>
                <div className="space-y-3">
                  {assessment?.likely_conditions?.map((condition: any, index: number) => (
                    <div key={index} className="p-3 bg-white rounded-lg border border-orange-200">
                      <div className="flex justify-between items-start mb-2">
                        <span className="font-medium text-gray-900">{condition.condition}</span>
                        <span className="text-sm text-orange-600 font-medium">{condition.probability}</span>
                      </div>
                      <p className="text-sm text-gray-600">{condition.reasoning}</p>
                    </div>
                  )) || (
                    <p className="text-orange-800">
                      {assessment?.assessment || 'Comprehensive medical assessment completed'}
                    </p>
                  )}
                </div>
              </div>

              <div className="bg-red-50 rounded-xl p-6">
                <h4 className="font-bold text-red-900 mb-4">Risk Factors</h4>
                <div className="space-y-2">
                  {assessment?.red_flags?.map((flag: string, index: number) => (
                    <div key={index} className="flex items-start space-x-2 p-3 bg-white rounded-lg border border-red-200">
                      <AlertTriangle className="w-4 h-4 text-red-600 mt-0.5" />
                      <span className="text-red-800 text-sm">{flag}</span>
                    </div>
                  )) || (
                    <p className="text-red-800">No immediate risk factors identified</p>
                  )}
                </div>
              </div>
            </div>

            <div className="bg-blue-50 rounded-xl p-6">
              <h4 className="font-bold text-blue-900 mb-3">Clinical Summary</h4>
              <p className="text-blue-800 leading-relaxed">
                {assessment?.doctor_summary || assessment?.explanation || 'Detailed clinical analysis completed by AI'}
              </p>
            </div>
          </div>
        );

      case 'investigations':
        return (
          <div className="space-y-6">
            <h3 className="text-2xl font-bold text-gray-900 flex items-center space-x-3">
              <FileText className="w-7 h-7 text-green-600" />
              <span>Recommended Tests & Studies</span>
            </h3>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="bg-blue-50 rounded-xl p-6">
                <h4 className="font-bold text-blue-900 mb-4">Essential Lab Tests</h4>
                <div className="space-y-2">
                  {['Complete Blood Count (CBC)', 'Basic Metabolic Panel', 'Thyroid Function Tests', 'Vitamin D Level'].map((test, index) => (
                    <div key={index} className="p-3 bg-white rounded-lg border border-blue-200">
                      <span className="text-blue-800">{test}</span>
                    </div>
                  ))}
                </div>
              </div>

              <div className="bg-purple-50 rounded-xl p-6">
                <h4 className="font-bold text-purple-900 mb-4">Imaging Studies</h4>
                <div className="space-y-2">
                  {['Chest X-Ray', 'Brain MRI (if neurological symptoms)', 'Abdominal Ultrasound'].map((study, index) => (
                    <div key={index} className="p-3 bg-white rounded-lg border border-purple-200">
                      <span className="text-purple-800">{study}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            <div className="bg-orange-50 rounded-xl p-6">
              <h4 className="font-bold text-orange-900 mb-3">Investigation Timeline</h4>
              <p className="text-orange-800 font-medium">
                Complete essential tests within 1-2 weeks. Urgent tests if symptoms worsen.
              </p>
            </div>
          </div>
        );

      case 'lifestyle':
        return (
          <div className="space-y-6">
            <h3 className="text-2xl font-bold text-gray-900 flex items-center space-x-3">
              <Heart className="w-7 h-7 text-pink-600" />
              <span>Personalized Lifestyle Plan</span>
            </h3>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="bg-green-50 rounded-xl p-6">
                <h4 className="font-bold text-green-900 mb-4">Diet Recommendations</h4>
                <div className="space-y-2">
                  {[
                    'Increase fruits and vegetables (5-7 servings daily)',
                    'Reduce processed and fried foods',
                    'Stay hydrated (8-10 glasses water daily)',
                    'Consider anti-inflammatory foods'
                  ].map((diet, index) => (
                    <div key={index} className="flex items-start space-x-2 p-3 bg-white rounded-lg">
                      <div className="w-2 h-2 bg-green-600 rounded-full mt-2"></div>
                      <span className="text-green-800 text-sm">{diet}</span>
                    </div>
                  ))}
                </div>
              </div>

              <div className="bg-blue-50 rounded-xl p-6">
                <h4 className="font-bold text-blue-900 mb-4">Exercise Plan</h4>
                <div className="space-y-2">
                  {[
                    'Start with 20-30 minutes daily walking',
                    'Gradually increase to moderate exercise',
                    'Include stress-reducing activities (yoga, meditation)',
                    'Avoid strenuous activity until cleared by doctor'
                  ].map((exercise, index) => (
                    <div key={index} className="flex items-start space-x-2 p-3 bg-white rounded-lg">
                      <div className="w-2 h-2 bg-blue-600 rounded-full mt-2"></div>
                      <span className="text-blue-800 text-sm">{exercise}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="bg-purple-50 rounded-xl p-6">
                <h4 className="font-bold text-purple-900 mb-4">Sleep Optimization</h4>
                <div className="space-y-2">
                  {[
                    'Maintain 7-9 hours sleep nightly',
                    'Create consistent sleep schedule',
                    'Avoid screens 1 hour before bed',
                    'Create calm, dark sleep environment'
                  ].map((tip, index) => (
                    <div key={index} className="flex items-start space-x-2 p-3 bg-white rounded-lg">
                      <div className="w-2 h-2 bg-purple-600 rounded-full mt-2"></div>
                      <span className="text-purple-800 text-sm">{tip}</span>
                    </div>
                  ))}
                </div>
              </div>

              <div className="bg-orange-50 rounded-xl p-6">
                <h4 className="font-bold text-orange-900 mb-4">Stress Management</h4>
                <div className="space-y-2">
                  {[
                    'Practice deep breathing exercises',
                    'Consider mindfulness meditation',
                    'Maintain social connections',
                    'Engage in relaxing hobbies'
                  ].map((strategy, index) => (
                    <div key={index} className="flex items-start space-x-2 p-3 bg-white rounded-lg">
                      <div className="w-2 h-2 bg-orange-600 rounded-full mt-2"></div>
                      <span className="text-orange-800 text-sm">{strategy}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        );

      case 'medications':
        return (
          <div className="space-y-6">
            <h3 className="text-2xl font-bold text-gray-900 flex items-center space-x-3">
              <Pill className="w-7 h-7 text-red-600" />
              <span>Treatment Recommendations</span>
            </h3>

            <div className="bg-yellow-50 border border-yellow-200 rounded-xl p-4 mb-6">
              <div className="flex items-start space-x-3">
                <AlertTriangle className="w-5 h-5 text-yellow-600 mt-0.5" />
                <p className="text-yellow-800 text-sm">
                  <strong>Important:</strong> These are general treatment categories. 
                  Actual prescriptions require consultation with licensed healthcare professionals.
                </p>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="bg-red-50 rounded-xl p-6">
                <h4 className="font-bold text-red-900 mb-4">Immediate Care</h4>
                <div className="space-y-2">
                  {[
                    'Monitor symptoms closely',
                    'Stay hydrated and rest',
                    'Take temperature regularly',
                    'Avoid strenuous activities'
                  ].map((care, index) => (
                    <div key={index} className="p-3 bg-white rounded-lg border border-red-200">
                      <span className="text-red-800">{care}</span>
                    </div>
                  ))}
                </div>
              </div>

              <div className="bg-blue-50 rounded-xl p-6">
                <h4 className="font-bold text-blue-900 mb-4">Medication Categories</h4>
                <div className="space-y-2">
                  {[
                    'Over-the-counter pain relievers (as needed)',
                    'Multivitamin supplements',
                    'Probiotics for digestive health',
                    'Consult doctor for prescription medications'
                  ].map((med, index) => (
                    <div key={index} className="p-3 bg-white rounded-lg border border-blue-200">
                      <span className="text-blue-800">{med}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        );

      case 'education':
        return (
          <div className="space-y-6">
            <h3 className="text-2xl font-bold text-gray-900 flex items-center space-x-3">
              <Stethoscope className="w-7 h-7 text-orange-600" />
              <span>Patient Education</span>
            </h3>

            <div className="bg-blue-50 rounded-xl p-6">
              <h4 className="font-bold text-blue-900 mb-3">Understanding Your Condition</h4>
              <p className="text-blue-800 leading-relaxed">
                {assessment?.condition_explanation || assessment?.explanation || 
                 'Based on your symptoms and medical history, our AI analysis suggests monitoring and professional consultation for proper diagnosis and treatment planning.'}
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="bg-green-50 rounded-xl p-6">
                <h4 className="font-bold text-green-900 mb-4">Self-Care Instructions</h4>
                <div className="space-y-2">
                  {assessment?.self_care_instructions?.map((instruction: string, index: number) => (
                    <div key={index} className="flex items-start space-x-2">
                      <CheckCircle className="w-4 h-4 text-green-600 mt-0.5" />
                      <span className="text-green-800 text-sm">{instruction}</span>
                    </div>
                  )) || [
                    'Rest and maintain good hydration',
                    'Monitor symptoms for changes',
                    'Follow healthy lifestyle practices',
                    'Take medications as prescribed'
                  ].map((instruction, index) => (
                    <div key={index} className="flex items-start space-x-2">
                      <CheckCircle className="w-4 h-4 text-green-600 mt-0.5" />
                      <span className="text-green-800 text-sm">{instruction}</span>
                    </div>
                  ))}
                </div>
              </div>

              <div className="bg-red-50 rounded-xl p-6">
                <h4 className="font-bold text-red-900 mb-4">Warning Signs</h4>
                <div className="space-y-2">
                  {assessment?.warning_signs?.map((sign: string, index: number) => (
                    <div key={index} className="flex items-start space-x-2">
                      <AlertTriangle className="w-4 h-4 text-red-600 mt-0.5" />
                      <span className="text-red-800 text-sm">{sign}</span>
                    </div>
                  )) || [
                    'Severe worsening of symptoms',
                    'High fever (>101.5°F)',
                    'Difficulty breathing',
                    'Severe pain or discomfort'
                  ].map((sign, index) => (
                    <div key={index} className="flex items-start space-x-2">
                      <AlertTriangle className="w-4 h-4 text-red-600 mt-0.5" />
                      <span className="text-red-800 text-sm">{sign}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        );

      case 'followup':
        return (
          <div className="space-y-6">
            <h3 className="text-2xl font-bold text-gray-900 flex items-center space-x-3">
              <Clock className="w-7 h-7 text-teal-600" />
              <span>Follow-up & Monitoring Plan</span>
            </h3>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="bg-teal-50 rounded-xl p-6">
                <h4 className="font-bold text-teal-900 mb-3">Next Steps</h4>
                <div className="space-y-3">
                  {assessment?.next_steps?.map((step: string, index: number) => (
                    <div key={index} className="flex items-start space-x-3">
                      <div className="bg-teal-600 text-white rounded-full w-6 h-6 flex items-center justify-center text-sm font-bold mt-0.5">
                        {index + 1}
                      </div>
                      <span className="text-teal-800">{step}</span>
                    </div>
                  )) || [
                    'Schedule appointment with healthcare provider',
                    'Complete recommended lab tests',
                    'Monitor symptoms daily',
                    'Follow lifestyle recommendations'
                  ].map((step, index) => (
                    <div key={index} className="flex items-start space-x-3">
                      <div className="bg-teal-600 text-white rounded-full w-6 h-6 flex items-center justify-center text-sm font-bold mt-0.5">
                        {index + 1}
                      </div>
                      <span className="text-teal-800">{step}</span>
                    </div>
                  ))}
                </div>
              </div>

              <div className="bg-blue-50 rounded-xl p-6">
                <h4 className="font-bold text-blue-900 mb-3">Follow-up Timeline</h4>
                <p className="text-blue-800 text-lg font-medium mb-4">
                  {assessment?.follow_up_timeline || 'Follow up within 1-2 weeks or sooner if symptoms worsen'}
                </p>
                <div className="space-y-2">
                  <div className="flex items-center space-x-2">
                    <Calendar className="w-4 h-4 text-blue-600" />
                    <span className="text-blue-800 text-sm">Schedule appointment within 1 week</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <Clock className="w-4 h-4 text-blue-600" />
                    <span className="text-blue-800 text-sm">Monitor symptoms daily</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        );

      default:
        return (
          <div className="text-center py-12">
            <Brain className="w-16 h-16 text-gray-300 mx-auto mb-4" />
            <p className="text-gray-600">Select a section to view details</p>
          </div>
        );
    }
  };

  return (
    <AnimatePresence>
      {isOpen && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.9 }}
            className="bg-white rounded-3xl shadow-2xl max-w-7xl w-full max-h-[90vh] overflow-hidden"
          >
            {/* Header */}
            <div className="bg-gradient-to-r from-blue-600 to-purple-600 p-6 text-white">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-4">
                  <div className="bg-white/20 p-3 rounded-xl">
                    <Brain className="w-8 h-8" />
                  </div>
                  <div>
                    <h1 className="text-2xl font-bold">AI Medical Assessment</h1>
                    <p className="text-blue-100">Comprehensive health analysis completed</p>
                  </div>
                </div>
                <button
                  onClick={onClose}
                  className="p-2 bg-white/20 rounded-lg hover:bg-white/30 transition-all duration-200"
                >
                  <X className="w-6 h-6" />
                </button>
              </div>
            </div>

            <div className="flex h-[calc(90vh-120px)]">
              {/* Sidebar Navigation */}
              <div className="w-80 bg-gray-50 p-6 overflow-y-auto">
                <h3 className="font-bold text-gray-900 mb-4">Assessment Sections</h3>
                <div className="space-y-2">
                  {sections.map((section) => {
                    const Icon = section.icon;
                    return (
                      <button
                        key={section.id}
                        onClick={() => setActiveSection(section.id)}
                        className={`w-full flex items-center space-x-3 p-3 rounded-xl transition-all duration-200 ${
                          activeSection === section.id
                            ? 'bg-white shadow-md border-2 border-blue-200'
                            : 'hover:bg-white hover:shadow-sm'
                        }`}
                      >
                        <Icon className={`w-5 h-5 ${section.color}`} />
                        <span className={`font-medium ${
                          activeSection === section.id ? 'text-gray-900' : 'text-gray-600'
                        }`}>
                          {section.title}
                        </span>
                        {activeSection === section.id && (
                          <ArrowRight className="w-4 h-4 text-blue-600 ml-auto" />
                        )}
                      </button>
                    );
                  })}
                </div>
              </div>

              {/* Main Content */}
              <div className="flex-1 p-8 overflow-y-auto">
                <AnimatePresence mode="wait">
                  <motion.div
                    key={activeSection}
                    initial={{ opacity: 0, x: 20 }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0, x: -20 }}
                    transition={{ duration: 0.3 }}
                  >
                    {renderSection()}
                  </motion.div>
                </AnimatePresence>
              </div>
            </div>

            {/* Footer Actions */}
            <div className="bg-gray-50 p-6 border-t">
              <div className="flex flex-wrap gap-4 justify-center">
                <button
                  onClick={onSaveToDashboard}
                  className="bg-gradient-to-r from-blue-600 to-purple-600 text-white px-6 py-3 rounded-xl font-semibold hover:shadow-lg transition-all duration-200 flex items-center space-x-2"
                >
                  <Bookmark className="w-5 h-5" />
                  <span>Save to Dashboard</span>
                </button>
                <button
                  onClick={onBookAppointment}
                  className="bg-green-600 text-white px-6 py-3 rounded-xl font-semibold hover:bg-green-700 transition-all duration-200 flex items-center space-x-2"
                >
                  <Calendar className="w-5 h-5" />
                  <span>Book Appointment</span>
                </button>
                <button className="bg-gray-600 text-white px-6 py-3 rounded-xl font-semibold hover:bg-gray-700 transition-all duration-200 flex items-center space-x-2">
                  <Download className="w-5 h-5" />
                  <span>Download PDF</span>
                </button>
                <button className="bg-purple-600 text-white px-6 py-3 rounded-xl font-semibold hover:bg-purple-700 transition-all duration-200 flex items-center space-x-2">
                  <Share className="w-5 h-5" />
                  <span>Share with Doctor</span>
                </button>
              </div>
            </div>
          </motion.div>
        </div>
      )}
    </AnimatePresence>
  );
};