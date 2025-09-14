import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  X, 
  Brain, 
  Heart, 
  AlertTriangle, 
  CheckCircle, 
  Clock,
  FileText,
  TrendingUp,
  Pill,
  Activity,
  Calendar,
  Download,
  Share,
  Bookmark
} from 'lucide-react';

interface DiagnosisData {
  preliminary_assessment: {
    primary_diagnosis: string;
    confidence_score: number;
    urgency_level: string;
    mental_health_risk: string;
    overall_health_status: string;
  };
  clinical_analysis: {
    probable_causes: string[];
    differential_diagnosis: string[];
    risk_factors: string[];
    symptom_pattern: string;
  };
  recommended_investigations: {
    essential_tests: string[];
    imaging_studies: string[];
    specialist_consultations: string[];
    monitoring_parameters: string[];
  };
  lifestyle_recommendations: {
    diet_modifications: string[];
    exercise_plan: string[];
    sleep_hygiene: string[];
    stress_management: string[];
    preventive_measures: string[];
  };
  medication_guidance: {
    otc_recommendations: string[];
    prescription_categories: string[];
    drug_interactions: string[];
    dosage_considerations: string[];
  };
  referral_recommendations: {
    specialist_needed: string[];
    urgency_timeline: string;
    preparation_notes: string[];
  };
  patient_education: {
    condition_explanation: string;
    warning_signs: string[];
    self_care_strategies: string[];
    prognosis: string;
  };
  follow_up_plan: {
    reassessment_timeline: string;
    symptom_tracking: string[];
    progress_indicators: string[];
    next_steps: string[];
  };
}

interface DiagnosisModalProps {
  isOpen: boolean;
  onClose: () => void;
  diagnosisData: DiagnosisData | null;
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
  const [isProcessing, setIsProcessing] = useState(false);

  if (!diagnosisData) return null;

  const getUrgencyColor = (urgency: string) => {
    switch (urgency?.toUpperCase()) {
      case 'EMERGENCY': return 'bg-red-100 text-red-800 border-red-300';
      case 'HIGH': return 'bg-orange-100 text-orange-800 border-orange-300';
      case 'MEDIUM': return 'bg-yellow-100 text-yellow-800 border-yellow-300';
      case 'LOW': return 'bg-green-100 text-green-800 border-green-300';
      default: return 'bg-gray-100 text-gray-800 border-gray-300';
    }
  };

  const getUrgencyIcon = (urgency: string) => {
    switch (urgency?.toUpperCase()) {
      case 'EMERGENCY': return AlertTriangle;
      case 'HIGH': return AlertTriangle;
      case 'MEDIUM': return Clock;
      case 'LOW': return CheckCircle;
      default: return Clock;
    }
  };

  const sections = [
    { id: 'overview', label: 'Overview', icon: Brain },
    { id: 'clinical', label: 'Clinical Analysis', icon: Activity },
    { id: 'investigations', label: 'Tests & Imaging', icon: FileText },
    { id: 'lifestyle', label: 'Lifestyle Plan', icon: Heart },
    { id: 'medications', label: 'Medications', icon: Pill },
    { id: 'referrals', label: 'Specialist Care', icon: TrendingUp },
    { id: 'education', label: 'Patient Guide', icon: CheckCircle },
    { id: 'followup', label: 'Follow-up Plan', icon: Calendar }
  ];

  const handleSaveReport = async () => {
    setIsProcessing(true);
    try {
      // Save diagnosis to patient dashboard
      await onSaveToDashboard();
      
      // Show success message
      setTimeout(() => {
        setIsProcessing(false);
        onClose();
      }, 1500);
    } catch (error) {
      console.error('Save failed:', error);
      setIsProcessing(false);
    }
  };

  return (
    <AnimatePresence>
      {isOpen && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
          <motion.div
            initial={{ opacity: 0, scale: 0.9, y: 20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.9, y: 20 }}
            className="bg-white rounded-3xl shadow-2xl max-w-6xl w-full max-h-[95vh] overflow-hidden"
          >
            {/* Header */}
            <div className="bg-gradient-to-r from-blue-600 to-purple-600 p-6 text-white">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-4">
                  <div className="bg-white/20 p-3 rounded-xl">
                    <Brain className="w-8 h-8" />
                  </div>
                  <div>
                    <h2 className="text-2xl font-bold">AI Medical Assessment</h2>
                    <p className="text-blue-100">Comprehensive health analysis completed</p>
                  </div>
                </div>
                <button
                  onClick={onClose}
                  className="p-2 hover:bg-white/20 rounded-lg transition-all duration-200"
                >
                  <X className="w-6 h-6" />
                </button>
              </div>

              {/* Primary Diagnosis Banner */}
              <div className="mt-6 bg-white/10 rounded-2xl p-6">
                <div className="flex items-center justify-between">
                  <div>
                    <h3 className="text-xl font-bold mb-2">Primary Diagnosis</h3>
                    <p className="text-2xl font-bold text-white">
                      {diagnosisData.preliminary_assessment.primary_diagnosis}
                    </p>
                  </div>
                  <div className="text-right">
                    <div className="flex items-center space-x-2 mb-2">
                      <span className="text-sm">Confidence:</span>
                      <div className="bg-white/20 rounded-full h-2 w-24">
                        <div 
                          className="bg-white h-2 rounded-full transition-all duration-1000"
                          style={{ width: `${(diagnosisData.preliminary_assessment.confidence_score || 0) * 100}%` }}
                        />
                      </div>
                      <span className="font-bold">
                        {Math.round((diagnosisData.preliminary_assessment.confidence_score || 0) * 100)}%
                      </span>
                    </div>
                    <div className={`px-3 py-1 rounded-full text-sm font-medium border ${getUrgencyColor(diagnosisData.preliminary_assessment.urgency_level)}`}>
                      {diagnosisData.preliminary_assessment.urgency_level} Priority
                    </div>
                  </div>
                </div>
              </div>
            </div>

            <div className="flex h-[calc(95vh-200px)]">
              {/* Sidebar Navigation */}
              <div className="w-64 bg-gray-50 p-4 overflow-y-auto">
                <nav className="space-y-2">
                  {sections.map((section) => {
                    const Icon = section.icon;
                    return (
                      <button
                        key={section.id}
                        onClick={() => setActiveSection(section.id)}
                        className={`w-full flex items-center space-x-3 px-4 py-3 rounded-xl text-left transition-all duration-200 ${
                          activeSection === section.id
                            ? 'bg-blue-600 text-white shadow-lg'
                            : 'text-gray-700 hover:bg-white hover:shadow-md'
                        }`}
                      >
                        <Icon className="w-5 h-5" />
                        <span className="font-medium">{section.label}</span>
                      </button>
                    );
                  })}
                </nav>
              </div>

              {/* Main Content */}
              <div className="flex-1 p-6 overflow-y-auto">
                <AnimatePresence mode="wait">
                  <motion.div
                    key={activeSection}
                    initial={{ opacity: 0, x: 20 }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0, x: -20 }}
                    transition={{ duration: 0.3 }}
                  >
                    {/* Overview Section */}
                    {activeSection === 'overview' && (
                      <div className="space-y-6">
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                          <div className="bg-blue-50 rounded-2xl p-6 border border-blue-200">
                            <h3 className="text-lg font-bold text-blue-900 mb-3">Health Status</h3>
                            <p className="text-blue-800">{diagnosisData.preliminary_assessment.overall_health_status}</p>
                          </div>
                          <div className="bg-purple-50 rounded-2xl p-6 border border-purple-200">
                            <h3 className="text-lg font-bold text-purple-900 mb-3">Mental Health</h3>
                            <p className="text-purple-800">{diagnosisData.preliminary_assessment.mental_health_risk}</p>
                          </div>
                        </div>

                        <div className="bg-gray-50 rounded-2xl p-6">
                          <h3 className="text-lg font-bold text-gray-900 mb-4">Alternative Diagnoses</h3>
                          <div className="space-y-3">
                            {diagnosisData.clinical_analysis.differential_diagnosis.map((diagnosis, index) => (
                              <div key={index} className="flex items-center space-x-3 p-3 bg-white rounded-xl">
                                <div className="w-2 h-2 bg-blue-600 rounded-full"></div>
                                <span className="text-gray-700">{diagnosis}</span>
                              </div>
                            ))}
                          </div>
                        </div>
                      </div>
                    )}

                    {/* Clinical Analysis Section */}
                    {activeSection === 'clinical' && (
                      <div className="space-y-6">
                        <div className="bg-white rounded-2xl p-6 border border-gray-200">
                          <h3 className="text-lg font-bold text-gray-900 mb-4">Probable Causes</h3>
                          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                            {diagnosisData.clinical_analysis.probable_causes.map((cause, index) => (
                              <div key={index} className="p-4 bg-orange-50 rounded-xl border border-orange-200">
                                <p className="text-orange-800">{cause}</p>
                              </div>
                            ))}
                          </div>
                        </div>

                        <div className="bg-white rounded-2xl p-6 border border-gray-200">
                          <h3 className="text-lg font-bold text-gray-900 mb-4">Risk Factors</h3>
                          <div className="space-y-3">
                            {diagnosisData.clinical_analysis.risk_factors.map((factor, index) => (
                              <div key={index} className="flex items-start space-x-3 p-3 bg-red-50 rounded-xl">
                                <AlertTriangle className="w-5 h-5 text-red-600 mt-0.5" />
                                <span className="text-red-800">{factor}</span>
                              </div>
                            ))}
                          </div>
                        </div>

                        <div className="bg-white rounded-2xl p-6 border border-gray-200">
                          <h3 className="text-lg font-bold text-gray-900 mb-4">Symptom Pattern Analysis</h3>
                          <p className="text-gray-700 leading-relaxed">{diagnosisData.clinical_analysis.symptom_pattern}</p>
                        </div>
                      </div>
                    )}

                    {/* Investigations Section */}
                    {activeSection === 'investigations' && (
                      <div className="space-y-6">
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                          <div className="bg-white rounded-2xl p-6 border border-gray-200">
                            <h3 className="text-lg font-bold text-gray-900 mb-4 flex items-center space-x-2">
                              <FileText className="w-5 h-5 text-blue-600" />
                              <span>Essential Lab Tests</span>
                            </h3>
                            <div className="space-y-2">
                              {diagnosisData.recommended_investigations.essential_tests.map((test, index) => (
                                <div key={index} className="p-3 bg-blue-50 rounded-lg">
                                  <span className="text-blue-800">{test}</span>
                                </div>
                              ))}
                            </div>
                          </div>

                          <div className="bg-white rounded-2xl p-6 border border-gray-200">
                            <h3 className="text-lg font-bold text-gray-900 mb-4 flex items-center space-x-2">
                              <Activity className="w-5 h-5 text-purple-600" />
                              <span>Imaging Studies</span>
                            </h3>
                            <div className="space-y-2">
                              {diagnosisData.recommended_investigations.imaging_studies.map((study, index) => (
                                <div key={index} className="p-3 bg-purple-50 rounded-lg">
                                  <span className="text-purple-800">{study}</span>
                                </div>
                              ))}
                            </div>
                          </div>
                        </div>

                        <div className="bg-white rounded-2xl p-6 border border-gray-200">
                          <h3 className="text-lg font-bold text-gray-900 mb-4">Monitoring Parameters</h3>
                          <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                            {diagnosisData.recommended_investigations.monitoring_parameters.map((param, index) => (
                              <div key={index} className="p-3 bg-green-50 rounded-lg text-center">
                                <span className="text-green-800 font-medium">{param}</span>
                              </div>
                            ))}
                          </div>
                        </div>
                      </div>
                    )}

                    {/* Lifestyle Section */}
                    {activeSection === 'lifestyle' && (
                      <div className="space-y-6">
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                          <div className="bg-white rounded-2xl p-6 border border-gray-200">
                            <h3 className="text-lg font-bold text-gray-900 mb-4">Diet Modifications</h3>
                            <div className="space-y-3">
                              {diagnosisData.lifestyle_recommendations.diet_modifications.map((diet, index) => (
                                <div key={index} className="flex items-start space-x-3">
                                  <div className="w-2 h-2 bg-green-600 rounded-full mt-2"></div>
                                  <span className="text-gray-700">{diet}</span>
                                </div>
                              ))}
                            </div>
                          </div>

                          <div className="bg-white rounded-2xl p-6 border border-gray-200">
                            <h3 className="text-lg font-bold text-gray-900 mb-4">Exercise Plan</h3>
                            <div className="space-y-3">
                              {diagnosisData.lifestyle_recommendations.exercise_plan.map((exercise, index) => (
                                <div key={index} className="flex items-start space-x-3">
                                  <div className="w-2 h-2 bg-blue-600 rounded-full mt-2"></div>
                                  <span className="text-gray-700">{exercise}</span>
                                </div>
                              ))}
                            </div>
                          </div>
                        </div>

                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                          <div className="bg-white rounded-2xl p-6 border border-gray-200">
                            <h3 className="text-lg font-bold text-gray-900 mb-4">Sleep Hygiene</h3>
                            <div className="space-y-3">
                              {diagnosisData.lifestyle_recommendations.sleep_hygiene.map((tip, index) => (
                                <div key={index} className="flex items-start space-x-3">
                                  <div className="w-2 h-2 bg-purple-600 rounded-full mt-2"></div>
                                  <span className="text-gray-700">{tip}</span>
                                </div>
                              ))}
                            </div>
                          </div>

                          <div className="bg-white rounded-2xl p-6 border border-gray-200">
                            <h3 className="text-lg font-bold text-gray-900 mb-4">Stress Management</h3>
                            <div className="space-y-3">
                              {diagnosisData.lifestyle_recommendations.stress_management.map((strategy, index) => (
                                <div key={index} className="flex items-start space-x-3">
                                  <div className="w-2 h-2 bg-orange-600 rounded-full mt-2"></div>
                                  <span className="text-gray-700">{strategy}</span>
                                </div>
                              ))}
                            </div>
                          </div>
                        </div>
                      </div>
                    )}

                    {/* Medications Section */}
                    {activeSection === 'medications' && (
                      <div className="space-y-6">
                        <div className="bg-yellow-50 border border-yellow-200 rounded-2xl p-6">
                          <div className="flex items-start space-x-3">
                            <AlertTriangle className="w-6 h-6 text-yellow-600 mt-1" />
                            <div>
                              <h4 className="font-bold text-yellow-900 mb-2">Important Medication Notice</h4>
                              <p className="text-yellow-800 text-sm">
                                These are general medication categories. Actual prescriptions require consultation with a licensed physician.
                              </p>
                            </div>
                          </div>
                        </div>

                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                          <div className="bg-white rounded-2xl p-6 border border-gray-200">
                            <h3 className="text-lg font-bold text-gray-900 mb-4">Over-the-Counter Options</h3>
                            <div className="space-y-3">
                              {diagnosisData.medication_guidance.otc_recommendations.map((med, index) => (
                                <div key={index} className="p-3 bg-green-50 rounded-lg">
                                  <span className="text-green-800">{med}</span>
                                </div>
                              ))}
                            </div>
                          </div>

                          <div className="bg-white rounded-2xl p-6 border border-gray-200">
                            <h3 className="text-lg font-bold text-gray-900 mb-4">Prescription Categories</h3>
                            <div className="space-y-3">
                              {diagnosisData.medication_guidance.prescription_categories.map((category, index) => (
                                <div key={index} className="p-3 bg-blue-50 rounded-lg">
                                  <span className="text-blue-800">{category}</span>
                                </div>
                              ))}
                            </div>
                          </div>
                        </div>

                        <div className="bg-white rounded-2xl p-6 border border-gray-200">
                          <h3 className="text-lg font-bold text-gray-900 mb-4">Drug Interactions & Considerations</h3>
                          <div className="space-y-3">
                            {diagnosisData.medication_guidance.drug_interactions.map((interaction, index) => (
                              <div key={index} className="p-3 bg-red-50 rounded-lg border border-red-200">
                                <span className="text-red-800">{interaction}</span>
                              </div>
                            ))}
                          </div>
                        </div>
                      </div>
                    )}

                    {/* Patient Education Section */}
                    {activeSection === 'education' && (
                      <div className="space-y-6">
                        <div className="bg-white rounded-2xl p-6 border border-gray-200">
                          <h3 className="text-lg font-bold text-gray-900 mb-4">Understanding Your Condition</h3>
                          <p className="text-gray-700 leading-relaxed text-lg">
                            {diagnosisData.patient_education.condition_explanation}
                          </p>
                        </div>

                        <div className="bg-red-50 rounded-2xl p-6 border border-red-200">
                          <h3 className="text-lg font-bold text-red-900 mb-4 flex items-center space-x-2">
                            <AlertTriangle className="w-5 h-5" />
                            <span>Warning Signs - Seek Immediate Care</span>
                          </h3>
                          <div className="space-y-2">
                            {diagnosisData.patient_education.warning_signs.map((sign, index) => (
                              <div key={index} className="flex items-start space-x-3">
                                <AlertTriangle className="w-4 h-4 text-red-600 mt-1" />
                                <span className="text-red-800">{sign}</span>
                              </div>
                            ))}
                          </div>
                        </div>

                        <div className="bg-white rounded-2xl p-6 border border-gray-200">
                          <h3 className="text-lg font-bold text-gray-900 mb-4">Self-Care Strategies</h3>
                          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                            {diagnosisData.patient_education.self_care_strategies.map((strategy, index) => (
                              <div key={index} className="p-4 bg-green-50 rounded-xl">
                                <span className="text-green-800">{strategy}</span>
                              </div>
                            ))}
                          </div>
                        </div>

                        <div className="bg-blue-50 rounded-2xl p-6 border border-blue-200">
                          <h3 className="text-lg font-bold text-blue-900 mb-3">Prognosis</h3>
                          <p className="text-blue-800 text-lg">{diagnosisData.patient_education.prognosis}</p>
                        </div>
                      </div>
                    )}

                    {/* Follow-up Plan Section */}
                    {activeSection === 'followup' && (
                      <div className="space-y-6">
                        <div className="bg-white rounded-2xl p-6 border border-gray-200">
                          <h3 className="text-lg font-bold text-gray-900 mb-4">Reassessment Timeline</h3>
                          <div className="p-4 bg-blue-50 rounded-xl border border-blue-200">
                            <p className="text-blue-800 text-lg font-medium">
                              {diagnosisData.follow_up_plan.reassessment_timeline}
                            </p>
                          </div>
                        </div>

                        <div className="bg-white rounded-2xl p-6 border border-gray-200">
                          <h3 className="text-lg font-bold text-gray-900 mb-4">Symptoms to Track</h3>
                          <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                            {diagnosisData.follow_up_plan.symptom_tracking.map((symptom, index) => (
                              <div key={index} className="p-3 bg-yellow-50 rounded-lg text-center">
                                <span className="text-yellow-800 font-medium">{symptom}</span>
                              </div>
                            ))}
                          </div>
                        </div>

                        <div className="bg-white rounded-2xl p-6 border border-gray-200">
                          <h3 className="text-lg font-bold text-gray-900 mb-4">Next Steps</h3>
                          <div className="space-y-3">
                            {diagnosisData.follow_up_plan.next_steps.map((step, index) => (
                              <div key={index} className="flex items-start space-x-3 p-3 bg-green-50 rounded-xl">
                                <div className="bg-green-600 text-white rounded-full w-6 h-6 flex items-center justify-center text-sm font-bold">
                                  {index + 1}
                                </div>
                                <span className="text-green-800">{step}</span>
                              </div>
                            ))}
                          </div>
                        </div>
                      </div>
                    )}
                  </motion.div>
                </AnimatePresence>
              </div>
            </div>

            {/* Footer Actions */}
            <div className="bg-gray-50 p-6 border-t border-gray-200">
              <div className="flex items-center justify-between">
                <div className="flex space-x-3">
                  <button className="flex items-center space-x-2 px-4 py-2 bg-gray-200 text-gray-700 rounded-lg hover:bg-gray-300 transition-all duration-200">
                    <Download className="w-4 h-4" />
                    <span>Download PDF</span>
                  </button>
                  <button className="flex items-center space-x-2 px-4 py-2 bg-gray-200 text-gray-700 rounded-lg hover:bg-gray-300 transition-all duration-200">
                    <Share className="w-4 h-4" />
                    <span>Share Report</span>
                  </button>
                  <button className="flex items-center space-x-2 px-4 py-2 bg-gray-200 text-gray-700 rounded-lg hover:bg-gray-300 transition-all duration-200">
                    <Bookmark className="w-4 h-4" />
                    <span>Save for Later</span>
                  </button>
                </div>
                
                <div className="flex space-x-3">
                  <button
                    onClick={onBookAppointment}
                    className="bg-green-600 text-white px-6 py-3 rounded-xl font-semibold hover:bg-green-700 transition-all duration-200 flex items-center space-x-2"
                  >
                    <Calendar className="w-5 h-5" />
                    <span>Book Appointment</span>
                  </button>
                  <button
                    onClick={handleSaveReport}
                    disabled={isProcessing}
                    className="bg-gradient-to-r from-blue-600 to-purple-600 text-white px-6 py-3 rounded-xl font-semibold disabled:opacity-50 disabled:cursor-not-allowed hover:shadow-lg transition-all duration-200 flex items-center space-x-2"
                  >
                    {isProcessing ? (
                      <>
                        <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
                        <span>Saving...</span>
                      </>
                    ) : (
                      <>
                        <Heart className="w-5 h-5" />
                        <span>Save to Dashboard</span>
                      </>
                    )}
                  </button>
                </div>
              </div>
            </div>
          </motion.div>
        </div>
      )}
    </AnimatePresence>
  );
};