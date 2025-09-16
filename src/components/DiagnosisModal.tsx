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
  Bookmark,
  User,
  Stethoscope,
  Eye,
  Shield,
  Zap
} from 'lucide-react';

interface DiagnosisData {
  preliminary_assessment: {
    primary_diagnosis: string;
    confidence_score: number;
    urgency_level: string;
    clinical_reasoning: string;
    mental_health_risk: string;
    overall_health_status: string;
  };
  differential_diagnosis: Array<{
    condition: string;
    probability: number;
    supporting_evidence: string[];
    distinguishing_features: string[];
  }>;
  clinical_analysis: {
    probable_causes: string[];
    risk_factors: string[];
    protective_factors: string[];
    symptom_pattern_analysis: string;
    red_flags: string[];
    clinical_pearls: string[];
  };
  recommended_investigations: {
    essential_tests: string[];
    imaging_studies: string[];
    specialist_consultations: string[];
    monitoring_parameters: string[];
    urgency_timeline: string;
  };
  treatment_recommendations: {
    immediate_interventions: string[];
    medication_categories: string[];
    non_pharmacological: string[];
    contraindications: string[];
    monitoring_requirements: string[];
  };
  lifestyle_optimization: {
    diet_modifications: string[];
    exercise_prescription: string[];
    sleep_hygiene: string[];
    stress_management: string[];
    preventive_measures: string[];
    environmental_modifications: string[];
  };
  patient_education: {
    condition_explanation: string;
    warning_signs: string[];
    self_care_strategies: string[];
    prognosis: string;
    when_to_seek_help: string;
  };
  doctor_handoff: {
    chief_complaint: string;
    history_present_illness: string;
    relevant_pmh: string;
    medications_allergies: string;
    clinical_impression: string;
    recommended_workup: string;
    priority_level: string;
  };
  follow_up_plan: {
    reassessment_timeline: string;
    symptom_tracking: string[];
    progress_indicators: string[];
    next_steps: string[];
    long_term_management: string[];
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
  const [showFullReport, setShowFullReport] = useState(false);

  if (!diagnosisData) return null;

  const getUrgencyColor = (urgency: string) => {
    switch (urgency?.toUpperCase()) {
      case 'EMERGENCY': return 'bg-red-500 text-white border-red-600';
      case 'HIGH': return 'bg-orange-500 text-white border-orange-600';
      case 'MEDIUM': return 'bg-yellow-500 text-white border-yellow-600';
      case 'LOW': return 'bg-green-500 text-white border-green-600';
      default: return 'bg-gray-500 text-white border-gray-600';
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
    { id: 'overview', label: 'Overview', icon: Brain, color: 'text-blue-600' },
    { id: 'clinical', label: 'Clinical Analysis', icon: Activity, color: 'text-purple-600' },
    { id: 'differential', label: 'Differential Diagnosis', icon: Stethoscope, color: 'text-green-600' },
    { id: 'investigations', label: 'Tests & Imaging', icon: FileText, color: 'text-orange-600' },
    { id: 'treatment', label: 'Treatment Plan', icon: Pill, color: 'text-red-600' },
    { id: 'lifestyle', label: 'Lifestyle Plan', icon: Heart, color: 'text-pink-600' },
    { id: 'education', label: 'Patient Education', icon: CheckCircle, color: 'text-indigo-600' },
    { id: 'followup', label: 'Follow-up Plan', icon: Calendar, color: 'text-teal-600' },
    { id: 'doctor', label: 'Doctor Handoff', icon: User, color: 'text-gray-600' }
  ];

  const handleSaveReport = async () => {
    setIsProcessing(true);
    try {
      await onSaveToDashboard();
      
      // Show success animation
      setTimeout(() => {
        setIsProcessing(false);
        onClose();
        // Redirect to diagnosis page
        window.location.href = '/diagnosis';
      }, 1500);
    } catch (error) {
      console.error('Save failed:', error);
      setIsProcessing(false);
    }
  };

  const downloadReport = () => {
    // Generate and download PDF report
    const reportContent = JSON.stringify(diagnosisData, null, 2);
    const blob = new Blob([reportContent], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `medical-assessment-${new Date().toISOString().split('T')[0]}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const UrgencyIcon = getUrgencyIcon(diagnosisData.preliminary_assessment.urgency_level);

  return (
    <AnimatePresence>
      {isOpen && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
          <motion.div
            initial={{ opacity: 0, scale: 0.9, y: 20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.9, y: 20 }}
            className="bg-white rounded-3xl shadow-2xl max-w-7xl w-full max-h-[95vh] overflow-hidden"
          >
            {/* Enhanced Header */}
            <div className="bg-gradient-to-r from-blue-600 via-purple-600 to-green-600 p-8 text-white">
              <div className="flex items-center justify-between mb-6">
                <div className="flex items-center space-x-4">
                  <div className="bg-white/20 p-4 rounded-2xl">
                    <Brain className="w-10 h-10" />
                  </div>
                  <div>
                    <h2 className="text-3xl font-bold">AI Medical Assessment</h2>
                    <p className="text-blue-100 text-lg">Comprehensive health analysis completed</p>
                  </div>
                </div>
                <button
                  onClick={onClose}
                  className="p-3 hover:bg-white/20 rounded-xl transition-all duration-200"
                >
                  <X className="w-8 h-8" />
                </button>
              </div>

              {/* Primary Diagnosis Banner */}
              <div className="bg-white/10 backdrop-blur-sm rounded-3xl p-8">
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                  <div className="lg:col-span-2">
                    <h3 className="text-xl font-bold mb-3 flex items-center space-x-2">
                      <Stethoscope className="w-6 h-6" />
                      <span>Primary Assessment</span>
                    </h3>
                    <p className="text-3xl font-bold text-white mb-2">
                      {diagnosisData.preliminary_assessment.primary_diagnosis}
                    </p>
                    <p className="text-blue-100 text-lg leading-relaxed">
                      {diagnosisData.preliminary_assessment.clinical_reasoning}
                    </p>
                  </div>
                  <div className="text-center lg:text-right">
                    <div className="mb-4">
                      <div className="text-sm text-blue-200 mb-2">AI Confidence Level</div>
                      <div className="flex items-center justify-center lg:justify-end space-x-3">
                        <div className="bg-white/20 rounded-full h-3 w-32">
                          <div 
                            className="bg-white h-3 rounded-full transition-all duration-1000"
                            style={{ width: `${(diagnosisData.preliminary_assessment.confidence_score || 0) * 100}%` }}
                          />
                        </div>
                        <span className="font-bold text-2xl">
                          {Math.round((diagnosisData.preliminary_assessment.confidence_score || 0) * 100)}%
                        </span>
                      </div>
                    </div>
                    <div className={`inline-flex items-center space-x-2 px-4 py-3 rounded-2xl font-bold text-lg ${getUrgencyColor(diagnosisData.preliminary_assessment.urgency_level)}`}>
                      <UrgencyIcon className="w-6 h-6" />
                      <span>{diagnosisData.preliminary_assessment.urgency_level} Priority</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            <div className="flex h-[calc(95vh-280px)]">
              {/* Enhanced Sidebar Navigation */}
              <div className="w-80 bg-gray-50 p-6 overflow-y-auto border-r border-gray-200">
                <nav className="space-y-2">
                  {sections.map((section) => {
                    const Icon = section.icon;
                    return (
                      <button
                        key={section.id}
                        onClick={() => setActiveSection(section.id)}
                        className={`w-full flex items-center space-x-4 px-4 py-4 rounded-2xl text-left transition-all duration-200 ${
                          activeSection === section.id
                            ? 'bg-blue-600 text-white shadow-lg transform scale-105'
                            : 'text-gray-700 hover:bg-white hover:shadow-md'
                        }`}
                      >
                        <Icon className={`w-6 h-6 ${activeSection === section.id ? 'text-white' : section.color}`} />
                        <span className="font-semibold">{section.label}</span>
                      </button>
                    );
                  })}
                </nav>

                {/* Quick Stats */}
                <div className="mt-8 space-y-4">
                  <div className="bg-white rounded-2xl p-4 shadow-sm">
                    <div className="flex items-center space-x-3">
                      <div className="bg-blue-100 p-2 rounded-lg">
                        <TrendingUp className="w-5 h-5 text-blue-600" />
                      </div>
                      <div>
                        <div className="text-sm text-gray-600">Health Score</div>
                        <div className="font-bold text-lg">85/100</div>
                      </div>
                    </div>
                  </div>
                  
                  <div className="bg-white rounded-2xl p-4 shadow-sm">
                    <div className="flex items-center space-x-3">
                      <div className="bg-green-100 p-2 rounded-lg">
                        <Shield className="w-5 h-5 text-green-600" />
                      </div>
                      <div>
                        <div className="text-sm text-gray-600">Risk Level</div>
                        <div className="font-bold text-lg text-green-600">Low</div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Enhanced Main Content */}
              <div className="flex-1 p-8 overflow-y-auto">
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
                      <div className="space-y-8">
                        <div>
                          <h2 className="text-3xl font-bold text-gray-900 mb-6">Assessment Overview</h2>
                        </div>

                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                          <div className="bg-gradient-to-br from-blue-50 to-blue-100 rounded-2xl p-6 border border-blue-200">
                            <div className="flex items-center space-x-3 mb-4">
                              <div className="bg-blue-600 p-3 rounded-xl">
                                <Heart className="w-6 h-6 text-white" />
                              </div>
                              <h3 className="text-xl font-bold text-blue-900">Overall Health Status</h3>
                            </div>
                            <p className="text-blue-800 text-lg leading-relaxed">
                              {diagnosisData.preliminary_assessment.overall_health_status}
                            </p>
                          </div>

                          <div className="bg-gradient-to-br from-purple-50 to-purple-100 rounded-2xl p-6 border border-purple-200">
                            <div className="flex items-center space-x-3 mb-4">
                              <div className="bg-purple-600 p-3 rounded-xl">
                                <Brain className="w-6 h-6 text-white" />
                              </div>
                              <h3 className="text-xl font-bold text-purple-900">Mental Health Assessment</h3>
                            </div>
                            <p className="text-purple-800 text-lg leading-relaxed">
                              {diagnosisData.preliminary_assessment.mental_health_risk}
                            </p>
                          </div>
                        </div>

                        {/* Protective Factors */}
                        <div className="bg-green-50 rounded-2xl p-6 border border-green-200">
                          <h3 className="text-xl font-bold text-green-900 mb-4 flex items-center space-x-2">
                            <Shield className="w-6 h-6" />
                            <span>Protective Health Factors</span>
                          </h3>
                          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                            {diagnosisData.clinical_analysis.protective_factors?.map((factor, index) => (
                              <div key={index} className="flex items-start space-x-3 p-3 bg-white rounded-xl">
                                <CheckCircle className="w-5 h-5 text-green-600 mt-0.5" />
                                <span className="text-green-800">{factor}</span>
                              </div>
                            )) || <p className="text-green-700">No specific protective factors identified</p>}
                          </div>
                        </div>

                        {/* Clinical Pearls */}
                        <div className="bg-indigo-50 rounded-2xl p-6 border border-indigo-200">
                          <h3 className="text-xl font-bold text-indigo-900 mb-4 flex items-center space-x-2">
                            <Zap className="w-6 h-6" />
                            <span>Clinical Insights</span>
                          </h3>
                          <div className="space-y-3">
                            {diagnosisData.clinical_analysis.clinical_pearls?.map((pearl, index) => (
                              <div key={index} className="p-4 bg-white rounded-xl border border-indigo-200">
                                <p className="text-indigo-800 font-medium">{pearl}</p>
                              </div>
                            )) || <p className="text-indigo-700">Clinical insights will be provided by healthcare professionals</p>}
                          </div>
                        </div>
                      </div>
                    )}

                    {/* Clinical Analysis Section */}
                    {activeSection === 'clinical' && (
                      <div className="space-y-8">
                        <h2 className="text-3xl font-bold text-gray-900">Clinical Analysis</h2>
                        
                        <div className="bg-white rounded-2xl p-6 border border-gray-200 shadow-lg">
                          <h3 className="text-xl font-bold text-gray-900 mb-4">Symptom Pattern Analysis</h3>
                          <div className="bg-blue-50 rounded-xl p-6">
                            <p className="text-blue-900 text-lg leading-relaxed">
                              {diagnosisData.clinical_analysis.symptom_pattern_analysis}
                            </p>
                          </div>
                        </div>
                        
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                          <div className="bg-white rounded-2xl p-6 border border-gray-200 shadow-lg">
                            <h4 className="text-xl font-bold text-gray-900 mb-4 flex items-center space-x-2">
                              <AlertTriangle className="w-6 h-6 text-orange-600" />
                              <span>Probable Causes</span>
                            </h4>
                            <div className="space-y-3">
                              {diagnosisData.clinical_analysis.probable_causes?.map((cause, index) => (
                                <div key={index} className="p-4 bg-orange-50 rounded-xl border border-orange-200">
                                  <span className="text-orange-800 font-medium">{cause}</span>
                                </div>
                              )) || <p className="text-gray-500">No specific causes identified</p>}
                            </div>
                          </div>

                          <div className="bg-white rounded-2xl p-6 border border-gray-200 shadow-lg">
                            <h4 className="text-xl font-bold text-gray-900 mb-4 flex items-center space-x-2">
                              <AlertTriangle className="w-6 h-6 text-red-600" />
                              <span>Risk Factors</span>
                            </h4>
                            <div className="space-y-3">
                              {diagnosisData.clinical_analysis.risk_factors?.map((factor, index) => (
                                <div key={index} className="flex items-start space-x-3 p-4 bg-red-50 rounded-xl border border-red-200">
                                  <AlertTriangle className="w-5 h-5 text-red-600 mt-0.5" />
                                  <span className="text-red-800">{factor}</span>
                                </div>
                              )) || <p className="text-gray-500">No significant risk factors identified</p>}
                            </div>
                          </div>
                        </div>

                        {/* Red Flags */}
                        {diagnosisData.clinical_analysis.red_flags && diagnosisData.clinical_analysis.red_flags.length > 0 && (
                          <div className="bg-red-50 rounded-2xl p-6 border-2 border-red-300">
                            <h4 className="text-xl font-bold text-red-900 mb-4 flex items-center space-x-2">
                              <AlertTriangle className="w-6 h-6" />
                              <span>Emergency Warning Signs</span>
                            </h4>
                            <div className="space-y-3">
                              {diagnosisData.clinical_analysis.red_flags.map((flag, index) => (
                                <div key={index} className="flex items-start space-x-3 p-4 bg-red-100 rounded-xl border border-red-300">
                                  <AlertTriangle className="w-5 h-5 text-red-700 mt-0.5" />
                                  <span className="text-red-900 font-medium">{flag}</span>
                                </div>
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                    )}

                    {/* Differential Diagnosis Section */}
                    {activeSection === 'differential' && (
                      <div className="space-y-8">
                        <h2 className="text-3xl font-bold text-gray-900">Differential Diagnosis</h2>
                        
                        <div className="space-y-6">
                          {diagnosisData.differential_diagnosis?.map((diagnosis, index) => (
                            <div key={index} className="bg-white rounded-2xl p-6 border border-gray-200 shadow-lg">
                              <div className="flex items-center justify-between mb-4">
                                <h3 className="text-xl font-bold text-gray-900">{diagnosis.condition}</h3>
                                <div className="flex items-center space-x-2">
                                  <span className="text-sm text-gray-600">Probability:</span>
                                  <div className="bg-gray-200 rounded-full h-3 w-24">
                                    <div 
                                      className="bg-blue-500 h-3 rounded-full transition-all duration-1000"
                                      style={{ width: `${diagnosis.probability * 100}%` }}
                                    />
                                  </div>
                                  <span className="font-bold text-lg">{Math.round(diagnosis.probability * 100)}%</span>
                                </div>
                              </div>
                              
                              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                                <div>
                                  <h4 className="font-bold text-gray-900 mb-3">Supporting Evidence</h4>
                                  <div className="space-y-2">
                                    {diagnosis.supporting_evidence?.map((evidence, idx) => (
                                      <div key={idx} className="flex items-start space-x-2">
                                        <div className="w-2 h-2 bg-green-600 rounded-full mt-2"></div>
                                        <span className="text-gray-700 text-sm">{evidence}</span>
                                      </div>
                                    )) || <p className="text-gray-500">No specific evidence listed</p>}
                                  </div>
                                </div>
                                
                                <div>
                                  <h4 className="font-bold text-gray-900 mb-3">Distinguishing Features</h4>
                                  <div className="space-y-2">
                                    {diagnosis.distinguishing_features?.map((feature, idx) => (
                                      <div key={idx} className="flex items-start space-x-2">
                                        <div className="w-2 h-2 bg-blue-600 rounded-full mt-2"></div>
                                        <span className="text-gray-700 text-sm">{feature}</span>
                                      </div>
                                    )) || <p className="text-gray-500">No distinguishing features listed</p>}
                                  </div>
                                </div>
                              </div>
                            </div>
                          )) || (
                            <div className="bg-white rounded-2xl p-8 text-center">
                              <Stethoscope className="w-16 h-16 text-gray-300 mx-auto mb-4" />
                              <p className="text-gray-600">No differential diagnoses provided</p>
                            </div>
                          )}
                        </div>
                      </div>
                    )}

                    {/* Investigations Section */}
                    {activeSection === 'investigations' && (
                      <div className="space-y-8">
                        <h2 className="text-3xl font-bold text-gray-900">Recommended Investigations</h2>
                        
                        <div className="bg-orange-50 rounded-2xl p-6 border border-orange-200">
                          <div className="flex items-center space-x-3 mb-4">
                            <Clock className="w-6 h-6 text-orange-600" />
                            <h3 className="text-xl font-bold text-orange-900">Timeline</h3>
                          </div>
                          <p className="text-orange-800 text-lg font-medium">
                            {diagnosisData.recommended_investigations.urgency_timeline}
                          </p>
                        </div>
                        
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                          <div className="bg-white rounded-2xl p-6 border border-gray-200 shadow-lg">
                            <h3 className="text-xl font-bold text-gray-900 mb-4 flex items-center space-x-2">
                              <FileText className="w-6 h-6 text-blue-600" />
                              <span>Essential Lab Tests</span>
                            </h3>
                            <div className="space-y-3">
                              {diagnosisData.recommended_investigations.essential_tests?.map((test, index) => (
                                <div key={index} className="p-4 bg-blue-50 rounded-xl border border-blue-200">
                                  <span className="text-blue-800 font-medium">{test}</span>
                                </div>
                              )) || <p className="text-gray-500">No specific lab tests recommended</p>}
                            </div>
                          </div>

                          <div className="bg-white rounded-2xl p-6 border border-gray-200 shadow-lg">
                            <h3 className="text-xl font-bold text-gray-900 mb-4 flex items-center space-x-2">
                              <Activity className="w-6 h-6 text-purple-600" />
                              <span>Imaging Studies</span>
                            </h3>
                            <div className="space-y-3">
                              {diagnosisData.recommended_investigations.imaging_studies?.map((study, index) => (
                                <div key={index} className="p-4 bg-purple-50 rounded-xl border border-purple-200">
                                  <span className="text-purple-800 font-medium">{study}</span>
                                </div>
                              )) || <p className="text-gray-500">No imaging studies recommended</p>}
                            </div>
                          </div>
                        </div>

                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                          <div className="bg-white rounded-2xl p-6 border border-gray-200 shadow-lg">
                            <h3 className="text-xl font-bold text-gray-900 mb-4 flex items-center space-x-2">
                              <User className="w-6 h-6 text-green-600" />
                              <span>Specialist Consultations</span>
                            </h3>
                            <div className="space-y-3">
                              {diagnosisData.recommended_investigations.specialist_consultations?.map((specialist, index) => (
                                <div key={index} className="p-4 bg-green-50 rounded-xl border border-green-200 text-center">
                                  <span className="text-green-800 font-medium">{specialist}</span>
                                </div>
                              )) || <p className="text-gray-500">No specialist consultations needed at this time</p>}
                            </div>
                          </div>

                          <div className="bg-white rounded-2xl p-6 border border-gray-200 shadow-lg">
                            <h3 className="text-xl font-bold text-gray-900 mb-4 flex items-center space-x-2">
                              <Activity className="w-6 h-6 text-orange-600" />
                              <span>Monitoring Parameters</span>
                            </h3>
                            <div className="space-y-3">
                              {diagnosisData.recommended_investigations.monitoring_parameters?.map((param, index) => (
                                <div key={index} className="p-4 bg-orange-50 rounded-xl border border-orange-200">
                                  <span className="text-orange-800 font-medium">{param}</span>
                                </div>
                              )) || <p className="text-gray-500">Standard vital signs monitoring recommended</p>}
                            </div>
                          </div>
                        </div>
                      </div>
                    )}

                    {/* Treatment Section */}
                    {activeSection === 'treatment' && (
                      <div className="space-y-8">
                        <h2 className="text-3xl font-bold text-gray-900">Treatment Recommendations</h2>
                        
                        {/* Immediate Interventions */}
                        {diagnosisData.treatment_recommendations.immediate_interventions && 
                         diagnosisData.treatment_recommendations.immediate_interventions.length > 0 && (
                          <div className="bg-red-50 rounded-2xl p-6 border-2 border-red-300">
                            <h3 className="text-xl font-bold text-red-900 mb-4 flex items-center space-x-2">
                              <Zap className="w-6 h-6" />
                              <span>Immediate Interventions</span>
                            </h3>
                            <div className="space-y-3">
                              {diagnosisData.treatment_recommendations.immediate_interventions.map((intervention, index) => (
                                <div key={index} className="flex items-start space-x-3 p-4 bg-red-100 rounded-xl">
                                  <div className="bg-red-600 text-white rounded-full w-6 h-6 flex items-center justify-center text-sm font-bold">
                                    {index + 1}
                                  </div>
                                  <span className="text-red-900 font-medium">{intervention}</span>
                                </div>
                              ))}
                            </div>
                          </div>
                        )}

                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                          <div className="bg-white rounded-2xl p-6 border border-gray-200 shadow-lg">
                            <h3 className="text-xl font-bold text-gray-900 mb-4 flex items-center space-x-2">
                              <Pill className="w-6 h-6 text-blue-600" />
                              <span>Medication Categories</span>
                            </h3>
                            <div className="space-y-3">
                              {diagnosisData.treatment_recommendations.medication_categories?.map((category, index) => (
                                <div key={index} className="p-4 bg-blue-50 rounded-xl border border-blue-200">
                                  <span className="text-blue-800 font-medium">{category}</span>
                                </div>
                              )) || <p className="text-gray-500">No specific medications recommended</p>}
                            </div>
                          </div>

                          <div className="bg-white rounded-2xl p-6 border border-gray-200 shadow-lg">
                            <h3 className="text-xl font-bold text-gray-900 mb-4 flex items-center space-x-2">
                              <Heart className="w-6 h-6 text-green-600" />
                              <span>Non-Pharmacological</span>
                            </h3>
                            <div className="space-y-3">
                              {diagnosisData.treatment_recommendations.non_pharmacological?.map((treatment, index) => (
                                <div key={index} className="p-4 bg-green-50 rounded-xl border border-green-200">
                                  <span className="text-green-800 font-medium">{treatment}</span>
                                </div>
                              )) || <p className="text-gray-500">No specific non-drug treatments recommended</p>}
                            </div>
                          </div>
                        </div>

                        {/* Contraindications */}
                        {diagnosisData.treatment_recommendations.contraindications && 
                         diagnosisData.treatment_recommendations.contraindications.length > 0 && (
                          <div className="bg-yellow-50 rounded-2xl p-6 border border-yellow-300">
                            <h3 className="text-xl font-bold text-yellow-900 mb-4 flex items-center space-x-2">
                              <Shield className="w-6 h-6" />
                              <span>Important Contraindications</span>
                            </h3>
                            <div className="space-y-3">
                              {diagnosisData.treatment_recommendations.contraindications.map((contraindication, index) => (
                                <div key={index} className="p-4 bg-yellow-100 rounded-xl border border-yellow-300">
                                  <span className="text-yellow-900 font-medium">{contraindication}</span>
                                </div>
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                    )}

                    {/* Lifestyle Section */}
                    {activeSection === 'lifestyle' && (
                      <div className="space-y-8">
                        <h2 className="text-3xl font-bold text-gray-900">Personalized Lifestyle Optimization</h2>
                        
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                          <div className="bg-white rounded-2xl p-6 border border-gray-200 shadow-lg">
                            <h3 className="text-xl font-bold text-gray-900 mb-4 flex items-center space-x-2">
                              <Heart className="w-6 h-6 text-green-600" />
                              <span>Diet Modifications</span>
                            </h3>
                            <div className="space-y-3">
                              {diagnosisData.lifestyle_optimization.diet_modifications?.map((diet, index) => (
                                <div key={index} className="flex items-start space-x-3 p-3 bg-green-50 rounded-xl">
                                  <div className="w-2 h-2 bg-green-600 rounded-full mt-2"></div>
                                  <span className="text-green-800">{diet}</span>
                                </div>
                              )) || <p className="text-gray-500">Continue current healthy diet</p>}
                            </div>
                          </div>

                          <div className="bg-white rounded-2xl p-6 border border-gray-200 shadow-lg">
                            <h3 className="text-xl font-bold text-gray-900 mb-4 flex items-center space-x-2">
                              <Activity className="w-6 h-6 text-blue-600" />
                              <span>Exercise Prescription</span>
                            </h3>
                            <div className="space-y-3">
                              {diagnosisData.lifestyle_optimization.exercise_prescription?.map((exercise, index) => (
                                <div key={index} className="flex items-start space-x-3 p-3 bg-blue-50 rounded-xl">
                                  <div className="w-2 h-2 bg-blue-600 rounded-full mt-2"></div>
                                  <span className="text-blue-800">{exercise}</span>
                                </div>
                              )) || <p className="text-gray-500">Maintain current activity level</p>}
                            </div>
                          </div>
                        </div>

                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                          <div className="bg-white rounded-2xl p-6 border border-gray-200 shadow-lg">
                            <h3 className="text-xl font-bold text-gray-900 mb-4 flex items-center space-x-2">
                              <Clock className="w-6 h-6 text-purple-600" />
                              <span>Sleep Hygiene</span>
                            </h3>
                            <div className="space-y-3">
                              {diagnosisData.lifestyle_optimization.sleep_hygiene?.map((tip, index) => (
                                <div key={index} className="flex items-start space-x-3 p-3 bg-purple-50 rounded-xl">
                                  <div className="w-2 h-2 bg-purple-600 rounded-full mt-2"></div>
                                  <span className="text-purple-800">{tip}</span>
                                </div>
                              )) || <p className="text-gray-500">Maintain good sleep habits</p>}
                            </div>
                          </div>

                          <div className="bg-white rounded-2xl p-6 border border-gray-200 shadow-lg">
                            <h3 className="text-xl font-bold text-gray-900 mb-4 flex items-center space-x-2">
                              <Brain className="w-6 h-6 text-orange-600" />
                              <span>Stress Management</span>
                            </h3>
                            <div className="space-y-3">
                              {diagnosisData.lifestyle_optimization.stress_management?.map((strategy, index) => (
                                <div key={index} className="flex items-start space-x-3 p-3 bg-orange-50 rounded-xl">
                                  <div className="w-2 h-2 bg-orange-600 rounded-full mt-2"></div>
                                  <span className="text-orange-800">{strategy}</span>
                                </div>
                              )) || <p className="text-gray-500">Continue current stress management techniques</p>}
                            </div>
                          </div>
                        </div>

                        {/* Environmental Modifications */}
                        <div className="bg-white rounded-2xl p-6 border border-gray-200 shadow-lg">
                          <h3 className="text-xl font-bold text-gray-900 mb-4 flex items-center space-x-2">
                            <Shield className="w-6 h-6 text-teal-600" />
                            <span>Environmental Modifications</span>
                          </h3>
                          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                            {diagnosisData.lifestyle_optimization.environmental_modifications?.map((modification, index) => (
                              <div key={index} className="p-4 bg-teal-50 rounded-xl border border-teal-200">
                                <span className="text-teal-800 font-medium">{modification}</span>
                              </div>
                            )) || <p className="text-gray-500">No specific environmental changes recommended</p>}
                          </div>
                        </div>
                      </div>
                    )}

                    {/* Patient Education Section */}
                    {activeSection === 'education' && (
                      <div className="space-y-8">
                        <h2 className="text-3xl font-bold text-gray-900">Patient Education & Guidance</h2>
                        
                        <div className="bg-white rounded-2xl p-8 border border-gray-200 shadow-lg">
                          <h3 className="text-2xl font-bold text-gray-900 mb-6">Understanding Your Condition</h3>
                          <div className="bg-blue-50 rounded-2xl p-6 border border-blue-200">
                            <p className="text-blue-900 text-lg leading-relaxed">
                              {diagnosisData.patient_education.condition_explanation}
                            </p>
                          </div>
                        </div>

                        <div className="bg-red-50 rounded-2xl p-6 border-2 border-red-300">
                          <h3 className="text-xl font-bold text-red-900 mb-4 flex items-center space-x-2">
                            <AlertTriangle className="w-6 h-6" />
                            <span>Warning Signs - Seek Immediate Care</span>
                          </h3>
                          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                            {diagnosisData.patient_education.warning_signs?.map((sign, index) => (
                              <div key={index} className="flex items-start space-x-3 p-4 bg-red-100 rounded-xl">
                                <AlertTriangle className="w-5 h-5 text-red-600 mt-0.5" />
                                <span className="text-red-800 font-medium">{sign}</span>
                              </div>
                            )) || <p className="text-red-800">Monitor for any worsening symptoms</p>}
                          </div>
                        </div>

                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                          <div className="bg-white rounded-2xl p-6 border border-gray-200 shadow-lg">
                            <h3 className="text-xl font-bold text-gray-900 mb-4">Self-Care Strategies</h3>
                            <div className="space-y-3">
                              {diagnosisData.patient_education.self_care_strategies?.map((strategy, index) => (
                                <div key={index} className="flex items-start space-x-3 p-3 bg-green-50 rounded-xl">
                                  <CheckCircle className="w-5 h-5 text-green-600 mt-0.5" />
                                  <span className="text-green-800">{strategy}</span>
                                </div>
                              )) || <p className="text-gray-500">Follow general health maintenance practices</p>}
                            </div>
                          </div>

                          <div className="bg-white rounded-2xl p-6 border border-gray-200 shadow-lg">
                            <h3 className="text-xl font-bold text-gray-900 mb-4">When to Seek Help</h3>
                            <div className="bg-yellow-50 rounded-xl p-4 border border-yellow-200">
                              <p className="text-yellow-900 font-medium">
                                {diagnosisData.patient_education.when_to_seek_help}
                              </p>
                            </div>
                          </div>
                        </div>

                        <div className="bg-blue-50 rounded-2xl p-6 border border-blue-200">
                          <h3 className="text-xl font-bold text-blue-900 mb-4">Prognosis & Outlook</h3>
                          <p className="text-blue-800 text-lg leading-relaxed">{diagnosisData.patient_education.prognosis}</p>
                        </div>
                      </div>
                    )}

                    {/* Doctor Handoff Section */}
                    {activeSection === 'doctor' && (
                      <div className="space-y-8">
                        <h2 className="text-3xl font-bold text-gray-900">Healthcare Provider Summary</h2>
                        
                        <div className="bg-gradient-to-r from-gray-50 to-gray-100 rounded-2xl p-6 border border-gray-300">
                          <div className="flex items-center space-x-3 mb-4">
                            <User className="w-8 h-8 text-gray-600" />
                            <h3 className="text-2xl font-bold text-gray-900">Clinical Summary for Healthcare Provider</h3>
                          </div>
                          <p className="text-gray-700 text-lg">
                            This structured summary is designed to save healthcare providers time by providing 
                            pre-analyzed patient information in standard medical format.
                          </p>
                        </div>

                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                          <div className="bg-white rounded-2xl p-6 border border-gray-200 shadow-lg">
                            <h4 className="text-lg font-bold text-gray-900 mb-3">Chief Complaint</h4>
                            <div className="bg-blue-50 rounded-xl p-4">
                              <p className="text-blue-900 font-mono text-sm">
                                {diagnosisData.doctor_handoff.chief_complaint}
                              </p>
                            </div>
                          </div>

                          <div className="bg-white rounded-2xl p-6 border border-gray-200 shadow-lg">
                            <h4 className="text-lg font-bold text-gray-900 mb-3">Priority Level</h4>
                            <div className={`inline-flex items-center space-x-2 px-4 py-2 rounded-xl font-bold ${getUrgencyColor(diagnosisData.doctor_handoff.priority_level)}`}>
                              <span>{diagnosisData.doctor_handoff.priority_level}</span>
                            </div>
                          </div>
                        </div>

                        <div className="bg-white rounded-2xl p-6 border border-gray-200 shadow-lg">
                          <h4 className="text-lg font-bold text-gray-900 mb-4">History of Present Illness (HPI)</h4>
                          <div className="bg-gray-50 rounded-xl p-6">
                            <pre className="text-gray-800 font-mono text-sm whitespace-pre-wrap">
                              {diagnosisData.doctor_handoff.history_present_illness}
                            </pre>
                          </div>
                        </div>

                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                          <div className="bg-white rounded-2xl p-6 border border-gray-200 shadow-lg">
                            <h4 className="text-lg font-bold text-gray-900 mb-3">Relevant PMH</h4>
                            <div className="bg-purple-50 rounded-xl p-4">
                              <p className="text-purple-900 text-sm">
                                {diagnosisData.doctor_handoff.relevant_pmh}
                              </p>
                            </div>
                          </div>

                          <div className="bg-white rounded-2xl p-6 border border-gray-200 shadow-lg">
                            <h4 className="text-lg font-bold text-gray-900 mb-3">Medications & Allergies</h4>
                            <div className="bg-orange-50 rounded-xl p-4">
                              <p className="text-orange-900 text-sm">
                                {diagnosisData.doctor_handoff.medications_allergies}
                              </p>
                            </div>
                          </div>
                        </div>

                        <div className="bg-white rounded-2xl p-6 border border-gray-200 shadow-lg">
                          <h4 className="text-lg font-bold text-gray-900 mb-4">Clinical Impression & Recommended Workup</h4>
                          <div className="space-y-4">
                            <div className="bg-green-50 rounded-xl p-4">
                              <h5 className="font-bold text-green-900 mb-2">Clinical Impression:</h5>
                              <p className="text-green-800">{diagnosisData.doctor_handoff.clinical_impression}</p>
                            </div>
                            <div className="bg-blue-50 rounded-xl p-4">
                              <h5 className="font-bold text-blue-900 mb-2">Recommended Workup:</h5>
                              <p className="text-blue-800">{diagnosisData.doctor_handoff.recommended_workup}</p>
                            </div>
                          </div>
                        </div>
                      </div>
                    )}

                    {/* Follow-up Plan Section */}
                    {activeSection === 'followup' && (
                      <div className="space-y-8">
                        <h2 className="text-3xl font-bold text-gray-900">Follow-up & Monitoring Plan</h2>
                        
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                          <div className="bg-blue-50 rounded-2xl p-6 border border-blue-200">
                            <h3 className="text-xl font-bold text-blue-900 mb-4">Reassessment Timeline</h3>
                            <div className="bg-white rounded-xl p-4">
                              <p className="text-blue-800 text-lg font-medium">
                                {diagnosisData.follow_up_plan.reassessment_timeline}
                              </p>
                            </div>
                          </div>

                          <div className="bg-green-50 rounded-2xl p-6 border border-green-200">
                            <h3 className="text-xl font-bold text-green-900 mb-4">Progress Indicators</h3>
                            <div className="space-y-2">
                              {diagnosisData.follow_up_plan.progress_indicators?.slice(0, 3).map((indicator, index) => (
                                <div key={index} className="flex items-start space-x-2 p-2 bg-white rounded-lg">
                                  <TrendingUp className="w-4 h-4 text-green-600 mt-1" />
                                  <span className="text-green-800 text-sm">{indicator}</span>
                                </div>
                              )) || <p className="text-gray-500">Monitor general health improvement</p>}
                            </div>
                          </div>
                        </div>

                        <div className="bg-white rounded-2xl p-6 border border-gray-200 shadow-lg">
                          <h3 className="text-xl font-bold text-gray-900 mb-4">Symptoms to Track</h3>
                          <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                            {diagnosisData.follow_up_plan.symptom_tracking?.map((symptom, index) => (
                              <div key={index} className="p-3 bg-yellow-50 rounded-xl border border-yellow-200 text-center">
                                <span className="text-yellow-800 font-medium">{symptom}</span>
                              </div>
                            )) || <p className="text-gray-500">Monitor general symptoms</p>}
                          </div>
                        </div>

                        <div className="bg-white rounded-2xl p-6 border border-gray-200 shadow-lg">
                          <h3 className="text-xl font-bold text-gray-900 mb-4">Immediate Next Steps</h3>
                          <div className="space-y-4">
                            {diagnosisData.follow_up_plan.next_steps?.map((step, index) => (
                              <div key={index} className="flex items-start space-x-4 p-4 bg-green-50 rounded-xl border border-green-200">
                                <div className="bg-green-600 text-white rounded-full w-8 h-8 flex items-center justify-center text-sm font-bold">
                                  {index + 1}
                                </div>
                                <span className="text-green-800 font-medium">{step}</span>
                              </div>
                            )) || <p className="text-gray-500">Continue current care plan</p>}
                          </div>
                        </div>
                      </div>
                    )}
                  </motion.div>
                </AnimatePresence>
              </div>
            </div>

            {/* Enhanced Footer Actions */}
            <div className="bg-gray-50 p-6 border-t border-gray-200">
              <div className="flex items-center justify-between">
                <div className="flex space-x-3">
                  <button 
                    onClick={downloadReport}
                    className="flex items-center space-x-2 px-4 py-2 bg-gray-200 text-gray-700 rounded-xl hover:bg-gray-300 transition-all duration-200"
                  >
                    <Download className="w-5 h-5" />
                    <span>Download Report</span>
                  </button>
                  <button className="flex items-center space-x-2 px-4 py-2 bg-gray-200 text-gray-700 rounded-xl hover:bg-gray-300 transition-all duration-200">
                    <Share className="w-5 h-5" />
                    <span>Share with Doctor</span>
                  </button>
                  <button className="flex items-center space-x-2 px-4 py-2 bg-gray-200 text-gray-700 rounded-xl hover:bg-gray-300 transition-all duration-200">
                    <Bookmark className="w-5 h-5" />
                    <span>Save for Later</span>
                  </button>
                  <button 
                    onClick={() => setShowFullReport(!showFullReport)}
                    className="flex items-center space-x-2 px-4 py-2 bg-gray-200 text-gray-700 rounded-xl hover:bg-gray-300 transition-all duration-200"
                  >
                    <Eye className="w-5 h-5" />
                    <span>{showFullReport ? 'Hide' : 'Show'} Full Report</span>
                  </button>
                </div>
                
                <div className="flex space-x-4">
                  <button
                    onClick={onBookAppointment}
                    className="bg-green-600 text-white px-8 py-3 rounded-xl font-semibold hover:bg-green-700 transition-all duration-200 flex items-center space-x-2"
                  >
                    <Calendar className="w-5 h-5" />
                    <span>Book Appointment</span>
                  </button>
                  <button
                    onClick={handleSaveReport}
                    disabled={isProcessing}
                    className="bg-gradient-to-r from-blue-600 to-purple-600 text-white px-8 py-3 rounded-xl font-semibold disabled:opacity-50 disabled:cursor-not-allowed hover:shadow-lg transition-all duration-200 flex items-center space-x-2"
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

            {/* Full Report Overlay */}
            {showFullReport && (
              <div className="absolute inset-0 bg-white z-10 overflow-y-auto">
                <div className="p-6">
                  <div className="flex items-center justify-between mb-6">
                    <h2 className="text-2xl font-bold text-gray-900">Complete Medical Assessment Report</h2>
                    <button
                      onClick={() => setShowFullReport(false)}
                      className="p-2 bg-gray-100 rounded-lg hover:bg-gray-200"
                    >
                      <X className="w-6 h-6" />
                    </button>
                  </div>
                  <pre className="bg-gray-50 rounded-xl p-6 text-sm text-gray-800 whitespace-pre-wrap font-mono">
                    {JSON.stringify(diagnosisData, null, 2)}
                  </pre>
                </div>
              </div>
            )}
          </motion.div>
        </div>
      )}
    </AnimatePresence>
  );
};