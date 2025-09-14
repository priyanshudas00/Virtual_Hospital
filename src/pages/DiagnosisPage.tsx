import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { useSearchParams } from 'react-router-dom';
import { 
  Brain, 
  FileText, 
  Calendar, 
  Download, 
  Share,
  AlertTriangle,
  CheckCircle,
  TrendingUp,
  Activity,
  Heart,
  Pill,
  Clock,
  User,
  Bookmark,
  RefreshCw
} from 'lucide-react';

interface DiagnosisReport {
  id: string;
  patient_id: string;
  assessment_data: any;
  created_at: string;
  status: string;
  report_type: string;
}

export const DiagnosisPage: React.FC = () => {
  const [searchParams] = useSearchParams();
  const [reports, setReports] = useState<DiagnosisReport[]>([]);
  const [selectedReport, setSelectedReport] = useState<DiagnosisReport | null>(null);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState('recent');

  useEffect(() => {
    loadDiagnosisReports();
    
    // Check if we have a specific report ID from intake form
    const reportId = searchParams.get('report_id');
    if (reportId) {
      loadSpecificReport(reportId);
    }
  }, [searchParams]);

  const loadDiagnosisReports = async () => {
    try {
      const response = await fetch(`${import.meta.env.VITE_API_URL}/api/diagnosis-reports`, {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        }
      });

      if (response.ok) {
        const data = await response.json();
        setReports(data.reports || []);
        
        // Auto-select the most recent report
        if (data.reports && data.reports.length > 0) {
          setSelectedReport(data.reports[0]);
        }
      }
    } catch (error) {
      console.error('Failed to load diagnosis reports:', error);
    } finally {
      setLoading(false);
    }
  };

  const loadSpecificReport = async (reportId: string) => {
    try {
      const response = await fetch(`${import.meta.env.VITE_API_URL}/api/diagnosis-reports/${reportId}`, {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        }
      });

      if (response.ok) {
        const report = await response.json();
        setSelectedReport(report);
        setActiveTab('current');
      }
    } catch (error) {
      console.error('Failed to load specific report:', error);
    }
  };

  const getUrgencyColor = (urgency: string) => {
    switch (urgency?.toUpperCase()) {
      case 'EMERGENCY': return 'bg-red-100 text-red-800 border-red-300';
      case 'HIGH': return 'bg-orange-100 text-orange-800 border-orange-300';
      case 'MEDIUM': return 'bg-yellow-100 text-yellow-800 border-yellow-300';
      case 'LOW': return 'bg-green-100 text-green-800 border-green-300';
      default: return 'bg-gray-100 text-gray-800 border-gray-300';
    }
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'long',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <p className="text-gray-600">Loading your diagnosis reports...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen py-8 px-4 sm:px-6 lg:px-8 bg-gradient-to-br from-blue-50 via-white to-purple-50">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-8"
        >
          <div className="bg-gradient-to-r from-blue-600 to-purple-600 w-20 h-20 rounded-3xl flex items-center justify-center mx-auto mb-6">
            <Brain className="w-10 h-10 text-white" />
          </div>
          <h1 className="text-4xl font-bold text-gray-900 mb-3">AI Diagnosis Center</h1>
          <p className="text-xl text-gray-600">Your comprehensive medical assessments and AI-powered insights</p>
        </motion.div>

        <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
          {/* Sidebar - Reports List */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            className="lg:col-span-1"
          >
            <div className="bg-white rounded-2xl shadow-xl p-6 sticky top-8">
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-xl font-bold text-gray-900">Your Reports</h2>
                <button
                  onClick={loadDiagnosisReports}
                  className="p-2 bg-blue-100 text-blue-600 rounded-lg hover:bg-blue-200 transition-all duration-200"
                >
                  <RefreshCw className="w-4 h-4" />
                </button>
              </div>

              {/* Tabs */}
              <div className="flex space-x-1 mb-4 bg-gray-100 rounded-lg p-1">
                <button
                  onClick={() => setActiveTab('recent')}
                  className={`flex-1 py-2 px-3 rounded-md text-sm font-medium transition-all duration-200 ${
                    activeTab === 'recent'
                      ? 'bg-white text-blue-600 shadow-sm'
                      : 'text-gray-600 hover:text-gray-900'
                  }`}
                >
                  Recent
                </button>
                <button
                  onClick={() => setActiveTab('all')}
                  className={`flex-1 py-2 px-3 rounded-md text-sm font-medium transition-all duration-200 ${
                    activeTab === 'all'
                      ? 'bg-white text-blue-600 shadow-sm'
                      : 'text-gray-600 hover:text-gray-900'
                  }`}
                >
                  All Reports
                </button>
              </div>

              {/* Reports List */}
              <div className="space-y-3 max-h-96 overflow-y-auto">
                {reports.length === 0 ? (
                  <div className="text-center py-8">
                    <FileText className="w-12 h-12 text-gray-300 mx-auto mb-3" />
                    <p className="text-gray-600 text-sm">No diagnosis reports yet</p>
                    <button
                      onClick={() => window.location.href = '/intake-form'}
                      className="mt-3 text-blue-600 text-sm font-medium hover:text-blue-700"
                    >
                      Complete Intake Form →
                    </button>
                  </div>
                ) : (
                  reports.map((report) => (
                    <div
                      key={report.id}
                      onClick={() => setSelectedReport(report)}
                      className={`p-4 rounded-xl cursor-pointer transition-all duration-200 border-2 ${
                        selectedReport?.id === report.id
                          ? 'bg-blue-50 border-blue-300'
                          : 'bg-gray-50 border-gray-200 hover:bg-blue-50 hover:border-blue-200'
                      }`}
                    >
                      <div className="flex items-start space-x-3">
                        <div className="bg-blue-100 p-2 rounded-lg">
                          <Brain className="w-4 h-4 text-blue-600" />
                        </div>
                        <div className="flex-1 min-w-0">
                          <h4 className="font-semibold text-gray-900 text-sm truncate">
                            {report.assessment_data?.preliminary_assessment?.primary_diagnosis || 'Medical Assessment'}
                          </h4>
                          <p className="text-xs text-gray-600 mt-1">
                            {formatDate(report.created_at)}
                          </p>
                          <div className="mt-2">
                            <span className={`inline-block px-2 py-1 rounded-full text-xs font-medium ${
                              getUrgencyColor(report.assessment_data?.preliminary_assessment?.urgency_level || 'LOW')
                            }`}>
                              {report.assessment_data?.preliminary_assessment?.urgency_level || 'LOW'}
                            </span>
                          </div>
                        </div>
                      </div>
                    </div>
                  ))
                )}
              </div>
            </div>
          </motion.div>

          {/* Main Content - Selected Report */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="lg:col-span-3"
          >
            {selectedReport ? (
              <div className="space-y-6">
                {/* Report Header */}
                <div className="bg-white rounded-2xl shadow-xl p-8">
                  <div className="flex items-center justify-between mb-6">
                    <div>
                      <h2 className="text-3xl font-bold text-gray-900 mb-2">
                        {selectedReport.assessment_data?.preliminary_assessment?.primary_diagnosis || 'Medical Assessment'}
                      </h2>
                      <div className="flex items-center space-x-4 text-gray-600">
                        <div className="flex items-center space-x-1">
                          <Calendar className="w-4 h-4" />
                          <span>{formatDate(selectedReport.created_at)}</span>
                        </div>
                        <div className="flex items-center space-x-1">
                          <User className="w-4 h-4" />
                          <span>Report ID: {selectedReport.id.slice(0, 8)}</span>
                        </div>
                      </div>
                    </div>
                    <div className="flex items-center space-x-3">
                      <div className="text-right">
                        <div className="text-sm text-gray-600 mb-1">AI Confidence</div>
                        <div className="flex items-center space-x-2">
                          <div className="bg-gray-200 rounded-full h-2 w-24">
                            <div 
                              className="bg-blue-500 h-2 rounded-full"
                              style={{ 
                                width: `${(selectedReport.assessment_data?.preliminary_assessment?.confidence_score || 0) * 100}%` 
                              }}
                            />
                          </div>
                          <span className="font-bold text-lg">
                            {Math.round((selectedReport.assessment_data?.preliminary_assessment?.confidence_score || 0) * 100)}%
                          </span>
                        </div>
                      </div>
                      <div className={`px-4 py-2 rounded-xl border-2 font-semibold ${
                        getUrgencyColor(selectedReport.assessment_data?.preliminary_assessment?.urgency_level || 'LOW')
                      }`}>
                        {selectedReport.assessment_data?.preliminary_assessment?.urgency_level || 'LOW'} Priority
                      </div>
                    </div>
                  </div>

                  {/* Quick Actions */}
                  <div className="flex flex-wrap gap-3">
                    <button className="flex items-center space-x-2 px-4 py-2 bg-blue-100 text-blue-700 rounded-lg hover:bg-blue-200 transition-all duration-200">
                      <Download className="w-4 h-4" />
                      <span>Download PDF</span>
                    </button>
                    <button className="flex items-center space-x-2 px-4 py-2 bg-green-100 text-green-700 rounded-lg hover:bg-green-200 transition-all duration-200">
                      <Share className="w-4 h-4" />
                      <span>Share with Doctor</span>
                    </button>
                    <button className="flex items-center space-x-2 px-4 py-2 bg-purple-100 text-purple-700 rounded-lg hover:bg-purple-200 transition-all duration-200">
                      <Bookmark className="w-4 h-4" />
                      <span>Save to Favorites</span>
                    </button>
                    <button
                      onClick={() => window.location.href = '/book-appointment'}
                      className="flex items-center space-x-2 px-4 py-2 bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-lg hover:shadow-lg transition-all duration-200"
                    >
                      <Calendar className="w-4 h-4" />
                      <span>Book Follow-up</span>
                    </button>
                  </div>
                </div>

                {/* Assessment Overview */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                  <div className="bg-white rounded-2xl shadow-lg p-6">
                    <div className="flex items-center space-x-3 mb-4">
                      <div className="bg-blue-100 p-3 rounded-xl">
                        <Heart className="w-6 h-6 text-blue-600" />
                      </div>
                      <h3 className="text-lg font-bold text-gray-900">Health Status</h3>
                    </div>
                    <p className="text-gray-700 leading-relaxed">
                      {selectedReport.assessment_data?.preliminary_assessment?.overall_health_status || 'Assessment in progress'}
                    </p>
                  </div>

                  <div className="bg-white rounded-2xl shadow-lg p-6">
                    <div className="flex items-center space-x-3 mb-4">
                      <div className="bg-purple-100 p-3 rounded-xl">
                        <Brain className="w-6 h-6 text-purple-600" />
                      </div>
                      <h3 className="text-lg font-bold text-gray-900">Mental Health</h3>
                    </div>
                    <p className="text-gray-700 leading-relaxed">
                      {selectedReport.assessment_data?.preliminary_assessment?.mental_health_risk || 'No specific concerns identified'}
                    </p>
                  </div>

                  <div className="bg-white rounded-2xl shadow-lg p-6">
                    <div className="flex items-center space-x-3 mb-4">
                      <div className="bg-green-100 p-3 rounded-xl">
                        <TrendingUp className="w-6 h-6 text-green-600" />
                      </div>
                      <h3 className="text-lg font-bold text-gray-900">Prognosis</h3>
                    </div>
                    <p className="text-gray-700 leading-relaxed">
                      {selectedReport.assessment_data?.patient_education?.prognosis || 'Positive outlook with proper care'}
                    </p>
                  </div>
                </div>

                {/* Detailed Analysis Sections */}
                <div className="space-y-6">
                  {/* Clinical Analysis */}
                  <div className="bg-white rounded-2xl shadow-lg p-8">
                    <h3 className="text-2xl font-bold text-gray-900 mb-6 flex items-center space-x-3">
                      <Activity className="w-7 h-7 text-blue-600" />
                      <span>Clinical Analysis</span>
                    </h3>
                    
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      <div>
                        <h4 className="font-bold text-gray-900 mb-3">Probable Causes</h4>
                        <div className="space-y-2">
                          {selectedReport.assessment_data?.clinical_analysis?.probable_causes?.map((cause: string, index: number) => (
                            <div key={index} className="p-3 bg-orange-50 rounded-lg border border-orange-200">
                              <span className="text-orange-800">{cause}</span>
                            </div>
                          )) || <p className="text-gray-500">No specific causes identified</p>}
                        </div>
                      </div>

                      <div>
                        <h4 className="font-bold text-gray-900 mb-3">Risk Factors</h4>
                        <div className="space-y-2">
                          {selectedReport.assessment_data?.clinical_analysis?.risk_factors?.map((factor: string, index: number) => (
                            <div key={index} className="flex items-start space-x-2 p-3 bg-red-50 rounded-lg border border-red-200">
                              <AlertTriangle className="w-4 h-4 text-red-600 mt-0.5" />
                              <span className="text-red-800 text-sm">{factor}</span>
                            </div>
                          )) || <p className="text-gray-500">No significant risk factors identified</p>}
                        </div>
                      </div>
                    </div>

                    <div className="mt-6">
                      <h4 className="font-bold text-gray-900 mb-3">Alternative Diagnoses</h4>
                      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                        {selectedReport.assessment_data?.clinical_analysis?.differential_diagnosis?.map((diagnosis: string, index: number) => (
                          <div key={index} className="p-3 bg-blue-50 rounded-lg border border-blue-200 text-center">
                            <span className="text-blue-800 font-medium">{diagnosis}</span>
                          </div>
                        )) || <p className="text-gray-500">No alternative diagnoses suggested</p>}
                      </div>
                    </div>
                  </div>

                  {/* Recommended Investigations */}
                  <div className="bg-white rounded-2xl shadow-lg p-8">
                    <h3 className="text-2xl font-bold text-gray-900 mb-6 flex items-center space-x-3">
                      <FileText className="w-7 h-7 text-green-600" />
                      <span>Recommended Tests & Investigations</span>
                    </h3>
                    
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      <div>
                        <h4 className="font-bold text-gray-900 mb-4 flex items-center space-x-2">
                          <Activity className="w-5 h-5 text-blue-600" />
                          <span>Essential Lab Tests</span>
                        </h4>
                        <div className="space-y-2">
                          {selectedReport.assessment_data?.recommended_investigations?.essential_tests?.map((test: string, index: number) => (
                            <div key={index} className="p-3 bg-blue-50 rounded-lg">
                              <span className="text-blue-800">{test}</span>
                            </div>
                          )) || <p className="text-gray-500">No specific lab tests recommended</p>}
                        </div>
                      </div>

                      <div>
                        <h4 className="font-bold text-gray-900 mb-4 flex items-center space-x-2">
                          <Brain className="w-5 h-5 text-purple-600" />
                          <span>Imaging Studies</span>
                        </h4>
                        <div className="space-y-2">
                          {selectedReport.assessment_data?.recommended_investigations?.imaging_studies?.map((study: string, index: number) => (
                            <div key={index} className="p-3 bg-purple-50 rounded-lg">
                              <span className="text-purple-800">{study}</span>
                            </div>
                          )) || <p className="text-gray-500">No imaging studies recommended</p>}
                        </div>
                      </div>
                    </div>

                    <div className="mt-6">
                      <h4 className="font-bold text-gray-900 mb-4 flex items-center space-x-2">
                        <Stethoscope className="w-5 h-5 text-green-600" />
                        <span>Specialist Consultations</span>
                      </h4>
                      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                        {selectedReport.assessment_data?.recommended_investigations?.specialist_consultations?.map((specialist: string, index: number) => (
                          <div key={index} className="p-3 bg-green-50 rounded-lg border border-green-200 text-center">
                            <span className="text-green-800 font-medium">{specialist}</span>
                          </div>
                        )) || <p className="text-gray-500">No specialist consultations needed at this time</p>}
                      </div>
                    </div>
                  </div>

                  {/* Lifestyle Recommendations */}
                  <div className="bg-white rounded-2xl shadow-lg p-8">
                    <h3 className="text-2xl font-bold text-gray-900 mb-6 flex items-center space-x-3">
                      <Heart className="w-7 h-7 text-red-600" />
                      <span>Personalized Lifestyle Plan</span>
                    </h3>
                    
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      <div>
                        <h4 className="font-bold text-gray-900 mb-3">Diet Modifications</h4>
                        <div className="space-y-2">
                          {selectedReport.assessment_data?.lifestyle_recommendations?.diet_modifications?.map((diet: string, index: number) => (
                            <div key={index} className="flex items-start space-x-2 p-3 bg-green-50 rounded-lg">
                              <div className="w-2 h-2 bg-green-600 rounded-full mt-2"></div>
                              <span className="text-green-800 text-sm">{diet}</span>
                            </div>
                          )) || <p className="text-gray-500">Continue current healthy diet</p>}
                        </div>
                      </div>

                      <div>
                        <h4 className="font-bold text-gray-900 mb-3">Exercise Plan</h4>
                        <div className="space-y-2">
                          {selectedReport.assessment_data?.lifestyle_recommendations?.exercise_plan?.map((exercise: string, index: number) => (
                            <div key={index} className="flex items-start space-x-2 p-3 bg-blue-50 rounded-lg">
                              <div className="w-2 h-2 bg-blue-600 rounded-full mt-2"></div>
                              <span className="text-blue-800 text-sm">{exercise}</span>
                            </div>
                          )) || <p className="text-gray-500">Maintain current activity level</p>}
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Medication Guidance */}
                  <div className="bg-white rounded-2xl shadow-lg p-8">
                    <h3 className="text-2xl font-bold text-gray-900 mb-6 flex items-center space-x-3">
                      <Pill className="w-7 h-7 text-orange-600" />
                      <span>Medication Guidance</span>
                    </h3>
                    
                    <div className="bg-yellow-50 border-2 border-yellow-200 rounded-xl p-4 mb-6">
                      <div className="flex items-start space-x-3">
                        <AlertTriangle className="w-5 h-5 text-yellow-600 mt-0.5" />
                        <p className="text-yellow-800 text-sm">
                          <strong>Important:</strong> These are general medication categories. 
                          Actual prescriptions require consultation with a licensed physician.
                        </p>
                      </div>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      <div>
                        <h4 className="font-bold text-gray-900 mb-3">Over-the-Counter Options</h4>
                        <div className="space-y-2">
                          {selectedReport.assessment_data?.medication_guidance?.otc_recommendations?.map((med: string, index: number) => (
                            <div key={index} className="p-3 bg-green-50 rounded-lg">
                              <span className="text-green-800">{med}</span>
                            </div>
                          )) || <p className="text-gray-500">No specific OTC medications recommended</p>}
                        </div>
                      </div>

                      <div>
                        <h4 className="font-bold text-gray-900 mb-3">Prescription Categories</h4>
                        <div className="space-y-2">
                          {selectedReport.assessment_data?.medication_guidance?.prescription_categories?.map((category: string, index: number) => (
                            <div key={index} className="p-3 bg-blue-50 rounded-lg">
                              <span className="text-blue-800">{category}</span>
                            </div>
                          )) || <p className="text-gray-500">No prescription medications indicated</p>}
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Follow-up Plan */}
                  <div className="bg-white rounded-2xl shadow-lg p-8">
                    <h3 className="text-2xl font-bold text-gray-900 mb-6 flex items-center space-x-3">
                      <Clock className="w-7 h-7 text-purple-600" />
                      <span>Follow-up & Monitoring Plan</span>
                    </h3>
                    
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      <div className="bg-blue-50 rounded-xl p-6 border border-blue-200">
                        <h4 className="font-bold text-blue-900 mb-3">Reassessment Timeline</h4>
                        <p className="text-blue-800 text-lg font-medium">
                          {selectedReport.assessment_data?.follow_up_plan?.reassessment_timeline || 'Follow up as needed'}
                        </p>
                      </div>

                      <div className="bg-green-50 rounded-xl p-6 border border-green-200">
                        <h4 className="font-bold text-green-900 mb-3">Next Steps</h4>
                        <div className="space-y-2">
                          {selectedReport.assessment_data?.follow_up_plan?.next_steps?.slice(0, 3).map((step: string, index: number) => (
                            <div key={index} className="flex items-start space-x-2">
                              <div className="bg-green-600 text-white rounded-full w-5 h-5 flex items-center justify-center text-xs font-bold mt-0.5">
                                {index + 1}
                              </div>
                              <span className="text-green-800 text-sm">{step}</span>
                            </div>
                          )) || <p className="text-gray-500">Continue current care plan</p>}
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            ) : (
              <div className="bg-white rounded-2xl shadow-lg p-12 text-center">
                <Brain className="w-20 h-20 text-gray-300 mx-auto mb-6" />
                <h3 className="text-2xl font-bold text-gray-900 mb-4">No Diagnosis Report Selected</h3>
                <p className="text-gray-600 mb-6">
                  Select a report from the sidebar to view detailed AI analysis, or complete a new intake form.
                </p>
                <button
                  onClick={() => window.location.href = '/intake-form'}
                  className="bg-gradient-to-r from-blue-600 to-purple-600 text-white px-8 py-3 rounded-xl font-semibold hover:shadow-lg transition-all duration-200 flex items-center space-x-2 mx-auto"
                >
                  <FileText className="w-5 h-5" />
                  <span>Complete New Assessment</span>
                </button>
              </div>
            )}
          </motion.div>
        </div>
      </div>
    </>
  );
};