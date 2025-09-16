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
  RefreshCw,
  Eye,
  Filter,
  Search,
  Star,
  Shield
} from 'lucide-react';

interface DiagnosisReport {
  id: string;
  patient_id: string;
  assessment_data: any;
  created_at: string;
  status: string;
  report_type: string;
  confidence_score: number;
  urgency_level: string;
  primary_diagnosis: string;
}

export const DiagnosisPage: React.FC = () => {
  const [searchParams] = useSearchParams();
  const [reports, setReports] = useState<DiagnosisReport[]>([]);
  const [selectedReport, setSelectedReport] = useState<DiagnosisReport | null>(null);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState('recent');
  const [searchTerm, setSearchTerm] = useState('');
  const [filterUrgency, setFilterUrgency] = useState('all');
  const [sortBy, setSortBy] = useState('date_desc');

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
        const processedReports = (data.reports || []).map((report: any) => ({
          id: report._id || report.id,
          patient_id: report.user_id,
          assessment_data: report.assessment_data,
          created_at: report.created_at,
          status: report.status || 'completed',
          report_type: report.report_type || 'triage_assessment',
          confidence_score: report.assessment_data?.preliminary_assessment?.confidence_score || 0,
          urgency_level: report.assessment_data?.preliminary_assessment?.urgency_level || 'LOW',
          primary_diagnosis: report.assessment_data?.preliminary_assessment?.primary_diagnosis || 'Medical Assessment'
        }));
        
        setReports(processedReports);
        
        // Auto-select the most recent report
        if (processedReports.length > 0) {
          setSelectedReport(processedReports[0]);
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
        const processedReport = {
          id: report._id || report.id,
          patient_id: report.user_id,
          assessment_data: report.assessment_data,
          created_at: report.created_at,
          status: report.status || 'completed',
          report_type: report.report_type || 'triage_assessment',
          confidence_score: report.assessment_data?.preliminary_assessment?.confidence_score || 0,
          urgency_level: report.assessment_data?.preliminary_assessment?.urgency_level || 'LOW',
          primary_diagnosis: report.assessment_data?.preliminary_assessment?.primary_diagnosis || 'Medical Assessment'
        };
        setSelectedReport(processedReport);
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

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'text-green-600';
    if (confidence >= 0.6) return 'text-yellow-600';
    return 'text-red-600';
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

  const filteredReports = reports.filter(report => {
    const matchesSearch = report.primary_diagnosis.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         report.assessment_data?.clinical_analysis?.probable_causes?.some((cause: string) => 
                           cause.toLowerCase().includes(searchTerm.toLowerCase()));
    
    const matchesUrgency = filterUrgency === 'all' || report.urgency_level.toLowerCase() === filterUrgency;
    
    return matchesSearch && matchesUrgency;
  });

  const sortedReports = [...filteredReports].sort((a, b) => {
    switch (sortBy) {
      case 'date_desc':
        return new Date(b.created_at).getTime() - new Date(a.created_at).getTime();
      case 'date_asc':
        return new Date(a.created_at).getTime() - new Date(b.created_at).getTime();
      case 'confidence_desc':
        return b.confidence_score - a.confidence_score;
      case 'urgency_desc':
        const urgencyOrder = { 'EMERGENCY': 4, 'HIGH': 3, 'MEDIUM': 2, 'LOW': 1 };
        return (urgencyOrder[b.urgency_level as keyof typeof urgencyOrder] || 0) - 
               (urgencyOrder[a.urgency_level as keyof typeof urgencyOrder] || 0);
      default:
        return 0;
    }
  });

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-50">
        <div className="text-center">
          <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <p className="text-gray-600 text-lg">Loading your medical assessments...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen py-8 px-4 sm:px-6 lg:px-8 bg-gray-50">
      <div className="max-w-7xl mx-auto">
        {/* Enhanced Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-8"
        >
          <div className="bg-gradient-to-r from-blue-600 to-purple-600 w-24 h-24 rounded-3xl flex items-center justify-center mx-auto mb-6">
            <Brain className="w-12 h-12 text-white" />
          </div>
          <h1 className="text-4xl font-bold text-gray-900 mb-3">AI Diagnosis Center</h1>
          <p className="text-xl text-gray-600">Your comprehensive medical assessments and AI-powered insights</p>
          
          {/* Quick Stats */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mt-8">
            <div className="bg-white rounded-xl p-4 shadow-md">
              <div className="text-2xl font-bold text-blue-600">{reports.length}</div>
              <div className="text-sm text-gray-600">Total Assessments</div>
            </div>
            <div className="bg-white rounded-xl p-4 shadow-md">
              <div className="text-2xl font-bold text-green-600">
                {reports.filter(r => r.urgency_level === 'LOW').length}
              </div>
              <div className="text-sm text-gray-600">Low Priority</div>
            </div>
            <div className="bg-white rounded-xl p-4 shadow-md">
              <div className="text-2xl font-bold text-orange-600">
                {reports.filter(r => ['MEDIUM', 'HIGH'].includes(r.urgency_level)).length}
              </div>
              <div className="text-sm text-gray-600">Needs Attention</div>
            </div>
            <div className="bg-white rounded-xl p-4 shadow-md">
              <div className="text-2xl font-bold text-purple-600">
                {Math.round(reports.reduce((acc, r) => acc + r.confidence_score, 0) / reports.length * 100) || 0}%
              </div>
              <div className="text-sm text-gray-600">Avg Confidence</div>
            </div>
          </div>
        </motion.div>

        <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
          {/* Enhanced Sidebar */}
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

              {/* Search and Filters */}
              <div className="space-y-4 mb-6">
                <div className="relative">
                  <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
                  <input
                    type="text"
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                    placeholder="Search reports..."
                    className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  />
                </div>

                <div className="grid grid-cols-2 gap-2">
                  <select
                    value={filterUrgency}
                    onChange={(e) => setFilterUrgency(e.target.value)}
                    className="px-3 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-blue-500"
                  >
                    <option value="all">All Urgency</option>
                    <option value="emergency">Emergency</option>
                    <option value="high">High</option>
                    <option value="medium">Medium</option>
                    <option value="low">Low</option>
                  </select>

                  <select
                    value={sortBy}
                    onChange={(e) => setSortBy(e.target.value)}
                    className="px-3 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-blue-500"
                  >
                    <option value="date_desc">Newest First</option>
                    <option value="date_asc">Oldest First</option>
                    <option value="confidence_desc">High Confidence</option>
                    <option value="urgency_desc">High Urgency</option>
                  </select>
                </div>
              </div>

              {/* Reports List */}
              <div className="space-y-3 max-h-96 overflow-y-auto">
                {sortedReports.length === 0 ? (
                  <div className="text-center py-8">
                    <FileText className="w-12 h-12 text-gray-300 mx-auto mb-3" />
                    <p className="text-gray-600 text-sm">No reports found</p>
                    <button
                      onClick={() => window.location.href = '/intake-form'}
                      className="mt-3 text-blue-600 text-sm font-medium hover:text-blue-700"
                    >
                      Complete New Assessment →
                    </button>
                  </div>
                ) : (
                  sortedReports.map((report) => (
                    <div
                      key={report.id}
                      onClick={() => setSelectedReport(report)}
                      className={`p-4 rounded-xl cursor-pointer transition-all duration-200 border-2 ${
                        selectedReport?.id === report.id
                          ?   'bg-blue-50 border-blue-300 shadow-md'
                          : 'bg-gray-50 border-gray-200 hover:bg-blue-50 hover:border-blue-200'
                      }`}
                    >
                      <div className="flex items-start space-x-3">
                        <div className="bg-blue-100 p-2 rounded-lg">
                          <Brain className="w-4 h-4 text-blue-600" />
                        </div>
                        <div className="flex-1 min-w-0">
                          <h4 className="font-semibold text-gray-900 text-sm truncate">
                            {report.primary_diagnosis}
                          </h4>
                          <p className="text-xs text-gray-600 mt-1">
                            {formatDate(report.created_at)}
                          </p>
                          <div className="flex items-center space-x-2 mt-2">
                            <span className={`inline-block px-2 py-1 rounded-full text-xs font-medium border ${getUrgencyColor(report.urgency_level)}`}>
                              {report.urgency_level}
                            </span>
                            <div className="flex items-center space-x-1">
                              <Star className="w-3 h-3 text-yellow-500" />
                              <span className={`text-xs font-medium ${getConfidenceColor(report.confidence_score)}`}>
                                {Math.round(report.confidence_score * 100)}%
                              </span>
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                  ))
                )}
              </div>

              {/* Quick Actions */}
              <div className="mt-6 space-y-3">
                <button
                  onClick={() => window.location.href = '/intake-form'}
                  className="w-full bg-gradient-to-r from-blue-600 to-purple-600 text-white py-3 px-4 rounded-xl font-semibold hover:shadow-lg transition-all duration-200 flex items-center justify-center space-x-2"
                >
                  <Brain className="w-5 h-5" />
                  <span>New Assessment</span>
                </button>
                <button
                  onClick={() => window.location.href = '/upload-reports'}
                  className="w-full bg-green-600 text-white py-3 px-4 rounded-xl font-semibold hover:bg-green-700 transition-all duration-200 flex items-center justify-center space-x-2"
                >
                  <FileText className="w-5 h-5" />
                  <span>Upload Reports</span>
                </button>
              </div>
            </div>
          </motion.div>

          {/* Enhanced Main Content */}
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
                        {selectedReport.primary_diagnosis}
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
                        <div className="flex items-center space-x-1">
                          <Shield className="w-4 h-4" />
                          <span>AI Powered</span>
                        </div>
                      </div>
                    </div>
                    <div className="flex items-center space-x-4">
                      <div className="text-right">
                        <div className="text-sm text-gray-600 mb-1">AI Confidence</div>
                        <div className="flex items-center space-x-2">
                          <div className="bg-gray-200 rounded-full h-2 w-24">
                            <div 
                              className="bg-blue-500 h-2 rounded-full transition-all duration-1000"
                              style={{ width: `${selectedReport.confidence_score * 100}%` }}
                            />
                          </div>
                          <span className={`font-bold text-lg ${getConfidenceColor(selectedReport.confidence_score)}`}>
                            {Math.round(selectedReport.confidence_score * 100)}%
                          </span>
                        </div>
                      </div>
                      <div className={`px-4 py-2 rounded-xl border-2 font-semibold ${getUrgencyColor(selectedReport.urgency_level)}`}>
                        {selectedReport.urgency_level} Priority
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

                {/* Assessment Overview Cards */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                  <div className="bg-white rounded-2xl shadow-lg p-6 border-l-4 border-blue-500">
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

                  <div className="bg-white rounded-2xl shadow-lg p-6 border-l-4 border-purple-500">
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

                  <div className="bg-white rounded-2xl shadow-lg p-6 border-l-4 border-green-500">
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

                    {/* Clinical Reasoning */}
                    <div className="mt-6 bg-blue-50 rounded-xl p-6">
                      <h4 className="font-bold text-blue-900 mb-3">Clinical Reasoning</h4>
                      <p className="text-blue-800 leading-relaxed">
                        {selectedReport.assessment_data?.preliminary_assessment?.clinical_reasoning || 'Detailed clinical reasoning provided by AI analysis'}
                      </p>
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
                        <User className="w-5 h-5 text-green-600" />
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

                    {/* Timeline */}
                    <div className="mt-6 bg-orange-50 rounded-xl p-6">
                      <h4 className="font-bold text-orange-900 mb-3 flex items-center space-x-2">
                        <Clock className="w-5 h-5" />
                        <span>Investigation Timeline</span>
                      </h4>
                      <p className="text-orange-800 font-medium">
                        {selectedReport.assessment_data?.recommended_investigations?.urgency_timeline || 'Complete investigations as recommended by healthcare provider'}
                      </p>
                    </div>
                  </div>

                  {/* Treatment Recommendations */}
                  <div className="bg-white rounded-2xl shadow-lg p-8">
                    <h3 className="text-2xl font-bold text-gray-900 mb-6 flex items-center space-x-3">
                      <Pill className="w-7 h-7 text-red-600" />
                      <span>Treatment Recommendations</span>
                    </h3>
                    
                    <div className="bg-yellow-50 border border-yellow-200 rounded-xl p-4 mb-6">
                      <div className="flex items-start space-x-3">
                        <AlertTriangle className="w-5 h-5 text-yellow-600 mt-0.5" />
                        <p className="text-yellow-800 text-sm">
                          <strong>Important:</strong> These are general treatment categories based on AI analysis. 
                          Actual prescriptions and treatments require consultation with licensed healthcare professionals.
                        </p>
                      </div>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      <div>
                        <h4 className="font-bold text-gray-900 mb-3">Immediate Interventions</h4>
                        <div className="space-y-2">
                          {selectedReport.assessment_data?.treatment_recommendations?.immediate_interventions?.map((intervention: string, index: number) => (
                            <div key={index} className="p-3 bg-red-50 rounded-lg border border-red-200">
                              <span className="text-red-800">{intervention}</span>
                            </div>
                          )) || <p className="text-gray-500">No immediate interventions required</p>}
                        </div>
                      </div>

                      <div>
                        <h4 className="font-bold text-gray-900 mb-3">Medication Categories</h4>
                        <div className="space-y-2">
                          {selectedReport.assessment_data?.treatment_recommendations?.medication_categories?.map((category: string, index: number) => (
                            <div key={index} className="p-3 bg-blue-50 rounded-lg">
                              <span className="text-blue-800">{category}</span>
                            </div>
                          )) || <p className="text-gray-500">No specific medications indicated</p>}
                        </div>
                      </div>
                    </div>

                    <div className="mt-6">
                      <h4 className="font-bold text-gray-900 mb-3">Non-Pharmacological Treatments</h4>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                        {selectedReport.assessment_data?.treatment_recommendations?.non_pharmacological?.map((treatment: string, index: number) => (
                          <div key={index} className="p-3 bg-green-50 rounded-lg border border-green-200">
                            <span className="text-green-800">{treatment}</span>
                          </div>
                        )) || <p className="text-gray-500">No specific non-drug treatments recommended</p>}
                      </div>
                    </div>
                  </div>

                  {/* Lifestyle Optimization */}
                  <div className="bg-white rounded-2xl shadow-lg p-8">
                    <h3 className="text-2xl font-bold text-gray-900 mb-6 flex items-center space-x-3">
                      <Heart className="w-7 h-7 text-pink-600" />
                      <span>Personalized Lifestyle Plan</span>
                    </h3>
                    
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      <div>
                        <h4 className="font-bold text-gray-900 mb-3">Diet Modifications</h4>
                        <div className="space-y-2">
                          {selectedReport.assessment_data?.lifestyle_optimization?.diet_modifications?.map((diet: string, index: number) => (
                            <div key={index} className="flex items-start space-x-2 p-3 bg-green-50 rounded-lg">
                              <div className="w-2 h-2 bg-green-600 rounded-full mt-2"></div>
                              <span className="text-green-800 text-sm">{diet}</span>
                            </div>
                          )) || <p className="text-gray-500">Continue current healthy diet</p>}
                        </div>
                      </div>

                      <div>
                        <h4 className="font-bold text-gray-900 mb-3">Exercise Prescription</h4>
                        <div className="space-y-2">
                          {selectedReport.assessment_data?.lifestyle_optimization?.exercise_prescription?.map((exercise: string, index: number) => (
                            <div key={index} className="flex items-start space-x-2 p-3 bg-blue-50 rounded-lg">
                              <div className="w-2 h-2 bg-blue-600 rounded-full mt-2"></div>
                              <span className="text-blue-800 text-sm">{exercise}</span>
                            </div>
                          )) || <p className="text-gray-500">Maintain current activity level</p>}
                        </div>
                      </div>
                    </div>

                    <div className="mt-6 grid grid-cols-1 md:grid-cols-2 gap-6">
                      <div>
                        <h4 className="font-bold text-gray-900 mb-3">Sleep Hygiene</h4>
                        <div className="space-y-2">
                          {selectedReport.assessment_data?.lifestyle_optimization?.sleep_hygiene?.map((tip: string, index: number) => (
                            <div key={index} className="flex items-start space-x-2 p-3 bg-purple-50 rounded-lg">
                              <div className="w-2 h-2 bg-purple-600 rounded-full mt-2"></div>
                              <span className="text-purple-800 text-sm">{tip}</span>
                            </div>
                          )) || <p className="text-gray-500">Maintain good sleep habits</p>}
                        </div>
                      </div>

                      <div>
                        <h4 className="font-bold text-gray-900 mb-3">Stress Management</h4>
                        <div className="space-y-2">
                          {selectedReport.assessment_data?.lifestyle_optimization?.stress_management?.map((strategy: string, index: number) => (
                            <div key={index} className="flex items-start space-x-2 p-3 bg-orange-50 rounded-lg">
                              <div className="w-2 h-2 bg-orange-600 rounded-full mt-2"></div>
                              <span className="text-orange-800 text-sm">{strategy}</span>
                            </div>
                          )) || <p className="text-gray-500">Continue current stress management techniques</p>}
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

                    <div className="mt-6">
                      <h4 className="font-bold text-gray-900 mb-3">Symptoms to Monitor</h4>
                      <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                        {selectedReport.assessment_data?.follow_up_plan?.symptom_tracking?.map((symptom: string, index: number) => (
                          <div key={index} className="p-3 bg-yellow-50 rounded-lg border border-yellow-200 text-center">
                            <span className="text-yellow-800 font-medium">{symptom}</span>
                          </div>
                        )) || <p className="text-gray-500">Monitor general symptoms</p>}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            ) : (
              <div className="bg-white rounded-2xl shadow-lg p-12 text-center">
                <Brain className="w-20 h-20 text-gray-300 mx-auto mb-6" />
                <h3 className="text-2xl font-bold text-gray-900 mb-4">No Assessment Selected</h3>
                <p className="text-gray-600 mb-6 text-lg">
                  Select a report from the sidebar to view detailed AI analysis, or complete a new assessment.
                </p>
                <div className="flex flex-col sm:flex-row gap-4 justify-center">
                  <button
                    onClick={() => window.location.href = '/intake-form'}
                    className="bg-gradient-to-r from-blue-600 to-purple-600 text-white px-8 py-3 rounded-xl font-semibold hover:shadow-lg transition-all duration-200 flex items-center space-x-2"
                  >
                    <FileText className="w-5 h-5" />
                    <span>Complete New Assessment</span>
                  </button>
                  <button
                    onClick={() => window.location.href = '/upload-reports'}
                    className="bg-green-600 text-white px-8 py-3 rounded-xl font-semibold hover:bg-green-700 transition-all duration-200 flex items-center space-x-2"
                  >
                    <FileText className="w-5 h-5" />
                    <span>Upload Medical Reports</span>
                  </button>
                </div>
              </div>
            )}
          </motion.div>
        </div>
      </div>
    </div>
  );
};