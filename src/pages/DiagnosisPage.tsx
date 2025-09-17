import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useAuth } from '../contexts/AuthContext';
import { apiClient } from '../lib/api';
import { DiagnosisModal } from '../components/DiagnosisModal';
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
  Shield,
  Plus,
  Stethoscope,
  Upload,
  BarChart3
} from 'lucide-react';

interface DiagnosisReport {
  _id: string;
  user_id: string;
  assessment_data: any;
  created_at: string;
  status: string;
  report_type: string;
  confidence_score: number;
  urgency_level: string;
  primary_diagnosis: string;
}

export const DiagnosisPage: React.FC = () => {
  const { user } = useAuth();
  const [reports, setReports] = useState<DiagnosisReport[]>([]);
  const [selectedReport, setSelectedReport] = useState<DiagnosisReport | null>(null);
  const [loading, setLoading] = useState(true);
  const [showIntakeModal, setShowIntakeModal] = useState(false);
  const [showDiagnosisModal, setShowDiagnosisModal] = useState(false);
  const [diagnosisData, setDiagnosisData] = useState<any>(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [filterUrgency, setFilterUrgency] = useState('all');
  const [sortBy, setSortBy] = useState('date_desc');

  useEffect(() => {
    loadDiagnosisReports();
  }, []);

  const loadDiagnosisReports = async () => {
    try {
      const data = await apiClient.getDiagnosisReports();
      const processedReports = (data.reports || []).map((report: any) => ({
        _id: report._id || report.id,
        user_id: report.user_id,
        assessment_data: report.assessment_data,
        created_at: report.created_at,
        status: report.status || 'completed',
        report_type: report.report_type || 'comprehensive_assessment',
        confidence_score: report.assessment_data?.preliminary_assessment?.confidence_score || 0.85,
        urgency_level: report.assessment_data?.preliminary_assessment?.urgency_level || 'MEDIUM',
        primary_diagnosis: report.assessment_data?.preliminary_assessment?.primary_diagnosis || 'Medical Assessment'
      }));
      
      setReports(processedReports);
      
      if (processedReports.length > 0) {
        setSelectedReport(processedReports[0]);
      }
    } catch (error) {
      console.error('Failed to load diagnosis reports:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleIntakeComplete = (assessmentData: any) => {
    setDiagnosisData(assessmentData);
    setShowIntakeModal(false);
    setShowDiagnosisModal(true);
    // Reload reports to show new assessment
    loadDiagnosisReports();
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
    const matchesSearch = report.primary_diagnosis.toLowerCase().includes(searchTerm.toLowerCase());
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
      <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-blue-50 via-white to-purple-50">
        <div className="text-center">
          <div className="relative">
            <div className="animate-spin rounded-full h-20 w-20 border-4 border-blue-200 border-t-blue-600 mx-auto mb-4"></div>
            <Brain className="absolute inset-0 m-auto w-8 h-8 text-blue-600" />
          </div>
          <p className="text-gray-600 text-lg font-medium">Loading your medical assessments...</p>
          <p className="text-gray-500 text-sm mt-2">Powered by Gemini AI</p>
        </div>
      </div>
    );
  }

  return (
    <>
      <div className="min-h-screen py-8 px-4 sm:px-6 lg:px-8 bg-gradient-to-br from-blue-50 via-white to-purple-50">
        <div className="max-w-7xl mx-auto">
          {/* Enhanced Header with Graphics */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-center mb-12"
          >
            <div className="relative mb-8">
              <div className="absolute inset-0 bg-gradient-to-r from-blue-600 to-purple-600 w-32 h-32 rounded-full opacity-20 blur-xl mx-auto"></div>
              <div className="relative bg-gradient-to-r from-blue-600 to-purple-600 w-24 h-24 rounded-3xl flex items-center justify-center mx-auto mb-6 shadow-2xl">
                <Brain className="w-12 h-12 text-white" />
              </div>
            </div>
            <h1 className="text-5xl font-bold bg-gradient-to-r from-blue-600 via-purple-600 to-pink-600 bg-clip-text text-transparent mb-4">
              MediScan AI Diagnosis Center
            </h1>
            <p className="text-xl text-gray-600 mb-8">Advanced AI-powered medical analysis and clinical decision support</p>
            
            {/* Enhanced Stats Cards */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mt-12">
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.1 }}
                className="bg-white/80 backdrop-blur-sm rounded-2xl p-6 shadow-xl border border-white/20"
              >
                <div className="flex items-center space-x-3 mb-2">
                  <div className="bg-blue-100 p-3 rounded-xl">
                    <BarChart3 className="w-6 h-6 text-blue-600" />
                  </div>
                  <div>
                    <div className="text-3xl font-bold text-blue-600">{reports.length}</div>
                    <div className="text-sm text-gray-600">Total Assessments</div>
                  </div>
                </div>
              </motion.div>
              
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.2 }}
                className="bg-white/80 backdrop-blur-sm rounded-2xl p-6 shadow-xl border border-white/20"
              >
                <div className="flex items-center space-x-3 mb-2">
                  <div className="bg-green-100 p-3 rounded-xl">
                    <CheckCircle className="w-6 h-6 text-green-600" />
                  </div>
                  <div>
                    <div className="text-3xl font-bold text-green-600">
                      {reports.filter(r => r.urgency_level === 'LOW').length}
                    </div>
                    <div className="text-sm text-gray-600">Low Priority</div>
                  </div>
                </div>
              </motion.div>
              
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.3 }}
                className="bg-white/80 backdrop-blur-sm rounded-2xl p-6 shadow-xl border border-white/20"
              >
                <div className="flex items-center space-x-3 mb-2">
                  <div className="bg-orange-100 p-3 rounded-xl">
                    <AlertTriangle className="w-6 h-6 text-orange-600" />
                  </div>
                  <div>
                    <div className="text-3xl font-bold text-orange-600">
                      {reports.filter(r => ['MEDIUM', 'HIGH'].includes(r.urgency_level)).length}
                    </div>
                    <div className="text-sm text-gray-600">Needs Attention</div>
                  </div>
                </div>
              </motion.div>
              
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.4 }}
                className="bg-white/80 backdrop-blur-sm rounded-2xl p-6 shadow-xl border border-white/20"
              >
                <div className="flex items-center space-x-3 mb-2">
                  <div className="bg-purple-100 p-3 rounded-xl">
                    <Star className="w-6 h-6 text-purple-600" />
                  </div>
                  <div>
                    <div className="text-3xl font-bold text-purple-600">
                      {Math.round(reports.reduce((acc, r) => acc + r.confidence_score, 0) / reports.length * 100) || 0}%
                    </div>
                    <div className="text-sm text-gray-600">Avg Confidence</div>
                  </div>
                </div>
              </motion.div>
            </div>
          </motion.div>

          <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
            {/* Enhanced Sidebar */}
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              className="lg:col-span-1"
            >
              <div className="bg-white/90 backdrop-blur-sm rounded-3xl shadow-2xl p-6 sticky top-8 border border-white/20">
                <div className="flex items-center justify-between mb-6">
                  <h2 className="text-xl font-bold text-gray-900">Medical Reports</h2>
                  <button
                    onClick={loadDiagnosisReports}
                    className="p-2 bg-blue-100 text-blue-600 rounded-xl hover:bg-blue-200 transition-all duration-200"
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
                      className="w-full pl-10 pr-4 py-3 border border-gray-200 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 bg-white/50"
                    />
                  </div>

                  <div className="grid grid-cols-2 gap-2">
                    <select
                      value={filterUrgency}
                      onChange={(e) => setFilterUrgency(e.target.value)}
                      className="px-3 py-2 border border-gray-200 rounded-xl text-sm focus:ring-2 focus:ring-blue-500 bg-white/50"
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
                      className="px-3 py-2 border border-gray-200 rounded-xl text-sm focus:ring-2 focus:ring-blue-500 bg-white/50"
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
                      <div className="bg-gradient-to-r from-blue-100 to-purple-100 w-16 h-16 rounded-2xl flex items-center justify-center mx-auto mb-4">
                        <FileText className="w-8 h-8 text-blue-600" />
                      </div>
                      <p className="text-gray-600 text-sm mb-4">No assessments found</p>
                      <button
                        onClick={() => setShowIntakeModal(true)}
                        className="text-blue-600 text-sm font-medium hover:text-blue-700 flex items-center space-x-1 mx-auto"
                      >
                        <Plus className="w-4 h-4" />
                        <span>Start New Assessment</span>
                      </button>
                    </div>
                  ) : (
                    sortedReports.map((report) => (
                      <motion.div
                        key={report._id}
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        onClick={() => setSelectedReport(report)}
                        className={`p-4 rounded-2xl cursor-pointer transition-all duration-300 border-2 ${
                          selectedReport?._id === report._id
                            ? 'bg-gradient-to-r from-blue-50 to-purple-50 border-blue-300 shadow-lg'
                            : 'bg-white/60 border-gray-200 hover:bg-blue-50 hover:border-blue-200 hover:shadow-md'
                        }`}
                      >
                        <div className="flex items-start space-x-3">
                          <div className="bg-gradient-to-r from-blue-500 to-purple-500 p-2 rounded-xl">
                            <Brain className="w-4 h-4 text-white" />
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
                      </motion.div>
                    ))
                  )}
                </div>

                {/* Enhanced Action Buttons */}
                <div className="mt-6 space-y-3">
                  <button
                    onClick={() => setShowIntakeModal(true)}
                    className="w-full bg-gradient-to-r from-blue-600 via-purple-600 to-pink-600 text-white py-4 px-4 rounded-2xl font-bold hover:shadow-xl transition-all duration-300 flex items-center justify-center space-x-2 transform hover:scale-105"
                  >
                    <Brain className="w-5 h-5" />
                    <span>New AI Assessment</span>
                  </button>
                  <button
                    onClick={() => window.location.href = '/imaging'}
                    className="w-full bg-gradient-to-r from-green-500 to-teal-500 text-white py-3 px-4 rounded-2xl font-semibold hover:shadow-lg transition-all duration-300 flex items-center justify-center space-x-2"
                  >
                    <Upload className="w-5 h-5" />
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
                <div className="space-y-8">
                  {/* Enhanced Report Header */}
                  <div className="bg-white/90 backdrop-blur-sm rounded-3xl shadow-2xl p-8 border border-white/20">
                    <div className="flex items-center justify-between mb-8">
                      <div>
                        <h2 className="text-4xl font-bold bg-gradient-to-r from-gray-900 to-gray-600 bg-clip-text text-transparent mb-3">
                          {selectedReport.primary_diagnosis}
                        </h2>
                        <div className="flex items-center space-x-6 text-gray-600">
                          <div className="flex items-center space-x-2">
                            <Calendar className="w-5 h-5" />
                            <span className="font-medium">{formatDate(selectedReport.created_at)}</span>
                          </div>
                          <div className="flex items-center space-x-2">
                            <User className="w-5 h-5" />
                            <span>ID: {selectedReport._id.slice(0, 8)}</span>
                          </div>
                          <div className="flex items-center space-x-2">
                            <Shield className="w-5 h-5" />
                            <span>Gemini AI Powered</span>
                          </div>
                        </div>
                      </div>
                      <div className="flex items-center space-x-6">
                        <div className="text-right">
                          <div className="text-sm text-gray-600 mb-2">AI Confidence</div>
                          <div className="flex items-center space-x-3">
                            <div className="bg-gray-200 rounded-full h-3 w-32">
                              <div 
                                className="bg-gradient-to-r from-blue-500 to-purple-500 h-3 rounded-full transition-all duration-1000"
                                style={{ width: `${selectedReport.confidence_score * 100}%` }}
                              />
                            </div>
                            <span className={`font-bold text-xl ${getConfidenceColor(selectedReport.confidence_score)}`}>
                              {Math.round(selectedReport.confidence_score * 100)}%
                            </span>
                          </div>
                        </div>
                        <div className={`px-6 py-3 rounded-2xl border-2 font-bold text-lg ${getUrgencyColor(selectedReport.urgency_level)}`}>
                          {selectedReport.urgency_level} Priority
                        </div>
                      </div>
                    </div>

                    {/* Enhanced Action Buttons */}
                    <div className="flex flex-wrap gap-4">
                      <button className="flex items-center space-x-2 px-6 py-3 bg-gradient-to-r from-blue-500 to-blue-600 text-white rounded-xl hover:shadow-lg transition-all duration-200">
                        <Download className="w-5 h-5" />
                        <span>Download PDF</span>
                      </button>
                      <button className="flex items-center space-x-2 px-6 py-3 bg-gradient-to-r from-green-500 to-green-600 text-white rounded-xl hover:shadow-lg transition-all duration-200">
                        <Share className="w-5 h-5" />
                        <span>Share with Doctor</span>
                      </button>
                      <button className="flex items-center space-x-2 px-6 py-3 bg-gradient-to-r from-purple-500 to-purple-600 text-white rounded-xl hover:shadow-lg transition-all duration-200">
                        <Bookmark className="w-5 h-5" />
                        <span>Save to Favorites</span>
                      </button>
                      <button
                        onClick={() => window.location.href = '/book-appointment'}
                        className="flex items-center space-x-2 px-6 py-3 bg-gradient-to-r from-pink-500 to-red-500 text-white rounded-xl hover:shadow-lg transition-all duration-200"
                      >
                        <Calendar className="w-5 h-5" />
                        <span>Book Follow-up</span>
                      </button>
                    </div>
                  </div>

                  {/* Enhanced Assessment Cards */}
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                    <motion.div
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: 0.2 }}
                      className="bg-white/90 backdrop-blur-sm rounded-2xl shadow-xl p-6 border-l-4 border-blue-500"
                    >
                      <div className="flex items-center space-x-3 mb-4">
                        <div className="bg-blue-100 p-3 rounded-xl">
                          <Heart className="w-6 h-6 text-blue-600" />
                        </div>
                        <h3 className="text-lg font-bold text-gray-900">Health Status</h3>
                      </div>
                      <p className="text-gray-700 leading-relaxed">
                        {selectedReport.assessment_data?.preliminary_assessment?.overall_health_status || 'Comprehensive assessment completed with AI analysis'}
                      </p>
                    </motion.div>

                    <motion.div
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: 0.3 }}
                      className="bg-white/90 backdrop-blur-sm rounded-2xl shadow-xl p-6 border-l-4 border-purple-500"
                    >
                      <div className="flex items-center space-x-3 mb-4">
                        <div className="bg-purple-100 p-3 rounded-xl">
                          <Brain className="w-6 h-6 text-purple-600" />
                        </div>
                        <h3 className="text-lg font-bold text-gray-900">AI Analysis</h3>
                      </div>
                      <p className="text-gray-700 leading-relaxed">
                        Advanced Gemini AI analysis with clinical reasoning and evidence-based recommendations
                      </p>
                    </motion.div>

                    <motion.div
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: 0.4 }}
                      className="bg-white/90 backdrop-blur-sm rounded-2xl shadow-xl p-6 border-l-4 border-green-500"
                    >
                      <div className="flex items-center space-x-3 mb-4">
                        <div className="bg-green-100 p-3 rounded-xl">
                          <TrendingUp className="w-6 h-6 text-green-600" />
                        </div>
                        <h3 className="text-lg font-bold text-gray-900">Prognosis</h3>
                      </div>
                      <p className="text-gray-700 leading-relaxed">
                        {selectedReport.assessment_data?.patient_education?.prognosis || 'Positive outlook with appropriate medical care and lifestyle modifications'}
                      </p>
                    </motion.div>
                  </div>

                  {/* Detailed Report Content */}
                  <div className="bg-white/90 backdrop-blur-sm rounded-3xl shadow-2xl p-8 border border-white/20">
                    <h3 className="text-2xl font-bold text-gray-900 mb-6 flex items-center space-x-3">
                      <Activity className="w-7 h-7 text-blue-600" />
                      <span>Comprehensive Medical Analysis</span>
                    </h3>
                    
                    <div className="prose max-w-none">
                      <div className="bg-blue-50 rounded-2xl p-6 mb-6">
                        <h4 className="font-bold text-blue-900 mb-3">Clinical Assessment</h4>
                        <p className="text-blue-800 leading-relaxed">
                          {JSON.stringify(selectedReport.assessment_data, null, 2)}
                        </p>
                      </div>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="bg-white/90 backdrop-blur-sm rounded-3xl shadow-2xl p-12 text-center border border-white/20">
                  <div className="relative mb-8">
                    <div className="absolute inset-0 bg-gradient-to-r from-blue-600 to-purple-600 w-24 h-24 rounded-full opacity-20 blur-xl mx-auto"></div>
                    <div className="relative bg-gradient-to-r from-blue-100 to-purple-100 w-20 h-20 rounded-2xl flex items-center justify-center mx-auto">
                      <Brain className="w-10 h-10 text-blue-600" />
                    </div>
                  </div>
                  <h3 className="text-3xl font-bold text-gray-900 mb-4">Welcome to MediScan AI</h3>
                  <p className="text-gray-600 mb-8 text-lg leading-relaxed">
                    Start your medical assessment journey with our advanced AI-powered diagnostic support system
                  </p>
                  <div className="flex flex-col sm:flex-row gap-4 justify-center">
                    <button
                      onClick={() => setShowIntakeModal(true)}
                      className="bg-gradient-to-r from-blue-600 via-purple-600 to-pink-600 text-white px-8 py-4 rounded-2xl font-bold text-lg hover:shadow-xl transition-all duration-300 flex items-center space-x-3 transform hover:scale-105"
                    >
                      <Stethoscope className="w-6 h-6" />
                      <span>Start Medical Assessment</span>
                    </button>
                    <button
                      onClick={() => window.location.href = '/imaging'}
                      className="bg-gradient-to-r from-green-500 to-teal-500 text-white px-8 py-4 rounded-2xl font-bold text-lg hover:shadow-xl transition-all duration-300 flex items-center space-x-3"
                    >
                      <Upload className="w-6 h-6" />
                      <span>Upload Medical Reports</span>
                    </button>
                  </div>
                </div>
              )}
            </motion.div>
          </div>
        </div>
      </div>

      {/* Intake Form Modal */}
      <AnimatePresence>
        {showIntakeModal && (
          <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center p-4 z-50">
            <motion.div
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.9 }}
              className="bg-white rounded-3xl shadow-2xl max-w-4xl w-full max-h-[90vh] overflow-hidden"
            >
              <div className="p-6 border-b border-gray-200">
                <div className="flex items-center justify-between">
                  <h2 className="text-2xl font-bold text-gray-900">Medical Intake Assessment</h2>
                  <button
                    onClick={() => setShowIntakeModal(false)}
                    className="p-2 bg-gray-100 rounded-xl hover:bg-gray-200 transition-all duration-200"
                  >
                    ✕
                  </button>
                </div>
              </div>
              <div className="p-6 overflow-y-auto max-h-[calc(90vh-120px)]">
                {/* Intake form component would be embedded here */}
                <div className="text-center py-12">
                  <Brain className="w-16 h-16 text-blue-600 mx-auto mb-4" />
                  <h3 className="text-xl font-bold text-gray-900 mb-2">Medical Intake Form</h3>
                  <p className="text-gray-600">Complete assessment form will be integrated here</p>
                  <button
                    onClick={() => {
                      // Simulate form completion
                      const mockAssessment = {
                        assessment_data: {
                          preliminary_assessment: {
                            primary_diagnosis: "Viral Upper Respiratory Infection",
                            confidence_score: 0.87,
                            urgency_level: "LOW"
                          }
                        }
                      };
                      handleIntakeComplete(mockAssessment);
                    }}
                    className="mt-4 bg-blue-600 text-white px-6 py-2 rounded-xl hover:bg-blue-700"
                  >
                    Simulate Assessment Complete
                  </button>
                </div>
              </div>
            </motion.div>
          </div>
        )}
      </AnimatePresence>

      {/* Diagnosis Modal */}
      <DiagnosisModal
        isOpen={showDiagnosisModal}
        onClose={() => setShowDiagnosisModal(false)}
        diagnosisData={diagnosisData}
        onSaveToDashboard={() => {
          setShowDiagnosisModal(false);
          loadDiagnosisReports();
        }}
        onBookAppointment={() => window.location.href = '/book-appointment'}
      />
    </>
  );
};