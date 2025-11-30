import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Brain, Plus, Calendar, TrendingUp, AlertCircle, CheckCircle, Clock, FileText, Download, Eye, Filter, Search } from 'lucide-react';
import IntakeFormModal from '../components/IntakeFormModal';

interface DiagnosisReport {
  id: string;
  date: string;
  patientName: string;
  chiefComplaint: string;
  confidence: number;
  urgency: 'low' | 'medium' | 'high' | 'critical';
  status: 'completed' | 'pending' | 'reviewed';
  findings: {
    summary: string;
    likelyConditions: string[];
    recommendations: string[];
    tests: string[];
  };
  mlAnalysis: {
    symptomScore: number;
    riskFactors: string[];
    predictedOutcome: string;
  };
}

const DiagnosisPage: React.FC = () => {
  const [reports, setReports] = useState<DiagnosisReport[]>([]);
  const [selectedReport, setSelectedReport] = useState<DiagnosisReport | null>(null);
  const [showIntakeForm, setShowIntakeForm] = useState(false);
  const [filterUrgency, setFilterUrgency] = useState<string>('all');
  const [searchTerm, setSearchTerm] = useState('');
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    // Simulate loading reports
    setTimeout(() => {
      setReports([
        {
          id: '1',
          date: '2024-01-15',
          patientName: 'John Doe',
          chiefComplaint: 'Chest pain and shortness of breath',
          confidence: 87.5,
          urgency: 'high',
          status: 'completed',
          findings: {
            summary: 'Patient presents with acute chest pain and dyspnea. ML analysis suggests possible cardiac etiology.',
            likelyConditions: ['Acute Coronary Syndrome', 'Pulmonary Embolism', 'Anxiety Disorder'],
            recommendations: ['Immediate ECG', 'Cardiac enzymes', 'Chest X-ray', 'Cardiology consultation'],
            tests: ['ECG', 'Troponin', 'D-dimer', 'Chest X-ray']
          },
          mlAnalysis: {
            symptomScore: 8.2,
            riskFactors: ['Age > 50', 'Smoking history', 'Hypertension'],
            predictedOutcome: 'Requires immediate medical attention'
          }
        },
        {
          id: '2',
          date: '2024-01-14',
          patientName: 'Jane Smith',
          chiefComplaint: 'Persistent headache and fatigue',
          confidence: 72.3,
          urgency: 'medium',
          status: 'completed',
          findings: {
            summary: 'Chronic headache pattern with associated fatigue. ML models suggest tension-type headache.',
            likelyConditions: ['Tension Headache', 'Migraine', 'Sleep Disorder'],
            recommendations: ['Sleep hygiene assessment', 'Stress management', 'Neurological evaluation if persistent'],
            tests: ['Blood pressure check', 'Basic metabolic panel', 'Sleep study consideration']
          },
          mlAnalysis: {
            symptomScore: 6.1,
            riskFactors: ['Stress', 'Poor sleep', 'Screen time'],
            predictedOutcome: 'Likely to improve with lifestyle modifications'
          }
        }
      ]);
      setIsLoading(false);
    }, 1000);
  }, []);

  const getUrgencyColor = (urgency: string) => {
    switch (urgency) {
      case 'critical': return 'text-red-600 bg-red-50 border-red-200';
      case 'high': return 'text-orange-600 bg-orange-50 border-orange-200';
      case 'medium': return 'text-yellow-600 bg-yellow-50 border-yellow-200';
      default: return 'text-green-600 bg-green-50 border-green-200';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed': return <CheckCircle className="w-4 h-4 text-green-500" />;
      case 'pending': return <Clock className="w-4 h-4 text-yellow-500" />;
      case 'reviewed': return <Eye className="w-4 h-4 text-blue-500" />;
      default: return <Clock className="w-4 h-4 text-gray-500" />;
    }
  };

  const filteredReports = reports.filter(report => {
    const matchesUrgency = filterUrgency === 'all' || report.urgency === filterUrgency;
    const matchesSearch = report.chiefComplaint.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         report.patientName.toLowerCase().includes(searchTerm.toLowerCase());
    return matchesUrgency && matchesSearch;
  });

  const handleNewDiagnosis = (formData: any) => {
    // Simulate ML analysis and create new report
    const newReport: DiagnosisReport = {
      id: Math.random().toString(36).substr(2, 9),
      date: new Date().toISOString().split('T')[0],
      patientName: `${formData.firstName} ${formData.lastName}`,
      chiefComplaint: formData.symptoms.join(', '),
      confidence: Math.random() * 20 + 75, // 75-95%
      urgency: Math.random() > 0.7 ? 'medium' : 'low',
      status: 'completed',
      findings: {
        summary: 'AI analysis completed based on provided symptoms and medical history.',
        likelyConditions: ['Condition A', 'Condition B', 'Condition C'],
        recommendations: ['Follow-up with primary care', 'Monitor symptoms', 'Lifestyle modifications'],
        tests: ['Basic blood work', 'Vital signs monitoring']
      },
      mlAnalysis: {
        symptomScore: Math.random() * 4 + 4, // 4-8
        riskFactors: ['Age', 'Medical history'],
        predictedOutcome: 'Good prognosis with proper care'
      }
    };
    
    setReports(prev => [newReport, ...prev]);
    setShowIntakeForm(false);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50">
      {/* Header */}
      <div className="bg-white/80 backdrop-blur-sm border-b border-gray-200/50 sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                AI Diagnosis Dashboard
              </h1>
              <p className="text-gray-600 mt-2">ML-powered medical analysis and diagnosis reports</p>
            </div>
            <button
              onClick={() => setShowIntakeForm(true)}
              className="bg-gradient-to-r from-blue-600 to-purple-600 text-white px-6 py-3 rounded-xl hover:shadow-lg transition-all duration-300 flex items-center space-x-2"
            >
              <Plus className="w-5 h-5" />
              <span>New Diagnosis</span>
            </button>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Stats Cards */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-white/80 backdrop-blur-sm rounded-2xl shadow-xl border border-gray-200/50 p-6"
          >
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600">Total Reports</p>
                <p className="text-2xl font-bold text-gray-800">{reports.length}</p>
              </div>
              <FileText className="w-8 h-8 text-blue-500" />
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="bg-white/80 backdrop-blur-sm rounded-2xl shadow-xl border border-gray-200/50 p-6"
          >
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600">High Priority</p>
                <p className="text-2xl font-bold text-orange-600">
                  {reports.filter(r => r.urgency === 'high' || r.urgency === 'critical').length}
                </p>
              </div>
              <AlertCircle className="w-8 h-8 text-orange-500" />
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="bg-white/80 backdrop-blur-sm rounded-2xl shadow-xl border border-gray-200/50 p-6"
          >
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600">Avg Confidence</p>
                <p className="text-2xl font-bold text-green-600">
                  {reports.length > 0 ? (reports.reduce((acc, r) => acc + r.confidence, 0) / reports.length).toFixed(1) : 0}%
                </p>
              </div>
              <TrendingUp className="w-8 h-8 text-green-500" />
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
            className="bg-white/80 backdrop-blur-sm rounded-2xl shadow-xl border border-gray-200/50 p-6"
          >
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600">This Month</p>
                <p className="text-2xl font-bold text-purple-600">
                  {reports.filter(r => new Date(r.date).getMonth() === new Date().getMonth()).length}
                </p>
              </div>
              <Calendar className="w-8 h-8 text-purple-500" />
            </div>
          </motion.div>
        </div>

        {/* Filters */}
        <div className="bg-white/80 backdrop-blur-sm rounded-2xl shadow-xl border border-gray-200/50 p-6 mb-8">
          <div className="flex flex-col md:flex-row md:items-center md:justify-between space-y-4 md:space-y-0">
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2">
                <Filter className="w-4 h-4 text-gray-500" />
                <select
                  value={filterUrgency}
                  onChange={(e) => setFilterUrgency(e.target.value)}
                  className="border border-gray-300 rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                >
                  <option value="all">All Urgency</option>
                  <option value="low">Low</option>
                  <option value="medium">Medium</option>
                  <option value="high">High</option>
                  <option value="critical">Critical</option>
                </select>
              </div>
            </div>
            
            <div className="flex items-center space-x-2">
              <Search className="w-4 h-4 text-gray-500" />
              <input
                type="text"
                placeholder="Search reports..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="border border-gray-300 rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-blue-500 focus:border-transparent w-64"
              />
            </div>
          </div>
        </div>

        {/* Reports List */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
          className="bg-white/80 backdrop-blur-sm rounded-2xl shadow-xl border border-gray-200/50 p-6"
        >
          <h2 className="text-xl font-semibold text-gray-800 mb-6 flex items-center">
            <Brain className="w-5 h-5 mr-2 text-purple-600" />
            Diagnosis Reports
          </h2>

          {isLoading ? (
            <div className="text-center py-12">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
              <p className="text-gray-500">Loading reports...</p>
            </div>
          ) : filteredReports.length === 0 ? (
            <div className="text-center py-12">
              <Brain className="w-16 h-16 text-gray-300 mx-auto mb-4" />
              <p className="text-gray-500 text-lg">No reports found</p>
              <p className="text-gray-400 text-sm">Start a new diagnosis to see results here</p>
            </div>
          ) : (
            <div className="space-y-4">
              {filteredReports.map((report) => (
                <motion.div
                  key={report.id}
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                  className="border border-gray-200 rounded-xl p-6 hover:shadow-lg transition-all duration-300 cursor-pointer"
                  onClick={() => setSelectedReport(report)}
                >
                  <div className="flex items-center justify-between mb-4">
                    <div className="flex items-center space-x-3">
                      {getStatusIcon(report.status)}
                      <div>
                        <h3 className="font-semibold text-gray-800">{report.patientName}</h3>
                        <p className="text-sm text-gray-500">{report.date}</p>
                      </div>
                    </div>
                    <div className="flex items-center space-x-3">
                      <span className={`px-3 py-1 rounded-full text-xs font-medium border ${getUrgencyColor(report.urgency)}`}>
                        {report.urgency.toUpperCase()}
                      </span>
                      <span className="text-sm text-gray-600">
                        {report.confidence.toFixed(1)}% confidence
                      </span>
                      <Eye className="w-4 h-4 text-gray-400" />
                    </div>
                  </div>
                  
                  <div className="mb-4">
                    <p className="text-sm text-gray-600 mb-2">Chief Complaint:</p>
                    <p className="text-gray-800">{report.chiefComplaint}</p>
                  </div>
                  
                  <div className="bg-gray-50 rounded-lg p-4">
                    <p className="text-sm text-gray-700">{report.findings.summary}</p>
                  </div>
                </motion.div>
              ))}
            </div>
          )}
        </motion.div>
      </div>

      {/* Intake Form Modal */}
      <AnimatePresence>
        {showIntakeForm && (
          <IntakeFormModal
            isOpen={showIntakeForm}
            onClose={() => setShowIntakeForm(false)}
            onSubmit={handleNewDiagnosis}
          />
        )}
      </AnimatePresence>

      {/* Detailed Report Modal */}
      <AnimatePresence>
        {selectedReport && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4"
            onClick={() => setSelectedReport(null)}
          >
            <motion.div
              initial={{ scale: 0.95, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.95, opacity: 0 }}
              className="bg-white rounded-2xl shadow-2xl max-w-4xl w-full max-h-[90vh] overflow-y-auto"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="p-6 border-b border-gray-200">
                <div className="flex items-center justify-between">
                  <h2 className="text-2xl font-bold text-gray-800">Detailed Analysis Report</h2>
                  <button
                    onClick={() => setSelectedReport(null)}
                    className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
                  >
                    ✕
                  </button>
                </div>
              </div>
              
              <div className="p-6">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
                  <div>
                    <h3 className="text-lg font-semibold text-gray-800 mb-3">Patient Information</h3>
                    <div className="space-y-2 text-sm">
                      <p><span className="font-medium">Name:</span> {selectedReport.patientName}</p>
                      <p><span className="font-medium">Date:</span> {selectedReport.date}</p>
                      <p><span className="font-medium">Status:</span> {selectedReport.status}</p>
                      <p><span className="font-medium">Confidence:</span> {selectedReport.confidence.toFixed(1)}%</p>
                    </div>
                  </div>
                  
                  <div>
                    <h3 className="text-lg font-semibold text-gray-800 mb-3">ML Analysis</h3>
                    <div className="space-y-2 text-sm">
                      <p><span className="font-medium">Symptom Score:</span> {selectedReport.mlAnalysis.symptomScore}/10</p>
                      <p><span className="font-medium">Predicted Outcome:</span> {selectedReport.mlAnalysis.predictedOutcome}</p>
                      <p><span className="font-medium">Risk Factors:</span> {selectedReport.mlAnalysis.riskFactors.join(', ')}</p>
                    </div>
                  </div>
                </div>
                
                <div className="mb-6">
                  <h3 className="text-lg font-semibold text-gray-800 mb-3">Chief Complaint</h3>
                  <p className="text-gray-700 bg-gray-50 p-4 rounded-lg">{selectedReport.chiefComplaint}</p>
                </div>
                
                <div className="mb-6">
                  <h3 className="text-lg font-semibold text-gray-800 mb-3">Clinical Summary</h3>
                  <p className="text-gray-700">{selectedReport.findings.summary}</p>
                </div>
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
                  <div>
                    <h3 className="text-lg font-semibold text-gray-800 mb-3">Likely Conditions</h3>
                    <ul className="space-y-2">
                      {selectedReport.findings.likelyConditions.map((condition, index) => (
                        <li key={index} className="flex items-start space-x-2">
                          <CheckCircle className="w-4 h-4 text-green-500 mt-0.5 flex-shrink-0" />
                          <span className="text-sm text-gray-700">{condition}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                  
                  <div>
                    <h3 className="text-lg font-semibold text-gray-800 mb-3">Recommended Tests</h3>
                    <ul className="space-y-2">
                      {selectedReport.findings.tests.map((test, index) => (
                        <li key={index} className="flex items-start space-x-2">
                          <FileText className="w-4 h-4 text-blue-500 mt-0.5 flex-shrink-0" />
                          <span className="text-sm text-gray-700">{test}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>
                
                <div className="mb-6">
                  <h3 className="text-lg font-semibold text-gray-800 mb-3">Recommendations</h3>
                  <ul className="space-y-2">
                    {selectedReport.findings.recommendations.map((recommendation, index) => (
                      <li key={index} className="flex items-start space-x-2">
                        <Brain className="w-4 h-4 text-purple-500 mt-0.5 flex-shrink-0" />
                        <span className="text-sm text-gray-700">{recommendation}</span>
                      </li>
                    ))}
                  </ul>
                </div>
                
                <div className="flex justify-end space-x-3">
                  <button className="px-4 py-2 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 transition-colors flex items-center space-x-2">
                    <Download className="w-4 h-4" />
                    <span>Download Report</span>
                  </button>
                  <button className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors">
                    Share with Doctor
                  </button>
                </div>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default DiagnosisPage;