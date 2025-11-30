import React, { useState, useRef } from 'react';
import { motion } from 'framer-motion';
import { Upload, FileText, Image, Brain, Activity, Download, Eye, AlertCircle, CheckCircle, Clock, Zap } from 'lucide-react';

interface AnalysisResult {
  id: string;
  type: 'image' | 'text';
  filename: string;
  uploadTime: string;
  status: 'processing' | 'completed' | 'error';
  confidence: number;
  findings: {
    summary: string;
    keyFindings: string[];
    recommendations: string[];
    urgency: 'low' | 'medium' | 'high' | 'critical';
  };
}

const ImagingPage: React.FC = () => {
  const [uploadedFiles, setUploadedFiles] = useState<File[]>([]);
  const [analysisResults, setAnalysisResults] = useState<AnalysisResult[]>([]);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [selectedResult, setSelectedResult] = useState<AnalysisResult | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(event.target.files || []);
    setUploadedFiles(prev => [...prev, ...files]);
    
    // Simulate analysis for each file
    files.forEach(file => {
      const newResult: AnalysisResult = {
        id: Math.random().toString(36).substr(2, 9),
        type: file.type.startsWith('image/') ? 'image' : 'text',
        filename: file.name,
        uploadTime: new Date().toLocaleString(),
        status: 'processing',
        confidence: 0,
        findings: {
          summary: '',
          keyFindings: [],
          recommendations: [],
          urgency: 'low'
        }
      };
      
      setAnalysisResults(prev => [...prev, newResult]);
      
      // Simulate ML analysis completion
      setTimeout(() => {
        setAnalysisResults(prev => prev.map(result => 
          result.id === newResult.id 
            ? {
                ...result,
                status: 'completed',
                confidence: Math.random() * 30 + 70, // 70-100%
                findings: {
                  summary: file.type.startsWith('image/') 
                    ? 'Medical imaging analysis completed. No acute abnormalities detected.'
                    : 'Lab report analysis completed. Values within normal ranges.',
                  keyFindings: [
                    'Normal anatomical structures',
                    'No signs of acute pathology',
                    'Good image quality for assessment'
                  ],
                  recommendations: [
                    'Continue routine monitoring',
                    'Follow up as clinically indicated',
                    'Correlate with clinical symptoms'
                  ],
                  urgency: Math.random() > 0.8 ? 'medium' : 'low'
                }
              }
            : result
        ));
      }, 3000 + Math.random() * 2000);
    });
  };

  const getUrgencyColor = (urgency: string) => {
    switch (urgency) {
      case 'critical': return 'text-red-600 bg-red-50';
      case 'high': return 'text-orange-600 bg-orange-50';
      case 'medium': return 'text-yellow-600 bg-yellow-50';
      default: return 'text-green-600 bg-green-50';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'processing': return <Clock className="w-4 h-4 text-blue-500 animate-spin" />;
      case 'completed': return <CheckCircle className="w-4 h-4 text-green-500" />;
      case 'error': return <AlertCircle className="w-4 h-4 text-red-500" />;
      default: return <Clock className="w-4 h-4 text-gray-500" />;
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50">
      {/* Header */}
      <div className="bg-white/80 backdrop-blur-sm border-b border-gray-200/50 sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                Medical Imaging & Reports
              </h1>
              <p className="text-gray-600 mt-2">Advanced AI analysis for medical documents and imaging</p>
            </div>
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2 text-sm text-gray-500">
                <Brain className="w-4 h-4" />
                <span>AI-Powered Analysis</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Upload Section */}
          <div className="lg:col-span-1">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="bg-white/80 backdrop-blur-sm rounded-2xl shadow-xl border border-gray-200/50 p-6"
            >
              <h2 className="text-xl font-semibold text-gray-800 mb-4 flex items-center">
                <Upload className="w-5 h-5 mr-2 text-blue-600" />
                Upload Medical Files
              </h2>
              
              <div
                onClick={() => fileInputRef.current?.click()}
                className="border-2 border-dashed border-gray-300 rounded-xl p-8 text-center cursor-pointer hover:border-blue-400 hover:bg-blue-50/50 transition-all duration-300"
              >
                <div className="flex flex-col items-center space-y-4">
                  <div className="p-4 bg-blue-100 rounded-full">
                    <Upload className="w-8 h-8 text-blue-600" />
                  </div>
                  <div>
                    <p className="text-lg font-medium text-gray-700">Drop files here or click to upload</p>
                    <p className="text-sm text-gray-500 mt-1">
                      Supports: DICOM, JPG, PNG, PDF, TXT
                    </p>
                  </div>
                </div>
              </div>
              
              <input
                ref={fileInputRef}
                type="file"
                multiple
                accept=".dcm,.jpg,.jpeg,.png,.pdf,.txt"
                onChange={handleFileUpload}
                className="hidden"
              />

              {uploadedFiles.length > 0 && (
                <div className="mt-6">
                  <h3 className="text-sm font-medium text-gray-700 mb-3">Uploaded Files</h3>
                  <div className="space-y-2">
                    {uploadedFiles.map((file, index) => (
                      <div key={index} className="flex items-center space-x-3 p-3 bg-gray-50 rounded-lg">
                        {file.type.startsWith('image/') ? (
                          <Image className="w-4 h-4 text-blue-500" />
                        ) : (
                          <FileText className="w-4 h-4 text-green-500" />
                        )}
                        <span className="text-sm text-gray-700 truncate flex-1">{file.name}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </motion.div>
          </div>

          {/* Analysis Results */}
          <div className="lg:col-span-2">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 }}
              className="bg-white/80 backdrop-blur-sm rounded-2xl shadow-xl border border-gray-200/50 p-6"
            >
              <h2 className="text-xl font-semibold text-gray-800 mb-6 flex items-center">
                <Activity className="w-5 h-5 mr-2 text-purple-600" />
                Analysis Results
              </h2>

              {analysisResults.length === 0 ? (
                <div className="text-center py-12">
                  <Brain className="w-16 h-16 text-gray-300 mx-auto mb-4" />
                  <p className="text-gray-500 text-lg">No files uploaded yet</p>
                  <p className="text-gray-400 text-sm">Upload medical files to see AI analysis results</p>
                </div>
              ) : (
                <div className="space-y-4">
                  {analysisResults.map((result) => (
                    <motion.div
                      key={result.id}
                      initial={{ opacity: 0, scale: 0.95 }}
                      animate={{ opacity: 1, scale: 1 }}
                      className="border border-gray-200 rounded-xl p-4 hover:shadow-lg transition-all duration-300 cursor-pointer"
                      onClick={() => setSelectedResult(result)}
                    >
                      <div className="flex items-center justify-between mb-3">
                        <div className="flex items-center space-x-3">
                          {getStatusIcon(result.status)}
                          <div>
                            <h3 className="font-medium text-gray-800">{result.filename}</h3>
                            <p className="text-sm text-gray-500">{result.uploadTime}</p>
                          </div>
                        </div>
                        <div className="flex items-center space-x-2">
                          {result.status === 'completed' && (
                            <>
                              <span className={`px-2 py-1 rounded-full text-xs font-medium ${getUrgencyColor(result.findings.urgency)}`}>
                                {result.findings.urgency.toUpperCase()}
                              </span>
                              <span className="text-sm text-gray-600">
                                {result.confidence.toFixed(1)}% confidence
                              </span>
                            </>
                          )}
                          <Eye className="w-4 h-4 text-gray-400" />
                        </div>
                      </div>
                      
                      {result.status === 'completed' && (
                        <div className="bg-gray-50 rounded-lg p-3">
                          <p className="text-sm text-gray-700">{result.findings.summary}</p>
                        </div>
                      )}
                      
                      {result.status === 'processing' && (
                        <div className="bg-blue-50 rounded-lg p-3">
                          <div className="flex items-center space-x-2">
                            <Zap className="w-4 h-4 text-blue-500" />
                            <p className="text-sm text-blue-700">AI analysis in progress...</p>
                          </div>
                        </div>
                      )}
                    </motion.div>
                  ))}
                </div>
              )}
            </motion.div>
          </div>
        </div>
      </div>

      {/* Detailed Results Modal */}
      {selectedResult && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4"
          onClick={() => setSelectedResult(null)}
        >
          <motion.div
            initial={{ scale: 0.95, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            className="bg-white rounded-2xl shadow-2xl max-w-4xl w-full max-h-[90vh] overflow-y-auto"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="p-6 border-b border-gray-200">
              <div className="flex items-center justify-between">
                <h2 className="text-2xl font-bold text-gray-800">Analysis Results</h2>
                <button
                  onClick={() => setSelectedResult(null)}
                  className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
                >
                  ✕
                </button>
              </div>
            </div>
            
            <div className="p-6">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <h3 className="text-lg font-semibold text-gray-800 mb-3">File Information</h3>
                  <div className="space-y-2 text-sm">
                    <p><span className="font-medium">Filename:</span> {selectedResult.filename}</p>
                    <p><span className="font-medium">Type:</span> {selectedResult.type}</p>
                    <p><span className="font-medium">Upload Time:</span> {selectedResult.uploadTime}</p>
                    <p><span className="font-medium">Confidence:</span> {selectedResult.confidence.toFixed(1)}%</p>
                  </div>
                </div>
                
                <div>
                  <h3 className="text-lg font-semibold text-gray-800 mb-3">Summary</h3>
                  <p className="text-sm text-gray-700">{selectedResult.findings.summary}</p>
                </div>
              </div>
              
              <div className="mt-6">
                <h3 className="text-lg font-semibold text-gray-800 mb-3">Key Findings</h3>
                <ul className="space-y-2">
                  {selectedResult.findings.keyFindings.map((finding, index) => (
                    <li key={index} className="flex items-start space-x-2">
                      <CheckCircle className="w-4 h-4 text-green-500 mt-0.5 flex-shrink-0" />
                      <span className="text-sm text-gray-700">{finding}</span>
                    </li>
                  ))}
                </ul>
              </div>
              
              <div className="mt-6">
                <h3 className="text-lg font-semibold text-gray-800 mb-3">Recommendations</h3>
                <ul className="space-y-2">
                  {selectedResult.findings.recommendations.map((recommendation, index) => (
                    <li key={index} className="flex items-start space-x-2">
                      <Zap className="w-4 h-4 text-blue-500 mt-0.5 flex-shrink-0" />
                      <span className="text-sm text-gray-700">{recommendation}</span>
                    </li>
                  ))}
                </ul>
              </div>
              
              <div className="mt-6 flex justify-end space-x-3">
                <button className="px-4 py-2 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 transition-colors flex items-center space-x-2">
                  <Download className="w-4 h-4" />
                  <span>Download Report</span>
                </button>
                <button className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors">
                  Save to Dashboard
                </button>
              </div>
            </div>
          </motion.div>
        </motion.div>
      )}
    </div>
  );
};

export default ImagingPage;