import React, { useState, useCallback } from 'react';
import { motion } from 'framer-motion';
import { useDropzone } from 'react-dropzone';
import { useAuth } from '../contexts/AuthContext';
import {
  Upload,
  FileText,
  Image as ImageIcon,
  Brain,
  CheckCircle,
  AlertTriangle,
  Download,
  Eye,
  Trash2,
  Plus
} from 'lucide-react';

interface UploadedReport {
  id: string;
  filename: string;
  type: string;
  uploadDate: string;
  extractedText: string;
  aiAnalysis: string;
  status: 'processing' | 'completed' | 'error';
}

export const UploadReportsPage: React.FC = () => {
  const { user } = useAuth();
  const [uploads, setUploads] = useState<UploadedReport[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const [selectedReport, setSelectedReport] = useState<UploadedReport | null>(null);

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    if (!user) return;

    setIsUploading(true);

    for (const file of acceptedFiles) {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('type', getReportType(file.name));
      formData.append('description', '');

      try {
        const response = await fetch(`${import.meta.env.VITE_API_URL}/api/upload-medical-report`, {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${localStorage.getItem('token')}`
          },
          body: formData
        });

        const result = await response.json();

        if (response.ok) {
          const newUpload: UploadedReport = {
            id: result.upload_id,
            filename: file.name,
            type: getReportType(file.name),
            uploadDate: new Date().toISOString(),
            extractedText: result.extracted_text || '',
            aiAnalysis: result.ai_analysis || '',
            status: 'completed'
          };

          setUploads(prev => [newUpload, ...prev]);
        } else {
          console.error('Upload failed:', result.error);
        }
      } catch (error) {
        console.error('Upload error:', error);
      }
    }

    setIsUploading(false);
  }, [user]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf'],
      'image/*': ['.png', '.jpg', '.jpeg', '.tiff'],
      'text/plain': ['.txt']
    },
    multiple: true
  });

  const getReportType = (filename: string): string => {
    const name = filename.toLowerCase();
    if (name.includes('blood') || name.includes('lab')) return 'lab_test';
    if (name.includes('xray') || name.includes('x-ray')) return 'xray';
    if (name.includes('mri')) return 'mri';
    if (name.includes('ct') || name.includes('scan')) return 'ct_scan';
    if (name.includes('echo') || name.includes('cardiac')) return 'cardiac';
    return 'general';
  };

  const getReportIcon = (type: string) => {
    switch (type) {
      case 'lab_test': return FileText;
      case 'xray': case 'mri': case 'ct_scan': return ImageIcon;
      case 'cardiac': return Heart;
      default: return FileText;
    }
  };

  const getReportColor = (type: string) => {
    switch (type) {
      case 'lab_test': return 'bg-blue-100 text-blue-600';
      case 'xray': return 'bg-purple-100 text-purple-600';
      case 'mri': return 'bg-green-100 text-green-600';
      case 'ct_scan': return 'bg-orange-100 text-orange-600';
      case 'cardiac': return 'bg-red-100 text-red-600';
      default: return 'bg-gray-100 text-gray-600';
    }
  };

  if (!user) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <AlertTriangle className="w-16 h-16 text-red-500 mx-auto mb-4" />
          <h2 className="text-2xl font-bold text-gray-900 mb-2">Authentication Required</h2>
          <p className="text-gray-600">Please sign in to upload medical reports.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen py-8 px-4 sm:px-6 lg:px-8 bg-gray-50">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-12"
        >
          <div className="bg-gradient-to-r from-purple-600 to-blue-600 w-16 h-16 rounded-2xl flex items-center justify-center mx-auto mb-4">
            <Upload className="w-8 h-8 text-white" />
          </div>
          <h1 className="text-3xl font-bold text-gray-900 mb-2">Upload Medical Reports</h1>
          <p className="text-gray-600">Upload lab results, imaging reports, and medical documents for AI analysis</p>
        </motion.div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Upload Section */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            className="lg:col-span-1"
          >
            <div className="bg-white rounded-2xl shadow-lg p-6 mb-6">
              <h2 className="text-xl font-bold text-gray-900 mb-4">Upload New Reports</h2>
              
              {/* File Upload Area */}
              <div
                {...getRootProps()}
                className={`border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-all duration-200 ${
                  isDragActive 
                    ? 'border-blue-500 bg-blue-50' 
                    : 'border-gray-300 hover:border-blue-500 hover:bg-blue-50'
                }`}
              >
                <input {...getInputProps()} />
                <Upload className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                {isDragActive ? (
                  <p className="text-blue-600 font-medium">Drop the files here...</p>
                ) : (
                  <div>
                    <p className="text-gray-600 mb-2">Drag & drop medical reports here, or click to select</p>
                    <p className="text-sm text-gray-500">Supports PDF, JPG, PNG, TIFF formats</p>
                  </div>
                )}
              </div>

              {isUploading && (
                <div className="mt-4 p-4 bg-blue-50 rounded-xl">
                  <div className="flex items-center space-x-3">
                    <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-blue-600"></div>
                    <span className="text-blue-800">Processing uploads...</span>
                  </div>
                </div>
              )}

              {/* Supported Report Types */}
              <div className="mt-6">
                <h3 className="font-semibold text-gray-900 mb-3">Supported Report Types</h3>
                <div className="space-y-2">
                  {[
                    { type: 'Lab Tests', desc: 'Blood work, urine tests, pathology' },
                    { type: 'X-Rays', desc: 'Chest, bone, dental X-rays' },
                    { type: 'MRI Scans', desc: 'Brain, spine, joint MRI reports' },
                    { type: 'CT Scans', desc: 'Body, head CT scan results' },
                    { type: 'Cardiac Tests', desc: 'ECG, echo, stress tests' },
                    { type: 'Other', desc: 'Any medical document or report' }
                  ].map((item, index) => (
                    <div key={index} className="flex items-start space-x-2">
                      <div className="w-2 h-2 bg-blue-600 rounded-full mt-2"></div>
                      <div>
                        <span className="font-medium text-gray-900">{item.type}:</span>
                        <span className="text-gray-600 text-sm ml-1">{item.desc}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </motion.div>

          {/* Reports List */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            className="lg:col-span-2"
          >
            <div className="bg-white rounded-2xl shadow-lg p-6">
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-xl font-bold text-gray-900">Your Medical Reports</h2>
                <span className="bg-blue-100 text-blue-800 px-3 py-1 rounded-full text-sm font-medium">
                  {uploads.length} reports
                </span>
              </div>

              {uploads.length === 0 ? (
                <div className="text-center py-12">
                  <FileText className="w-16 h-16 text-gray-300 mx-auto mb-4" />
                  <h3 className="text-lg font-medium text-gray-900 mb-2">No reports uploaded yet</h3>
                  <p className="text-gray-600">Upload your first medical report to get started with AI analysis</p>
                </div>
              ) : (
                <div className="space-y-4">
                  {uploads.map((report) => {
                    const Icon = getReportIcon(report.type);
                    return (
                      <div key={report.id} className="border border-gray-200 rounded-xl p-6 hover:shadow-md transition-all duration-200">
                        <div className="flex items-start justify-between mb-4">
                          <div className="flex items-start space-x-4">
                            <div className={`p-3 rounded-xl ${getReportColor(report.type)}`}>
                              <Icon className="w-6 h-6" />
                            </div>
                            <div>
                              <h3 className="font-bold text-gray-900 mb-1">{report.filename}</h3>
                              <p className="text-sm text-gray-600 mb-2">
                                Uploaded on {new Date(report.uploadDate).toLocaleDateString()}
                              </p>
                              <span className={`inline-block px-2 py-1 rounded-full text-xs font-medium ${
                                report.status === 'completed' ? 'bg-green-100 text-green-800' :
                                report.status === 'processing' ? 'bg-yellow-100 text-yellow-800' :
                                'bg-red-100 text-red-800'
                              }`}>
                                {report.status}
                              </span>
                            </div>
                          </div>
                          <div className="flex space-x-2">
                            <button
                              onClick={() => setSelectedReport(report)}
                              className="p-2 bg-blue-100 text-blue-600 rounded-lg hover:bg-blue-200 transition-all duration-200"
                            >
                              <Eye className="w-4 h-4" />
                            </button>
                            <button className="p-2 bg-gray-100 text-gray-600 rounded-lg hover:bg-gray-200 transition-all duration-200">
                              <Download className="w-4 h-4" />
                            </button>
                            <button className="p-2 bg-red-100 text-red-600 rounded-lg hover:bg-red-200 transition-all duration-200">
                              <Trash2 className="w-4 h-4" />
                            </button>
                          </div>
                        </div>

                        {report.status === 'completed' && report.aiAnalysis && (
                          <div className="bg-blue-50 rounded-xl p-4">
                            <div className="flex items-center space-x-2 mb-2">
                              <Brain className="w-5 h-5 text-blue-600" />
                              <span className="font-medium text-blue-900">AI Analysis Summary</span>
                            </div>
                            <p className="text-blue-800 text-sm">
                              {report.aiAnalysis.substring(0, 200)}...
                            </p>
                            <button
                              onClick={() => setSelectedReport(report)}
                              className="text-blue-600 text-sm font-medium mt-2 hover:text-blue-700"
                            >
                              View Full Analysis →
                            </button>
                          </div>
                        )}

                        {report.extractedText && (
                          <div className="mt-4 p-4 bg-gray-50 rounded-xl">
                            <h4 className="font-medium text-gray-900 mb-2">Extracted Text Preview</h4>
                            <p className="text-gray-700 text-sm">
                              {report.extractedText.substring(0, 150)}...
                            </p>
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          </motion.div>
        </div>

        {/* Report Detail Modal */}
        {selectedReport && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              className="bg-white rounded-2xl shadow-xl max-w-4xl w-full max-h-[90vh] overflow-y-auto"
            >
              <div className="p-6 border-b border-gray-200">
                <div className="flex items-center justify-between">
                  <h2 className="text-2xl font-bold text-gray-900">{selectedReport.filename}</h2>
                  <button
                    onClick={() => setSelectedReport(null)}
                    className="p-2 bg-gray-100 rounded-lg hover:bg-gray-200 transition-all duration-200"
                  >
                    ✕
                  </button>
                </div>
              </div>

              <div className="p-6 space-y-6">
                {/* AI Analysis */}
                {selectedReport.aiAnalysis && (
                  <div>
                    <div className="flex items-center space-x-2 mb-4">
                      <Brain className="w-6 h-6 text-blue-600" />
                      <h3 className="text-xl font-bold text-gray-900">AI Medical Analysis</h3>
                    </div>
                    <div className="bg-blue-50 rounded-xl p-6">
                      <pre className="whitespace-pre-wrap text-sm text-gray-700 font-sans">
                        {selectedReport.aiAnalysis}
                      </pre>
                    </div>
                  </div>
                )}

                {/* Extracted Text */}
                {selectedReport.extractedText && (
                  <div>
                    <h3 className="text-xl font-bold text-gray-900 mb-4">Extracted Text</h3>
                    <div className="bg-gray-50 rounded-xl p-6">
                      <pre className="whitespace-pre-wrap text-sm text-gray-700 font-mono">
                        {selectedReport.extractedText}
                      </pre>
                    </div>
                  </div>
                )}

                {/* Actions */}
                <div className="flex space-x-4">
                  <button className="flex-1 bg-blue-600 text-white py-3 px-4 rounded-xl font-semibold hover:bg-blue-700 transition-all duration-200 flex items-center justify-center space-x-2">
                    <Download className="w-5 h-5" />
                    <span>Download Report</span>
                  </button>
                  <button className="flex-1 bg-green-600 text-white py-3 px-4 rounded-xl font-semibold hover:bg-green-700 transition-all duration-200 flex items-center justify-center space-x-2">
                    <Plus className="w-5 h-5" />
                    <span>Add to Medical History</span>
                  </button>
                </div>
              </div>
            </motion.div>
          </div>
        )}

        {/* Disclaimer */}
        <div className="mt-8 p-6 bg-yellow-50 border border-yellow-200 rounded-xl">
          <div className="flex items-start space-x-3">
            <AlertTriangle className="w-6 h-6 text-yellow-600 mt-1" />
            <div>
              <h4 className="font-bold text-yellow-900 mb-2">Important Medical Disclaimer</h4>
              <p className="text-yellow-800 text-sm">
                AI analysis of medical reports is for informational purposes only and should not replace 
                professional medical interpretation. Always consult with qualified healthcare professionals 
                for accurate diagnosis and treatment decisions. Ensure all uploaded documents comply with 
                privacy regulations and do not contain sensitive personal information of others.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};