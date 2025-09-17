import React, { useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useDropzone } from 'react-dropzone';
import { useAuth } from '../contexts/AuthContext';
import { apiClient } from '../lib/api';
import { 
  Upload, 
  Image as ImageIcon, 
  Brain, 
  Scan, 
  CheckCircle, 
  AlertTriangle,
  Eye,
  Download,
  FileText,
  Activity,
  Heart,
  Zap,
  Camera,
  Monitor,
  Microscope,
  Shield,
  Star,
  Clock,
  User
} from 'lucide-react';

interface AnalysisResult {
  success: boolean;
  report_id: string;
  analysis: any;
  error?: string;
}

export const ImagingPage: React.FC = () => {
  const { user } = useAuth();
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [imageType, setImageType] = useState('auto');
  const [clinicalContext, setClinicalContext] = useState('');
  const [analysis, setAnalysis] = useState<any>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [uploadedReports, setUploadedReports] = useState<any[]>([]);
  const [activeTab, setActiveTab] = useState<'imaging' | 'reports'>('imaging');

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (file) {
      setSelectedImage(file);
      const reader = new FileReader();
      reader.onload = (e) => {
        setImagePreview(e.target?.result as string);
      };
      reader.readAsDataURL(file);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png', '.tiff', '.bmp'],
      'application/dicom': ['.dcm', '.dicom']
    },
    multiple: false,
    maxSize: 50 * 1024 * 1024 // 50MB
  });

  const handleAnalyzeImage = async () => {
    if (!selectedImage) return;

    setIsAnalyzing(true);
    
    try {
      const formData = new FormData();
      formData.append('file', selectedImage);
      formData.append('image_type', imageType);
      formData.append('clinical_context', clinicalContext);

      const result = await apiClient.uploadMedicalImage(formData);
      setAnalysis(result.analysis);
      
      // Add to uploaded reports
      setUploadedReports(prev => [{
        id: result.report_id,
        filename: selectedImage.name,
        type: imageType,
        analysis: result.analysis,
        uploadDate: new Date().toISOString()
      }, ...prev]);
      
    } catch (error) {
      console.error('Image analysis failed:', error);
      alert('Failed to analyze image. Please try again.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleAnalyzeTextReport = async (file: File) => {
    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('report_type', 'lab_report');

      const result = await apiClient.uploadMedicalReport(formData);
      
      // Add to uploaded reports
      setUploadedReports(prev => [{
        id: result.report_id,
        filename: file.name,
        type: 'text_report',
        analysis: result.analysis,
        extractedText: result.extracted_text,
        uploadDate: new Date().toISOString()
      }, ...prev]);
      
    } catch (error) {
      console.error('Report analysis failed:', error);
      alert('Failed to analyze report. Please try again.');
    }
  };

  const imageTypes = [
    { value: 'auto', label: 'Auto-detect', icon: Brain, color: 'from-blue-500 to-purple-500' },
    { value: 'chest_xray', label: 'Chest X-Ray', icon: Activity, color: 'from-red-500 to-pink-500' },
    { value: 'brain_mri', label: 'Brain MRI', icon: Brain, color: 'from-purple-500 to-indigo-500' },
    { value: 'ct_scan', label: 'CT Scan', icon: Scan, color: 'from-green-500 to-teal-500' },
    { value: 'ultrasound', label: 'Ultrasound', icon: Heart, color: 'from-orange-500 to-red-500' },
    { value: 'mammography', label: 'Mammography', icon: Heart, color: 'from-pink-500 to-rose-500' },
    { value: 'bone_xray', label: 'Bone X-Ray', icon: Activity, color: 'from-gray-500 to-gray-600' }
  ];

  if (!user) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-blue-50 via-white to-purple-50">
        <div className="text-center">
          <AlertTriangle className="w-16 h-16 text-red-500 mx-auto mb-4" />
          <h2 className="text-2xl font-bold text-gray-900 mb-2">Authentication Required</h2>
          <p className="text-gray-600">Please sign in to access medical imaging analysis.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen py-8 px-4 sm:px-6 lg:px-8 bg-gradient-to-br from-blue-50 via-white to-purple-50">
      <div className="max-w-7xl mx-auto">
        {/* Enhanced Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-12"
        >
          <div className="relative mb-8">
            <div className="absolute inset-0 bg-gradient-to-r from-purple-600 to-pink-600 w-32 h-32 rounded-full opacity-20 blur-xl mx-auto"></div>
            <div className="relative bg-gradient-to-r from-purple-600 to-pink-600 w-24 h-24 rounded-3xl flex items-center justify-center mx-auto shadow-2xl">
              <Brain className="w-12 h-12 text-white" />
            </div>
          </div>
          <h1 className="text-5xl font-bold bg-gradient-to-r from-purple-600 via-pink-600 to-red-600 bg-clip-text text-transparent mb-4">
            Medical Imaging & Reports
          </h1>
          <p className="text-xl text-gray-600">Advanced AI analysis of medical images and reports using Gemini Vision</p>
        </motion.div>

        {/* Enhanced Tabs */}
        <div className="mb-8">
          <div className="bg-white/80 backdrop-blur-sm rounded-2xl shadow-xl p-2 border border-white/20">
            <div className="flex space-x-1">
              <button
                onClick={() => setActiveTab('imaging')}
                className={`flex items-center space-x-2 px-6 py-3 rounded-xl font-semibold transition-all duration-300 ${
                  activeTab === 'imaging'
                    ? 'bg-gradient-to-r from-purple-600 to-pink-600 text-white shadow-lg'
                    : 'text-gray-600 hover:bg-gray-100'
                }`}
              >
                <Camera className="w-5 h-5" />
                <span>Medical Imaging</span>
              </button>
              <button
                onClick={() => setActiveTab('reports')}
                className={`flex items-center space-x-2 px-6 py-3 rounded-xl font-semibold transition-all duration-300 ${
                  activeTab === 'reports'
                    ? 'bg-gradient-to-r from-blue-600 to-purple-600 text-white shadow-lg'
                    : 'text-gray-600 hover:bg-gray-100'
                }`}
              >
                <FileText className="w-5 h-5" />
                <span>Text Reports ({uploadedReports.length})</span>
              </button>
            </div>
          </div>
        </div>

        <AnimatePresence mode="wait">
          {/* Medical Imaging Tab */}
          {activeTab === 'imaging' && (
            <motion.div
              key="imaging"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              className="grid grid-cols-1 lg:grid-cols-2 gap-8"
            >
              {/* Upload Section */}
              <div className="space-y-6">
                {/* Image Types Grid */}
                <div className="bg-white/90 backdrop-blur-sm rounded-3xl shadow-2xl p-6 border border-white/20">
                  <h2 className="text-2xl font-bold text-gray-900 mb-6">Supported Imaging Types</h2>
                  <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                    {imageTypes.map((type) => {
                      const Icon = type.icon;
                      return (
                        <motion.div
                          key={type.value}
                          whileHover={{ scale: 1.05 }}
                          className={`bg-gradient-to-r ${type.color} p-4 rounded-2xl text-white text-center shadow-lg`}
                        >
                          <Icon className="w-8 h-8 mx-auto mb-2" />
                          <h3 className="font-bold text-sm">{type.label}</h3>
                        </motion.div>
                      );
                    })}
                  </div>
                </div>

                {/* Upload Area */}
                <div className="bg-white/90 backdrop-blur-sm rounded-3xl shadow-2xl p-8 border border-white/20">
                  <h2 className="text-2xl font-bold text-gray-900 mb-6">Upload Medical Image</h2>
                  
                  <div
                    {...getRootProps()}
                    className={`border-3 border-dashed rounded-2xl p-12 text-center cursor-pointer transition-all duration-300 ${
                      isDragActive 
                        ? 'border-purple-500 bg-purple-50 scale-105' 
                        : 'border-gray-300 hover:border-purple-500 hover:bg-purple-50'
                    }`}
                  >
                    <input {...getInputProps()} />
                    <div className="relative">
                      <div className="absolute inset-0 bg-gradient-to-r from-purple-600 to-pink-600 w-20 h-20 rounded-full opacity-20 blur-xl mx-auto"></div>
                      <Upload className="relative w-16 h-16 text-gray-400 mx-auto mb-4" />
                    </div>
                    {isDragActive ? (
                      <p className="text-purple-600 font-bold text-xl">Drop the medical image here...</p>
                    ) : (
                      <div>
                        <p className="text-gray-600 mb-2 text-lg font-medium">Drag & drop medical image here, or click to select</p>
                        <p className="text-sm text-gray-500">Supports JPEG, PNG, TIFF, DICOM formats (max 50MB)</p>
                      </div>
                    )}
                  </div>

                  {/* Image Preview */}
                  {imagePreview && (
                    <motion.div
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      className="mt-8"
                    >
                      <h3 className="font-bold text-gray-900 mb-4">Image Preview</h3>
                      <div className="relative bg-gray-100 rounded-2xl overflow-hidden shadow-lg">
                        <img
                          src={imagePreview}
                          alt="Medical scan preview"
                          className="w-full h-80 object-contain"
                        />
                        <div className="absolute top-4 right-4 bg-white/90 backdrop-blur-sm rounded-xl p-3 shadow-lg">
                          <p className="text-sm font-bold text-gray-900">{selectedImage?.name}</p>
                          <p className="text-xs text-gray-600">
                            {((selectedImage?.size || 0) / (1024 * 1024)).toFixed(2)} MB
                          </p>
                        </div>
                      </div>
                    </motion.div>
                  )}

                  {/* Configuration */}
                  {selectedImage && (
                    <motion.div
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      className="mt-8 space-y-6"
                    >
                      <div>
                        <label className="block text-sm font-bold text-gray-700 mb-4">
                          Select Image Type
                        </label>
                        <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                          {imageTypes.map((type) => {
                            const Icon = type.icon;
                            return (
                              <label key={type.value} className="relative">
                                <input
                                  type="radio"
                                  value={type.value}
                                  checked={imageType === type.value}
                                  onChange={(e) => setImageType(e.target.value)}
                                  className="sr-only"
                                />
                                <div className={`p-4 border-2 rounded-xl cursor-pointer transition-all duration-200 ${
                                  imageType === type.value
                                    ? 'border-purple-500 bg-purple-50 shadow-lg'
                                    : 'border-gray-200 hover:border-purple-300 hover:bg-purple-50'
                                }`}>
                                  <Icon className="w-6 h-6 mx-auto mb-2 text-purple-600" />
                                  <span className="text-sm font-medium text-center block">{type.label}</span>
                                </div>
                              </label>
                            );
                          })}
                        </div>
                      </div>

                      <div>
                        <label className="block text-sm font-bold text-gray-700 mb-2">
                          Clinical Context (Optional)
                        </label>
                        <textarea
                          value={clinicalContext}
                          onChange={(e) => setClinicalContext(e.target.value)}
                          rows={3}
                          className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-purple-500 focus:border-purple-500 bg-white/50"
                          placeholder="Provide clinical context: patient symptoms, reason for imaging, relevant history..."
                        />
                      </div>

                      <button
                        onClick={handleAnalyzeImage}
                        disabled={isAnalyzing}
                        className="w-full bg-gradient-to-r from-purple-600 via-pink-600 to-red-600 text-white py-4 px-6 rounded-2xl font-bold text-lg disabled:opacity-50 disabled:cursor-not-allowed hover:shadow-2xl transition-all duration-300 flex items-center justify-center space-x-3 transform hover:scale-105"
                      >
                        {isAnalyzing ? (
                          <>
                            <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-white"></div>
                            <span>AI Analyzing Image...</span>
                            <Brain className="w-6 h-6" />
                          </>
                        ) : (
                          <>
                            <Zap className="w-6 h-6" />
                            <span>Analyze with Gemini Vision AI</span>
                            <Brain className="w-6 h-6" />
                          </>
                        )}
                      </button>
                    </motion.div>
                  )}
                </div>
              </div>

              {/* Analysis Results */}
              <div>
                {analysis ? (
                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="space-y-6"
                  >
                    {/* Analysis Header */}
                    <div className="bg-white/90 backdrop-blur-sm rounded-3xl shadow-2xl p-8 border border-white/20">
                      <div className="flex items-center space-x-4 mb-6">
                        <div className="bg-gradient-to-r from-green-500 to-teal-500 p-4 rounded-2xl">
                          <CheckCircle className="w-8 h-8 text-white" />
                        </div>
                        <div>
                          <h2 className="text-2xl font-bold text-gray-900">Analysis Complete</h2>
                          <p className="text-gray-600">Gemini Vision AI Analysis Results</p>
                        </div>
                      </div>
                      
                      <div className="bg-gradient-to-r from-blue-50 to-purple-50 rounded-2xl p-6">
                        <p className="text-gray-800 leading-relaxed">
                          {analysis.image_analysis?.overall_impression || 'Medical image analysis completed successfully'}
                        </p>
                      </div>
                    </div>

                    {/* Detailed Findings */}
                    <div className="bg-white/90 backdrop-blur-sm rounded-3xl shadow-2xl p-8 border border-white/20">
                      <h3 className="text-2xl font-bold text-gray-900 mb-6">Detailed Findings</h3>
                      <div className="space-y-4">
                        {analysis.image_analysis?.findings?.map((finding: any, index: number) => (
                          <motion.div
                            key={index}
                            initial={{ opacity: 0, x: -20 }}
                            animate={{ opacity: 1, x: 0 }}
                            transition={{ delay: index * 0.1 }}
                            className="border border-gray-200 rounded-2xl p-6 bg-gradient-to-r from-white to-gray-50"
                          >
                            <div className="flex justify-between items-start mb-3">
                              <h4 className="font-bold text-gray-900 text-lg">{finding.finding}</h4>
                              <span className={`px-3 py-1 rounded-full text-sm font-bold ${
                                finding.severity === 'Normal' ? 'bg-green-100 text-green-800' :
                                finding.severity === 'Abnormal' ? 'bg-red-100 text-red-800' :
                                'bg-yellow-100 text-yellow-800'
                              }`}>
                                {finding.severity}
                              </span>
                            </div>
                            <p className="text-sm text-gray-600 mb-3">
                              <strong>Location:</strong> {finding.location}
                            </p>
                            <div className="flex items-center space-x-3">
                              <span className="text-sm text-gray-600 font-medium">Confidence:</span>
                              <div className="bg-gray-200 rounded-full h-3 w-32">
                                <div 
                                  className="bg-gradient-to-r from-blue-500 to-purple-500 h-3 rounded-full transition-all duration-1000"
                                  style={{ width: `${(finding.confidence || 0.8) * 100}%` }}
                                />
                              </div>
                              <span className="text-sm font-bold text-purple-600">
                                {Math.round((finding.confidence || 0.8) * 100)}%
                              </span>
                            </div>
                          </motion.div>
                        )) || (
                          <div className="text-center py-8">
                            <Brain className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                            <p className="text-gray-600">No specific findings detected</p>
                          </div>
                        )}
                      </div>
                    </div>

                    {/* Recommendations */}
                    <div className="bg-white/90 backdrop-blur-sm rounded-3xl shadow-2xl p-8 border border-white/20">
                      <h3 className="text-2xl font-bold text-gray-900 mb-6">AI Recommendations</h3>
                      <div className="space-y-3">
                        {analysis.image_analysis?.recommendations?.map((rec: string, index: number) => (
                          <motion.div
                            key={index}
                            initial={{ opacity: 0, x: -20 }}
                            animate={{ opacity: 1, x: 0 }}
                            transition={{ delay: index * 0.1 }}
                            className="flex items-start space-x-3 p-4 bg-gradient-to-r from-blue-50 to-purple-50 rounded-xl"
                          >
                            <div className="bg-gradient-to-r from-blue-500 to-purple-500 rounded-full p-1 mt-1">
                              <div className="w-2 h-2 bg-white rounded-full"></div>
                            </div>
                            <span className="text-gray-700 font-medium">{rec}</span>
                          </motion.div>
                        )) || (
                          <p className="text-gray-600">No specific recommendations at this time</p>
                        )}
                      </div>
                    </div>

                    {/* Action Buttons */}
                    <div className="flex flex-wrap gap-4">
                      <button className="flex items-center space-x-2 px-6 py-3 bg-gradient-to-r from-blue-500 to-blue-600 text-white rounded-xl hover:shadow-lg transition-all duration-200">
                        <Download className="w-5 h-5" />
                        <span>Download Report</span>
                      </button>
                      <button className="flex items-center space-x-2 px-6 py-3 bg-gradient-to-r from-green-500 to-green-600 text-white rounded-xl hover:shadow-lg transition-all duration-200">
                        <Share className="w-5 h-5" />
                        <span>Share with Doctor</span>
                      </button>
                      <button className="flex items-center space-x-2 px-6 py-3 bg-gradient-to-r from-purple-500 to-purple-600 text-white rounded-xl hover:shadow-lg transition-all duration-200">
                        <Bookmark className="w-5 h-5" />
                        <span>Save to Profile</span>
                      </button>
                    </div>
                  </motion.div>
                ) : (
                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="bg-white/90 backdrop-blur-sm rounded-3xl shadow-2xl p-12 text-center border border-white/20"
                  >
                    <div className="relative mb-8">
                      <div className="absolute inset-0 bg-gradient-to-r from-purple-600 to-pink-600 w-20 h-20 rounded-full opacity-20 blur-xl mx-auto"></div>
                      <div className="relative bg-gradient-to-r from-purple-100 to-pink-100 w-16 h-16 rounded-2xl flex items-center justify-center mx-auto">
                        <Brain className="w-8 h-8 text-purple-600" />
                      </div>
                    </div>
                    <h3 className="text-2xl font-bold text-gray-900 mb-4">Ready for AI Analysis</h3>
                    <p className="text-gray-600 text-lg leading-relaxed">
                      Upload a medical image to get started with advanced Gemini Vision AI analysis. 
                      Our system can detect various conditions and provide detailed insights.
                    </p>
                  </motion.div>
                )}
              </div>
            </motion.div>
          )}

          {/* Text Reports Tab */}
          {activeTab === 'reports' && (
            <motion.div
              key="reports"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              className="space-y-6"
            >
              {/* Text Report Upload */}
              <div className="bg-white/90 backdrop-blur-sm rounded-3xl shadow-2xl p-8 border border-white/20">
                <h2 className="text-2xl font-bold text-gray-900 mb-6">Upload Medical Reports</h2>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div
                    {...getRootProps()}
                    className="border-3 border-dashed border-blue-300 rounded-2xl p-8 text-center cursor-pointer hover:border-blue-500 hover:bg-blue-50 transition-all duration-300"
                  >
                    <input {...getInputProps()} />
                    <FileText className="w-12 h-12 text-blue-400 mx-auto mb-4" />
                    <p className="text-blue-600 font-medium">Upload Lab Reports, Discharge Summaries</p>
                    <p className="text-sm text-gray-500 mt-2">PDF, TXT, DOC formats</p>
                  </div>
                  
                  <div className="space-y-4">
                    <h3 className="font-bold text-gray-900">Supported Report Types</h3>
                    <div className="space-y-2">
                      {[
                        { type: 'Lab Tests', desc: 'Blood work, urine analysis, pathology reports' },
                        { type: 'Discharge Summary', desc: 'Hospital discharge documents' },
                        { type: 'Prescription', desc: 'Medication lists and prescriptions' },
                        { type: 'Clinical Notes', desc: 'Doctor visit notes and observations' }
                      ].map((item, index) => (
                        <div key={index} className="flex items-start space-x-3 p-3 bg-blue-50 rounded-xl">
                          <div className="w-2 h-2 bg-blue-600 rounded-full mt-2"></div>
                          <div>
                            <span className="font-medium text-blue-900">{item.type}:</span>
                            <span className="text-blue-700 text-sm ml-2">{item.desc}</span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>

              {/* Uploaded Reports List */}
              <div className="bg-white/90 backdrop-blur-sm rounded-3xl shadow-2xl p-8 border border-white/20">
                <h3 className="text-2xl font-bold text-gray-900 mb-6">Your Medical Reports</h3>
                {uploadedReports.length === 0 ? (
                  <div className="text-center py-12">
                    <FileText className="w-16 h-16 text-gray-300 mx-auto mb-4" />
                    <h4 className="text-xl font-medium text-gray-900 mb-2">No reports uploaded yet</h4>
                    <p className="text-gray-600">Upload your first medical report to get AI analysis</p>
                  </div>
                ) : (
                  <div className="space-y-4">
                    {uploadedReports.map((report) => (
                      <motion.div
                        key={report.id}
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        className="border border-gray-200 rounded-2xl p-6 hover:shadow-lg transition-all duration-300 bg-gradient-to-r from-white to-gray-50"
                      >
                        <div className="flex items-start justify-between mb-4">
                          <div className="flex items-start space-x-4">
                            <div className="bg-gradient-to-r from-blue-500 to-purple-500 p-3 rounded-xl">
                              <FileText className="w-6 h-6 text-white" />
                            </div>
                            <div>
                              <h4 className="font-bold text-gray-900 mb-1">{report.filename}</h4>
                              <p className="text-sm text-gray-600 mb-2">
                                Uploaded on {new Date(report.uploadDate).toLocaleDateString()}
                              </p>
                              <span className="inline-block px-3 py-1 bg-green-100 text-green-800 rounded-full text-xs font-medium">
                                Analyzed
                              </span>
                            </div>
                          </div>
                          <div className="flex space-x-2">
                            <button className="p-2 bg-blue-100 text-blue-600 rounded-xl hover:bg-blue-200 transition-all duration-200">
                              <Eye className="w-4 h-4" />
                            </button>
                            <button className="p-2 bg-gray-100 text-gray-600 rounded-xl hover:bg-gray-200 transition-all duration-200">
                              <Download className="w-4 h-4" />
                            </button>
                          </div>
                        </div>

                        {report.analysis && (
                          <div className="bg-blue-50 rounded-xl p-4">
                            <div className="flex items-center space-x-2 mb-2">
                              <Brain className="w-5 h-5 text-blue-600" />
                              <span className="font-medium text-blue-900">AI Analysis Summary</span>
                            </div>
                            <p className="text-blue-800 text-sm">
                              {JSON.stringify(report.analysis).substring(0, 200)}...
                            </p>
                          </div>
                        )}
                      </motion.div>
                    ))}
                  </div>
                )}
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Enhanced Disclaimer */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
          className="mt-12 p-6 bg-gradient-to-r from-yellow-50 to-orange-50 border border-yellow-200 rounded-2xl"
        >
          <div className="flex items-start space-x-4">
            <div className="bg-yellow-100 p-3 rounded-xl">
              <AlertTriangle className="w-6 h-6 text-yellow-600" />
            </div>
            <div>
              <h4 className="font-bold text-yellow-900 mb-2">Important Medical Disclaimer</h4>
              <p className="text-yellow-800 text-sm leading-relaxed">
                MediScan AI provides preliminary analysis using advanced Gemini AI technology. This is not a substitute for professional medical diagnosis. 
                All findings should be reviewed by qualified healthcare professionals. In case of emergency, contact emergency services immediately.
              </p>
            </div>
          </div>
        </motion.div>
      </div>
    </>
  );
};