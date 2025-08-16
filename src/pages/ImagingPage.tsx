import React, { useState, useCallback } from 'react';
import { motion } from 'framer-motion';
import { useDropzone } from 'react-dropzone';
import { 
  Upload, 
  Image as ImageIcon, 
  Brain, 
  Scan, 
  CheckCircle, 
  AlertTriangle,
  Eye,
  Download
} from 'lucide-react';

export const ImagingPage: React.FC = () => {
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [analysis, setAnalysis] = useState<any>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

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
      'image/*': ['.jpeg', '.jpg', '.png', '.dicom']
    },
    multiple: false
  });

  const handleAnalyze = async () => {
    if (!selectedImage) return;

    setIsAnalyzing(true);
    
    // Simulate AI analysis
    setTimeout(() => {
      const mockAnalysis = {
        imageType: 'Chest X-Ray',
        findings: [
          {
            finding: 'Normal lung fields',
            confidence: 0.94,
            severity: 'Normal',
            location: 'Bilateral lung fields'
          },
          {
            finding: 'Clear cardiac silhouette',
            confidence: 0.91,
            severity: 'Normal',
            location: 'Mediastinum'
          },
          {
            finding: 'No acute abnormalities',
            confidence: 0.88,
            severity: 'Normal',
            location: 'Overall'
          }
        ],
        overallAssessment: 'Normal chest X-ray with no acute findings',
        recommendations: [
          'No immediate intervention required',
          'Continue routine screening',
          'Correlate with clinical symptoms if present'
        ],
        riskLevel: 'Low'
      };
      
      setAnalysis(mockAnalysis);
      setIsAnalyzing(false);
    }, 4000);
  };

  const imageTypes = [
    { name: 'X-Ray', icon: ImageIcon, description: 'Chest, spine, extremities' },
    { name: 'CT Scan', icon: Brain, description: 'Brain, abdomen, chest' },
    { name: 'MRI', icon: Scan, description: 'Soft tissues, joints' },
    { name: 'Ultrasound', icon: Eye, description: 'Organs, pregnancy' }
  ];

  return (
    <div className="min-h-screen py-8 px-4 sm:px-6 lg:px-8">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-12"
        >
          <div className="bg-gradient-to-r from-purple-600 to-pink-600 w-16 h-16 rounded-2xl flex items-center justify-center mx-auto mb-4">
            <Brain className="w-8 h-8 text-white" />
          </div>
          <h1 className="text-3xl font-bold text-gray-900 mb-2">Medical Imaging Analysis</h1>
          <p className="text-gray-600">AI-powered analysis of X-rays, CT scans, MRIs, and other medical images</p>
        </motion.div>

        {/* Supported Image Types */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8"
        >
          {imageTypes.map((type, index) => {
            const Icon = type.icon;
            return (
              <div key={index} className="bg-white rounded-xl p-4 shadow-md text-center">
                <Icon className="w-8 h-8 text-purple-600 mx-auto mb-2" />
                <h3 className="font-semibold text-gray-900 mb-1">{type.name}</h3>
                <p className="text-xs text-gray-600">{type.description}</p>
              </div>
            );
          })}
        </motion.div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Upload Section */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.2 }}
          >
            <div className="bg-white rounded-2xl shadow-lg p-8">
              <h2 className="text-xl font-bold text-gray-900 mb-6">Upload Medical Image</h2>
              
              {/* File Upload Area */}
              <div
                {...getRootProps()}
                className={`border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-all duration-200 ${
                  isDragActive 
                    ? 'border-purple-500 bg-purple-50' 
                    : 'border-gray-300 hover:border-purple-500 hover:bg-purple-50'
                }`}
              >
                <input {...getInputProps()} />
                <Upload className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                {isDragActive ? (
                  <p className="text-purple-600 font-medium">Drop the image here...</p>
                ) : (
                  <div>
                    <p className="text-gray-600 mb-2">Drag & drop medical image here, or click to select</p>
                    <p className="text-sm text-gray-500">Supports JPEG, PNG, DICOM formats</p>
                  </div>
                )}
              </div>

              {/* Image Preview */}
              {imagePreview && (
                <div className="mt-6">
                  <h3 className="font-semibold text-gray-900 mb-3">Image Preview</h3>
                  <div className="relative">
                    <img
                      src={imagePreview}
                      alt="Medical scan preview"
                      className="w-full h-64 object-contain bg-gray-100 rounded-xl"
                    />
                    <div className="absolute top-2 right-2 bg-white rounded-lg p-2 shadow-md">
                      <p className="text-sm text-gray-600">{selectedImage?.name}</p>
                    </div>
                  </div>
                </div>
              )}

              {/* Analyze Button */}
              {selectedImage && (
                <button
                  onClick={handleAnalyze}
                  disabled={isAnalyzing}
                  className="w-full mt-6 bg-gradient-to-r from-purple-600 to-pink-600 text-white py-4 px-6 rounded-xl font-semibold text-lg disabled:opacity-50 disabled:cursor-not-allowed hover:shadow-lg transition-all duration-200 flex items-center justify-center space-x-2"
                >
                  {isAnalyzing ? (
                    <>
                      <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
                      <span>Analyzing Image...</span>
                    </>
                  ) : (
                    <>
                      <Brain className="w-5 h-5" />
                      <span>Analyze with AI</span>
                    </>
                  )}
                </button>
              )}
            </div>
          </motion.div>

          {/* Analysis Results */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.3 }}
          >
            {analysis ? (
              <div className="space-y-6">
                {/* Overall Assessment */}
                <div className="bg-white rounded-2xl shadow-lg p-8">
                  <div className="flex items-start space-x-4 mb-6">
                    <div className={`p-3 rounded-xl ${
                      analysis.riskLevel === 'High' ? 'bg-red-100' :
                      analysis.riskLevel === 'Medium' ? 'bg-yellow-100' :
                      'bg-green-100'
                    }`}>
                      {analysis.riskLevel === 'High' ? (
                        <AlertTriangle className="w-6 h-6 text-red-600" />
                      ) : (
                        <CheckCircle className="w-6 h-6 text-green-600" />
                      )}
                    </div>
                    <div className="flex-1">
                      <h2 className="text-xl font-bold text-gray-900 mb-2">Analysis Results</h2>
                      <p className="text-lg text-gray-700 mb-2">{analysis.imageType}</p>
                      <p className="text-gray-600">{analysis.overallAssessment}</p>
                    </div>
                  </div>
                </div>

                {/* Detailed Findings */}
                <div className="bg-white rounded-2xl shadow-lg p-8">
                  <h3 className="text-xl font-bold text-gray-900 mb-6">Detailed Findings</h3>
                  <div className="space-y-4">
                    {analysis.findings.map((finding: any, index: number) => (
                      <div key={index} className="border border-gray-200 rounded-xl p-4">
                        <div className="flex justify-between items-start mb-2">
                          <h4 className="font-semibold text-gray-900">{finding.finding}</h4>
                          <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                            finding.severity === 'Normal' ? 'bg-green-100 text-green-800' :
                            finding.severity === 'Abnormal' ? 'bg-red-100 text-red-800' :
                            'bg-yellow-100 text-yellow-800'
                          }`}>
                            {finding.severity}
                          </span>
                        </div>
                        <p className="text-sm text-gray-600 mb-2">Location: {finding.location}</p>
                        <div className="flex items-center space-x-2">
                          <span className="text-sm text-gray-600">Confidence:</span>
                          <div className="bg-gray-200 rounded-full h-2 w-32">
                            <div 
                              className="bg-blue-500 h-2 rounded-full"
                              style={{ width: `${finding.confidence * 100}%` }}
                            ></div>
                          </div>
                          <span className="text-sm font-medium">{Math.round(finding.confidence * 100)}%</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Recommendations */}
                <div className="bg-white rounded-2xl shadow-lg p-8">
                  <h3 className="text-xl font-bold text-gray-900 mb-6">Recommendations</h3>
                  <div className="space-y-3">
                    {analysis.recommendations.map((rec: string, index: number) => (
                      <div key={index} className="flex items-start space-x-3">
                        <div className="bg-purple-100 rounded-full p-1 mt-1">
                          <div className="w-2 h-2 bg-purple-600 rounded-full"></div>
                        </div>
                        <span className="text-gray-700">{rec}</span>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Download Report */}
                <button className="w-full bg-gray-800 text-white py-3 px-6 rounded-xl font-semibold hover:bg-gray-700 transition-all duration-200 flex items-center justify-center space-x-2">
                  <Download className="w-5 h-5" />
                  <span>Download Analysis Report</span>
                </button>
              </div>
            ) : (
              <div className="bg-white rounded-2xl shadow-lg p-8 text-center">
                <Brain className="w-16 h-16 text-gray-300 mx-auto mb-4" />
                <h3 className="text-xl font-bold text-gray-900 mb-2">Ready to Analyze</h3>
                <p className="text-gray-600">
                  Upload a medical image to get started with AI-powered analysis. 
                  Our advanced algorithms can detect various conditions and abnormalities.
                </p>
              </div>
            )}
          </motion.div>
        </div>

        {/* Disclaimer */}
        <div className="mt-8 p-4 bg-gray-50 rounded-xl">
          <p className="text-sm text-gray-600 text-center">
            <strong>Medical Disclaimer:</strong> AI image analysis is for educational and research purposes only. 
            Always consult with qualified radiologists and healthcare professionals for medical diagnosis and treatment decisions.
          </p>
        </div>
      </div>
    </div>
  );
};