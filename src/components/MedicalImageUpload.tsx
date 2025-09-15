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
  FileText,
  Activity,
  Heart,
  Zap
} from 'lucide-react';

interface MedicalImageUploadProps {
  onAnalysisComplete: (analysis: any) => void;
}

export const MedicalImageUpload: React.FC<MedicalImageUploadProps> = ({ 
  onAnalysisComplete 
}) => {
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [imageType, setImageType] = useState('auto');
  const [bodyPart, setBodyPart] = useState('');
  const [clinicalContext, setClinicalContext] = useState('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState<any>(null);

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
    maxSize: 50 * 1024 * 1024 // 50MB max
  });

  const handleAnalyze = async () => {
    if (!selectedImage) return;

    setIsAnalyzing(true);
    
    try {
      const formData = new FormData();
      formData.append('image', selectedImage);
      formData.append('image_type', imageType);
      formData.append('body_part', bodyPart);
      formData.append('clinical_context', clinicalContext);
      formData.append('study_date', new Date().toISOString().split('T')[0]);

      const response = await fetch(`${import.meta.env.VITE_API_URL}/api/imaging/upload-image`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        },
        body: formData
      });

      const result = await response.json();

      if (response.ok) {
        setAnalysisResult(result);
        onAnalysisComplete(result);
      } else {
        throw new Error(result.error || 'Analysis failed');
      }
    } catch (error) {
      console.error('Image analysis failed:', error);
      alert('Failed to analyze image. Please try again.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  const imageTypes = [
    { value: 'auto', label: 'Auto-detect', icon: Brain },
    { value: 'chest_xray', label: 'Chest X-Ray', icon: Activity },
    { value: 'brain_mri', label: 'Brain MRI', icon: Brain },
    { value: 'ct_scan', label: 'CT Scan', icon: Scan },
    { value: 'ultrasound', label: 'Ultrasound', icon: Heart },
    { value: 'mammography', label: 'Mammography', icon: Heart },
    { value: 'bone_xray', label: 'Bone X-Ray', icon: Activity }
  ];

  const bodyParts = [
    'Chest/Thorax', 'Head/Brain', 'Abdomen', 'Pelvis', 'Spine',
    'Upper Extremity', 'Lower Extremity', 'Neck', 'Heart', 'Other'
  ];

  return (
    <div className="space-y-6">
      {/* Image Upload Area */}
      <div className="bg-white rounded-2xl shadow-lg p-8">
        <h2 className="text-2xl font-bold text-gray-900 mb-6 flex items-center space-x-3">
          <ImageIcon className="w-7 h-7 text-purple-600" />
          <span>Medical Image Analysis</span>
        </h2>
        
        <div
          {...getRootProps()}
          className={`border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-all duration-200 ${
            isDragActive 
              ? 'border-purple-500 bg-purple-50' 
              : 'border-gray-300 hover:border-purple-500 hover:bg-purple-50'
          }`}
        >
          <input {...getInputProps()} />
          <Upload className="w-16 h-16 text-gray-400 mx-auto mb-4" />
          {isDragActive ? (
            <p className="text-purple-600 font-medium text-lg">Drop the medical image here...</p>
          ) : (
            <div>
              <p className="text-gray-600 mb-2 text-lg">Drag & drop medical image here, or click to select</p>
              <p className="text-sm text-gray-500">Supports JPEG, PNG, TIFF, DICOM formats (max 50MB)</p>
            </div>
          )}
        </div>

        {/* Image Preview */}
        {imagePreview && (
          <div className="mt-6">
            <h3 className="font-semibold text-gray-900 mb-3">Image Preview</h3>
            <div className="relative bg-gray-100 rounded-xl overflow-hidden">
              <img
                src={imagePreview}
                alt="Medical scan preview"
                className="w-full h-80 object-contain"
              />
              <div className="absolute top-4 right-4 bg-white rounded-lg p-3 shadow-lg">
                <p className="text-sm font-medium text-gray-900">{selectedImage?.name}</p>
                <p className="text-xs text-gray-600">
                  {(selectedImage?.size || 0 / (1024 * 1024)).toFixed(2)} MB
                </p>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Analysis Configuration */}
      {selectedImage && (
        <div className="bg-white rounded-2xl shadow-lg p-8">
          <h3 className="text-xl font-bold text-gray-900 mb-6">Analysis Configuration</h3>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-3">
                Image Type
              </label>
              <div className="grid grid-cols-2 gap-2">
                {imageTypes.map((type) => {
                  const Icon = type.icon;
                  return (
                    <label key={type.value} className="flex items-center p-3 border rounded-xl cursor-pointer hover:bg-purple-50 transition-all duration-200">
                      <input
                        type="radio"
                        value={type.value}
                        checked={imageType === type.value}
                        onChange={(e) => setImageType(e.target.value)}
                        className="mr-3"
                      />
                      <Icon className="w-4 h-4 mr-2 text-purple-600" />
                      <span className="text-sm font-medium">{type.label}</span>
                    </label>
                  );
                })}
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-3">
                Body Part
              </label>
              <select
                value={bodyPart}
                onChange={(e) => setBodyPart(e.target.value)}
                className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-purple-500 focus:border-purple-500"
              >
                <option value="">Select body part</option>
                {bodyParts.map((part) => (
                  <option key={part} value={part}>{part}</option>
                ))}
              </select>
            </div>
          </div>

          <div className="mt-6">
            <label className="block text-sm font-medium text-gray-700 mb-3">
              Clinical Context (Optional)
            </label>
            <textarea
              value={clinicalContext}
              onChange={(e) => setClinicalContext(e.target.value)}
              rows={3}
              className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-purple-500 focus:border-purple-500"
              placeholder="Provide any relevant clinical information (symptoms, history, reason for imaging)..."
            />
          </div>

          <button
            onClick={handleAnalyze}
            disabled={isAnalyzing}
            className="w-full mt-6 bg-gradient-to-r from-purple-600 to-pink-600 text-white py-4 px-6 rounded-xl font-semibold text-lg disabled:opacity-50 disabled:cursor-not-allowed hover:shadow-lg transition-all duration-200 flex items-center justify-center space-x-3"
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
        </div>
      )}

      {/* Safety Notice */}
      <div className="bg-yellow-50 border border-yellow-200 rounded-xl p-4">
        <div className="flex items-start space-x-3">
          <AlertTriangle className="w-5 h-5 text-yellow-600 mt-0.5" />
          <div>
            <h4 className="font-bold text-yellow-900 mb-1">AI Image Analysis Notice</h4>
            <p className="text-yellow-800 text-sm">
              This is an automated preliminary analysis using AI. It is not a certified medical diagnosis. 
              A qualified radiologist must always provide the final and definitive interpretation.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};