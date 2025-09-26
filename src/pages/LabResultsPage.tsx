import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { 
  Activity, 
  TrendingUp, 
  TrendingDown, 
  AlertTriangle, 
  CheckCircle,
  Upload,
  FileText,
  Calendar
} from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar } from 'recharts';

export const LabResultsPage: React.FC = () => {
  const [selectedTest, setSelectedTest] = useState('complete-blood-count');

  const labTests = {
    'complete-blood-count': {
      name: 'Complete Blood Count (CBC)',
      date: '2024-01-15',
      results: [
        { parameter: 'White Blood Cells', value: 6.2, unit: 'K/μL', range: '4.5-11.0', status: 'normal' },
        { parameter: 'Red Blood Cells', value: 4.8, unit: 'M/μL', range: '4.2-5.4', status: 'normal' },
        { parameter: 'Hemoglobin', value: 14.2, unit: 'g/dL', range: '12.0-15.5', status: 'normal' },
        { parameter: 'Hematocrit', value: 42.1, unit: '%', range: '36.0-46.0', status: 'normal' },
        { parameter: 'Platelets', value: 285, unit: 'K/μL', range: '150-400', status: 'normal' }
      ]
    },
    'lipid-panel': {
      name: 'Lipid Panel',
      date: '2024-01-15',
      results: [
        { parameter: 'Total Cholesterol', value: 195, unit: 'mg/dL', range: '<200', status: 'normal' },
        { parameter: 'LDL Cholesterol', value: 118, unit: 'mg/dL', range: '<100', status: 'high' },
        { parameter: 'HDL Cholesterol', value: 52, unit: 'mg/dL', range: '>40', status: 'normal' },
        { parameter: 'Triglycerides', value: 125, unit: 'mg/dL', range: '<150', status: 'normal' }
      ]
    },
    'metabolic-panel': {
      name: 'Basic Metabolic Panel',
      date: '2024-01-15',
      results: [
        { parameter: 'Glucose', value: 92, unit: 'mg/dL', range: '70-100', status: 'normal' },
        { parameter: 'Sodium', value: 140, unit: 'mmol/L', range: '136-145', status: 'normal' },
        { parameter: 'Potassium', value: 4.1, unit: 'mmol/L', range: '3.5-5.1', status: 'normal' },
        { parameter: 'Chloride', value: 102, unit: 'mmol/L', range: '98-107', status: 'normal' },
        { parameter: 'Creatinine', value: 0.9, unit: 'mg/dL', range: '0.6-1.2', status: 'normal' }
      ]
    }
  };

  const trendData = [
    { month: 'Jan', cholesterol: 210, glucose: 95, hemoglobin: 14.0 },
    { month: 'Apr', cholesterol: 205, glucose: 88, hemoglobin: 14.1 },
    { month: 'Jul', cholesterol: 200, glucose: 90, hemoglobin: 14.2 },
    { month: 'Oct', cholesterol: 195, glucose: 92, hemoglobin: 14.2 }
  ];

  const currentTest = labTests[selectedTest as keyof typeof labTests];

  const aiInsights = {
    summary: 'Overall lab results show good health markers with one area of attention.',
    keyFindings: [
      'LDL cholesterol is slightly elevated at 118 mg/dL',
      'All other parameters are within normal ranges',
      'Kidney function markers are excellent',
      'Blood sugar levels are well controlled'
    ],
    recommendations: [
      'Consider dietary modifications to reduce LDL cholesterol',
      'Increase physical activity to 150 minutes per week',
      'Recheck lipid panel in 3 months',
      'Continue current lifestyle for other parameters'
    ],
    riskAssessment: 'Low to moderate cardiovascular risk'
  };

  return (
    <div className="min-h-screen py-8 px-4 sm:px-6 lg:px-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-12"
        >
          <div className="bg-gradient-to-r from-green-600 to-blue-600 w-16 h-16 rounded-2xl flex items-center justify-center mx-auto mb-4">
            <Activity className="w-8 h-8 text-white" />
          </div>
          <h1 className="text-3xl font-bold text-gray-900 mb-2">Lab Results Analysis</h1>
          <p className="text-gray-600">AI-powered interpretation of your laboratory test results</p>
        </motion.div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Test Selection Sidebar */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            className="lg:col-span-1"
          >
            <div className="bg-white rounded-2xl shadow-lg p-6 mb-6">
              <h2 className="text-xl font-bold text-gray-900 mb-4">Recent Lab Tests</h2>
              <div className="space-y-2">
                {Object.entries(labTests).map(([key, test]) => (
                  <button
                    key={key}
                    onClick={() => setSelectedTest(key)}
                    className={`w-full text-left p-3 rounded-xl transition-all duration-200 ${
                      selectedTest === key
                        ? 'bg-blue-50 border-2 border-blue-200 text-blue-900'
                        : 'hover:bg-gray-50 border-2 border-transparent'
                    }`}
                  >
                    <div className="flex items-center space-x-3">
                      <FileText className="w-5 h-5 text-gray-600" />
                      <div>
                        <p className="font-medium">{test.name}</p>
                        <p className="text-sm text-gray-500 flex items-center space-x-1">
                          <Calendar className="w-3 h-3" />
                          <span>{test.date}</span>
                        </p>
                      </div>
                    </div>
                  </button>
                ))}
              </div>
              
              <button className="w-full mt-4 bg-gradient-to-r from-green-600 to-blue-600 text-white py-3 px-4 rounded-xl font-semibold hover:shadow-lg transition-all duration-200 flex items-center justify-center space-x-2">
                <Upload className="w-5 h-5" />
                <span>Upload New Results</span>
              </button>
            </div>

            {/* AI Insights Summary */}
            <div className="bg-white rounded-2xl shadow-lg p-6">
              <h3 className="text-lg font-bold text-gray-900 mb-4 flex items-center space-x-2">
                <Activity className="w-5 h-5" />
                <span>AI Insights</span>
              </h3>
              <div className="space-y-4">
                <div className="p-3 bg-blue-50 rounded-xl">
                  <p className="text-sm text-blue-800">{aiInsights.summary}</p>
                </div>
                <div className="p-3 bg-orange-50 rounded-xl">
                  <p className="text-sm text-orange-800 font-medium">Risk Level: {aiInsights.riskAssessment}</p>
                </div>
              </div>
            </div>
          </motion.div>

          {/* Main Content */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="lg:col-span-2 space-y-6"
          >
            {/* Current Test Results */}
            <div className="bg-white rounded-2xl shadow-lg p-8">
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-2xl font-bold text-gray-900">{currentTest.name}</h2>
                <div className="flex items-center space-x-2 text-gray-600">
                  <Calendar className="w-4 h-4" />
                  <span>{currentTest.date}</span>
                </div>
              </div>

              <div className="space-y-4">
                {currentTest.results.map((result, index) => (
                  <div key={index} className="flex items-center justify-between p-4 bg-gray-50 rounded-xl">
                    <div className="flex-1">
                      <h4 className="font-medium text-gray-900">{result.parameter}</h4>
                      <p className="text-sm text-gray-600">Normal range: {result.range}</p>
                    </div>
                    <div className="flex items-center space-x-4">
                      <div className="text-right">
                        <p className="font-bold text-lg">{result.value}</p>
                        <p className="text-sm text-gray-600">{result.unit}</p>
                      </div>
                      <div className={`p-2 rounded-full ${
                        result.status === 'normal' ? 'bg-green-100' :
                        result.status === 'high' ? 'bg-red-100' :
                        'bg-yellow-100'
                      }`}>
                        {result.status === 'normal' ? (
                          <CheckCircle className="w-5 h-5 text-green-600" />
                        ) : result.status === 'high' ? (
                          <TrendingUp className="w-5 h-5 text-red-600" />
                        ) : (
                          <TrendingDown className="w-5 h-5 text-yellow-600" />
                        )}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Trend Analysis */}
            <div className="bg-white rounded-2xl shadow-lg p-8">
              <h3 className="text-xl font-bold text-gray-900 mb-6">Trend Analysis</h3>
              <div className="h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={trendData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="month" />
                    <YAxis />
                    <Tooltip />
                    <Line type="monotone" dataKey="cholesterol" stroke="#ef4444" strokeWidth={2} name="Total Cholesterol" />
                    <Line type="monotone" dataKey="glucose" stroke="#3b82f6" strokeWidth={2} name="Glucose" />
                    <Line type="monotone" dataKey="hemoglobin" stroke="#10b981" strokeWidth={2} name="Hemoglobin" />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Detailed AI Analysis */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Key Findings */}
              <div className="bg-white rounded-2xl shadow-lg p-6">
                <h3 className="text-lg font-bold text-gray-900 mb-4 flex items-center space-x-2">
                  <AlertTriangle className="w-5 h-5 text-orange-600" />
                  <span>Key Findings</span>
                </h3>
                <div className="space-y-3">
                  {aiInsights.keyFindings.map((finding, index) => (
                    <div key={index} className="flex items-start space-x-3">
                      <div className="bg-orange-100 rounded-full p-1 mt-1">
                        <div className="w-2 h-2 bg-orange-600 rounded-full"></div>
                      </div>
                      <p className="text-gray-700 text-sm">{finding}</p>
                    </div>
                  ))}
                </div>
              </div>

              {/* Recommendations */}
              <div className="bg-white rounded-2xl shadow-lg p-6">
                <h3 className="text-lg font-bold text-gray-900 mb-4 flex items-center space-x-2">
                  <CheckCircle className="w-5 h-5 text-green-600" />
                  <span>Recommendations</span>
                </h3>
                <div className="space-y-3">
                  {aiInsights.recommendations.map((rec, index) => (
                    <div key={index} className="flex items-start space-x-3">
                      <div className="bg-green-100 rounded-full p-1 mt-1">
                        <div className="w-2 h-2 bg-green-600 rounded-full"></div>
                      </div>
                      <p className="text-gray-700 text-sm">{rec}</p>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </motion.div>
        </div>

        {/* Disclaimer */}
        <div className="mt-8 p-4 bg-gray-50 rounded-xl">
          <p className="text-sm text-gray-600 text-center">
            <strong>Medical Disclaimer:</strong> AI lab result interpretation is for informational purposes only. 
            Always consult with your healthcare provider for proper medical interpretation and treatment decisions.
          </p>
        </div>
      </div>
    </div>
  );
};