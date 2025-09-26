import React, { useState } from 'react';
import { motion } from 'framer-motion';
import {
  BarChart3,
  Users,
  Activity,
  TrendingUp,
  Heart,
  Brain,
  Eye,
  Shield,
  Clock,
  CheckCircle,
  AlertTriangle,
  Calendar
} from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell, BarChart, Bar } from 'recharts';

export const DashboardPage: React.FC = () => {
  const [timeFilter, setTimeFilter] = useState('7days');

  const statsData = [
    { label: 'Total Diagnoses', value: '2,847', change: '+12%', icon: Activity, color: 'bg-blue-500' },
    { label: 'Active Patients', value: '1,234', change: '+8%', icon: Users, color: 'bg-green-500' },
    { label: 'Accuracy Rate', value: '94.7%', change: '+2.3%', icon: CheckCircle, color: 'bg-purple-500' },
    { label: 'Response Time', value: '1.2s', change: '-15%', icon: Clock, color: 'bg-orange-500' }
  ];

  const diagnosisData = [
    { month: 'Jan', diagnoses: 245, accuracy: 92 },
    { month: 'Feb', diagnoses: 312, accuracy: 94 },
    { month: 'Mar', diagnoses: 389, accuracy: 95 },
    { month: 'Apr', diagnoses: 431, accuracy: 93 },
    { month: 'May', diagnoses: 502, accuracy: 96 },
    { month: 'Jun', diagnoses: 478, accuracy: 94 }
  ];

  const conditionDistribution = [
    { name: 'Respiratory', value: 35, color: '#3b82f6' },
    { name: 'Cardiovascular', value: 28, color: '#ef4444' },
    { name: 'Neurological', value: 15, color: '#8b5cf6' },
    { name: 'Gastrointestinal', value: 12, color: '#f59e0b' },
    { name: 'Other', value: 10, color: '#10b981' }
  ];

  const recentDiagnoses = [
    {
      id: 1,
      patient: 'Patient #1247',
      condition: 'Viral Upper Respiratory Infection',
      confidence: 0.92,
      urgency: 'Low',
      time: '5 minutes ago'
    },
    {
      id: 2,
      patient: 'Patient #1248',
      condition: 'Hypertension - Stage 1',
      confidence: 0.87,
      urgency: 'Medium',
      time: '12 minutes ago'
    },
    {
      id: 3,
      patient: 'Patient #1249',
      condition: 'Type 2 Diabetes',
      confidence: 0.94,
      urgency: 'Medium',
      time: '18 minutes ago'
    },
    {
      id: 4,
      patient: 'Patient #1250',
      condition: 'Migraine',
      confidence: 0.89,
      urgency: 'Low',
      time: '25 minutes ago'
    }
  ];

  const systemHealth = [
    { component: 'AI Diagnosis Engine', status: 'Healthy', uptime: '99.9%', responseTime: '0.8s' },
    { component: 'Medical Imaging AI', status: 'Healthy', uptime: '99.7%', responseTime: '2.1s' },
    { component: 'Lab Analysis AI', status: 'Warning', uptime: '98.5%', responseTime: '1.9s' },
    { component: 'Treatment Planner', status: 'Healthy', uptime: '99.8%', responseTime: '1.2s' }
  ];

  return (
    <div className="min-h-screen py-8 px-4 sm:px-6 lg:px-8 bg-gray-50">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8"
        >
          <div className="flex justify-between items-center mb-6">
            <div>
              <h1 className="text-3xl font-bold text-gray-900">AI Hospital Dashboard</h1>
              <p className="text-gray-600">Real-time analytics and system monitoring</p>
            </div>
            <div className="flex items-center space-x-3">
              <select
                value={timeFilter}
                onChange={(e) => setTimeFilter(e.target.value)}
                className="bg-white border border-gray-300 rounded-lg px-3 py-2 focus:ring-2 focus:ring-blue-500"
              >
                <option value="7days">Last 7 days</option>
                <option value="30days">Last 30 days</option>
                <option value="90days">Last 90 days</option>
              </select>
              <div className="flex items-center space-x-2 text-sm text-gray-600">
                <Calendar className="w-4 h-4" />
                <span>{new Date().toLocaleDateString()}</span>
              </div>
            </div>
          </div>
        </motion.div>

        {/* Stats Grid */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8"
        >
          {statsData.map((stat, index) => {
            const Icon = stat.icon;
            return (
              <div key={index} className="bg-white rounded-2xl shadow-lg p-6 hover:shadow-xl transition-all duration-300">
                <div className="flex items-center justify-between mb-4">
                  <div className={`${stat.color} p-3 rounded-xl`}>
                    <Icon className="w-6 h-6 text-white" />
                  </div>
                  <span className={`text-sm font-medium ${
                    stat.change.startsWith('+') ? 'text-green-600' : 'text-red-600'
                  }`}>
                    {stat.change}
                  </span>
                </div>
                <div>
                  <p className="text-2xl font-bold text-gray-900 mb-1">{stat.value}</p>
                  <p className="text-sm text-gray-600">{stat.label}</p>
                </div>
              </div>
            );
          })}
        </motion.div>

        {/* Charts Row */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 mb-8">
          {/* Diagnosis Trends */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.2 }}
            className="lg:col-span-2 bg-white rounded-2xl shadow-lg p-8"
          >
            <h2 className="text-xl font-bold text-gray-900 mb-6">Diagnosis Trends</h2>
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={diagnosisData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="month" />
                  <YAxis yAxisId="left" />
                  <YAxis yAxisId="right" orientation="right" />
                  <Tooltip />
                  <Bar yAxisId="left" dataKey="diagnoses" fill="#3b82f6" name="Diagnoses" />
                  <Line yAxisId="right" type="monotone" dataKey="accuracy" stroke="#ef4444" strokeWidth={3} name="Accuracy %" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </motion.div>

          {/* Condition Distribution */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.3 }}
            className="bg-white rounded-2xl shadow-lg p-8"
          >
            <h2 className="text-xl font-bold text-gray-900 mb-6">Top Conditions</h2>
            <div className="h-60">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={conditionDistribution}
                    cx="50%"
                    cy="50%"
                    innerRadius={40}
                    outerRadius={80}
                    paddingAngle={5}
                    dataKey="value"
                  >
                    {conditionDistribution.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </div>
            <div className="space-y-2">
              {conditionDistribution.map((condition, index) => (
                <div key={index} className="flex items-center space-x-2">
                  <div className="w-3 h-3 rounded-full" style={{ backgroundColor: condition.color }}></div>
                  <span className="text-sm text-gray-700">{condition.name}</span>
                  <span className="text-sm font-medium text-gray-900">{condition.value}%</span>
                </div>
              ))}
            </div>
          </motion.div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Recent Diagnoses */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4 }}
            className="bg-white rounded-2xl shadow-lg p-8"
          >
            <h2 className="text-xl font-bold text-gray-900 mb-6">Recent Diagnoses</h2>
            <div className="space-y-4">
              {recentDiagnoses.map((diagnosis) => (
                <div key={diagnosis.id} className="border border-gray-200 rounded-xl p-4 hover:bg-gray-50 transition-all duration-200">
                  <div className="flex justify-between items-start mb-2">
                    <div>
                      <h4 className="font-semibold text-gray-900">{diagnosis.patient}</h4>
                      <p className="text-sm text-gray-700">{diagnosis.condition}</p>
                    </div>
                    <div className="text-right">
                      <span className={`inline-block px-2 py-1 rounded-full text-xs font-medium ${
                        diagnosis.urgency === 'High' ? 'bg-red-100 text-red-800' :
                        diagnosis.urgency === 'Medium' ? 'bg-yellow-100 text-yellow-800' :
                        'bg-green-100 text-green-800'
                      }`}>
                        {diagnosis.urgency}
                      </span>
                    </div>
                  </div>
                  <div className="flex justify-between items-center text-sm text-gray-600">
                    <div className="flex items-center space-x-2">
                      <span>Confidence:</span>
                      <div className="bg-gray-200 rounded-full h-2 w-20">
                        <div 
                          className="bg-blue-500 h-2 rounded-full"
                          style={{ width: `${diagnosis.confidence * 100}%` }}
                        ></div>
                      </div>
                      <span>{Math.round(diagnosis.confidence * 100)}%</span>
                    </div>
                    <span>{diagnosis.time}</span>
                  </div>
                </div>
              ))}
            </div>
          </motion.div>

          {/* System Health */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.5 }}
            className="bg-white rounded-2xl shadow-lg p-8"
          >
            <h2 className="text-xl font-bold text-gray-900 mb-6">System Health</h2>
            <div className="space-y-4">
              {systemHealth.map((system, index) => (
                <div key={index} className="border border-gray-200 rounded-xl p-4">
                  <div className="flex justify-between items-start mb-3">
                    <h4 className="font-semibold text-gray-900">{system.component}</h4>
                    <div className="flex items-center space-x-2">
                      {system.status === 'Healthy' ? (
                        <CheckCircle className="w-5 h-5 text-green-600" />
                      ) : (
                        <AlertTriangle className="w-5 h-5 text-yellow-600" />
                      )}
                      <span className={`text-sm font-medium ${
                        system.status === 'Healthy' ? 'text-green-600' : 'text-yellow-600'
                      }`}>
                        {system.status}
                      </span>
                    </div>
                  </div>
                  <div className="grid grid-cols-2 gap-4 text-sm text-gray-600">
                    <div>
                      <span className="block">Uptime</span>
                      <span className="font-medium text-gray-900">{system.uptime}</span>
                    </div>
                    <div>
                      <span className="block">Response Time</span>
                      <span className="font-medium text-gray-900">{system.responseTime}</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </motion.div>
        </div>
      </div>
    </div>
  );
};