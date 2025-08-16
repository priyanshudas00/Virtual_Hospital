import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { useAuth } from '../contexts/AuthContext';
import {
  Calendar,
  Clock,
  FileText,
  CreditCard,
  User,
  Bell,
  Activity,
  Heart,
  TrendingUp,
  AlertCircle,
  CheckCircle,
  Plus,
  Download,
  Eye,
  Edit
} from 'lucide-react';
import { format } from 'date-fns';

interface Appointment {
  id: string;
  type: string;
  status: 'scheduled' | 'in_progress' | 'completed' | 'cancelled';
  date: string;
  time: string;
  cost: number;
  paymentStatus: 'pending' | 'paid' | 'failed';
}

interface MedicalRecord {
  id: string;
  date: string;
  type: 'diagnosis' | 'lab_result' | 'imaging' | 'prescription';
  title: string;
  summary: string;
  aiConfidence?: number;
}

export const PatientDashboardPage: React.FC = () => {
  const { user, updateProfile } = useAuth();
  const [activeTab, setActiveTab] = useState('overview');
  const [appointments, setAppointments] = useState<Appointment[]>([]);
  const [medicalRecords, setMedicalRecords] = useState<MedicalRecord[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Load user data
    loadDashboardData();
  }, []);

  const loadDashboardData = async () => {
    try {
      // Simulate loading appointments and medical records
      setAppointments([
        {
          id: '1',
          type: 'AI Consultation',
          status: 'scheduled',
          date: '2024-01-20',
          time: '10:00',
          cost: 49.99,
          paymentStatus: 'paid'
        },
        {
          id: '2',
          type: 'Lab Test Analysis',
          status: 'completed',
          date: '2024-01-15',
          time: '14:30',
          cost: 39.99,
          paymentStatus: 'paid'
        }
      ]);

      setMedicalRecords([
        {
          id: '1',
          date: '2024-01-15',
          type: 'diagnosis',
          title: 'AI Symptom Analysis',
          summary: 'Viral upper respiratory infection diagnosed with 87% confidence',
          aiConfidence: 0.87
        },
        {
          id: '2',
          date: '2024-01-10',
          type: 'lab_result',
          title: 'Complete Blood Count',
          summary: 'All parameters within normal ranges',
        }
      ]);
    } catch (error) {
      console.error('Error loading dashboard data:', error);
    } finally {
      setLoading(false);
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'scheduled': return 'bg-blue-100 text-blue-800';
      case 'in_progress': return 'bg-yellow-100 text-yellow-800';
      case 'completed': return 'bg-green-100 text-green-800';
      case 'cancelled': return 'bg-red-100 text-red-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  const getPaymentStatusColor = (status: string) => {
    switch (status) {
      case 'paid': return 'text-green-600';
      case 'pending': return 'text-yellow-600';
      case 'failed': return 'text-red-600';
      default: return 'text-gray-600';
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  return (
    <div className="min-h-screen py-8 px-4 sm:px-6 lg:px-8 bg-gray-50">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8"
        >
          <div className="bg-white rounded-2xl shadow-lg p-6">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-4">
                <div className="bg-gradient-to-r from-blue-600 to-green-600 w-16 h-16 rounded-full flex items-center justify-center">
                  <User className="w-8 h-8 text-white" />
                </div>
                <div>
                  <h1 className="text-2xl font-bold text-gray-900">
                    Welcome back, {user?.firstName} {user?.lastName}
                  </h1>
                  <p className="text-gray-600">Patient ID: {user?.id?.slice(0, 8)}</p>
                </div>
              </div>
              <div className="flex items-center space-x-3">
                <button className="p-2 bg-gray-100 rounded-lg hover:bg-gray-200 transition-all duration-200">
                  <Bell className="w-5 h-5 text-gray-600" />
                </button>
                <button
                  onClick={() => window.location.href = '/book-appointment'}
                  className="bg-gradient-to-r from-blue-600 to-green-600 text-white px-4 py-2 rounded-lg font-semibold hover:shadow-lg transition-all duration-200 flex items-center space-x-2"
                >
                  <Plus className="w-4 h-4" />
                  <span>Book Appointment</span>
                </button>
              </div>
            </div>
          </div>
        </motion.div>

        {/* Quick Stats */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8"
        >
          <div className="bg-white rounded-2xl shadow-lg p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600">Total Appointments</p>
                <p className="text-2xl font-bold text-gray-900">{appointments.length}</p>
              </div>
              <Calendar className="w-8 h-8 text-blue-600" />
            </div>
          </div>
          <div className="bg-white rounded-2xl shadow-lg p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600">Medical Records</p>
                <p className="text-2xl font-bold text-gray-900">{medicalRecords.length}</p>
              </div>
              <FileText className="w-8 h-8 text-green-600" />
            </div>
          </div>
          <div className="bg-white rounded-2xl shadow-lg p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600">Health Score</p>
                <p className="text-2xl font-bold text-gray-900">85/100</p>
              </div>
              <Heart className="w-8 h-8 text-red-600" />
            </div>
          </div>
          <div className="bg-white rounded-2xl shadow-lg p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600">Last Visit</p>
                <p className="text-2xl font-bold text-gray-900">5 days</p>
              </div>
              <Clock className="w-8 h-8 text-purple-600" />
            </div>
          </div>
        </motion.div>

        {/* Navigation Tabs */}
        <div className="mb-8">
          <div className="bg-white rounded-2xl shadow-lg p-2">
            <div className="flex space-x-1">
              {[
                { id: 'overview', label: 'Overview', icon: Activity },
                { id: 'appointments', label: 'Appointments', icon: Calendar },
                { id: 'records', label: 'Medical Records', icon: FileText },
                { id: 'payments', label: 'Payments', icon: CreditCard },
                { id: 'profile', label: 'Profile', icon: User }
              ].map((tab) => {
                const Icon = tab.icon;
                return (
                  <button
                    key={tab.id}
                    onClick={() => setActiveTab(tab.id)}
                    className={`flex items-center space-x-2 px-4 py-2 rounded-xl font-medium transition-all duration-200 ${
                      activeTab === tab.id
                        ? 'bg-blue-600 text-white'
                        : 'text-gray-600 hover:bg-gray-100'
                    }`}
                  >
                    <Icon className="w-4 h-4" />
                    <span>{tab.label}</span>
                  </button>
                );
              })}
            </div>
          </div>
        </div>

        {/* Tab Content */}
        <motion.div
          key={activeTab}
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.3 }}
        >
          {/* Overview Tab */}
          {activeTab === 'overview' && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              {/* Recent Appointments */}
              <div className="bg-white rounded-2xl shadow-lg p-6">
                <h3 className="text-xl font-bold text-gray-900 mb-4">Recent Appointments</h3>
                <div className="space-y-4">
                  {appointments.slice(0, 3).map((appointment) => (
                    <div key={appointment.id} className="flex items-center justify-between p-4 bg-gray-50 rounded-xl">
                      <div>
                        <h4 className="font-medium text-gray-900">{appointment.type}</h4>
                        <p className="text-sm text-gray-600">
                          {format(new Date(appointment.date), 'MMM dd, yyyy')} at {appointment.time}
                        </p>
                      </div>
                      <span className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(appointment.status)}`}>
                        {appointment.status}
                      </span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Health Insights */}
              <div className="bg-white rounded-2xl shadow-lg p-6">
                <h3 className="text-xl font-bold text-gray-900 mb-4">Health Insights</h3>
                <div className="space-y-4">
                  <div className="p-4 bg-green-50 rounded-xl border border-green-200">
                    <div className="flex items-center space-x-2 mb-2">
                      <CheckCircle className="w-5 h-5 text-green-600" />
                      <span className="font-medium text-green-800">Good Health Trend</span>
                    </div>
                    <p className="text-sm text-green-700">
                      Your recent lab results show improvement in key health markers.
                    </p>
                  </div>
                  <div className="p-4 bg-blue-50 rounded-xl border border-blue-200">
                    <div className="flex items-center space-x-2 mb-2">
                      <TrendingUp className="w-5 h-5 text-blue-600" />
                      <span className="font-medium text-blue-800">AI Recommendation</span>
                    </div>
                    <p className="text-sm text-blue-700">
                      Consider scheduling a routine check-up within the next month.
                    </p>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Appointments Tab */}
          {activeTab === 'appointments' && (
            <div className="bg-white rounded-2xl shadow-lg p-6">
              <div className="flex items-center justify-between mb-6">
                <h3 className="text-xl font-bold text-gray-900">My Appointments</h3>
                <button
                  onClick={() => window.location.href = '/book-appointment'}
                  className="bg-gradient-to-r from-blue-600 to-green-600 text-white px-4 py-2 rounded-lg font-semibold hover:shadow-lg transition-all duration-200 flex items-center space-x-2"
                >
                  <Plus className="w-4 h-4" />
                  <span>Book New</span>
                </button>
              </div>
              <div className="space-y-4">
                {appointments.map((appointment) => (
                  <div key={appointment.id} className="border border-gray-200 rounded-xl p-6">
                    <div className="flex items-center justify-between mb-4">
                      <div>
                        <h4 className="text-lg font-bold text-gray-900">{appointment.type}</h4>
                        <p className="text-gray-600">
                          {format(new Date(appointment.date), 'EEEE, MMMM dd, yyyy')} at {appointment.time}
                        </p>
                      </div>
                      <div className="text-right">
                        <span className={`px-3 py-1 rounded-full text-sm font-medium ${getStatusColor(appointment.status)}`}>
                          {appointment.status}
                        </span>
                        <p className={`text-sm mt-1 ${getPaymentStatusColor(appointment.paymentStatus)}`}>
                          ${appointment.cost} - {appointment.paymentStatus}
                        </p>
                      </div>
                    </div>
                    <div className="flex space-x-3">
                      <button className="flex items-center space-x-1 px-3 py-1 bg-blue-100 text-blue-700 rounded-lg hover:bg-blue-200 transition-all duration-200">
                        <Eye className="w-4 h-4" />
                        <span>View Details</span>
                      </button>
                      {appointment.status === 'scheduled' && (
                        <button className="flex items-center space-x-1 px-3 py-1 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 transition-all duration-200">
                          <Edit className="w-4 h-4" />
                          <span>Reschedule</span>
                        </button>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Medical Records Tab */}
          {activeTab === 'records' && (
            <div className="bg-white rounded-2xl shadow-lg p-6">
              <h3 className="text-xl font-bold text-gray-900 mb-6">Medical Records</h3>
              <div className="space-y-4">
                {medicalRecords.map((record) => (
                  <div key={record.id} className="border border-gray-200 rounded-xl p-6">
                    <div className="flex items-center justify-between mb-4">
                      <div>
                        <h4 className="text-lg font-bold text-gray-900">{record.title}</h4>
                        <p className="text-gray-600">{format(new Date(record.date), 'MMMM dd, yyyy')}</p>
                      </div>
                      <div className="flex items-center space-x-2">
                        {record.aiConfidence && (
                          <span className="px-2 py-1 bg-blue-100 text-blue-800 rounded-full text-xs font-medium">
                            AI: {Math.round(record.aiConfidence * 100)}%
                          </span>
                        )}
                        <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                          record.type === 'diagnosis' ? 'bg-purple-100 text-purple-800' :
                          record.type === 'lab_result' ? 'bg-green-100 text-green-800' :
                          record.type === 'imaging' ? 'bg-blue-100 text-blue-800' :
                          'bg-orange-100 text-orange-800'
                        }`}>
                          {record.type.replace('_', ' ')}
                        </span>
                      </div>
                    </div>
                    <p className="text-gray-700 mb-4">{record.summary}</p>
                    <div className="flex space-x-3">
                      <button className="flex items-center space-x-1 px-3 py-1 bg-blue-100 text-blue-700 rounded-lg hover:bg-blue-200 transition-all duration-200">
                        <Eye className="w-4 h-4" />
                        <span>View Full Report</span>
                      </button>
                      <button className="flex items-center space-x-1 px-3 py-1 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 transition-all duration-200">
                        <Download className="w-4 h-4" />
                        <span>Download PDF</span>
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Payments Tab */}
          {activeTab === 'payments' && (
            <div className="bg-white rounded-2xl shadow-lg p-6">
              <h3 className="text-xl font-bold text-gray-900 mb-6">Payment History</h3>
              <div className="space-y-4">
                {appointments.map((appointment) => (
                  <div key={appointment.id} className="flex items-center justify-between p-4 border border-gray-200 rounded-xl">
                    <div>
                      <h4 className="font-medium text-gray-900">{appointment.type}</h4>
                      <p className="text-sm text-gray-600">
                        {format(new Date(appointment.date), 'MMM dd, yyyy')}
                      </p>
                    </div>
                    <div className="text-right">
                      <p className="font-bold text-gray-900">${appointment.cost}</p>
                      <p className={`text-sm ${getPaymentStatusColor(appointment.paymentStatus)}`}>
                        {appointment.paymentStatus}
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Profile Tab */}
          {activeTab === 'profile' && (
            <div className="bg-white rounded-2xl shadow-lg p-6">
              <h3 className="text-xl font-bold text-gray-900 mb-6">Profile Information</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">First Name</label>
                  <input
                    type="text"
                    value={user?.firstName || ''}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                    readOnly
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Last Name</label>
                  <input
                    type="text"
                    value={user?.lastName || ''}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                    readOnly
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Email</label>
                  <input
                    type="email"
                    value={user?.email || ''}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                    readOnly
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Phone</label>
                  <input
                    type="tel"
                    value={user?.phone || ''}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                    readOnly
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Date of Birth</label>
                  <input
                    type="date"
                    value={user?.dateOfBirth || ''}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                    readOnly
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Insurance Provider</label>
                  <input
                    type="text"
                    value={user?.insuranceProvider || ''}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                    readOnly
                  />
                </div>
              </div>
              <div className="mt-6">
                <button className="bg-gradient-to-r from-blue-600 to-green-600 text-white px-6 py-2 rounded-lg font-semibold hover:shadow-lg transition-all duration-200">
                  Edit Profile
                </button>
              </div>
            </div>
          )}
        </motion.div>
      </div>
    </div>
  );
};