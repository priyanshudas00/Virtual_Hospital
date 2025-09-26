import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { useAuth } from '../contexts/AuthContext';
import {
  MapPin,
  Phone,
  Clock,
  Star,
  DollarSign,
  Navigation,
  Filter,
  Search,
  Heart,
  Stethoscope,
  Building,
  Award,
  Shield,
  Zap
} from 'lucide-react';

interface Doctor {
  _id: string;
  name: string;
  specialization: string;
  qualifications: string;
  experience: number;
  address: string;
  phone: string;
  consultation_fee: number;
  cost_category: string;
  rating: number;
  availability: string[];
  distance?: number;
}

interface Hospital {
  _id: string;
  name: string;
  type: string;
  specialties: string[];
  address: string;
  phone: string;
  emergency_services: boolean;
  cost_category: string;
  rating: number;
  bed_capacity: number;
  facilities: string[];
  distance?: number;
}

export const FindHealthcarePage: React.FC = () => {
  const { user } = useAuth();
  const [doctors, setDoctors] = useState<Doctor[]>([]);
  const [hospitals, setHospitals] = useState<Hospital[]>([]);
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState<'doctors' | 'hospitals'>('doctors');
  const [filters, setFilters] = useState({
    location: '',
    urgency: 'routine',
    financial_capability: 'medium',
    specialization: '',
    emergency_only: false
  });

  useEffect(() => {
    if (user?.location) {
      setFilters(prev => ({ ...prev, location: user.location }));
      searchHealthcare();
    }
  }, [user]);

  const searchHealthcare = async () => {
    setLoading(true);
    
    try {
      const response = await fetch(`${import.meta.env.VITE_API_URL}/api/find-healthcare`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        },
        body: JSON.stringify(filters)
      });

      const result = await response.json();

      if (response.ok) {
        setDoctors(result.doctors || []);
        setHospitals(result.hospitals || []);
      } else {
        console.error('Search failed:', result.error);
      }
    } catch (error) {
      console.error('Search error:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleFilterChange = (key: string, value: any) => {
    setFilters(prev => ({ ...prev, [key]: value }));
  };

  const getCostCategoryColor = (category: string) => {
    switch (category) {
      case 'low': return 'bg-green-100 text-green-800';
      case 'medium': return 'bg-blue-100 text-blue-800';
      case 'high': return 'bg-purple-100 text-purple-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  const getUrgencyColor = (urgency: string) => {
    switch (urgency) {
      case 'emergency': return 'bg-red-500';
      case 'urgent': return 'bg-orange-500';
      case 'routine': return 'bg-green-500';
      default: return 'bg-gray-500';
    }
  };

  return (
    <div className="min-h-screen py-8 px-4 sm:px-6 lg:px-8 bg-gray-50">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-8"
        >
          <div className="bg-gradient-to-r from-green-600 to-blue-600 w-16 h-16 rounded-2xl flex items-center justify-center mx-auto mb-4">
            <MapPin className="w-8 h-8 text-white" />
          </div>
          <h1 className="text-3xl font-bold text-gray-900 mb-2">Find Healthcare Providers</h1>
          <p className="text-gray-600">Discover nearby doctors and hospitals based on your needs</p>
        </motion.div>

        {/* Search Filters */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="bg-white rounded-2xl shadow-lg p-6 mb-8"
        >
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Location</label>
              <input
                type="text"
                value={filters.location}
                onChange={(e) => handleFilterChange('location', e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                placeholder="Enter city or area"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Urgency</label>
              <select
                value={filters.urgency}
                onChange={(e) => handleFilterChange('urgency', e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              >
                <option value="routine">Routine</option>
                <option value="urgent">Urgent</option>
                <option value="emergency">Emergency</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Budget</label>
              <select
                value={filters.financial_capability}
                onChange={(e) => handleFilterChange('financial_capability', e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              >
                <option value="low">Budget-Friendly</option>
                <option value="medium">Moderate</option>
                <option value="high">Premium</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Specialization</label>
              <select
                value={filters.specialization}
                onChange={(e) => handleFilterChange('specialization', e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              >
                <option value="">All Specializations</option>
                <option value="cardiology">Cardiology</option>
                <option value="neurology">Neurology</option>
                <option value="orthopedics">Orthopedics</option>
                <option value="dermatology">Dermatology</option>
                <option value="pediatrics">Pediatrics</option>
                <option value="psychiatry">Psychiatry</option>
                <option value="general_medicine">General Medicine</option>
              </select>
            </div>
          </div>

          <div className="flex items-center justify-between">
            <label className="flex items-center space-x-2">
              <input
                type="checkbox"
                checked={filters.emergency_only}
                onChange={(e) => handleFilterChange('emergency_only', e.target.checked)}
                className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
              />
              <span className="text-sm text-gray-700">Emergency services only</span>
            </label>

            <button
              onClick={searchHealthcare}
              disabled={loading}
              className="bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 flex items-center space-x-2"
            >
              {loading ? (
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
              ) : (
                <Search className="w-4 h-4" />
              )}
              <span>Search</span>
            </button>
          </div>
        </motion.div>

        {/* Tabs */}
        <div className="mb-6">
          <div className="bg-white rounded-2xl shadow-lg p-2">
            <div className="flex space-x-1">
              <button
                onClick={() => setActiveTab('doctors')}
                className={`flex items-center space-x-2 px-4 py-2 rounded-xl font-medium transition-all duration-200 ${
                  activeTab === 'doctors'
                    ? 'bg-blue-600 text-white'
                    : 'text-gray-600 hover:bg-gray-100'
                }`}
              >
                <Stethoscope className="w-4 h-4" />
                <span>Doctors ({doctors.length})</span>
              </button>
              <button
                onClick={() => setActiveTab('hospitals')}
                className={`flex items-center space-x-2 px-4 py-2 rounded-xl font-medium transition-all duration-200 ${
                  activeTab === 'hospitals'
                    ? 'bg-blue-600 text-white'
                    : 'text-gray-600 hover:bg-gray-100'
                }`}
              >
                <Building className="w-4 h-4" />
                <span>Hospitals ({hospitals.length})</span>
              </button>
            </div>
          </div>
        </div>

        {/* Results */}
        <motion.div
          key={activeTab}
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.3 }}
        >
          {/* Doctors Tab */}
          {activeTab === 'doctors' && (
            <div className="space-y-6">
              {doctors.length === 0 ? (
                <div className="bg-white rounded-2xl shadow-lg p-12 text-center">
                  <Stethoscope className="w-16 h-16 text-gray-300 mx-auto mb-4" />
                  <h3 className="text-xl font-bold text-gray-900 mb-2">No doctors found</h3>
                  <p className="text-gray-600">Try adjusting your search filters or location</p>
                </div>
              ) : (
                doctors.map((doctor) => (
                  <div key={doctor._id} className="bg-white rounded-2xl shadow-lg p-6 hover:shadow-xl transition-all duration-300">
                    <div className="flex items-start justify-between mb-4">
                      <div className="flex items-start space-x-4">
                        <div className="bg-blue-100 p-3 rounded-xl">
                          <Stethoscope className="w-8 h-8 text-blue-600" />
                        </div>
                        <div>
                          <h3 className="text-xl font-bold text-gray-900 mb-1">Dr. {doctor.name}</h3>
                          <p className="text-blue-600 font-medium mb-2">{doctor.specialization}</p>
                          <p className="text-gray-600 text-sm mb-2">{doctor.qualifications}</p>
                          <div className="flex items-center space-x-4 text-sm text-gray-600">
                            <div className="flex items-center space-x-1">
                              <Award className="w-4 h-4" />
                              <span>{doctor.experience} years exp.</span>
                            </div>
                            <div className="flex items-center space-x-1">
                              <Star className="w-4 h-4 text-yellow-500" />
                              <span>{doctor.rating}/5</span>
                            </div>
                            {doctor.distance && (
                              <div className="flex items-center space-x-1">
                                <MapPin className="w-4 h-4" />
                                <span>{doctor.distance.toFixed(1)} km away</span>
                              </div>
                            )}
                          </div>
                        </div>
                      </div>
                      <div className="text-right">
                        <div className="flex items-center space-x-2 mb-2">
                          <span className={`px-2 py-1 rounded-full text-xs font-medium ${getCostCategoryColor(doctor.cost_category)}`}>
                            {doctor.cost_category} cost
                          </span>
                          <div className={`w-3 h-3 rounded-full ${getUrgencyColor(filters.urgency)}`}></div>
                        </div>
                        <p className="text-2xl font-bold text-green-600">₹{doctor.consultation_fee}</p>
                        <p className="text-sm text-gray-600">Consultation</p>
                      </div>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
                      <div className="flex items-center space-x-2 text-gray-600">
                        <MapPin className="w-4 h-4" />
                        <span className="text-sm">{doctor.address}</span>
                      </div>
                      <div className="flex items-center space-x-2 text-gray-600">
                        <Phone className="w-4 h-4" />
                        <span className="text-sm">{doctor.phone}</span>
                      </div>
                    </div>

                    <div className="mb-4">
                      <h4 className="font-medium text-gray-900 mb-2">Available Times</h4>
                      <div className="flex flex-wrap gap-2">
                        {doctor.availability.map((time, index) => (
                          <span key={index} className="px-3 py-1 bg-green-100 text-green-800 rounded-full text-sm">
                            {time}
                          </span>
                        ))}
                      </div>
                    </div>

                    <div className="flex space-x-3">
                      <button className="flex-1 bg-blue-600 text-white py-2 px-4 rounded-lg hover:bg-blue-700 transition-all duration-200 flex items-center justify-center space-x-2">
                        <Phone className="w-4 h-4" />
                        <span>Call Now</span>
                      </button>
                      <button className="flex-1 bg-green-600 text-white py-2 px-4 rounded-lg hover:bg-green-700 transition-all duration-200 flex items-center justify-center space-x-2">
                        <Navigation className="w-4 h-4" />
                        <span>Get Directions</span>
                      </button>
                      <button className="px-4 py-2 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 transition-all duration-200">
                        <Heart className="w-4 h-4" />
                      </button>
                    </div>
                  </div>
                ))
              )}
            </div>
          )}

          {/* Hospitals Tab */}
          {activeTab === 'hospitals' && (
            <div className="space-y-6">
              {hospitals.length === 0 ? (
                <div className="bg-white rounded-2xl shadow-lg p-12 text-center">
                  <Building className="w-16 h-16 text-gray-300 mx-auto mb-4" />
                  <h3 className="text-xl font-bold text-gray-900 mb-2">No hospitals found</h3>
                  <p className="text-gray-600">Try adjusting your search filters or location</p>
                </div>
              ) : (
                hospitals.map((hospital) => (
                  <div key={hospital._id} className="bg-white rounded-2xl shadow-lg p-6 hover:shadow-xl transition-all duration-300">
                    <div className="flex items-start justify-between mb-4">
                      <div className="flex items-start space-x-4">
                        <div className="bg-red-100 p-3 rounded-xl">
                          <Building className="w-8 h-8 text-red-600" />
                        </div>
                        <div>
                          <h3 className="text-xl font-bold text-gray-900 mb-1">{hospital.name}</h3>
                          <p className="text-red-600 font-medium mb-2">{hospital.type} Hospital</p>
                          <div className="flex items-center space-x-4 text-sm text-gray-600 mb-2">
                            <div className="flex items-center space-x-1">
                              <Star className="w-4 h-4 text-yellow-500" />
                              <span>{hospital.rating}/5</span>
                            </div>
                            <div className="flex items-center space-x-1">
                              <Shield className="w-4 h-4" />
                              <span>{hospital.bed_capacity} beds</span>
                            </div>
                            {hospital.distance && (
                              <div className="flex items-center space-x-1">
                                <MapPin className="w-4 h-4" />
                                <span>{hospital.distance.toFixed(1)} km away</span>
                              </div>
                            )}
                          </div>
                          <div className="flex flex-wrap gap-2">
                            {hospital.specialties.slice(0, 3).map((specialty, index) => (
                              <span key={index} className="px-2 py-1 bg-blue-100 text-blue-800 rounded-full text-xs">
                                {specialty}
                              </span>
                            ))}
                            {hospital.specialties.length > 3 && (
                              <span className="px-2 py-1 bg-gray-100 text-gray-600 rounded-full text-xs">
                                +{hospital.specialties.length - 3} more
                              </span>
                            )}
                          </div>
                        </div>
                      </div>
                      <div className="text-right">
                        <div className="flex items-center space-x-2 mb-2">
                          <span className={`px-2 py-1 rounded-full text-xs font-medium ${getCostCategoryColor(hospital.cost_category)}`}>
                            {hospital.cost_category} cost
                          </span>
                          {hospital.emergency_services && (
                            <span className="px-2 py-1 bg-red-100 text-red-800 rounded-full text-xs font-medium flex items-center space-x-1">
                              <Zap className="w-3 h-3" />
                              <span>Emergency</span>
                            </span>
                          )}
                        </div>
                      </div>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
                      <div className="flex items-center space-x-2 text-gray-600">
                        <MapPin className="w-4 h-4" />
                        <span className="text-sm">{hospital.address}</span>
                      </div>
                      <div className="flex items-center space-x-2 text-gray-600">
                        <Phone className="w-4 h-4" />
                        <span className="text-sm">{hospital.phone}</span>
                      </div>
                    </div>

                    <div className="mb-4">
                      <h4 className="font-medium text-gray-900 mb-2">Facilities</h4>
                      <div className="flex flex-wrap gap-2">
                        {hospital.facilities.slice(0, 4).map((facility, index) => (
                          <span key={index} className="px-3 py-1 bg-green-100 text-green-800 rounded-full text-sm">
                            {facility}
                          </span>
                        ))}
                        {hospital.facilities.length > 4 && (
                          <span className="px-3 py-1 bg-gray-100 text-gray-600 rounded-full text-sm">
                            +{hospital.facilities.length - 4} more
                          </span>
                        )}
                      </div>
                    </div>

                    <div className="flex space-x-3">
                      <button className="flex-1 bg-red-600 text-white py-2 px-4 rounded-lg hover:bg-red-700 transition-all duration-200 flex items-center justify-center space-x-2">
                        <Phone className="w-4 h-4" />
                        <span>Call Hospital</span>
                      </button>
                      <button className="flex-1 bg-green-600 text-white py-2 px-4 rounded-lg hover:bg-green-700 transition-all duration-200 flex items-center justify-center space-x-2">
                        <Navigation className="w-4 h-4" />
                        <span>Get Directions</span>
                      </button>
                      <button className="px-4 py-2 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 transition-all duration-200">
                        <Heart className="w-4 h-4" />
                      </button>
                    </div>
                  </div>
                ))
              )}
            </div>
          )}
        </motion.div>
      </div>
    </div>
  );
};