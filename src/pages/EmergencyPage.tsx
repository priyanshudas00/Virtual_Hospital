import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import {
  AlertTriangle,
  Phone,
  MapPin,
  Clock,
  Heart,
  Brain,
  Activity,
  Zap,
  Shield,
  Navigation
} from 'lucide-react';

export const EmergencyPage: React.FC = () => {
  const [emergencyStatus, setEmergencyStatus] = useState<string | null>(null);
  const [currentTime, setCurrentTime] = useState(new Date());
  const [userLocation, setUserLocation] = useState<{lat: number, lon: number} | null>(null);

  useEffect(() => {
    const timer = setInterval(() => setCurrentTime(new Date()), 1000);
    
    // Get user location
    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        (position) => {
          setUserLocation({
            lat: position.coords.latitude,
            lon: position.coords.longitude
          });
        },
        (error) => {
          console.error('Location access denied:', error);
        }
      );
    }

    return () => clearInterval(timer);
  }, []);

  const emergencySymptoms = [
    {
      category: 'Cardiac Emergency',
      icon: Heart,
      color: 'bg-red-500',
      symptoms: [
        'Severe chest pain or pressure',
        'Difficulty breathing with chest pain',
        'Pain radiating to arm, jaw, or back',
        'Sudden collapse or unconsciousness'
      ],
      action: 'CALL 911 IMMEDIATELY'
    },
    {
      category: 'Stroke/Brain Emergency',
      icon: Brain,
      color: 'bg-purple-500',
      symptoms: [
        'Sudden severe headache',
        'Face drooping or numbness',
        'Arm weakness or numbness',
        'Speech difficulty or confusion'
      ],
      action: 'CALL 911 - TIME IS BRAIN'
    },
    {
      category: 'Severe Trauma',
      icon: AlertTriangle,
      color: 'bg-orange-500',
      symptoms: [
        'Severe bleeding that won\'t stop',
        'Suspected broken bones',
        'Head injury with confusion',
        'Severe burns'
      ],
      action: 'CALL 911 IMMEDIATELY'
    },
    {
      category: 'Breathing Emergency',
      icon: Activity,
      color: 'bg-blue-500',
      symptoms: [
        'Severe difficulty breathing',
        'Choking',
        'Severe allergic reaction',
        'Asthma attack not responding to inhaler'
      ],
      action: 'CALL 911 NOW'
    }
  ];

  const nearbyHospitals = [
    {
      name: 'City General Hospital',
      distance: '2.1 miles',
      estimatedTime: '8 minutes',
      emergencyCapable: true,
      phone: '(555) 123-4567'
    },
    {
      name: 'St. Mary\'s Medical Center',
      distance: '3.7 miles',
      estimatedTime: '12 minutes',
      emergencyCapable: true,
      phone: '(555) 987-6543'
    },
    {
      name: 'Regional Emergency Clinic',
      distance: '1.5 miles',
      estimatedTime: '5 minutes',
      emergencyCapable: false,
      phone: '(555) 456-7890'
    }
  ];

  const handleEmergencyCall = () => {
    if (window.confirm('This will attempt to call emergency services. Continue?')) {
      // In a real app, this would trigger the call
      window.open('tel:911', '_self');
    }
  };

  const assessEmergency = (symptoms: string) => {
    // Simple AI-based emergency assessment
    const highRiskKeywords = ['chest pain', 'heart attack', 'stroke', 'unconscious', 'severe bleeding', 'choking'];
    const mediumRiskKeywords = ['headache', 'nausea', 'dizziness', 'fever', 'pain'];
    
    const lowerSymptoms = symptoms.toLowerCase();
    
    if (highRiskKeywords.some(keyword => lowerSymptoms.includes(keyword))) {
      return 'CRITICAL - CALL 911 IMMEDIATELY';
    } else if (mediumRiskKeywords.some(keyword => lowerSymptoms.includes(keyword))) {
      return 'URGENT - Seek medical attention';
    } else {
      return 'Monitor symptoms - Consider telehealth consultation';
    }
  };

  return (
    <div className="min-h-screen py-8 px-4 sm:px-6 lg:px-8 bg-red-50">
      <div className="max-w-6xl mx-auto">
        {/* Emergency Header */}
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          className="text-center mb-8"
        >
          <div className="bg-red-600 w-20 h-20 rounded-full flex items-center justify-center mx-auto mb-4 animate-pulse">
            <AlertTriangle className="w-10 h-10 text-white" />
          </div>
          <h1 className="text-4xl font-bold text-red-900 mb-2">Emergency Care</h1>
          <p className="text-red-700 text-lg">Get immediate AI-powered emergency assistance</p>
          <div className="flex items-center justify-center space-x-2 mt-4 text-red-800">
            <Clock className="w-5 h-5" />
            <span className="font-mono text-lg">{currentTime.toLocaleTimeString()}</span>
          </div>
        </motion.div>

        {/* Emergency Call Button */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-8"
        >
          <button
            onClick={handleEmergencyCall}
            className="bg-red-600 hover:bg-red-700 text-white text-2xl font-bold py-6 px-12 rounded-2xl shadow-2xl transform hover:scale-105 transition-all duration-200 flex items-center space-x-4 mx-auto"
          >
            <Phone className="w-8 h-8" />
            <span>CALL 911</span>
          </button>
          <p className="text-red-700 mt-2">For life-threatening emergencies</p>
        </motion.div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Emergency Symptoms Checker */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            className="lg:col-span-2"
          >
            <div className="bg-white rounded-2xl shadow-xl p-8 mb-8">
              <h2 className="text-2xl font-bold text-gray-900 mb-6 flex items-center space-x-2">
                <Zap className="w-6 h-6 text-yellow-600" />
                <span>Emergency Assessment</span>
              </h2>
              
              <div className="space-y-4 mb-6">
                <textarea
                  placeholder="Describe the emergency symptoms quickly..."
                  className="w-full h-32 p-4 border-2 border-gray-300 rounded-xl focus:border-red-500 focus:ring-2 focus:ring-red-200"
                  onChange={(e) => {
                    if (e.target.value) {
                      const assessment = assessEmergency(e.target.value);
                      setEmergencyStatus(assessment);
                    }
                  }}
                />
                
                {emergencyStatus && (
                  <div className={`p-4 rounded-xl border-2 ${
                    emergencyStatus.includes('CRITICAL') 
                      ? 'bg-red-50 border-red-500 text-red-900' 
                      : emergencyStatus.includes('URGENT')
                      ? 'bg-yellow-50 border-yellow-500 text-yellow-900'
                      : 'bg-blue-50 border-blue-500 text-blue-900'
                  }`}>
                    <div className="flex items-center space-x-2">
                      <AlertTriangle className="w-5 h-5" />
                      <span className="font-bold">AI Assessment: {emergencyStatus}</span>
                    </div>
                  </div>
                )}
              </div>
              
              {/* Emergency Categories */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {emergencySymptoms.map((emergency, index) => {
                  const Icon = emergency.icon;
                  return (
                    <div key={index} className="border border-gray-200 rounded-xl p-4 hover:shadow-lg transition-all duration-200">
                      <div className="flex items-center space-x-3 mb-3">
                        <div className={`${emergency.color} p-2 rounded-lg`}>
                          <Icon className="w-5 h-5 text-white" />
                        </div>
                        <h3 className="font-bold text-gray-900">{emergency.category}</h3>
                      </div>
                      <ul className="space-y-1 text-sm text-gray-600 mb-3">
                        {emergency.symptoms.map((symptom, idx) => (
                          <li key={idx} className="flex items-start space-x-1">
                            <span className="text-red-500 mt-1">•</span>
                            <span>{symptom}</span>
                          </li>
                        ))}
                      </ul>
                      <div className="bg-red-100 text-red-800 text-xs font-bold py-1 px-2 rounded text-center">
                        {emergency.action}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </motion.div>

          {/* Location & Hospitals */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            className="space-y-6"
          >
            {/* Location */}
            <div className="bg-white rounded-2xl shadow-xl p-6">
              <h3 className="text-xl font-bold text-gray-900 mb-4 flex items-center space-x-2">
                <MapPin className="w-5 h-5 text-blue-600" />
                <span>Your Location</span>
              </h3>
              {userLocation ? (
                <div className="space-y-2">
                  <p className="text-sm text-gray-600">Lat: {userLocation.lat.toFixed(4)}</p>
                  <p className="text-sm text-gray-600">Lon: {userLocation.lon.toFixed(4)}</p>
                  <button className="w-full bg-blue-600 text-white py-2 px-4 rounded-lg hover:bg-blue-700 transition-all duration-200 flex items-center justify-center space-x-2">
                    <Navigation className="w-4 h-4" />
                    <span>Share Location with 911</span>
                  </button>
                </div>
              ) : (
                <p className="text-gray-600">Location access required for emergency services</p>
              )}
            </div>

            {/* Nearby Hospitals */}
            <div className="bg-white rounded-2xl shadow-xl p-6">
              <h3 className="text-xl font-bold text-gray-900 mb-4 flex items-center space-x-2">
                <Shield className="w-5 h-5 text-green-600" />
                <span>Nearby Hospitals</span>
              </h3>
              <div className="space-y-3">
                {nearbyHospitals.map((hospital, index) => (
                  <div key={index} className="border border-gray-200 rounded-lg p-4">
                    <div className="flex justify-between items-start mb-2">
                      <h4 className="font-semibold text-gray-900">{hospital.name}</h4>
                      {hospital.emergencyCapable && (
                        <span className="bg-green-100 text-green-800 text-xs px-2 py-1 rounded-full">
                          Emergency
                        </span>
                      )}
                    </div>
                    <div className="text-sm text-gray-600 space-y-1">
                      <div className="flex items-center space-x-1">
                        <MapPin className="w-3 h-3" />
                        <span>{hospital.distance}</span>
                      </div>
                      <div className="flex items-center space-x-1">
                        <Clock className="w-3 h-3" />
                        <span>~{hospital.estimatedTime}</span>
                      </div>
                      <div className="flex items-center space-x-1">
                        <Phone className="w-3 h-3" />
                        <span>{hospital.phone}</span>
                      </div>
                    </div>
                    <button className="w-full mt-2 bg-gray-100 hover:bg-gray-200 text-gray-800 py-2 px-3 rounded text-sm font-medium transition-all duration-200">
                      Get Directions
                    </button>
                  </div>
                ))}
              </div>
            </div>

            {/* Emergency Contacts */}
            <div className="bg-white rounded-2xl shadow-xl p-6">
              <h3 className="text-xl font-bold text-gray-900 mb-4">Emergency Contacts</h3>
              <div className="space-y-3">
                <button className="w-full bg-red-600 text-white py-3 px-4 rounded-lg font-semibold hover:bg-red-700 transition-all duration-200 flex items-center justify-center space-x-2">
                  <Phone className="w-4 h-4" />
                  <span>911 - Emergency</span>
                </button>
                <button className="w-full bg-blue-600 text-white py-3 px-4 rounded-lg font-semibold hover:bg-blue-700 transition-all duration-200 flex items-center justify-center space-x-2">
                  <Phone className="w-4 h-4" />
                  <span>Poison Control: (800) 222-1222</span>
                </button>
                <button className="w-full bg-purple-600 text-white py-3 px-4 rounded-lg font-semibold hover:bg-purple-700 transition-all duration-200 flex items-center justify-center space-x-2">
                  <Phone className="w-4 h-4" />
                  <span>Mental Health Crisis: 988</span>
                </button>
              </div>
            </div>
          </motion.div>
        </div>

        {/* Disclaimer */}
        <div className="mt-8 p-6 bg-red-100 border border-red-300 rounded-xl">
          <div className="flex items-start space-x-3">
            <AlertTriangle className="w-6 h-6 text-red-600 mt-1" />
            <div>
              <h4 className="font-bold text-red-900 mb-2">EMERGENCY DISCLAIMER</h4>
              <p className="text-red-800 text-sm">
                This AI emergency assistant is NOT a substitute for professional emergency medical services. 
                For life-threatening emergencies, ALWAYS call 911 immediately. This system provides informational 
                guidance only and should not delay seeking appropriate emergency medical care.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};