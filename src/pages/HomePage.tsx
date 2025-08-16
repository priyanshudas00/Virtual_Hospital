import React from 'react';
import { motion } from 'framer-motion';
import { Link } from 'react-router-dom';
import {
  Brain,
  Heart,
  Activity,
  Stethoscope,
  AlertTriangle,
  Shield,
  Zap,
  Clock,
  Users,
  Award
} from 'lucide-react';

export const HomePage: React.FC = () => {
  const features = [
    {
      icon: Brain,
      title: 'AI Diagnosis Engine',
      description: 'Advanced machine learning models analyze symptoms and provide accurate diagnostic predictions.',
      color: 'bg-blue-500'
    },
    {
      icon: Activity,
      title: 'Medical Imaging AI',
      description: 'Deep learning algorithms interpret X-rays, MRIs, and CT scans with radiologist-level accuracy.',
      color: 'bg-purple-500'
    },
    {
      icon: Heart,
      title: 'Smart Treatment Plans',
      description: 'Personalized treatment recommendations based on patient history and evidence-based medicine.',
      color: 'bg-red-500'
    },
    {
      icon: Stethoscope,
      title: 'Lab Result Analysis',
      description: 'Automated interpretation of blood tests, urine analysis, and biomarker assessments.',
      color: 'bg-green-500'
    },
    {
      icon: AlertTriangle,
      title: 'Emergency Detection',
      description: 'Real-time critical condition identification with instant alerts and intervention protocols.',
      color: 'bg-orange-500'
    },
    {
      icon: Shield,
      title: 'Secure & Compliant',
      description: 'HIPAA-compliant platform with end-to-end encryption and privacy protection.',
      color: 'bg-indigo-500'
    }
  ];

  const stats = [
    { label: 'Diagnostic Accuracy', value: '94.7%', icon: Award },
    { label: 'Response Time', value: '<2 sec', icon: Clock },
    { label: 'Patients Served', value: '10K+', icon: Users },
    { label: 'Uptime', value: '99.9%', icon: Zap }
  ];

  return (
    <div className="overflow-hidden">
      {/* Hero Section */}
      <section className="relative px-4 pt-16 pb-24 sm:px-6 lg:px-8">
        <div className="max-w-7xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="text-center"
          >
            <h1 className="text-4xl sm:text-6xl font-bold mb-6">
              <span className="bg-gradient-to-r from-blue-600 via-purple-600 to-green-600 bg-clip-text text-transparent">
                Revolutionary AI
              </span>
              <br />
              <span className="text-gray-900">Virtual Hospital</span>
            </h1>
            <p className="text-xl text-gray-600 mb-8 max-w-3xl mx-auto leading-relaxed">
              Experience the future of healthcare with our advanced AI platform. Get instant diagnosis, 
              medical imaging analysis, lab result interpretation, and personalized treatment plans - 
              all powered by cutting-edge artificial intelligence.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Link
                to="/diagnosis"
                className="bg-gradient-to-r from-blue-600 to-purple-600 text-white px-8 py-4 rounded-xl font-semibold text-lg hover:shadow-lg transform hover:-translate-y-1 transition-all duration-200"
              >
                Start Diagnosis
              </Link>
              <Link
                to="/emergency"
                className="bg-gradient-to-r from-red-500 to-orange-500 text-white px-8 py-4 rounded-xl font-semibold text-lg hover:shadow-lg transform hover:-translate-y-1 transition-all duration-200"
              >
                Emergency Care
              </Link>
            </div>
          </motion.div>

          {/* Stats */}
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.2 }}
            className="mt-20 grid grid-cols-2 lg:grid-cols-4 gap-8"
          >
            {stats.map((stat, index) => {
              const Icon = stat.icon;
              return (
                <div key={index} className="text-center">
                  <div className="bg-white rounded-2xl p-6 shadow-lg hover:shadow-xl transition-all duration-300">
                    <Icon className="w-8 h-8 text-blue-600 mx-auto mb-2" />
                    <div className="text-3xl font-bold text-gray-900 mb-1">{stat.value}</div>
                    <div className="text-sm text-gray-600">{stat.label}</div>
                  </div>
                </div>
              );
            })}
          </motion.div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-20 bg-white/50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="text-center mb-16"
          >
            <h2 className="text-3xl sm:text-4xl font-bold text-gray-900 mb-4">
              Advanced AI Medical Technologies
            </h2>
            <p className="text-xl text-gray-600 max-w-3xl mx-auto">
              Our platform integrates multiple AI technologies to provide comprehensive healthcare solutions
            </p>
          </motion.div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            {features.map((feature, index) => {
              const Icon = feature.icon;
              return (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5, delay: index * 0.1 }}
                  className="bg-white rounded-2xl p-8 shadow-lg hover:shadow-xl transition-all duration-300 border border-gray-100"
                >
                  <div className={`${feature.color} w-12 h-12 rounded-xl flex items-center justify-center mb-6`}>
                    <Icon className="w-6 h-6 text-white" />
                  </div>
                  <h3 className="text-xl font-bold text-gray-900 mb-3">{feature.title}</h3>
                  <p className="text-gray-600 leading-relaxed">{feature.description}</p>
                </motion.div>
              );
            })}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 bg-gradient-to-r from-blue-600 to-purple-600">
        <div className="max-w-4xl mx-auto text-center px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
          >
            <h2 className="text-3xl sm:text-4xl font-bold text-white mb-6">
              Ready to Experience AI Healthcare?
            </h2>
            <p className="text-xl text-blue-100 mb-8">
              Join thousands of patients who trust our AI-powered medical platform for accurate, 
              fast, and reliable healthcare solutions.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Link
                to="/diagnosis"
                className="bg-white text-blue-600 px-8 py-4 rounded-xl font-semibold text-lg hover:bg-blue-50 transition-all duration-200"
              >
                Get Started Now
              </Link>
              <Link
                to="/dashboard"
                className="border-2 border-white text-white px-8 py-4 rounded-xl font-semibold text-lg hover:bg-white hover:text-blue-600 transition-all duration-200"
              >
                View Dashboard
              </Link>
            </div>
          </motion.div>
        </div>
      </section>
    </div>
  );
};