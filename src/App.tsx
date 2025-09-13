import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { AuthProvider, useAuth } from './contexts/AuthContext';
import { Header } from './components/Header';
import { HomePage } from './pages/HomePage';
import { DiagnosisPage } from './pages/DiagnosisPage';
import { ImagingPage } from './pages/ImagingPage';
import { LabResultsPage } from './pages/LabResultsPage';
import { TreatmentPage } from './pages/TreatmentPage';
import { EmergencyPage } from './pages/EmergencyPage';
import { IntakeFormPage } from './pages/IntakeFormPage';
import { UploadReportsPage } from './pages/UploadReportsPage';
import { FindHealthcarePage } from './pages/FindHealthcarePage';
import { DashboardPage } from './pages/DashboardPage';
import { AuthPage } from './pages/AuthPage';
import { AppointmentBookingPage } from './pages/AppointmentBookingPage';
import { PatientDashboardPage } from './pages/PatientDashboardPage';

const ProtectedRoute: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const { user, loading } = useAuth();
  
  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
      </div>
    );
  }
  
  if (!user) {
    return <AuthPage />;
  }
  
  return <>{children}</>;
};

const AppContent: React.FC = () => {
  const { user } = useAuth();

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-green-50">
      {user && <Header />}
      <Routes>
        <Route path="/auth" element={<AuthPage />} />
        <Route path="/" element={user ? <PatientDashboardPage /> : <HomePage />} />
        <Route path="/home" element={<HomePage />} />
        <Route path="/diagnosis" element={
          <ProtectedRoute>
            <DiagnosisPage />
          </ProtectedRoute>
        } />
        <Route path="/imaging" element={
          <ProtectedRoute>
            <ImagingPage />
          </ProtectedRoute>
        } />
        <Route path="/lab-results" element={
          <ProtectedRoute>
            <LabResultsPage />
          </ProtectedRoute>
        } />
        <Route path="/treatment" element={
          <ProtectedRoute>
            <TreatmentPage />
          </ProtectedRoute>
        } />
        <Route path="/emergency" element={<EmergencyPage />} />
        <Route path="/intake-form" element={
          <ProtectedRoute>
            <IntakeFormPage />
          </ProtectedRoute>
        } />
        <Route path="/upload-reports" element={
          <ProtectedRoute>
            <UploadReportsPage />
          </ProtectedRoute>
        } />
        <Route path="/find-healthcare" element={
          <ProtectedRoute>
            <FindHealthcarePage />
          </ProtectedRoute>
        } />
        <Route path="/dashboard" element={
          <ProtectedRoute>
            <PatientDashboardPage />
          </ProtectedRoute>
        } />
        <Route path="/admin-dashboard" element={
          <ProtectedRoute>
            <DashboardPage />
          </ProtectedRoute>
        } />
        <Route path="/book-appointment" element={
          <ProtectedRoute>
            <AppointmentBookingPage />
          </ProtectedRoute>
        } />
      </Routes>
    </div>
  );
};

function App() {
  return (
    <AuthProvider>
      <Router>
        <AppContent />
      </Router>
    </AuthProvider>
  );
}

export default App;