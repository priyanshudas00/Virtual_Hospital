import React, { createContext, useContext, useState, useEffect } from 'react';
import { apiClient } from '../lib/api';

interface User {
  id: string;
  email: string;
  firstName: string;
  lastName: string;
  role: 'patient' | 'doctor' | 'admin';
  phone?: string;
  dateOfBirth?: string;
  profile?: any;
}

interface AuthContextType {
  user: User | null;
  loading: boolean;
  signUp: (email: string, password: string, userData: any) => Promise<void>;
  signIn: (email: string, password: string) => Promise<void>;
  signOut: () => Promise<void>;
  updateProfile: (profileData: any) => Promise<void>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Check for existing token
    const token = localStorage.getItem('token');
    if (token) {
      // Validate token and get user data
      validateToken();
    } else {
      setLoading(false);
    }
  }, []);

  const validateToken = async () => {
    try {
      const response = await fetch(`${import.meta.env.VITE_API_URL || 'http://localhost:3002'}/api/auth/profile`, {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        }
      });

      if (response.ok) {
        const data = await response.json();
        setUser({
          id: data.user._id,
          email: data.user.email,
          firstName: data.user.profile?.firstName || '',
          lastName: data.user.profile?.lastName || '',
          role: data.user.profile?.role || 'patient',
          phone: data.user.profile?.phone || '',
          dateOfBirth: data.user.profile?.dateOfBirth || '',
          profile: data.user.profile
        });
      } else {
        localStorage.removeItem('token');
      }
    } catch (error) {
      console.error('Token validation failed:', error);
      localStorage.removeItem('token');
    } finally {
      setLoading(false);
    }
  };

  const signUp = async (email: string, password: string, userData: any) => {
    try {
      const data = await apiClient.register({
        email,
        password,
        ...userData
      });

      localStorage.setItem('token', data.token);
      setUser({
        id: data.user._id,
        email,
        firstName: data.user.profile?.firstName || '',
        lastName: data.user.profile?.lastName || '',
        role: data.user.profile?.role || 'patient',
        phone: data.user.profile?.phone || '',
        dateOfBirth: data.user.profile?.dateOfBirth || '',
        profile: data.user.profile
      });
    } catch (error) {
      throw error;
    }
  };

  const signIn = async (email: string, password: string) => {
    try {
      const data = await apiClient.login({ email, password });

      localStorage.setItem('token', data.token);
      setUser({
        id: data.user._id,
        email,
        firstName: data.user.profile?.firstName || '',
        lastName: data.user.profile?.lastName || '',
        role: data.user.profile?.role || 'patient',
        phone: data.user.profile?.phone || '',
        dateOfBirth: data.user.profile?.dateOfBirth || '',
        profile: data.user.profile
      });
    } catch (error) {
      throw error;
    }
  };

  const signOut = async () => {
    localStorage.removeItem('token');
    setUser(null);
  };

  const updateProfile = async (profileData: any) => {
    try {
      const response = await fetch(`${import.meta.env.VITE_API_URL || 'http://localhost:3002'}/api/auth/profile`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        },
        body: JSON.stringify(profileData)
      });

      if (response.ok) {
        const data = await response.json();
        setUser(prev => prev ? { ...prev, ...data.user } : null);
      }
    } catch (error) {
      console.error('Profile update failed:', error);
      throw error;
    }
  };

  const value = {
    user,
    loading,
    signUp,
    signIn,
    signOut,
    updateProfile,
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
};