import { createClient } from '@supabase/supabase-js';

const supabaseUrl = import.meta.env.VITE_SUPABASE_URL || 'https://fapcuzzofbudjlledtwa.supabase.co';
const supabaseAnonKey = import.meta.env.VITE_SUPABASE_ANON_KEY || 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImZhcGN1enpvZmJ1ZGpsbGVkdHdhIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MzU0Mzk5NzQsImV4cCI6MjA1MTAxNTk3NH0.Ej8Ej8Ej8Ej8Ej8Ej8Ej8Ej8Ej8Ej8Ej8Ej8Ej8';

// Validate environment variables
if (!supabaseUrl || supabaseUrl === 'https://your-project.supabase.co') {
  console.error('VITE_SUPABASE_URL is not configured. Please set it in your .env file.');
}

if (!supabaseAnonKey || supabaseAnonKey === 'your-anon-key') {
  console.error('VITE_SUPABASE_ANON_KEY is not configured. Please set it in your .env file.');
}

export const supabase = createClient(supabaseUrl, supabaseAnonKey);

// Database schema types
export interface UserProfile {
  id: string;
  email: string;
  firstName: string;
  lastName: string;
  role: 'patient' | 'family_member';
  patientId?: string;
  phone?: string;
  dateOfBirth?: string;
  emergencyContact?: string;
  medicalHistory?: string[];
  allergies?: string[];
  currentMedications?: string[];
  insuranceProvider?: string;
  insuranceNumber?: string;
  createdAt: string;
  updatedAt: string;
}

export interface Appointment {
  id: string;
  patientId: string;
  type: 'consultation' | 'diagnosis' | 'imaging' | 'lab_test' | 'emergency';
  status: 'scheduled' | 'in_progress' | 'completed' | 'cancelled';
  scheduledAt: string;
  symptoms?: string;
  diagnosis?: string;
  treatment?: string;
  cost: number;
  paymentStatus: 'pending' | 'paid' | 'failed';
  paymentId?: string;
  createdAt: string;
  updatedAt: string;
}

export interface MedicalRecord {
  id: string;
  patientId: string;
  appointmentId: string;
  type: 'diagnosis' | 'lab_result' | 'imaging' | 'prescription';
  data: any;
  aiConfidence?: number;
  reviewedBy?: string;
  createdAt: string;
}

export interface Payment {
  id: string;
  patientId: string;
  appointmentId: string;
  amount: number;
  currency: string;
  status: 'pending' | 'succeeded' | 'failed' | 'refunded';
  stripePaymentIntentId: string;
  paymentMethod: string;
  createdAt: string;
}