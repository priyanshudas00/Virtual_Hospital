const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:3002';

class ApiClient {
  private baseURL: string;

  constructor() {
    this.baseURL = API_URL;
  }

  private async request(endpoint: string, options: RequestInit = {}) {
    const token = localStorage.getItem('token');
    
    const config: RequestInit = {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...(token && { Authorization: `Bearer ${token}` }),
        ...options.headers,
      },
    };

    const response = await fetch(`${this.baseURL}${endpoint}`, config);
    
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.message || 'API request failed');
    }

    return response.json();
  }

  // Auth endpoints
  async register(userData: any) {
    return this.request('/api/auth/register', {
      method: 'POST',
      body: JSON.stringify(userData),
    });
  }

  async login(credentials: any) {
    return this.request('/api/auth/login', {
      method: 'POST',
      body: JSON.stringify(credentials),
    });
  }

  // Medical analysis endpoints
  async analyzeSymptoms(data: any) {
    return this.request('/api/analyze/symptoms', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  async submitIntakeForm(formData: any) {
    return this.request('/api/intake-form', {
      method: 'POST',
      body: JSON.stringify(formData),
    });
  }

  async uploadMedicalReport(formData: FormData) {
    const token = localStorage.getItem('token');
    
    const response = await fetch(`${this.baseURL}/api/analyze/report-text`, {
      method: 'POST',
      headers: {
        ...(token && { Authorization: `Bearer ${token}` }),
      },
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.message || 'Upload failed');
    }

    return response.json();
  }

  async uploadMedicalImage(formData: FormData) {
    const token = localStorage.getItem('token');
    
    const response = await fetch(`${this.baseURL}/api/analyze/report-image`, {
      method: 'POST',
      headers: {
        ...(token && { Authorization: `Bearer ${token}` }),
      },
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.message || 'Upload failed');
    }

    return response.json();
  }

  // Dashboard endpoints
  async getDashboardData() {
    return this.request('/api/dashboard');
  }

  async getDiagnosisReports() {
    return this.request('/api/diagnosis-reports');
  }

  async getSpecificReport(reportId: string) {
    return this.request(`/api/diagnosis-reports/${reportId}`);
  }
}

export const apiClient = new ApiClient();