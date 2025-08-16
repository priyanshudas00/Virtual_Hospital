import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';
import { createServer } from 'http';
import { Server } from 'socket.io';
import Stripe from 'stripe';
import bcrypt from 'bcryptjs';
import jwt from 'jsonwebtoken';

dotenv.config();

const app = express();
const server = createServer(app);
const io = new Server(server, {
  cors: {
    origin: "http://localhost:5173",
    methods: ["GET", "POST"]
  }
});

// Initialize Stripe
const stripe = new Stripe(process.env.STRIPE_SECRET_KEY || 'sk_test_your_key', {
  apiVersion: '2023-10-16',
});

const PORT = process.env.PORT || 3001;
const JWT_SECRET = process.env.JWT_SECRET || 'your-jwt-secret';

// Middleware
app.use(cors());
app.use(express.json());

// In-memory storage (replace with database in production)
const users = new Map();
const appointments = new Map();
const medicalRecords = new Map();
const payments = new Map();

// Authentication middleware
const authenticateToken = (req, res, next) => {
  const authHeader = req.headers['authorization'];
  const token = authHeader && authHeader.split(' ')[1];

  if (!token) {
    return res.sendStatus(401);
  }

  jwt.verify(token, JWT_SECRET, (err, user) => {
    if (err) return res.sendStatus(403);
    req.user = user;
    next();
  });
};

// Auth Routes
app.post('/api/auth/register', async (req, res) => {
  try {
    const { email, password, ...userData } = req.body;
    
    // Check if user exists
    if (users.has(email)) {
      return res.status(400).json({ error: 'User already exists' });
    }
    
    // Hash password
    const hashedPassword = await bcrypt.hash(password, 10);
    
    // Create user
    const userId = Date.now().toString();
    const user = {
      id: userId,
      email,
      password: hashedPassword,
      ...userData,
      createdAt: new Date().toISOString(),
    };
    
    users.set(email, user);
    
    // Generate JWT
    const token = jwt.sign({ userId, email }, JWT_SECRET, { expiresIn: '24h' });
    
    res.json({ 
      token, 
      user: { ...user, password: undefined } 
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.post('/api/auth/login', async (req, res) => {
  try {
    const { email, password } = req.body;
    
    const user = users.get(email);
    if (!user) {
      return res.status(400).json({ error: 'Invalid credentials' });
    }
    
    const validPassword = await bcrypt.compare(password, user.password);
    if (!validPassword) {
      return res.status(400).json({ error: 'Invalid credentials' });
    }
    
    const token = jwt.sign({ userId: user.id, email }, JWT_SECRET, { expiresIn: '24h' });
    
    res.json({ 
      token, 
      user: { ...user, password: undefined } 
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Appointment Routes
app.post('/api/appointments', authenticateToken, async (req, res) => {
  try {
    const appointmentId = Date.now().toString();
    const appointment = {
      id: appointmentId,
      patientId: req.user.userId,
      ...req.body,
      status: 'scheduled',
      paymentStatus: 'pending',
      createdAt: new Date().toISOString(),
    };
    
    appointments.set(appointmentId, appointment);
    
    res.json(appointment);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.get('/api/appointments', authenticateToken, (req, res) => {
  try {
    const userAppointments = Array.from(appointments.values())
      .filter(apt => apt.patientId === req.user.userId);
    
    res.json(userAppointments);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Payment Routes
app.post('/api/create-payment-intent', authenticateToken, async (req, res) => {
  try {
    const { amount, appointmentId } = req.body;
    
    const paymentIntent = await stripe.paymentIntents.create({
      amount: amount, // amount in cents
      currency: 'usd',
      metadata: {
        appointmentId,
        patientId: req.user.userId,
      },
    });
    
    res.json({
      id: paymentIntent.id,
      clientSecret: paymentIntent.client_secret,
      amount: paymentIntent.amount,
      currency: paymentIntent.currency,
      status: paymentIntent.status,
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.post('/api/confirm-payment', authenticateToken, async (req, res) => {
  try {
    const { paymentIntentId, appointmentId } = req.body;
    
    const paymentIntent = await stripe.paymentIntents.retrieve(paymentIntentId);
    
    if (paymentIntent.status === 'succeeded') {
      // Update appointment payment status
      const appointment = appointments.get(appointmentId);
      if (appointment) {
        appointment.paymentStatus = 'paid';
        appointment.paymentId = paymentIntentId;
        appointments.set(appointmentId, appointment);
      }
      
      // Store payment record
      const paymentId = Date.now().toString();
      const payment = {
        id: paymentId,
        patientId: req.user.userId,
        appointmentId,
        amount: paymentIntent.amount,
        currency: paymentIntent.currency,
        status: paymentIntent.status,
        stripePaymentIntentId: paymentIntentId,
        createdAt: new Date().toISOString(),
      };
      
      payments.set(paymentId, payment);
      
      res.json({ success: true, payment });
    } else {
      res.status(400).json({ error: 'Payment not successful' });
    }
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Medical Records Routes
app.post('/api/medical-records', authenticateToken, (req, res) => {
  try {
    const recordId = Date.now().toString();
    const record = {
      id: recordId,
      patientId: req.user.userId,
      ...req.body,
      createdAt: new Date().toISOString(),
    };
    
    medicalRecords.set(recordId, record);
    
    res.json(record);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.get('/api/medical-records', authenticateToken, (req, res) => {
  try {
    const userRecords = Array.from(medicalRecords.values())
      .filter(record => record.patientId === req.user.userId);
    
    res.json(userRecords);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Mock AI Services
const diagnoseSymptoms = (symptoms) => {
  // Simulate AI diagnosis
  const conditions = [
    { name: 'Viral Upper Respiratory Infection', confidence: 0.87 },
    { name: 'Common Cold', confidence: 0.72 },
    { name: 'Allergic Rhinitis', confidence: 0.45 }
  ];
  
  return {
    primaryDiagnosis: conditions[0].name,
    confidence: conditions[0].confidence,
    alternativeDiagnoses: conditions.slice(1).map(c => ({
      condition: c.name,
      probability: c.confidence
    })),
    urgency: 'Low',
    recommendedActions: [
      'Rest and stay hydrated',
      'Take over-the-counter pain relievers',
      'Monitor symptoms for 3-5 days'
    ]
  };
};

const analyzeImage = (imageData) => {
  // Simulate medical image analysis
  return {
    imageType: 'Chest X-Ray',
    findings: [
      {
        finding: 'Normal lung fields',
        confidence: 0.94,
        severity: 'Normal',
        location: 'Bilateral lung fields'
      }
    ],
    overallAssessment: 'Normal chest X-ray with no acute findings',
    recommendations: ['No immediate intervention required'],
    riskLevel: 'Low'
  };
};

// API Routes
app.get('/api/health', (req, res) => {
  res.json({ status: 'healthy', timestamp: new Date().toISOString() });
});

app.post('/api/diagnose', (req, res) => {
  const { symptoms } = req.body;
  
  setTimeout(() => {
    const diagnosis = diagnoseSymptoms(symptoms);
    res.json(diagnosis);
  }, 2000); // Simulate processing time
});

app.post('/api/analyze-image', (req, res) => {
  const { imageData } = req.body;
  
  setTimeout(() => {
    const analysis = analyzeImage(imageData);
    res.json(analysis);
  }, 3000);
});

app.post('/api/lab-results', (req, res) => {
  const { results } = req.body;
  
  const analysis = {
    summary: 'Overall lab results show good health markers',
    keyFindings: ['All parameters within normal ranges'],
    recommendations: ['Continue current lifestyle'],
    riskAssessment: 'Low risk'
  };
  
  res.json(analysis);
});

// User Profile Routes
app.get('/api/profile', authenticateToken, (req, res) => {
  try {
    const user = Array.from(users.values()).find(u => u.id === req.user.userId);
    if (!user) {
      return res.status(404).json({ error: 'User not found' });
    }
    
    res.json({ ...user, password: undefined });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.put('/api/profile', authenticateToken, (req, res) => {
  try {
    const user = Array.from(users.values()).find(u => u.id === req.user.userId);
    if (!user) {
      return res.status(404).json({ error: 'User not found' });
    }
    
    const updatedUser = { ...user, ...req.body, updatedAt: new Date().toISOString() };
    users.set(user.email, updatedUser);
    
    res.json({ ...updatedUser, password: undefined });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Socket.IO for real-time features
io.on('connection', (socket) => {
  console.log('Client connected:', socket.id);
  
  socket.on('emergency_alert', (data) => {
    // Broadcast emergency to relevant systems
    io.emit('emergency_response', {
      patientId: data.patientId,
      location: data.location,
      symptoms: data.symptoms,
      timestamp: new Date()
    });
  });
  
  socket.on('appointment_update', (data) => {
    // Broadcast appointment updates
    io.emit('appointment_notification', {
      appointmentId: data.appointmentId,
      status: data.status,
      timestamp: new Date()
    });
  });
  
  socket.on('disconnect', () => {
    console.log('Client disconnected:', socket.id);
  });
});

server.listen(PORT, () => {
  console.log(`🏥 AI Hospital Backend running on port ${PORT}`);
  console.log(`💳 Stripe integration enabled`);
  console.log(`🔐 JWT authentication enabled`);
});