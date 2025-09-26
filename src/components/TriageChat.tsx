import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Send, 
  Bot, 
  User, 
  AlertTriangle, 
  Clock, 
  CheckCircle,
  Brain,
  Heart,
  Activity
} from 'lucide-react';

interface Message {
  id: string;
  type: 'user' | 'ai' | 'system';
  content: string;
  timestamp: Date;
  isQuestion?: boolean;
  questionType?: string;
}

interface TriageChatProps {
  onAssessmentComplete: (assessment: any) => void;
  initialComplaint?: string;
}

export const TriageChat: React.FC<TriageChatProps> = ({ 
  onAssessmentComplete, 
  initialComplaint 
}) => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [currentInput, setCurrentInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [interactionId, setInteractionId] = useState<string | null>(null);
  const [currentQuestion, setCurrentQuestion] = useState<any>(null);
  const [progress, setProgress] = useState(0);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (initialComplaint) {
      startTriageSession(initialComplaint);
    }
  }, [initialComplaint]);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const startTriageSession = async (complaint: string) => {
    setIsLoading(true);
    
    try {
      const response = await fetch(`${import.meta.env.VITE_API_URL}/api/triage/start-session`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        },
        body: JSON.stringify({
          complaint,
          primary_symptom: complaint.split(' ')[0],
          severity: 5,
          onset: 'gradual'
        })
      });

      const result = await response.json();

      if (result.emergency_detected) {
        setMessages([
          {
            id: '1',
            type: 'system',
            content: result.response,
            timestamp: new Date()
          }
        ]);
        return;
      }

      if (result.success) {
        setSessionId(result.session_id);
        setInteractionId(result.interaction_id);
        
        // Add initial messages
        setMessages([
          {
            id: '1',
            type: 'user',
            content: complaint,
            timestamp: new Date()
          },
          {
            id: '2',
            type: 'ai',
            content: "I understand you're experiencing some symptoms. Let me ask you a few questions to better understand your situation and provide appropriate guidance.",
            timestamp: new Date()
          }
        ]);

        // Add first question
        if (result.questions && result.questions.length > 0) {
          const firstQuestion = result.questions[0];
          setCurrentQuestion(firstQuestion);
          
          setMessages(prev => [...prev, {
            id: `q-${Date.now()}`,
            type: 'ai',
            content: firstQuestion.question,
            timestamp: new Date(),
            isQuestion: true,
            questionType: firstQuestion.type
          }]);
        }
      }
    } catch (error) {
      console.error('Failed to start triage session:', error);
      setMessages([{
        id: 'error',
        type: 'system',
        content: 'Sorry, I encountered an error. Please try again or contact support.',
        timestamp: new Date()
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSendMessage = async () => {
    if (!currentInput.trim() || !currentQuestion || !interactionId) return;

    const userMessage: Message = {
      id: `user-${Date.now()}`,
      type: 'user',
      content: currentInput,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setIsLoading(true);

    try {
      const response = await fetch(`${import.meta.env.VITE_API_URL}/api/triage/answer-question`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        },
        body: JSON.stringify({
          interaction_id: interactionId,
          question: currentQuestion.question,
          answer: currentInput,
          question_type: currentQuestion.type
        })
      });

      const result = await response.json();

      if (result.emergency_detected) {
        setMessages(prev => [...prev, {
          id: `emergency-${Date.now()}`,
          type: 'system',
          content: result.response,
          timestamp: new Date()
        }]);
        return;
      }

      if (result.assessment_complete) {
        // Assessment is complete
        setMessages(prev => [...prev, {
          id: `complete-${Date.now()}`,
          type: 'ai',
          content: "Thank you for providing all the information. I've completed my analysis of your symptoms. Let me show you the results.",
          timestamp: new Date()
        }]);

        setProgress(100);
        
        // Trigger assessment modal
        setTimeout(() => {
          onAssessmentComplete(result.assessment);
        }, 1000);

      } else {
        // Continue with next questions
        setProgress(result.progress || 0);
        
        if (result.questions && result.questions.length > 0) {
          const nextQuestion = result.questions[0];
          setCurrentQuestion(nextQuestion);
          
          setMessages(prev => [...prev, {
            id: `q-${Date.now()}`,
            type: 'ai',
            content: nextQuestion.question,
            timestamp: new Date(),
            isQuestion: true,
            questionType: nextQuestion.type
          }]);
        }
      }
    } catch (error) {
      console.error('Failed to send message:', error);
      setMessages(prev => [...prev, {
        id: `error-${Date.now()}`,
        type: 'system',
        content: 'Sorry, I encountered an error processing your response. Please try again.',
        timestamp: new Date()
      }]);
    } finally {
      setIsLoading(false);
      setCurrentInput('');
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const getMessageIcon = (type: string, questionType?: string) => {
    switch (type) {
      case 'user':
        return <User className="w-5 h-5" />;
      case 'system':
        return <AlertTriangle className="w-5 h-5" />;
      default:
        if (questionType === 'urgency') return <AlertTriangle className="w-5 h-5" />;
        if (questionType === 'severity') return <Activity className="w-5 h-5" />;
        if (questionType === 'location') return <Heart className="w-5 h-5" />;
        return <Bot className="w-5 h-5" />;
    }
  };

  const getMessageColor = (type: string, questionType?: string) => {
    switch (type) {
      case 'user':
        return 'bg-blue-600 text-white';
      case 'system':
        return 'bg-red-100 text-red-800 border border-red-300';
      default:
        if (questionType === 'urgency') return 'bg-orange-100 text-orange-800';
        if (questionType === 'severity') return 'bg-purple-100 text-purple-800';
        return 'bg-gray-100 text-gray-800';
    }
  };

  return (
    <div className="flex flex-col h-[600px] bg-white rounded-2xl shadow-xl">
      {/* Header */}
      <div className="bg-gradient-to-r from-blue-600 to-purple-600 p-4 rounded-t-2xl">
        <div className="flex items-center justify-between text-white">
          <div className="flex items-center space-x-3">
            <div className="bg-white/20 p-2 rounded-lg">
              <Brain className="w-6 h-6" />
            </div>
            <div>
              <h3 className="font-bold text-lg">AI Medical Triage</h3>
              <p className="text-blue-100 text-sm">Intelligent symptom assessment</p>
            </div>
          </div>
          {progress > 0 && (
            <div className="text-right">
              <div className="text-sm mb-1">{Math.round(progress)}% Complete</div>
              <div className="bg-white/20 rounded-full h-2 w-24">
                <div 
                  className="bg-white h-2 rounded-full transition-all duration-500"
                  style={{ width: `${progress}%` }}
                />
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        <AnimatePresence>
          {messages.map((message) => (
            <motion.div
              key={message.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className={`flex items-start space-x-3 ${
                message.type === 'user' ? 'flex-row-reverse space-x-reverse' : ''
              }`}
            >
              <div className={`p-2 rounded-lg ${getMessageColor(message.type, message.questionType)}`}>
                {getMessageIcon(message.type, message.questionType)}
              </div>
              <div className={`max-w-xs lg:max-w-md px-4 py-3 rounded-2xl ${
                message.type === 'user' 
                  ? 'bg-blue-600 text-white ml-auto' 
                  : message.type === 'system'
                  ? 'bg-red-50 border border-red-200 text-red-800'
                  : 'bg-gray-100 text-gray-800'
              }`}>
                <p className="text-sm leading-relaxed">{message.content}</p>
                <div className="text-xs opacity-70 mt-1">
                  {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                </div>
              </div>
            </motion.div>
          ))}
        </AnimatePresence>

        {isLoading && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="flex items-center space-x-3"
          >
            <div className="bg-gray-100 p-2 rounded-lg">
              <Bot className="w-5 h-5 text-gray-600" />
            </div>
            <div className="bg-gray-100 px-4 py-3 rounded-2xl">
              <div className="flex items-center space-x-2">
                <div className="flex space-x-1">
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                </div>
                <span className="text-sm text-gray-600">AI is thinking...</span>
              </div>
            </div>
          </motion.div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="border-t border-gray-200 p-4">
        <div className="flex items-center space-x-3">
          <div className="flex-1 relative">
            <textarea
              value={currentInput}
              onChange={(e) => setCurrentInput(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder={currentQuestion ? "Type your answer..." : "Describe your symptoms..."}
              className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 resize-none"
              rows={2}
              disabled={isLoading || !currentQuestion}
            />
          </div>
          <button
            onClick={handleSendMessage}
            disabled={!currentInput.trim() || isLoading || !currentQuestion}
            className="bg-blue-600 text-white p-3 rounded-xl hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200"
          >
            <Send className="w-5 h-5" />
          </button>
        </div>
        
        {currentQuestion && (
          <div className="mt-2 text-xs text-gray-500">
            Question type: {currentQuestion.type} • Priority: {currentQuestion.priority}
          </div>
        )}
      </div>
    </div>
  );
};