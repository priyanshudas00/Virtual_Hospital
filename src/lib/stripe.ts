import { loadStripe } from '@stripe/stripe-js';

const stripePublishableKey = import.meta.env.VITE_STRIPE_PUBLISHABLE_KEY || 'pk_test_your_key';

export const stripePromise = loadStripe(stripePublishableKey);

export interface PaymentIntent {
  id: string;
  amount: number;
  currency: string;
  status: string;
  clientSecret: string;
}

export const createPaymentIntent = async (amount: number, appointmentId: string): Promise<PaymentIntent> => {
  const response = await fetch('/api/create-payment-intent', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      amount: amount * 100, // Convert to cents
      appointmentId,
    }),
  });

  if (!response.ok) {
    throw new Error('Failed to create payment intent');
  }

  return response.json();
};

export const confirmPayment = async (paymentIntentId: string, paymentMethodId: string) => {
  const response = await fetch('/api/confirm-payment', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      paymentIntentId,
      paymentMethodId,
    }),
  });

  if (!response.ok) {
    throw new Error('Failed to confirm payment');
  }

  return response.json();
};