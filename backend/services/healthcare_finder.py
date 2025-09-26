import os
import logging
import requests
import asyncio
from typing import Dict, List, Tuple, Optional
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import googlemaps
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)

class IntelligentHealthcareFinder:
    """Advanced healthcare provider finder with AI-powered matching"""
    
    def __init__(self):
        self.google_maps_key = os.getenv('GOOGLE_MAPS_API_KEY')
        self.gmaps = googlemaps.Client(key=self.google_maps_key) if self.google_maps_key else None
        self.geolocator = Nominatim(user_agent="virtual_hospital_v2.0")
        
        # Medical specialization mapping
        self.specialization_mapping = {
            'cardiac': ['cardiology', 'cardiovascular surgery', 'interventional cardiology'],
            'heart': ['cardiology', 'cardiac surgery', 'cardiovascular medicine'],
            'chest pain': ['cardiology', 'emergency medicine', 'internal medicine'],
            'neurological': ['neurology', 'neurosurgery', 'neuropsychiatry'],
            'brain': ['neurology', 'neurosurgery', 'neuroimaging'],
            'headache': ['neurology', 'internal medicine', 'family medicine'],
            'migraine': ['neurology', 'headache specialist', 'pain management'],
            'diabetes': ['endocrinology', 'diabetes specialist', 'internal medicine'],
            'thyroid': ['endocrinology', 'thyroid specialist'],
            'kidney': ['nephrology', 'urology', 'internal medicine'],
            'liver': ['gastroenterology', 'hepatology', 'internal medicine'],
            'skin': ['dermatology', 'dermatopathology'],
            'bone': ['orthopedics', 'rheumatology', 'sports medicine'],
            'joint': ['orthopedics', 'rheumatology', 'physical medicine'],
            'mental health': ['psychiatry', 'psychology', 'behavioral health'],
            'anxiety': ['psychiatry', 'psychology', 'anxiety specialist'],
            'depression': ['psychiatry', 'psychology', 'mental health'],
            'respiratory': ['pulmonology', 'respiratory medicine', 'chest medicine'],
            'lung': ['pulmonology', 'thoracic surgery', 'respiratory therapy'],
            'pregnancy': ['obstetrics', 'gynecology', 'maternal-fetal medicine'],
            'pediatric': ['pediatrics', 'pediatric subspecialties'],
            'eye': ['ophthalmology', 'optometry', 'retina specialist'],
            'ear': ['otolaryngology', 'ent', 'audiology'],
            'cancer': ['oncology', 'hematology-oncology', 'radiation oncology'],
            'pain': ['pain management', 'anesthesiology', 'physical medicine']
        }
        
        # Emergency service requirements
        self.emergency_services = [
            'emergency_department', 'trauma_center', 'cardiac_catheterization',
            'stroke_center', 'intensive_care', 'surgical_services'
        ]
    
    async def find_optimal_providers(self, patient_location: str, medical_profile: Dict,
                                   search_criteria: Dict) -> Dict:
        """Find optimal healthcare providers using intelligent matching"""
        try:
            # Geocode patient location with enhanced accuracy
            location_data = await self._enhanced_geocoding(patient_location)
            if not location_data:
                return {'error': 'Could not determine precise location coordinates'}
            
            # Analyze medical needs and determine search strategy
            search_strategy = self._determine_search_strategy(medical_profile, search_criteria)
            
            # Multi-source provider search
            all_providers = {}
            
            # Search doctors
            doctors = await self._search_doctors(location_data, search_strategy)
            all_providers['doctors'] = doctors
            
            # Search hospitals
            hospitals = await self._search_hospitals(location_data, search_strategy)
            all_providers['hospitals'] = hospitals
            
            # Search specialized clinics
            clinics = await self._search_specialized_clinics(location_data, search_strategy)
            all_providers['clinics'] = clinics
            
            # Search urgent care centers
            urgent_care = await self._search_urgent_care(location_data, search_strategy)
            all_providers['urgent_care'] = urgent_care
            
            # Apply intelligent ranking and filtering
            ranked_providers = await self._intelligent_provider_ranking(
                all_providers, medical_profile, search_criteria, location_data
            )
            
            # Generate accessibility analysis
            accessibility_analysis = self._analyze_accessibility(
                ranked_providers, search_criteria, location_data
            )
            
            return {
                'success': True,
                'patient_location': location_data,
                'search_strategy': search_strategy,
                'providers': ranked_providers,
                'accessibility_analysis': accessibility_analysis,
                'total_found': sum(len(providers) for providers in all_providers.values()),
                'search_metadata': {
                    'timestamp': datetime.utcnow().isoformat(),
                    'search_radius': search_strategy.get('radius_km', 50),
                    'specializations_searched': search_strategy.get('specializations', []),
                    'urgency_level': search_criteria.get('urgency', 'routine')
                }
            }
            
        except Exception as e:
            logger.error(f"Provider search failed: {e}")
            return {
                'success': False,
                'error': f'Provider search failed: {str(e)}'
            }
    
    async def _enhanced_geocoding(self, location: str) -> Optional[Dict]:
        """Enhanced geocoding with multiple fallbacks"""
        try:
            # Try Google Maps Geocoding API first (most accurate)
            if self.gmaps:
                geocode_result = self.gmaps.geocode(location)
                if geocode_result:
                    result = geocode_result[0]
                    return {
                        'latitude': result['geometry']['location']['lat'],
                        'longitude': result['geometry']['location']['lng'],
                        'formatted_address': result['formatted_address'],
                        'address_components': result['address_components'],
                        'place_id': result['place_id'],
                        'accuracy': 'high',
                        'source': 'Google Maps'
                    }
            
            # Fallback to Nominatim
            location_data = self.geolocator.geocode(location, exactly_one=True, timeout=10)
            if location_data:
                return {
                    'latitude': location_data.latitude,
                    'longitude': location_data.longitude,
                    'formatted_address': location_data.address,
                    'accuracy': 'medium',
                    'source': 'Nominatim'
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Enhanced geocoding failed: {e}")
            return None
    
    def _determine_search_strategy(self, medical_profile: Dict, search_criteria: Dict) -> Dict:
        """Determine intelligent search strategy based on medical needs"""
        strategy = {
            'specializations': [],
            'services_required': [],
            'urgency_level': search_criteria.get('urgency', 'routine'),
            'radius_km': 50,  # Default radius
            'emergency_capable': False,
            'telemedicine_suitable': False
        }
        
        # Extract primary condition
        primary_diagnosis = medical_profile.get('primary_diagnosis', '').lower()
        urgency_level = medical_profile.get('urgency_level', 'LOW')
        
        # Map conditions to specializations
        for condition_keyword, specializations in self.specialization_mapping.items():
            if condition_keyword in primary_diagnosis:
                strategy['specializations'].extend(specializations)
        
        # Default to general medicine if no specific specialization
        if not strategy['specializations']:
            strategy['specializations'] = ['internal medicine', 'family medicine', 'general practice']
        
        # Adjust strategy based on urgency
        if urgency_level in ['EMERGENCY', 'HIGH']:
            strategy['radius_km'] = 25  # Smaller radius for urgent care
            strategy['emergency_capable'] = True
            strategy['services_required'].extend(self.emergency_services)
        elif urgency_level == 'MEDIUM':
            strategy['radius_km'] = 35
            strategy['services_required'] = ['urgent_care', 'same_day_appointments']
        else:
            strategy['telemedicine_suitable'] = True
        
        # Consider financial constraints
        financial_capability = search_criteria.get('financial_capability', 'medium')
        if financial_capability == 'low':
            strategy['services_required'].append('community_health_center')
            strategy['insurance_required'] = True
        
        return strategy
    
    async def _search_doctors(self, location_data: Dict, search_strategy: Dict) -> List[Dict]:
        """Search for doctors using multiple data sources"""
        doctors = []
        
        try:
            # Google Places API search
            if self.gmaps:
                for specialization in search_strategy['specializations'][:3]:  # Limit API calls
                    places_results = await self._google_places_search(
                        location_data,
                        'doctor',
                        specialization,
                        search_strategy['radius_km']
                    )
                    
                    for place in places_results:
                        doctor_data = await self._enrich_doctor_data(place, search_strategy)
                        if doctor_data:
                            doctors.append(doctor_data)
            
            # Add synthetic doctors for demonstration (in production, use real database)
            synthetic_doctors = self._generate_synthetic_doctors(location_data, search_strategy)
            doctors.extend(synthetic_doctors)
            
            # Remove duplicates and sort by relevance
            doctors = self._deduplicate_providers(doctors)
            doctors = sorted(doctors, key=lambda x: x.get('relevance_score', 0), reverse=True)
            
            return doctors[:25]  # Return top 25 doctors
            
        except Exception as e:
            logger.error(f"Doctor search failed: {e}")
            return []
    
    async def _search_hospitals(self, location_data: Dict, search_strategy: Dict) -> List[Dict]:
        """Search for hospitals with comprehensive data"""
        hospitals = []
        
        try:
            # Google Places API search for hospitals
            if self.gmaps:
                places_results = await self._google_places_search(
                    location_data,
                    'hospital',
                    'medical center',
                    search_strategy['radius_km']
                )
                
                for place in places_results:
                    hospital_data = await self._enrich_hospital_data(place, search_strategy)
                    if hospital_data:
                        hospitals.append(hospital_data)
            
            # Add comprehensive synthetic hospitals
            synthetic_hospitals = self._generate_synthetic_hospitals(location_data, search_strategy)
            hospitals.extend(synthetic_hospitals)
            
            # Filter based on emergency requirements
            if search_strategy.get('emergency_capable'):
                hospitals = [h for h in hospitals if h.get('emergency_services', False)]
            
            # Sort by relevance and emergency capability
            hospitals = sorted(hospitals, key=lambda x: (
                x.get('emergency_services', False),
                x.get('relevance_score', 0)
            ), reverse=True)
            
            return hospitals[:20]  # Return top 20 hospitals
            
        except Exception as e:
            logger.error(f"Hospital search failed: {e}")
            return []
    
    def _generate_synthetic_doctors(self, location_data: Dict, search_strategy: Dict) -> List[Dict]:
        """Generate realistic synthetic doctor data for demonstration"""
        doctors = []
        
        specializations = search_strategy.get('specializations', ['general medicine'])
        
        for i, specialization in enumerate(specializations[:5]):
            for j in range(3):  # 3 doctors per specialization
                doctor = {
                    '_id': f"doc_{specialization.replace(' ', '_')}_{j+1}",
                    'name': f"Dr. {self._generate_doctor_name()}",
                    'specialization': specialization.title(),
                    'qualifications': f"MD, {self._get_specialization_qualification(specialization)}",
                    'experience': np.random.randint(5, 25),
                    'address': f"{np.random.randint(100, 999)} Medical Plaza, {location_data.get('formatted_address', 'Unknown').split(',')[0]}",
                    'phone': f"+1-{np.random.randint(200, 999)}-{np.random.randint(200, 999)}-{np.random.randint(1000, 9999)}",
                    'consultation_fee': self._calculate_consultation_fee(specialization, search_strategy),
                    'cost_category': self._determine_cost_category(specialization),
                    'rating': round(np.random.uniform(3.5, 5.0), 1),
                    'availability': self._generate_availability(),
                    'distance': np.random.uniform(1.0, search_strategy.get('radius_km', 50)),
                    'emergency_availability': specialization in ['emergency medicine', 'internal medicine'],
                    'telemedicine': np.random.choice([True, False]),
                    'languages': ['English', np.random.choice(['Spanish', 'Hindi', 'Mandarin', 'French'])],
                    'insurance_accepted': ['Blue Cross', 'Aetna', 'Medicare', 'Medicaid'],
                    'relevance_score': self._calculate_doctor_relevance(specialization, search_strategy)
                }
                doctors.append(doctor)
        
        return doctors
    
    def _generate_synthetic_hospitals(self, location_data: Dict, search_strategy: Dict) -> List[Dict]:
        """Generate realistic synthetic hospital data"""
        hospitals = []
        
        hospital_types = ['General Hospital', 'Medical Center', 'Specialty Hospital', 'Teaching Hospital']
        
        for i, hospital_type in enumerate(hospital_types):
            hospital = {
                '_id': f"hosp_{hospital_type.replace(' ', '_').lower()}_{i+1}",
                'name': f"{location_data.get('formatted_address', 'City').split(',')[0]} {hospital_type}",
                'type': hospital_type,
                'specialties': self._generate_hospital_specialties(search_strategy),
                'address': f"{np.random.randint(1000, 9999)} Hospital Drive, {location_data.get('formatted_address', 'Unknown').split(',')[0]}",
                'phone': f"+1-{np.random.randint(200, 999)}-{np.random.randint(200, 999)}-{np.random.randint(1000, 9999)}",
                'emergency_services': hospital_type in ['General Hospital', 'Medical Center'],
                'cost_category': np.random.choice(['low', 'medium', 'high']),
                'rating': round(np.random.uniform(3.0, 5.0), 1),
                'bed_capacity': np.random.randint(100, 500),
                'facilities': self._generate_hospital_facilities(),
                'distance': np.random.uniform(2.0, search_strategy.get('radius_km', 50)),
                'trauma_level': np.random.choice(['Level I', 'Level II', 'Level III', None]),
                'accreditation': ['Joint Commission', 'NABH'],
                'insurance_accepted': ['All major insurance', 'Medicare', 'Medicaid'],
                'wait_times': {
                    'emergency': f"{np.random.randint(15, 60)} minutes",
                    'urgent_care': f"{np.random.randint(30, 120)} minutes",
                    'scheduled': f"{np.random.randint(1, 14)} days"
                },
                'relevance_score': self._calculate_hospital_relevance(hospital_type, search_strategy)
            }
            hospitals.append(hospital)
        
        return hospitals
    
    async def _intelligent_provider_ranking(self, all_providers: Dict, medical_profile: Dict,
                                          search_criteria: Dict, location_data: Dict) -> Dict:
        """Apply intelligent ranking algorithm to providers"""
        ranked_providers = {}
        
        for provider_type, providers in all_providers.items():
            if not providers:
                ranked_providers[provider_type] = []
                continue
            
            # Apply multi-factor ranking
            for provider in providers:
                score = self._calculate_comprehensive_score(
                    provider, medical_profile, search_criteria, location_data
                )
                provider['comprehensive_score'] = score
                provider['ranking_factors'] = self._explain_ranking_factors(provider, score)
            
            # Sort by comprehensive score
            sorted_providers = sorted(providers, key=lambda x: x['comprehensive_score'], reverse=True)
            ranked_providers[provider_type] = sorted_providers
        
        return ranked_providers
    
    def _calculate_comprehensive_score(self, provider: Dict, medical_profile: Dict,
                                     search_criteria: Dict, location_data: Dict) -> float:
        """Calculate comprehensive provider matching score"""
        score = 0.0
        max_score = 1.0
        
        # Distance factor (40% weight)
        distance = provider.get('distance', float('inf'))
        if distance <= 5:
            score += 0.4
        elif distance <= 15:
            score += 0.3
        elif distance <= 30:
            score += 0.2
        else:
            score += 0.1
        
        # Specialization match (30% weight)
        provider_specialization = provider.get('specialization', '').lower()
        required_specializations = medical_profile.get('specialist_needed', [])
        
        if any(spec.lower() in provider_specialization for spec in required_specializations):
            score += 0.3
        elif 'general' in provider_specialization or 'internal' in provider_specialization:
            score += 0.2
        
        # Urgency match (20% weight)
        urgency = search_criteria.get('urgency', 'routine')
        if urgency == 'emergency' and provider.get('emergency_services', False):
            score += 0.2
        elif urgency == 'urgent' and provider.get('urgent_care', False):
            score += 0.15
        else:
            score += 0.1
        
        # Cost compatibility (10% weight)
        financial_capability = search_criteria.get('financial_capability', 'medium')
        provider_cost = provider.get('cost_category', 'medium')
        
        if financial_capability == provider_cost:
            score += 0.1
        elif abs(['low', 'medium', 'high'].index(financial_capability) - 
                ['low', 'medium', 'high'].index(provider_cost)) == 1:
            score += 0.05
        
        return min(score, max_score)
    
    def _analyze_accessibility(self, providers: Dict, search_criteria: Dict, 
                             location_data: Dict) -> Dict:
        """Analyze accessibility factors for healthcare access"""
        return {
            'transportation': {
                'public_transport_available': True,  # Would check real transit data
                'parking_availability': 'Available at most locations',
                'wheelchair_accessible': 'Most facilities are ADA compliant'
            },
            'financial_accessibility': {
                'insurance_coverage': 'Multiple insurance options available',
                'payment_plans': 'Most providers offer payment plans',
                'community_health_options': len([p for providers_list in providers.values() 
                                               for p in providers_list 
                                               if p.get('cost_category') == 'low'])
            },
            'language_services': {
                'interpreter_services': 'Available at major hospitals',
                'multilingual_staff': 'Common languages supported',
                'cultural_competency': 'Culturally diverse healthcare teams'
            },
            'wait_times': {
                'emergency': 'Immediate for life-threatening conditions',
                'urgent': '1-4 hours for urgent care',
                'routine': '1-14 days for scheduled appointments'
            }
        }
    
    # Helper methods for realistic data generation
    def _generate_doctor_name(self) -> str:
        """Generate realistic doctor names"""
        first_names = ['Sarah', 'Michael', 'Jennifer', 'David', 'Lisa', 'Robert', 'Maria', 'James', 'Patricia', 'John']
        last_names = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis', 'Rodriguez', 'Martinez']
        return f"{np.random.choice(first_names)} {np.random.choice(last_names)}"
    
    def _get_specialization_qualification(self, specialization: str) -> str:
        """Get appropriate qualifications for specialization"""
        qualifications = {
            'cardiology': 'Board Certified Cardiologist',
            'neurology': 'Board Certified Neurologist',
            'orthopedics': 'Board Certified Orthopedic Surgeon',
            'dermatology': 'Board Certified Dermatologist',
            'internal medicine': 'Board Certified Internist',
            'family medicine': 'Board Certified Family Physician'
        }
        return qualifications.get(specialization, 'Board Certified Physician')
    
    def _calculate_consultation_fee(self, specialization: str, search_strategy: Dict) -> Dict:
        """Calculate realistic consultation fees"""
        base_fees = {
            'cardiology': {'min': 300, 'max': 800},
            'neurology': {'min': 350, 'max': 900},
            'orthopedics': {'min': 250, 'max': 700},
            'dermatology': {'min': 200, 'max': 500},
            'internal medicine': {'min': 150, 'max': 400},
            'family medicine': {'min': 100, 'max': 300}
        }
        
        fee_range = base_fees.get(specialization, {'min': 150, 'max': 400})
        
        return {
            'currency': 'USD',
            'consultation_fee': np.random.randint(fee_range['min'], fee_range['max']),
            'follow_up_fee': np.random.randint(fee_range['min'] // 2, fee_range['max'] // 2),
            'telemedicine_fee': np.random.randint(fee_range['min'] // 3, fee_range['max'] // 3)
        }

# Global healthcare finder instance
healthcare_finder = IntelligentHealthcareFinder()

# Import numpy for random number generation
import numpy as np