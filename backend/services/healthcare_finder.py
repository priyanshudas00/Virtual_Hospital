import os
import logging
import requests
from typing import Dict, List, Tuple, Optional
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import googlemaps
from datetime import datetime
import asyncio

logger = logging.getLogger(__name__)

class HealthcareProviderFinder:
    def __init__(self):
        self.google_maps_key = os.getenv('GOOGLE_MAPS_API_KEY')
        self.gmaps = googlemaps.Client(key=self.google_maps_key) if self.google_maps_key else None
        self.geolocator = Nominatim(user_agent="virtual_hospital_v1.0")
        
    async def find_providers(self, patient_location: str, medical_profile: Dict, 
                           financial_capability: str, urgency: str) -> Dict:
        """Find healthcare providers based on patient needs"""
        try:
            # Geocode patient location
            location_coords = await self._geocode_location(patient_location)
            if not location_coords:
                return {'error': 'Could not determine location coordinates'}
            
            # Determine search criteria based on medical profile
            search_criteria = self._determine_search_criteria(medical_profile, urgency)
            
            # Search for providers
            providers_data = {}
            
            # Find doctors
            doctors = await self._find_doctors(
                location_coords, 
                search_criteria, 
                financial_capability,
                urgency
            )
            providers_data['doctors'] = doctors
            
            # Find hospitals
            hospitals = await self._find_hospitals(
                location_coords, 
                search_criteria, 
                financial_capability,
                urgency
            )
            providers_data['hospitals'] = hospitals
            
            # Find specialized clinics
            clinics = await self._find_specialized_clinics(
                location_coords, 
                search_criteria, 
                financial_capability
            )
            providers_data['clinics'] = clinics
            
            # Rank and filter results
            ranked_providers = self._rank_providers(providers_data, medical_profile, urgency)
            
            return {
                'success': True,
                'patient_location': location_coords,
                'search_criteria': search_criteria,
                'providers': ranked_providers,
                'total_found': sum(len(providers) for providers in providers_data.values()),
                'search_timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Provider search failed: {e}")
            return {
                'success': False,
                'error': f'Provider search failed: {str(e)}'
            }
    
    async def _geocode_location(self, location: str) -> Optional[Dict]:
        """Convert location string to coordinates"""
        try:
            # Try Google Maps Geocoding API first
            if self.gmaps:
                geocode_result = self.gmaps.geocode(location)
                if geocode_result:
                    coords = geocode_result[0]['geometry']['location']
                    return {
                        'latitude': coords['lat'],
                        'longitude': coords['lng'],
                        'formatted_address': geocode_result[0]['formatted_address'],
                        'source': 'Google Maps'
                    }
            
            # Fallback to Nominatim
            location_data = self.geolocator.geocode(location)
            if location_data:
                return {
                    'latitude': location_data.latitude,
                    'longitude': location_data.longitude,
                    'formatted_address': location_data.address,
                    'source': 'Nominatim'
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Geocoding failed: {e}")
            return None
    
    def _determine_search_criteria(self, medical_profile: Dict, urgency: str) -> Dict:
        """Determine search criteria based on medical assessment"""
        criteria = {
            'specializations': [],
            'services_needed': [],
            'urgency_level': urgency,
            'search_radius_km': 50  # Default 50km radius
        }
        
        # Extract primary diagnosis
        primary_diagnosis = medical_profile.get('primary_diagnosis', '').lower()
        
        # Map conditions to specializations
        specialization_mapping = {
            'cardiac': ['cardiology', 'cardiovascular'],
            'heart': ['cardiology', 'cardiovascular'],
            'chest pain': ['cardiology', 'emergency medicine'],
            'neurological': ['neurology', 'neurosurgery'],
            'brain': ['neurology', 'neurosurgery'],
            'headache': ['neurology', 'general medicine'],
            'migraine': ['neurology', 'general medicine'],
            'diabetes': ['endocrinology', 'general medicine'],
            'thyroid': ['endocrinology'],
            'kidney': ['nephrology', 'urology'],
            'liver': ['gastroenterology', 'hepatology'],
            'skin': ['dermatology'],
            'bone': ['orthopedics', 'rheumatology'],
            'joint': ['orthopedics', 'rheumatology'],
            'mental health': ['psychiatry', 'psychology'],
            'anxiety': ['psychiatry', 'psychology'],
            'depression': ['psychiatry', 'psychology'],
            'respiratory': ['pulmonology', 'general medicine'],
            'lung': ['pulmonology', 'general medicine'],
            'pregnancy': ['obstetrics', 'gynecology'],
            'pediatric': ['pediatrics'],
            'eye': ['ophthalmology'],
            'ear': ['otolaryngology', 'ent']
        }
        
        # Determine required specializations
        for condition, specializations in specialization_mapping.items():
            if condition in primary_diagnosis:
                criteria['specializations'].extend(specializations)
        
        # Default to general medicine if no specific specialization
        if not criteria['specializations']:
            criteria['specializations'] = ['general medicine', 'family medicine']
        
        # Adjust search radius based on urgency
        if urgency == 'EMERGENCY':
            criteria['search_radius_km'] = 25
            criteria['services_needed'] = ['emergency_services', '24_hour_availability']
        elif urgency == 'HIGH':
            criteria['search_radius_km'] = 35
            criteria['services_needed'] = ['urgent_care', 'same_day_appointments']
        
        return criteria
    
    async def _find_doctors(self, location_coords: Dict, criteria: Dict, 
                          financial_capability: str, urgency: str) -> List[Dict]:
        """Find doctors using Google Places API and database"""
        doctors = []
        
        try:
            # Search using Google Places API
            if self.gmaps:
                places_results = await self._search_google_places(
                    location_coords, 
                    'doctor', 
                    criteria['specializations'],
                    criteria['search_radius_km']
                )
                
                for place in places_results:
                    doctor_data = await self._enrich_doctor_data(place, criteria, financial_capability)
                    if doctor_data:
                        doctors.append(doctor_data)
            
            # Add database doctors (if any)
            db_doctors = await self._search_database_doctors(location_coords, criteria, financial_capability)
            doctors.extend(db_doctors)
            
            # Remove duplicates and sort by relevance
            doctors = self._deduplicate_providers(doctors)
            doctors = sorted(doctors, key=lambda x: x.get('relevance_score', 0), reverse=True)
            
            return doctors[:20]  # Return top 20 doctors
            
        except Exception as e:
            logger.error(f"Doctor search failed: {e}")
            return []
    
    async def _find_hospitals(self, location_coords: Dict, criteria: Dict, 
                            financial_capability: str, urgency: str) -> List[Dict]:
        """Find hospitals using Google Places API and database"""
        hospitals = []
        
        try:
            # Search using Google Places API
            if self.gmaps:
                places_results = await self._search_google_places(
                    location_coords, 
                    'hospital', 
                    criteria.get('services_needed', []),
                    criteria['search_radius_km']
                )
                
                for place in places_results:
                    hospital_data = await self._enrich_hospital_data(place, criteria, financial_capability)
                    if hospital_data:
                        hospitals.append(hospital_data)
            
            # Add database hospitals
            db_hospitals = await self._search_database_hospitals(location_coords, criteria, financial_capability)
            hospitals.extend(db_hospitals)
            
            # Remove duplicates and sort
            hospitals = self._deduplicate_providers(hospitals)
            hospitals = sorted(hospitals, key=lambda x: x.get('relevance_score', 0), reverse=True)
            
            return hospitals[:15]  # Return top 15 hospitals
            
        except Exception as e:
            logger.error(f"Hospital search failed: {e}")
            return []
    
    async def _search_google_places(self, location_coords: Dict, place_type: str, 
                                  keywords: List[str], radius_km: int) -> List[Dict]:
        """Search Google Places API for healthcare providers"""
        try:
            if not self.gmaps:
                return []
            
            # Convert radius to meters
            radius_meters = radius_km * 1000
            
            # Search query
            query = f"{place_type} {' '.join(keywords[:2])}"  # Limit keywords
            
            # Perform search
            places_result = self.gmaps.places_nearby(
                location=(location_coords['latitude'], location_coords['longitude']),
                radius=radius_meters,
                type=place_type,
                keyword=query
            )
            
            return places_result.get('results', [])
            
        except Exception as e:
            logger.error(f"Google Places search failed: {e}")
            return []
    
    async def _enrich_doctor_data(self, place_data: Dict, criteria: Dict, financial_capability: str) -> Optional[Dict]:
        """Enrich doctor data from Google Places"""
        try:
            # Get place details
            place_details = self.gmaps.place(
                place_data['place_id'],
                fields=['name', 'formatted_address', 'formatted_phone_number', 
                       'rating', 'opening_hours', 'website', 'reviews']
            )['result']
            
            # Calculate distance
            distance = self._calculate_distance(
                (criteria.get('patient_lat'), criteria.get('patient_lng')),
                (place_data['geometry']['location']['lat'], 
                 place_data['geometry']['location']['lng'])
            )
            
            # Estimate consultation fee based on area and rating
            estimated_fee = self._estimate_consultation_fee(
                place_details.get('rating', 3.0),
                financial_capability,
                place_data.get('price_level', 2)
            )
            
            return {
                'provider_id': place_data['place_id'],
                'name': place_details.get('name', 'Unknown Doctor'),
                'address': place_details.get('formatted_address', ''),
                'phone': place_details.get('formatted_phone_number', ''),
                'rating': place_details.get('rating', 0),
                'distance_km': distance,
                'estimated_fee': estimated_fee,
                'specializations': self._extract_specializations(place_details),
                'availability': self._parse_opening_hours(place_details.get('opening_hours', {})),
                'reviews_summary': self._summarize_reviews(place_details.get('reviews', [])),
                'source': 'Google Places',
                'relevance_score': self._calculate_relevance_score(place_details, criteria, distance)
            }
            
        except Exception as e:
            logger.error(f"Doctor data enrichment failed: {e}")
            return None
    
    def _calculate_distance(self, coord1: Tuple, coord2: Tuple) -> float:
        """Calculate distance between two coordinates"""
        try:
            if None in coord1 or None in coord2:
                return float('inf')
            return geodesic(coord1, coord2).kilometers
        except:
            return float('inf')
    
    def _estimate_consultation_fee(self, rating: float, financial_capability: str, price_level: int) -> Dict:
        """Estimate consultation fees based on various factors"""
        base_fees = {
            'low': {'min': 200, 'max': 800},      # Budget-friendly
            'medium': {'min': 500, 'max': 1500},  # Moderate
            'high': {'min': 1000, 'max': 5000}    # Premium
        }
        
        fee_range = base_fees.get(financial_capability, base_fees['medium'])
        
        # Adjust based on rating and price level
        rating_multiplier = 1 + (rating - 3.0) * 0.2  # Higher rating = higher fee
        price_multiplier = 1 + (price_level - 2) * 0.3  # Higher price level = higher fee
        
        estimated_min = int(fee_range['min'] * rating_multiplier * price_multiplier)
        estimated_max = int(fee_range['max'] * rating_multiplier * price_multiplier)
        
        return {
            'currency': 'INR',
            'min_fee': estimated_min,
            'max_fee': estimated_max,
            'average_fee': (estimated_min + estimated_max) // 2
        }
    
    def _extract_specializations(self, place_details: Dict) -> List[str]:
        """Extract specializations from place details"""
        specializations = []
        
        name = place_details.get('name', '').lower()
        
        # Common specialization keywords
        spec_keywords = {
            'cardio': 'Cardiology',
            'neuro': 'Neurology',
            'ortho': 'Orthopedics',
            'derma': 'Dermatology',
            'pediatric': 'Pediatrics',
            'gynec': 'Gynecology',
            'ent': 'ENT',
            'eye': 'Ophthalmology',
            'dental': 'Dentistry',
            'psychiatr': 'Psychiatry',
            'general': 'General Medicine'
        }
        
        for keyword, specialization in spec_keywords.items():
            if keyword in name:
                specializations.append(specialization)
        
        return specializations if specializations else ['General Medicine']
    
    def _calculate_relevance_score(self, place_details: Dict, criteria: Dict, distance: float) -> float:
        """Calculate relevance score for provider"""
        score = 0.0
        
        # Distance score (closer is better)
        if distance <= 5:
            score += 0.4
        elif distance <= 15:
            score += 0.3
        elif distance <= 30:
            score += 0.2
        else:
            score += 0.1
        
        # Rating score
        rating = place_details.get('rating', 0)
        score += (rating / 5.0) * 0.3
        
        # Specialization match
        provider_specs = self._extract_specializations(place_details)
        required_specs = criteria.get('specializations', [])
        
        if any(spec.lower() in [req.lower() for req in required_specs] for spec in provider_specs):
            score += 0.3
        
        return min(score, 1.0)

# Global healthcare finder instance
healthcare_finder = HealthcareProviderFinder()