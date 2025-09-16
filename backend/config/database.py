import os
import logging
from pymongo import MongoClient
from motor.motor_asyncio import AsyncIOMotorClient
import redis
from datetime import datetime

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self):
        self.mongo_client = None
        self.async_mongo_client = None
        self.redis_client = None
        self.db = None
        self.async_db = None
        
    def connect_mongodb(self):
        """Connect to MongoDB with production settings"""
        try:
            mongodb_uri = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/')
            self.mongo_client = MongoClient(
                mongodb_uri,
                maxPoolSize=50,
                minPoolSize=5,
                maxIdleTimeMS=30000,
                serverSelectionTimeoutMS=5000,
                socketTimeoutMS=20000
            )
            self.db = self.mongo_client.virtual_hospital
            
            # Test connection
            self.mongo_client.admin.command('ping')
            logger.info("✅ MongoDB connected successfully")
            
            # Create indexes for performance
            self.create_indexes()
            
        except Exception as e:
            logger.error(f"❌ MongoDB connection failed: {e}")
            raise
    
    def connect_async_mongodb(self):
        """Connect to MongoDB with async support"""
        try:
            mongodb_uri = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/')
            self.async_mongo_client = AsyncIOMotorClient(mongodb_uri)
            self.async_db = self.async_mongo_client.virtual_hospital
            logger.info("✅ Async MongoDB connected successfully")
            
        except Exception as e:
            logger.error(f"❌ Async MongoDB connection failed: {e}")
            raise
    
    def connect_redis(self):
        """Connect to Redis for caching and session management"""
        try:
            redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
            self.redis_client = redis.from_url(
                redis_url, 
                decode_responses=True,
                max_connections=20,
                retry_on_timeout=True
            )
            
            # Test connection
            self.redis_client.ping()
            logger.info("✅ Redis connected successfully")
            
        except Exception as e:
            logger.warning(f"⚠️ Redis connection failed: {e}")
            # Redis is optional for basic functionality
    
    def create_indexes(self):
        """Create optimized database indexes"""
        try:
            # User indexes
            self.db.users.create_index("email", unique=True)
            self.db.users.create_index("username", unique=True)
            self.db.users.create_index("profile.personal_info.phone")
            
            # Interaction indexes for fast retrieval
            self.db.interactions.create_index([("user_id", 1), ("started_at", -1)])
            self.db.interactions.create_index("status")
            self.db.interactions.create_index("ai_assessment.triage_level")
            self.db.interactions.create_index("session_id", unique=True)
            
            # Medical reports indexes
            self.db.medical_reports.create_index([("user_id", 1), ("upload_metadata.upload_timestamp", -1)])
            self.db.medical_reports.create_index("report_classification.type")
            self.db.medical_reports.create_index("ai_analysis.flags.severity")
            self.db.medical_reports.create_index("radiologist_review.reviewed")
            self.db.medical_reports.create_index("upload_metadata.file_hash")
            
            # Healthcare provider geospatial indexes
            self.db.doctors.create_index([("practice_info.address.coordinates", "2dsphere")])
            self.db.hospitals.create_index([("location.address.coordinates", "2dsphere")])
            self.db.doctors.create_index("profile.specializations")
            self.db.hospitals.create_index("services.emergency_services")
            self.db.doctors.create_index("verification.verified")
            
            # Patient history and timeline indexes
            self.db.patient_timeline.create_index([("user_id", 1), ("timestamp", -1)])
            self.db.patient_timeline.create_index("event_type")
            
            logger.info("✅ Database indexes created successfully")
            
        except Exception as e:
            logger.error(f"❌ Index creation failed: {e}")
    
    def get_collection(self, collection_name):
        """Get MongoDB collection with error handling"""
        if not self.db:
            self.connect_mongodb()
        return self.db[collection_name]
    
    def cache_set(self, key, value, expiry=3600):
        """Set cache value with error handling"""
        if self.redis_client:
            try:
                import json
                self.redis_client.setex(key, expiry, json.dumps(value))
            except Exception as e:
                logger.warning(f"Cache set failed: {e}")
    
    def cache_get(self, key):
        """Get cache value with error handling"""
        if self.redis_client:
            try:
                import json
                value = self.redis_client.get(key)
                return json.loads(value) if value else None
            except Exception as e:
                logger.warning(f"Cache get failed: {e}")
        return None
    
    def health_check(self):
        """Comprehensive health check for all connections"""
        status = {
            'mongodb': False,
            'redis': False,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        try:
            self.mongo_client.admin.command('ping')
            status['mongodb'] = True
        except:
            pass
        
        try:
            if self.redis_client:
                self.redis_client.ping()
                status['redis'] = True
        except:
            pass
        
        return status

# Global database manager instance
db_manager = DatabaseManager()