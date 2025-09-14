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
        """Connect to MongoDB"""
        try:
            mongodb_uri = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/')
            self.mongo_client = MongoClient(mongodb_uri)
            self.db = self.mongo_client.virtual_hospital
            
            # Test connection
            self.mongo_client.admin.command('ping')
            logger.info("✅ MongoDB connected successfully")
            
            # Create indexes
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
        """Connect to Redis for caching"""
        try:
            redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
            self.redis_client = redis.from_url(redis_url, decode_responses=True)
            
            # Test connection
            self.redis_client.ping()
            logger.info("✅ Redis connected successfully")
            
        except Exception as e:
            logger.warning(f"⚠️ Redis connection failed: {e}")
            # Redis is optional, continue without it
    
    def create_indexes(self):
        """Create database indexes for performance"""
        try:
            # User indexes
            self.db.users.create_index("email", unique=True)
            self.db.users.create_index("user_id")
            
            # Patient data indexes
            self.db.intake_forms.create_index("user_id")
            self.db.intake_forms.create_index("created_at")
            
            # AI reports indexes
            self.db.ai_reports.create_index("user_id")
            self.db.ai_reports.create_index("report_type")
            self.db.ai_reports.create_index("created_at")
            
            # Medical uploads indexes
            self.db.medical_uploads.create_index("user_id")
            self.db.medical_uploads.create_index("upload_type")
            
            # Healthcare provider indexes
            self.db.doctors.create_index([("location", "2dsphere")])
            self.db.hospitals.create_index([("location", "2dsphere")])
            self.db.doctors.create_index("specialization")
            self.db.hospitals.create_index("emergency_services")
            
            logger.info("✅ Database indexes created successfully")
            
        except Exception as e:
            logger.error(f"❌ Index creation failed: {e}")
    
    def get_collection(self, collection_name):
        """Get MongoDB collection"""
        if not self.db:
            self.connect_mongodb()
        return self.db[collection_name]
    
    def cache_set(self, key, value, expiry=3600):
        """Set cache value"""
        if self.redis_client:
            try:
                import json
                self.redis_client.setex(key, expiry, json.dumps(value))
            except Exception as e:
                logger.warning(f"Cache set failed: {e}")
    
    def cache_get(self, key):
        """Get cache value"""
        if self.redis_client:
            try:
                import json
                value = self.redis_client.get(key)
                return json.loads(value) if value else None
            except Exception as e:
                logger.warning(f"Cache get failed: {e}")
        return None

# Global database manager instance
db_manager = DatabaseManager()