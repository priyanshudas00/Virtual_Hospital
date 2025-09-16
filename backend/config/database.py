import os
import logging
from pymongo import MongoClient
from datetime import datetime

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self):
        self.client = None
        self.db = None
        
    def connect(self):
        """Connect to MongoDB"""
        try:
            mongodb_uri = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/')
            self.client = MongoClient(mongodb_uri)
            self.db = self.client.mediscan_ai
            
            # Test connection
            self.client.admin.command('ping')
            logger.info("✅ MongoDB connected successfully")
            
            # Create indexes
            self.create_indexes()
            
        except Exception as e:
            logger.error(f"❌ MongoDB connection failed: {e}")
            raise
    
    def create_indexes(self):
        """Create optimized database indexes"""
        try:
            # User indexes
            self.db.users.create_index("email", unique=True)
            self.db.users.create_index("username", unique=True)
            
            # Interaction indexes
            self.db.interactions.create_index([("user_id", 1), ("created_at", -1)])
            self.db.interactions.create_index("session_id", unique=True)
            self.db.interactions.create_index("triage_level")
            
            # Medical reports indexes
            self.db.medical_reports.create_index([("user_id", 1), ("uploaded_at", -1)])
            self.db.medical_reports.create_index("report_type")
            self.db.medical_reports.create_index("ai_analysis.flags.severity")
            
            logger.info("✅ Database indexes created")
            
        except Exception as e:
            logger.error(f"❌ Index creation failed: {e}")
    
    def get_collection(self, name):
        """Get collection with error handling"""
        if not self.db:
            self.connect()
        return self.db[name]

# Global database manager
db_manager = DatabaseManager()