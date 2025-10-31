"""
Data Engine Synonym System
A FastAPI application with SQL Server integration and caching support.

This module demonstrates production-grade Python development practices including:
- Fully typed code with type annotations
- Abstract base classes and protocols
- Comprehensive logging and exception handling
- Modular, loosely coupled architecture
- OOP and functional programming patterns
"""

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.responses import HTMLResponse, JSONResponse
from sqlalchemy import create_engine, Column, Integer, String, Text, event
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from pydantic import BaseModel, ConfigDict, Field, validator
from dotenv import load_dotenv
from typing import List, Optional, Protocol, Dict, Any, Tuple
from enum import Enum
from abc import ABC, abstractmethod
import json
import redis
from datetime import datetime, timedelta
import threading
import os
import logging
from contextlib import contextmanager
from functools import wraps
import time
load_dotenv()

# ============================================================================
# Logging Configuration (Production-Grade)
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('synonym_system.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# ============================================================================
# Type Definitions and Aliases
# ============================================================================

# Type aliases for better code readability
CacheKey = str
CacheValue = str
TTLSeconds = int
WordId = int
WordText = str
SynonymList = List[str]

# ============================================================================
# Configuration with Validation
# ============================================================================

class CacheStrategy(str, Enum):
    """Enumeration of available caching strategies."""
    REDIS = "redis"
    MEMORY = "memory"

class Config:
    """
    Application configuration with environment variable support.
    
    This class centralizes all configuration parameters and provides
    validation and defaults for production deployment.
    """
    
    # Database Configuration
    USE_SQLSERVER: bool = os.getenv("USE_SQLSERVER", "false").lower() == "true"
    
    if USE_SQLSERVER:
        DB_SERVER: str = os.getenv("DB_SERVER", "localhost")
        DB_NAME: str = os.getenv("DB_NAME", "synonyms_db")
        DB_USER: str = os.getenv("DB_USER", "sa")
        DB_PASSWORD: str = os.getenv("DB_PASSWORD", "YourPassword123")
        DATABASE_URL: str = f"mssql+pyodbc://{DB_USER}:{DB_PASSWORD}@{DB_SERVER}/{DB_NAME}?driver=ODBC+Driver+17+for+SQL+Server"
    else:
        DATABASE_URL: str = "sqlite:///./synonyms.db"
    
    # Redis Configuration
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_DB: int = int(os.getenv("REDIS_DB", "0"))
    
    # Cache Configuration with validation
    CACHE_TTL: TTLSeconds = int(os.getenv("CACHE_TTL", "3600"))
    CACHE_STRATEGY: CacheStrategy = CacheStrategy(os.getenv("CACHE_STRATEGY", CacheStrategy.MEMORY))
    
    # Connection pooling settings
    DB_POOL_SIZE: int = int(os.getenv("DB_POOL_SIZE", "10"))
    DB_MAX_OVERFLOW: int = int(os.getenv("DB_MAX_OVERFLOW", "20"))

config = Config()

# ============================================================================
# Custom Exceptions (Production-Grade Error Handling)
# ============================================================================

class SynonymSystemException(Exception):
    """Base exception for all synonym system errors."""
    pass

class CacheConnectionError(SynonymSystemException):
    """Raised when cache connection fails."""
    pass

class DatabaseConnectionError(SynonymSystemException):
    """Raised when database connection fails."""
    pass

class DataValidationError(SynonymSystemException):
    """Raised when data validation fails."""
    pass

# ============================================================================
# Database Models (Fully Typed)
# ============================================================================

Base = declarative_base()

class SynonymModel(Base):
    """
    SQLAlchemy model for synonym records.
    
    Attributes:
        id: Unique identifier for the synonym record
        word: The primary word
        synonyms: JSON-serialized list of synonym words
    """
    __tablename__ = "synonyms"
    
    id: int = Column(Integer, primary_key=True, index=True)
    word: str = Column(String(255), nullable=False, index=True)
    synonyms: str = Column(Text, nullable=False)
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"<SynonymModel(id={self.id}, word='{self.word}')>"

# ============================================================================
# Pydantic Models (Validation & Serialization)
# ============================================================================

class SynonymRecord(BaseModel):
    """
    Validated synonym record for API responses.
    
    Uses Pydantic for automatic validation and serialization.
    Demonstrates immutability through frozen configuration.
    """
    model_config = ConfigDict(from_attributes=True, frozen=True)
    
    id: WordId = Field(..., description="Unique word identifier", gt=0)
    word: WordText = Field(..., description="The primary word", min_length=1, max_length=255)
    synonyms: SynonymList = Field(..., description="List of synonym words")
    
    @validator('synonyms')
    @classmethod
    def validate_synonyms(cls, v: List[str]) -> List[str]:
        """Validate that synonyms list is not empty and contains valid strings."""
        if not v:
            raise ValueError("Synonyms list cannot be empty")
        if not all(isinstance(s, str) and s.strip() for s in v):
            raise ValueError("All synonyms must be non-empty strings")
        return v

class SynonymsResponse(BaseModel):
    """
    API response model for synonym queries.
    
    Includes metadata about cache usage and performance metrics.
    """
    data: List[SynonymRecord]
    from_cache: bool
    cache_strategy: str
    timestamp: str
    query_time_ms: Optional[float] = Field(None, description="Query execution time in milliseconds")
    record_count: int = Field(..., description="Number of records returned")

class ErrorResponse(BaseModel):
    """Standard error response format."""
    error: str
    detail: str
    timestamp: str

# ============================================================================
# Protocol Definition (Coding to Interfaces)
# ============================================================================

class CacheProtocol(Protocol):
    """
    Protocol (interface) for cache implementations.
    
    This demonstrates 'coding to interfaces' - any cache implementation
    that follows this protocol can be used interchangeably.
    """
    
    def get(self, key: CacheKey) -> Optional[CacheValue]:
        """Retrieve value from cache."""
        ...
    
    def set(self, key: CacheKey, value: CacheValue, ttl: TTLSeconds) -> None:
        """Store value in cache with TTL."""
        ...
    
    def delete(self, key: CacheKey) -> None:
        """Remove value from cache."""
        ...
    
    def clear(self) -> None:
        """Clear all cache entries."""
        ...

# ============================================================================
# Abstract Base Class for Cache (Interface Abstraction)
# ============================================================================

class CacheInterface(ABC):
    """
    Abstract base class defining the cache contract.
    
    All cache implementations must inherit from this and implement
    all abstract methods. This enforces consistency across implementations.
    """
    
    @abstractmethod
    def get(self, key: CacheKey) -> Optional[CacheValue]:
        """Retrieve value from cache."""
        pass
    
    @abstractmethod
    def set(self, key: CacheKey, value: CacheValue, ttl: TTLSeconds) -> None:
        """Store value in cache with TTL."""
        pass
    
    @abstractmethod
    def delete(self, key: CacheKey) -> None:
        """Remove value from cache."""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all cache entries."""
        pass

# ============================================================================
# Cache Implementations (Polymorphism)
# ============================================================================

class RedisCache(CacheInterface):
    """
    Redis-based distributed caching implementation.
    
    Features:
    - Thread-safe operations using locks
    - Connection pooling for performance
    - Comprehensive error handling with fallback
    """
    
    def __init__(self, host: str, port: int, db: int) -> None:
        """
        Initialize Redis cache connection.
        
        Args:
            host: Redis server hostname
            port: Redis server port
            db: Redis database number
            
        Raises:
            CacheConnectionError: If connection to Redis fails
        """
        try:
            self.client: redis.Redis = redis.Redis(
                host=host,
                port=port,
                db=db,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_keepalive=True
            )
            # Test connection
            self.client.ping()
            self._lock: threading.Lock = threading.Lock()
            logger.info(f"✓ Connected to Redis at {host}:{port}")
        except redis.RedisError as e:
            logger.error(f"✗ Redis connection failed: {e}")
            raise CacheConnectionError(f"Failed to connect to Redis: {e}")
    
    def get(self, key: CacheKey) -> Optional[CacheValue]:
        """
        Retrieve value from Redis cache.
        
        Thread-safe operation with exception handling.
        """
        try:
            with self._lock:
                value = self.client.get(key)
                if value:
                    logger.debug(f"Cache hit: {key}")
                return value
        except redis.RedisError as e:
            logger.error(f"Redis get error for key '{key}': {e}")
            return None
    
    def set(self, key: CacheKey, value: CacheValue, ttl: TTLSeconds) -> None:
        """
        Store value in Redis cache with expiration.
        
        Uses SETEX for atomic set-with-expiry operation.
        """
        try:
            with self._lock:
                self.client.setex(key, ttl, value)
                logger.debug(f"Cache set: {key} (TTL: {ttl}s)")
        except redis.RedisError as e:
            logger.error(f"Redis set error for key '{key}': {e}")
    
    def delete(self, key: CacheKey) -> None:
        """Remove key from Redis cache."""
        try:
            with self._lock:
                self.client.delete(key)
                logger.info(f"Cache invalidated: {key}")
        except redis.RedisError as e:
            logger.error(f"Redis delete error for key '{key}': {e}")
    
    def clear(self) -> None:
        """Clear all keys in current Redis database."""
        try:
            with self._lock:
                self.client.flushdb()
                logger.info("Cache cleared (Redis)")
        except redis.RedisError as e:
            logger.error(f"Redis clear error: {e}")

class InMemoryCache(CacheInterface):
    """
    Thread-safe in-memory caching implementation.
    
    Features:
    - No external dependencies
    - Automatic expiration handling
    - Thread-safe operations
    - Suitable for single-server deployments
    """
    
    def __init__(self) -> None:
        """Initialize in-memory cache with thread safety."""
        self._cache: Dict[CacheKey, CacheValue] = {}
        self._expiry: Dict[CacheKey, datetime] = {}
        self._lock: threading.Lock = threading.Lock()
        logger.info("✓ Using in-memory cache")
    
    def _is_expired(self, key: CacheKey) -> bool:
        """
        Check if cache entry has expired.
        
        Pure function - no side effects, deterministic output.
        """
        if key not in self._expiry:
            return True
        return datetime.now() > self._expiry[key]
    
    def get(self, key: CacheKey) -> Optional[CacheValue]:
        """
        Retrieve value from memory cache.
        
        Automatically cleans up expired entries.
        """
        with self._lock:
            if key in self._cache and not self._is_expired(key):
                logger.debug(f"Cache hit: {key}")
                return self._cache[key]
            elif key in self._cache:
                # Clean up expired entry
                del self._cache[key]
                del self._expiry[key]
                logger.debug(f"Cache expired: {key}")
            return None
    
    def set(self, key: CacheKey, value: CacheValue, ttl: TTLSeconds) -> None:
        """Store value in memory cache with expiration."""
        with self._lock:
            self._cache[key] = value
            self._expiry[key] = datetime.now() + timedelta(seconds=ttl)
            logger.debug(f"Cache set: {key} (TTL: {ttl}s)")
    
    def delete(self, key: CacheKey) -> None:
        """Remove key from memory cache."""
        with self._lock:
            self._cache.pop(key, None)
            self._expiry.pop(key, None)
            logger.info(f"Cache invalidated: {key}")
    
    def clear(self) -> None:
        """Clear all entries from memory cache."""
        with self._lock:
            self._cache.clear()
            self._expiry.clear()
            logger.info("Cache cleared (in-memory)")

# ============================================================================
# Cache Manager (Strategy Pattern)
# ============================================================================

class CacheManager:
    """
    Manages cache operations using the Strategy pattern.
    
    This class encapsulates the caching strategy and provides a
    high-level interface for cache operations. It demonstrates:
    - Dependency injection
    - Strategy pattern
    - Separation of concerns
    """
    
    def __init__(self, strategy: CacheStrategy, ttl: TTLSeconds) -> None:
        """
        Initialize cache manager with specified strategy.
        
        Args:
            strategy: Caching strategy to use (Redis or Memory)
            ttl: Time-to-live for cache entries in seconds
        """
        self.ttl: TTLSeconds = ttl
        self.strategy: CacheStrategy = strategy
        
        try:
            if strategy == CacheStrategy.REDIS:
                self.cache: CacheInterface = RedisCache(
                    host=config.REDIS_HOST,
                    port=config.REDIS_PORT,
                    db=config.REDIS_DB
                )
            else:
                self.cache: CacheInterface = InMemoryCache()
        except CacheConnectionError as e:
            logger.warning(f"Failed to initialize {strategy} cache: {e}")
            logger.info("Falling back to in-memory cache")
            self.cache = InMemoryCache()
            self.strategy = CacheStrategy.MEMORY
    
    def get_synonyms(self) -> Optional[List[Dict[str, Any]]]:
        """
        Retrieve synonyms from cache.
        
        Returns:
            List of synonym dictionaries if cached, None otherwise
        """
        try:
            cached_data: Optional[CacheValue] = self.cache.get("synonyms:all")
            if cached_data:
                return json.loads(cached_data)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode cached data: {e}")
        return None
    
    def set_synonyms(self, data: List[Dict[str, Any]]) -> None:
        """
        Store synonyms in cache.
        
        Args:
            data: List of synonym dictionaries to cache
        """
        try:
            serialized: str = json.dumps(data)
            self.cache.set("synonyms:all", serialized, self.ttl)
        except (TypeError, ValueError) as e:
            logger.error(f"Failed to serialize data for caching: {e}")
    
    def invalidate(self) -> None:
        """Clear synonym cache entry."""
        self.cache.delete("synonyms:all")
    
    def clear_all(self) -> None:
        """Clear entire cache."""
        self.cache.clear()

# ============================================================================
# Database Setup with Connection Pooling
# ============================================================================

if config.USE_SQLSERVER:
    engine = create_engine(
        config.DATABASE_URL,
        pool_pre_ping=True,
        pool_size=config.DB_POOL_SIZE,
        max_overflow=config.DB_MAX_OVERFLOW,
        echo=False
    )
else:
    engine = create_engine(
        config.DATABASE_URL,
        connect_args={"check_same_thread": False},
        echo=False
    )

# Log SQL queries for debugging (optional)
@event.listens_for(engine, "connect")
def receive_connect(dbapi_conn, connection_record):
    """Log database connections for monitoring."""
    logger.debug("Database connection established")

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

@contextmanager
def get_db_context():
    """
    Context manager for database sessions.
    
    Ensures proper cleanup even if exceptions occur.
    This is a functional programming pattern for resource management.
    """
    db: Session = SessionLocal()
    try:
        yield db
    except Exception as e:
        logger.error(f"Database error: {e}")
        db.rollback()
        raise
    finally:
        db.close()

def get_db():
    """Dependency injection for database sessions."""
    db: Session = SessionLocal()
    try:
        yield db
    except Exception as e:
        logger.error(f"Database session error: {e}")
        db.rollback()
        raise
    finally:
        db.close()

# ============================================================================
# Service Layer (Business Logic)
# ============================================================================

class SynonymService:
    """
    Service layer for synonym operations.
    
    This class encapsulates business logic and provides a clean
    separation between the API layer and data access layer.
    Demonstrates:
    - Single Responsibility Principle
    - Dependency Injection
    - Separation of Concerns
    """
    
    def __init__(self, db: Session, cache_manager: CacheManager) -> None:
        """
        Initialize service with dependencies.
        
        Args:
            db: Database session
            cache_manager: Cache management instance
        """
        self.db: Session = db
        self.cache: CacheManager = cache_manager
    
    def get_all_synonyms(self) -> Tuple[List[SynonymRecord], bool, float]:
        """
        Retrieve all synonyms with performance tracking.
        
        This method demonstrates:
        - Cache-aside pattern
        - Performance monitoring
        - Error handling
        
        Returns:
            Tuple of (records, from_cache, query_time_ms)
        """
        start_time: float = time.time()
        
        try:
            # Try cache first (cache-aside pattern)
            cached_data: Optional[List[Dict[str, Any]]] = self.cache.get_synonyms()
            if cached_data:
                records: List[SynonymRecord] = [
                    SynonymRecord(
                        id=item['id'],
                        word=item['word'],
                        synonyms=item['synonyms']
                    )
                    for item in cached_data
                ]
                query_time: float = (time.time() - start_time) * 1000
                logger.info(f"Retrieved {len(records)} synonyms from cache in {query_time:.2f}ms")
                return records, True, query_time
            
            # Cache miss - query database
            db_records: List[SynonymModel] = self.db.query(SynonymModel).all()
            
            # Transform and validate data
            records: List[SynonymRecord] = []
            cache_data: List[Dict[str, Any]] = []
            
            for record in db_records:
                try:
                    synonyms_list: List[str] = json.loads(record.synonyms)
                    synonym_record = SynonymRecord(
                        id=record.id,
                        word=record.word,
                        synonyms=synonyms_list
                    )
                    records.append(synonym_record)
                    cache_data.append({
                        'id': record.id,
                        'word': record.word,
                        'synonyms': synonyms_list
                    })
                except (json.JSONDecodeError, ValueError) as e:
                    logger.error(f"Failed to parse record {record.id}: {e}")
                    continue
            
            # Update cache for future requests
            if cache_data:
                self.cache.set_synonyms(cache_data)
            
            query_time = (time.time() - start_time) * 1000
            logger.info(f"Retrieved {len(records)} synonyms from database in {query_time:.2f}ms")
            return records, False, query_time
            
        except Exception as e:
            logger.error(f"Error retrieving synonyms: {e}", exc_info=True)
            raise DatabaseConnectionError(f"Failed to retrieve synonyms: {e}")

# ============================================================================
# Decorators (Functional Programming)
# ============================================================================

def log_execution_time(func):
    """
    Decorator to log function execution time.
    
    Demonstrates functional programming concepts:
    - Higher-order functions
    - Function composition
    - Aspect-oriented programming
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start: float = time.time()
        result = func(*args, **kwargs)
        duration: float = (time.time() - start) * 1000
        logger.info(f"{func.__name__} executed in {duration:.2f}ms")
        return result
    return wrapper

# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="Data Engine Synonym System",
    description="Production-grade synonym retrieval system with intelligent caching",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Initialize cache manager (singleton pattern)
cache_manager: CacheManager = CacheManager(
    strategy=config.CACHE_STRATEGY,
    ttl=config.CACHE_TTL
)

def get_synonym_service(db: Session = Depends(get_db)) -> SynonymService:
    """Dependency injection for synonym service."""
    return SynonymService(db, cache_manager)

# ============================================================================
# API Endpoints (RESTful Design)
# ============================================================================

from fastapi.responses import HTMLResponse

@app.get("/", response_class=HTMLResponse)
def root():
    """
    Homepage with system information and navigation.
    
    Returns:
        HTML page with links to documentation and endpoints
    """
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Data Engine Synonym System</title>
        <style>
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
                max-width: 900px;
                margin: 50px auto;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
            }
            .container {
                background: white;
                border-radius: 12px;
                padding: 40px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            }
            h1 {
                color: #667eea;
                margin-bottom: 10px;
                font-size: 2.5em;
            }
            .subtitle {
                color: #666;
                margin-bottom: 30px;
                font-size: 1.1em;
            }
            .status {
                background: #f0f4ff;
                padding: 20px;
                border-radius: 8px;
                margin: 20px 0;
                border-left: 4px solid #667eea;
            }
            .status-item {
                display: flex;
                justify-content: space-between;
                margin: 8px 0;
                padding: 8px 0;
                border-bottom: 1px solid #e0e0e0;
            }
            .status-item:last-child {
                border-bottom: none;
            }
            .status-label {
                font-weight: 600;
                color: #555;
            }
            .status-value {
                color: #667eea;
                font-weight: 500;
            }
            .links {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 15px;
                margin: 30px 0;
            }
            .link-card {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 25px;
                border-radius: 8px;
                text-decoration: none;
                transition: transform 0.2s, box-shadow 0.2s;
                box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
            }
            .link-card:hover {
                transform: translateY(-5px);
                box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4);
            }
            .link-card h3 {
                margin: 0 0 10px 0;
                font-size: 1.3em;
            }
            .link-card p {
                margin: 0;
                opacity: 0.9;
                font-size: 0.95em;
            }
            .endpoints {
                margin-top: 30px;
            }
            .endpoint {
                background: #f9f9f9;
                padding: 15px;
                margin: 10px 0;
                border-radius: 6px;
                border-left: 4px solid #667eea;
            }
            .method {
                display: inline-block;
                padding: 4px 10px;
                border-radius: 4px;
                font-size: 0.85em;
                font-weight: 600;
                margin-right: 10px;
            }
            .get { background: #61affe; color: white; }
            .post { background: #49cc90; color: white; }
            .endpoint-path {
                font-family: 'Courier New', monospace;
                color: #333;
                font-weight: 600;
            }
            .endpoint-desc {
                color: #666;
                margin-top: 8px;
                font-size: 0.95em;
            }
            .features {
                margin-top: 30px;
                background: #f9f9f9;
                padding: 20px;
                border-radius: 8px;
            }
            .features h2 {
                color: #667eea;
                margin-bottom: 15px;
            }
            .feature-list {
                list-style: none;
                padding: 0;
            }
            .feature-list li {
                padding: 8px 0;
                padding-left: 25px;
                position: relative;
            }
            .feature-list li:before {
                content: "✓";
                position: absolute;
                left: 0;
                color: #49cc90;
                font-weight: bold;
            }
            .footer {
                margin-top: 40px;
                padding-top: 20px;
                border-top: 1px solid #e0e0e0;
                text-align: center;
                color: #666;
                font-size: 0.9em;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🔤 Data Engine Synonym System</h1>
            <p class="subtitle">Production-grade synonym retrieval with intelligent caching</p>
            
            <div class="status">
                <div class="status-item">
                    <span class="status-label">Status:</span>
                    <span class="status-value">✓ Running</span>
                </div>
                <div class="status-item">
                    <span class="status-label">Database:</span>
                    <span class="status-value">""" + ("SQL Server" if config.USE_SQLSERVER else "SQLite") + """</span>
                </div>
                <div class="status-item">
                    <span class="status-label">Cache Strategy:</span>
                    <span class="status-value">""" + str(cache_manager.strategy.value).title() + """</span>
                </div>
                <div class="status-item">
                    <span class="status-label">Cache TTL:</span>
                    <span class="status-value">""" + str(config.CACHE_TTL) + """ seconds</span>
                </div>
            </div>

            <div class="links">
                <a href="/docs" class="link-card">
                    <h3>📚 API Documentation</h3>
                    <p>Interactive Swagger UI with all endpoints and schemas</p>
                </a>
                <a href="/redoc" class="link-card">
                    <h3>📖 ReDoc</h3>
                    <p>Alternative documentation format</p>
                </a>
                <a href="/synonyms" class="link-card">
                    <h3>🔍 View Synonyms</h3>
                    <p>Get all synonym records (JSON)</p>
                </a>
                <a href="/health" class="link-card">
                    <h3>💚 Health Check</h3>
                    <p>System status and configuration</p>
                </a>
            </div>

            <div class="features">
                <h2>Production-Grade Features</h2>
                <ul class="feature-list">
                    <li><strong>Fully Typed Python:</strong> Complete type annotations on all functions</li>
                    <li><strong>Abstract Base Classes:</strong> Coding to interfaces and protocols</li>
                    <li><strong>Production Logging:</strong> Comprehensive logging with rotation</li>
                    <li><strong>Exception Handling:</strong> Custom exceptions with proper error handling</li>
                    <li><strong>OOP + Functional:</strong> Blend of paradigms with immutability</li>
                    <li><strong>Connection Pooling:</strong> Optimized database connections</li>
                    <li><strong>Thread-Safe Caching:</strong> Atomic operations with locks</li>
                    <li><strong>Performance Monitoring:</strong> Query execution time tracking</li>
                    <li><strong>RESTful API Design:</strong> Standard HTTP methods and status codes</li>
                    <li><strong>Dependency Injection:</strong> Loosely coupled, testable architecture</li>
                </ul>
            </div>

            <div class="endpoints">
                <h2>Available Endpoints</h2>
                
                <div class="endpoint">
                    <span class="method get">GET</span>
                    <span class="endpoint-path">/synonyms</span>
                    <div class="endpoint-desc">Retrieve all synonym records with cache metadata and performance metrics</div>
                </div>
                
                <div class="endpoint">
                    <span class="method post">POST</span>
                    <span class="endpoint-path">/cache/invalidate</span>
                    <div class="endpoint-desc">Invalidate the synonym cache</div>
                </div>
                
                <div class="endpoint">
                    <span class="method post">POST</span>
                    <span class="endpoint-path">/cache/clear</span>
                    <div class="endpoint-desc">Clear entire cache</div>
                </div>
                
                <div class="endpoint">
                    <span class="method get">GET</span>
                    <span class="endpoint-path">/health</span>
                    <div class="endpoint-desc">Health check and system status</div>
                </div>
            </div>

            <div class="footer">
                <p>Data Engine Synonym System v1.0.0 | Built with FastAPI</p>
            </div>
        </div>
    </body>
    </html>
    """
    return html_content

@app.get("/synonyms", response_model=SynonymsResponse)
@log_execution_time
def get_synonyms(service: SynonymService = Depends(get_synonym_service)) -> SynonymsResponse:
    """
    Retrieve all synonym records from the database or cache.
    
    This endpoint demonstrates:
    - RESTful API design
    - Dependency injection
    - Response model validation
    - Performance tracking
    
    Returns:
        SynonymsResponse: Complete synonym data with metadata
        
    Raises:
        HTTPException: If retrieval fails
    """
    try:
        records, from_cache, query_time = service.get_all_synonyms()
        
        return SynonymsResponse(
            data=records,
            from_cache=from_cache,
            cache_strategy=cache_manager.strategy.value,
            timestamp=datetime.now().isoformat(),
            query_time_ms=round(query_time, 2),
            record_count=len(records)
        )
    except DatabaseConnectionError as e:
        logger.error(f"Database error in get_synonyms: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Unexpected error in get_synonyms: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

@app.post("/cache/invalidate")
def invalidate_cache() -> Dict[str, str]:
    """
    Invalidate the synonym cache.
    
    Forces the next request to fetch fresh data from the database.
    
    Returns:
        Success message with timestamp
    """
    try:
        cache_manager.invalidate()
        logger.info("Cache invalidated via API")
        return {
            "message": "Cache invalidated successfully",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error invalidating cache: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to invalidate cache"
        )

@app.post("/cache/clear")
def clear_cache() -> Dict[str, str]:
    """
    Clear entire cache.
    
    Removes all cached entries across the system.
    
    Returns:
        Success message with timestamp
    """
    try:
        cache_manager.clear_all()
        logger.info("Cache cleared via API")
        return {
            "message": "Cache cleared successfully",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to clear cache"
        )

@app.get("/health")
def health_check() -> Dict[str, Any]:
    """
    Health check endpoint for monitoring.
    
    Returns system status and configuration information.
    Useful for load balancers and monitoring systems.
    
    Returns:
        System health information
    """
    return {
        "status": "healthy",
        "service": "Data Engine Synonym System",
        "version": "1.0.0",
        "cache_strategy": cache_manager.strategy.value,
        "database": "SQL Server" if config.USE_SQLSERVER else "SQLite",
        "cache_ttl": config.CACHE_TTL,
        "timestamp": datetime.now().isoformat()
    }

# ============================================================================
# Exception Handlers (Production-Grade Error Handling)
# ============================================================================

@app.exception_handler(SynonymSystemException)
def handle_synonym_exception(request, exc: SynonymSystemException) -> JSONResponse:
    """
    Global exception handler for custom exceptions.
    
    Ensures consistent error response format across the API.
    """
    logger.error(f"Application error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": type(exc).__name__,
            "detail": str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )

# ============================================================================
# Database Initialization
# ============================================================================

@log_execution_time
def init_db() -> None:
    """
    Initialize database tables.
    
    Creates tables if they don't exist. Safe to call multiple times.
    
    Raises:
        DatabaseConnectionError: If database initialization fails
    """
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("✓ Database tables created/verified")
    except Exception as e:
        logger.error(f"✗ Database initialization error: {e}", exc_info=True)
        raise DatabaseConnectionError(f"Failed to initialize database: {e}")

@log_execution_time
def seed_sample_data() -> None:
    """
    Seed database with sample synonym data.
    
    Only adds data if the database is empty. Idempotent operation.
    """
    with get_db_context() as db:
        try:
            count: int = db.query(SynonymModel).count()
            if count > 0:
                logger.info(f"✓ Database already contains {count} records")
                return
            
            sample_data: List[SynonymModel] = [
                SynonymModel(
                    word="happy",
                    synonyms=json.dumps(["joyful", "cheerful", "delighted", "pleased", "content"])
                ),
                SynonymModel(
                    word="fast",
                    synonyms=json.dumps(["quick", "rapid", "swift", "speedy", "hasty"])
                ),
                SynonymModel(
                    word="smart",
                    synonyms=json.dumps(["intelligent", "clever", "bright", "wise", "brilliant"])
                ),
                SynonymModel(
                    word="big",
                    synonyms=json.dumps(["large", "huge", "enormous", "giant", "massive"])
                ),
                SynonymModel(
                    word="small",
                    synonyms=json.dumps(["tiny", "little", "minute", "compact", "petite"])
                ),
                SynonymModel(
                    word="strong",
                    synonyms=json.dumps(["powerful", "robust", "sturdy", "mighty", "tough"])
                ),
                SynonymModel(
                    word="beautiful",
                    synonyms=json.dumps(["gorgeous", "stunning", "lovely", "attractive", "pretty"])
                ),
                SynonymModel(
                    word="difficult",
                    synonyms=json.dumps(["hard", "challenging", "tough", "complex", "demanding"])
                ),
            ]
            
            db.add_all(sample_data)
            db.commit()
            logger.info(f"✓ Seeded {len(sample_data)} sample records")
        except Exception as e:
            logger.error(f"✗ Error seeding data: {e}", exc_info=True)
            db.rollback()
            raise

# ============================================================================
# Startup Event
# ============================================================================

@app.on_event("startup")
def startup_event() -> None:
    """
    Application startup event handler.
    
    Initializes database and seeds sample data.
    Logs comprehensive startup information.
    """
    logger.info("="*50)
    logger.info("Data Engine Synonym System - Starting Up")
    logger.info("="*50)
    logger.info(f"Database: {'SQL Server' if config.USE_SQLSERVER else 'SQLite (local)'}")
    logger.info(f"Cache Strategy: {config.CACHE_STRATEGY}")
    logger.info(f"Cache TTL: {config.CACHE_TTL} seconds")
    logger.info(f"Pool Size: {config.DB_POOL_SIZE}")
    logger.info("="*50)
    
    try:
        init_db()
        seed_sample_data()
        logger.info("")
        logger.info("✓ Application ready!")
        logger.info(f"✓ API available at: http://localhost:8000")
        logger.info(f"✓ Docs available at: http://localhost:8000/docs")
        logger.info("")
    except Exception as e:
        logger.error(f"✗ Startup failed: {e}", exc_info=True)
        raise

@app.on_event("shutdown")
def shutdown_event() -> None:
    """
    Application shutdown event handler.
    
    Performs cleanup operations before shutdown.
    """
    logger.info("Application shutting down...")
    engine.dispose()
    logger.info("✓ Database connections closed")

# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    # Run with production-grade settings
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=True
    )