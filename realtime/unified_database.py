#!/usr/bin/env python3
"""
Unified Database System
========================

Comprehensive database integration system that handles:
1. SQLite for structured data (provenance, experiments, decisions)
2. JSON for flexible data storage (vocalization database, phrase libraries)
3. Cloud synchronization for distributed systems
4. Caching layer for performance optimization
5. Backup and recovery mechanisms

This module integrates all commented database features from:
- data_logging.py (SQLite provenance logging)
- advanced_technical_enhancements.py (cloud sync, concatenative database)
- dual_path_analyzer.py (phrase database loading)
- populate_from_database.py (database population)

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import sqlite3
import json
import pickle
import threading
import time
import uuid
import hashlib
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict, field
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import logging
import asyncio
import aiohttp
from queue import Queue
import shutil
from abc import ABC, abstractmethod
import numpy as np

# ============================================================================
# 1. DATABASE CONFIGURATION
# ============================================================================

@dataclass
class DatabaseConfig:
    """Configuration for unified database system"""
    # Paths
    sqlite_path: str = "realtime_system.db"
    json_path: str = "vocalization_database.json"
    cache_path: str = "database_cache"
    backup_path: str = "database_backups"

    # Performance settings
    max_cache_size: int = 10000
    cache_ttl: int = 3600  # 1 hour
    max_sync_threads: int = 4

    # Cloud sync settings
    enable_cloud_sync: bool = False
    cloud_endpoint: str = "https://api.example.com/sync"
    cloud_api_key: str = ""
    sync_interval: int = 300  # 5 minutes

    # Compression settings
    compress_backups: bool = True
    compression_format: str = "gzip"

    # Backup settings
    auto_backup: bool = True
    backup_interval: int = 86400  # 24 hours
    max_backups: int = 7

# ============================================================================
# 2. ABSTRACT DATABASE INTERFACES
# ============================================================================

class DatabaseInterface(ABC):
    """Abstract interface for database implementations"""

    @abstractmethod
    def connect(self) -> bool:
        """Connect to database"""
        pass

    @abstractmethod
    def disconnect(self):
        """Disconnect from database"""
        pass

    @abstractmethod
    def execute_query(self, query: str, params: Tuple = ()) -> Any:
        """Execute a query"""
        pass

    @abstractmethod
    def begin_transaction(self):
        """Begin transaction"""
        pass

    @abstractmethod
    def commit_transaction(self):
        """Commit transaction"""
        pass

    @abstractmethod
    def rollback_transaction(self):
        """Rollback transaction"""
        pass

class CacheInterface(ABC):
    """Abstract interface for caching implementations"""

    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        pass

    @abstractmethod
    def set(self, key: str, value: Any, ttl: int = None):
        """Set value in cache"""
        pass

    @abstractmethod
    def delete(self, key: str):
        """Delete value from cache"""
        pass

    @abstractmethod
    def clear(self):
        """Clear all cache"""
        pass

    @abstractmethod
    def size(self) -> int:
        """Get cache size"""
        pass

# ============================================================================
# 3. SQLITE DATABASE IMPLEMENTATION
# ============================================================================

class SQLiteDatabase(DatabaseInterface):
    """SQLite database implementation for structured data"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.connection = None
        self.lock = threading.Lock()

    def connect(self) -> bool:
        """Connect to SQLite database"""
        try:
            self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
            self.connection.row_factory = sqlite3.Row

            # Enable foreign keys
            self.execute_query("PRAGMA foreign_keys = ON")

            # Create tables if they don't exist
            self._create_tables()

            return True
        except Exception as e:
            logging.error(f"Failed to connect to SQLite database: {e}")
            return False

    def disconnect(self):
        """Disconnect from SQLite database"""
        if self.connection:
            self.connection.close()
            self.connection = None

    def execute_query(self, query: str, params: Tuple = ()) -> Any:
        """Execute a query"""
        with self.lock:
            try:
                cursor = self.connection.cursor()
                cursor.execute(query, params)

                if query.strip().upper().startswith('SELECT'):
                    return cursor.fetchall()
                else:
                    self.connection.commit()
                    return cursor.rowcount
            except Exception as e:
                logging.error(f"Database query failed: {e}, Query: {query}")
                raise

    def begin_transaction(self):
        """Begin transaction"""
        self.connection.execute("BEGIN TRANSACTION")

    def commit_transaction(self):
        """Commit transaction"""
        self.connection.commit()

    def rollback_transaction(self):
        """Rollback transaction"""
        self.connection.rollback()

    def _create_tables(self):
        """Create database tables"""
        # Decisions table for provenance logging
        self.execute_query('''
            CREATE TABLE IF NOT EXISTS decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                timestamp TEXT,
                input_features TEXT,
                context_probabilities TEXT,
                phrase_selection TEXT,
                synthesis_method TEXT,
                output_audio BLOB,
                processing_time_ms REAL,
                adaptation_parameters TEXT,
                safety_applied INTEGER,
                cognitive_context TEXT,
                visual_context TEXT,
                experimental_conditions TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Experiments table for A/B testing
        self.execute_query('''
            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT,
                name TEXT,
                description TEXT,
                start_time TEXT,
                end_time TEXT,
                conditions TEXT,
                metrics TEXT,
                status TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Species data table
        self.execute_query('''
            CREATE TABLE IF NOT EXISTS species_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                species_name TEXT,
                total_phrases INTEGER,
                total_sentences INTEGER,
                vocabulary_size INTEGER,
                modality_distribution TEXT,
                acoustic_features TEXT,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(species_name)
            )
        ''')

        # Phrase cache table
        self.execute_query('''
            CREATE TABLE IF NOT EXISTS phrase_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                phrase_key TEXT,
                species TEXT,
                modality TEXT,
                audio_data BLOB,
                acoustic_features TEXT,
                context_info TEXT,
                access_count INTEGER DEFAULT 0,
                last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(phrase_key, species)
            )
        ''')

        # Indexes for performance
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_decisions_session ON decisions(session_id)",
            "CREATE INDEX IF NOT EXISTS idx_decisions_timestamp ON decisions(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_experiments_status ON experiments(status)",
            "CREATE INDEX IF NOT EXISTS idx_species_name ON species_data(species_name)",
            "CREATE INDEX IF NOT EXISTS idx_phrase_key ON phrase_cache(phrase_key)",
            "CREATE INDEX IF NOT EXISTS idx_phrase_species ON phrase_cache(species)",
        ]

        for index in indexes:
            self.execute_query(index)

# ============================================================================
# 4. FILE-BASED CACHE IMPLEMENTATION
# ============================================================================

class FileBasedCache(CacheInterface):
    """File-based cache implementation"""

    def __init__(self, cache_path: str, max_size: int = 10000, ttl: int = 3600):
        self.cache_path = Path(cache_path)
        self.max_size = max_size
        self.ttl = ttl
        self.lock = threading.Lock()
        self.metadata = {}

        # Create cache directory
        self.cache_path.mkdir(parents=True, exist_ok=True)

        # Load existing cache metadata
        self._load_metadata()

    def _load_metadata(self):
        """Load cache metadata"""
        metadata_file = self.cache_path / "metadata.json"
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    self.metadata = json.load(f)
                self._cleanup_expired()
            except Exception as e:
                logging.error(f"Failed to load cache metadata: {e}")
                self.metadata = {}

    def _save_metadata(self):
        """Save cache metadata"""
        metadata_file = self.cache_path / "metadata.json"
        try:
            with open(metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            logging.error(f"Failed to save cache metadata: {e}")

    def _cleanup_expired(self):
        """Clean up expired cache entries"""
        current_time = time.time()
        expired_keys = []

        for key, data in self.metadata.items():
            if current_time - data['timestamp'] > self.ttl:
                expired_keys.append(key)
                file_path = self.cache_path / f"{key}.cache"
                if file_path.exists():
                    file_path.unlink()

        for key in expired_keys:
            del self.metadata[key]

        if expired_keys:
            self._save_metadata()

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        with self.lock:
            if key not in self.metadata:
                return None

            data = self.metadata[key]

            # Check if expired
            if time.time() - data['timestamp'] > self.ttl:
                self.delete(key)
                return None

            # Load from file
            file_path = self.cache_path / f"{key}.cache"
            if file_path.exists() and file_path.stat().st_size > 0:
                try:
                    with open(file_path, 'rb') as f:
                        return pickle.load(f)
                except Exception as e:
                    logging.error(f"Failed to load cache entry {key}: {e}")
                    self.delete(key)
                    return None

            return None

    def set(self, key: str, value: Any, ttl: int = None):
        """Set value in cache"""
        with self.lock:
            # Check size limit
            if self.size() >= self.max_size:
                self._evict_oldest()

            # Save to file
            file_path = self.cache_path / f"{key}.cache"
            try:
                with open(file_path, 'wb') as f:
                    pickle.dump(value, f)
            except Exception as e:
                logging.error(f"Failed to save cache entry {key}: {e}")
                return

            # Update metadata
            self.metadata[key] = {
                'timestamp': time.time(),
                'size': os.path.getsize(file_path),
                'ttl': ttl or self.ttl
            }

            self._save_metadata()

    def delete(self, key: str):
        """Delete value from cache"""
        with self.lock:
            if key in self.metadata:
                del self.metadata[key]
                file_path = self.cache_path / f"{key}.cache"
                if file_path.exists():
                    file_path.unlink()
                self._save_metadata()

    def clear(self):
        """Clear all cache"""
        with self.lock:
            for file_path in self.cache_path.glob("*.cache"):
                file_path.unlink()
            self.metadata.clear()
            self._save_metadata()

    def size(self) -> int:
        """Get cache size"""
        return len(self.metadata)

    def _evict_oldest(self):
        """Evict oldest cache entries"""
        if not self.metadata:
            return

        # Sort by timestamp
        sorted_keys = sorted(
            self.metadata.keys(),
            key=lambda k: self.metadata[k]['timestamp']
        )

        # Evict oldest 10%
        evict_count = max(1, len(sorted_keys) // 10)
        for key in sorted_keys[:evict_count]:
            self.delete(key)

# ============================================================================
# 5. CLOUD SYNC IMPLEMENTATION
# ============================================================================

class CloudSync:
    """Cloud synchronization for database"""

    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.sync_queue = Queue()
        self.sync_lock = threading.Lock()
        self.last_sync = 0
        self.running = False

    async def start_sync(self):
        """Start cloud synchronization"""
        if not self.config.enable_cloud_sync:
            return

        self.running = True

        # Start sync thread
        sync_thread = threading.Thread(target=self._sync_worker, daemon=True)
        sync_thread.start()

        # Start periodic sync
        asyncio.create_task(self._periodic_sync())

    async def stop_sync(self):
        """Stop cloud synchronization"""
        self.running = False

    async def _periodic_sync(self):
        """Periodic sync task"""
        while self.running:
            await asyncio.sleep(self.config.sync_interval)
            await self.sync_all()

    def _sync_worker(self):
        """Background sync worker"""
        while self.running:
            try:
                # Process sync queue
                while not self.sync_queue.empty():
                    data = self.sync_queue.get()
                    asyncio.create_task(self._sync_to_cloud(data))

                time.sleep(1)
            except Exception as e:
                logging.error(f"Sync worker error: {e}")

    async def _sync_to_cloud(self, data: Dict[str, Any]):
        """Sync data to cloud"""
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    'Authorization': f'Bearer {self.config.cloud_api_key}',
                    'Content-Type': 'application/json'
                }

                async with session.post(
                    self.config.cloud_endpoint,
                    json=data,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        logging.info("Data synced to cloud successfully")
                    else:
                        logging.error(f"Cloud sync failed: {response.status}")
        except Exception as e:
            logging.error(f"Cloud sync error: {e}")

    def sync_data(self, table_name: str, data: Dict[str, Any]):
        """Queue data for sync"""
        sync_data = {
            'table': table_name,
            'data': data,
            'timestamp': datetime.now().isoformat(),
            'system_id': str(uuid.uuid4())
        }

        self.sync_queue.put(sync_data)

    async def sync_all(self):
        """Sync all pending data"""
        with self.sync_lock:
            current_time = time.time()

            if current_time - self.last_sync < self.config.sync_interval:
                return

            # Get all pending data
            pending_data = []
            while not self.sync_queue.empty():
                pending_data.append(self.sync_queue.get())

            if pending_data:
                await self._sync_to_cloud({
                    'batch': True,
                    'data': pending_data
                })

            self.last_sync = current_time

# ============================================================================
# 6. BACKUP AND RECOVERY
# ============================================================================

class DatabaseBackup:
    """Database backup and recovery system"""

    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.backup_path = Path(config.backup_path)
        self.backup_path.mkdir(parents=True, exist_ok=True)

    def create_backup(self, database_path: str) -> str:
        """Create database backup"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"database_backup_{timestamp}"

            if self.config.compress_backups:
                backup_file = self.backup_path / f"{backup_name}.db.gz"
                import gzip
                with open(database_path, 'rb') as f_in:
                    with gzip.open(backup_file, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
            else:
                backup_file = self.backup_path / f"{backup_name}.db"
                shutil.copy2(database_path, backup_file)

            # Cleanup old backups
            self._cleanup_old_backups()

            logging.info(f"Backup created: {backup_file}")
            return str(backup_file)

        except Exception as e:
            logging.error(f"Backup failed: {e}")
            return None

    def restore_backup(self, backup_file: str, target_path: str) -> bool:
        """Restore database from backup"""
        try:
            if backup_file.endswith('.gz'):
                import gzip
                with gzip.open(backup_file, 'rb') as f_in:
                    with open(target_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
            else:
                shutil.copy2(backup_file, target_path)

            logging.info(f"Restored from backup: {backup_file}")
            return True

        except Exception as e:
            logging.error(f"Restore failed: {e}")
            return False

    def _cleanup_old_backups(self):
        """Clean up old backups"""
        if self.config.max_backups <= 0:
            return

        # Get all backup files
        backup_files = list(self.backup_path.glob("*.db*"))
        backup_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

        # Remove old backups
        for backup_file in backup_files[self.config.max_backups:]:
            backup_file.unlink()

    def get_backup_list(self) -> List[Dict[str, Any]]:
        """Get list of available backups"""
        backups = []

        for backup_file in self.backup_path.glob("*.db*"):
            stat = backup_file.stat()
            backups.append({
                'name': backup_file.name,
                'path': str(backup_file),
                'size': stat.st_size,
                'created': datetime.fromtimestamp(stat.st_mtime).isoformat()
            })

        return sorted(backups, key=lambda x: x['created'], reverse=True)

# ============================================================================
# 7. UNIFIED DATABASE MANAGER
# ============================================================================

class UnifiedDatabaseManager:
    """Main unified database manager"""

    def __init__(self, config: DatabaseConfig = None):
        self.config = config or DatabaseConfig()
        self.sqlite_db = None
        self.cache = None
        self.cloud_sync = None
        self.backup = None
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self._initialize_components()

    def _initialize_components(self):
        """Initialize database components"""
        # SQLite database
        self.sqlite_db = SQLiteDatabase(self.config.sqlite_path)
        if not self.sqlite_db.connect():
            raise RuntimeError("Failed to connect to SQLite database")

        # Cache
        self.cache = FileBasedCache(
            self.config.cache_path,
            self.config.max_cache_size,
            self.config.cache_ttl
        )

        # Cloud sync
        self.cloud_sync = CloudSync(self.config)

        # Backup system
        self.backup = DatabaseBackup(self.config)

    async def start(self):
        """Start database manager"""
        # Start cloud sync
        await self.cloud_sync.start_sync()

        # Start auto backup
        if self.config.auto_backup:
            asyncio.create_task(self._auto_backup())

        self.logger.info("Database manager started")

    async def stop(self):
        """Stop database manager"""
        await self.cloud_sync.stop_sync()
        self.sqlite_db.disconnect()
        self.logger.info("Database manager stopped")

    async def _auto_backup(self):
        """Auto backup task"""
        while True:
            await asyncio.sleep(self.config.backup_interval)
            self.create_backup()

    def create_backup(self) -> str:
        """Create database backup"""
        return self.backup.create_backup(self.config.sqlite_path)

    def restore_backup(self, backup_file: str) -> bool:
        """Restore database from backup"""
        return self.backup.restore_backup(backup_file, self.config.sqlite_path)

    # ============================================================================
    # 8. DATABASE OPERATIONS
    # ============================================================================

    def log_decision(self, decision_record: Dict[str, Any]):
        """Log decision to database (implementing data_logging.py feature)"""
        # Check cache first
        cache_key = f"decision_{decision_record['session_id']}_{decision_record['timestamp']}"
        cached = self.cache.get(cache_key)
        if cached:
            return

        # Insert into database
        self.sqlite_db.execute_query('''
            INSERT INTO decisions (
                session_id, timestamp, input_features, context_probabilities,
                phrase_selection, synthesis_method, output_audio, processing_time_ms,
                adaptation_parameters, safety_applied, cognitive_context,
                visual_context, experimental_conditions
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            decision_record.get('session_id'),
            decision_record.get('timestamp'),
            json.dumps(decision_record.get('input_features', {})),
            json.dumps(decision_record.get('context_probabilities', {})),
            json.dumps(decision_record.get('phrase_selection', {})),
            decision_record.get('synthesis_method'),
            np.array(decision_record.get('output_audio', [])).tobytes(),
            decision_record.get('processing_time_ms', 0.0),
            json.dumps(decision_record.get('adaptation_parameters', {})),
            int(decision_record.get('safety_applied', False)),
            json.dumps(decision_record.get('cognitive_context')),
            json.dumps(decision_record.get('visual_context')),
            json.dumps(decision_record.get('experimental_conditions', {}))
        ))

        # Cache the decision
        self.cache.set(cache_key, decision_record, ttl=self.config.cache_ttl)

        # Queue for cloud sync
        self.cloud_sync.sync_data('decisions', decision_record)

    def load_phrase_database(self, phrase_key: str, species: str) -> Optional[Dict]:
        """Load phrase from database (implementing dual_path_analyzer.py feature)"""
        # Check cache first
        cache_key = f"phrase_{phrase_key}_{species}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached

        # Query database
        result = self.sqlite_db.execute_query('''
            SELECT audio_data, acoustic_features, context_info
            FROM phrase_cache
            WHERE phrase_key = ? AND species = ?
        ''', (phrase_key, species))

        if result:
            phrase_data = {
                'audio_data': result[0]['audio_data'],
                'acoustic_features': json.loads(result[0]['acoustic_features'] or '{}'),
                'context_info': json.loads(result[0]['context_info'] or '{}')
            }

            # Update access count
            self.sqlite_db.execute_query('''
                UPDATE phrase_cache
                SET access_count = access_count + 1, last_accessed = CURRENT_TIMESTAMP
                WHERE phrase_key = ? AND species = ?
            ''', (phrase_key, species))

            # Cache the result
            self.cache.set(cache_key, phrase_data, ttl=self.config.cache_ttl)

            return phrase_data

        return None

    def save_phrase_to_cache(self, phrase_key: str, species: str, audio_data: bytes,
                           acoustic_features: Dict, context_info: Dict):
        """Save phrase to cache (implementing concatenative database feature)"""
        # Check if already exists
        existing = self.sqlite_db.execute_query('''
            SELECT id FROM phrase_cache WHERE phrase_key = ? AND species = ?
        ''', (phrase_key, species))

        if existing:
            # Update existing
            self.sqlite_db.execute_query('''
                UPDATE phrase_cache SET
                    audio_data = ?,
                    acoustic_features = ?,
                    context_info = ?,
                    created_at = CURRENT_TIMESTAMP
                WHERE phrase_key = ? AND species = ?
            ''', (
                audio_data,
                json.dumps(acoustic_features),
                json.dumps(context_info),
                phrase_key,
                species
            ))
        else:
            # Insert new
            self.sqlite_db.execute_query('''
                INSERT INTO phrase_cache (
                    phrase_key, species, audio_data, acoustic_features, context_info
                ) VALUES (?, ?, ?, ?, ?)
            ''', (
                phrase_key,
                species,
                audio_data,
                json.dumps(acoustic_features),
                json.dumps(context_info)
            ))

        # Cache the result
        cache_key = f"phrase_{phrase_key}_{species}"
        phrase_data = {
            'audio_data': audio_data,
            'acoustic_features': acoustic_features,
            'context_info': context_info
        }
        self.cache.set(cache_key, phrase_data, ttl=self.config.cache_ttl)

    def get_experiment_data(self, experiment_id: str) -> Optional[Dict]:
        """Get experiment data"""
        cache_key = f"experiment_{experiment_id}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached

        result = self.sqlite_db.execute_query('''
            SELECT * FROM experiments WHERE experiment_id = ?
        ''', (experiment_id,))

        if result:
            experiment = dict(result[0])
            self.cache.set(cache_key, experiment, ttl=self.config.cache_ttl)
            return experiment

        return None

    def save_experiment_data(self, experiment_data: Dict):
        """Save experiment data"""
        self.sqlite_db.execute_query('''
            INSERT OR REPLACE INTO experiments (
                experiment_id, name, description, start_time, end_time,
                conditions, metrics, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            experiment_data.get('experiment_id'),
            experiment_data.get('name'),
            experiment_data.get('description'),
            experiment_data.get('start_time'),
            experiment_data.get('end_time'),
            json.dumps(experiment_data.get('conditions', {})),
            json.dumps(experiment_data.get('metrics', {})),
            experiment_data.get('status', 'active')
        ))

        # Cache the result
        cache_key = f"experiment_{experiment_data['experiment_id']}"
        self.cache.set(cache_key, experiment_data, ttl=self.config.cache_ttl)

    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        stats = {}

        # SQLite stats
        try:
            # Table counts
            tables = ['decisions', 'experiments', 'species_data', 'phrase_cache']
            for table in tables:
                count = self.sqlite_db.execute_query(f'SELECT COUNT(*) FROM {table}')[0][0]
                stats[f'{table}_count'] = count

            # Database size
            if os.path.exists(self.config.sqlite_path):
                stats['database_size_bytes'] = os.path.getsize(self.config.sqlite_path)

        except Exception as e:
            self.logger.error(f"Failed to get database stats: {e}")

        # Cache stats
        stats['cache_size'] = self.cache.size()
        stats['cache_max_size'] = self.config.max_cache_size

        # Cloud sync status
        stats['cloud_sync_enabled'] = self.config.enable_cloud_sync
        stats['sync_queue_size'] = self.cloud_sync.sync_queue.qsize()

        # Backup status
        stats['backups'] = self.backup.get_backup_list()

        return stats

# ============================================================================
# 9. DATABASE UTILITIES
# ============================================================================

def create_database_manager(config_path: str = None) -> UnifiedDatabaseManager:
    """Create database manager from config file"""
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        config = DatabaseConfig(**config_data)
    else:
        config = DatabaseConfig()

    return UnifiedDatabaseManager(config)

# Example usage
if __name__ == "__main__":
    # Create database manager
    db_manager = create_database_manager()

    # Start database manager
    asyncio.run(db_manager.start())

    # Example usage
    decision_record = {
        'session_id': str(uuid.uuid4()),
        'timestamp': datetime.now().isoformat(),
        'input_features': {'f0': 7400, 'duration': 0.1},
        'context_probabilities': {'contact': 0.8, 'alarm': 0.2},
        'phrase_selection': {'phrase_key': 'F0_7400'},
        'synthesis_method': 'microharmonic',
        'output_audio': [0.1, 0.2, 0.3] * 1000,
        'processing_time_ms': 45.6,
        'adaptation_parameters': {'learning_rate': 0.01},
        'safety_applied': True,
        'cognitive_context': None,
        'visual_context': None,
        'experimental_conditions': {}
    }

    # Log decision
    db_manager.log_decision(decision_record)

    # Get stats
    stats = db_manager.get_database_stats()
    print(f"Database stats: {stats}")

    # Stop database manager
    asyncio.run(db_manager.stop())