import os
import json
import logging
import threading
import requests
import time
from pathlib import Path
from typing import Set, Optional, Dict
from datetime import datetime, timedelta
from urllib.parse import urlparse

# Configure logging
logging.basicConfig(
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

PEERS_FILE = "peers.json"
HEALTH_CHECK_INTERVAL = 30  # seconds
PEER_TIMEOUT = 120  # seconds

DEFAULT_PEERS = {
    "version": "1.0",
    "peers": []
}

class LocalDiscovery:
    """Simple local file-based peer discovery system with health checking."""
    
    def __init__(self, peers_file: Optional[str] = None):
        self.peers: Dict[str, Dict] = {}  # address -> peer info
        self.lock = threading.Lock()
        self.peers_file = Path(peers_file or PEERS_FILE)
        self.is_running = False
        self.health_check_thread = None
        
    def _ensure_peers_file(self):
        """Ensure the peers file exists with proper structure."""
        try:
            if not self.peers_file.exists():
                # Create directory if it doesn't exist
                self.peers_file.parent.mkdir(parents=True, exist_ok=True)
                # Write default structure
                with open(self.peers_file, 'w') as f:
                    json.dump(DEFAULT_PEERS, f, indent=2)
                logger.info(f"Created new peers file at {self.peers_file}")
            else:
                # Validate existing file
                try:
                    with open(self.peers_file, 'r') as f:
                        data = json.load(f)
                        if not isinstance(data, dict) or "peers" not in data:
                            raise ValueError("Invalid peers file structure")
                except (json.JSONDecodeError, ValueError) as e:
                    logger.warning(f"Invalid peers file: {e}. Recreating with default structure.")
                    with open(self.peers_file, 'w') as f:
                        json.dump(DEFAULT_PEERS, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to ensure peers file: {e}")
            raise
            
    def _load_peers(self):
        """Load peers from the local file."""
        try:
            self._ensure_peers_file()
            
            with open(self.peers_file, 'r') as f:
                data = json.load(f)
                with self.lock:
                    self.peers = {peer["address"]: peer for peer in data["peers"]}
                logger.info(f"Loaded {len(self.peers)} peers from file")
        except Exception as e:
            logger.error(f"Failed to load peers: {e}")
            raise
            
    def _save_peers(self):
        """Save peers to the local file."""
        try:
            with self.lock:
                data = {
                    "version": "1.0",
                    "peers": list(self.peers.values())
                }
                    
            with open(self.peers_file, 'w') as f:
                json.dump(data, f, indent=2)
            logger.debug(f"Saved {len(self.peers)} peers to file")
            
        except Exception as e:
            logger.error(f"Failed to save peers: {e}")
            raise
            
    def _check_peer_health(self, address: str) -> bool:
        """Check if a peer is healthy by making a request to its health endpoint."""
        try:
            # Parse the URL to ensure it's valid
            parsed = urlparse(address)
            if not parsed.scheme or not parsed.netloc:
                logger.warning(f"Invalid peer address: {address}")
                return False
                
            # Try to connect to the peer's health endpoint
            health_url = f"{address}/health"
            response = requests.get(health_url, timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.debug(f"Health check failed for {address}: {e}")
            return False
            
    def _health_check_loop(self):
        """Background thread for checking peer health."""
        while self.is_running:
            try:
                with self.lock:
                    current_time = datetime.utcnow()
                    for address, peer_info in list(self.peers.items()):
                        # Skip self
                        if peer_info.get("is_self", False):
                            continue
                            
                        # Check if peer has timed out
                        last_seen = datetime.fromisoformat(peer_info["last_seen"])
                        if current_time - last_seen > timedelta(seconds=PEER_TIMEOUT):
                            logger.info(f"Peer timed out: {address}")
                            del self.peers[address]
                            continue
                            
                        # Check peer health
                        is_healthy = self._check_peer_health(address)
                        if not is_healthy:
                            logger.warning(f"Peer unhealthy: {address}")
                            peer_info["status"] = "unhealthy"
                        else:
                            peer_info["status"] = "active"
                            peer_info["last_seen"] = current_time.isoformat()
                            
                self._save_peers()
                
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
                
            time.sleep(HEALTH_CHECK_INTERVAL)
            
    def start_discovery(self):
        """Start the local discovery system."""
        if self.is_running:
            return
            
        self._load_peers()
        self.is_running = True
        
        # Start health check thread
        self.health_check_thread = threading.Thread(
            target=self._health_check_loop,
            daemon=True
        )
        self.health_check_thread.start()
        
        logger.info("Started local peer discovery")
        
    def add_peer(self, address: str, metadata: Optional[Dict] = None):
        """Add a new peer."""
        try:
            with self.lock:
                # Parse the URL to ensure it's valid
                parsed = urlparse(address)
                if not parsed.scheme or not parsed.netloc:
                    raise ValueError(f"Invalid peer address: {address}")
                    
                peer_info = {
                    "address": address,
                    "last_seen": datetime.utcnow().isoformat(),
                    "metadata": metadata or {},
                    "status": "active"
                }
                
                self.peers[address] = peer_info
                self._save_peers()
                logger.info(f"Added new peer: {address}")
                    
        except Exception as e:
            logger.error(f"Failed to add peer: {e}")
            raise
            
    def remove_peer(self, address: str):
        """Remove a peer."""
        try:
            with self.lock:
                if address in self.peers:
                    del self.peers[address]
                    self._save_peers()
                    logger.info(f"Removed peer: {address}")
                    
        except Exception as e:
            logger.error(f"Failed to remove peer: {e}")
            raise
            
    def get_peers(self) -> Dict[str, Dict]:
        """Get the current set of peers with their metadata."""
        with self.lock:
            return self.peers.copy()
            
    def stop(self):
        """Stop the discovery system."""
        self.is_running = False
        if self.health_check_thread:
            self.health_check_thread.join(timeout=1)
        logger.info("Stopped local peer discovery") 