import socket
import threading
import time
import logging
import base58
import hashlib
import os

logger = logging.getLogger(__name__)

DISCOVERY_PORT = 31338
DISCOVERY_TIMEOUT = 5  # seconds

class SimplePeerDiscovery:
    """Simple UDP-based peer discovery for Petals network."""
    
    def __init__(self):
        self.heard_peers = set()
        self.sock = None
        self.is_running = False
        self.lock = threading.Lock()
        self._peer_id = None
        
    @property
    def peer_id(self):
        """Generate or return a stable peer ID for this instance."""
        if self._peer_id is None:
            try:
                # Create a stable peer ID based on machine-specific information
                machine_id = self._get_machine_id()
                
                # Generate a hash that will be consistent for this machine
                identity_hash = hashlib.sha256(machine_id.encode()).digest()
                
                # Create a proper libp2p peer ID using multihash
                # Format: <hash-func-code><digest-length><digest-value>
                mh = bytes([0x12]) + bytes([len(identity_hash)]) + identity_hash
                
                # Encode with base58btc (what libp2p expects)
                self._peer_id = base58.b58encode(mh).decode()
                
                logger.debug(f"Generated peer ID: {self._peer_id}")
                
            except Exception as e:
                logger.error(f"Failed to generate peer ID: {e}")
                raise RuntimeError(f"Failed to generate valid peer ID: {e}")
            
        return self._peer_id
    
    def _get_machine_id(self) -> str:
        """Get a stable machine identifier."""
        try:
            # Try to use the system's machine ID first
            if os.path.exists('/etc/machine-id'):
                with open('/etc/machine-id', 'r') as f:
                    return f.read().strip()
            # Fallback to using hardware info on macOS
            import subprocess
            result = subprocess.run(['system_profiler', 'SPHardwareDataType'], 
                                 capture_output=True, text=True)
            # Extract serial number or hardware UUID
            for line in result.stdout.split('\n'):
                if 'Serial Number' in line or 'Hardware UUID' in line:
                    return line.split(':')[1].strip()
        except Exception as e:
            logger.warning(f"Failed to get machine ID: {e}")
        
        # Final fallback: use hostname + MAC address if available
        hostname = socket.gethostname()
        try:
            import uuid
            mac = hex(uuid.getnode())[2:]
        except:
            mac = "000000000000"
        return f"{hostname}-{mac}"
        
    def start_discovery(self):
        """Start discovering peers on the local network."""
        if self.is_running:
            return
            
        self.is_running = True
        self.heard_peers.clear()
        
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self.sock.bind(('', DISCOVERY_PORT))
        self.sock.settimeout(1.0)
        
        self.listen_thread = threading.Thread(target=self._listen, daemon=True)
        self.listen_thread.start()
        logger.info("Started peer discovery")
        
    def _listen(self):
        """Listen for peer announcements."""
        while self.is_running:
            try:
                data, _ = self.sock.recvfrom(2048)
                with self.lock:
                    for line in data.decode().splitlines():
                        self.heard_peers.add(line.strip())
            except socket.timeout:
                pass
            except Exception as e:
                logger.error(f"Error in discovery listener: {e}")
                
    def get_peers(self, model_name=None):
        """Get list of discovered peer addresses."""
        with self.lock:
            return list(self.heard_peers)
            
    def start_advertising(self, port, model_name, device="mps"):
        """Start advertising this peer on the local network."""
        if not self.is_running:
            self.start_discovery()
            
        def announce():
            while self.is_running:
                try:
                    # Get local IP address
                    hostname = socket.gethostname()
                    local_ip = socket.gethostbyname(hostname)
                    
                    # Create peer address with peer ID
                    peer_addr = f"/ip4/{local_ip}/tcp/{port}/p2p/{self.peer_id}"
                    
                    # Send announcement
                    self.sock.sendto(peer_addr.encode(), ('<broadcast>', DISCOVERY_PORT))
                    time.sleep(1.0)
                except Exception as e:
                    logger.error(f"Error in peer announcement: {e}")
                    
        self.announce_thread = threading.Thread(target=announce, daemon=True)
        self.announce_thread.start()
        logger.info(f"Started advertising peer on port {port}")
        
    def stop(self):
        """Stop peer discovery and advertising."""
        self.is_running = False
        if self.sock:
            self.sock.close()
            self.sock = None
        logger.info("Stopped peer discovery") 