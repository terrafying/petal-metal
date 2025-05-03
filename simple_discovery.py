import socket
import threading
import time
import logging

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
                    
                    # Create peer address
                    peer_addr = f"/ip4/{local_ip}/tcp/{port}"
                    
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