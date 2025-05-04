import socket
import threading
import time
import logging
import json
import hashlib
import uuid
import os
import subprocess
from typing import Optional, List, Dict, Set
import base58
import requests
from pathlib import Path
import traceback
import psutil

# Configure logging
logging.basicConfig(
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

DISCOVERY_PORT = 31338
BROADCAST_INTERVAL = 1.0  # seconds
MDNS_SERVICE_TYPE = "_petals._tcp.local."
GITHUB_API_URL = "https://api.github.com"
PEERS_FILE = "known_peers.json"

class UnifiedDiscovery:
    """Unified peer discovery system combining UDP broadcast, mDNS/Bonjour, UPnP, and git-based discovery."""
    
    def __init__(self):
        self.peers: Set[str] = set()  # Set of peer addresses
        self.sock: Optional[socket.socket] = None
        self.is_running = False
        self.lock = threading.Lock()
        self._peer_id = None
        self._mdns_browser = None
        self._mdns_zeroconf = None
        self._upnp_client = None
        self._known_peers_file = Path(PEERS_FILE)
        self._load_known_peers()
        
    def _load_known_peers(self):
        """Load known peers from file."""
        try:
            if self._known_peers_file.exists():
                with open(self._known_peers_file, 'r') as f:
                    known_peers = json.load(f)
                    with self.lock:
                        self.peers.update(known_peers)
                logger.info(f"Loaded {len(known_peers)} known peers from file")
        except Exception as e:
            logger.warning(f"Failed to load known peers: {e}")
            
    def _save_known_peers(self):
        """Save known peers to file."""
        try:
            with self.lock:
                known_peers = list(self.peers)
            with open(self._known_peers_file, 'w') as f:
                json.dump(known_peers, f)
            logger.debug(f"Saved {len(known_peers)} known peers to file")
        except Exception as e:
            logger.warning(f"Failed to save known peers: {e}")
    
    @property
    def peer_id(self) -> str:
        """Get or generate a stable peer ID."""
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
                # Fallback to UUID-based ID
                self._peer_id = base58.b58encode(uuid.uuid4().bytes).decode()
                logger.warning(f"Using fallback peer ID: {self._peer_id}")
                
        return self._peer_id
    
    def _get_machine_id(self) -> str:
        """Get a stable machine identifier."""
        try:
            # Try to use the system's machine ID first
            if os.path.exists('/etc/machine-id'):
                with open('/etc/machine-id', 'r') as f:
                    return f.read().strip()
            # Fallback to using hardware info on macOS
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
            mac = hex(uuid.getnode())[2:]
        except:
            mac = "000000000000"
        return f"{hostname}-{mac}"
    
    def start_discovery(self):
        """Start all discovery methods."""
        if self.is_running:
            return
            
        self.is_running = True
        self.peers.clear()
        
        # Start UDP broadcast discovery
        self._start_udp_discovery()
        
        # Start mDNS/Bonjour discovery if available
        self._start_mdns_discovery()
        
        # Start UPnP discovery if available
        self._start_upnp_discovery()
        
        logger.info("Started unified peer discovery")
        
    def _start_udp_discovery(self):
        """Start UDP broadcast discovery."""
        try:
            logger.debug("Initializing UDP socket...")
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            
            try:
                logger.debug(f"Binding UDP socket to port {DISCOVERY_PORT}...")
                self.sock.bind(('', DISCOVERY_PORT))
            except socket.error as bind_error:
                logger.error(f"Failed to bind UDP socket: {str(bind_error)}")
                logger.debug(f"Socket bind error details:\n{traceback.format_exc()}")
                # Try to find what's using the port
                try:
                    for conn in psutil.net_connections():
                        if conn.laddr.port == DISCOVERY_PORT:
                            logger.error(f"Port {DISCOVERY_PORT} is in use by process {conn.pid}")
                except Exception as e:
                    logger.debug(f"Failed to check port usage: {e}")
                raise
            
            self.sock.settimeout(1.0)
            
            # Start listener thread
            logger.debug("Starting UDP listener thread...")
            self.listen_thread = threading.Thread(target=self._listen_udp, daemon=True)
            self.listen_thread.start()
            logger.info("Started UDP discovery")
            
        except Exception as e:
            logger.error(f"Failed to start UDP discovery: {str(e)}")
            logger.debug(f"UDP discovery error details:\n{traceback.format_exc()}")
            
    def _start_mdns_discovery(self):
        """Start mDNS/Bonjour discovery if available."""
        try:
            from zeroconf import ServiceBrowser, Zeroconf
            
            class PetalsListener:
                def __init__(self, discovery):
                    self.discovery = discovery
                
                def add_service(self, zeroconf, type, name):
                    try:
                        info = zeroconf.get_service_info(type, name)
                        if info:
                            try:
                                addr = info.properties.get(b'ip', b'').decode() or socket.inet_ntoa(info.addresses[0])
                                port = info.port
                                peer_id = info.properties.get(b'peer_id', b'').decode()
                                
                                if peer_id:
                                    peer_addr = f"/ip4/{addr}/tcp/{port}/p2p/{peer_id}"
                                    with self.discovery.lock:
                                        self.discovery.peers.add(peer_addr)
                                    logger.debug(f"Found peer via mDNS: {peer_addr}")
                                else:
                                    logger.warning(f"Peer ID missing in mDNS service: {name}")
                            except Exception as e:
                                logger.warning(f"Failed to process mDNS service: {str(e)}")
                                logger.debug(f"mDNS service processing error details:\n{traceback.format_exc()}")
                    except Exception as e:
                        logger.error(f"Error in mDNS add_service: {str(e)}")
                        logger.debug(f"mDNS add_service error details:\n{traceback.format_exc()}")
                
                def remove_service(self, zeroconf, type, name):
                    """Handle service removal."""
                    try:
                        info = zeroconf.get_service_info(type, name)
                        if info:
                            try:
                                addr = info.properties.get(b'ip', b'').decode() or socket.inet_ntoa(info.addresses[0])
                                port = info.port
                                peer_id = info.properties.get(b'peer_id', b'').decode()
                                
                                if peer_id:
                                    peer_addr = f"/ip4/{addr}/tcp/{port}/p2p/{peer_id}"
                                    with self.discovery.lock:
                                        if peer_addr in self.discovery.peers:
                                            self.discovery.peers.remove(peer_addr)
                                            logger.debug(f"Removed peer via mDNS: {peer_addr}")
                            except Exception as e:
                                logger.warning(f"Failed to process mDNS service removal: {str(e)}")
                                logger.debug(f"mDNS service removal error details:\n{traceback.format_exc()}")
                    except Exception as e:
                        logger.error(f"Error in mDNS remove_service: {str(e)}")
                        logger.debug(f"mDNS remove_service error details:\n{traceback.format_exc()}")
                
                def update_service(self, zeroconf, type, name):
                    """Handle service updates."""
                    logger.debug(f"mDNS service updated: {name}")
            
            logger.debug("Initializing mDNS/Zeroconf...")
            self._mdns_zeroconf = Zeroconf()
            self._mdns_browser = ServiceBrowser(
                self._mdns_zeroconf,
                MDNS_SERVICE_TYPE,
                PetalsListener(self)
            )
            logger.info("Started mDNS discovery")
            
        except ImportError:
            logger.info("mDNS discovery not available (zeroconf not installed)")
        except Exception as e:
            logger.error(f"Failed to start mDNS discovery: {str(e)}")
            logger.debug(f"mDNS discovery error details:\n{traceback.format_exc()}")
            
    def _start_upnp_discovery(self):
        """Start UPnP discovery if available."""
        try:
            import miniupnpc
            
            self._upnp_client = miniupnpc.UPnP()
            self._upnp_client.discoverdelay = 200
            self._upnp_client.discover()
            self._upnp_client.selectigd()
            logger.info("Started UPnP discovery")
            
        except ImportError:
            logger.info("UPnP discovery not available (miniupnpc not installed)")
        except Exception as e:
            logger.error(f"Failed to start UPnP discovery: {e}")
            
    def _listen_udp(self):
        """Listen for UDP peer announcements."""
        while self.is_running:
            try:
                data, addr = self.sock.recvfrom(2048)
                try:
                    peer_info = json.loads(data.decode())
                    peer_addr = self._format_peer_address(
                        peer_info['ip'],
                        peer_info['port'],
                        peer_info.get('peer_id', self.peer_id)
                    )
                    
                    with self.lock:
                        self.peers.add(peer_addr)
                        logger.debug(f"Found peer via UDP: {peer_addr}")
                        
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON from {addr}")
                except Exception as e:
                    logger.warning(f"Failed to process peer info: {e}")
                    
            except socket.timeout:
                pass
            except Exception as e:
                if self.is_running:
                    logger.error(f"Error in UDP listener: {e}")
                    
    def _format_peer_address(self, ip: str, port: int, peer_id: str) -> str:
        """Format a peer address in libp2p format."""
        return f"/ip4/{ip}/tcp/{port}/p2p/{peer_id}"
    
    def get_peers(self) -> List[str]:
        """Get list of discovered peer addresses."""
        with self.lock:
            peers = list(self.peers)
            
        # If no peers found through real-time discovery, try git-based discovery
        if not peers:
            peers = self._discover_peers_from_git()
            if peers:
                with self.lock:
                    self.peers.update(peers)
                self._save_known_peers()
                
        return peers
        
    def _discover_peers_from_git(self) -> List[str]:
        """Discover peers from git repository."""
        try:
            # Try to get peers from GitHub repository
            repo_url = "https://github.com/bigscience-workshop/petals"
            api_url = f"{GITHUB_API_URL}/repos/bigscience-workshop/petals/contents/known_peers.json"
            
            response = requests.get(api_url)
            if response.status_code == 200:
                content = response.json()
                if 'content' in content:
                    import base64
                    peers_data = json.loads(base64.b64decode(content['content']).decode())
                    logger.info(f"Found {len(peers_data)} peers from git repository")
                    return peers_data
                    
        except Exception as e:
            logger.warning(f"Failed to discover peers from git: {e}")
            
        return []
    
    def start_advertising(self, port: int, model_name: str, device: str = "mps"):
        """Start advertising this peer using all available methods."""
        if not self.is_running:
            self.start_discovery()
            
        # Start UDP advertising
        self._start_udp_advertising(port, model_name, device)
        
        # Start mDNS advertising if available
        self._start_mdns_advertising(port, model_name, device)
        
        # Start UPnP port mapping if available
        self._start_upnp_advertising(port)
        
        logger.info(f"Started advertising peer on port {port}")
        
    def _start_udp_advertising(self, port: int, model_name: str, device: str):
        """Start UDP broadcast advertising."""
        def announce():
            while self.is_running:
                try:
                    # Get local IP address
                    hostname = socket.gethostname()
                    local_ip = socket.gethostbyname(hostname)
                    
                    # Create peer info
                    peer_info = {
                        'ip': local_ip,
                        'port': port,
                        'peer_id': self.peer_id,
                        'model': model_name,
                        'device': device
                    }
                    
                    # Send announcement
                    self.sock.sendto(
                        json.dumps(peer_info).encode(),
                        ('<broadcast>', DISCOVERY_PORT)
                    )
                    time.sleep(BROADCAST_INTERVAL)
                except Exception as e:
                    if self.is_running:
                        logger.error(f"Error in UDP advertising: {e}")
                    
        self.announce_thread = threading.Thread(target=announce, daemon=True)
        self.announce_thread.start()
        
    def _start_mdns_advertising(self, port: int, model_name: str, device: str):
        """Start mDNS/Bonjour advertising if available."""
        try:
            from zeroconf import ServiceInfo
            
            hostname = socket.gethostname()
            local_ip = socket.gethostbyname(hostname)
            service_name = f"Petals-{hostname}-{port}"
            
            info = ServiceInfo(
                type_=MDNS_SERVICE_TYPE,
                name=f"{service_name}.{MDNS_SERVICE_TYPE}",
                addresses=[socket.inet_aton(local_ip)],
                port=port,
                properties={
                    'model': model_name,
                    'device': device,
                    'peer_id': self.peer_id,
                    'ip': local_ip
                }
            )
            
            if self._mdns_zeroconf:
                self._mdns_zeroconf.register_service(info)
                logger.info("Started mDNS advertising")
                
        except ImportError:
            pass
        except Exception as e:
            logger.error(f"Failed to start mDNS advertising: {e}")
            
    def _start_upnp_advertising(self, port: int):
        """Start UPnP port mapping if available."""
        try:
            if self._upnp_client:
                self._upnp_client.addportmapping(
                    port, 'TCP', self._upnp_client.lanaddr, port,
                    'Petals P2P', ''
                )
                logger.info(f"Added UPnP port mapping for port {port}")
                
        except Exception as e:
            logger.error(f"Failed to add UPnP port mapping: {e}")
            
    def stop(self):
        """Stop all discovery and advertising."""
        self.is_running = False
        
        # Stop UDP socket
        if self.sock:
            try:
                self.sock.close()
            except:
                pass
                
        # Stop mDNS
        if self._mdns_zeroconf:
            try:
                self._mdns_zeroconf.close()
            except:
                pass
                
        # Stop UPnP
        if self._upnp_client:
            try:
                self._upnp_client.deleteportmapping(self._upnp_client.lanaddr, 'TCP')
            except:
                pass
                
        logger.info("Stopped unified peer discovery") 