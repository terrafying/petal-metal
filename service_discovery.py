from typing import List, Optional
import socket
import threading
from zeroconf import ServiceBrowser, ServiceInfo, Zeroconf
import logging
import time
import base58
import hashlib
import os
import multihash
import multibase

logger = logging.getLogger(__name__)

class PetalsServiceDiscovery:
    """
    Service discovery for Petals network using mDNS/Bonjour.
    Particularly optimized for macOS but works cross-platform.
    """
    SERVICE_TYPE = "_petals._tcp.local."
    HASH_FUNC = 0x12  # sha2-256 in multihash
    
    def __init__(self):
        self.zeroconf = Zeroconf()
        self.browser = None
        self.services = {}
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
                
                # Create a proper libp2p peer ID:
                # 1. Create a multihash (sha2-256 + digest)
                # multihash format: <hash-func-code><digest-length><digest-value>
                mh = bytes([self.HASH_FUNC]) + bytes([len(identity_hash)]) + identity_hash
                
                # 2. Encode with base58btc (what libp2p expects)
                self._peer_id = base58.b58encode(mh).decode()
                
                logger.debug(f"Generated peer ID: {self._peer_id}")
                
                # Validate the peer ID format
                try:
                    # Attempt to decode it - this will fail if format is wrong
                    decoded = base58.b58decode(self._peer_id)
                    hash_func, _ = multihash.decode(decoded)
                    if hash_func != self.HASH_FUNC:  # Check if it's sha2-256
                        raise ValueError(f"Wrong hash algorithm: {hash_func}")
                except Exception as e:
                    logger.error(f"Validation failed for generated peer ID: {e}")
                    self._peer_id = None
                    raise RuntimeError("Failed to validate peer ID")
                
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
        
    def start_advertising(self, port: int, model_name: str, device: str = "mps") -> None:
        """
        Advertise this Petals server on the local network.
        
        Args:
            port: The port number the server is listening on
            model_name: Name of the model being served
            device: Device type being used (e.g., "mps" for Apple Silicon)
        """
        hostname = socket.gethostname()
        service_name = f"Petals-{hostname}-{port}"
        
        # Get the local IP address
        local_ip = socket.gethostbyname(hostname)
        
        # Generate peer ID before advertising
        peer_id = self.peer_id  # This will raise an error if generation fails
        
        info = ServiceInfo(
            type_=self.SERVICE_TYPE,
            name=f"{service_name}.{self.SERVICE_TYPE}",
            addresses=[socket.inet_aton(local_ip)],
            port=port,
            properties={
                'model': model_name,
                'device': device,
                'peer_id': peer_id,
                'ip': local_ip
            }
        )
        
        try:
            self.zeroconf.register_service(info)
            logger.info(f"Advertising Petals service on port {port} with model {model_name}")
            logger.info(f"Service peer ID: {peer_id}")
        except Exception as e:
            logger.error(f"Failed to register service: {e}")

    def start_discovery(self) -> None:
        """Start discovering Petals services on the local network."""
        class PetalsListener:
            def __init__(self, outer):
                self.outer = outer

            def add_service(self, zc: Zeroconf, type_: str, name: str) -> None:
                info = zc.get_service_info(type_, name)
                if info:
                    with self.outer.lock:
                        self.outer.services[name] = info
                        logger.info(f"Found Petals service: {name}")

            def remove_service(self, zc: Zeroconf, type_: str, name: str) -> None:
                with self.outer.lock:
                    if name in self.outer.services:
                        del self.outer.services[name]
                        logger.info(f"Petals service removed: {name}")

        self.browser = ServiceBrowser(self.zeroconf, self.SERVICE_TYPE, PetalsListener(self))
        logger.info("Started discovering Petals services")

    def get_peers(self, model_name: Optional[str] = None) -> List[str]:
        """
        Get list of peer addresses for the specified model.
        
        Args:
            model_name: Optional model name to filter by
            
        Returns:
            List of peer addresses in the format required by Petals
        """
        peers = []
        with self.lock:
            for service in self.services.values():
                properties = service.properties
                if model_name and properties.get(b'model', b'').decode() != model_name:
                    continue
                
                # Get the IP address and peer ID from properties
                addr = properties.get(b'ip', b'').decode() or socket.inet_ntoa(service.addresses[0])
                peer_id = properties.get(b'peer_id', b'').decode()
                port = service.port
                
                if not peer_id:
                    logger.warning(f"Service {service.name} doesn't have a peer ID, skipping")
                    continue
                
                try:
                    # Validate the peer ID format
                    decoded = base58.b58decode(peer_id)
                    hash_func, _ = multihash.decode(decoded)
                    if hash_func != self.HASH_FUNC:
                        logger.warning(f"Invalid hash function in peer ID from service {service.name}")
                        continue
                except Exception as e:
                    logger.warning(f"Invalid peer ID format from service {service.name}: {e}")
                    continue
                
                # Format the peer address in Petals format with proper peer ID
                peer = f"/ip4/{addr}/tcp/{port}/p2p/{peer_id}"
                peers.append(peer)
                logger.debug(f"Found peer: {peer}")
        
        return peers

    def stop(self) -> None:
        """Stop service discovery and clean up."""
        if self.browser:
            self.browser.cancel()
        self.zeroconf.close()
        logger.info("Stopped Petals service discovery") 