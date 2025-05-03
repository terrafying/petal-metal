from typing import List, Optional
import socket
import threading
from zeroconf import ServiceBrowser, ServiceInfo, Zeroconf
import logging
import time

logger = logging.getLogger(__name__)

class PetalsServiceDiscovery:
    """
    Service discovery for Petals network using mDNS/Bonjour.
    Particularly optimized for macOS but works cross-platform.
    """
    SERVICE_TYPE = "_petals._tcp.local."
    
    def __init__(self):
        self.zeroconf = Zeroconf()
        self.browser = None
        self.services = {}
        self.lock = threading.Lock()
        
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
        
        info = ServiceInfo(
            type_=self.SERVICE_TYPE,
            name=f"{service_name}.{self.SERVICE_TYPE}",
            addresses=[socket.inet_aton(socket.gethostbyname(hostname))],
            port=port,
            properties={
                'model': model_name,
                'device': device,
                'hostname': hostname
            }
        )
        
        try:
            self.zeroconf.register_service(info)
            logger.info(f"Advertising Petals service on port {port} with model {model_name}")
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
                
                # Get the first IPv4 address
                addr = socket.inet_ntoa(service.addresses[0])
                port = service.port
                
                # Format the peer address in Petals format
                peer = f"/ip4/{addr}/tcp/{port}/p2p/{properties.get(b'hostname', b'').decode()}"
                peers.append(peer)
        
        return peers

    def stop(self) -> None:
        """Stop service discovery and clean up."""
        if self.browser:
            self.browser.cancel()
        self.zeroconf.close()
        logger.info("Stopped Petals service discovery") 