from typing import Dict, List, Optional, Sequence

from petals.data_structures import UID_DELIMITER

class RemoteSequenceInfo:
    """
    Information about a sequence of remote transformer blocks.
    """

    def __init__(self, block_uids: Sequence[str]):
        self.block_uids = block_uids
        self.block_indices = [int(uid.split(UID_DELIMITER)[1]) for uid in block_uids]
        self.servers: Dict[str, Dict] = {}  # Map of server address to server info

    def add_server(self, server_address: str, server_info: Dict):
        """Add a server to the sequence."""
        self.servers[server_address] = server_info

    def remove_server(self, server_address: str):
        """Remove a server from the sequence."""
        self.servers.pop(server_address, None)

    def get_server_info(self, server_address: str) -> Optional[Dict]:
        """Get information about a server."""
        return self.servers.get(server_address)

    def get_servers(self) -> Dict[str, Dict]:
        """Get all servers in the sequence."""
        return self.servers
