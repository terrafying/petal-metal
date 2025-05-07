import asyncio
import math
import threading
import time
from concurrent.futures import Future
from contextlib import asynccontextmanager
from functools import partial
from typing import Dict, Optional, Set

import requests
from hivemind.dht import DHT, DHTNode
from hivemind.moe.client.remote_expert_worker import RemoteExpertWorker
from hivemind.p2p import P2P, P2PContext, PeerID, ServicerBase
from hivemind.proto import dht_pb2
from hivemind.utils import get_logger
import torch

from petals.constants import REACHABILITY_API_URL

logger = get_logger(__name__)


def validate_reachability(peer_id, wait_time: float = 7 * 60, retry_delay: float = 15) -> None:
    """verify that your peer is reachable from a (centralized) validator, whether directly or through a relay"""
    for attempt_no in range(math.floor(wait_time / retry_delay) + 1):
        try:
            r = requests.get(f"{REACHABILITY_API_URL}/api/v1/is_reachable/{peer_id}", timeout=10)
            r.raise_for_status()
            response = r.json()

            if response["success"]:
                logger.info("Server is reachable from the Internet. It will appear at https://health.petals.dev soon")
                return

            if attempt_no == 0:
                # Usually, libp2p manages to set up relays before we finish loading blocks.
                # In other cases, we may need to wait for up to `wait_time` seconds before it's done.
                logger.info("Detected a NAT or a firewall, connecting to libp2p relays. This takes a few minutes")
            time.sleep(retry_delay)
        except Exception as e:
            logger.warning(f"Skipping reachability check because health.petals.dev is down: {repr(e)}")
            return

    raise RuntimeError(
        f"Server has not become reachable from the Internet:\n\n"
        f"{response['message']}\n\n"
        f"You need to fix your port forwarding and/or firewall settings. How to do that:\n\n"
        f"    1. Choose a specific port for the Petals server, for example, 31337.\n"
        f"    2. Ensure that this port is accessible from the Internet and not blocked by your firewall.\n"
        f"    3. Add these arguments to explicitly announce your IP address and port to other peers:\n"
        f"        python -m petals.cli.run_server ... --public_ip {response['your_ip']} --port 31337\n"
        f"    4. If it does not help, ask for help in our Discord: https://discord.gg/Wuk8BnrEPH\n"
    )


def check_direct_reachability(max_peers: int = 5, threshold: float = 0.5, **kwargs) -> Optional[bool]:
    """test if your peer is accessible by others in the swarm with the specified network options in **kwargs"""

    async def _check_direct_reachability():
        target_dht = await DHTNode.create(client_mode=True, **kwargs)
        try:
            protocol = ReachabilityProtocol(probe=target_dht.protocol.p2p)
            async with protocol.serve(target_dht.protocol.p2p):
                successes = requests = 0
                for remote_peer in list(target_dht.protocol.routing_table.peer_id_to_uid.keys()):
                    probe_available = await protocol.call_check(remote_peer=remote_peer, check_peer=target_dht.peer_id)
                    if probe_available is None:
                        continue  # remote peer failed to check probe
                    successes += probe_available
                    requests += 1
                    if requests >= max_peers:
                        break

            logger.debug(f"Direct reachability: {successes}/{requests}")
            return (successes / requests) >= threshold if requests > 0 else None
        finally:
            await target_dht.shutdown()

    return RemoteExpertWorker.run_coroutine(_check_direct_reachability())


STRIPPED_PROBE_ARGS = dict(
    dht_mode="client", use_relay=False, auto_nat=False, nat_port_map=False, no_listen=True, startup_timeout=60
)


class ReachabilityProtocol:
    """Protocol for checking server reachability."""

    def __init__(self, *, initial_peers: Optional[Set[str]] = None):
        self.initial_peers = initial_peers or set()
        self.reachable_peers: Dict[str, float] = {}

    def update_peer_reachability(self, peer_id: str, is_reachable: bool):
        """Update the reachability status of a peer."""
        if is_reachable:
            self.reachable_peers[peer_id] = torch.cuda.current_time()
        elif peer_id in self.reachable_peers:
            del self.reachable_peers[peer_id]

    def get_reachable_peers(self) -> Set[str]:
        """Get the set of currently reachable peers."""
        return set(self.reachable_peers.keys())

    def shutdown(self):
        """Shutdown the reachability protocol."""
        self.reachable_peers.clear()

    @asynccontextmanager
    async def serve(self, p2p: P2P):
        try:
            await self.add_p2p_handlers(p2p)
            yield self
        finally:
            await self.remove_p2p_handlers(p2p)

    @classmethod
    def attach_to_dht(cls, dht: DHT, await_ready: bool = False, **kwargs) -> Optional["ReachabilityProtocol"]:
        protocol = cls(**kwargs)
        ready = Future()

        async def _serve_with_probe():
            try:
                common_p2p = await dht.replicate_p2p()
                protocol._event_loop = asyncio.get_event_loop()
                protocol._stop = asyncio.Event()

                initial_peers = [str(addr) for addr in await common_p2p.get_visible_maddrs(latest=True)]
                for info in await common_p2p.list_peers():
                    initial_peers.extend(f"{addr}/p2p/{info.peer_id}" for addr in info.addrs)
                protocol.probe = await P2P.create(initial_peers, **STRIPPED_PROBE_ARGS)

                ready.set_result(True)
                logger.debug("Reachability service started")

                async with protocol.serve(common_p2p):
                    await protocol._stop.wait()
            except Exception as e:
                logger.debug("Reachability service failed:", exc_info=True)

                if not ready.done():
                    ready.set_exception(e)
            finally:
                if protocol is not None and protocol.probe is not None:
                    await protocol.probe.shutdown()
                logger.debug("Reachability service shut down")

        threading.Thread(target=partial(asyncio.run, _serve_with_probe()), daemon=True).start()
        if await_ready:
            ready.result()  # Propagates startup exceptions, if any
        return protocol

    def shutdown(self):
        if self._event_loop is not None and self._stop is not None:
            self._event_loop.call_soon_threadsafe(self._stop.set)
