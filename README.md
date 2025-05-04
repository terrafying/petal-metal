# Enhanced Petals for Apple Silicon

This repository extends the [Petals](https://github.com/bigscience-workshop/petals) distributed system with optimizations for Apple Silicon and improved local network discovery. Run large language models like BLOOM-176B and Llama across multiple Apple Silicon machines efficiently.

## 🚀 Key Enhancements

- **Native Apple Silicon Support**: Optimized for MPS (Metal Performance Shaders) with float16 precision
- **Zero-Config Service Discovery**: Automatic peer discovery using mDNS/Bonjour on local networks
- **Resource Management**: Intelligent process locking and resource monitoring to prevent system overload
- **Enhanced Stability**: Graceful handling of peer connections and process management
- **Auto-Swarm Configuration**: Automatically detects and joins existing swarms or creates new ones

## 🔧 Quick Start

### Running a Server Node

```bash
# Clone this repository
git clone https://github.com/yourusername/petals-metal
cd petals-metal

# Install dependencies
pip install -r requirements.txt

# Start a server node - automatically joins existing swarm or creates new one
./run-local.sh
```

The server will automatically:
- Check system resources
- Acquire process locks to prevent multiple instances
- Detect existing Petals swarms on the local network
- Create new swarm if none exists, or join existing swarm
- Advertise itself on the local network
- Use MPS acceleration for optimal performance

### Connecting as a Client

```python
from connect_client import PetalsClient

# Initialize and connect to available peers
client = PetalsClient(model_name="bigscience/bloom-7b1-petals")
try:
    if client.connect():
        # Generate text
        response = client.generate("Hello, how are you?")
        print(response)
finally:
    client.close()
```

## 🛠 Advanced Features

### Unified Service Discovery
Our implementation uses a unified discovery system that combines multiple discovery methods:
```python
from unified_discovery import UnifiedDiscovery

# Start discovering peers
discovery = UnifiedDiscovery()
discovery.start_discovery()

# Get list of available peers
peers = discovery.get_peers(model_name="bigscience/bloom-7b1-petals")

# Check if we should join existing swarm
if peers:
    print(f"Joining existing swarm with peer: {peers[0]}")
else:
    print("Creating new swarm")
```

### Resource Management
The system includes intelligent resource management:
```python
from process_lock import PetalsProcessLock

# Acquire process lock with resource checking
with PetalsProcessLock().acquire_context() as lock:
    # Run your Petals operations
    # System resources are automatically monitored
    # Only one server instance can run at a time
```

## 🔐 System Requirements & Safety

- macOS with Apple Silicon (M1/M2/M3)
- Python 3.8 or higher
- At least 16GB RAM recommended
- Sufficient disk space for model weights

The system automatically monitors:
- CPU usage (prevents overload > 90%)
- Memory usage (prevents overload > 90%)
- Disk space (requires at least 5% free)

## 🤝 Network Configuration

The system uses smart network configuration:
- Automatic port selection (default 31330)
- Unified discovery system combining multiple methods
- Automatic swarm formation and joining
- TCP for peer communication
- File-based locking for process management

### Swarm Behavior
- First node on network creates a new swarm
- Subsequent nodes automatically discover and join existing swarm
- Graceful handling of peer disconnections
- Automatic re-advertisement of services

## 🔍 Debugging & Monitoring

The system provides comprehensive logging:
```python
import logging
logging.basicConfig(level=logging.INFO)
```

Monitor:
- Peer discovery events
- Swarm formation and joining
- Resource usage
- Process locks
- Server status

## 💡 Contributing

This is an enhanced fork of the Petals project, focusing on Apple Silicon optimization and local network usage. Contributions are welcome, especially in areas of:
- Performance optimization for MPS
- Network discovery improvements
- Resource management
- Client/server stability
- Swarm coordination

## 📝 License

This project maintains the same license as the original Petals project, with additional features provided under the same terms.

## 🔗 References

- Original Petals Project: [Petals](https://github.com/bigscience-workshop/petals)
- BLOOM Model: [bigscience/bloom](https://huggingface.co/bigscience/bloom)
- Apple Metal Performance Shaders: [Documentation](https://developer.apple.com/metal/) 