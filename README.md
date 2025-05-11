# Enhanced Petals for Apple Silicon

This repository extends the [Petals](https://github.com/bigscience-workshop/petals) distributed system with optimizations for Apple Silicon, improved local network discovery, and real-time swarm visualization. Run large language models like BLOOM-176B and Llama across multiple Apple Silicon machines efficiently and gain insights into their dynamic behavior.

## 🚀 Key Enhancements

- **Native Apple Silicon Support**: Optimized for MPS (Metal Performance Shaders) with float16 precision
- **Zero-Config Service Discovery**: Automatic peer discovery using mDNS/Bonjour on local networks
- **Resource Management**: Intelligent process locking and resource monitoring to prevent system overload
- **Enhanced Stability**: Graceful handling of peer connections and process management
- **Auto-Swarm Configuration**: Automatically detects and joins existing swarms or creates new ones
- **Real-time Swarm Visualization**: Dynamic Pyglet-based visualization of internal swarm states and pattern evolution.

## 🔧 Quick Start

### Running the Demo with Visualization

The primary way to experience the enhanced features, including the dynamic visualization, is by running the `swarm_vignettes.py` script. This showcases various swarm behaviors and their visual representation.

```bash
# Clone this repository (if you haven't already)
# git clone https://github.com/yourusername/petals-metal
# cd petals-metal

# Install dependencies (ensure pyglet is included)
pip install -r requirements.txt

# Run the pattern evolution swarm demo with Pyglet visualization
python swarm_vignettes.py
```

The demo will automatically:
- Initialize models and the pattern management system.
- Launch a Pyglet window displaying a real-time visualization of the evolving patterns (a "mandala").
- Show dynamic elements like "shooting stars" representing activity (press 'S' to spawn more stars).
- Allow for some interaction (e.g., mouse dragging may perturb the main visual pattern).
- Log detailed information to the console (set to DEBUG level by default in this script).

### Running a Server Node (Headless)

If you need to run a headless server node (e.g., for a distributed setup without immediate local visualization on that node), the `./run-local.sh` script can still be used. (Ensure this script is configured for your desired headless server behavior).

```bash
# To start a server node (potentially without local GUI visualization)
./run-local.sh
```

The server (when run via `./run-local.sh`) typically:
- Checks system resources.
- Acquires process locks to prevent multiple instances.
- Detects existing Petals swarms on the local network.
- Creates a new swarm if none exists, or joins an existing swarm.
- Advertises itself on the local network.
- Uses MPS acceleration for optimal performance.

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

The system provides comprehensive logging and direct visual feedback for monitoring and debugging:

- **Detailed Console Logging**: The `swarm_vignettes.py` script, by default, configures logging to the `DEBUG` level, providing a rich stream of information about:
    ```python
    import logging
    # Example: logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ```
- **Pyglet Visualization Window**: The primary tool for observing the live state of the pattern evolution, including the mandala and dynamic star elements.

Monitor:
- Peer discovery events (if applicable to the running script)
- Swarm formation and joining (if applicable)
- **Real-time pattern evolution via Pyglet visualization**
- **Dynamic event visualization (e.g., shooting stars, tensor perturbations)**
- Resource usage (as logged by underlying processes)
- Process locks (if applicable)
- Server status (if running a server component)
- **Detailed star state changes (position, lifetime) via DEBUG logs**
- **Async task scheduling and execution flow via DEBUG logs**

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