# Petals on Apple Silicon: Demonstration Guide

This guide will walk you through creating a video or animated GIF demonstration of the Enhanced Petals for Apple Silicon project.

## 🎞️ Capturing Your "Motion"

To get that "Disney-like" animation or an MP4, you'll want to record your screen while performing the steps below. Here are some popular tools:

*   **macOS**: QuickTime Player (comes built-in; File > New Screen Recording)
*   **Cross-platform**: OBS Studio (free, open-source, and powerful)
*   **Other tools**: Kap (GIFs on macOS)
After recording, you can edit the footage to add annotations, zoom in on important details, and create a polished, distributable video.

## 🎬 Demonstration Steps

Follow these steps to showcase the key features of your project.

### Prerequisites

1.  **Clone the Repository**: If you haven't already, clone your project:
    ```bash
    # Replace 'terrafying' with your actual GitHub username/organization if different
    git clone https://github.com/terrafying/petals-2-metal new-patella && cd new-patella
    ```
2.  **Install Dependencies**: Ensure all dependencies are installed:
    ```bash
    pip install -r requirements.txt
    ```

### Step 1: Start the Petals Server Node

1.  Open a new terminal window.
2.  Navigate to your project directory (`petals-metal`).
3.  Run the local server script:
    ```bash
    ./run-local.sh
    ```
4.  **What to look for (and highlight in your recording)**:
    *   Messages about system resource checks.
    *   Logs indicating detection of existing Petals swarms or creation of a new one.
    *   Confirmation that the server is using MPS (Metal Performance Shaders) acceleration.
    *   The server advertising itself on the local network.

### Step 2: Connect and Generate Text with a Client

1.  Open a *second* new terminal window (keep the server running in the first one).
2.  Navigate to your project directory (`petals-metal`).
3.  Run the `demo_client.py` script:
    ```bash
    python demo_client.py
    ```
4.  **What to look for (and highlight in your recording)**:
    *   The client successfully connecting to the swarm/server.
    *   The generated text output (e.g., "Hello, how are you?" followed by the model's response).
    *   The client gracefully closing the connection.

## ✨ "Disney" Touches (Post-Production)

Once you have your screen recording, consider these to enhance it:

*   **Speed up an_parts**: Condense parts where you're just typing or waiting for processes.
*   **Zoom and Pan**: Focus on key terminal outputs or code sections.
*   **Annotations/Text Overlays**: Explain what's happening or highlight important messages from the scripts.
*   **Clear Narration or Background Music**: Depending on your style.

Good luck creating your distributable masterpiece!

---

## 🎬 Advanced Demo: Showcasing Resilience and Self-Organization ("Chaos and Symmetry")

This scenario helps demonstrate the "hidden symmetry" of your system – its ability to self-organize from the "chaos" of individual nodes appearing on the network. It showcases the zero-config discovery and auto-swarm formation features.

**Objective**: To show multiple server nodes discovering each other, forming a swarm, and a client connecting to this distributed intelligence.

### Setup:

*   You'll ideally need two machines on the same local network to run two Petals server nodes simultaneously.
*   If running on a single machine, you would need to ensure your `run-local.sh` script and Petals configuration can support multiple instances (e.g., by using different ports or other isolating mechanisms). This guide assumes the multi-machine setup for clarity, adapt as needed for a single-machine simulation.

### Steps:

1.  **Start the First Server Node**:
    *   On `Machine A` (or your first terminal instance):
        ```bash
        # In your petals-metal directory
        ./run-local.sh
        ```
    *   **Observe**: Note the logs indicating it has started, potentially creating a new swarm if it's the first one.

2.  **Start the Second Server Node**:
    *   On `Machine B` (or your second terminal instance, configured for a separate server):
        ```bash
        # In your petals-metal directory
        ./run-local.sh
        ```
    *   **Observe (The "Symmetry" Emerges)**:
        *   Watch the logs on **both** `Machine A` and `Machine B`.
        *   You should see messages related to mDNS/Bonjour service discovery.
        *   Look for indications that `Machine B` has discovered the swarm created by `Machine A` (or vice-versa if `Machine B` was slightly faster to fully initialize its swarm logic) and has joined it.
        *   This is the "hidden symmetry": nodes automatically finding each other and organizing into a functional unit.

3.  **Connect the Client to the Swarm**:
    *   On a third machine, or one of the server machines (if its resources allow and it doesn't interfere with the server), run the client:
        ```bash
        # In your petals-metal directory
        python demo_client.py
        ```
    *   **Observe**:
        *   The client should connect to the swarm. It doesn't need to know *which* specific server to talk to initially; the swarm handles distributing the work.
        *   The client successfully generates text, demonstrating the collective power of the organized swarm.

### What this Demonstrates (Your "Sphearful e'e"):

*   **Zero-Configuration Discovery**: No manual IP addresses or port configurations were needed for the servers to find each other.
*   **Automatic Swarm Formation**: The nodes intelligently formed a cooperative group.
*   **Distributed Service**: The client can leverage the combined resources of the swarm.
*   **Robustness (Conceptual)**: While harder to demo predictably without more sophisticated tooling, this setup is the foundation for a system that can be resilient to nodes joining or leaving. You can narrate how, if one server node were to drop (gracefully), the client or new clients could still potentially be serviced by the remaining nodes.

This advanced demonstration really brings to life the "enhanced" part of your Petals fork, focusing on the intelligent local networking and self-organization capabilities. 