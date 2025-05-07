import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from dataclasses import dataclass
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.spatial.distance import pdist, squareform

@dataclass
class ShapeConfig:
    """Configuration for shape visualization."""
    embedding_dim: int = 768  # Default BERT/GPT embedding dimension
    num_components: int = 3   # For dimensionality reduction
    perplexity: int = 30     # For t-SNE
    learning_rate: float = 200.0
    n_iter: int = 1000
    preserve_ratio: float = 0.95  # Ratio of variance to preserve
    geometric_weight: float = 0.7  # Weight for geometric structure preservation

class ShapeVisualizer:
    def __init__(self, config: Optional[ShapeConfig] = None):
        self.config = config or ShapeConfig()
        self.pca = PCA(n_components=self.config.num_components)
        self.tsne = TSNE(
            n_components=self.config.num_components,
            perplexity=self.config.perplexity,
            learning_rate=self.config.learning_rate,
            n_iter=self.config.n_iter
        )
        
    def _compute_geometric_structure(self, embeddings: np.ndarray) -> np.ndarray:
        """Compute geometric structure matrix using pairwise distances."""
        # Compute pairwise distances
        distances = squareform(pdist(embeddings))
        
        # Normalize distances
        distances = distances / np.max(distances)
        
        # Create geometric structure matrix
        structure_matrix = np.exp(-distances / np.mean(distances))
        
        return structure_matrix
        
    def _preserve_geometric_structure(self, 
                                    original: np.ndarray,
                                    reduced: np.ndarray,
                                    structure_matrix: np.ndarray) -> np.ndarray:
        """Adjust reduced dimensions to better preserve geometric structure."""
        # Compute current pairwise distances in reduced space
        reduced_distances = squareform(pdist(reduced))
        reduced_distances = reduced_distances / np.max(reduced_distances)
        
        # Compute adjustment based on structure preservation
        adjustment = structure_matrix - np.exp(-reduced_distances / np.mean(reduced_distances))
        
        # Apply geometric structure preservation
        adjusted = reduced + self.config.geometric_weight * adjustment.dot(reduced)
        
        return adjusted
        
    def visualize_embedding_space(self, embeddings: torch.Tensor, languages: List[str]) -> plt.Figure:
        """Visualize embeddings in a lower-dimensional space."""
        # Convert embeddings to numpy array
        embeddings_np = embeddings.detach().cpu().numpy()
        
        # Determine number of components based on data size
        n_components = min(3, embeddings_np.shape[0], embeddings_np.shape[1])
        
        # Initialize PCA with adaptive components
        self.pca = PCA(n_components=n_components)
        reduced = self.pca.fit_transform(embeddings_np)
        
        # Create figure
        fig = plt.figure(figsize=(10, 8))
        
        if n_components == 3:
            ax = fig.add_subplot(111, projection='3d')
            scatter = ax.scatter(reduced[:, 0], reduced[:, 1], reduced[:, 2],
                               c=np.arange(len(languages)),
                               cmap='viridis')
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            ax.set_zlabel('PC3')
        else:
            ax = fig.add_subplot(111)
            scatter = ax.scatter(reduced[:, 0], 
                               reduced[:, 1] if reduced.shape[1] > 1 else np.zeros_like(reduced[:, 0]),
                               c=np.arange(len(languages)),
                               cmap='viridis')
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2' if reduced.shape[1] > 1 else '')
        
        plt.colorbar(scatter, label='Language Index')
        plt.title('Embedding Space Visualization')
        
        return fig
        
    def visualize_language_matrix(self,
                                language_weights: Dict[str, float],
                                coupling_matrix: Optional[torch.Tensor] = None) -> plt.Figure:
        """Visualize language weights and coupling matrix with enhanced geometric structure."""
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot language weights with enhanced visualization
        languages = list(language_weights.keys())
        weights = list(language_weights.values())
        
        # Create bar plot with enhanced styling
        bars = sns.barplot(x=languages, y=weights, ax=ax1, palette='viridis')
        ax1.set_title("Language Weights", fontsize=12, pad=20)
        ax1.set_xticklabels(languages, rotation=45, ha='right')
        
        # Add value labels on top of bars
        for bar in bars.patches:
            ax1.text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height(),
                f'{bar.get_height():.2f}',
                ha='center',
                va='bottom'
            )
        
        # Plot coupling matrix if provided
        if coupling_matrix is not None:
            if isinstance(coupling_matrix, torch.Tensor):
                coupling_matrix = coupling_matrix.detach().cpu().numpy()
                
            # Create heatmap with enhanced styling
            sns.heatmap(
                coupling_matrix,
                annot=True,
                cmap='viridis',
                ax=ax2,
                xticklabels=languages,
                yticklabels=languages,
                fmt='.2f',
                cbar_kws={'label': 'Coupling Strength'}
            )
            ax2.set_title("Language Coupling Matrix", fontsize=12, pad=20)
            
        plt.tight_layout()
        return fig
        
    def visualize_pattern_evolution(self,
                                  patterns: List[torch.Tensor],
                                  timesteps: List[int]) -> plt.Figure:
        """Visualize how patterns evolve over time with enhanced geometric structure."""
        # Convert patterns to numpy
        patterns_np = [p.detach().cpu().numpy() if isinstance(p, torch.Tensor) else p 
                      for p in patterns]
        
        # Create figure with enhanced layout
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        # Plot different aspects of pattern evolution
        for i, (pattern, timestep) in enumerate(zip(patterns_np, timesteps)):
            if i < len(axes):
                # Plot pattern magnitude with enhanced styling
                sns.lineplot(
                    data=pattern,
                    ax=axes[i],
                    label=f"Timestep {timestep}",
                    linewidth=2,
                    alpha=0.8
                )
                
                # Add confidence intervals
                if len(pattern.shape) > 1:
                    mean = np.mean(pattern, axis=0)
                    std = np.std(pattern, axis=0)
                    axes[i].fill_between(
                        range(len(mean)),
                        mean - std,
                        mean + std,
                        alpha=0.2
                    )
                
                axes[i].set_title(f"Pattern Evolution at t={timestep}", fontsize=12, pad=20)
                axes[i].grid(True, alpha=0.3)
                
        plt.tight_layout()
        return fig
        
    def visualize_swarm_dynamics(self,
                               positions: List[torch.Tensor],
                               velocities: List[torch.Tensor],
                               languages: List[str]) -> plt.Figure:
        """Visualize swarm dynamics in embedding space with enhanced geometric structure."""
        # Convert to numpy
        positions_np = [p.detach().cpu().numpy() if isinstance(p, torch.Tensor) else p 
                       for p in positions]
        velocities_np = [v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v 
                        for v in velocities]
        
        # Create 3D plot with enhanced styling
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot positions and velocities with enhanced visualization
        for pos, vel, lang in zip(positions_np, velocities_np, languages):
            # Plot position with enhanced styling
            scatter = ax.scatter(
                pos[0], pos[1], pos[2],
                label=lang,
                s=100,
                alpha=0.7
            )
            
            # Plot velocity vector with enhanced styling
            ax.quiver(
                pos[0], pos[1], pos[2],
                vel[0], vel[1], vel[2],
                length=0.2,
                color='gray',
                alpha=0.5,
                arrow_length_ratio=0.3
            )
            
            # Add trajectory line
            ax.plot(
                [pos[0], pos[0] + vel[0]],
                [pos[1], pos[1] + vel[1]],
                [pos[2], pos[2] + vel[2]],
                '--',
                alpha=0.3,
                color='gray'
            )
            
        ax.set_title("Swarm Dynamics in Embedding Space", fontsize=12, pad=20)
        ax.legend(bbox_to_anchor=(1.15, 1))
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        return fig
        
    def visualize_emotional_state(self, emotional_states: List[Dict[str, torch.Tensor]], timesteps: List[int]) -> plt.Figure:
        """Visualize emotional state evolution over time."""
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(111)
        
        # Convert tensors to scalar values
        intensity_values = [state["intensity"].mean().item() for state in emotional_states]
        coherence_values = [state["coherence"].mean().item() for state in emotional_states]
        complexity_values = [state["complexity"].mean().item() for state in emotional_states]
        
        # Plot each component
        ax.plot(timesteps, intensity_values, 'r-', label='Intensity', linewidth=2)
        ax.plot(timesteps, coherence_values, 'g-', label='Coherence', linewidth=2)
        ax.plot(timesteps, complexity_values, 'b-', label='Complexity', linewidth=2)
        
        # Add shaded regions for variance
        ax.fill_between(timesteps, 
                       [max(0, v - 0.1) for v in intensity_values],
                       [min(1, v + 0.1) for v in intensity_values],
                       color='r', alpha=0.2)
        ax.fill_between(timesteps, 
                       [max(0, v - 0.1) for v in coherence_values],
                       [min(1, v + 0.1) for v in coherence_values],
                       color='g', alpha=0.2)
        ax.fill_between(timesteps, 
                       [max(0, v - 0.1) for v in complexity_values],
                       [min(1, v + 0.1) for v in complexity_values],
                       color='b', alpha=0.2)
        
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Emotional State Value')
        ax.set_title('Emotional State Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Set y-axis limits with some padding
        ax.set_ylim(-0.1, 1.1)
        
        return fig

def visualize_swarm_example():
    """Example usage of the shape visualizer with enhanced geometric structure."""
    # Create sample data
    embeddings = torch.randn(10, 768)  # 10 samples, 768-dim embeddings
    labels = [f"Sample {i}" for i in range(10)]
    
    # Language weights and coupling
    language_weights = {
        "en": 1.0,
        "zh": 0.8,
        "ja": 0.7,
        "ko": 0.6
    }
    coupling_matrix = torch.randn(4, 4)
    
    # Pattern evolution
    patterns = [torch.randn(100) for _ in range(4)]
    timesteps = [0, 1, 2, 3]
    
    # Swarm dynamics
    positions = [torch.randn(3) for _ in range(4)]
    velocities = [torch.randn(3) for _ in range(4)]
    languages = ["en", "zh", "ja", "ko"]
    
    # Emotional states
    emotional_states = [
        {"intensity": 0.5, "complexity": 0.6, "coherence": 0.7},
        {"intensity": 0.6, "complexity": 0.7, "coherence": 0.8},
        {"intensity": 0.7, "complexity": 0.8, "coherence": 0.9}
    ]
    
    # Create visualizer with enhanced configuration
    config = ShapeConfig(
        preserve_ratio=0.95,
        geometric_weight=0.7
    )
    visualizer = ShapeVisualizer(config)
    
    # Generate visualizations
    embedding_fig = visualizer.visualize_embedding_space(embeddings, languages)
    language_fig = visualizer.visualize_language_matrix(language_weights, coupling_matrix)
    pattern_fig = visualizer.visualize_pattern_evolution(patterns, timesteps)
    swarm_fig = visualizer.visualize_swarm_dynamics(positions, velocities, languages)
    emotional_fig = visualizer.visualize_emotional_state(emotional_states, [0, 1, 2])
    
    # Show all plots
    plt.show()

if __name__ == "__main__":
    visualize_swarm_example() 