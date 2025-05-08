import time
import torch
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import logging
from pattern_manager import PatternManager, SwarmNode
import asyncio

logger = logging.getLogger(__name__)

@dataclass
class NarrativeThread:
    """Represents a narrative thread with emotional and thematic elements."""
    theme: str
    emotional_valence: float  # -1 to 1
    complexity: float  # 0 to 1
    depth: float  # 0 to 1
    resonance: float  # 0 to 1
    pattern: torch.Tensor
    mandala: str
    concretion: str
    context: Dict
    timestamp: float
    usage_count: int = 0
    affinity_score: float = 1.0

class Storyteller:
    def __init__(self, pattern_manager: PatternManager):
        self.pattern_manager = pattern_manager
        self.narrative_threads: List[NarrativeThread] = []
        self.theme_weights = {
            "harmony": 1.0,
            "transformation": 0.9,
            "resonance": 0.8,
            "emergence": 0.7,
            "synthesis": 0.6
        }
        self.emotional_spectrum = {
            "joy": 1.0,
            "wonder": 0.9,
            "peace": 0.8,
            "awe": 0.7,
            "serenity": 0.6
        }
        
    def _extract_thematic_elements(self, pattern: torch.Tensor) -> Dict:
        """Extract emergent thematic elements from a pattern's geometric structure."""
        # Normalize pattern for some features
        pattern_norm = pattern / (pattern.norm() + 1e-6)

        # Ensure processed tensor has a batch-like dimension for consistent operations
        if pattern.dim() == 2: # e.g. [seq, hidden]
            p_proc = pattern.unsqueeze(0) # -> [1, seq, hidden]
            pn_proc = pattern_norm.unsqueeze(0)
        elif pattern.dim() == 1: # e.g. [hidden]
            p_proc = pattern.unsqueeze(0).unsqueeze(0) # -> [1, 1, hidden]
            pn_proc = pattern_norm.unsqueeze(0).unsqueeze(0)
        else: # Already [batch, seq, hidden] or higher (assuming typical 3D)
            p_proc = pattern
            pn_proc = pattern_norm

        # Calculate geometric/statistical features
        # Features from the original pattern (pre-normalization)
        feat_mean_raw_abs = p_proc.abs().mean().item()
        feat_std_raw = p_proc.std().item()

        # Features from the normalized pattern
        feat_mean_norm_abs = pn_proc.abs().mean().item()
        feat_std_norm = pn_proc.std().item()
        
        feat_skewness_norm = 0.0
        p_flat_norm = pn_proc.flatten()
        if p_flat_norm.numel() > 2 : # Skewness needs at least 3 elements and var > 0
            # Basic skewness: sum((x_i - mu)^3 / n) / sigma^3
            # PyTorch doesn't have a direct torch.skew for older versions.
            # Using a simplified check or a proper library would be better.
            # For now, let's use a placeholder related to mean and median if available or skip
            # If we want a robust skewness, we might need to implement it or ensure PyTorch version.
            # Let's calculate it manually for now, assuming PyTorch 1.8+ for torch.mean and torch.std behavior
            mean = torch.mean(p_flat_norm)
            std = torch.std(p_flat_norm, unbiased=False)
            if std > 1e-6: # Avoid division by zero if std is tiny
                skew_val = torch.mean(((p_flat_norm - mean) / std)**3)
                feat_skewness_norm = skew_val.item()

        feat_roughness_norm = 0.0
        if pn_proc.shape[-1] > 1:
            feat_roughness_norm = pn_proc.diff(dim=-1).abs().mean().item()
        
        calculated_features = {
            "raw_mean_abs": feat_mean_raw_abs,
            "raw_std": feat_std_raw,
            "norm_mean_abs": feat_mean_norm_abs, # Energy of normalized pattern
            "norm_std": feat_std_norm,          # Dispersion of normalized pattern
            "norm_roughness": feat_roughness_norm, # Complexity/Roughness proxy
            "norm_skewness": feat_skewness_norm     # Asymmetry proxy
        }

        # Generate a descriptive theme string
        desc_parts = []
        # Example thresholds (these need tuning based on observed values)
        if calculated_features["norm_mean_abs"] > 0.05: desc_parts.append("active")
        else: desc_parts.append("quiet")

        if calculated_features["norm_std"] > 0.08: desc_parts.append("varied")
        elif calculated_features["norm_std"] < 0.02: desc_parts.append("uniform")
        else: desc_parts.append("moderate_variance")

        if calculated_features["norm_roughness"] > 0.05: desc_parts.append("textured")
        elif calculated_features["norm_roughness"] < 0.01: desc_parts.append("smooth")
        
        if abs(calculated_features["norm_skewness"]) > 0.5:
            skew_dir = "pos_skew" if calculated_features["norm_skewness"] > 0 else "neg_skew"
            desc_parts.append(skew_dir)

        dominant_theme_str = "_" .join(desc_parts) if desc_parts else "neutral_pattern"
        thematic_strength_val = sum(abs(v) for k, v in calculated_features.items() if "norm" in k) / 4.0 # Crude strength

        return {
            "features": calculated_features, 
            "dominant_theme": dominant_theme_str,
            "thematic_strength": thematic_strength_val
        }
        
    def _calculate_emotional_valence(self, pattern: torch.Tensor) -> float:
        """Calculate emotional valence from pattern features."""
        
        # Calculate emotional features by taking the mean across batch and sequence dimensions
        if pattern.dim() > 1:
            mean_features = pattern.mean(dim=tuple(range(pattern.dim() -1)))
        else:
            mean_features = pattern

        if mean_features.dim() > 1:
            mean_features = mean_features.flatten()
            
        features = {
            "joy": mean_features[0].item() if mean_features.numel() > 0 else 0.5,
            "wonder": mean_features[1].item() if mean_features.numel() > 1 else 0.5,
            "peace": mean_features[2].item() if mean_features.numel() > 2 else 0.5,
            "awe": mean_features[3].item() if mean_features.numel() > 3 else 0.5,
            "serenity": mean_features[4].item() if mean_features.numel() > 4 else 0.5
        }
        
        # Calculate weighted emotional valence
        valence = sum(
            features[emotion] * weight 
            for emotion, weight in self.emotional_spectrum.items()
        ) / sum(self.emotional_spectrum.values())
        
        return valence
        
    def _weave_narrative_thread(self, 
                              pattern: torch.Tensor,
                              mandala: str,
                              concretion: str,
                              context: Dict) -> NarrativeThread:
        """Weave a narrative thread from pattern elements."""
        # Extract thematic elements
        thematic_info = self._extract_thematic_elements(pattern)
        context["thematic_features"] = thematic_info["features"] # Store detailed features in context
        
        # Calculate emotional valence
        emotional_valence = self._calculate_emotional_valence(pattern)
        
        # Calculate complexity and depth
        complexity = pattern.std().item()
        depth = pattern.mean().item()
        
        # Calculate resonance with existing threads
        resonance = 0.0
        if self.narrative_threads:
            resonances = [
                self.pattern_manager.compute_resonance(pattern, thread.pattern)
                for thread in self.narrative_threads
            ]
            resonance = max(resonances) if resonances else 0.0
        
        return NarrativeThread(
            theme=thematic_info["dominant_theme"],
            emotional_valence=emotional_valence,
            complexity=complexity,
            depth=depth,
            resonance=resonance,
            pattern=pattern,
            mandala=mandala,
            concretion=concretion,
            context=context,
            timestamp=time.time()
        )
        
    def add_thread(self, 
                  pattern: torch.Tensor,
                  mandala: str,
                  concretion: str,
                  context: Dict) -> None:
        """Add a new narrative thread."""
        thread = self._weave_narrative_thread(pattern, mandala, concretion, context)
        self.narrative_threads.append(thread)
        
        # Update thread affinity scores
        self._update_thread_affinities()
        
    def _update_thread_affinities(self) -> None:
        """Update affinity scores between threads."""
        for i, thread1 in enumerate(self.narrative_threads):
            for j, thread2 in enumerate(self.narrative_threads):
                if i != j:
                    # Calculate pattern similarity
                    similarity = self.pattern_manager.compute_resonance(
                        thread1.pattern,
                        thread2.pattern
                    )
                    
                    # Update affinity scores
                    thread1.affinity_score = max(thread1.affinity_score, similarity)
                    thread2.affinity_score = max(thread2.affinity_score, similarity)
                    
    def get_connected_threads(self, thread: NarrativeThread, threshold: float = 0.5) -> List[NarrativeThread]:
        """Get threads connected to the given thread by affinity."""
        return [
            t for t in self.narrative_threads
            if t != thread and self.pattern_manager.compute_resonance(thread.pattern, t.pattern) > threshold
        ]
        
    def generate_narrative(self, 
                         max_threads: int = 5,
                         min_affinity: float = 0.3) -> Dict:
        """Generate a narrative from connected threads."""
        if not self.narrative_threads:
            return {
                "narrative": "No threads to weave.",
                "themes": [],
                "emotional_arc": [],
                "visual_elements": []
            }
            
        # Sort threads by affinity score
        sorted_threads = sorted(
            self.narrative_threads,
            key=lambda x: x.affinity_score,
            reverse=True
        )
        
        # Select top threads
        selected_threads = sorted_threads[:max_threads]
        
        # Generate narrative elements
        themes = [thread.theme for thread in selected_threads]
        emotional_arc = [thread.emotional_valence for thread in selected_threads]
        visual_elements = [
            {
                "mandala": thread.mandala,
                "concretion": thread.concretion,
                "theme": thread.theme,
                "emotional_valence": thread.emotional_valence
            }
            for thread in selected_threads
        ]
        
        # Weave narrative
        narrative = self._weave_narrative_text(selected_threads)
        
        return {
            "narrative": narrative,
            "themes": themes,
            "emotional_arc": emotional_arc,
            "visual_elements": visual_elements
        }
        
    def _weave_narrative_text(self, threads: List[NarrativeThread]) -> str:
        """Weave narrative text from selected threads."""
        if not threads:
            return ""
            
        # Sort threads by emotional valence for narrative flow
        sorted_threads = sorted(threads, key=lambda x: x.emotional_valence)
        
        # Generate narrative segments
        segments = []
        for thread in sorted_threads:
            # Create segment based on theme and emotional valence
            segment = self._create_narrative_segment(thread)
            segments.append(segment)
            
        # Join segments with transitions
        narrative = self._join_narrative_segments(segments)
        
        return narrative
        
    def _create_narrative_segment(self, thread: NarrativeThread) -> str:
        """Create a narrative segment from a thread's emergent theme."""
        # Map emotional valence to narrative tone
        if thread.emotional_valence > 0.7:
            tone = "uplifting"
        elif thread.emotional_valence > 0.3:
            tone = "contemplative"
        else:
            tone = "mysterious"
            
        # The theme is now thread.theme (e.g., "active_varied_textured")
        # The detailed numerical features are in thread.context["thematic_features"]
        
        theme_description_parts = thread.theme.split('_')
        if not theme_description_parts or thread.theme == "neutral_pattern":
            narrative_element = "a neutral and unassuming pattern"
        else:
            # Reconstruct a more readable description
            # Example: "active_varied_textured" -> "an active, varied, and textured pattern"
            if len(theme_description_parts) == 1:
                desc = theme_description_parts[0]
            elif len(theme_description_parts) == 2:
                desc = f"{theme_description_parts[0]} and {theme_description_parts[1]} pattern"
            else:
                first_parts = ", ".join(theme_description_parts[:-1])
                desc = f"{first_parts}, and {theme_description_parts[-1]} pattern"
            narrative_element = f"a pattern that is {desc}"

        # Optionally, incorporate specific values from thread.context["thematic_features"]
        # For example: norm_std = thread.context["thematic_features"].get("norm_std", 0.0)
        # segment_details = f" (variance: {norm_std:.2f})"
        # For now, keeping it simpler by using the descriptive string primarily.

        segment = f"In the {tone} dance of {narrative_element}, "
        segment += f"its character defined by complexity {thread.complexity:.2f} and depth {thread.depth:.2f}, "
        segment += f"creating a resonance of {thread.resonance:.2f} with what came before."
        
        return segment
        
    def _join_narrative_segments(self, segments: List[str]) -> str:
        """Join narrative segments with appropriate transitions."""
        if not segments:
            return ""
            
        # Define transitions based on emotional flow
        transitions = [
            "As the patterns unfold,",
            "In the space between,",
            "Through the veil of perception,",
            "Within the dance of creation,",
            "Amidst the symphony of forms,"
        ]
        
        # Join segments with transitions
        narrative = segments[0]
        for i, segment in enumerate(segments[1:], 1):
            transition = transitions[i % len(transitions)]
            narrative += f"\n\n{transition} {segment}"
            
        return narrative
        
    def get_narrative_stats(self) -> Dict:
        """Get statistics about the narrative threads."""
        if not self.narrative_threads:
            return {
                "total_threads": 0,
                "thematic_distribution": {},
                "emotional_range": (0, 0),
                "average_complexity": 0,
                "average_depth": 0
            }
            
        # Calculate thematic distribution
        themes = [thread.theme for thread in self.narrative_threads]
        thematic_distribution = {
            theme: themes.count(theme) / len(themes)
            for theme in set(themes)
        }
        
        # Calculate emotional range
        emotional_vals = [thread.emotional_valence for thread in self.narrative_threads]
        emotional_range = (min(emotional_vals), max(emotional_vals))
        
        # Calculate averages
        avg_complexity = sum(t.complexity for t in self.narrative_threads) / len(self.narrative_threads)
        avg_depth = sum(t.depth for t in self.narrative_threads) / len(self.narrative_threads)
        
        return {
            "total_threads": len(self.narrative_threads),
            "thematic_distribution": thematic_distribution,
            "emotional_range": emotional_range,
            "average_complexity": avg_complexity,
            "average_depth": avg_depth
        } 