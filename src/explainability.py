"""
Explainability Module for Face Recognition System

This module provides transparency and interpretability for face recognition decisions:
- Attention map visualization (Grad-CAM on CBAM)
- Natural language explanations
- kNN neighbor visualization
- Uncertainty quantification
- Decision breakdown with confidence scores

Author: Enhanced for Hackathon
Date: November 2025
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import io


class ExplainabilityEngine:
    """
    Main engine for generating explanations for face recognition decisions.
    """
    
    def __init__(self, model, device='cpu'):
        """
        Initialize the explainability engine.
        
        Args:
            model: FaceEmbeddingCNN model instance
            device: torch device (cpu or cuda)
        """
        self.model = model
        self.device = device
        self.model.eval()
        
        # Storage for intermediate activations
        self.activations = {}
        self.gradients = {}
        
        # Register hooks for CBAM attention
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks to capture CBAM attention."""
        def forward_hook(module, input, output, name):
            self.activations[name] = output.detach()
        
        def backward_hook(module, grad_input, grad_output, name):
            self.gradients[name] = grad_output[0].detach()
        
        # Hook into CBAM modules
        for i, module in enumerate(self.model.backbone):
            if hasattr(module, 'channel_att'):  # It's a CBAM module
                module.register_forward_hook(
                    lambda m, inp, out, name=f'cbam_{i}': forward_hook(m, inp, out, name)
                )
                module.register_backward_hook(
                    lambda m, g_in, g_out, name=f'cbam_{i}': backward_hook(m, g_in, g_out, name)
                )
    
    def generate_attention_map(self, image_tensor: torch.Tensor, 
                              target_layer: str = 'cbam_3') -> np.ndarray:
        """
        Generate attention heatmap using Grad-CAM on CBAM.
        
        Args:
            image_tensor: Input image tensor (1, 3, H, W)
            target_layer: Name of the CBAM layer to visualize
        
        Returns:
            Heatmap as numpy array (H, W) with values in [0, 1]
        """
        image_tensor = image_tensor.to(self.device)
        image_tensor.requires_grad = True
        
        # Forward pass
        embedding = self.model(image_tensor, mode='metric')
        
        # Backward pass
        self.model.zero_grad()
        embedding.sum().backward()
        
        # Get activations and gradients
        if target_layer in self.activations and target_layer in self.gradients:
            activation = self.activations[target_layer]
            gradient = self.gradients[target_layer]
            
            # Global average pooling of gradients
            weights = torch.mean(gradient, dim=[2, 3], keepdim=True)
            
            # Weighted combination of activation maps
            cam = torch.sum(weights * activation, dim=1, keepdim=True)
            cam = F.relu(cam)  # ReLU to focus on positive contributions
            
            # Normalize to [0, 1]
            cam = cam - cam.min()
            if cam.max() > 0:
                cam = cam / cam.max()
            
            # Resize to input size
            cam = F.interpolate(cam, size=image_tensor.shape[2:], 
                              mode='bilinear', align_corners=False)
            
            # Convert to numpy for percentile normalization
            cam_np = cam.squeeze().cpu().numpy()
            
            # Percentile normalization to suppress edge artifacts (ignore top 5% outliers)
            p5 = np.percentile(cam_np, 5)
            p95 = np.percentile(cam_np, 95)
            cam_np = np.clip(cam_np, p5, p95)
            cam_np = (cam_np - p5) / (p95 - p5 + 1e-8)
            
            return cam_np
        else:
            # Fallback: return uniform attention
            h, w = image_tensor.shape[2:]
            return np.ones((h, w)) * 0.5
    
    def overlay_attention_on_image(self, image: np.ndarray, 
                                   attention_map: np.ndarray,
                                   alpha: float = 0.5) -> np.ndarray:
        """
        Overlay attention heatmap on original image.
        
        Args:
            image: Original image (H, W, 3) in RGB, values [0, 255]
            attention_map: Attention heatmap (H, W) in [0, 1]
            alpha: Blending factor
        
        Returns:
            Image with heatmap overlay (H, W, 3) in RGB
        """
        # Resize attention map if needed
        if attention_map.shape != image.shape[:2]:
            attention_map = cv2.resize(attention_map, (image.shape[1], image.shape[0]))
        
        # Convert attention map to colormap (red = high attention, blue = low)
        heatmap = cv2.applyColorMap(
            (attention_map * 255).astype(np.uint8), 
            cv2.COLORMAP_JET
        )
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Blend with original image
        overlay = (alpha * heatmap + (1 - alpha) * image).astype(np.uint8)
        
        return overlay
    
    def explain_distance(self, distance: float, threshold: float) -> Dict:
        """
        Explain the verification decision based on distance and threshold.
        
        Args:
            distance: Embedding distance between query and reference
            threshold: Decision threshold
        
        Returns:
            Dictionary with explanation components
        """
        confidence = max(0, min(100, (1 - distance / threshold) * 100))
        
        # Classify confidence level
        if confidence >= 80:
            level = "High"
            color = "green"
            message = "Excellent match with very high certainty"
        elif confidence >= 60:
            level = "Medium"
            color = "yellow"
            message = "Good match with acceptable confidence"
        else:
            level = "Low"
            color = "red"
            message = "Uncertain match, verification may be unreliable"
        
        # Decision
        decision = "Accept" if distance < threshold else "Reject"
        
        # Margin analysis
        margin = abs(distance - threshold)
        margin_percent = (margin / threshold) * 100
        
        if decision == "Accept":
            margin_text = f"Distance is {margin:.3f} below threshold ({margin_percent:.1f}% margin)"
        else:
            margin_text = f"Distance is {margin:.3f} above threshold ({margin_percent:.1f}% margin)"
        
        return {
            'confidence': confidence,
            'level': level,
            'color': color,
            'message': message,
            'decision': decision,
            'distance': distance,
            'threshold': threshold,
            'margin': margin,
            'margin_percent': margin_percent,
            'margin_text': margin_text
        }
    
    def explain_quality_factors(self, image: np.ndarray, 
                                face_bbox: Optional[Tuple] = None) -> Dict:
        """
        Analyze image quality factors affecting recognition.
        
        Args:
            image: Face image (H, W, 3) in RGB
            face_bbox: Optional bounding box (x, y, w, h)
        
        Returns:
            Dictionary with quality metrics and explanations
        """
        # Convert to grayscale for quality checks
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # 1. Blur detection (Laplacian variance)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        blur_quality = "Good" if blur_score > 100 else "Moderate" if blur_score > 50 else "Poor"
        blur_message = f"Blur level: {blur_quality} (score: {blur_score:.1f})"
        
        # 2. Lighting analysis
        brightness = gray.mean()
        brightness_quality = "Good" if 50 < brightness < 200 else "Moderate" if 30 < brightness < 220 else "Poor"
        brightness_message = f"Lighting: {brightness_quality} (brightness: {brightness:.1f})"
        
        # 3. Contrast
        contrast = gray.std()
        contrast_quality = "Good" if contrast > 40 else "Moderate" if contrast > 25 else "Poor"
        contrast_message = f"Contrast: {contrast_quality} (std: {contrast:.1f})"
        
        # 4. Overall quality score (0-100)
        blur_normalized = min(100, blur_score / 2)
        brightness_normalized = 100 * (1 - abs(brightness - 125) / 125)
        contrast_normalized = min(100, contrast * 2)
        
        overall_quality = (blur_normalized * 0.4 + brightness_normalized * 0.3 + 
                          contrast_normalized * 0.3)
        
        return {
            'blur_score': blur_score,
            'blur_quality': blur_quality,
            'blur_message': blur_message,
            'brightness': brightness,
            'brightness_quality': brightness_quality,
            'brightness_message': brightness_message,
            'contrast': contrast,
            'contrast_quality': contrast_quality,
            'contrast_message': contrast_message,
            'overall_quality': overall_quality,
            'overall_message': f"Overall quality: {overall_quality:.1f}/100"
        }
    
    def generate_knn_explanation(self, query_name: str, 
                                 neighbor_names: List[str],
                                 neighbor_distances: List[float]) -> Dict:
        """
        Generate explanation based on kNN neighbors.
        
        Args:
            query_name: Predicted identity
            neighbor_names: List of k-nearest neighbor identities
            neighbor_distances: List of distances to neighbors
        
        Returns:
            Dictionary with kNN analysis
        """
        # Count matches
        matches = [name for name in neighbor_names if name == query_name]
        match_count = len(matches)
        k = len(neighbor_names)
        
        # Agreement percentage
        agreement = (match_count / k) * 100
        
        # Average distance to matching neighbors
        if match_count > 0:
            match_distances = [dist for name, dist in zip(neighbor_names, neighbor_distances) 
                             if name == query_name]
            avg_match_distance = np.mean(match_distances)
        else:
            avg_match_distance = float('inf')
        
        # Confidence classification
        if agreement >= 80:
            confidence_level = "Very High"
            message = f"{match_count}/{k} neighbors agree - strong consensus"
        elif agreement >= 60:
            confidence_level = "High"
            message = f"{match_count}/{k} neighbors agree - good confidence"
        elif agreement >= 40:
            confidence_level = "Moderate"
            message = f"{match_count}/{k} neighbors agree - uncertain"
        else:
            confidence_level = "Low"
            message = f"Only {match_count}/{k} neighbors agree - unreliable"
        
        # Diversity analysis
        unique_identities = len(set(neighbor_names))
        diversity = (unique_identities / k) * 100
        
        return {
            'k': k,
            'matches': match_count,
            'agreement_percent': agreement,
            'confidence_level': confidence_level,
            'message': message,
            'avg_match_distance': avg_match_distance,
            'neighbor_names': neighbor_names,
            'neighbor_distances': neighbor_distances,
            'unique_identities': unique_identities,
            'diversity_percent': diversity
        }
    
    def generate_comprehensive_explanation(self, 
                                          image: np.ndarray,
                                          image_tensor: torch.Tensor,
                                          predicted_name: str,
                                          distance: float,
                                          threshold: float,
                                          neighbor_names: Optional[List[str]] = None,
                                          neighbor_distances: Optional[List[float]] = None) -> Dict:
        """
        Generate comprehensive multi-factor explanation.
        
        Args:
            image: Original face image (H, W, 3)
            image_tensor: Preprocessed tensor (1, 3, H, W)
            predicted_name: Predicted identity
            distance: Embedding distance
            threshold: Decision threshold
            neighbor_names: Optional list of kNN neighbor names
            neighbor_distances: Optional list of kNN distances
        
        Returns:
            Complete explanation dictionary
        """
        explanation = {}
        
        # 1. Distance-based explanation
        explanation['distance'] = self.explain_distance(distance, threshold)
        
        # 2. Quality factors
        explanation['quality'] = self.explain_quality_factors(image)
        
        # 3. Attention map
        attention_map = self.generate_attention_map(image_tensor)
        explanation['attention_map'] = attention_map
        
        # 4. kNN explanation (if available)
        if neighbor_names and neighbor_distances:
            explanation['knn'] = self.generate_knn_explanation(
                predicted_name, neighbor_names, neighbor_distances
            )
        
        # 5. Generate natural language summary
        explanation['summary'] = self._generate_summary(explanation, predicted_name)
        
        return explanation
    
    def _generate_summary(self, explanation: Dict, predicted_name: str) -> str:
        """
        Generate human-readable natural language summary.
        
        Args:
            explanation: Complete explanation dictionary
            predicted_name: Predicted identity
        
        Returns:
            Natural language explanation text
        """
        lines = [f"🎯 Recognition Result: {predicted_name}\n"]
        
        # Distance analysis
        dist_exp = explanation['distance']
        lines.append(f"📊 Confidence: {dist_exp['confidence']:.1f}% ({dist_exp['level']})")
        lines.append(f"   {dist_exp['message']}")
        lines.append(f"   Distance: {dist_exp['distance']:.3f} vs Threshold: {dist_exp['threshold']:.3f}")
        lines.append(f"   {dist_exp['margin_text']}\n")
        
        # Quality analysis
        qual_exp = explanation['quality']
        lines.append(f"📸 Image Quality:")
        lines.append(f"   • {qual_exp['blur_message']}")
        lines.append(f"   • {qual_exp['brightness_message']}")
        lines.append(f"   • {qual_exp['contrast_message']}")
        lines.append(f"   • {qual_exp['overall_message']}\n")
        
        # kNN analysis (if available)
        if 'knn' in explanation:
            knn_exp = explanation['knn']
            lines.append(f"🔍 Neighbor Analysis:")
            lines.append(f"   {knn_exp['message']}")
            lines.append(f"   Agreement: {knn_exp['agreement_percent']:.1f}%")
            lines.append(f"   Avg distance to matches: {knn_exp['avg_match_distance']:.3f}\n")
        
        # Recommendations
        lines.append(f"💡 Recommendations:")
        if qual_exp['overall_quality'] < 60:
            lines.append("   • Improve image quality for better results")
            if qual_exp['blur_quality'] == "Poor":
                lines.append("   • Hold camera steady to reduce blur")
            if qual_exp['brightness_quality'] == "Poor":
                lines.append("   • Adjust lighting conditions")
        if dist_exp['confidence'] < 60:
            lines.append("   • Consider re-registering with multiple poses")
        if dist_exp['decision'] == "Reject":
            lines.append("   • Face not recognized - may need registration")
        
        return '\n'.join(lines)
    
    def visualize_explanation(self, image: np.ndarray,
                             explanation: Dict,
                             save_path: Optional[str] = None) -> np.ndarray:
        """
        Create comprehensive visualization with all explanation components.
        
        Args:
            image: Original image (H, W, 3)
            explanation: Explanation dictionary
            save_path: Optional path to save visualization
        
        Returns:
            Visualization image
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Face Recognition Explanation', fontsize=16, fontweight='bold')
        
        # 1. Original image
        axes[0, 0].imshow(image)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # 2. Attention map overlay
        attention_overlay = self.overlay_attention_on_image(
            image, explanation['attention_map']
        )
        axes[0, 1].imshow(attention_overlay)
        axes[0, 1].set_title('Attention Map (Red=High, Blue=Low)')
        axes[0, 1].axis('off')
        
        # 3. Confidence breakdown
        axes[1, 0].axis('off')
        dist_exp = explanation['distance']
        qual_exp = explanation['quality']
        
        text_content = f"""
DECISION: {dist_exp['decision']}
Confidence: {dist_exp['confidence']:.1f}%

Distance: {dist_exp['distance']:.3f}
Threshold: {dist_exp['threshold']:.3f}
Margin: {dist_exp['margin']:.3f}

Image Quality: {qual_exp['overall_quality']:.1f}/100
• Blur: {qual_exp['blur_quality']}
• Lighting: {qual_exp['brightness_quality']}
• Contrast: {qual_exp['contrast_quality']}
        """
        axes[1, 0].text(0.1, 0.5, text_content, fontsize=10, 
                       verticalalignment='center', family='monospace')
        axes[1, 0].set_title('Decision Breakdown')
        
        # 4. kNN visualization (if available)
        if 'knn' in explanation:
            knn_exp = explanation['knn']
            axes[1, 1].barh(range(knn_exp['k']), 
                           [-d for d in knn_exp['neighbor_distances']])
            axes[1, 1].set_yticks(range(knn_exp['k']))
            axes[1, 1].set_yticklabels([f"N{i+1}: {name}" 
                                       for i, name in enumerate(knn_exp['neighbor_names'])])
            axes[1, 1].set_xlabel('Distance (lower=more similar)')
            axes[1, 1].set_title(f"k-NN Neighbors (k={knn_exp['k']})")
            axes[1, 1].invert_xaxis()
        else:
            axes[1, 1].axis('off')
            axes[1, 1].text(0.5, 0.5, 'kNN data not available', 
                           ha='center', va='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        # Convert to numpy array
        fig.canvas.draw()
        img_array = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
        img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt.close(fig)
        
        return img_array


# Convenience functions
def explain_recognition(model, image, image_tensor, predicted_name, distance, threshold,
                       neighbor_names=None, neighbor_distances=None, device='cpu'):
    """
    One-shot function to generate complete explanation.
    
    Returns:
        Explanation dictionary
    """
    engine = ExplainabilityEngine(model, device)
    return engine.generate_comprehensive_explanation(
        image, image_tensor, predicted_name, distance, threshold,
        neighbor_names, neighbor_distances
    )


def visualize_attention_only(model, image, image_tensor, device='cpu'):
    """
    Quick function to get attention map overlay.
    
    Returns:
        Image with attention overlay (H, W, 3)
    """
    engine = ExplainabilityEngine(model, device)
    attention_map = engine.generate_attention_map(image_tensor)
    return engine.overlay_attention_on_image(image, attention_map)


if __name__ == "__main__":
    print("✓ Explainability module loaded successfully")
    print("\nFeatures:")
    print("  • Attention map visualization (Grad-CAM on CBAM)")
    print("  • Natural language explanations")
    print("  • Quality factor analysis")
    print("  • kNN neighbor explanations")
    print("  • Comprehensive multi-factor explanations")
