"""
xAI CBAM Attention Map Analysis & Solutions

FINDINGS:
=========

Question 1: What is being visualized?
--------------------------------------
ANSWER: The attention map visualizes the FACE VERIFICATION (embedding extraction) process,
NOT the emotion recognition process.

Evidence:
- Line 1314 in app.py: `image_tensor = val_transform(pil_image).unsqueeze(0).to(DEVICE)`
- Line 1317: `trial_embedding = verification_model(image_tensor, mode='metric').cpu()`
- Line 1478: `self.current_face_tensor = image_tensor` (stores the verification tensor)
- Line 2271: `embedding = self.model(image_tensor, mode='metric')` in explainability.py

The CBAM attention map shows which facial regions the FaceEmbeddingCNN model focuses on
when extracting the 256-dimensional face embedding for verification/matching.

Emotion recognition uses DeepFace (separate model), not the verification model.


Question 2: Why does CBAM focus on image edges (non-facial features)?
----------------------------------------------------------------------
ROOT CAUSES:

1. **Input Preprocessing Artifacts**
   - The face is cropped with padding (crop_face_with_padding in utils.py)
   - Padding adds border regions that may have high contrast
   - cv2.resize() may introduce edge artifacts
   - Color conversion BGR->RGB->PIL->Tensor creates boundary effects

2. **CBAM Channel Attention Mechanism**
   - CBAM applies global pooling which treats all spatial positions equally
   - Edge pixels with high contrast gradients activate strongly
   - Channel attention weights may amplify edge channels

3. **Training Data Bias**
   - If training data had similar padding/edge patterns, model learned to attend to edges
   - LFW dataset preprocessing may have created consistent edge patterns

4. **Grad-CAM Visualization Bias**
   - Grad-CAM uses gradient magnitude, which is high at edges
   - ReLU in CAM generation keeps only positive gradients
   - Upsampling the attention map can blur boundaries

VISUAL EVIDENCE FROM YOUR IMAGE:
- Strong red/yellow attention at top edge
- Strong cyan/green attention at left edge  
- Strong green/yellow attention at right edge
- Bottom edge also shows elevated attention
- Facial features (eyes, nose, mouth) show moderate attention
- Background chin/jaw area shows cyan attention (moderately high)

This is PROBLEMATIC because:
- Edges are not discriminative facial features
- Model may be using image artifacts instead of true face features
- Reduces model robustness and interpretability
- xAI explanation is misleading
