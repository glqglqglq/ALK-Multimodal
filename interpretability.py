import numpy as np
import torch
import shap

def run_modality_attribution(model, background_data, test_data, device):
    """
    Perform SHAP analysis to quantify the contribution of each modality.
    """
    model.eval()
    explainer = shap.GradientExplainer(model, [background_data[0].to(device), background_data[1].to(device)])
    
    # Calculate SHAP values for the ALK-positive class
    shap_values = explainer.shap_values([test_data[0].to(device), test_data[1].to(device)])
    
    # Mean absolute SHAP values for global importance
    pathology_importance = np.abs(shap_values[1][0]).mean()
    radiology_importance = np.abs(shap_values[1][1]).mean()
    
    return pathology_importance, radiology_importance

def compute_linear_cka(feat1, feat2):
    """
    Compute Linear Centered Kernel Alignment (CKA) similarity.
    """
    # Center the features
    feat1 = feat1 - feat1.mean(axis=0)
    feat2 = feat2 - feat2.mean(axis=0)
    
    dot_prod = np.linalg.norm(feat2.T @ feat1, ord='fro')**2
    norm1 = np.linalg.norm(feat1.T @ feat1, ord='fro')
    norm2 = np.linalg.norm(feat2.T @ feat2, ord='fro')
    
    return dot_prod / (norm1 * norm2)