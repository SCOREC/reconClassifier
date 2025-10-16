"""
Model evaluation metrics for X-point detection.

This module provides functions to compute detailed performance metrics
for the X-point detection model, including per-frame and global statistics.
"""

import numpy as np
import json
from pathlib import Path
import torch
from torch.amp import autocast


class ModelEvaluator:
    """
    Evaluates model performance on X-point detection task.
    
    Computes metrics including:
    - True Positives (TP): X-point pixels correctly identified
    - False Positives (FP): Background pixels incorrectly labeled as X-points
    - False Negatives (FN): X-point pixels that were missed
    - True Negatives (TN): Background pixels correctly identified
    
    Metrics calculated:
    - Accuracy: (TP + TN) / (TP + TN + FP + FN)
    - Precision: TP / (TP + FP)
    - Recall: TP / (TP + FN)
    - F1 Score: 2 * (Precision * Recall) / (Precision + Recall)
    - IoU: TP / (TP + FP + FN)
    """
    
    def __init__(self, threshold=0.5):
        """
        Initialize evaluator.
        
        Parameters:
        threshold: float - Probability threshold for binary classification (default: 0.5)
        """
        self.threshold = threshold
        self.reset()
    
    def reset(self):
        """Reset all accumulated metrics."""
        self.global_tp = 0
        self.global_fp = 0
        self.global_fn = 0
        self.global_tn = 0
        self.frame_metrics = []
    
    def compute_frame_metrics(self, pred_probs, ground_truth):
        """
        Compute metrics for a single frame.
        
        Parameters:
        pred_probs: np.ndarray - Predicted probabilities, shape [H, W]
        ground_truth: np.ndarray - Ground truth binary mask, shape [H, W]
        
        Returns:
        dict - Dictionary containing TP, FP, FN, TN and derived metrics
        """
        # Binarize predictions
        pred_binary = (pred_probs > self.threshold).astype(np.float32)
        gt_binary = (ground_truth > 0.5).astype(np.float32)
        
        # Compute confusion matrix elements
        tp = np.sum((pred_binary == 1) & (gt_binary == 1))
        fp = np.sum((pred_binary == 1) & (gt_binary == 0))
        fn = np.sum((pred_binary == 0) & (gt_binary == 1))
        tn = np.sum((pred_binary == 0) & (gt_binary == 0))
        
        # Compute derived metrics
        total = tp + fp + fn + tn
        accuracy = (tp + tn) / total if total > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        
        return {
            'tp': int(tp),
            'fp': int(fp),
            'fn': int(fn),
            'tn': int(tn),
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'iou': float(iou)
        }
    
    def add_frame(self, pred_probs, ground_truth, frame_id=None):
        """
        Add a frame's results to the evaluation.
        
        Parameters:
        pred_probs: np.ndarray - Predicted probabilities
        ground_truth: np.ndarray - Ground truth binary mask
        frame_id: int or str - Optional frame identifier
        """
        metrics = self.compute_frame_metrics(pred_probs, ground_truth)
        
        # Add to global counts
        self.global_tp += metrics['tp']
        self.global_fp += metrics['fp']
        self.global_fn += metrics['fn']
        self.global_tn += metrics['tn']
        
        # Store frame metrics
        if frame_id is not None:
            metrics['frame_id'] = frame_id
        self.frame_metrics.append(metrics)
    
    def get_global_metrics(self):
        """
        Compute global metrics across all frames.
        
        Returns:
        dict - Global metrics computed from accumulated confusion matrix
        """
        total = self.global_tp + self.global_fp + self.global_fn + self.global_tn
        
        metrics = {
            'global_tp': int(self.global_tp),
            'global_fp': int(self.global_fp),
            'global_fn': int(self.global_fn),
            'global_tn': int(self.global_tn),
            'total_pixels': int(total),
            'accuracy': (self.global_tp + self.global_tn) / total if total > 0 else 0.0,
            'precision': self.global_tp / (self.global_tp + self.global_fp) 
                        if (self.global_tp + self.global_fp) > 0 else 0.0,
            'recall': self.global_tp / (self.global_tp + self.global_fn) 
                     if (self.global_tp + self.global_fn) > 0 else 0.0,
            'iou': self.global_tp / (self.global_tp + self.global_fp + self.global_fn)
                  if (self.global_tp + self.global_fp + self.global_fn) > 0 else 0.0,
        }
        
        # Compute F1 from global precision and recall
        if (metrics['precision'] + metrics['recall']) > 0:
            metrics['f1_score'] = 2 * metrics['precision'] * metrics['recall'] / \
                                 (metrics['precision'] + metrics['recall'])
        else:
            metrics['f1_score'] = 0.0
        
        return metrics
    
    def get_frame_statistics(self):
        """
        Compute statistics across all frames.
        
        Returns:
        dict - Mean and standard deviation for each metric across frames
        """
        if not self.frame_metrics:
            return {}
        
        metrics_arrays = {
            key: np.array([frame[key] for frame in self.frame_metrics])
            for key in ['accuracy', 'precision', 'recall', 'f1_score', 'iou']
        }
        
        stats = {}
        for metric_name, values in metrics_arrays.items():
            stats[f'{metric_name}_mean'] = float(np.mean(values))
            stats[f'{metric_name}_std'] = float(np.std(values))
            stats[f'{metric_name}_min'] = float(np.min(values))
            stats[f'{metric_name}_max'] = float(np.max(values))
        
        return stats
    
    def print_summary(self):
        """Print comprehensive evaluation summary."""
        print("\n" + "="*70)
        print("MODEL EVALUATION METRICS")
        print("="*70)
        
        global_metrics = self.get_global_metrics()
        
        print("\nGlobal Metrics (across all frames):")
        print(f"  Total pixels evaluated:    {global_metrics['total_pixels']:,}")
        print(f"  True Positives (TP):       {global_metrics['global_tp']:,}")
        print(f"  False Positives (FP):      {global_metrics['global_fp']:,}")
        print(f"  False Negatives (FN):      {global_metrics['global_fn']:,}")
        print(f"  True Negatives (TN):       {global_metrics['global_tn']:,}")
        print(f"\n  Accuracy:                  {global_metrics['accuracy']:.4f}")
        print(f"  Precision:                 {global_metrics['precision']:.4f}")
        print(f"  Recall:                    {global_metrics['recall']:.4f}")
        print(f"  F1 Score:                  {global_metrics['f1_score']:.4f}")
        print(f"  IoU:                       {global_metrics['iou']:.4f}")
        
        if self.frame_metrics:
            print(f"\nPer-Frame Statistics ({len(self.frame_metrics)} frames):")
            stats = self.get_frame_statistics()
            
            for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'iou']:
                mean = stats[f'{metric}_mean']
                std = stats[f'{metric}_std']
                min_val = stats[f'{metric}_min']
                max_val = stats[f'{metric}_max']
                print(f"  {metric.replace('_', ' ').title():20s} "
                      f"mean={mean:.4f} ±{std:.4f} "
                      f"[{min_val:.4f}, {max_val:.4f}]")
        
        print("="*70 + "\n")
    
    def save_json(self, output_file):
        """
        Save evaluation results to JSON file.
        
        Parameters:
        output_file: Path or str - File path to save evaluation data
        """
        evaluation_data = {
            'global_metrics': self.get_global_metrics(),
            'frame_statistics': self.get_frame_statistics(),
            'per_frame_metrics': self.frame_metrics,
            'threshold': self.threshold,
            'num_frames': len(self.frame_metrics)
        }
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(evaluation_data, f, indent=2)
        
        print(f"Evaluation metrics saved to: {output_path}")


def evaluate_model_on_dataset(model, dataset, device, use_amp=False, 
                              amp_dtype=torch.float16, threshold=0.5):
    """
    Evaluate model on entire dataset and return metrics.
    
    Parameters:
    model: nn.Module - The trained model
    dataset: Dataset - Dataset to evaluate on (XPointDataset, not patch dataset)
    device: torch.device - Device to run evaluation on
    use_amp: bool - Whether to use automatic mixed precision
    amp_dtype: torch.dtype - Data type for mixed precision
    threshold: float - Threshold for binary classification
    
    Returns:
    ModelEvaluator - Evaluator object with computed metrics
    """
    model.eval()
    evaluator = ModelEvaluator(threshold=threshold)
    
    with torch.no_grad():
        for item in dataset:
            fnum = item["fnum"]
            all_torch = item["all"].unsqueeze(0).to(device)
            mask_gt = item["mask"][0].cpu().numpy()  # Remove channel dimension
            
            # Get prediction
            with autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
                pred_mask = model(all_torch)
                pred_prob = torch.sigmoid(pred_mask)
            
            # Convert to numpy (handle BFloat16)
            pred_prob_np = pred_prob[0, 0].float().cpu().numpy()
            
            # Add to evaluator
            evaluator.add_frame(pred_prob_np, mask_gt, frame_id=fnum)
    
    return evaluator