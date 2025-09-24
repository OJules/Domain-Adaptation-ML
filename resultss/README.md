# 📊 Results Directory - Domain Adaptation Analysis

This directory contains outputs, performance metrics, and visualizations from the domain adaptation experiments.

## 📁 Contents

### **Performance Metrics**
- `cross_domain_accuracy.csv` - Classification accuracy across source/target domains
- `adaptation_scores.json` - Domain adaptation performance indicators
- `transfer_learning_metrics.csv` - Feature transferability measurements
- `domain_distance_metrics.json` - Statistical distance between domains

### **Visualizations**
- `domain_alignment_plots.png` - Feature distribution alignment visualization
- `tsne_domain_separation.png` - Domain separation in reduced dimensions
- `adaptation_learning_curves.png` - Training progression across domains
- `confusion_matrices.png` - Classification performance breakdown
- `feature_importance_comparison.png` - Feature relevance across domains

### **Model Outputs**
- `adapted_model_weights.pkl` - Trained domain adaptation models
- `feature_representations.npy` - Learned domain-invariant features
- `adaptation_history.json` - Training history and convergence metrics

### **Comparative Analysis**
- `baseline_vs_adapted.csv` - Performance comparison with/without adaptation
- `method_comparison.png` - Different adaptation techniques comparison
- `robustness_analysis.csv` - Performance under various domain shifts

## 🎯 How to Generate Results

Run the main script to populate this directory:
```bash
python domain_adaptation_main.py
```

The analysis workflow:
1. **Source Domain Training** - Initial model development
2. **Domain Gap Analysis** - Distribution shift measurement
3. **Adaptation Process** - Feature alignment and transfer
4. **Target Domain Evaluation** - Performance assessment

## 📈 Key Performance Indicators

Expected metrics include:
- **Source Accuracy** - Performance on original domain
- **Target Accuracy** - Performance after adaptation
- **Adaptation Gain** - Improvement from domain adaptation
- **Domain Distance** - Statistical measures of domain shift

---

*Note: Results are generated automatically when running the domain adaptation analysis.*
