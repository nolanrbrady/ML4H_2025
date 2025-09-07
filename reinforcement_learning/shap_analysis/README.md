# 🧠 SHAP Analysis Suite for Reinforcement Learning Models

This directory contains a comprehensive SHAP analysis suite for analyzing PPO, A2C, and SAC models trained on the ALPACA environment.

## 🎯 Quick Start for Paper Writing

### 1. Most Common Tasks (run from `reinforcement_learning/` directory)

```bash
# Quick PPO analysis (2-3 minutes)
python run_shap.py --fast

# Standard PPO analysis (5-10 minutes)  
python run_shap.py

# Compare all algorithms (10-15 minutes)
python run_shap.py --compare --fast

# Detailed analysis for publication (20-30 minutes)
python run_shap.py --detailed
```

### 2. Algorithm-Specific Analysis

```bash
# Analyze specific algorithms
python run_shap.py --algorithm ppo
python run_shap.py --algorithm a2c  
python run_shap.py --algorithm sac
```

### 3. View Results

```bash
# From shap_analysis/ directory - Quick summary of latest results
python shap_summary.py
```

## 📁 File Structure

```
shap_analysis/
├── 📄 Core Scripts
│   ├── unified_shap_analysis.py    # Main multi-algorithm analyzer
│   ├── quick_analysis.py           # Simple interface for paper writing
│   └── shap_summary.py             # Results summary tool
│
├── 📋 Configuration
│   ├── shap_requirements.txt       # Python dependencies
│   └── README.md                   # This file
│
└── 📊 Generated Results (auto-created)
    ├── ppo_shap_analysis_YYYYMMDD_HHMMSS/
    ├── a2c_shap_analysis_YYYYMMDD_HHMMSS/
    ├── sac_shap_analysis_YYYYMMDD_HHMMSS/
    └── comparative_analysis_YYYYMMDD_HHMMSS/
```

## 🚀 Paper Writing Workflow

### Step 1: Run Comparative Analysis
```bash
python run_shap.py --compare
```
**Output**: `comparative_analysis_YYYYMMDD_HHMMSS/`
- `feature_importance_comparison.png` - Main figure for paper
- `treatment_comparison.png` - Treatment-specific analysis
- `comparative_summary.csv` - Statistics table

### Step 2: Individual Algorithm Analysis
```bash
python run_shap.py --algorithm ppo --detailed
python run_shap.py --algorithm a2c --detailed  
python run_shap.py --algorithm sac --detailed
```
**Output**: Individual `{algorithm}_shap_analysis_*/` directories with detailed visualizations

### Step 3: Generate Summary Statistics
```bash
cd shap_analysis
python shap_summary.py
```
**Output**: Console summary with key insights and medical interpretations

## 📊 Key Output Files for Papers

### For Algorithm Comparison Sections:
- `comparative_analysis_*/feature_importance_comparison.png`
- `comparative_analysis_*/top_features_heatmap.png`
- `comparative_analysis_*/treatment_comparison.png`
- `comparative_analysis_*/comparative_summary.csv`

### For Individual Algorithm Analysis:
- `{algorithm}_shap_analysis_*/overall_feature_importance.png`
- `{algorithm}_shap_analysis_*/action_specific_importance_heatmap.png`
- `{algorithm}_shap_analysis_*/summary_AD_Treatment_active.png`
- `{algorithm}_shap_analysis_*/top_features_key_treatments.png`

### For Methods/Supplementary:
- `{algorithm}_shap_analysis_*/waterfall_sample*_*.png` (individual examples)
- `{algorithm}_shap_analysis_*/metadata.csv` (analysis parameters)

## 🖼️ Example Figures

### Individual Algorithm Results

#### PPO Model Analysis
![Overall Feature Importance](ppo_shap_analysis_YYYYMMDD_HHMMSS/overall_feature_importance.png)
*Figure: Overall feature importance for PPO model showing brain volumes and biomarkers driving treatment decisions*

![Action-Specific Heatmap](ppo_shap_analysis_YYYYMMDD_HHMMSS/action_specific_importance_heatmap.png)
*Figure: Treatment-specific feature importance heatmap for PPO model*

![AD Treatment Summary](ppo_shap_analysis_YYYYMMDD_HHMMSS/summary_AD_Treatment_active.png)
*Figure: SHAP summary plot for AD Treatment decisions in PPO model*

![Key Treatment Comparison](ppo_shap_analysis_YYYYMMDD_HHMMSS/top_features_key_treatments.png)
*Figure: Top features influencing key treatment decisions in PPO model*

#### A2C Model Analysis
![Overall Feature Importance](a2c_shap_analysis_YYYYMMDD_HHMMSS/overall_feature_importance.png)
*Figure: Overall feature importance for A2C model*

![Action-Specific Heatmap](a2c_shap_analysis_YYYYMMDD_HHMMSS/action_specific_importance_heatmap.png)
*Figure: Treatment-specific feature importance heatmap for A2C model*

#### SAC Model Analysis
![Overall Feature Importance](sac_shap_analysis_YYYYMMDD_HHMMSS/overall_feature_importance.png)
*Figure: Overall feature importance for SAC model*

![Action-Specific Heatmap](sac_shap_analysis_YYYYMMDD_HHMMSS/action_specific_importance_heatmap.png)
*Figure: Treatment-specific feature importance heatmap for SAC model*

### Comparative Analysis Results
![Algorithm Comparison](comparative_analysis_YYYYMMDD_HHMMSS/feature_importance_comparison.png)
*Figure: Feature importance comparison across PPO, A2C, and SAC models*

![Treatment Comparison](comparative_analysis_YYYYMMDD_HHMMSS/treatment_comparison.png)
*Figure: Treatment-specific importance patterns across algorithms*

## ⚡ Analysis Modes

| Mode | Samples | Time | Use Case |
|------|---------|------|----------|
| **Fast** (`--fast`) | 200 | 2-3 min | Quick testing, iteration |
| **Standard** (default) | 500 | 5-10 min | Most analyses |
| **Detailed** (`--detailed`) | 2000 | 20-30 min | Publication-quality |

## 🔧 Advanced Usage

### Custom Parameters (from shap_analysis/ directory)
```bash
# Custom sample sizes
python unified_shap_analysis.py --algorithm ppo --n_samples 1000 --n_background 100

# Specific model directory
python unified_shap_analysis.py --algorithm ppo --model_dir ../ppo_alpaca_20250614_101546
```

### Batch Analysis for Multiple Runs
```bash
# Analyze all available models
python unified_shap_analysis.py --algorithm all
```

## 📈 Interpreting Results

### Feature Importance Values
- **Higher values** = More influence on treatment decisions
- **Positive SHAP** = Feature increases treatment likelihood
- **Negative SHAP** = Feature decreases treatment likelihood

### Medical Context
- **Brain volumes** (Hippocampus, Entorhinal) = Disease staging
- **Biomarkers** (Tau, Amyloid) = AD pathology indicators  
- **Cognitive scores** (TRABSCOR) = Functional outcomes
- **Demographics** (Age, Gender) = Patient characteristics

### Algorithm Differences
- **PPO**: Policy gradient with clipping
- **A2C**: Advantage actor-critic
- **SAC**: Soft actor-critic (entropy regularization)

## 🏥 Medical Insights

The analysis automatically generates medical interpretations:

✅ **Good Patterns**:
- High importance of cognitive measures
- Use of brain imaging data
- Biomarker consideration
- Age-appropriate treatment selection

⚠️ **Potential Concerns**:
- Low cognitive score importance
- Over-reliance on brain volumes
- Missing demographic considerations

## 🤝 Troubleshooting

### Common Issues

1. **No models found**
   ```
   Error: No {ALGORITHM} model directories found
   ```
   **Solution**: Train models first using `ppo.py`, `a2c.py`, or `sac.py`

2. **Path errors**
   ```
   Error: Must run from reinforcement_learning/ directory
   ```
   **Solution**: Use `python run_shap.py` from the `reinforcement_learning/` directory

3. **Memory issues**
   ```
   OutOfMemoryError during SHAP analysis
   ```
   **Solution**: Use `--fast` mode or reduce custom `--n_samples`

4. **Missing dependencies**
   ```
   ModuleNotFoundError: No module named 'shap'
   ```
   **Solution**: `pip install -r shap_analysis/shap_requirements.txt`

## 🎓 Citation

When using this analysis in publications, consider citing:

- SHAP: Lundberg & Lee (2017). "A Unified Approach to Interpreting Model Predictions"
- Stable-Baselines3: Raffin et al. (2021)
- Your specific RL algorithm papers (PPO, A2C, SAC)

---

**For immediate paper writing**: Start with `python run_shap.py --compare` to get comparative visualizations, then run detailed individual analyses with `python run_shap.py --algorithm {ppo|a2c|sac} --detailed` as needed. 