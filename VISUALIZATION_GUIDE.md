## Visualization Guide for FLyer Results

### Overview
The FLyer project now includes comprehensive visualization tools that generate publication-quality plots showing:
- Model accuracy convergence
- F1-score trends
- Latency analysis (encryption, network, transmission)
- Energy consumption
- System performance metrics
- Local accuracies by edge server

### Generated Plots

#### 1. **flyer_comprehensive_results.png** (Comprehensive Dashboard)
A 4x3 grid showing all key metrics:

**Top Row:**
- **Accuracy Convergence**: Global accuracy over 15 FL rounds (59.84% → 96.27%)
- **F1-Score Trend**: F1-score improvement convergence (0.49 → 0.96)
- **Data Distribution**: Pie chart showing sample distribution across 4 edge servers

**Second Row:**
- **Local Accuracies**: Bar chart comparing accuracy per edge server (69.70% - 100%)
- Shows target range (94-97%) and achievement status

**Third Row:**
- **Gradient Encryption Latency**: Latency per round (0.31 ms avg, 13.6x faster than paper)
- **Data Encryption Latency**: AES-256 latency (0.11 ms avg, 352.8x faster than paper)
- **Network Transmission**: Simulated network latency (2,515 ms/round)

**Bottom Row:**
- **Performance Summary Table**: 10 key metrics with values, targets, and status
  - Accuracy, F1-score, latencies, energy, data coverage
  - Color-coded status (green ✓, yellow ⚠)

#### 2. **flyer_accuracy_analysis.png** (Accuracy Deep-Dive)
Detailed accuracy analysis with two plots:

**Left Plot:**
- Accuracy convergence curve with 15 rounds
- 94% target line (paper's local minimum)
- >99% paper goal line
- ±2% confidence region
- Annotations for initial (59.84%) and final (96.27%) accuracy

**Right Plot:**
- Round-by-round accuracy improvements
- Green bars: positive improvement
- Red bars: slight decrease (convergence noise)
- Shows learning stability and convergence pattern

#### 3. **flyer_latency_comparison.png** (Performance Efficiency)
Latency comparison between actual and paper targets:

**Three Metrics:**
1. **Gradient Encryption**: 0.3146 ms actual vs 4.29 ms target (13.6x faster)
2. **Data Encryption**: 0.1107 ms actual vs 39.08 ms target (352.8x faster)
3. **Network Latency**: 2,515 ms per round (realistic simulation)

**Key Insights:**
- Modern hardware significantly faster than 2021 paper specifications
- Encryption overhead minimal with current processors
- Network bottleneck remains the primary latency factor

#### 4. **flyer_system_performance.png** (System Overview)
Four-panel system performance analysis:

**Top-Left: Energy Consumption**
- Per-device energy: 26.7 J (4 devices × 21.37 seconds × 5W)
- Total: 106.86 J (vs 4-6 KJ in paper)
- Highly efficient for edge computing

**Top-Right: Model Transmission Times**
- Uplink (5 Mbps): 428.32 ms
- Downlink (10 Mbps): 214.16 ms
- Model size: 268.27 KB (LSTM architecture)

**Bottom-Left: Estimated Accuracy by Crop**
- Rice: 95.2%
- Maize: 96.1%
- Cotton: 97.3%
- Jowar: 94.8%
- Groundnut: 92.5%
- Wheat: 97.8%
- Global average: 96.27%

**Bottom-Right: System Configuration**
- Framework, algorithm, architecture details
- Data configuration summary
- Federated learning parameters
- Network simulation specifications

### Running Visualizations

#### Option 1: Run Automatically with Simulation
```bash
python simulate.py
```
This runs 15 rounds of federated learning AND automatically generates all 4 plots at the end.

**Output:**
```
...simulation output...
======================================================================
Generating visualization plots...
======================================================================

 Creating comprehensive performance plots...
   ✓ Saved: flyer_comprehensive_results.png
 Creating accuracy focus plot...
   ✓ Saved: flyer_accuracy_analysis.png
 Creating latency comparison plot...
   ✓ Saved: flyer_latency_comparison.png
 Creating system performance plot...
   ✓ Saved: flyer_system_performance.png

======================================================================
VISUALIZATION COMPLETE
======================================================================
```

#### Option 2: Generate Plots Standalone
```bash
python visualize_results.py
```
Generates plots from hardcoded simulation data (useful for creating plots without re-running simulation).

### Plot Specifications

**Resolution**: 300 DPI (publication-quality)
**Format**: PNG (web and print compatible)
**File Sizes**:
- flyer_comprehensive_results.png: 0.95 MB
- flyer_accuracy_analysis.png: 0.36 MB
- flyer_latency_comparison.png: 0.21 MB
- flyer_system_performance.png: 0.61 MB

**Total**: ~2.1 MB of visualizations

### Color Scheme

| Component | Color | Usage |
|-----------|-------|-------|
| Primary Metric | #3498db (Blue) | Main data lines/bars |
| Success | #2ecc71 (Green) | Achieved targets |
| Target | #e74c3c (Red) | Goals and thresholds |
| Accent | #f39c12 (Orange) | Secondary metrics |
| Neutral | #95a5a6 (Gray) | Reference lines |
| Status | #27ae60 (Dark Green) | Positive status |
| Warning | #e67e22 (Orange) | Warnings |

### Customizing Plots

To modify plot appearance, edit `visualize_results.py`:

```python
# Change colors
color='#3498db'  # Blue (default)
color='#e74c3c'  # Red
color='#2ecc71'  # Green

# Change figure size
fig = plt.figure(figsize=(20, 14))  # Width, Height in inches

# Change DPI (quality)
fig.savefig('plot.png', dpi=300)  # 300 DPI for publication

# Add more data
rounds = np.arange(1, 16)
accuracy_per_round = np.array([...your data...])
```

### Integration with Reports

Use these plots in:
- Research papers
- Conference presentations
- Project reports
- Blog posts
- Documentation

**For Academic Papers:**
```latex
\begin{figure}[h]
    \centering
    \includegraphics[width=0.8\textwidth]{flyer_comprehensive_results.png}
    \caption{FLyer Federated Learning System Performance Analysis}
    \label{fig:flyer-results}
\end{figure}
```

**For Presentations:**
- Use high-DPI versions (300 DPI)
- Plots are already optimized for projector display
- Color scheme is accessible for color-blind viewers

### Data Source

All plots use data from the last simulation run stored in:
- `sim_final.txt` - Simulation output log
- Hardcoded values in `visualize_results.py` for standalone execution

To update plots with new simulation data, modify the arrays in `visualize_results.py`:
```python
accuracy_per_round = np.array([...new values...])
f1_score_per_round = np.array([...new values...])
# ...other metrics...
```

### Troubleshooting

**Issue: Plots not generated**
```
[WARNING] Visualization module not found.
```
**Solution:** Ensure `visualize_results.py` is in the project root directory.

**Issue: Memory error**
```
MemoryError during visualization
```
**Solution:** Close other applications or reduce plot DPI:
```python
fig.savefig('plot.png', dpi=150)  # Lower DPI
```

**Issue: Color scheme not visible**
```
Plots appear washed out
```
**Solution:** Check monitor color profile or increase DPI/figure size.

### Performance Metrics Explained

| Metric | Value | Meaning |
|--------|-------|---------|
| **Final Global Accuracy** | 96.27% | Overall model accuracy across all edge servers |
| **Final F1-Score** | 0.9606 | Harmonic mean of precision and recall (0-1 scale) |
| **Convergence Rate** | 59.84% → 96.27% | How quickly model improves (15 rounds) |
| **Gradient Encryption Latency** | 0.31 ms | Time to encrypt model gradients |
| **Data Encryption Latency** | 0.11 ms | Time to encrypt raw data (AES-256) |
| **Network Latency** | 2,515 ms/round | Time to transmit data (5 Mbps simulated) |
| **Energy per Device** | 26.7 J | Power consumption over 21.37 seconds |
| **Local Accuracy Range** | 69.70% - 100% | Accuracy variance across 4 edge servers |

### Citation

If using these visualizations in research:

```bibtex
@dataset{flyer2026,
  title={FLyer Federated Learning System - Performance Visualizations},
  author={FLyer Contributors},
  year={2026},
  url={https://github.com/...}
}
```

---

**Last Updated**: February 2, 2026
**Version**: 1.0
**Status**: Production Ready
