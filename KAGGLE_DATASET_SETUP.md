# Using Real Kaggle Dataset - Setup Instructions

## 🎯 Quick Start

The code is now ready to use the **real Kaggle dataset**! Follow these steps:

### Step 1: Download the Dataset from Kaggle

1. **Go to Kaggle**: https://www.kaggle.com/datasets/sanchitagholap/crop-and-fertilizer-dataset-for-westernmaharashtra

2. **Click "Download"** button (top-right)
   - You may need a free Kaggle account
   - File size: ~377 KB

3. **Unzip the downloaded file**
   - Look for `Crop and fertilizer dataset.csv`

### Step 2: Place CSV in Flower Directory

1. **Copy the CSV file** to your FLyer directory:
   ```
   y:\Coding\Flower\Crop and fertilizer dataset.csv
   ```

2. The script will **automatically detect** the file when you run it

### Step 3: Run with Real Data

```bash
python data_prep.py
```

**Expected Output**:
```
======================================================================
LOADING AGRICULTURAL DATA
======================================================================

✓ Loading real Kaggle dataset from: y:\Coding\Flower\Crop and fertilizer dataset.csv
  Original dataset shape: (XXXX, XX)
  After filtering crops: XXXX rows
  After filtering districts: XXXX rows
  ✓ Dataset loaded successfully: XXXX samples
  ✓ Crops: ['bananas', 'cassava', 'maize', 'rice', 'seed_cotton', 'yams']
  ✓ Districts: ['Pune', 'Ahmednagar', 'Solapur', 'Satara']

=== FLyer Federated Learning - Data Partitions ===
Total Edge Servers (Tier-III clients): 4
  Edge Server 1 (Pune): Train=X, Test=Y, Total=Z
  Edge Server 2 (Ahmednagar): Train=X, Test=Y, Total=Z
  Edge Server 3 (Solapur): Train=X, Test=Y, Total=Z
  Edge Server 4 (Satara): Train=X, Test=Y, Total=Z

Total Samples: XXXX
  ✓ REAL DATA from Kaggle dataset!
Training/Test Split: XXXX/XXXX
```

### Step 4: Run Federated Learning Simulation

```bash
python simulate.py
```

Now the simulation will use **real agricultural data** from Western Maharashtra! 🎉

---

## ✅ What the Code Does Automatically

The updated `data_prep.py` now:

1. **Searches for the CSV file**
   - Supports multiple filename variations
   - Looks in the current directory

2. **Loads the data**
   - Reads all rows from CSV
   - Cleans column names

3. **Filters for required data**
   - **Keeps 6 crops only**: rice, maize, cassava, seed_cotton, yams, bananas
   - **Keeps 4 districts only**: Pune, Ahmednagar, Solapur, Satara
   - **Extracts 6 features**: Nitrogen, Phosphorus, Potassium, pH, Rainfall, Temperature

4. **Prepares for federated learning**
   - Standardizes (normalizes) features
   - Encodes crop labels (0-5)
   - Splits into train/test (80/20)
   - Partitions across 4 edge servers by district
   - Reshapes for LSTM input (samples, 1 timestep, 6 features)

5. **Fallback mode**
   - If CSV not found, automatically uses **synthetic data**
   - No errors or crashes
   - Prints clear instructions for downloading real data

---

## 📊 Dataset Information

### From Kaggle Dataset
- **Crops** (6): rice, maize, cassava, seed_cotton, yams, bananas
- **Features** (6): Nitrogen, Phosphorus, Potassium, pH, Rainfall, Temperature
- **Districts** (4): Pune, Ahmednagar, Solapur, Satara
- **Total Samples**: Will depend on how many samples are in CSV after filtering

### Expected Statistics (Based on Dataset Description)
- **Original dataset**: ~2,000-3,000 rows
- **After filtering**: Will be determined by actual data
- **Per district**: Samples distributed by district

---

## 🔧 Troubleshooting

### Problem: "Dataset NOT FOUND"

**Solution**: Make sure CSV is in the right location:
```
y:\Coding\Flower\Crop and fertilizer dataset.csv
```

**Supported filenames** (the script will find any of these):
- `Crop and fertilizer dataset.csv`
- `Crop_and_fertilizer_dataset.csv`
- `crop_fertilizer.csv`
- `crop_data.csv`
- Or with full Kaggle name: `crop_and_fertilizer_dataset_for_westernmaharashtra.csv`

**If still not found**:
1. Check the actual filename of your downloaded file
2. Rename it to one of the supported names
3. Ensure it's in `y:\Coding\Flower\`

### Problem: "Missing Columns"

**Possible causes**:
1. Wrong CSV file (verify it's from the correct Kaggle dataset)
2. Column names have extra spaces or different capitalization

**Solution**: The code automatically cleans column names (removes spaces), so this usually works automatically.

### Problem: "Fewer samples than expected"

**Reasons**:
1. Dataset may not have all 6 crops
2. Dataset may not have all 4 districts
3. The original Kaggle dataset may be smaller than expected

**Check**: Run `python data_prep.py` to see filtering results.

---

## 🎯 Running the Full Pipeline with Real Data

```bash
# 1. First, verify data loading works
python data_prep.py

# 2. Then run federated learning with real data
python simulate.py

# 3. Check the results
```

You'll see metrics like:
- **Accuracy**: Should be high with real data (crop recommendations are well-defined)
- **Encryption latency**: Real data with different sample sizes may show slightly different latency
- **Network latency**: Same as before (simulated based on network speeds)

---

## 📈 Expected Differences with Real Data

| Metric | Synthetic Data | Real Data |
|--------|---|---|
| Accuracy | ~95%+ | Typically >95-99%+ |
| Consistency | Very high (artificial) | More realistic variation |
| Sample distribution | Perfectly balanced | May be imbalanced by crop/district |
| Latency | Same (synthetic) | May vary based on sample count |

---

## 🔄 Switching Between Real and Synthetic

You can easily switch back to synthetic data by:
1. Removing/renaming the CSV file
2. Running the code again
3. It automatically falls back to synthetic generation

Or manually in code:
```python
# Force synthetic data
df_kaggle = None
use_synthetic = True
```

---

## 📋 Quick Reference

| Task | Command |
|------|---------|
| Check data | `python data_prep.py` |
| Run FL simulation | `python simulate.py` |
| View current data | Edit `data_prep.py` and print dataset info |
| Switch to synthetic | Remove CSV file or rename it |
| Change district filter | Edit `TARGET_DISTRICTS` in `data_prep.py` |
| Change crop filter | Edit `TARGET_CROPS` in `data_prep.py` |

---

## ✅ When Real Data is Successfully Loaded

You'll know it's working when:

1. ✓ Output shows "✓ Loading real Kaggle dataset from: ..."
2. ✓ Shows "Using REAL data from Kaggle dataset"
3. ✓ Shows "Total Samples: XXXX" (not 5,400 synthetic)
4. ✓ Shows actual crop and district names

---

## 📚 More Information

- **Paper**: FLyer: Federated Learning-Based Crop Yield Prediction for Agriculture 5.0
- **Kaggle Dataset**: https://www.kaggle.com/datasets/sanchitagholap/crop-and-fertilizer-dataset-for-westernmaharashtra
- **Your code directory**: `y:\Coding\Flower\`

---

**Ready to use real agricultural data with your federated learning system!** 🚀
