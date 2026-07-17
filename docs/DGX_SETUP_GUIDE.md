# DGX / Cloud GPU Setup Guide

This guide explains how to migrate your XAUUSD Predictor training workflow from a local laptop to an NVIDIA DGX H200 (or any Jupyter-based Cloud GPU environment like RunPod, AWS SageMaker, etc.).

## Why use DGX?
- **No Thermal Throttling:** Avoids local laptop overheating (thermal shutdowns) during heavy XGBoost or FAISS operations.
- **Speed:** Drastically reduces training time.
- **Scalability:** Allows for larger grid searches and hyperparameter tuning.

---

## Step 1: Accessing the Terminal
Most cloud GPU environments provide a Jupyter interface. To run git commands and training scripts, you need a terminal.
1. Open your JupyterLab or Classic Jupyter interface.
2. Click on **File > New > Terminal** (or use the "New" button on the right side and select "Terminal").

## Step 2: Clone the Repository
In the terminal, download your code from GitHub.

```bash
# Clone the repository
git clone https://github.com/Nadex19-Adi/XAUUSD-Predictor.git

# Navigate into the project folder
cd XAUUSD-Predictor
```

> **Note:** If `git` is not installed in your sandbox, you can download the zip directly:
> ```bash
> wget https://github.com/Nadex19-Adi/XAUUSD-Predictor/archive/refs/heads/main.zip
> unzip main.zip
> cd XAUUSD-Predictor-main
> ```

## Step 3: Install Dependencies
Set up the Python environment with the required packages.

```bash
pip install -r requirements.txt
```
*(DGX environments usually have pre-installed libraries like pandas and scikit-learn, but running this ensures all specific dependencies like FAISS and FastAPI are installed).*

## Step 4: Transfer Local Data (If not on GitHub)
Since large datasets (e.g., `xauusd_master_5m.csv`) are ignored in `.gitignore`, you must transfer them manually.

### Option A: Via Jupyter UI (For files < 1GB)
1. On your laptop, compress your `data/` folder into `data.zip`.
2. In the Jupyter web interface, use the **Upload** button to upload `data.zip` to the `XAUUSD-Predictor` directory.
3. Unzip it via terminal:
   ```bash
   unzip data.zip
   ```

### Option B: Via Google Drive (For files > 1GB)
1. Upload `data.zip` to Google Drive and set sharing to "Anyone with the link".
2. Copy the link.
3. In the DGX terminal, run:
   ```bash
   pip install gdown
   gdown "YOUR_GOOGLE_DRIVE_LINK"
   unzip data.zip
   ```

## Step 5: Run the Training
With code and data in place, you can now run your heavy workloads.

**Option A: Run via Terminal (Recommended)**
```bash
# To run the entire pipeline (Resample -> Features -> FAISS -> Train)
python main.py

# Or just the training script
python training/train.py
```

**Option B: Run inside a Jupyter Notebook Cell**
Create a new `.ipynb` file in the project root and run:
```python
!pip install -r requirements.txt
!python training/train.py
```

## GPU Optimization Tip for XGBoost
To fully utilize the DGX H200 GPU, ensure your XGBoost parameters in `src/core/config.py` or your training script include:
```python
"tree_method": "hist",
"device": "cuda"
```
*(For older versions of XGBoost, use `"tree_method": "gpu_hist"`).*
