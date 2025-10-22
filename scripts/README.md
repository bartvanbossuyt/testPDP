# 🧩 Scripts

This folder contains the **main Python scripts** that power the **Point-Descriptor-Precedence (PDP)** framework and its visual analytics tools.  
These scripts form the computational core of the *PDP-Analysis* project and handle everything from dataset creation to final reporting.

---

## ⚙️ Overview

The scripts in this folder are designed to:
- Generate and preprocess datasets  
- Transform trajectories into **PDP representations**  
- Compute **inequality** and **distance matrices**  
- Perform **visual analytics** (heatmaps, clustering, MDS, Top-K, etc.)  
- Create **static and dynamic visualizations**  
- Export automated **reports** and figures  

---

## 📜 Main (Base) Files

| Script | Description |
|--------|-------------|
| **`av.py`** | Core configuration file — defines global settings such as PDP variants, buffer/rough parameters, and visualization options. |
| **`GUI.py`** | Optional graphical user interface for activating PDP settings interactively. |
| **`N_D_CreateDataset.py`** | Generates or prepares input datasets for PDP analysis. |
| **`N_PDP.py`** | Core transformation: converts spatiotemporal data into PDP-based inequality and distance matrices. |
| **`N_T_Report.py`** | Generates structured PDF reports compiling static and dynamic visualizations. |
| **`N_T_OB.py`** | Utility script for handling observation-based transformations or object tracking extensions. |
| **`N_Moving_Objects.py`** | Handles data processing for moving-object visualizations and simulation examples. |
| **`N_VA_HeatMap.py`** | Creates heatmaps from PDP distance matrices to visualize configuration differences. |
| **`N_VA_HClust.py`** | Performs hierarchical clustering and generates dendrograms of configuration similarities. |
| **`N_VA_ClusterMap.py`** | Combines clustering and heatmap views for enhanced interpretation of configuration distances. |
| **`N_VA_Mds.py`** | Runs multidimensional scaling (MDS) for low-dimensional representation of configuration similarities. |
| **`N_VA_Mds_autoencoder.py`** | Experimental variant using autoencoders for nonlinear dimensionality reduction. |
| **`N_VA_TopK.py`** | Identifies and plots the top-K most similar configurations per case. |
| **`N_VA_DynamicAbsolute.py`** | Generates dynamic PDP visualizations of moving objects in absolute coordinates. |
| **`N_VA_StaticAbsolute.py`** | Creates static visualizations showing object positions in absolute space. |
| **`N_VA_StaticRelative.py`** | Creates static visualizations relative to a reference frame or object. |
| **`N_VA_StaticFinetuned.py`** | Produces refined, high-quality static visuals for reporting and publications. |
| **`N_VA_Inverse.py`** | Performs inverse or contrast analyses on existing PDP matrices (advanced use). |

---

## 🧠 PDP Variants

Each analysis can run under different **PDP variants**, defined in `av.py`:
- **Fundamental** – base qualitative relationships  
- **Buffer** – adds tolerance zones around each point  
- **Rough** – merges nearly identical values  
- **Buffer-Rough** – combines both mechanisms  

---

## 🚀 Usage

1. Prepare your dataset (columns: `con, tst, id, x, y`)  
2. Adjust parameters and active variants in `av.py`  
3. Run the main PDP script `N_Moving_Objects.py`
