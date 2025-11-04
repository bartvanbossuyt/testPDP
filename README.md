# 🧭 PDP-Analysi

**PDP-Analysis** is a Python framework for qualitative spatiotemporal analysis based on the **Point-Descriptor-Precedence (PDP)** representation.  
It enables the identification, visualization, and comparison of micro-scale movement patterns — such as lane changes, overtakes, or interactions — in both simulated and real-world datasets.

Developed collaboratively at Ghent University, this repository integrates multiple PDP variants and visualization modules designed for research, experimentation, and teaching.

---

## ⚙️ What is PDP?

The **Point-Descriptor-Precedence (PDP)** representation expresses the relative motion between moving objects using relational symbols  
(`<`, `=`, `>`). It captures subtle qualitative differences in movement behavior without relying solely on numeric precision.

### PDP Variants
- **Fundamental** — base qualitative relationships  
- **Buffer** — adds spatial tolerance zones  
- **Rough** — merges nearly identical values  
- **Buffer-Rough** — combines both buffer and roughness effects  

---

## 🧩 Features

- 🧠 **Qualitative transformation:** Convert trajectory data into symbolic PDP representations  
- 📊 **Visual analytics:** Generate heatmaps, hierarchical clusters, MDS, t-SNE, and Top-K plots  
- 🛰️ **Flexible input:** Works with any dataset containing configurations, timestamps, and coordinates  
- 🧩 **Variant comparison:** Analyze and contrast multiple PDP modes  
- 🧾 **Automated reporting:** Create PDF reports combining all visual outputs  
- 👥 **Collaborative setup:** Each team member can use their own dataset and settings  

---

## 📁 Repository Structure
PDP-Analysis/  
│  
├── 📁 scripts/ # Main PDP algorithms and visualization tools  
├── 📁 videos/ # Instructional videos explaining the PDP workflow (shared link)  
├── 📁 visualisations/ # Streamlit-based tools for interactively viewing and analyzing PDP outputs  
├── TO DO 📁 docs/ # Documentation, methodology, and background materials  
├── TO DO 📁 data/ # Local datasets (ignored by Git)  
├── TO DO 📁 results/ # Generated matrices, figures, and reports (ignored by Git)  
│  
├── .gitignore  
├── TO DO requirements.txt  
└── README.md  

---

## 🚀 Quick Start

### 1️⃣ Setup
TO ADD

## 👥 Contributors

This project is developed collaboratively by four team members,  
each applying the PDP framework to their own datasets and experiments.

| Role | Contributor |
|------|--------------|
| Research & Concept | Nico Van de Weghe |
| Codebase & Framework | Collaborative team (Changbo Zhang, Bart Van Bossuyt, Olivier Vermeulen, Jana Verdoodt) |
| Visualization & Documentation | Team members, Ghent University |

---

## 📚 Related Publications

- Qayyum, A., De Baets, B., Baig, M. S., Witlox, F., De Tré, G., & Van de Weghe, N. (2021).  
  *The Point-Descriptor-Precedence representation for point configurations and movements.*  
  *International Journal of Geographical Information Science.*

- Qayyum, A. et al. (2022).  
  *Application of the Point-Descriptor-Precedence representation for micro-scale traffic analysis at a non-signalized T-junction.*  
  *Geo-Spatial Information Science.*

- Qayyum, A. et al. (2023).  
  *Identifying micro-scale lane-changing maneuvers for improving traffic safety.*

---

## 🧩 License

This repository is intended for academic and research use.  
If you use or adapt this framework, please cite the related publications.

---

## 🧭 Acknowledgment

Developed within the **CartoGIS** and **KERMIT** research groups,  
Department of Geography, **Ghent University**.
