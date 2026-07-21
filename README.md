---
title: Emhigh_Analytic
emoji: 🧬
colorFrom: indigo
colorTo: pink
sdk: streamlit
sdk_version: 1.30.0
app_file: app.py
pinned: false
---

# 🧬 Emhigh_Analytic

Emhigh_Analytic is a premium, glassmorphic dark-themed data analytics dashboard built with Python and Streamlit. It provides users with automated data cleaning, interactive visualizations, statistical hypothesis testing, one-click AutoML, and executive PDF summary report generation.

## 🚀 Key Features

* **📊 Dataset Overview:** Instantly inspect dataset shapes, types, missing values, duplicates, and numerical correlations.
* **🧹 Data Cleaning Studio:** Apply structural conversions, impute null values (mean/median/mode/constant/drop), crop/cap IQR outliers, and scale features.
* **📈 Visual Analytics Studio:** Build interactive, customized Plotly plots (Histograms, Box Plots, Scatters, Line/Bar/Pie Charts, Violins) on the fly.
* **🧮 Statistical Hypothesis Hub:** Run independent Two-Sample T-Tests, One-way ANOVA, OLS regressions, and correlation matrices.
* **🤖 AutoML Engines:** Automatically preprocess features, encode categoricals, standardise inputs, train models (classification & regression), and visualises feature importances.
* **📝 Summaries & PDF Builder:** Compile observations alongside dataset metrics into a professional, downloadable PDF summary.

---

## 🛠️ Local Development & Running

1. **Clone the Repository:**
   ```bash
   git clone <your-repository-url>
   cd data_analyst_dashboard
   ```

2. **Create and Activate Virtual Environment:**
   ```bash
   # Windows:
   python -m venv venv
   .\venv\Scripts\activate

   # macOS/Linux:
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch Server:**
   ```bash
   streamlit run app.py
   ```

---

## ☁️ Deployment Guidelines

### Option A: Streamlit Community Cloud (Recommended)
1. Commit all files and push this repository to GitHub.
2. Visit [share.streamlit.io](https://share.streamlit.io/) and log in with your GitHub account.
3. Click **New App**, select your repository, branch (`master` or `main`), and set the main file path to `app.py`.
4. Click **Deploy!** Your app will be live on a public URL in seconds.

### Option B: Hugging Face Spaces
1. Go to [Hugging Face Spaces](https://huggingface.co/spaces) and click **Create new Space**.
2. Name your space, select **Streamlit** as the SDK, and choose a license.
3. Hugging Face will create a repository. Clone it locally, or add it as a remote to this repository:
   ```bash
   git remote add hf https://huggingface.co/spaces/<your-username>/<your-space-name>
   git push -u hf master --force
   ```
4. The YAML metadata block at the top of this `README.md` will configure Hugging Face automatically to run `app.py` with Streamlit.

---

## 📂 Project Directory Structure

```
├── app.py                     # Main Streamlit layout and router
├── data_cleaning.py           # Infill, scaling, duplicates, and outlier processors
├── eda.py                     # Plotly chart definitions and summaries
├── ml_module.py               # AutoML classification & regression pipelines
├── utils.py                   # Caching utilities and ReportLab PDF compiler
├── requirements.txt           # Deployment Python dependencies
├── .gitignore                 # Excluded dynamic directories and temp cache files
└── README.md                  # Hugging Face config and deployment documentation
```
