# Automated Essay Scoring for English Essays of Bangladeshi Secondary Students

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.23924760)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

## Project Overview
State-of-the-art Automated Essay Scoring (AES) systems trained on standardized Western corpora exhibit significant performance degradation when deployed zero-shot on non-native, handwritten student scripts from developing educational contexts. This repository contains the open-science artifacts, dataset, and reproducible execution pipeline for addressing this domain shift in Bangladeshi secondary schools. 

Rather than chasing global benchmark optimization, this study curates the **Bangladeshi Secondary Student Essay Dataset (BSSD)** and engineers a **Warm-Start Hybrid BERT** framework. Through post-hoc Explainable AI (XAI) audits via SHAP, the architecture deconstructs and marginalizes the human grading "Halo Effect"—a local heuristic bias where teachers disproportionately reward text volume over semantic quality.

## Repository Structure
The repository is strictly organized to separate raw data, extraction pipelines, neural architectures, and statistical evaluations. 

- **`data/`**
  * `bssd_anonymized_corpus.csv`: The primary dataset containing $N=308$ digitized essays.
  * `final_preprocessed_data.csv`: Contains the fully processed BSSD corpus, including normalized ground-truth target scores and the extracted 5-dimensional pseudo-trait tensor required for downstream architectural fusion and statistical evaluation.
  * `asap_dataset_prompt_1.csv`: The Western benchmark dataset containing $\sim$1,700 essays from Kaggle's ASAP Prompt 1, utilized strictly to train the baseline BERT teacher model and quantify the zero-shot cross-domain covariate shift.
  * `ocr_audit_logs.csv`: Contains the dataset-wide workflow logs (initial Gemini OCR outputs vs. final Human-in-the-Loop corrected text) used for the OCR degradation and token correction audit..
  * `audit_30_sample_selection.xlsx`: Contains the 30-essay length-stratified random sample utilized for the blind double-transcription Inter-Annotator Agreement (IAA) audit.
- **`artifacts/`**
  * `02_traditional_ml_baselines/`: Serialized TF-IDF vectorizer and tuned Random Forest models (`.joblib`) establishing the traditional machine-learning ceiling.
  * `03_Experiment_A_Western_Baseline/`: Contains the decapitated `western_asap_teacher.pth` backbone and the zero-shot cross-domain predictions.
  * `03_Experiment_B_Hybrid_Local/`: Houses the locked `champion_warm_start_072.pth` PyTorch weights, `trait_scaler.pkl`, and the master `all_predictions.csv` file. These explicitly power the hardware safety lock to guarantee exact reproduction of the published 0.7274 QWK.
  * `03_Experiment_C_Recent_Architectures`: Contains the `xgboost_predictions.csv` artifact, supplying the locked Out-Of-Fold (OOF) predictions for the intermediate Frozen BERT + XGBoost and PCA-reduced tree architectures.
- **`notebooks/`**
  * `01_Data_Preprocessing.ipynb`: BSSD dataset curation, text sanitization, OCR degradation audits, and 5-dimensional pseudo-trait extraction (Mechanics, Casing, Lexical Richness, Prompt Relevance, Word Volume).
  * `02_Traditional_ML_Baselines.ipynb`: TF-IDF baseline evaluations (Random Forest, SVR, XGBoost) establishing the empirical ceiling of shallow architectures.
  * `03_Experiment_A_Western_Baseline.ipynb`: Zero-shot deployment of an ASAP-trained BERT model on the BSSD, empirically quantifying the covariate shift penalty.
  * `03_Experiment_B_Hybrid_Local.ipynb`: The champion **Warm-Start Hybrid BERT** fusing deep semantic token representations with handcrafted linguistic traits.
  * `03_Experiment_C_Recent_Architecture.ipynb`: Evaluation of recent tree-based architectures (XGBoost, PCA) against the BSSD corpus.
  * `04_XAI_Interpretability_and_Feedback.ipynb`: Kernel-SHAP feature ablation and LLM-distilled pedagogical diagnostic feedback generation.
  * `05_Statistical_Significance.ipynb`: ANOVA and fold-wise paired hypothesis testing.
- **`requirements.txt`**: Core environment dependencies.
- **`LICENSE`**: (CC BY-NC 4.0)

## Dataset Characteristics & Ethical Statement
The BSSD comprises handwritten English essays collected from secondary institutions in Bangladesh. Prior to digitization via Human-in-the-Loop OCR, all personally identifiable information (PII) and specific institutional markers were permanently redacted to protect minor populations. 
The provided `bssd_anonymized_corpus.csv` contains the following foundational columns:
- `essay_id`: Unique anonymized identifier.
- `prompt_name`: The curriculum-aligned essay topic.
- `essay_text`: The raw transcribed text retaining structural and syntactical noise.
- `raw_score`: The unedited human teacher evaluation.
- `scale_max`: The maximum possible score boundary.
- `school_id`: Anonymized regional identifier used strictly for Intra-School Z-Score Standardization to mitigate strict/lenient rater bias.

## Key Empirical Findings
1. **The Domain Penalty:** Standard Western models (ASAP-trained) fail when deployed zero-shot on the BSSD, collapsing to a Quadratic Weighted Kappa (QWK) of 0.6189.
2. **The Machine-Learned Halo Effect:** Traditional ML baselines (QWK 0.6987) deceptively perform well by explicitly mimicking human grader fatigue, assigning maximal predictive weight to raw word count rather than linguistic quality.
3. **The Champion Architecture:** The Warm-Start Hybrid BERT architecture achieves a superior QWK of **0.7274**, effectively decoupling structural English proficiency from the localized length bias.

## Reproduction Instructions
The codebase uses dynamic relative pathing (e.g., `../data/`) to ensure out-of-the-box execution across any local or cloud environment. The notebooks are designed for sequential execution.

**1. Environment Setup**
To guarantee cross-platform compatibility and isolate dependencies, we highly recommend utilizing Conda:
```bash
conda create -n aes python=3.8 -y
conda activate aes
pip install -r requirements.txt
```
*(Note for Cloud Users: If executing directly in ephemeral environments like Google Colab, you may bypass Conda and uncomment the inline `!pip install` blocks at the top of each notebook.)*

**2. Artifact-Safe Execution (Notebook 03_B)**
Training a 110-million parameter transformer on a highly constrained localized dataset ($N=308$) introduces inherent Vapnik-Chervonenkis (VC) data starvation, making live retraining susceptible to weight-initialization variance. 
To guarantee strict mathematical reproducibility of the exact metrics published in the study, `03_Experiment_B_Hybrid_Local.ipynb` defaults to an **Artifact-First** paradigm, safely loading the locked historical Out-Of-Fold (OOF) predictions. To execute live stochastic retraining, users must manually flip the execution boolean to `FORCE_RETRAIN = True`.

**3. API Configuration (Optional for Notebook 4)**
Notebook 04 executes a generative feedback Proof-of-Concept requiring the Gemini API. 
To run this specific cell locally, export your free API key to your environment before launching Jupyter/VS Code:
`export GEMINI_API_KEY="your_api_key_here"`
If running in Google Colab, simply add `GEMINI_API_KEY` to your Colab Secrets.

## Citation
If you utilize the BSSD dataset or this Hybrid BERT methodology in your research, please cite our repository via Zenodo:

```bibtex
@dataset{rahaman_uddin_bssd_2026,
  author       = {Rahaman, Tareq and Uddin, Helal},
  title        = {{Automated Essay Scoring for English Essays of Bangladeshi Secondary Students (BSSD)}},
  month        = oct,
  year         = 2026,
  publisher    = {Zenodo},
  version      = {1.0.0},
  doi          = {10.5281/zenodo.23924760.svg},
  url          = {https://doi.org/10.5281/zenodo.23924760}
}
```
