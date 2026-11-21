<div align="center">

# DataVita

**End-to-end data analytics product**

Data preparation, modelling, quality checks, and interactive KPI dashboards in one deliverable.

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://datavit.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Portfolio](https://img.shields.io/badge/Portfolio-mounir--gmamsi-1b3550?style=for-the-badge&logo=netlify&logoColor=white)](https://mounir-gmamsi.netlify.app/)

</div>

---

## Overview

DataVita is a complete analytics workflow designed as a usable product, not only a notebook. It covers data preparation, relational modelling, quality validation, KPI analysis, and an interactive dashboard that stakeholders can explore.

**Demo:** [datavit.streamlit.app](https://datavit.streamlit.app/)

---

## What the product delivers

| Capability | Description |
|------------|-------------|
| Data preparation | Clean and transform source data for analysis |
| Data modelling | Relational structure for consistent reporting |
| Data quality | Validation checks before insights are published |
| KPI analytics | Metrics and views for decision support |
| Interactive UI | Streamlit dashboard for exploration |
| Export | Parquet output for downstream pipelines |

---

## Technology stack

### Languages

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![SQL](https://img.shields.io/badge/SQL-4479A1?style=flat-square&logo=mysql&logoColor=white)

### Data processing and storage

![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-4169E1?style=flat-square&logo=postgresql&logoColor=white)
![SQLAlchemy](https://img.shields.io/badge/SQLAlchemy-D71F00?style=flat-square&logo=sqlalchemy&logoColor=white)
![DuckDB](https://img.shields.io/badge/DuckDB-FFF000?style=flat-square&logo=duckdb&logoColor=black)
![Parquet](https://img.shields.io/badge/Parquet-FEFEFE?style=flat-square&logo=apacheparquet&logoColor=black)
![PyArrow](https://img.shields.io/badge/PyArrow-F65E05?style=flat-square&logo=apachearrow&logoColor=white)

### Analytics, ML and interface

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=flat-square&logo=plotly&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![imbalanced-learn](https://img.shields.io/badge/imbalanced--learn-1B7F79?style=flat-square)
![Git](https://img.shields.io/badge/Git-F05032?style=flat-square&logo=git&logoColor=white)

| Layer | Technologies |
|-------|----------------|
| Application | Streamlit, Plotly |
| Processing | Python, Pandas, NumPy, PyArrow |
| Query / storage | SQL, PostgreSQL, SQLAlchemy, DuckDB |
| Machine learning | scikit-learn, imbalanced-learn |
| Export | Parquet |
| Configuration | python-dotenv |

---

## Solution flow

| Step | Stage | Technologies |
|:----:|-------|--------------|
| 1 | Ingest project data | Python, Pandas |
| 2 | Prepare and transform | Python, Pandas, SQL |
| 3 | Build relational model | PostgreSQL, SQLAlchemy, DuckDB |
| 4 | Run quality checks | Validation logic in Python |
| 5a | Present KPI dashboard | Streamlit, Plotly |
| 5b | Export for downstream use | Parquet, PyArrow |

```mermaid
flowchart TD
  A[1. Project data] --> B[2. Prepare and transform]
  B --> C[3. Relational model]
  C --> D[4. Quality checks]
  D --> E[5a. KPI dashboard]
  D --> F[5b. Parquet export]
```

---

## Repository structure

| File / folder | Purpose |
|---------------|---------|
| `DataVita_V2.py` | Main Streamlit application |
| `export_to_parquet.py` | Parquet export utility |
| `data/` | Project data assets |
| `requirements.txt` | Python dependencies |

---

## Run locally

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run DataVita_V2.py
```

On macOS or Linux, activate with `source .venv/bin/activate`.

---

## Author

**Mounir Gmamsi**  
[Portfolio](https://mounir-gmamsi.netlify.app/) · [GitHub](https://github.com/Mr-Gms-33) · [LinkedIn](https://www.linkedin.com/in/mounir-gmamsi-1595951b3)
