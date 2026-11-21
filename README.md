<div align="center">

# DataVita

**End-to-end data analytics product**

From data preparation and modelling to quality checks and interactive KPI dashboards.

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://datavit.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Portfolio](https://img.shields.io/badge/Portfolio-mounir--gmamsi-1b3550?style=for-the-badge&logo=netlify&logoColor=white)](https://mounir-gmamsi.netlify.app/)

</div>

---

## Overview

DataVita is a complete analytics workflow: prepare and structure data, apply quality checks, analyse KPIs, and present results in an interactive dashboard. It demonstrates practical delivery beyond notebooks â€” a usable data product that stakeholders can explore.

---

## Features

- **ETL / data preparation** with SQL and Python
- **Relational data modelling** (PostgreSQL-oriented design)
- **Data quality** checks and validation-minded profiling
- **KPI analytics** with interactive visualization
- **Export** to Parquet for downstream use
- **ML-ready** components for structured evaluation use cases

---

## Tech stack

### Languages & core

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![SQL](https://img.shields.io/badge/SQL-4479A1?style=flat-square&logo=mysql&logoColor=white)

### Data & databases

![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-4169E1?style=flat-square&logo=postgresql&logoColor=white)
![SQLAlchemy](https://img.shields.io/badge/SQLAlchemy-D71F00?style=flat-square&logo=sqlalchemy&logoColor=white)
![DuckDB](https://img.shields.io/badge/DuckDB-FFF000?style=flat-square&logo=duckdb&logoColor=black)
![Apache Parquet](https://img.shields.io/badge/Parquet-FEFEFE?style=flat-square&logo=apacheparquet&logoColor=black)
![PyArrow](https://img.shields.io/badge/PyArrow-F65E05?style=flat-square&logo=apachearrow&logoColor=white)

### Analytics, ML & UI

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=flat-square&logo=plotly&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![imbalanced-learn](https://img.shields.io/badge/imbalanced--learn-1B7F79?style=flat-square)

### Tooling

![Git](https://img.shields.io/badge/Git-F05032?style=flat-square&logo=git&logoColor=white)
![python-dotenv](https://img.shields.io/badge/python--dotenv-ECD53F?style=flat-square&logo=dotenv&logoColor=black)

| Layer | Technologies |
|-------|----------------|
| Application | Streamlit, Plotly |
| Data processing | Pandas, NumPy, PyArrow |
| Storage / query | PostgreSQL (`psycopg2`), SQLAlchemy, DuckDB |
| Machine learning | scikit-learn, imbalanced-learn |
| Export | Parquet |
| Config | python-dotenv |

---

## Architecture (high level)

```text
Raw / project data
        â”‚
        â–¼
  Prepare & transform   (Python Â· Pandas Â· SQL)
        â”‚
        â–¼
  Relational model      (PostgreSQL concepts Â· SQLAlchemy Â· DuckDB)
        â”‚
        â–¼
  Quality checks        (validation Â· profiling mindset)
        â”‚
        â”œâ”€â”€â–º KPI dashboard   (Streamlit Â· Plotly)
        â””â”€â”€â–º Parquet export  (PyArrow)
```

---

## Repository structure

| Path | Description |
|------|-------------|
| `DataVita_V2.py` | Main Streamlit application |
| `export_to_parquet.py` | Data export utility |
| `data/` | Project data assets |
| `requirements.txt` | Python dependencies |

---

## Getting started

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
# source .venv/bin/activate

pip install -r requirements.txt
streamlit run DataVita_V2.py
```

**Hosted demo:** [https://datavit.streamlit.app/](https://datavit.streamlit.app/)

---

## Author

**Mounir Gmamsi** â€” Data Science / Analytics  
[Portfolio](https://mounir-gmamsi.netlify.app/) Â· [GitHub](https://github.com/Mr-Gms-33) Â· [LinkedIn](https://www.linkedin.com/in/mounir-gmamsi-1595951b3)
