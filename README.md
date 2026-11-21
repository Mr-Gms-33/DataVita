# DataVita

End-to-end **data analytics product**: ingest and model data, enforce quality checks, and deliver interactive insights for decision support.

**Live demo:** [https://datavit.streamlit.app/](https://datavit.streamlit.app/)  
**Author:** Mounir Gmamsi Â· [Portfolio](https://mounir-gmamsi.netlify.app/)

---

## Why this project

Recruiters and hiring managers often ask for proof of **SQL + Python + BI thinking**, not only notebooks. DataVita shows a full path from raw-style data work to a usable dashboard, with attention to **data quality** and a clear **relational model**.

---

## What it includes

- ETL / data preparation with **SQL** and **Python (Pandas)**
- **Relational data model** (PostgreSQL-oriented design)
- **Data quality** checks and profiling-minded validation
- KPI-style analytics and an interactive **Streamlit** dashboard
- Export helpers (e.g. Parquet) for downstream use

---

## Stack

| Layer | Tools |
|-------|--------|
| Language | Python |
| Data | SQL, PostgreSQL concepts, Pandas |
| App | Streamlit |
| Packaging | `requirements.txt` |

---

## Repository layout

| Path | Role |
|------|------|
| `DataVita_V2.py` | Main Streamlit application |
| `export_to_parquet.py` | Data export utility |
| `data/` | Project data assets |
| `requirements.txt` | Python dependencies |

---

## Run locally

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run DataVita_V2.py
```

Or open the hosted demo: [datavit.streamlit.app](https://datavit.streamlit.app/)

---

## Skills demonstrated (for HR / technical screen)

- Data modelling and structured analytics delivery  
- Quality-aware pipelines (checks, documentation mindset)  
- Stakeholder-ready visualization  
- Engineering background applied to data products  

---

## License / use

Portfolio project for learning and interview discussions. Feel free to explore the demo and ask about design choices, metrics, or how I would harden this for production.
