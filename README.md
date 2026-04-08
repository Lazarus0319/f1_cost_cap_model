# 🏎️ F1 Cost Cap Compliance Model

A Python-based financial compliance tool that tracks, models, and visualises 
Formula 1 team spending against the FIA Financial Regulations (2021–2023).

Built as part of an independent research project into F1 financial regulation,
complementing a published paper on cost cap impact on competitive balance.

---

## 🎯 Why This Exists

The FIA introduced the F1 cost cap in 2021 to level the playing field.
Red Bull Racing became the first — and so far only — team to breach it,
overspending by 1.6% ($2.4M) in 2021 and receiving a $7M fine plus
a 10% reduction in aerodynamic testing time.

This tool models what that compliance monitoring looks like in practice:
tracking monthly spend by category, projecting full-season totals, and
flagging teams approaching or exceeding the cap.

---

## 📊 What It Does

- Tracks 5 teams across 3 seasons (2021, 2022, 2023)
- Models 7 FIA-defined spending categories per team per month
- Projects full-season spend from any mid-season snapshot
- Assigns compliance risk scores: 🔴 RED / 🟡 AMBER / 🟢 GREEN
- Generates static charts (matplotlib) and an interactive dashboard (Streamlit)

---

## 🗂️ Project Structure

---

---

## ⚙️ How To Run

**1. Clone and set up environment**
```bash
git clone https://github.com/Lazarus0319/f1_cost_cap_model.git
cd f1_cost_cap_model
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**2. Run the compliance engine (terminal output)**
```bash
python -m src.compliance_engine
```

**3. Generate static charts**
```bash
python -m src.visualiser
```

**4. Launch interactive dashboard**
```bash
streamlit run dashboard/app.py
```

---

## 📈 Key Findings

| Season | Cap | RedBull | Mercedes | Ferrari | McLaren | Alpine |
|--------|-----|---------|----------|---------|---------|--------|
| 2021 | $147.4M | 🔴 101.6% | 🟡 98.7% | 🟡 97.8% | 🟢 85.5% | 🟢 80.1% |
| 2022 | $142.4M | 🟡 98.4% | 🟡 98.2% | 🟡 97.2% | 🟢 85.0% | 🟢 78.6% |
| 2023 | $137.4M | 🟡 98.4% | 🟡 98.1% | 🟡 96.9% | 🟢 85.9% | 🟢 78.6% |

**RedBull 2021 is the only FIA-confirmed breach in F1 history.**
Post-breach, Red Bull pulled back to ~98.4% in 2022 and 2023 —
consistent with a team that learned exactly where the line is.

---

## 🗃️ Data Sources

| Data Point | Source | Type |
|------------|--------|------|
| 2021 cap: $147.4M | FIA Financial Regulations | ✅ Real |
| 2022 cap: $142.4M | FIA (inflation adjustment) | ✅ Real |
| 2023 cap: $137.4M | FIA ($135M + 2 extra races) | ✅ Real |
| RedBull 2021 spend: $149.8M | FIA Accepted Breach Agreement, Oct 2022 | ✅ Real |
| RedBull 2021 breach: 1.6% | Motor Sport Magazine / FIA | ✅ Real |
| All other team figures | Estimated — actual submissions confidential to FIA CCA | ⚠️ Estimated |
| Category proportions | Based on FIA Financial Regulations structure | ⚠️ Estimated |

> **Note:** The FIA does not publish individual team spending figures for 
> compliant seasons. Estimates are informed by The Race, Motor Sport Magazine, 
> and FIA Financial Regulations documentation. This is standard practice in 
> financial modelling when underlying data is commercially sensitive.

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.13 | Core language |
| Pandas | Data manipulation |
| Matplotlib | Static charts |
| Plotly | Interactive charts |
| Streamlit | Dashboard framework |

---

## 🔗 Related Work

This project complements an independent research paper analysing the impact 
of the FIA cost cap on competitive balance across three regulatory eras 
(2010–2015, 2016–2020, 2021–2024), using HHI, rank persistence, volatility, 
and tier mobility metrics.

📄 Paper available on SSRN (link coming soon)  
💻 Dissertation analysis repo: (link coming soon)

---

## 👤 Author

**Lakshya Agarwal**  
BSc (Hons) Investment and Financial Risk Management, Year 1  
Interested in financial regulation, sports economics, and data analysis.

*Built independently as part of preparation for a placement year application 
in Cost Analysis / Financial Regulation within Formula 1.*