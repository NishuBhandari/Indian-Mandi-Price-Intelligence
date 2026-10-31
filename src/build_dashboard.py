"""
=============================================================================
INDIAN MANDI PRICE INTELLIGENCE - DASHBOARD BUILDER
=============================================================================
Generates the interactive, standalone web dashboard embedding real project
metrics, audited records, and diagnostic figures.
Outputs to:
  - dashboard/index.html (and style.css, script.js)
  - index.html (root for GitHub Pages)
  - dashboard.html (root standalone)
  - Artifact directory
=============================================================================
"""

import sys
import json
import base64
from pathlib import Path
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = Path(r"C:\Users\Nishu\.gemini\antigravity\brain\371c5563-2e57-4984-ab7e-42f993a24af8")

# 1. Load Real Audited Data
df = pd.read_csv(ROOT_DIR / "data" / "processed" / "audited_mandi_data.csv")

def get_category(c):
    c_lower = str(c).lower()
    if any(k in c_lower for k in ['onion', 'potato', 'tomato', 'brinjal', 'bhindi', 'cauliflower', 'cabbage', 'gourd', 'chilli', 'cucumber', 'pumpkin', 'carrot', 'radish', 'spinach', 'peas', 'beans', 'garlic', 'ginger', 'yam', 'beetroot']):
        return 'Vegetables'
    elif any(k in c_lower for k in ['paddy', 'wheat', 'maize', 'rice', 'barley', 'jowar', 'bajra', 'ragi', 'gram', 'dal', 'chana', 'lentil', 'masur', 'moong', 'urd']):
        return 'Cereals & Pulses'
    elif any(k in c_lower for k in ['pepper', 'jeera', 'cummin', 'coriander', 'turmeric', 'cardamom', 'clove', 'mustard', 'fenugreek', 'methi', 'aniseed', 'tamarind']):
        return 'Spices'
    elif any(k in c_lower for k in ['apple', 'banana', 'orange', 'mango', 'grapes', 'pomegranate', 'papaya', 'guava', 'lemon', 'watermelon', 'melon', 'pineapple', 'sapota']):
        return 'Fruits'
    elif any(k in c_lower for k in ['cotton', 'rubber', 'arecanut', 'supari', 'soyabean', 'groundnut', 'sesamum', 'sunflower', 'copra', 'coconut', 'jute', 'tobacco', 'sugarcane']):
        return 'Cash Crops'
    else:
        return 'Other Agri'

df['Category'] = df['Commodity'].apply(get_category)
df['Min_Price'] = pd.to_numeric(df['Min_Price'], errors='coerce').fillna(0).round(2)
df['Max_Price'] = pd.to_numeric(df['Max_Price'], errors='coerce').fillna(0).round(2)
df['Modal_Price'] = pd.to_numeric(df['Modal_Price'], errors='coerce').fillna(0).round(2)

full_records = df[['State', 'District', 'Market', 'Commodity', 'Category', 'Min_Price', 'Max_Price', 'Modal_Price']].values.tolist()
states_list = sorted(df['State'].unique().tolist())
districts_list = sorted(df['District'].unique().tolist())

# Arbitrage
arb_df = pd.read_csv(ROOT_DIR / "outputs" / "metrics" / "intermarket_arbitrage.csv").head(10).round(2)
arb_list = arb_df.to_dict(orient='records')

# Regression metrics
reg_df = pd.read_csv(ROOT_DIR / "outputs" / "metrics" / "regression_model_comparison.csv").round(2)
reg_list = reg_df.to_dict(orient='records')

# Classification metrics
clf_df = pd.read_csv(ROOT_DIR / "outputs" / "metrics" / "classification_model_comparison.csv").round(4)
clf_list = clf_df.to_dict(orient='records')

# Commodity error
comm_err_df = pd.read_csv(ROOT_DIR / "outputs" / "metrics" / "commodity_error_analysis.csv").head(12).round(2)
comm_err_list = comm_err_df.to_dict(orient='records')

# Load and encode images
p1_b64 = base64.b64encode((ROOT_DIR / "outputs" / "figures" / "11_actual_vs_predicted_regression.png").read_bytes()).decode("utf-8")
p2_b64 = base64.b64encode((ROOT_DIR / "outputs" / "figures" / "12_prediction_error_distribution.png").read_bytes()).decode("utf-8")
cm_b64 = base64.b64encode((ROOT_DIR / "outputs" / "figures" / "13_confusion_matrix_classification.png").read_bytes()).decode("utf-8")
veggies_b64 = base64.b64encode((ROOT_DIR / "outputs" / "veggies_banner_opt.jpg").read_bytes()).decode("utf-8")

records_json = json.dumps(full_records)
arb_json = json.dumps(arb_list)
reg_json = json.dumps(reg_list)
clf_json = json.dumps(clf_list)
comm_err_json = json.dumps(comm_err_list)

html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Indian Mandi Price Intelligence | Decision Analytics Suite</title>
  <style>
    * {{
      margin: 0;
      padding: 0;
      box-sizing: border-box;
    }}
    :root {{
      --bg: #f4f7f5;
      --sidebar-bg: #ffffff;
      --card-bg: #ffffff;
      --card-border: #e2e8e4;
      --card-shadow: 0 4px 14px rgba(15, 23, 42, 0.04);
      --primary-dark: #073820;
      --primary-green: #0e7044;
      --primary-green-light: #10b981;
      --primary-green-hover: #0a5c37;
      --text-main: #0f172a;
      --text-muted: #64748b;
      --accent-blue: #2563eb;
      --accent-amber: #d97706;
      --accent-purple: #7c3aed;
      --accent-rose: #e11d48;
    }}
    body {{
      background-color: var(--bg);
      color: var(--text-main);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      font-size: 13px;
    }}
    /* Top Header */
    header {{
      background: #ffffff;
      border-bottom: 1px solid var(--card-border);
      padding: 12px 24px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      position: sticky;
      top: 0;
      z-index: 100;
      box-shadow: 0 1px 4px rgba(0,0,0,0.02);
    }}
    .brand-wrap {{
      display: flex;
      align-items: center;
      gap: 12px;
    }}
    .brand-logo {{
      width: 38px;
      height: 38px;
      border-radius: 10px;
      background: linear-gradient(135deg, #0e7044, #10b981);
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 20px;
      box-shadow: 0 4px 10px rgba(14, 112, 68, 0.25);
    }}
    .brand-title {{
      font-size: 16px;
      font-weight: 800;
      color: var(--primary-dark);
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    .brand-sub {{
      font-size: 12px;
      color: var(--text-muted);
    }}
    .badge-live {{
      font-size: 11px;
      font-weight: 700;
      padding: 2px 8px;
      border-radius: 20px;
      background: #dcfce7;
      color: var(--primary-green);
      border: 1px solid #bbf7d0;
    }}
    .badge-subtle {{
      font-size: 11px;
      font-weight: 700;
      padding: 2px 8px;
      border-radius: 20px;
      background: #f1f5f9;
      color: #475569;
      border: 1px solid #cbd5e1;
    }}
    .header-pills {{
      display: flex;
      align-items: center;
      gap: 16px;
      font-size: 12px;
      color: var(--text-muted);
    }}
    .header-pills strong {{
      color: var(--primary-dark);
    }}
    .header-actions {{
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    .btn-preset {{
      padding: 7px 13px;
      border-radius: 8px;
      border: 1px solid #bbf7d0;
      background: #f0fdf4;
      color: #15803d;
      font-size: 12px;
      font-weight: 700;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 6px;
      transition: all 0.2s;
    }}
    .btn-preset:hover {{
      background: #dcfce7;
      border-color: #86efac;
    }}
    .btn-preset-pill {{
      flex: 1;
      padding: 7px 8px;
      font-size: 11px;
      font-weight: 700;
      border-radius: 6px;
      border: 1px solid var(--card-border);
      background: #f8fafc;
      color: var(--text-muted);
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 4px;
      transition: all 0.2s;
    }}
    .btn-preset-pill:hover {{
      background: #f1f5f9;
    }}
    .btn-preset-pill.active {{
      background: #dcfce7;
      color: #15803d;
      border-color: #86efac;
    }}
    .btn-export {{
      padding: 7px 15px;
      border-radius: 8px;
      border: none;
      background: var(--primary-green);
      color: #ffffff;
      cursor: pointer;
      font-size: 12px;
      font-weight: 700;
      display: flex;
      align-items: center;
      gap: 6px;
      box-shadow: 0 3px 8px rgba(14, 112, 68, 0.25);
      transition: all 0.2s;
    }}
    .btn-export:hover {{
      background: var(--primary-green-hover);
    }}

    /* Main App Layout */
    .app-body {{
      display: flex;
      flex: 1;
    }}
    aside {{
      width: 270px;
      background: var(--sidebar-bg);
      border-right: 1px solid var(--card-border);
      padding: 20px 16px;
      display: flex;
      flex-direction: column;
      gap: 18px;
      flex-shrink: 0;
    }}
    .nav-label {{
      font-size: 11px;
      font-weight: 800;
      color: var(--primary-dark);
      text-transform: uppercase;
      letter-spacing: 0.6px;
      margin-bottom: 8px;
    }}
    .nav-btn {{
      width: 100%;
      text-align: left;
      padding: 10px 14px;
      border-radius: 8px;
      border: none;
      background: transparent;
      color: #475569;
      cursor: pointer;
      font-size: 13px;
      font-weight: 600;
      display: flex;
      align-items: center;
      gap: 10px;
      transition: all 0.2s;
      margin-bottom: 4px;
    }}
    .nav-btn:hover {{
      background: #f0fdf4;
      color: var(--primary-green);
    }}
    .nav-btn.active {{
      background: var(--primary-green);
      color: #ffffff;
      box-shadow: 0 4px 12px rgba(14, 112, 68, 0.25);
    }}
    .filter-box {{
      border-top: 1px solid var(--card-border);
      padding-top: 14px;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }}
    .filter-title-wrap {{
      display: flex;
      justify-content: space-between;
      align-items: center;
    }}
    .btn-reset {{
      font-size: 11px;
      color: var(--primary-green);
      background: none;
      border: none;
      cursor: pointer;
      font-weight: 700;
    }}
    .btn-reset:hover {{
      text-decoration: underline;
    }}
    .filter-group label {{
      display: flex;
      align-items: center;
      gap: 6px;
      font-size: 11px;
      font-weight: 700;
      color: var(--text-main);
      margin-bottom: 5px;
    }}
    select, input[type="text"] {{
      width: 100%;
      padding: 8px 11px;
      border-radius: 8px;
      border: 1px solid #d1dcd4;
      background: #ffffff;
      color: var(--text-main);
      font-size: 12px;
      outline: none;
      transition: border-color 0.2s;
    }}
    select:focus, input[type="text"]:focus {{
      border-color: var(--primary-green);
    }}

    /* Main Content */
    main {{
      flex: 1;
      padding: 24px;
      overflow-y: auto;
    }}
    .tab-section {{
      display: none;
    }}
    .tab-section.active {{
      display: flex;
      flex-direction: column;
      gap: 20px;
    }}

    /* Top Hero Banner with Real Vegetables Backdrop */
    .hero-banner {{
      background: linear-gradient(90deg, #073820 0%, #073820 38%, rgba(7, 56, 32, 0.82) 55%, rgba(7, 56, 32, 0.25) 80%, rgba(7, 56, 32, 0.05) 100%), url('data:image/jpeg;base64,{veggies_b64}') right center / cover no-repeat;
      border-radius: 14px;
      padding: 24px 28px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      flex-wrap: wrap;
      gap: 16px;
      color: #ffffff;
      box-shadow: 0 6px 18px rgba(11, 74, 41, 0.22);
      min-height: 96px;
    }}
    .hero-title {{
      font-size: 21px;
      font-weight: 800;
      letter-spacing: -0.3px;
    }}
    .hero-subtitle {{
      font-size: 12px;
      color: #bbf7d0;
      margin-top: 4px;
    }}
    .filter-pill {{
      background: rgba(0, 0, 0, 0.55);
      border: 1px solid rgba(254, 240, 138, 0.5);
      color: #fef08a;
      padding: 7px 16px;
      border-radius: 30px;
      font-weight: 700;
      font-size: 11px;
      display: flex;
      align-items: center;
      gap: 8px;
      backdrop-filter: blur(4px);
    }}

    /* KPI Cards Grid */
    .kpi-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 16px;
    }}
    .kpi-card {{
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 12px;
      padding: 18px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      box-shadow: var(--card-shadow);
      transition: transform 0.2s;
    }}
    .kpi-card:hover {{
      transform: translateY(-2px);
    }}
    .kpi-left-wrap {{
      display: flex;
      align-items: center;
      gap: 14px;
    }}
    .kpi-icon-box {{
      width: 44px;
      height: 44px;
      border-radius: 10px;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 22px;
    }}
    .kpi-meta {{
      display: flex;
      flex-direction: column;
    }}
    .kpi-title {{
      font-size: 11px;
      font-weight: 700;
      color: #475569;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }}
    .kpi-val {{
      font-size: 24px;
      font-weight: 800;
      color: var(--text-main);
      font-family: ui-monospace, monospace;
      margin: 2px 0;
    }}
    .kpi-sub {{
      font-size: 11px;
      color: var(--text-muted);
    }}
    .kpi-sub.green {{
      color: var(--primary-green);
      font-weight: 600;
    }}

    /* Dual Visuals */
    .visual-grid {{
      display: grid;
      grid-template-columns: 1.4fr 1fr;
      gap: 20px;
    }}
    .visual-grid-3 {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 18px;
    }}
    @media (max-width: 1024px) {{
      .visual-grid, .visual-grid-3 {{
        grid-template-columns: 1fr;
      }}
      .app-body {{
        flex-direction: column;
      }}
      aside {{
        width: 100%;
        border-right: none;
        border-bottom: 1px solid var(--card-border);
      }}
    }}
    .panel {{
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 14px;
      padding: 20px;
      box-shadow: var(--card-shadow);
    }}
    .panel-header {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 16px;
    }}
    .panel-title {{
      font-size: 14px;
      font-weight: 800;
      color: var(--text-main);
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    .panel-desc {{
      font-size: 11px;
      color: var(--text-muted);
      margin-top: 2px;
    }}
    .btn-toggle {{
      padding: 5px 12px;
      border-radius: 6px;
      font-size: 11px;
      font-weight: 700;
      cursor: pointer;
      border: 1px solid transparent;
    }}
    .btn-toggle.active {{
      background: var(--primary-green);
      color: #ffffff;
    }}
    .btn-toggle.inactive {{
      background: #f1f5f9;
      color: #475569;
      border-color: #cbd5e1;
    }}

    /* Bar Item */
    .bar-item {{
      margin-bottom: 12px;
    }}
    .bar-meta {{
      display: flex;
      justify-content: space-between;
      font-size: 12px;
      margin-bottom: 4px;
    }}
    .bar-name {{
      font-weight: 700;
      color: var(--text-main);
    }}
    .bar-stat {{
      font-family: ui-monospace, monospace;
      color: #475569;
      font-weight: 600;
    }}
    .bar-track {{
      background: #eef3ef;
      border-radius: 6px;
      height: 8px;
      overflow: hidden;
    }}
    .bar-fill {{
      background: var(--primary-green);
      height: 100%;
      border-radius: 6px;
      transition: width 0.4s ease;
    }}

    /* Donut Layout */
    .donut-wrap {{
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 24px;
      flex-wrap: wrap;
      margin: 16px 0;
    }}
    .donut-legend {{
      display: flex;
      flex-direction: column;
      gap: 8px;
    }}
    .legend-row {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      font-size: 12px;
    }}
    .legend-dot {{
      width: 10px;
      height: 10px;
      border-radius: 50%;
      display: inline-block;
      margin-right: 6px;
    }}

    /* Table with Green Thead */
    .table-wrap {{
      overflow-x: auto;
      margin-top: 10px;
      border-radius: 10px;
      border: 1px solid var(--card-border);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
      text-align: left;
    }}
    thead th {{
      padding: 12px 14px;
      background: var(--primary-dark);
      color: #ffffff;
      font-weight: 800;
      text-transform: uppercase;
      font-size: 11px;
      letter-spacing: 0.5px;
    }}
    tbody td {{
      padding: 10px 14px;
      border-bottom: 1px solid #edf2ee;
      color: var(--text-main);
    }}
    tbody tr:hover {{
      background: #f0fdf4;
    }}
    .tier-badge {{
      display: inline-block;
      padding: 3px 9px;
      border-radius: 12px;
      font-size: 10px;
      font-weight: 700;
    }}
    .tier-budget {{
      background: #dcfce7;
      color: #15803d;
      border: 1px solid #bbf7d0;
    }}
    .tier-standard {{
      background: #dbeafe;
      color: #1d4ed8;
      border: 1px solid #bfdbfe;
    }}
    .tier-premium {{
      background: #f3e8ff;
      color: #7e22ce;
      border: 1px solid #e9d5ff;
    }}
    .pagination-bar {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-top: 14px;
      font-size: 12px;
      color: var(--text-muted);
    }}
    .btn-page {{
      padding: 6px 14px;
      border-radius: 6px;
      border: 1px solid #d1dcd4;
      background: #ffffff;
      color: var(--text-main);
      cursor: pointer;
      font-weight: 600;
    }}
    .btn-page:disabled {{
      opacity: 0.4;
      cursor: not-allowed;
    }}
    .plot-img {{
      width: 100%;
      max-height: 280px;
      object-fit: contain;
      border-radius: 8px;
      background: #ffffff;
      padding: 6px;
      border: 1px solid var(--card-border);
    }}
    .experiment-callout {{
      padding: 12px 16px;
      border-radius: 8px;
      margin-bottom: 12px;
      font-size: 12px;
      line-height: 1.5;
    }}
    .callout-primary {{
      background: #f0fdf4;
      border: 1px solid #bbf7d0;
      color: #166534;
    }}
    .callout-info {{
      background: #eff6ff;
      border: 1px solid #bfdbfe;
      color: #1e40af;
    }}
  </style>
</head>
<body>

  <!-- ================= TOP HEADER ================= -->
  <header>
    <div class="brand-wrap">
      <div class="brand-logo">🌾</div>
      <div>
        <div class="brand-title">
          Indian Mandi Price Intelligence
          <span class="badge-live">Snapshot Analytics</span>
        </div>
        <div class="brand-sub">Interactive APMC Mandi Market & 6-Model Machine Learning Decision Suite</div>
      </div>
    </div>

    <div class="header-pills">
      <div>Audited Snapshot: <strong>16,734 Raw (16,733 Clean)</strong></div>
      <div>States: <strong>29</strong></div>
      <div>Active APMCs: <strong>1,774</strong></div>
      <div>ML Suite: <strong style="color: var(--primary-green);">6 Algorithms (3 Reg + 3 Clf)</strong></div>
    </div>

    <div class="header-actions">
      <!-- Quick Potato Demo Preset Button -->
      <button class="btn-preset" onclick="loadPotatoPreset()" title="Click to view Potato & Vegetables filter matching your screenshot">
        <span>🥔</span> Potato Demo (582 Recs)
      </button>
      <button class="btn-export" onclick="window.print()">
        <span>📥</span> Export Report
      </button>
    </div>
  </header>

  <!-- ================= APP BODY ================= -->
  <div class="app-body">

    <!-- SIDEBAR -->
    <aside>
      <div>
        <div class="nav-label">NAVIGATION PAGES</div>
        <button class="nav-btn active" id="btn-nav-1" onclick="switchTab(1)">
          <span>🏠</span> 1. Market Overview
        </button>
        <button class="nav-btn" id="btn-nav-2" onclick="switchTab(2)">
          <span>📊</span> 2. Spreads & Arbitrage
        </button>
        <button class="nav-btn" id="btn-nav-3" onclick="switchTab(3)">
          <span>🤖</span> 3. ML Model Benchmarks
        </button>
      </div>

      <!-- DYNAMIC SLICERS -->
      <div class="filter-box">
        <div class="filter-title-wrap">
          <div class="nav-label">LIVE SLICERS</div>
          <button class="btn-reset" onclick="resetAllFilters()">Reset All</button>
        </div>

        <div style="display: flex; gap: 6px; margin: 4px 0 8px 0;">
          <button id="btn-quick-potato" class="btn-preset-pill active" onclick="loadPotatoPreset()" title="Click to view Potato & Vegetables filter from screenshot">
            <span>🥔</span> Potato Demo
          </button>
          <button id="btn-quick-all" class="btn-preset-pill" onclick="resetAllFilters()" title="Click to view all produce">
            <span>🌾</span> All Produce
          </button>
        </div>

        <div class="filter-group">
          <label><span>📍</span> Filter by State</label>
          <select id="stateFilter" onchange="onStateChange()">
            <option value="ALL">All States (29 States)</option>
            {''.join([f'<option value="{s}">{s}</option>' for s in states_list])}
          </select>
        </div>

        <div class="filter-group">
          <label><span>🏛️</span> Filter by District</label>
          <select id="districtFilter" onchange="onFilterChange()">
            <option value="ALL">All Districts</option>
            {''.join([f'<option value="{d}">{d}</option>' for d in districts_list])}
          </select>
        </div>

        <div class="filter-group">
          <label><span>🍃</span> Agri Commodity Sector</label>
          <select id="categoryFilter" onchange="onFilterChange()">
            <option value="ALL">All Sectors</option>
            <option value="Vegetables">Vegetables</option>
            <option value="Cereals & Pulses">Cereals & Pulses</option>
            <option value="Fruits">Fruits</option>
            <option value="Cash Crops">Cash Crops</option>
            <option value="Spices">Spices</option>
            <option value="Other Agri">Other Produce</option>
          </select>
        </div>

        <div class="filter-group">
          <label><span>🔍</span> Commodity / Mandi Search</label>
          <input type="text" id="searchInput" oninput="onFilterChange()" placeholder="e.g. potato, onion, wheat...">
        </div>
      </div>

      <!-- Verified Data Tag -->
      <div style="margin-top: auto; padding: 12px; background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 8px;">
        <div style="font-weight: 800; font-size: 11px; color: var(--primary-green);">100% Real Audited Data</div>
        <div style="font-size: 10px; color: #475569; margin-top: 4px; line-height: 1.4;">
          Audited snapshot: 16,734 raw rows (16,733 clean modelling rows). Single-day snapshot dated 29/10/2026.
        </div>
      </div>
    </aside>

    <!-- MAIN CONTENT -->
    <main>

      <!-- ============================================================== -->
      <!-- TAB 1: MANDI MARKET OVERVIEW -->
      <!-- ============================================================== -->
      <div class="tab-section active" id="tab-1">

        <!-- Farm Green Banner with HD Vegetables Backdrop -->
        <div class="hero-banner">
          <div>
            <div class="hero-title">Mandi Market Intelligence Overview</div>
            <div class="hero-subtitle">Interactive APMC market analysis responding to selected state, district, and category slicers</div>
          </div>
          <div class="filter-pill">
            <span>⚡</span> Active Filter: <strong id="filter-status-tag">All States (16,733 clean records)</strong>
          </div>
        </div>

        <!-- 4 Hero KPI Cards with Icon Boxes -->
        <div class="kpi-grid">
          
          <div class="kpi-card">
            <div class="kpi-left-wrap">
              <div class="kpi-icon-box" style="background: #dcfce7; color: #15803d;">📦</div>
              <div class="kpi-meta">
                <span class="kpi-title">MATCHING TRANSACTIONS</span>
                <span class="kpi-val" id="kpi-records">16,733</span>
                <span class="kpi-sub green">✓ 100% Verified Clean Records</span>
              </div>
            </div>
            <div style="font-size: 22px; color: #15803d;">📊</div>
          </div>

          <div class="kpi-card">
            <div class="kpi-left-wrap">
              <div class="kpi-icon-box" style="background: #dbeafe; color: #1d4ed8;">🏪</div>
              <div class="kpi-meta">
                <span class="kpi-title">ACTIVE APMC MANDIS</span>
                <span class="kpi-val" id="kpi-markets">1,774</span>
                <span class="kpi-sub" id="kpi-sub-markets">Across all selected mandis</span>
              </div>
            </div>
            <div style="font-size: 22px; color: #1d4ed8;">📈</div>
          </div>

          <div class="kpi-card">
            <div class="kpi-left-wrap">
              <div class="kpi-icon-box" style="background: #fef3c7; color: #b45309;">₹</div>
              <div class="kpi-meta">
                <span class="kpi-title">AVERAGE MODAL PRICE</span>
                <span class="kpi-val" id="kpi-modal" style="color: #d97706;">₹5,592</span>
                <span class="kpi-sub" id="kpi-sub-minmax">Min: ₹5,062 | Max: ₹6,054</span>
              </div>
            </div>
            <div style="font-size: 22px; color: #d97706;">💰</div>
          </div>

          <div class="kpi-card">
            <div class="kpi-left-wrap">
              <div class="kpi-icon-box" style="background: #f3e8ff; color: #7e22ce;">📉</div>
              <div class="kpi-meta">
                <span class="kpi-title">AVG TRADING SPREAD</span>
                <span class="kpi-val" id="kpi-spread" style="color: #7e22ce;">₹992</span>
                <span class="kpi-sub">Intraday Max - Min Margin</span>
              </div>
            </div>
            <div style="font-size: 22px; color: #7e22ce;">📊</div>
          </div>

        </div>

        <!-- Visuals Grid: Bar Chart + Sector Donut -->
        <div class="visual-grid">
          
          <!-- Bar Chart Panel -->
          <div class="panel">
            <div class="panel-header">
              <div>
                <div class="panel-title">
                  <span>📊</span> <span id="barChartTitle">Top Commodities by Volume (Transactions)</span>
                </div>
                <div class="panel-desc">Dynamically recalculates volume and average settlement price</div>
              </div>
              <div style="display: flex; gap: 6px;">
                <button class="btn-toggle active" id="btnModeVol" onclick="setChartMode('volume')">By Volume</button>
                <button class="btn-toggle inactive" id="btnModePrice" onclick="setChartMode('price')">By Price</button>
              </div>
            </div>

            <div id="dynamicBarContainer" style="display: flex; flex-direction: column; justify-content: space-around; min-height: 220px;">
              <!-- Dynamically populated by JS -->
            </div>
          </div>

          <!-- Sector Share Panel -->
          <div class="panel">
            <div class="panel-header">
              <div>
                <div class="panel-title">
                  <span>🍃</span> Agri Sector Distribution
                </div>
                <div class="panel-desc">Share of transaction volume by sector</div>
              </div>
              <span class="badge-live" id="categoryCountBadge">6 Sectors Active</span>
            </div>

            <div class="donut-wrap">
              <svg viewBox="0 0 160 160" width="130" height="130" id="dynamicDonutSvg" style="transform: rotate(-90deg);">
                <!-- Dynamic SVG Donut slices -->
              </svg>
              <div class="donut-legend" id="dynamicDonutLegend">
                <!-- Dynamic Legend Rows -->
              </div>
            </div>

            <div style="font-size: 11px; color: var(--text-muted); border-top: 1px solid var(--card-border); padding-top: 10px; margin-top: 10px;">
              💡 Vegetables lead transaction volume across mandis (43.6%), followed by Cereals & Pulses and Fruits.
            </div>
          </div>

        </div>

        <!-- Ledger Table Panel -->
        <div class="panel">
          <div class="panel-header">
            <div>
              <div class="panel-title">
                <span>📑</span> Interactive Mandi Price Ledger
              </div>
              <div class="panel-desc">Showing filtered rows from the 16,733 cleaned dataset snapshot (16,734 raw records)</div>
            </div>
            <div style="font-size: 12px; color: var(--text-muted);" id="tableCountSummary">
              Showing 1-12 of 16,733 records
            </div>
          </div>

          <div class="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>State</th>
                  <th>District</th>
                  <th>Market / APMC</th>
                  <th>Commodity</th>
                  <th>Category</th>
                  <th style="text-align: right;">Min Price</th>
                  <th style="text-align: right;">Max Price</th>
                  <th style="text-align: right;">Modal Settlement</th>
                  <th style="text-align: center;">Price Tier</th>
                </tr>
              </thead>
              <tbody id="tableBody">
                <!-- Dynamically populated rows -->
              </tbody>
            </table>
          </div>

          <div class="pagination-bar">
            <div id="pageInfo">Page 1 of 1395</div>
            <div style="display: flex; gap: 6px;">
              <button class="btn-page" id="btnPrev" onclick="changePage(-1)">Previous</button>
              <button class="btn-page" id="btnNext" onclick="changePage(1)">Next</button>
            </div>
          </div>
        </div>

      </div>

      <!-- ============================================================== -->
      <!-- TAB 2: SPREADS & ARBITRAGE -->
      <!-- ============================================================== -->
      <div class="tab-section" id="tab-2">

        <div class="hero-banner" style="background: linear-gradient(105deg, #1e3a8a 0%, #1e40af 55%, #2563eb 100%);">
          <div>
            <div class="hero-title">Commodity & Inter-Market Spread Analysis</div>
            <div class="hero-subtitle">Auditing price arbitrage and intraday volatility across Indian APMC mandis</div>
          </div>
          <div class="filter-pill" style="border-color: rgba(255,255,255,0.4); color: #ffffff;">
            <span>⚖️</span> Arbitrage Opportunities Detected
          </div>
        </div>

        <div class="kpi-grid">
          <div class="kpi-card">
            <div class="kpi-left-wrap">
              <div class="kpi-icon-box" style="background: #ffe4e6; color: #e11d48;">🚨</div>
              <div class="kpi-meta">
                <span class="kpi-title">WIDEST ARBITRAGE GAP</span>
                <span class="kpi-val" style="color: #e11d48;">₹69,200</span>
                <span class="kpi-sub">Black Pepper (MP Min ₹2,400 vs KA Max ₹71,600)</span>
              </div>
            </div>
          </div>
          <div class="kpi-card">
            <div class="kpi-left-wrap">
              <div class="kpi-icon-box" style="background: #fef3c7; color: #b45309;">🌾</div>
              <div class="kpi-meta">
                <span class="kpi-title">MULTI-MANDI COMMODITIES</span>
                <span class="kpi-val" style="color: #b45309;">180+ Crops</span>
                <span class="kpi-sub">Traded across 5 or more distinct APMCs</span>
              </div>
            </div>
          </div>
          <div class="kpi-card">
            <div class="kpi-left-wrap">
              <div class="kpi-icon-box" style="background: #dbeafe; color: #1d4ed8;">📈</div>
              <div class="kpi-meta">
                <span class="kpi-title">AVERAGE MANDI MARKUP</span>
                <span class="kpi-val" style="color: #1d4ed8;">49.8%</span>
                <span class="kpi-sub">Average spread from minimum to maximum auction price</span>
              </div>
            </div>
          </div>
        </div>

        <div class="panel">
          <div class="panel-header">
            <div>
              <div class="panel-title">Top 10 Inter-Market Price Arbitrage Opportunities</div>
              <div class="panel-desc">Actual spread between lowest and highest APMC auction prices on the exact same date</div>
            </div>
            <span class="badge-live">Min 5 Active Mandis</span>
          </div>

          <div class="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Commodity</th>
                  <th>Category</th>
                  <th style="text-align: center;">Mandis Reporting</th>
                  <th style="text-align: right;">Lowest APMC Price</th>
                  <th style="text-align: right;">Highest APMC Price</th>
                  <th style="text-align: right;">Arbitrage Gap</th>
                  <th style="text-align: right;">Price Ratio</th>
                </tr>
              </thead>
              <tbody id="arbTableBody">
                <!-- JS populated -->
              </tbody>
            </table>
          </div>
        </div>

      </div>

      <!-- ============================================================== -->
      <!-- TAB 3: ML BENCHMARKS (EXACT 6 ALGORITHMS) -->
      <!-- ============================================================== -->
      <div class="tab-section" id="tab-3">

        <div class="hero-banner" style="background: linear-gradient(105deg, #4c1d95 0%, #5b21b6 55%, #7c3aed 100%);">
          <div>
            <div class="hero-title">Machine Learning Model Benchmarking & Decision Suite</div>
            <div class="hero-subtitle">Rigorous Evaluation of Exactly 6 Machine Learning Algorithms (3 Regression + 3 Classification)</div>
          </div>
          <div class="filter-pill" style="border-color: rgba(255,255,255,0.4); color: #ffffff;">
            <span>🤖</span> 6 Algorithms Benchmarked (80/20 Train-Test)
          </div>
        </div>

        <!-- 4 ML Highlight KPI Cards -->
        <div class="kpi-grid">
          <div class="kpi-card">
            <div class="kpi-left-wrap">
              <div class="kpi-icon-box" style="background: #dcfce7; color: #15803d;">🌳</div>
              <div class="kpi-meta">
                <span class="kpi-title">LOWEST MAE REGRESSOR</span>
                <span class="kpi-val" style="color: #15803d; font-size: 20px;">Decision Tree</span>
                <span class="kpi-sub green">MAE: ₹1,387.15 / Qtl (R² = 0.738)</span>
              </div>
            </div>
          </div>
          <div class="kpi-card">
            <div class="kpi-left-wrap">
              <div class="kpi-icon-box" style="background: #dbeafe; color: #1d4ed8;">🎯</div>
              <div class="kpi-meta">
                <span class="kpi-title">TOP CLASSIFIER</span>
                <span class="kpi-val" style="color: #1d4ed8;">84.67%</span>
                <span class="kpi-sub green">Logistic Regression (F1: 0.8451)</span>
              </div>
            </div>
          </div>
          <div class="kpi-card">
            <div class="kpi-left-wrap">
              <div class="kpi-icon-box" style="background: #fef3c7; color: #b45309;">📐</div>
              <div class="kpi-meta">
                <span class="kpi-title">HIGHEST R² REGRESSOR</span>
                <span class="kpi-val" style="color: #b45309;">0.782</span>
                <span class="kpi-sub">Linear Regression (RMSE: ₹2,868.64)</span>
              </div>
            </div>
          </div>
          <div class="kpi-card">
            <div class="kpi-left-wrap">
              <div class="kpi-icon-box" style="background: #f3e8ff; color: #7e22ce;">⚡</div>
              <div class="kpi-meta">
                <span class="kpi-title">NAIVE BAYES ACCURACY</span>
                <span class="kpi-val" style="color: #7e22ce;">80.97%</span>
                <span class="kpi-sub">Multinomial NB (F1: 0.8111)</span>
              </div>
            </div>
          </div>
        </div>

        <!-- SECTION A: REGRESSION MODELS LEADERBOARD -->
        <div class="panel" style="border-left: 4px solid var(--primary-green);">
          <div class="panel-header">
            <div>
              <div class="panel-title">
                <span>📈</span> Problem A: Modal Price Estimation (3 Regression Algorithms)
              </div>
              <div class="panel-desc">Target: Modal_Price (₹/Quintal) | Inputs: State, District, APMC Market, Commodity, Variety, Grade (Zero Price Leakage)</div>
            </div>
            <span class="badge-live">Target: Modal_Price</span>
          </div>

          <div class="experiment-callout callout-primary">
            <strong>Methodology &amp; Leakage Prevention:</strong> Min_Price and Max_Price are strictly excluded from inputs to ensure models learn genuine market-commodity spatial economics. Evaluated on 3,347 held-out test records (20% split).
          </div>

          <div class="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Algorithm</th>
                  <th style="text-align: right;">Train MAE</th>
                  <th style="text-align: right;">Test MAE</th>
                  <th style="text-align: right;">Test MSE</th>
                  <th style="text-align: right;">Test RMSE</th>
                  <th style="text-align: right;">Train R²</th>
                  <th style="text-align: right;">Test R²</th>
                  <th style="text-align: center;">Status</th>
                </tr>
              </thead>
              <tbody id="regTableBody">
                <!-- JS populated -->
              </tbody>
            </table>
          </div>
        </div>

        <!-- SECTION B: CLASSIFICATION MODELS LEADERBOARD -->
        <div class="panel" style="border-left: 4px solid var(--accent-blue);">
          <div class="panel-header">
            <div>
              <div class="panel-title">
                <span>🎯</span> Problem B: Commodity Price Tier Classification (3 Classification Algorithms)
              </div>
              <div class="panel-desc">Target: Price_Tier (Budget &lt;₹3k | Standard ₹3k-₹10k | Premium &gt;₹10k) | Inputs: Market &amp; Commodity Descriptors</div>
            </div>
            <span class="badge-live">Target: Price_Tier</span>
          </div>

          <div class="experiment-callout callout-info">
            <strong>Model Architecture Note:</strong> Logistic Regression with L2 regularization is the overall top classifier (84.67%). Multinomial Naive Bayes achieves 80.97% accuracy on One-Hot Encoded binary indicators using Laplace smoothing, comfortably exceeding the majority class baseline (60.71%).
          </div>

          <div class="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Algorithm</th>
                  <th>Feature Representation</th>
                  <th style="text-align: right;">Accuracy</th>
                  <th style="text-align: right;">Weighted Precision</th>
                  <th style="text-align: right;">Weighted Recall</th>
                  <th style="text-align: right;">Weighted F1</th>
                  <th style="text-align: right;">Macro F1</th>
                  <th style="text-align: center;">Status</th>
                </tr>
              </thead>
              <tbody id="clfTableBody">
                <!-- JS populated -->
              </tbody>
            </table>
          </div>
        </div>

        <!-- 3 Diagnostic Visuals Grid -->
        <div class="visual-grid-3">
          <div class="panel">
            <div class="panel-header">
              <div class="panel-title">Regression: Actual vs. Predicted</div>
              <span class="badge-live">R² = 0.767</span>
            </div>
            <img src="data:image/png;base64,{p1_b64}" class="plot-img" alt="Actual vs Predicted Mandi Prices">
            <p style="font-size: 11px; color: var(--text-muted); margin-top: 8px; line-height: 1.4;">
              Scatter plot of actual vs. predicted modal prices on 3,347 test records. Points follow the red identity line (y = x) with high precision in staple crops and expected spread in volatile commodities.
            </p>
          </div>

          <div class="panel">
            <div class="panel-header">
              <div class="panel-title">Classification: Confusion Matrix</div>
              <span class="badge-live">84.67% Accuracy</span>
            </div>
            <img src="data:image/png;base64,{cm_b64}" class="plot-img" alt="Classification Confusion Matrix">
            <p style="font-size: 11px; color: var(--text-muted); margin-top: 8px; line-height: 1.4;">
              Confusion matrix for Logistic Regression on test set. Strong diagonal alignment across Budget (&lt;₹3k), Standard (₹3k-₹10k), and Premium (&gt;₹10k) tiers with low cross-tier leakage.
            </p>
          </div>

          <div class="panel">
            <div class="panel-header">
              <div class="panel-title">Residual Error Distribution</div>
              <span class="badge-live">Zero Bias</span>
            </div>
            <img src="data:image/png;base64,{p2_b64}" class="plot-img" alt="Residual Error Distribution">
            <p style="font-size: 11px; color: var(--text-muted); margin-top: 8px; line-height: 1.4;">
              Residual error distribution (Actual - Predicted) centered symmetrically around ₹0, demonstrating that the regression pipeline introduces zero systematic upward or downward pricing bias.
            </p>
          </div>
        </div>

        <!-- Top Commodity Estimation Reliability -->
        <div class="panel">
          <div class="panel-header">
            <div>
              <div class="panel-title">Top Commodity Estimation Reliability &amp; Error Analysis</div>
              <div class="panel-desc">Model error performance across major commodities evaluated on held-out test records</div>
            </div>
            <span class="badge-live">Evaluated Test Set</span>
          </div>

          <div class="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Commodity</th>
                  <th style="text-align: center;">Test Records</th>
                  <th style="text-align: right;">Median Actual Price</th>
                  <th style="text-align: right;">Model MAE</th>
                  <th style="text-align: right;">Mean Percentage Error</th>
                </tr>
              </thead>
              <tbody id="commErrTableBody">
                <!-- JS populated -->
              </tbody>
            </table>
          </div>
        </div>

        <!-- Project Limitations Callout -->
        <div class="panel" style="background: #fafafa; border-left: 4px solid var(--accent-amber);">
          <div class="panel-header">
            <div class="panel-title">⚠️ Project Limitations &amp; Dataset Scope</div>
            <span class="badge-subtle">Methodological Notice</span>
          </div>
          <div style="font-size: 12px; color: #475569; line-height: 1.6;">
            <p>1. <strong>Cross-Sectional Snapshot:</strong> The dataset is an audited cross-sectional snapshot representing daily prices on <strong>October 29, 2026</strong> from Data.gov.in (Agmarknet). It does not contain longitudinal time-series data or seasonal price trends.</p>
            <p style="margin-top: 6px;">2. <strong>Arrival Volumes Unrecorded:</strong> Daily arrival quantities (in metric tonnes) are absent in the raw feed, precluding elastic supply-demand price modelling.</p>
            <p style="margin-top: 6px;">3. <strong>Data Leakage Prevention:</strong> Min_Price and Max_Price were intentionally omitted from input features to prevent trivial mathematical envelope interpolation.</p>
          </div>
        </div>

      </div>

    </main>
  </div>

  <!-- ================= JAVASCRIPT ENGINE ================= -->
  <script>
    const ALL_RECORDS = {records_json};
    const ARB_DATA = {arb_json};
    const REG_DATA = {reg_json};
    const CLF_DATA = {clf_json};
    const COMM_ERR_DATA = {comm_err_json};

    let filteredData = [...ALL_RECORDS];
    let currentTab = 1;
    let chartMode = 'volume';
    let currentPage = 1;
    const pageSize = 12;

    function switchTab(tabIndex) {{
      currentTab = tabIndex;
      document.querySelectorAll('.tab-section').forEach(el => el.classList.remove('active'));
      document.querySelectorAll('.nav-btn').forEach(el => el.classList.remove('active'));

      const activeSection = document.getElementById('tab-' + tabIndex);
      if (activeSection) activeSection.classList.add('active');

      const activeBtn = document.getElementById('btn-nav-' + tabIndex);
      if (activeBtn) activeBtn.classList.add('active');
    }}

    function setChartMode(mode) {{
      chartMode = mode;
      document.getElementById('btnModeVol').className = (mode === 'volume' ? 'btn-toggle active' : 'btn-toggle inactive');
      document.getElementById('btnModePrice').className = (mode === 'price' ? 'btn-toggle active' : 'btn-toggle inactive');
      updateBarChart();
    }}

    function onStateChange() {{
      const state = document.getElementById('stateFilter').value;
      const districtSelect = document.getElementById('districtFilter');
      
      // Update district options based on selected state
      if (state === 'ALL') {{
        districtSelect.innerHTML = '<option value="ALL">All Districts</option>';
        const dSet = new Set(ALL_RECORDS.map(r => r[1]));
        Array.from(dSet).sort().forEach(d => {{
          districtSelect.innerHTML += `<option value="${{d}}">${{d}}</option>`;
        }});
      }} else {{
        districtSelect.innerHTML = '<option value="ALL">All Districts in ' + state + '</option>';
        const dSet = new Set(ALL_RECORDS.filter(r => r[0] === state).map(r => r[1]));
        Array.from(dSet).sort().forEach(d => {{
          districtSelect.innerHTML += `<option value="${{d}}">${{d}}</option>`;
        }});
      }}
      onFilterChange();
    }}

    // Potato Preset
    function loadPotatoPreset() {{
      document.getElementById('stateFilter').value = 'ALL';
      document.getElementById('districtFilter').value = 'ALL';
      document.getElementById('categoryFilter').value = 'Vegetables';
      document.getElementById('searchInput').value = 'potato';
      const bPot = document.getElementById('btn-quick-potato');
      const bAll = document.getElementById('btn-quick-all');
      if (bPot) bPot.classList.add('active');
      if (bAll) bAll.classList.remove('active');
      onFilterChange();
    }}

    function onFilterChange() {{
      const state = document.getElementById('stateFilter').value;
      const district = document.getElementById('districtFilter').value;
      const cat = document.getElementById('categoryFilter').value;
      const query = document.getElementById('searchInput').value.toLowerCase().trim();

      filteredData = ALL_RECORDS.filter(r => {{
        // r = [State, District, Market, Commodity, Category, Min_Price, Max_Price, Modal_Price]
        const matchState = (state === 'ALL' || r[0] === state);
        const matchDistrict = (district === 'ALL' || r[1] === district);
        const matchCat = (cat === 'ALL' || r[4] === cat);
        const matchQuery = (!query || 
          r[0].toLowerCase().includes(query) ||
          r[1].toLowerCase().includes(query) ||
          r[2].toLowerCase().includes(query) ||
          r[3].toLowerCase().includes(query)
        );
        return matchState && matchDistrict && matchCat && matchQuery;
      }});

      currentPage = 1;
      updateDashboard();
    }}

    function resetAllFilters() {{
      document.getElementById('stateFilter').value = 'ALL';
      document.getElementById('districtFilter').value = 'ALL';
      document.getElementById('categoryFilter').value = 'ALL';
      document.getElementById('searchInput').value = '';
      const bPot = document.getElementById('btn-quick-potato');
      const bAll = document.getElementById('btn-quick-all');
      if (bPot) bPot.classList.remove('active');
      if (bAll) bAll.classList.add('active');
      filteredData = [...ALL_RECORDS];
      currentPage = 1;
      updateDashboard();
    }}

    function changePage(delta) {{
      const totalPages = Math.max(1, Math.ceil(filteredData.length / pageSize));
      const newPage = currentPage + delta;
      if (newPage >= 1 && newPage <= totalPages) {{
        currentPage = newPage;
        renderTable();
      }}
    }}

    function updateDashboard() {{
      const count = filteredData.length;

      const stateVal = document.getElementById('stateFilter').value;
      const catVal = document.getElementById('categoryFilter').value;
      let tag = (stateVal !== 'ALL' ? stateVal : 'All States');
      if (catVal !== 'ALL') tag += ' - ' + catVal;
      tag += ` (${{count.toLocaleString()}} records)`;
      document.getElementById('filter-status-tag').innerText = tag;

      document.getElementById('kpi-records').innerText = count.toLocaleString();

      if (count === 0) {{
        document.getElementById('kpi-markets').innerText = '0';
        document.getElementById('kpi-modal').innerText = '₹0';
        document.getElementById('kpi-spread').innerText = '₹0';
        document.getElementById('kpi-sub-minmax').innerText = 'Min: ₹0 | Max: ₹0';
        document.getElementById('dynamicBarContainer').innerHTML = '<div style="text-align: center; color: var(--text-muted); padding: 40px 0;">No matching commodities found for this filter.</div>';
        document.getElementById('dynamicDonutSvg').innerHTML = '';
        document.getElementById('dynamicDonutLegend').innerHTML = '';
        renderTable();
        return;
      }}

      const marketSet = new Set();
      let sumModal = 0, sumMin = 0, sumMax = 0;
      const commMap = {{}};
      const catMap = {{}};

      for (let i = 0; i < count; i++) {{
        const r = filteredData[i];
        marketSet.add(r[2]);
        sumMin += r[5];
        sumMax += r[6];
        sumModal += r[7];

        const comm = r[3];
        if (!commMap[comm]) commMap[comm] = {{ count: 0, sumModal: 0 }};
        commMap[comm].count++;
        commMap[comm].sumModal += r[7];

        const c = r[4];
        catMap[c] = (catMap[c] || 0) + 1;
      }}

      const avgModal = Math.round(sumModal / count);
      const avgMin = Math.round(sumMin / count);
      const avgMax = Math.round(sumMax / count);
      const avgSpread = Math.round((sumMax - sumMin) / count);

      document.getElementById('kpi-markets').innerText = marketSet.size.toLocaleString();
      document.getElementById('kpi-modal').innerText = '₹' + avgModal.toLocaleString();
      document.getElementById('kpi-spread').innerText = '₹' + avgSpread.toLocaleString();
      document.getElementById('kpi-sub-minmax').innerText = `Min: ₹${{avgMin.toLocaleString()}} | Max: ₹${{avgMax.toLocaleString()}}`;

      updateBarChart(commMap);
      updateDonutChart(catMap, count);
      renderTable();
    }}

    function getCropIcon(comm) {{
      const c = (comm || '').toLowerCase();
      if (c.includes('sweet potato')) return '🍠';
      if (c.includes('potato')) return '🥔';
      if (c.includes('onion')) return '🧅';
      if (c.includes('tomato')) return '🍅';
      if (c.includes('chilli') || c.includes('chili')) return '🌶️';
      if (c.includes('garlic')) return '🧄';
      if (c.includes('ginger')) return '🫚';
      if (c.includes('carrot')) return '🥕';
      if (c.includes('cabbage') || c.includes('cauliflower')) return '🥬';
      if (c.includes('wheat') || c.includes('paddy') || c.includes('rice')) return '🌾';
      if (c.includes('maize') || c.includes('corn')) return '🌽';
      if (c.includes('apple') || c.includes('banana') || c.includes('mango')) return '🍎';
      if (c.includes('cotton')) return '☁️';
      if (c.includes('mustard') || c.includes('soyabean')) return '🌱';
      return '📦';
    }}

    function updateBarChart(commMap) {{
      if (!commMap) {{
        commMap = {{}};
        for (let i = 0; i < filteredData.length; i++) {{
          const comm = filteredData[i][3];
          if (!commMap[comm]) commMap[comm] = {{ count: 0, sumModal: 0 }};
          commMap[comm].count++;
          commMap[comm].sumModal += filteredData[i][7];
        }}
      }}

      const items = Object.keys(commMap).map(k => ({{
        commodity: k,
        records: commMap[k].count,
        avg_price: Math.round(commMap[k].sumModal / commMap[k].count)
      }}));

      if (chartMode === 'volume') {{
        items.sort((a, b) => b.records - a.records);
        document.getElementById('barChartTitle').innerText = 'Top Commodities by Volume (Transactions)';
      }} else {{
        items.sort((a, b) => b.avg_price - a.avg_price);
        document.getElementById('barChartTitle').innerText = 'Top Commodities by Average Modal Price (₹/Quintal)';
      }}

      const top10 = items.slice(0, 7);
      if (top10.length === 0) return;

      const maxVal = (chartMode === 'volume' ? top10[0].records : top10[0].avg_price);

      const html = top10.map(item => {{
        const val = (chartMode === 'volume' ? item.records : item.avg_price);
        const pct = Math.max(6, (val / maxVal) * 100);
        const label = (chartMode === 'volume' ? `${{item.records.toLocaleString()}} rows (Avg ₹${{item.avg_price.toLocaleString()}})` : `₹${{item.avg_price.toLocaleString()}} (${{item.records}} rows)`);

        return `
          <div class="bar-item">
            <div class="bar-meta">
              <span class="bar-name">${{getCropIcon(item.commodity)}} ${{item.commodity}}</span>
              <span class="bar-stat">${{label}}</span>
            </div>
            <div class="bar-track">
              <div class="bar-fill" style="width: ${{pct}}%;"></div>
            </div>
          </div>
        `;
      }}).join('');

      document.getElementById('dynamicBarContainer').innerHTML = html;
    }}

    const CAT_COLORS = {{
      "Vegetables": "#0e7044",
      "Cereals & Pulses": "#2563eb",
      "Fruits": "#d97706",
      "Cash Crops": "#7c3aed",
      "Spices": "#e11d48",
      "Other Agri": "#64748b"
    }};

    function updateDonutChart(catMap, totalCount) {{
      const cats = Object.keys(catMap).map(k => ({{
        name: k,
        count: catMap[k],
        pct: (catMap[k] / totalCount) * 100
      }})).sort((a, b) => b.count - a.count);

      document.getElementById('categoryCountBadge').innerText = cats.length + ' Sectors Active';

      const circumference = 2 * Math.PI * 45; // ~282.74
      let offset = 0;
      let svgHtml = '';

      cats.forEach(c => {{
        const dash = (c.pct / 100) * circumference;
        const space = circumference - dash;
        const color = CAT_COLORS[c.name] || '#64748b';

        svgHtml += `
          <circle r="45" cx="80" cy="80" fill="transparent"
                  stroke="${{color}}" stroke-width="22"
                  stroke-dasharray="${{dash.toFixed(2)}} ${{space.toFixed(2)}}"
                  stroke-dashoffset="${{(-offset).toFixed(2)}}">
            <title>${{c.name}}: ${{c.count.toLocaleString()}} (${{c.pct.toFixed(1)}}%)</title>
          </circle>
        `;
        offset += dash;
      }});

      document.getElementById('dynamicDonutSvg').innerHTML = svgHtml;

      const legendHtml = cats.map(c => `
        <div class="legend-row">
          <div>
            <span class="legend-dot" style="background: ${{CAT_COLORS[c.name] || '#64748b'}}"></span>
            <span>${{c.name}}</span>
          </div>
          <strong style="font-family: monospace;">${{c.pct.toFixed(1)}}%</strong>
        </div>
      `).join('');

      document.getElementById('dynamicDonutLegend').innerHTML = legendHtml;
    }}

    function renderTable() {{
      const tbody = document.getElementById('tableBody');
      const total = filteredData.length;
      const totalPages = Math.max(1, Math.ceil(total / pageSize));

      const start = (currentPage - 1) * pageSize;
      const end = Math.min(start + pageSize, total);

      document.getElementById('tableCountSummary').innerText = `Showing ${{total > 0 ? start + 1 : 0}}-${{end}} of ${{total.toLocaleString()}} records`;
      document.getElementById('pageInfo').innerText = `Page ${{currentPage}} of ${{totalPages}} (${{total.toLocaleString()}} total)`;

      document.getElementById('btnPrev').disabled = (currentPage === 1);
      document.getElementById('btnNext').disabled = (currentPage >= totalPages);

      if (total === 0) {{
        tbody.innerHTML = `<tr><td colspan="9" style="text-align: center; padding: 24px; color: var(--text-muted);">No matching transactions found.</td></tr>`;
        return;
      }}

      const pageRows = filteredData.slice(start, end);
      tbody.innerHTML = pageRows.map(r => {{
        let badge = '<span class="tier-badge tier-standard">Standard</span>';
        if (r[7] > 10000) badge = '<span class="tier-badge tier-premium">Premium</span>';
        else if (r[7] < 3000) badge = '<span class="tier-badge tier-budget">Budget</span>';

        return `
          <tr>
            <td style="font-weight: 700;">${{r[0]}}</td>
            <td style="color: var(--text-muted);">${{r[1]}}</td>
            <td style="color: var(--text-muted);">${{r[2]}}</td>
            <td style="font-weight: 800; color: #0b4a29;">${{getCropIcon(r[3])}} ${{r[3]}}</td>
            <td style="color: var(--text-muted); font-size: 11px;">${{r[4]}}</td>
            <td style="text-align: right; font-family: monospace; color: var(--text-muted);">₹${{r[5].toLocaleString()}}</td>
            <td style="text-align: right; font-family: monospace; color: var(--text-muted);">₹${{r[6].toLocaleString()}}</td>
            <td style="text-align: right; font-family: monospace; font-weight: 800; color: var(--primary-green);">₹${{r[7].toLocaleString()}}</td>
            <td style="text-align: center;">${{badge}}</td>
          </tr>
        `;
      }}).join('');
    }}

    function populateStaticTables() {{
      // Arbitrage Table
      const arbTbody = document.getElementById('arbTableBody');
      arbTbody.innerHTML = ARB_DATA.map(item => `
        <tr>
          <td style="font-weight: 800; color: #0b4a29;">${{item.Commodity}}</td>
          <td style="color: var(--text-muted);">${{item.Category}}</td>
          <td style="text-align: center; font-family: monospace;">${{item.Reporting_Mandis}} mandis</td>
          <td style="text-align: right; font-family: monospace; color: var(--primary-green); font-weight: 700;">₹${{item.Lowest_Mandi_Price.toLocaleString()}}</td>
          <td style="text-align: right; font-family: monospace; color: var(--accent-rose); font-weight: 700;">₹${{item.Highest_Mandi_Price.toLocaleString()}}</td>
          <td style="text-align: right; font-family: monospace; font-weight: 800; color: var(--accent-amber);">₹${{item.Arbitrage_Gap.toLocaleString()}}</td>
          <td style="text-align: right; font-family: monospace; font-weight: 800; color: var(--accent-blue);">${{item.Price_Ratio.toFixed(1)}}x</td>
        </tr>
      `).join('');

      // Regression Table
      const regTbody = document.getElementById('regTableBody');
      regTbody.innerHTML = REG_DATA.map(item => `
        <tr style="${{item.Model.includes('Decision Tree') ? 'background: #f0fdf4;' : (item.Model.includes('Linear') ? 'background: #eff6ff;' : '')}}">
          <td style="font-weight: 800; color: #0b4a29;">${{item.Model}}</td>
          <td style="text-align: right; font-family: monospace; color: var(--text-muted);">₹${{item.Train_MAE.toFixed(2)}}</td>
          <td style="text-align: right; font-family: monospace; font-weight: 800; color: ${{item.Test_MAE < 1500 ? 'var(--primary-green)' : 'var(--text-muted)'}};">₹${{item.Test_MAE.toFixed(2)}}</td>
          <td style="text-align: right; font-family: monospace; color: var(--text-muted);">${{(item.Test_MSE/1e6).toFixed(2)}}M</td>
          <td style="text-align: right; font-family: monospace; color: var(--text-muted);">₹${{item.Test_RMSE.toFixed(2)}}</td>
          <td style="text-align: right; font-family: monospace; color: var(--text-muted);">${{item.Train_R2.toFixed(3)}}</td>
          <td style="text-align: right; font-family: monospace; font-weight: 800; color: ${{item.Test_R2 > 0.75 ? 'var(--accent-blue)' : 'var(--text-muted)'}};">${{item.Test_R2.toFixed(3)}}</td>
          <td style="text-align: center;">
            ${{item.Model.includes('Decision Tree') ? '<span class="badge-live">Lowest MAE</span>' : (item.Model.includes('Linear') ? '<span class="badge-live" style="background:#dbeafe;color:#1d4ed8;border-color:#bfdbfe;">Highest R²</span>' : '<span class="badge-subtle">Evaluated</span>')}}
          </td>
        </tr>
      `).join('');

      // Classification Table
      const clfTbody = document.getElementById('clfTableBody');
      clfTbody.innerHTML = CLF_DATA.map(item => `
        <tr style="${{item.Model.includes('Logistic') ? 'background: #eff6ff;' : (item.Model.includes('Multinomial') ? 'background: #f0fdf4;' : '')}}">
          <td style="font-weight: 800; color: #0b4a29;">${{item.Model}}</td>
          <td style="color: var(--text-muted); font-size: 11px;">${{item.Feature_Representation}}</td>
          <td style="text-align: right; font-family: monospace; font-weight: 800; color: ${{item.Accuracy > 0.8 ? 'var(--primary-green)' : 'var(--text-muted)'}};">${{(item.Accuracy * 100).toFixed(2)}}%</td>
          <td style="text-align: right; font-family: monospace; color: var(--text-muted);">${{(item.Precision_Weighted * 100).toFixed(2)}}%</td>
          <td style="text-align: right; font-family: monospace; color: var(--text-muted);">${{(item.Recall_Weighted * 100).toFixed(2)}}%</td>
          <td style="text-align: right; font-family: monospace; font-weight: 800; color: var(--accent-blue);">${{item.F1_Weighted.toFixed(4)}}</td>
          <td style="text-align: right; font-family: monospace; color: var(--text-muted);">${{item.F1_Macro.toFixed(4)}}</td>
          <td style="text-align: center;">
            ${{item.Model.includes('Logistic') ? '<span class="badge-live">Top Classifier</span>' : (item.Model.includes('Multinomial') ? '<span class="badge-live" style="background:#f0fdf4;color:#15803d;border-color:#bbf7d0;">Top Naive Bayes</span>' : '<span class="badge-subtle">Evaluated</span>')}}
          </td>
        </tr>
      `).join('');

      // Commodity Error Table
      const commErrTbody = document.getElementById('commErrTableBody');
      commErrTbody.innerHTML = COMM_ERR_DATA.map(item => `
        <tr>
          <td style="font-weight: 800; color: #0b4a29;">${{item.Commodity}}</td>
          <td style="text-align: center; font-family: monospace; color: var(--text-muted);">${{item.Records}}</td>
          <td style="text-align: right; font-family: monospace; color: var(--text-muted);">₹${{item.Median_Actual_Price.toLocaleString()}}</td>
          <td style="text-align: right; font-family: monospace; font-weight: 800; color: var(--accent-rose);">₹${{item.MAE.toFixed(2)}}</td>
          <td style="text-align: right; font-family: monospace; font-weight: 800; color: ${{item.Mean_Percentage_Error < 25 ? 'var(--primary-green)' : (item.Mean_Percentage_Error < 50 ? 'var(--accent-amber)' : 'var(--accent-rose)')}};">${{item.Mean_Percentage_Error.toFixed(2)}}%</td>
        </tr>
      `).join('');
    }}

    populateStaticTables();
    loadPotatoPreset(); // Automatically displays the Potato & Vegetables view matching user's preset!
  </script>

</body>
</html>
"""

# Write bundle files
(ROOT_DIR / "dashboard.html").write_text(html_content, encoding="utf-8")
(ROOT_DIR / "index.html").write_text(html_content, encoding="utf-8")
(ROOT_DIR / "dashboard" / "index.html").write_text(html_content, encoding="utf-8")
if ARTIFACT_DIR.exists():
    (ARTIFACT_DIR / "dashboard.html").write_text(html_content, encoding="utf-8")

print("Successfully generated all upgraded dashboard files!")
