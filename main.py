#!/usr/bin/env python
# coding: utf-8
# ============================================================
#  PLN UPT Surabaya – Asset Intelligence v3.0
#  File   : app.py
#  Jalankan: streamlit run app.py
# ============================================================

import os, warnings
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from model import train_ml_pipeline, MODEL_PATH, ENCODER_PATH

warnings.filterwarnings('ignore')

# ── Palet Warna PLN ───────────────────────────────────────────────────────────
PLN_BLUE   = "#003DA5"
PLN_BLUE2  = "#0055CC"
PLN_YELLOW = "#F5C400"
DANGER     = "#DC3545"
SUCCESS    = "#198754"
WARNING_C  = "#FFC107"
LIGHT_BG   = "#F0F4FF"
BLACK      = "black"

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG  (harus baris pertama sebelum widget apapun)
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="PLN UPT Surabaya – Asset Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
  .stApp {{ background-color:{LIGHT_BG}; }}
  [data-testid="stSidebar"] {{
      background:linear-gradient(180deg,{PLN_BLUE} 0%,{PLN_BLUE2} 100%);
  }}
  [data-testid="stSidebar"] * {{ color:white !important; }}
  [data-testid="stSidebar"] .stRadio > label {{
      color:{PLN_YELLOW} !important; font-weight:700;
  }}
  .pln-header {{
      background:linear-gradient(90deg,{PLN_BLUE} 60%,{PLN_YELLOW} 100%);
      padding:18px 28px; border-radius:12px; margin-bottom:18px;
      display:flex; align-items:center; gap:14px;
  }}
  .pln-header h1 {{ color:white; margin:0; font-size:1.65rem; font-weight:800; }}
  .pln-header span {{ font-size:2.2rem; }}
  .kpi-card {{
      background:white; border-radius:12px; padding:20px 24px;
      border-left:6px solid {PLN_BLUE};
      box-shadow:0 2px 8px rgba(0,61,165,.12); text-align:center;
  }}
  .kpi-card.danger  {{ border-left-color:{DANGER}; }}
  .kpi-card.warning {{ border-left-color:{PLN_YELLOW}; }}
  .kpi-card.success {{ border-left-color:{SUCCESS}; }}
  .kpi-value {{ font-size:2.1rem; font-weight:800; color:{PLN_BLUE}; }}
  .kpi-label {{ font-size:.82rem; color:#222; margin-top:4px;
                font-weight:700; text-transform:uppercase; letter-spacing:.5px; }}
  .kpi-sub   {{ font-size:.75rem; color:#555; }}
  .section-card  {{ background:white; border-radius:12px; padding:20px;
                    box-shadow:0 2px 8px rgba(0,61,165,.09); margin-bottom:18px; }}
  .section-title {{ font-size:1.05rem; font-weight:700; color:{PLN_BLUE};
                    border-bottom:2px solid {PLN_YELLOW};
                    padding-bottom:6px; margin-bottom:14px; }}
  [data-testid="stDataFrame"] {{ width:100% !important; }}
  [data-testid="stDataFrame"] iframe {{ min-height:480px; width:100% !important; }}
  .stButton > button {{
      background:linear-gradient(90deg,{PLN_BLUE},{PLN_BLUE2});
      color:white; border:none; border-radius:8px;
      padding:10px 28px; font-weight:700; font-size:1rem; transition:.2s;
  }}
  .stButton > button:hover {{
      background:linear-gradient(90deg,{PLN_YELLOW},#e0a800);
      color:{PLN_BLUE}; transform:translateY(-1px);
  }}
</style>
""", unsafe_allow_html=True)

# ── Helper layout ─────────────────────────────────────────────────────────────
FONT   = dict(color=BLACK, family='Arial')
AXIS   = dict(title_font=dict(color=BLACK), tickfont=dict(color=BLACK))
LEGEND = dict(font=dict(color=BLACK), orientation='h', y=-0.28)

def base_layout(h=300, **kw):
    return dict(height=h, paper_bgcolor='white', plot_bgcolor='white',
                font=FONT, margin=dict(t=20, b=10, l=10, r=10), **kw)

def pln_header(icon, title, subtitle=""):
    sub = f'<p style="color:#dde;margin:0;font-size:.9rem">{subtitle}</p>' if subtitle else ''
    st.markdown(f"""
    <div class="pln-header"><span>{icon}</span>
      <div><h1>{title}</h1>{sub}</div>
    </div>""", unsafe_allow_html=True)

def kpi_card(val, label, sub="", variant=""):
    s = f'<div class="kpi-sub">{sub}</div>' if sub else ''
    return f"""<div class="kpi-card {variant}">
        <div class="kpi-value">{val}</div>
        <div class="kpi-label">{label}</div>{s}
    </div>"""


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD & PREPROCESSING DATA
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data
def load_data():
    df_g = pd.read_excel('dataset.xlsx', sheet_name='Data RAW-GROUND', header=1)
    df_f = pd.read_excel('dataset.xlsx', sheet_name='Data RAW-FASA',   header=1)

    df_g['NOMOR TOWER'] = df_g['NOMOR TOWER'].astype(str)
    df_f['NOMOR TOWER'] = df_f['NOMOR TOWER'].astype(str)
    df_f['TAHUN ISOLATOR TERPASANG'] = pd.to_numeric(
        df_f['TAHUN ISOLATOR TERPASANG'], errors='coerce')

    fasa = df_f.groupby(['GI', 'NOMOR TOWER']).agg(
        TAHUN_ISO=('TAHUN ISOLATOR TERPASANG', 'min'),
        POLUTAN=('POLUTAN ISOLATOR',
                 lambda x: x.mode()[0] if not x.empty else 'Normal')
    ).reset_index().rename(columns={
        'TAHUN_ISO': 'TAHUN ISOLATOR TERPASANG',
        'POLUTAN':   'POLUTAN ISOLATOR'
    })

    dm = pd.merge(df_g, fasa, on=['GI', 'NOMOR TOWER'], how='left')

    # Buat NAMA TOWER jika belum ada di Excel
    if 'NAMA TOWER' not in dm.columns:
        dm['NAMA TOWER'] = dm['GI'].astype(str) + '-T' + dm['NOMOR TOWER'].astype(str)

    dm['TAHUN ISOLATOR TERPASANG'] = pd.to_numeric(
        dm['TAHUN ISOLATOR TERPASANG'], errors='coerce')
    dm['UMUR_ASET'] = 2025 - dm['TAHUN ISOLATOR TERPASANG']
    dm['UMUR_ASET'] = dm['UMUR_ASET'].fillna(dm['UMUR_ASET'].median())

    bins   = [0, 10, 20, 30, 50, 999]
    labels = ['0-10 Thn', '11-20 Thn', '21-30 Thn', '31-50 Thn', '>50 Thn']
    dm['KATEGORI_UMUR'] = pd.cut(dm['UMUR_ASET'], bins=bins,
                                  labels=labels, right=True)

    for c in ['HALAMAN TOWER', 'POLUTAN ISOLATOR']:
        dm[c] = dm[c].fillna('Normal')

    def label_fn(v):
        if pd.isna(v): return 'Aman'
        s = str(v)
        if any(x in s for x in ['P3', 'P2', 'Patah', 'Hilang', 'Retak']):
            return 'Kritis'
        if any(x in s for x in ['P1', 'Korosi Ringan']):
            return 'Waspada'
        return 'Aman'

    dm['STATUS_RISIKO'] = dm['BESI SIKU BODY TOWER'].apply(label_fn)
    return dm

df = load_data()


(model, encoders, akurasi, report, cm_mat,
 cv_scores, feat_imp, X_te, y_te,
 present_labels, present_names) = train_ml_pipeline(df, len(df))


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚡ PLN Asset Intelligence")
    st.markdown("**UPT Surabaya**")
    st.markdown("---")

    # Status model
    model_src = "💾 Model: dari disk" if os.path.exists(MODEL_PATH) else "🔁 Model: baru dilatih"
    st.caption(model_src)
    if os.path.exists(MODEL_PATH):
        if st.button("🗑️ Hapus & Latih Ulang"):
            os.remove(MODEL_PATH)
            if os.path.exists(ENCODER_PATH):
                os.remove(ENCODER_PATH)
            st.cache_resource.clear()
            st.rerun()

    st.markdown("---")
    all_gi = ['Semua GI'] + sorted(df['GI'].dropna().unique().tolist())
    sel_gi = st.selectbox("🔍 Filter Gardu Induk", all_gi)
    st.markdown("---")
    menu = st.radio("📋 Navigasi Modul:", [
        "🏠 Dashboard Executive",
        "📊 Visualisasi Data",
        "🤖 Prediksi AI (ML)",
        "📋 Data Inspeksi Lengkap",
    ])
    st.markdown("---")
    st.markdown("**Versi:** 3.0.0 | **Update:** Mei 2025")

df_view   = df[df['GI'] == sel_gi].copy() if sel_gi != 'Semua GI' else df.copy()
color_map = {'Aman': SUCCESS, 'Waspada': PLN_YELLOW, 'Kritis': DANGER}


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 – DASHBOARD EXECUTIVE
# ══════════════════════════════════════════════════════════════════════════════
if menu == "🏠 Dashboard Executive":
    pln_header("⚡", "Dashboard Manajemen Aset Transmisi",
               "Sistem Informasi Prediktif – PT PLN (Persero) UPT Surabaya")

    kritis  = (df_view['STATUS_RISIKO'] == 'Kritis').sum()
    waspada = (df_view['STATUS_RISIKO'] == 'Waspada').sum()
    aman    = (df_view['STATUS_RISIKO'] == 'Aman').sum()
    total   = len(df_view)
    pct_k   = f"{kritis/total*100:.1f}%" if total else "0%"
    rata_u  = int(df_view['UMUR_ASET'].mean()) if total else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.markdown(kpi_card(f"{total:,}", "Total Tower", "Unit aset"),
                unsafe_allow_html=True)
    c2.markdown(kpi_card(f"{kritis}", "Tower KRITIS", f"≈{pct_k} dari total", "danger"),
                unsafe_allow_html=True)
    c3.markdown(kpi_card(f"{waspada}", "Tower WASPADA", "Pantau berkala", "warning"),
                unsafe_allow_html=True)
    c4.markdown(kpi_card(f"{aman}", "Tower AMAN", "Kondisi normal", "success"),
                unsafe_allow_html=True)
    c5.markdown(kpi_card(f"{rata_u} Thn", "Rata-rata Umur", "Tahun terpasang"),
                unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 1 ──────────────────────────────────────────────────────────────────
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown('<div class="section-card"><div class="section-title">📌 Proporsi Status Risiko</div>',
                    unsafe_allow_html=True)
        fig = px.pie(df_view, names='STATUS_RISIKO', color='STATUS_RISIKO',
                     color_discrete_map=color_map, hole=0.55)
        fig.update_traces(textposition='outside', textinfo='percent+label',
                          textfont=dict(color=BLACK, size=12, family='Arial'),
                          marker=dict(line=dict(color='white', width=2)))
        fig.update_layout(**base_layout(280), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-card"><div class="section-title">🏭 Distribusi Risiko per Gardu Induk</div>',
                    unsafe_allow_html=True)
        gi_s = df_view.groupby(['GI', 'STATUS_RISIKO']).size().reset_index(name='Jumlah')
        fig  = px.bar(gi_s, x='GI', y='Jumlah', color='STATUS_RISIKO',
                      color_discrete_map=color_map, barmode='stack', text_auto=True)
        fig.update_traces(textfont=dict(color=BLACK, size=10))
        fig.update_layout(**base_layout(280), legend=LEGEND,
                          xaxis=dict(title='', **AXIS),
                          yaxis=dict(title='Jumlah Tower', **AXIS))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Row 2 ──────────────────────────────────────────────────────────────────
    col3, col4 = st.columns(2)
    with col3:
        st.markdown('<div class="section-card"><div class="section-title">🏗️ Sebaran Umur Aset per Kategori</div>',
                    unsafe_allow_html=True)
        ks  = df_view.groupby(['KATEGORI_UMUR', 'STATUS_RISIKO']).size().reset_index(name='Jumlah')
        fig = px.bar(ks, x='KATEGORI_UMUR', y='Jumlah', color='STATUS_RISIKO',
                     color_discrete_map=color_map, barmode='group', text='Jumlah',
                     category_orders={'KATEGORI_UMUR': [
                         '0-10 Thn', '11-20 Thn', '21-30 Thn', '31-50 Thn', '>50 Thn']})
        fig.update_traces(textposition='outside', textfont=dict(color=BLACK, size=10))
        fig.update_layout(**base_layout(270), legend=LEGEND,
                          xaxis=dict(title='Kategori Umur', **AXIS),
                          yaxis=dict(title='Jumlah Tower', **AXIS))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col4:
        st.markdown('<div class="section-card"><div class="section-title">🌫️ Distribusi Level Polutan Isolator</div>',
                    unsafe_allow_html=True)
        pc  = df_view['POLUTAN ISOLATOR'].value_counts().reset_index()
        pc.columns = ['Polutan', 'Jumlah']
        fig = px.bar(pc, x='Jumlah', y='Polutan', orientation='h',
                     color='Jumlah', color_continuous_scale=['#d4e6ff', PLN_BLUE],
                     text='Jumlah')
        fig.update_traces(textposition='outside', textfont=dict(color=BLACK, size=11))
        fig.update_layout(**base_layout(270), coloraxis_showscale=False,
                          xaxis=dict(title='Jumlah Tower', **AXIS),
                          yaxis=dict(title='', tickfont=dict(color=BLACK)))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Top-10 Kritis ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-card"><div class="section-title">🚨 10 Tower Prioritas Kritis – Butuh SPK Segera</div>',
                unsafe_allow_html=True)
    kdf = df_view[df_view['STATUS_RISIKO'] == 'Kritis'][[
        'NAMA TOWER', 'GI', 'NOMOR TOWER', 'HALAMAN TOWER',
        'POLUTAN ISOLATOR', 'UMUR_ASET', 'BESI SIKU BODY TOWER'
    ]].sort_values('UMUR_ASET', ascending=False).head(10)

    if kdf.empty:
        st.info("Tidak ada tower berstatus KRITIS pada filter ini.")
    else:
        st.dataframe(kdf.reset_index(drop=True), use_container_width=True, height=320)
    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 – VISUALISASI DATA
# ══════════════════════════════════════════════════════════════════════════════
elif menu == "📊 Visualisasi Data":
    pln_header("📊", "Visualisasi & Eksplorasi Data Aset",
               "Analisis mendalam kondisi jaringan transmisi")

    tab1, tab2, tab3 = st.tabs([
        "📈 Distribusi & Sebaran",
        "🏭 Analisis per GI",
        "⏳ Umur & Polutan"
    ])

    # ── Tab 1 ──────────────────────────────────────────────────────────────────
    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            fig = px.histogram(df_view, x='UMUR_ASET', color='STATUS_RISIKO',
                               color_discrete_map=color_map, barmode='overlay',
                               nbins=25, opacity=0.75,
                               title='Distribusi Umur Aset vs Status Risiko',
                               labels={'UMUR_ASET': 'Umur Aset (Tahun)', 'count': 'Jumlah'})
            fig.update_layout(**base_layout(320), legend=LEGEND,
                              xaxis=dict(title='Umur Aset (Tahun)', **AXIS),
                              yaxis=dict(title='Jumlah Tower', **AXIS),
                              title_font=dict(color=BLACK))
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            sb  = df_view.groupby(['KATEGORI_UMUR', 'STATUS_RISIKO']).size().reset_index(name='n')
            fig = px.sunburst(sb, path=['KATEGORI_UMUR', 'STATUS_RISIKO'], values='n',
                              color='STATUS_RISIKO', color_discrete_map=color_map,
                              title='Sunburst: Kategori Umur & Status Risiko')
            fig.update_traces(textfont=dict(color=BLACK, size=12))
            fig.update_layout(**base_layout(320), title_font=dict(color=BLACK))
            st.plotly_chart(fig, use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            hs  = df_view.groupby(['HALAMAN TOWER', 'STATUS_RISIKO']).size().reset_index(name='n')
            fig = px.bar(hs, x='HALAMAN TOWER', y='n', color='STATUS_RISIKO',
                         color_discrete_map=color_map, barmode='group', text='n',
                         title='Kondisi Halaman Tower vs Status Risiko',
                         labels={'n': 'Jumlah', 'HALAMAN TOWER': 'Kondisi Halaman'})
            fig.update_traces(textposition='outside', textfont=dict(color=BLACK, size=10))
            fig.update_layout(**base_layout(300), legend=LEGEND,
                              xaxis=dict(title='Kondisi Halaman', **AXIS),
                              yaxis=dict(title='Jumlah Tower', **AXIS),
                              title_font=dict(color=BLACK))
            st.plotly_chart(fig, use_container_width=True)

        with c4:
            bub = df_view.groupby(['POLUTAN ISOLATOR', 'KATEGORI_UMUR']).agg(
                Jumlah=('NOMOR TOWER', 'count'),
                Pct_Kritis=('STATUS_RISIKO', lambda x: (x == 'Kritis').mean() * 100)
            ).reset_index()
            fig = px.scatter(bub, x='KATEGORI_UMUR', y='POLUTAN ISOLATOR',
                             size='Jumlah', color='Pct_Kritis',
                             color_continuous_scale=['#d4edda', '#fff3cd', DANGER],
                             title='Bubble: Polutan × Umur (ukuran=jumlah, warna=% Kritis)',
                             size_max=55, text='Jumlah',
                             category_orders={'KATEGORI_UMUR': [
                                 '0-10 Thn', '11-20 Thn', '21-30 Thn', '31-50 Thn', '>50 Thn']})
            fig.update_traces(textfont=dict(color=BLACK, size=10))
            fig.update_layout(**base_layout(300),
                              xaxis=dict(title='Kategori Umur', **AXIS),
                              yaxis=dict(title='Level Polutan', **AXIS),
                              title_font=dict(color=BLACK),
                              coloraxis_colorbar=dict(
                                  title='% Kritis',
                                  tickfont=dict(color=BLACK),
                                  title_font=dict(color=BLACK)))
            st.plotly_chart(fig, use_container_width=True)

    # ── Tab 2 ──────────────────────────────────────────────────────────────────
    with tab2:
        gi_d = df_view.groupby('GI').agg(
            Total=('NOMOR TOWER', 'count'),
            Kritis=('STATUS_RISIKO',  lambda x: (x == 'Kritis').sum()),
            Waspada=('STATUS_RISIKO', lambda x: (x == 'Waspada').sum()),
            Aman=('STATUS_RISIKO',   lambda x: (x == 'Aman').sum()),
            Umur_Rata=('UMUR_ASET', 'mean')
        ).reset_index()
        gi_d['Pct_Kritis'] = (gi_d['Kritis'] / gi_d['Total'] * 100).round(1)

        gi_melt = gi_d.melt(id_vars='GI', value_vars=['Kritis', 'Waspada', 'Aman'],
                             var_name='Status', value_name='Jumlah')
        fig = px.bar(gi_melt, x='GI', y='Jumlah', color='Status',
                     color_discrete_map={'Kritis': DANGER, 'Waspada': PLN_YELLOW, 'Aman': SUCCESS},
                     barmode='stack', text_auto=True,
                     title='Komposisi Status Risiko per Gardu Induk')
        fig.update_traces(textfont=dict(color=BLACK, size=10))
        fig.update_layout(**base_layout(360), legend=LEGEND,
                          xaxis=dict(title='', **AXIS),
                          yaxis=dict(title='Jumlah Tower', **AXIS),
                          title_font=dict(color=BLACK))
        st.plotly_chart(fig, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            srt = gi_d.sort_values('Pct_Kritis')
            fig = px.bar(srt, x='Pct_Kritis', y='GI', orientation='h',
                         color='Pct_Kritis', color_continuous_scale=['#ffe5e5', DANGER],
                         title='% Tower Kritis per GI',
                         text=srt['Pct_Kritis'].apply(lambda v: f"{v}%"))
            fig.update_traces(textposition='outside', textfont=dict(color=BLACK))
            fig.update_layout(**base_layout(320), coloraxis_showscale=False,
                              xaxis=dict(title='% Kritis', **AXIS),
                              yaxis=dict(title='', tickfont=dict(color=BLACK)),
                              title_font=dict(color=BLACK))
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            srt = gi_d.sort_values('Umur_Rata')
            fig = px.bar(srt, x='Umur_Rata', y='GI', orientation='h',
                         color='Umur_Rata', color_continuous_scale=['#d4e6ff', PLN_BLUE],
                         title='Rata-rata Umur Aset per GI (Tahun)',
                         text=srt['Umur_Rata'].apply(lambda v: f"{v:.1f}"))
            fig.update_traces(textposition='outside', textfont=dict(color=BLACK))
            fig.update_layout(**base_layout(320), coloraxis_showscale=False,
                              xaxis=dict(title='Umur Rata-rata (Tahun)', **AXIS),
                              yaxis=dict(title='', tickfont=dict(color=BLACK)),
                              title_font=dict(color=BLACK))
            st.plotly_chart(fig, use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            fig = px.scatter(gi_d, x='Umur_Rata', y='Pct_Kritis',
                             size='Total', color='GI', text='GI', size_max=50,
                             title='Matriks Risiko: Umur vs % Kritis per GI',
                             labels={'Umur_Rata': 'Umur Rata-rata (Thn)',
                                     'Pct_Kritis': '% Kritis'})
            fig.update_traces(textposition='top center',
                              textfont=dict(color=BLACK, size=9))
            fig.update_layout(**base_layout(330), showlegend=False,
                              xaxis=dict(title='Umur Rata-rata (Thn)', **AXIS),
                              yaxis=dict(title='% Tower Kritis', **AXIS),
                              title_font=dict(color=BLACK))
            st.plotly_chart(fig, use_container_width=True)

        with c4:
            pivot = df_view.groupby(['GI', 'KATEGORI_UMUR']).size().unstack(fill_value=0)
            fig   = px.imshow(pivot, text_auto=True, aspect='auto',
                              color_continuous_scale=['white', PLN_BLUE],
                              title='Heatmap: Jumlah Tower per GI & Kategori Umur')
            fig.update_traces(textfont=dict(color=BLACK, size=10))
            fig.update_layout(**base_layout(330),
                              xaxis=dict(title='Kategori Umur',
                                         tickfont=dict(color=BLACK)),
                              yaxis=dict(title='', tickfont=dict(color=BLACK)),
                              title_font=dict(color=BLACK))
            st.plotly_chart(fig, use_container_width=True)

    # ── Tab 3 ──────────────────────────────────────────────────────────────────
    with tab3:
        c1, c2 = st.columns(2)
        with c1:
            tree = df_view.groupby(
                ['KATEGORI_UMUR', 'STATUS_RISIKO']).size().reset_index(name='n')
            fig  = px.treemap(tree, path=['KATEGORI_UMUR', 'STATUS_RISIKO'], values='n',
                              color='STATUS_RISIKO', color_discrete_map=color_map,
                              title='Treemap: Kategori Umur → Status Risiko')
            fig.update_traces(textfont=dict(color=BLACK, size=12))
            fig.update_layout(**base_layout(340), title_font=dict(color=BLACK))
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            fig = px.violin(df_view, x='POLUTAN ISOLATOR', y='UMUR_ASET',
                            color='STATUS_RISIKO', box=True, points='outliers',
                            color_discrete_map=color_map,
                            title='Distribusi Umur Aset per Level Polutan & Risiko')
            fig.update_layout(**base_layout(340), legend=LEGEND,
                              xaxis=dict(title='Level Polutan', **AXIS),
                              yaxis=dict(title='Umur Aset (Tahun)', **AXIS),
                              title_font=dict(color=BLACK))
            st.plotly_chart(fig, use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            yr  = df_view.dropna(subset=['TAHUN ISOLATOR TERPASANG'])
            yr  = yr.groupby('TAHUN ISOLATOR TERPASANG').size().reset_index(name='Jumlah')
            fig = px.area(yr, x='TAHUN ISOLATOR TERPASANG', y='Jumlah',
                          title='Tren Pemasangan Isolator per Tahun',
                          line_shape='spline',
                          color_discrete_sequence=[PLN_BLUE])
            fig.update_layout(**base_layout(280),
                              xaxis=dict(title='Tahun Pemasangan', **AXIS),
                              yaxis=dict(title='Jumlah Tower', **AXIS),
                              title_font=dict(color=BLACK))
            st.plotly_chart(fig, use_container_width=True)

        with c4:
            ps  = df_view.groupby(['POLUTAN ISOLATOR', 'STATUS_RISIKO']).size().reset_index(name='n')
            fig = px.bar(ps, x='POLUTAN ISOLATOR', y='n', color='STATUS_RISIKO',
                         color_discrete_map=color_map, barmode='stack', text_auto=True,
                         title='Komposisi Status per Level Polutan')
            fig.update_traces(textfont=dict(color=BLACK, size=10))
            fig.update_layout(**base_layout(280), legend=LEGEND,
                              xaxis=dict(title='Level Polutan', **AXIS),
                              yaxis=dict(title='Jumlah Tower', **AXIS),
                              title_font=dict(color=BLACK))
            st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 – PREDIKSI AI (ML)
# ══════════════════════════════════════════════════════════════════════════════
elif menu == "🤖 Prediksi AI (ML)":
    pln_header("🤖", "Predictive Maintenance – AI Asset Risk",
               "Ensemble: Random Forest + Gradient Boosting | 5-Fold Cross Validation")

    # KPI akurasi
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(kpi_card(f"{akurasi*100:.1f}%", "Akurasi Model", "Test Set"),
                unsafe_allow_html=True)
    c2.markdown(kpi_card(f"{cv_scores.mean()*100:.1f}%", "CV Akurasi (5-fold)",
                          f"±{cv_scores.std()*100:.1f}%"),
                unsafe_allow_html=True)
    c3.markdown(kpi_card(f"{report['macro avg']['precision']*100:.1f}%",
                          "Precision (macro)"), unsafe_allow_html=True)
    c4.markdown(kpi_card(f"{report['macro avg']['recall']*100:.1f}%",
                          "Recall (macro)"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="section-card"><div class="section-title">🔢 Confusion Matrix</div>',
                    unsafe_allow_html=True)
        fig = px.imshow(cm_mat, text_auto=True, aspect='auto',
                        x=present_names, y=present_names,
                        color_continuous_scale=['white', PLN_BLUE],
                        labels={'x': 'Prediksi', 'y': 'Aktual'})
        fig.update_traces(textfont=dict(color=BLACK, size=13, family='Arial Black'))
        fig.update_layout(**base_layout(290), coloraxis_showscale=False,
                          xaxis=dict(title='Prediksi', **AXIS),
                          yaxis=dict(title='Aktual', **AXIS))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-card"><div class="section-title">📊 Feature Importance</div>',
                    unsafe_allow_html=True)
        fi_df = pd.DataFrame({'Fitur': list(feat_imp.keys()),
                               'Importance': list(feat_imp.values())
                               }).sort_values('Importance')
        fig = px.bar(fi_df, x='Importance', y='Fitur', orientation='h',
                     color='Importance', color_continuous_scale=['#d4e6ff', PLN_BLUE],
                     text=fi_df['Importance'].apply(lambda v: f"{v:.3f}"))
        fig.update_traces(textposition='outside', textfont=dict(color=BLACK, size=11))
        fig.update_layout(**base_layout(290), coloraxis_showscale=False,
                          xaxis=dict(title='Importance Score', **AXIS),
                          yaxis=dict(title='', tickfont=dict(color=BLACK)))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="section-card"><div class="section-title">📉 CV Score per Fold</div>',
                    unsafe_allow_html=True)
        n_folds = len(cv_scores)
        fold_df = pd.DataFrame({
            'Fold':     [f"Fold {i+1}" for i in range(n_folds)],
            'Accuracy': cv_scores * 100
        })
        fig = px.bar(fold_df, x='Fold', y='Accuracy',
                     text=fold_df['Accuracy'].apply(lambda v: f"{v:.1f}%"),
                     color='Accuracy', color_continuous_scale=['#d4e6ff', PLN_BLUE])
        fig.add_hline(y=cv_scores.mean() * 100, line_dash='dash',
                       line_color=PLN_YELLOW,
                       annotation_text=f"Rata-rata: {cv_scores.mean()*100:.1f}%",
                       annotation_font_color=BLACK)
        fig.update_traces(textposition='outside', textfont=dict(color=BLACK, size=11))
        fig.update_layout(**base_layout(290), coloraxis_showscale=False,
                          xaxis=dict(title='', **AXIS),
                          yaxis=dict(title='Akurasi (%)', range=[0, 110], **AXIS))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Classification Report
    with st.expander("📋 Lihat Classification Report Lengkap"):
        rpt_rows = []
        for cls in ['Aman', 'Waspada', 'Kritis']:
            rpt_rows.append({
                'Kelas':     cls,
                'Precision': f"{report[cls]['precision']*100:.1f}%",
                'Recall':    f"{report[cls]['recall']*100:.1f}%",
                'F1-Score':  f"{report[cls]['f1-score']*100:.1f}%",
                'Support':   int(report[cls]['support'])
            })
        st.dataframe(pd.DataFrame(rpt_rows).set_index('Kelas'),
                     use_container_width=True)

    st.markdown("---")

    # ── Form Prediksi ──────────────────────────────────────────────────────────
    st.subheader("🔮 Simulasi Prediksi Aset Baru")
    st.markdown('<div class="section-card">', unsafe_allow_html=True)

    fc1, fc2, fc3, fc4 = st.columns(4)
    with fc1: in_umur    = st.number_input("⏳ Umur Aset (Tahun)", 0, 60, 10, 1)
    with fc2: in_gi      = st.selectbox("🏭 Gardu Induk (GI)",
                                         sorted(df['GI'].dropna().unique()))
    with fc3: in_halaman = st.selectbox("🌿 Kondisi Halaman",
                                         df['HALAMAN TOWER'].unique())
    with fc4: in_polutan = st.selectbox("🌫️ Tingkat Polutan",
                                         df['POLUTAN ISOLATOR'].unique())

    if st.button("⚡ Prediksi Risiko Aset", use_container_width=True):
        try:
            gi_enc  = encoders['GI'].transform([str(in_gi)])[0]
            hal_enc = encoders['HALAMAN TOWER'].transform([in_halaman])[0]
            pol_enc = encoders['POLUTAN ISOLATOR'].transform([in_polutan])[0]
            inp     = np.array([[in_umur, gi_enc, hal_enc, pol_enc]])
            pred    = model.predict(inp)[0]
            proba   = model.predict_proba(inp)[0]

            # Mapping hasil prediksi — handle 2 atau 3 kelas
            classes = model.classes_
            prob_map = {c: 0.0 for c in [0, 1, 2]}
            for c, p in zip(classes, proba):
                prob_map[c] = p

            label_map = {
                0: ('AMAN ✅',    SUCCESS,    'white'),
                1: ('WASPADA ⚠️', PLN_YELLOW, '#222'),
                2: ('KRITIS 🚨',  DANGER,     'white'),
            }
            lbl, col_res, tcol = label_map.get(pred, ('AMAN ✅', SUCCESS, 'white'))

            st.markdown(f"""
            <div style="background:{col_res};color:{tcol};border-radius:12px;
                        padding:18px 24px;margin-top:14px;text-align:center;">
                <h2 style="margin:0">Status Prediksi: {lbl}</h2>
                <p style="margin:6px 0 0">
                    P(Aman) = <b>{prob_map[0]*100:.1f}%</b> &nbsp;|&nbsp;
                    P(Waspada) = <b>{prob_map[1]*100:.1f}%</b> &nbsp;|&nbsp;
                    P(Kritis) = <b>{prob_map[2]*100:.1f}%</b>
                </p>
            </div>""", unsafe_allow_html=True)

            # Gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=prob_map[2] * 100,
                delta={'reference': 50, 'font': {'color': BLACK}},
                title={'text': "Probabilitas KRITIS (%)",
                        'font': {'color': BLACK, 'size': 14}},
                number={'font': {'color': BLACK}},
                gauge={
                    'axis': {'range': [0, 100], 'tickfont': {'color': BLACK}},
                    'bar':  {'color': col_res},
                    'steps': [
                        {'range': [0, 30],   'color': '#d4edda'},
                        {'range': [30, 60],  'color': '#fff3cd'},
                        {'range': [60, 100], 'color': '#f8d7da'},
                    ],
                    'threshold': {'line': {'color': DANGER, 'width': 3}, 'value': 60}
                }
            ))
            fig.update_layout(height=260, paper_bgcolor='white', font=FONT,
                               margin=dict(t=30, b=10, l=20, r=20))
            st.plotly_chart(fig, use_container_width=True)

        except ValueError as e:
            st.error(f"❌ Error: {e} — pastikan nilai input ada dalam data latih.")

    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 – DATA INSPEKSI LENGKAP
# ══════════════════════════════════════════════════════════════════════════════
elif menu == "📋 Data Inspeksi Lengkap":
    pln_header("📋", "Data Inspeksi Aset Lengkap",
               "Tabel interaktif seluruh data tower – filter, scroll, & export")

    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        f_status = st.multiselect("Filter Status Risiko",
                                   ['Aman', 'Waspada', 'Kritis'],
                                   default=['Aman', 'Waspada', 'Kritis'])
    with fc2:
        u_min = int(df_view['UMUR_ASET'].min())
        u_max = int(df_view['UMUR_ASET'].max())
        f_umur = st.slider("Filter Umur Aset (Tahun)", u_min, u_max, (u_min, u_max))
    with fc3:
        srch = st.text_input("🔍 Cari Nama / Nomor Tower", "")

    tbl = df_view[df_view['STATUS_RISIKO'].isin(f_status)].copy()
    tbl = tbl[(tbl['UMUR_ASET'] >= f_umur[0]) & (tbl['UMUR_ASET'] <= f_umur[1])]
    if srch:
        mask = (tbl['NAMA TOWER'].str.contains(srch, case=False, na=False) |
                tbl['NOMOR TOWER'].str.contains(srch, case=False, na=False))
        tbl = tbl[mask]

    st.markdown(f"**Menampilkan {len(tbl):,} dari {len(df_view):,} tower**")

    COLS    = ['NAMA TOWER', 'GI', 'NOMOR TOWER', 'HALAMAN TOWER',
               'POLUTAN ISOLATOR', 'UMUR_ASET', 'KATEGORI_UMUR',
               'BESI SIKU BODY TOWER', 'STATUS_RISIKO']
    cols_ok = [c for c in COLS if c in tbl.columns]

    st.dataframe(
        tbl[cols_ok].reset_index(drop=True),
        use_container_width=True,
        height=600,
        column_config={
            "UMUR_ASET":    st.column_config.NumberColumn("Umur (Thn)", format="%d"),
            "STATUS_RISIKO": st.column_config.TextColumn("Status Risiko"),
        }
    )

    csv = tbl[cols_ok].to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Data (CSV)", csv,
                        "inspeksi_tower.csv", "text/csv",
                        use_container_width=True)