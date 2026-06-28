"""
SETRUM — Sistem Evaluasi Trips & Rute untuk Mobilitas EV
Aplikasi web Streamlit — prototipe untuk SIC SATRIA DATA 2026

REVISI 4: 
- Integrasi parsing dinamis langsung dari Data_SPKLU_Rapi.xlsx - Sheet1.csv
- Penyesuaian tema antarmuka ke Biru Abu-Abu (Blue-Gray).
- Penulisan ulang formula menggunakan format LaTeX.
"""

import numpy as np
import pandas as pd
import streamlit as st
import re
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

# =========================================================
# KONFIGURASI HALAMAN & STYLE
# =========================================================

st.set_page_config(
    page_title="SETRUM — Cek Kesiapan Infrastruktur EV",
    page_icon="🔌",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Tema Profesional Biru Abu-abu
BLUE_GRAY = "#2C3E50"
BLUE_LIGHT = "#EBF0F6"
BLUE_GRADIENT_1 = "#1E3B70"
BLUE_GRADIENT_2 = "#29539B"
AMBER = "#E0A526"
AMBER_LIGHT = "#FFF6DA"
RED = "#C1502E"
RED_LIGHT = "#FBE7E2"
GRAY = "#5A5A5A"

CUSTOM_CSS = f"""
<style>
    .main {{ background-color: #FAFBFA; }}
    h1, h2, h3 {{ color: {BLUE_GRAY}; font-family: 'Segoe UI', sans-serif; }}
    .hero-box {{
        background: linear-gradient(135deg, {BLUE_GRADIENT_1} 0%, {BLUE_GRADIENT_2} 100%);
        padding: 28px 32px; border-radius: 14px; color: white; margin-bottom: 24px;
    }}
    .hero-box h1 {{ color: white; margin-bottom: 4px; }}
    .hero-box p {{ color: #D4E0F0; font-size: 17px; margin-bottom: 0; }}
    .stat-card {{
        background-color: white; border: 1px solid #E0E0E0; border-radius: 12px;
        padding: 18px; text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }}
    .stat-card .big {{ font-size: 28px; font-weight: 700; color: {BLUE_GRAY}; }}
    .stat-card .label {{ font-size: 13px; color: {GRAY}; }}
    .result-box {{ border-radius: 12px; padding: 20px 24px; margin-top: 12px; }}
    .badge {{ display: inline-block; padding: 4px 14px; border-radius: 20px; font-weight: 600; font-size: 14px; }}
    div[data-testid="stMetricValue"] {{ color: {BLUE_GRAY}; }}
    .footer-note {{
        font-size: 12px; color: #999999; margin-top: 40px;
        border-top: 1px solid #EEEEEE; padding-top: 12px;
    }}
    .data-tag {{
        display: inline-block; background-color: {BLUE_LIGHT}; color: {BLUE_GRAY};
        font-size: 11px; font-weight: 600; padding: 2px 10px; border-radius: 10px; margin-bottom: 8px;
    }}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# =========================================================
# DATA & PREPROCESSING DINAMIS
# =========================================================

@st.cache_data
def load_spklu_semua():
    # Membaca data mentah
    df = pd.read_csv("Data_SPKLU_Rapi.xlsx - Sheet1.csv")
    
    # Ekstraksi Daya (kW) dan Kategori dari kolom 'Tipe Charger'
    df['daya_kw'] = df['Tipe Charger'].str.extract(r'(\d+)').astype(float)
    df['kategori_charger'] = df['Tipe Charger'].str.extract(r'([a-zA-Z]+)')[0].str.lower()
    
    # Fungsi ekstraksi kota sederhana dari 'Alamat'
    def extract_kota(alamat):
        alamat = str(alamat).lower()
        if pd.isna(alamat): return "Lainnya"
        
        # Kata kunci kota utama
        keywords = ["jakarta", "batam", "mamuju", "majene", "pasangkayu", "polewali", "bandung"]
        for kw in keywords:
            if kw in alamat:
                return kw.title() if kw != "polewali" else "Polewali Mandar"
                
        # Regex penangkap nama kota/kabupaten
        match = re.search(r'(kota\s+[\w\s]+|kabupaten\s+[\w\s]+|kab\.\s+[\w\s]+)', alamat)
        if match: return match.group(1).title()
        
        # Fallback split
        parts = alamat.split(',')
        if len(parts) >= 2: return parts[-2].strip().title()
        return "Lainnya"

    df['kota'] = df['Alamat'].apply(extract_kota)
    return df

@st.cache_data
def load_data_kota(df_spklu):
    # Agregasi otomatis
    kota_df = df_spklu.groupby('kota').agg(
        jumlah_spklu=('Nama Stasiun', 'count'),
        rata2_daya_kw=('daya_kw', 'mean'),
        jumlah_ultrafast=('kategori_charger', lambda x: (x == 'ultrafast').sum())
    ).reset_index()
    return kota_df[kota_df['jumlah_spklu'] > 0]

@st.cache_data
def load_data_ev():
    try:
        return pd.read_csv("data_ev_valid.csv")
    except FileNotFoundError:
        return pd.DataFrame({
            "model": ["Wuling Air EV", "Wuling Binguo EV", "MG 4 EV", "BYD Atto 3", "Hyundai Ioniq 5"],
            "harga_juta": [243, 348, 433, 515, 782],
            "sumber": ["SEVA.id", "SEVA.id", "Moladin", "Kompas", "AstraOtoshop"]
        })

spklu_semua = load_spklu_semua()
kota_df = load_data_kota(spklu_semua)
data_ev = load_data_ev()

# =========================================================
# PEMODELAN MATEMATIKA
# =========================================================

# --- 1. Skor Infrastruktur (Min-Max Normalization) ---
N_MIN = kota_df["jumlah_spklu"].min()
N_MAX = kota_df["jumlah_spklu"].max()
kota_df["skor_infra"] = ((kota_df["jumlah_spklu"] - N_MIN) / (N_MAX - N_MIN) * 100).round(1)

# --- 2. Skor Kualitas Charging ---
BOBOT_KATEGORI = {"standard": 1, "medium": 2, "fast": 4, "ultrafast": 8}
W_MAX = max(BOBOT_KATEGORI.values())

# --- 3. Skor Gabungan Kesiapan Infrastruktur ---
ALPHA = 0.6

# --- 4. Skor Kesesuaian Budget ---
K_PELURUHAN = 2

def skor_budget(harga_juta, budget_juta):
    if harga_juta <= budget_juta:
        return 100.0
    selisih_relatif = (harga_juta - budget_juta) / budget_juta
    return round(100 * np.exp(-K_PELURUHAN * selisih_relatif), 1)

# --- 5. Skor Akhir Rekomendasi ---
BETA = 0.5

# --- 6. Analisis Klaster (GMM) ---
@st.cache_data
def jalankan_clustering(df_kota):
    if len(df_kota) < 3:
        df = df_kota.copy()
        df["klaster"] = "Siap"
        return df
        
    fitur = ["jumlah_spklu", "rata2_daya_kw", "jumlah_ultrafast"]
    X = df_kota[fitur].fillna(0).values
    X_scaled = StandardScaler().fit_transform(X)
    
    n_comp = min(3, len(df_kota))
    gmm = GaussianMixture(n_components=n_comp, random_state=42)
    labels = gmm.fit_predict(X_scaled)
    
    df = df_kota.copy()
    df["klaster_raw"] = labels
    urutan = df.groupby("klaster_raw")["jumlah_spklu"].mean().sort_values().index
    
    mapping = {urutan[i]: ["Belum Layak", "Bersyarat", "Siap"][i] for i in range(len(urutan))}
    df["klaster"] = df["klaster_raw"].map(mapping)
    return df

kota_df = jalankan_clustering(kota_df)
WARNA_KLASTER = {"Siap": BLUE_GRAY, "Bersyarat": AMBER, "Belum Layak": RED}

# =========================================================
# HEADER / HERO
# =========================================================

st.markdown(f"""
<div class="hero-box">
    <h1>🔌 SETRUM</h1>
    <p>Cek kesiapan infrastruktur EV kotamu & kecocokan model EV dengan budgetmu — 100% berbasis data riil.</p>
</div>
""", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown('<div class="stat-card"><div class="big">65%</div><div class="label">konsumen khawatir kehabisan baterai (Populix)</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div class="stat-card"><div class="big">{len(spklu_semua):,}</div><div class="label">SPKLU dalam dataset</div></div>'.replace(",", "."), unsafe_allow_html=True)
with col3:
    n_kota = kota_df["kota"].nunique()
    st.markdown(f'<div class="stat-card"><div class="big">{n_kota}</div><div class="label">kota dianalisis</div></div>', unsafe_allow_html=True)
with col4:
    st.markdown(f'<div class="stat-card"><div class="big">{len(data_ev)}</div><div class="label">model EV tervalidasi</div></div>', unsafe_allow_html=True)

st.write("")

# =========================================================
# TAB UTAMA
# =========================================================

tab1, tab2, tab3 = st.tabs(["✅ Cek Kesiapan & Rekomendasi", "📊 Peta Kesiapan Kota", "🧮 Tentang Model"])

# ---------- TAB 1: FEASIBILITY CHECK ----------
with tab1:
    st.subheader("Apakah EV cocok untukmu di kotamu?")
    st.markdown('<span class="data-tag">📍 Skor infrastruktur otomatis dari data mentah</span> <span class="data-tag">💰 Harga EV tertelusur</span>', unsafe_allow_html=True)
    
    colA, colB = st.columns([1, 2])
    with colA:
        kota_pilih = st.selectbox("Kota domisili", kota_df["kota"].sort_values().tolist())
        anggaran = st.slider("Anggaran (juta Rupiah)", 150, 900, 300, step=10)
        cek = st.button("✅ Cek Kelayakan", type="primary", use_container_width=True)

        row_kota = kota_df[kota_df["kota"] == kota_pilih].iloc[0]
        st.caption(f"📍 {kota_pilih}: **{int(row_kota['jumlah_spklu'])} SPKLU** tercatat, rata-rata daya **{row_kota['rata2_daya_kw']:.1f} kW**.")

    if cek or "feas_result" in st.session_state:
        skor_infra = float(row_kota["skor_infra"])
        spklu_kota = spklu_semua[spklu_semua["kota"] == kota_pilih].copy()
        
        if len(spklu_kota) > 0:
            spklu_kota["bobot"] = spklu_kota["kategori_charger"].map(BOBOT_KATEGORI).fillna(1)
            skor_kualitas = round(spklu_kota["bobot"].mean() / W_MAX * 100, 1)
        else:
            skor_kualitas = 0.0

        skor_kesiapan = round(ALPHA * skor_infra + (1 - ALPHA) * skor_kualitas, 1)

        df_rekom = data_ev.copy()
        df_rekom["Skor Budget"] = df_rekom["harga_juta"].apply(lambda h: skor_budget(h, anggaran))
        df_rekom["Skor Akhir"] = (BETA * skor_kesiapan + (1 - BETA) * df_rekom["Skor Budget"]).round(1)
        df_rekom = df_rekom.sort_values("Skor Akhir", ascending=False).reset_index(drop=True)
        
        df_rekom_tampil = df_rekom.rename(columns={
            "model": "Model EV", "harga_juta": "Harga (juta Rp)", "sumber": "Sumber Harga"
        })[["Model EV", "Harga (juta Rp)", "Skor Budget", "Skor Akhir", "Sumber Harga"]]

        st.session_state["feas_result"] = df_rekom_tampil
        skor_top = df_rekom_tampil.iloc[0]["Skor Akhir"]

        if skor_top >= 75:
            label, warna, bg = "Sangat Layak", BLUE_GRAY, BLUE_LIGHT
        elif skor_top >= 50:
            label, warna, bg = "Layak dengan Catatan", AMBER, AMBER_LIGHT
        else:
            label, warna, bg = "Belum Layak", RED, RED_LIGHT

        mobil_terbaik = df_rekom_tampil.iloc[0]["Model EV"]
        harga_terbaik = df_rekom_tampil.iloc[0]["Harga (juta Rp)"]

        with colB:
            st.markdown(f"""
            <div class="result-box" style="background-color:{bg}; border: 1px solid {warna};">
                <span class="badge" style="background-color:{warna}; color:white;">{label}</span>
                <div style="font-size:36px; font-weight:700; color:{warna}; margin-top:8px;">{skor_top}/100</div>
                <div style="color:{GRAY}; margin-bottom:10px;">Skor akhir untuk kombinasi kota & anggaranmu</div>
                <div style="font-size:15px; color:{GRAY};">📌 Rekomendasi utama untukmu:</div>
                <div style="font-size:20px; font-weight:700; color:{warna};">{mobil_terbaik}</div>
                <div style="font-size:14px; color:{GRAY};">Rp {harga_terbaik} juta (OTR)</div>
            </div>
            """, unsafe_allow_html=True)

            m1, m2 = st.columns(2)
            m1.metric("Skor Infrastruktur Kota", f"{skor_infra}/100")
            m2.metric("Skor Kesiapan Gabungan", f"{skor_kesiapan}/100")

        st.write("")
        st.write(f"**Rekomendasi lengkap model EV:**")
        st.dataframe(df_rekom_tampil, use_container_width=True, hide_index=True)

# ---------- TAB 2: PETA KESIAPAN KOTA ----------
with tab2:
    st.subheader("Hasil Analisis Klaster — Kesiapan Infrastruktur EV per Kota")
    
    colA, colB = st.columns([3, 2])
    with colA:
        st.write("**Sebaran kota berdasarkan klaster:**")
        for klaster in ["Siap", "Bersyarat", "Belum Layak"]:
            if "klaster" not in kota_df.columns: continue
            subset = kota_df[kota_df["klaster"] == klaster]
            if subset.empty: continue
            
            st.markdown(f"<span class='badge' style='background-color:{WARNA_KLASTER[klaster]};color:white;'>{klaster}</span>", unsafe_allow_html=True)
            st.dataframe(
                subset[["kota", "jumlah_spklu", "rata2_daya_kw", "jumlah_ultrafast", "skor_infra"]]
                .rename(columns={"kota": "Kota", "jumlah_spklu": "Jumlah SPKLU", "rata2_daya_kw": "Rata2 Daya (kW)",
                                  "jumlah_ultrafast": "Jml Ultrafast", "skor_infra": "Skor Infrastruktur"}),
                use_container_width=True, hide_index=True
            )

    with colB:
        st.write("**Posisi kota: jumlah SPKLU vs rata-rata daya**")
        if "klaster" in kota_df.columns:
            st.scatter_chart(kota_df, x="jumlah_spklu", y="rata2_daya_kw", color="klaster", size=100)

# ---------- TAB 3: TENTANG MODEL ----------
with tab3:
    st.subheader("Pemodelan Matematika di Balik SETRUM")
    st.write("Transparansi penuh: berikut semua formula yang dipakai, dirender dalam notasi matematis.")

    st.markdown("#### 1. Skor Infrastruktur Kota — Min-Max Normalization")
    st.latex(r"S_{\text{infra}}(\text{kota}) = \frac{N_{\text{kota}} - N_{\text{min}}}{N_{\text{max}} - N_{\text{min}}} \times 100")
    st.caption(f"Dimana $N_{{\\text{{kota}}}}$ adalah jumlah SPKLU, $N_{{\\text{{min}}}} = {int(N_MIN)}$, dan $N_{{\\text{{max}}}} = {int(N_MAX)}$.")

    st.markdown("#### 2. Skor Kualitas Charging — Rata-Rata Tertimbang")
    st.latex(r"S_{\text{kualitas}}(\text{kota}) = \left( \frac{\sum_{i=1}^{n} (w_i \times n_i)}{\sum_{i=1}^{n} n_i} \right) \times \left( \frac{100}{w_{\text{max}}} \right)")
    st.caption("Bobot kategori ($w_i$): standard=1, medium=2, fast=4, ultrafast=8.")

    st.markdown("#### 3. Skor Kesiapan Gabungan")
    st.latex(r"S_{\text{kesiapan}} = \alpha \cdot S_{\text{infra}} + (1-\alpha) \cdot S_{\text{kualitas}}")
    st.caption(f"Asumsi bobot $\\alpha = {ALPHA}$.")

    st.markdown("#### 4. Skor Kesesuaian Budget — Fungsi Peluruhan Eksponensial")
    st.latex(r"S_{\text{budget}} = \begin{cases} 100, & \text{jika } h \le b \\ 100 \times e^{-k \cdot \left(\frac{h-b}{b}\right)}, & \text{jika } h > b \end{cases}")
    st.caption(f"Dimana $h$ adalah harga EV, $b$ adalah budget, dan tingkat peluruhan $k = {K_PELURUHAN}$.")

    st.markdown("#### 5. Skor Akhir Rekomendasi")
    st.latex(r"S_{\text{akhir}} = \beta \cdot S_{\text{kesiapan}} + (1-\beta) \cdot S_{\text{budget}}")
    st.caption(f"Dengan asumsi kesimbangan preferensi, $\\beta = {BETA}$.")

    st.markdown("#### 6. Analisis Klaster — Gaussian Mixture Model (GMM)")
    st.write("Vektor variabel input distandarisasi menggunakan *Z-score* sebelum dikelompokkan:")
    st.latex(r"\mathbf{X} = \left[ N_{\text{SPKLU}}, \bar{P}_{\text{kW}}, N_{\text{ultrafast}} \right]^T")

# =========================================================
# FOOTER
# =========================================================
st.markdown(f"""
<div class="footer-note">
    SETRUM — Prototipe untuk Statistics Infographic Competition, SATRIA DATA 2026.<br>
    Data diolah secara dinamis dari file Excel yang diunggah. Terdapat <b>{len(spklu_semua):,}</b> baris data SPKLU yang ditarik secara otomatis.
</div>
""".replace(",", "."), unsafe_allow_html=True)
