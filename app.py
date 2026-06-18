"""
SETRUM — Sistem Evaluasi Trips & Rute untuk Mobilitas EV
Aplikasi web Streamlit — prototipe untuk SIC SATRIA DATA 2026

Tema Visual: "Tech Blue & Slate" (Biru, Abu, Putih)
"""

import math
import pandas as pd
import numpy as np
import streamlit as st
import folium
from streamlit_folium import st_folium
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

# =========================================================
# KONFIGURASI HALAMAN & STYLE
# =========================================================

st.set_page_config(
    page_title="SETRUM — Kalkulator Kelayakan EV",
    page_icon="🔌",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Palet warna bernuansa Tech Blue & Slate (Biru, Abu, Putih)
BG_MAIN = "#F8FAFC"         # Background Utama (Slate 50)
TEXT_MAIN = "#0F172A"       # Teks Utama & Tombol (Slate 900)
BLUE_ACCENT = "#2563EB"     # Aksen & Header (Blue 600)
TEXT_MUTED = "#64748B"      # Teks Sekunder (Slate 500)
BG_CARD = "#FFFFFF"         # Background Kotak (White)

# Injeksi Custom CSS untuk "memaksa" Streamlit berubah wujud
CUSTOM_CSS = f"""
<style>
    /* Menyembunyikan elemen bawaan Streamlit */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}

    /* Latar Belakang Utama Aplikasi */
    .stApp {{
        background-color: {BG_MAIN};
    }}
    
    /* Mengubah warna font dasar */
    html, body, [class*="css"]  {{
        font-family: 'Nunito', 'Segoe UI', sans-serif;
        color: {TEXT_MAIN};
    }}

    h1, h2, h3, h4, h5, h6 {{
        color: {TEXT_MAIN} !important;
        font-weight: 800 !important;
    }}

    /* Kustomisasi Tampilan TABS (Biar mirip navigasi Pill) */
    .stTabs [data-baseweb="tab-list"] {{
        background-color: {BG_CARD};
        border-radius: 50px;
        padding: 5px;
        border: 1px solid #E2E8F0;
        gap: 5px;
    }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 50px;
        padding: 10px 20px;
        color: {TEXT_MUTED};
        font-weight: bold;
    }}
    .stTabs [aria-selected="true"] {{
        background-color: {BLUE_ACCENT} !important;
        color: {BG_CARD} !important;
    }}

    /* Kustomisasi Tombol Primary */
    .stButton>button {{
        background-color: {BLUE_ACCENT};
        color: {BG_CARD};
        border-radius: 20px;
        border: none;
        font-weight: 900;
        padding: 10px 24px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(37,99,235,0.2);
    }}
    .stButton>button:hover {{
        background-color: #1D4ED8;
        color: {BG_CARD};
        transform: translateY(-2px);
    }}

    /* Elemen Header / Hero (Biru Teknologi) */
    .hero-box {{
        background-color: {BLUE_ACCENT};
        padding: 40px 48px;
        border-radius: 40px;
        margin-bottom: 30px;
        box-shadow: 0 8px 20px rgba(37,99,235,0.3);
        position: relative;
        overflow: hidden;
    }}
    .hero-title {{
        color: {BG_CARD};
        font-size: 3rem;
        font-weight: 900;
        margin-bottom: 10px;
        line-height: 1.1;
    }}
    .hero-subtitle {{
        color: #DBEAFE;
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 24px;
    }}

    /* Kartu Statistik di Header */
    .stat-card {{
        background-color: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(5px);
        border-radius: 24px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 10px rgba(0,0,0,0.05);
    }}
    .stat-card .big {{
        font-size: 32px;
        font-weight: 900;
        color: {BLUE_ACCENT};
        line-height: 1;
        margin-bottom: 5px;
    }}
    .stat-card .label {{
        font-size: 13px;
        color: {TEXT_MUTED};
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}

    /* Kotak Hasil Rekomendasi (Highlight) */
    .result-box {{
        background-color: #EFF6FF;
        border: 3px solid {BLUE_ACCENT};
        border-radius: 32px;
        padding: 30px;
        margin-top: 10px;
        box-shadow: 0 6px 15px rgba(37,99,235,0.2);
    }}
    
    .badge {{
        display: inline-block;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 800;
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }}

    div[data-testid="stMetricValue"] {{
        color: {BLUE_ACCENT};
        font-weight: 900;
    }}
    div[data-testid="stMetricLabel"] {{
        color: {TEXT_MUTED};
        font-weight: bold;
    }}

    .footer-note {{
        font-size: 13px;
        color: {TEXT_MUTED};
        font-weight: bold;
        margin-top: 50px;
        border-top: 2px dashed #CBD5E1;
        padding-top: 20px;
        text-align: center;
    }}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# =========================================================
# DATA EV (25 MOBIL TERBARU)
# =========================================================

@st.cache_data
def load_data_ev():
    return pd.DataFrame([
        {"model": "Wuling Air EV Lite", "kapasitas_kwh": 17.3, "jarak_klaim_km": 200, "harga_juta": 190.0, "max_dc_kw": 0},
        {"model": "Wuling Air EV Standard Range", "kapasitas_kwh": 17.3, "jarak_klaim_km": 200, "harga_juta": 224.0, "max_dc_kw": 0},
        {"model": "Wuling Air EV Long Range", "kapasitas_kwh": 26.7, "jarak_klaim_km": 300, "harga_juta": 275.0, "max_dc_kw": 0},
        {"model": "Wuling BinguoEV Long Range", "kapasitas_kwh": 31.9, "jarak_klaim_km": 333, "harga_juta": 317.0, "max_dc_kw": 50},
        {"model": "Wuling BinguoEV Premium Range", "kapasitas_kwh": 37.9, "jarak_klaim_km": 410, "harga_juta": 372.0, "max_dc_kw": 50},
        {"model": "Wuling Cloud EV", "kapasitas_kwh": 50.6, "jarak_klaim_km": 460, "harga_juta": 398.0, "max_dc_kw": 50},
        {"model": "BYD Dolphin Dynamic", "kapasitas_kwh": 44.9, "jarak_klaim_km": 410, "harga_juta": 365.0, "max_dc_kw": 60},
        {"model": "BYD Dolphin Premium", "kapasitas_kwh": 60.48, "jarak_klaim_km": 490, "harga_juta": 425.0, "max_dc_kw": 80},
        {"model": "BYD Atto 3 Advanced", "kapasitas_kwh": 49.92, "jarak_klaim_km": 410, "harga_juta": 465.0, "max_dc_kw": 70},
        {"model": "BYD Atto 3 Superior", "kapasitas_kwh": 60.48, "jarak_klaim_km": 480, "harga_juta": 515.0, "max_dc_kw": 80},
        {"model": "BYD Seal Premium", "kapasitas_kwh": 82.56, "jarak_klaim_km": 650, "harga_juta": 629.0, "max_dc_kw": 150},
        {"model": "BYD Seal Performance AWD", "kapasitas_kwh": 82.56, "jarak_klaim_km": 580, "harga_juta": 719.0, "max_dc_kw": 150},
        {"model": "Hyundai Ioniq 5 Standard Range", "kapasitas_kwh": 58.0, "jarak_klaim_km": 384, "harga_juta": 782.0, "max_dc_kw": 350},
        {"model": "Hyundai Ioniq 5 Long Range", "kapasitas_kwh": 72.6, "jarak_klaim_km": 451, "harga_juta": 859.0, "max_dc_kw": 350},
        {"model": "Hyundai Ioniq 6 Signature", "kapasitas_kwh": 77.4, "jarak_klaim_km": 614, "harga_juta": 1220.0, "max_dc_kw": 350},
        {"model": "Hyundai All-New Kona Electric Standard Range", "kapasitas_kwh": 48.4, "jarak_klaim_km": 448, "harga_juta": 499.0, "max_dc_kw": 100},
        {"model": "Hyundai All-New Kona Electric Long Range", "kapasitas_kwh": 65.4, "jarak_klaim_km": 549, "harga_juta": 590.0, "max_dc_kw": 100},
        {"model": "Chery Omoda E5", "kapasitas_kwh": 61.0, "jarak_klaim_km": 430, "harga_juta": 488.8, "max_dc_kw": 80},
        {"model": "MG 4 EV Ignite", "kapasitas_kwh": 51.0, "jarak_klaim_km": 425, "harga_juta": 395.0, "max_dc_kw": 140},
        {"model": "MG 4 EV Magnify", "kapasitas_kwh": 51.0, "jarak_klaim_km": 425, "harga_juta": 423.0, "max_dc_kw": 140},
        {"model": "MG ZS EV", "kapasitas_kwh": 50.3, "jarak_klaim_km": 403, "harga_juta": 453.0, "max_dc_kw": 80},
        {"model": "Neta V", "kapasitas_kwh": 40.7, "jarak_klaim_km": 401, "harga_juta": 317.0, "max_dc_kw": 50},
        {"model": "Neta V-II", "kapasitas_kwh": 36.1, "jarak_klaim_km": 401, "harga_juta": 299.0, "max_dc_kw": 50},
        {"model": "Kia EV6 GT-Line", "kapasitas_kwh": 77.4, "jarak_klaim_km": 506, "harga_juta": 1299.0, "max_dc_kw": 350},
        {"model": "Toyota bZ4X", "kapasitas_kwh": 71.4, "jarak_klaim_km": 500, "harga_juta": 1190.0, "max_dc_kw": 150}
    ])

# Koreksi 25% pengurangan jarak untuk AC dan Stop-and-Go di Tropis
FAKTOR_KOREKSI_TROPIS = 0.75  

@st.cache_data
def load_data_kota():
    df = pd.DataFrame({
        "kota": ["Kota Tier-1 A", "Kota Tier-1 B", "Kota Tier-2 A", "Kota Tier-2 B",
                 "Kota Tier-2 C", "Kota Sub-Urban A", "Kota Sub-Urban B"],
        "jumlah_spklu": [120, 95, 35, 28, 22, 8, 5],
        "luas_km2": [661, 350, 167, 305, 199, 450, 380],
        "jarak_tempuh_harian_km": [18, 15, 22, 25, 20, 35, 40],
        "pdrb_per_kapita_juta": [285, 240, 95, 80, 70, 55, 48],
        "skor_persepsi_infrastruktur": [4.2, 3.9, 3.1, 2.8, 2.6, 2.0, 1.8],
        "skor_persepsi_harga": [3.5, 3.6, 3.0, 3.2, 2.9, 2.5, 2.3],
    })
    df["spklu_per_100km2"] = (df["jumlah_spklu"] / df["luas_km2"] * 100).round(2)
    return df

@st.cache_data
def load_data_spklu():
    return pd.DataFrame([
        {"nama": "SPKLU Rest Area KM19", "lat": -6.337, "lon": 106.866, "jarak_dari_asal_km": 19, "daya_kw": 50},
        {"nama": "SPKLU Rest Area KM57", "lat": -6.421, "lon": 107.180, "jarak_dari_asal_km": 57, "daya_kw": 60},
        {"nama": "SPKLU Rest Area KM88", "lat": -6.598, "lon": 107.450, "jarak_dari_asal_km": 88, "daya_kw": 50},
        {"nama": "SPKLU Rest Area KM125", "lat": -6.821, "lon": 107.580, "jarak_dari_asal_km": 125, "daya_kw": 100},
        {"nama": "SPKLU Pusat Kota Tujuan", "lat": -6.917, "lon": 107.619, "jarak_dari_asal_km": 150, "daya_kw": 60},
    ])

data_ev = load_data_ev()
data_ev["jarak_riil_km"] = (data_ev["jarak_klaim_km"] * FAKTOR_KOREKSI_TROPIS).round(1)
kota_dummy = load_data_kota()
spklu_dummy = load_data_spklu()

# Hasil clustering (di-cache supaya tidak hitung ulang tiap interaksi)
@st.cache_data
def jalankan_clustering(df_kota):
    fitur = ["spklu_per_100km2", "jarak_tempuh_harian_km", "pdrb_per_kapita_juta",
             "skor_persepsi_infrastruktur", "skor_persepsi_harga"]
    X = df_kota[fitur].values
    X_scaled = StandardScaler().fit_transform(X)
    gmm = GaussianMixture(n_components=3, random_state=42)
    labels = gmm.fit_predict(X_scaled)
    df = df_kota.copy()
    df["klaster_raw"] = labels
    urutan = df.groupby("klaster_raw")["skor_persepsi_infrastruktur"].mean().sort_values().index
    mapping = {urutan[0]: "Belum Layak", urutan[1]: "Bersyarat", urutan[2]: "Siap"}
    df["klaster"] = df["klaster_raw"].map(mapping)
    return df

kota_dummy = jalankan_clustering(kota_dummy)

# Pemetaan warna Biru-Abu untuk output visual klaster
WARNA_KLASTER = {"Siap": BLUE_ACCENT, "Bersyarat": TEXT_MUTED, "Belum Layak": "#94A3B8"}
BG_KLASTER = {"Siap": BG_CARD, "Bersyarat": "#F8FAFC", "Belum Layak": "#E2E8F0"}

# =========================================================
# FUNGSI ALGORITMA 
# =========================================================

MARGIN_AMAN = 0.8
KECEPATAN_RATA2_KMH = 60
WAKTU_CHARGING_MENIT = {50: 45, 60: 35, 100: 25}

def trip_planner(model_ev, jarak_total_km, df_spklu, df_ev):
    spek = df_ev[df_ev["model"] == model_ev].iloc[0]
    jangkauan_riil = spek["jarak_riil_km"]
    jangkauan_aman = jangkauan_riil * MARGIN_AMAN
    jumlah_charging_dibutuhkan = max(0, math.ceil(jarak_total_km / jangkauan_aman) - 1)

    titik_charging = []
    jarak_tertempuh = 0
    df_sorted = df_spklu.sort_values("jarak_dari_asal_km")

    for _ in range(jumlah_charging_dibutuhkan):
        target = jarak_tertempuh + jangkauan_aman
        kandidat = df_sorted[(df_sorted["jarak_dari_asal_km"] <= target) & (df_sorted["jarak_dari_asal_km"] > jarak_tertempuh)]
        if kandidat.empty:
            break
        titik = kandidat.iloc[-1]
        titik_charging.append(titik)
        jarak_tertempuh = titik["jarak_dari_asal_km"]

    waktu_charging_total = sum(WAKTU_CHARGING_MENIT.get(t["daya_kw"], 40) for t in titik_charging)
    waktu_jalan_menit = (jarak_total_km / KECEPATAN_RATA2_KMH) * 60
    waktu_total_menit = waktu_jalan_menit + waktu_charging_total

    return {
        "model_ev": model_ev,
        "jangkauan_riil_km": round(jangkauan_riil, 1),
        "jumlah_charging": len(titik_charging),
        "titik_charging": [t["nama"] for t in titik_charging],
        "waktu_jalan_jam": round(waktu_jalan_menit / 60, 1),
        "waktu_charging_menit": waktu_charging_total,
        "waktu_total_jam": round(waktu_total_menit / 60, 1),
    }

W_INFRASTRUKTUR, W_JARAK, W_BIAYA = 0.4, 0.35, 0.25

def skor_infrastruktur_kota(nama_kota, df_kota):
    row = df_kota[df_kota["kota"] == nama_kota]
    if row.empty:
        return 50
    nilai = row.iloc[0]["spklu_per_100km2"]
    maksimum = df_kota["spklu_per_100km2"].max()
    return min(100, (nilai / maksimum) * 100)

def skor_kesesuaian_jarak(jarak_harian_km, model_ev, df_ev):
    spek = df_ev[df_ev["model"] == model_ev].iloc[0]
    rasio = spek["jarak_riil_km"] / max(jarak_harian_km, 1)
    if rasio >= 5: return 100
    if rasio >= 3: return 85
    if rasio >= 1.5: return 65
    if rasio >= 1: return 40
    return 15

def skor_keterjangkauan_biaya(anggaran_juta, model_ev, df_ev):
    spek = df_ev[df_ev["model"] == model_ev].iloc[0]
    if spek["harga_juta"] <= anggaran_juta:
        return 100
    selisih_persen = (spek["harga_juta"] - anggaran_juta) / anggaran_juta * 100
    return max(0, 100 - selisih_persen * 2)

def ev_feasibility_check(nama_kota, jarak_harian_km, anggaran_juta, df_ev, df_kota):
    skor_infra = skor_infrastruktur_kota(nama_kota, df_kota)
    rekomendasi = []
    for _, ev in df_ev.iterrows():
        s_jarak = skor_kesesuaian_jarak(jarak_harian_km, ev["model"], df_ev)
        s_biaya = skor_keterjangkauan_biaya(anggaran_juta, ev["model"], df_ev)
        skor_total = (W_INFRASTRUKTUR * skor_infra) + (W_JARAK * s_jarak) + (W_BIAYA * s_biaya)
        rekomendasi.append({
            "Model EV": ev["model"],
            "Skor Total": round(skor_total, 1),
            "Skor Infrastruktur": round(skor_infra, 1),
            "Skor Kesesuaian Jarak": round(s_jarak, 1),
            "Skor Keterjangkauan Biaya": round(s_biaya, 1),
            "Harga (juta Rp)": ev["harga_juta"],
            "Jangkauan Riil (km)": ev["jarak_riil_km"],
            "Max DC (kW)": ev["max_dc_kw"]
        })
    return pd.DataFrame(rekomendasi).sort_values("Skor Total", ascending=False).reset_index(drop=True)

def buat_peta_trip(hasil_trip, df_spklu):
    titik_terpilih_nama = hasil_trip["titik_charging"]
    peta = folium.Map(location=[-6.6, 107.2], zoom_start=9, tiles="cartodbpositron")
    
    # Warna Pin Map disesuaikan dengan tema
    folium.Marker([-6.200, 106.816], popup="Asal", icon=folium.Icon(color="blue", icon="play")).add_to(peta)
    folium.Marker([-6.917, 107.619], popup="Tujuan", icon=folium.Icon(color="lightgray", icon="flag")).add_to(peta)
    
    for _, row in df_spklu.iterrows():
        dipakai = row["nama"] in titik_terpilih_nama
        # Ubah warna pin: Biru jika dipakai, Abu-abu jika dilewati
        pin_color = BLUE_ACCENT if dipakai else TEXT_MUTED
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=10 if dipakai else 6,
            popup=f"{row['nama']} ({row['daya_kw']} kW)",
            color=BLUE_ACCENT if dipakai else TEXT_MUTED,
            fill=True, fill_color=pin_color, fill_opacity=0.9,
        ).add_to(peta)
    return peta

# =========================================================
# HEADER / HERO SECTION (Custom HTML untuk warna Biru)
# =========================================================

st.markdown(f"""
<div class="hero-box">
    <div class="hero-title">⚡ SETRUM</div>
    <div class="hero-subtitle">Sistem Evaluasi Trips & Rute Mobilitas EV. Cek kecocokan mobil listrik impian dengan gaya hidupmu.</div>
    <div style="display:flex; gap: 15px; flex-wrap: wrap;">
        <div class="stat-card" style="flex:1;">
            <div class="big">65%</div>
            <div class="label">Khawatir Baterai</div>
        </div>
        <div class="stat-card" style="flex:1;">
            <div class="big">1:75</div>
            <div class="label">Rasio SPKLU/EV</div>
        </div>
        <div class="stat-card" style="flex:1;">
            <div class="big">358K+</div>
            <div class="label">Unit Terdaftar</div>
        </div>
        <div class="stat-card" style="flex:1;">
            <div class="big">4.892</div>
            <div class="label">SPKLU Aktif</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# =========================================================
# TAB UTAMA
# =========================================================

tab1, tab2, tab3 = st.tabs(["✅ Cek Kelayakan", "🗺️ Perencana Rute", "📊 Peta Kesiapan Kota"])

# ---------- TAB 1: FEASIBILITY CHECK ----------
with tab1:
    st.markdown(f"<h3 style='color: {TEXT_MAIN};'>Apakah EV cocok untukmu?</h3>", unsafe_allow_html=True)
    st.write("Sesuaikan profil mobilitas dan anggaranmu. Kami akan mencocokkannya dengan infrastruktur kota.")
    st.write("---")

    colA, colB = st.columns([1, 2])
    with colA:
        kota_pilih = st.selectbox("Kota Domisili", kota_dummy["kota"].tolist())
        jarak_harian = st.slider("Jarak tempuh harian (km)", 5, 150, 40, step=5)
        anggaran = st.slider("Anggaran (juta Rupiah)", 150, 1500, 450, step=50)
        cek = st.button("⚡ Analisis Sekarang", type="primary", use_container_width=True)

    if cek or "feas_result" in st.session_state:
        df_hasil = ev_feasibility_check(kota_pilih, jarak_harian, anggaran, data_ev, kota_dummy)
        st.session_state["feas_result"] = df_hasil
        skor_top = df_hasil.iloc[0]["Skor Total"]

        if skor_top >= 75:
            label, bg_warna, text_warna = "Sangat Direkomendasikan", BLUE_ACCENT, BG_CARD
        elif skor_top >= 50:
            label, bg_warna, text_warna = "Alternatif Layak", BG_CARD, TEXT_MAIN
        else:
            label, bg_warna, text_warna = "Belum Layak", "#E2E8F0", TEXT_MUTED

        with colB:
            st.markdown(f"""
            <div class="result-box">
                <span class="badge" style="background-color: {BLUE_ACCENT}; color: {BG_CARD};">{label}</span>
                <h2 style="color: {TEXT_MAIN}; margin-top: 15px; margin-bottom: 5px; font-size: 2.2rem;">{df_hasil.iloc[0]['Model EV']}</h2>
                <div style="display:flex; gap: 10px; margin-bottom: 15px;">
                    <span style="background-color: white; padding: 5px 12px; border-radius: 10px; border: 1px solid #E2E8F0; font-weight: bold; color: {TEXT_MUTED};">🚗 Rp {df_hasil.iloc[0]['Harga (juta Rp)']} Jt</span>
                    <span style="background-color: white; padding: 5px 12px; border-radius: 10px; border: 1px solid #E2E8F0; font-weight: bold; color: {TEXT_MUTED};">📍 {df_hasil.iloc[0]['Jangkauan Riil (km)']} km riil</span>
                </div>
                <div style="font-size:3.5rem; font-weight:900; color:{BLUE_ACCENT}; line-height: 1;">{skor_top} <span style="font-size: 1rem; color: {TEXT_MUTED};">/100 SKOR KELAYAKAN</span></div>
            </div>
            """, unsafe_allow_html=True)

        st.write("")
        st.markdown(f"<h4 style='color: {TEXT_MAIN};'>Alternatif EV Lainnya</h4>", unsafe_allow_html=True)
        # Menampilkan tabel alternatif
        st.dataframe(
            df_hasil.style.background_gradient(subset=["Skor Total"], cmap="Blues"),
            use_container_width=True, hide_index=True
        )


# ---------- TAB 2: TRIP PLANNER ----------
with tab2:
    st.markdown(f"<h3 style='color: {TEXT_MAIN};'>Perencana Rute EV</h3>", unsafe_allow_html=True)
    st.write("Pilih mobilmu dan tentukan jarak rute. Algoritma kami memperhitungkan koreksi AC 25% dan margin keamanan 20%.")
    st.write("---")

    colA, colB = st.columns([1, 2])
    with colA:
        model_pilih = st.selectbox("Pilih EV", data_ev["model"].tolist())
        jarak_total = st.slider("Jarak total rute (Contoh Jkt-Bdg: 150 km)", 20, 800, 150, step=10)
        hitung = st.button("🗺️ Kalkulasi Rute", type="primary", use_container_width=True)

    if hitung or "trip_result" in st.session_state:
        hasil = trip_planner(model_pilih, jarak_total, spklu_dummy, data_ev)
        st.session_state["trip_result"] = hasil

        with colB:
            m1, m2, m3 = st.columns(3)
            m1.metric("Jarak Riil", f"{hasil['jangkauan_riil_km']} km", help="Dikoreksi -25% untuk pemakaian AC iklim tropis")
            m2.metric("Berhenti (Cas)", f"{hasil['jumlah_charging']} Kali")
            m3.metric("Est. Total Waktu", f"{hasil['waktu_total_jam']} Jam")

            if hasil["titik_charging"]:
                st.warning(f"🔋 Perlu Charging di: {', '.join(hasil['titik_charging'])}")
            else:
                st.success("✅ Rute aman! Tidak perlu charging di jalan.")

        st.write("")
        peta = buat_peta_trip(hasil, spklu_dummy)
        st_folium(peta, width=None, height=420)

    st.caption("⚠️ Data SPKLU di atas adalah ilustrasi (dummy) untuk keperluan prototipe.")


# ---------- TAB 3: PETA KESIAPAN KOTA ----------
with tab3:
    st.markdown(f"<h3 style='color: {TEXT_MAIN};'>Hasil Analisis Klaster — Kesiapan EV</h3>", unsafe_allow_html=True)
    st.write("Pengelompokan kota berdasarkan kepadatan SPKLU, pola mobilitas, daya beli, dan persepsi.")

    colA, colB = st.columns([3, 2])

    with colA:
        st.write("**Sebaran kota berdasarkan klaster:**")
        for klaster in ["Siap", "Bersyarat", "Belum Layak"]:
            subset = kota_dummy[kota_dummy["klaster"] == klaster]
            if subset.empty:
                continue
            st.markdown(f"<span class='badge' style='background-color:{WARNA_KLASTER[klaster]};color:white;'>{klaster}</span>", unsafe_allow_html=True)
            st.dataframe(
                subset[["kota", "spklu_per_100km2", "jarak_tempuh_harian_km", "pdrb_per_kapita_juta"]]
                .rename(columns={"kota": "Kota", "spklu_per_100km2": "SPKLU/100km²",
                                  "jarak_tempuh_harian_km": "Jarak Harian (km)", "pdrb_per_kapita_juta": "PDRB/kapita (juta Rp)"}),
                use_container_width=True, hide_index=True
            )

    with colB:
        st.write("**Posisi kota: mobilitas vs persepsi infrastruktur**")
        chart_data = kota_dummy[["kota", "jarak_tempuh_harian_km", "skor_persepsi_infrastruktur", "klaster"]].copy()
        
        st.scatter_chart(
            chart_data, x="jarak_tempuh_harian_km", y="skor_persepsi_infrastruktur",
            color="klaster", size=100,
        )

# =========================================================
# FOOTER
# =========================================================
st.markdown("""
<div class="footer-note">
    <strong>SETRUM</strong> — Prototipe untuk Statistics Infographic Competition, SATRIA DATA 2026.<br>
</div>
""", unsafe_allow_html=True)
