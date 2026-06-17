"""
SETRUM — Sistem Evaluasi Trips & Rute untuk Mobilitas EV
Aplikasi web Streamlit — prototipe untuk SIC SATRIA DATA 2026

Cara jalankan lokal: streamlit run app.py
Cara deploy: push ke GitHub, lalu hubungkan di share.streamlit.io
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

# Palet warna sesuai dokumen rencana SETRUM
GREEN = "#1B7A43"
GREEN_LIGHT = "#E8F5EC"
AMBER = "#E0A526"
AMBER_LIGHT = "#FFF6DA"
RED = "#C1502E"
RED_LIGHT = "#FBE7E2"
BLUE = "#1F5C8B"
BLUE_LIGHT = "#E3EEF5"
GRAY = "#5A5A5A"

CUSTOM_CSS = f"""
<style>
    .main {{
        background-color: #FAFBFA;
    }}
    h1, h2, h3 {{
        color: {GREEN};
        font-family: 'Segoe UI', sans-serif;
    }}
    .hero-box {{
        background: linear-gradient(135deg, {GREEN} 0%, #2E9E5E 100%);
        padding: 28px 32px;
        border-radius: 14px;
        color: white;
        margin-bottom: 24px;
    }}
    .hero-box h1 {{
        color: white;
        margin-bottom: 4px;
    }}
    .hero-box p {{
        color: #E8F5EC;
        font-size: 17px;
        margin-bottom: 0;
    }}
    .stat-card {{
        background-color: white;
        border: 1px solid #E0E0E0;
        border-radius: 12px;
        padding: 18px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }}
    .stat-card .big {{
        font-size: 28px;
        font-weight: 700;
        color: {GREEN};
    }}
    .stat-card .label {{
        font-size: 13px;
        color: {GRAY};
    }}
    .result-box {{
        border-radius: 12px;
        padding: 20px 24px;
        margin-top: 12px;
    }}
    .badge {{
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 14px;
    }}
    div[data-testid="stMetricValue"] {{
        color: {GREEN};
    }}
    .footer-note {{
        font-size: 12px;
        color: #999999;
        margin-top: 40px;
        border-top: 1px solid #EEEEEE;
        padding-top: 12px;
    }}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# =========================================================
# DATA (GANTI DENGAN DATA RIIL SEBELUM RILIS FINAL)
# =========================================================
# Catatan: seluruh data di bawah ini bersifat sementara untuk keperluan
# prototipe. Sebelum dipakai untuk submission final, ganti dengan data
# yang sudah diverifikasi (lihat dokumen rencana, bagian Lampiran & bagian 4).

@st.cache_data
def load_data_ev():
    return pd.DataFrame([
        {"model": "Wuling Air EV Lite", "kapasitas_kwh": 18.0, "jarak_klaim_km": 200, "harga_juta": 190, "jenis": "Mobil"},
        {"model": "Wuling Air EV Long Range", "kapasitas_kwh": 26.7, "jarak_klaim_km": 300, "harga_juta": 250, "jenis": "Mobil"},
        {"model": "BYD Dolphin", "kapasitas_kwh": 44.9, "jarak_klaim_km": 410, "harga_juta": 425, "jenis": "Mobil"},
        {"model": "BYD Atto 3", "kapasitas_kwh": 60.4, "jarak_klaim_km": 420, "harga_juta": 480, "jenis": "Mobil"},
        {"model": "Hyundai Ioniq 5 SR", "kapasitas_kwh": 58.0, "jarak_klaim_km": 384, "harga_juta": 730, "jenis": "Mobil"},
        {"model": "Hyundai Ioniq 5 LR", "kapasitas_kwh": 72.6, "jarak_klaim_km": 451, "harga_juta": 820, "jenis": "Mobil"},
        {"model": "Wuling Cloud EV", "kapasitas_kwh": 50.6, "jarak_klaim_km": 460, "harga_juta": 298, "jenis": "Mobil"},
    ])

FAKTOR_KOREKSI_TROPIS = 0.85  # asumsi sementara — validasi dengan literatur/data riil

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

WARNA_KLASTER = {"Siap": GREEN, "Bersyarat": AMBER, "Belum Layak": RED}
BG_KLASTER = {"Siap": GREEN_LIGHT, "Bersyarat": AMBER_LIGHT, "Belum Layak": RED_LIGHT}

# =========================================================
# FUNGSI ALGORITMA (sama seperti versi notebook)
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
        })
    return pd.DataFrame(rekomendasi).sort_values("Skor Total", ascending=False).reset_index(drop=True)

def buat_peta_trip(hasil_trip, df_spklu):
    titik_terpilih_nama = hasil_trip["titik_charging"]
    peta = folium.Map(location=[-6.6, 107.2], zoom_start=9, tiles="cartodbpositron")
    folium.Marker([-6.200, 106.816], popup="Asal", icon=folium.Icon(color="blue", icon="play")).add_to(peta)
    folium.Marker([-6.917, 107.619], popup="Tujuan", icon=folium.Icon(color="red", icon="flag")).add_to(peta)
    for _, row in df_spklu.iterrows():
        dipakai = row["nama"] in titik_terpilih_nama
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=10 if dipakai else 6,
            popup=f"{row['nama']} ({row['daya_kw']} kW)",
            color=GREEN if dipakai else "#999999",
            fill=True, fill_color=GREEN if dipakai else "#cccccc", fill_opacity=0.85,
        ).add_to(peta)
    return peta

# =========================================================
# HEADER / HERO
# =========================================================

st.markdown(f"""
<div class="hero-box">
    <h1>🔌 SETRUM</h1>
    <p>Sistem Evaluasi Trips & Rute untuk Mobilitas EV — bantu kamu putuskan, sebelum beli.</p>
</div>
""", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown('<div class="stat-card"><div class="big">65%</div><div class="label">konsumen khawatir kehabisan baterai</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="stat-card"><div class="big">1 : 75</div><div class="label">rasio SPKLU terhadap populasi EV</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="stat-card"><div class="big">358K+</div><div class="label">unit EV terdaftar di Indonesia</div></div>', unsafe_allow_html=True)
with col4:
    st.markdown('<div class="stat-card"><div class="big">4.778</div><div class="label">SPKLU aktif per 2025</div></div>', unsafe_allow_html=True)

st.caption("Sumber: Populix Electric Vehicle Dynamic 2024, PLN, CNN Indonesia 2026. Data ringkasan kontekstual, bukan bagian dari hasil analisis prototipe ini.")

st.write("")

# =========================================================
# TAB UTAMA
# =========================================================

tab1, tab2, tab3 = st.tabs(["🗺️ Trip Planner", "✅ Feasibility Check", "📊 Peta Kesiapan Kota"])

# ---------- TAB 1: TRIP PLANNER ----------
with tab1:
    st.subheader("Rencanakan perjalanan EV-mu")
    st.write("Simulasi rute contoh: **Jakarta → Bandung** (±150 km). Pilih model EV-mu untuk melihat titik isi ulang daya yang disarankan.")

    colA, colB = st.columns([1, 2])
    with colA:
        model_pilih = st.selectbox("Model EV", data_ev["model"].tolist())
        jarak_total = st.slider("Jarak total perjalanan (km)", 20, 500, 150, step=10)
        hitung = st.button("🔍 Hitung Rute", type="primary", use_container_width=True)

    if hitung or "trip_result" in st.session_state:
        hasil = trip_planner(model_pilih, jarak_total, spklu_dummy, data_ev)
        st.session_state["trip_result"] = hasil

        with colB:
            m1, m2, m3 = st.columns(3)
            m1.metric("Jangkauan riil", f"{hasil['jangkauan_riil_km']} km", help="Sudah dikoreksi untuk kondisi tropis Indonesia")
            m2.metric("Jumlah charging", f"{hasil['jumlah_charging']}x")
            m3.metric("Estimasi total waktu", f"{hasil['waktu_total_jam']} jam")

            if hasil["titik_charging"]:
                st.info("📍 Titik isi ulang yang disarankan: " + ", ".join(hasil["titik_charging"]))
            else:
                st.success("✅ Tidak perlu charging di tengah jalan — baterai cukup untuk jarak ini.")

        st.write("")
        peta = buat_peta_trip(hasil, spklu_dummy)
        st_folium(peta, width=None, height=420)

    st.caption("⚠️ Data SPKLU & koordinat di atas adalah data contoh (dummy) untuk keperluan prototipe. Ganti dengan data riil PLN/ESDM sebelum dipakai untuk keputusan nyata.")

# ---------- TAB 2: FEASIBILITY CHECK ----------
with tab2:
    st.subheader("Apakah EV cocok untuk rutinitasmu?")
    st.write("Masukkan kondisi harianmu — kami cocokkan dengan data infrastruktur kotamu dan rekomendasikan model EV yang paling sesuai.")

    colA, colB = st.columns([1, 2])
    with colA:
        kota_pilih = st.selectbox("Kota domisili", kota_dummy["kota"].tolist())
        jarak_harian = st.slider("Jarak tempuh harian (km)", 5, 100, 20, step=5)
        anggaran = st.slider("Anggaran (juta Rupiah)", 150, 900, 300, step=50)
        cek = st.button("✅ Cek Kelayakan", type="primary", use_container_width=True)

    if cek or "feas_result" in st.session_state:
        df_hasil = ev_feasibility_check(kota_pilih, jarak_harian, anggaran, data_ev, kota_dummy)
        st.session_state["feas_result"] = df_hasil
        skor_top = df_hasil.iloc[0]["Skor Total"]

        if skor_top >= 75:
            label, warna, bg = "Sangat Layak", GREEN, GREEN_LIGHT
        elif skor_top >= 50:
            label, warna, bg = "Layak dengan Catatan", AMBER, AMBER_LIGHT
        else:
            label, warna, bg = "Belum Layak", RED, RED_LIGHT

        with colB:
            st.markdown(f"""
            <div class="result-box" style="background-color:{bg}; border: 1px solid {warna};">
                <span class="badge" style="background-color:{warna}; color:white;">{label}</span>
                <div style="font-size:36px; font-weight:700; color:{warna}; margin-top:8px;">{skor_top}/100</div>
                <div style="color:{GRAY};">Skor kelayakan tertinggi berdasarkan kondisimu</div>
            </div>
            """, unsafe_allow_html=True)

        st.write("")
        st.write("**Rekomendasi model EV (diurutkan dari paling sesuai):**")
        st.dataframe(
            df_hasil.style.background_gradient(subset=["Skor Total"], cmap="Greens"),
            use_container_width=True, hide_index=True
        )

    st.caption("⚠️ Bobot skor (infrastruktur/jarak/biaya) saat ini bernilai sementara. Nilai final akan ditentukan dari hasil analisis klaster pada data riil 401 responden.")

# ---------- TAB 3: PETA KESIAPAN KOTA ----------
with tab3:
    st.subheader("Hasil Analisis Klaster — Kesiapan EV per Kota")
    st.write("Pengelompokan kota berdasarkan kepadatan SPKLU, pola mobilitas, daya beli, dan persepsi infrastruktur/harga.")
    st.warning("📌 Catatan metodologis: hasil di bawah memakai **data contoh 7 kota** untuk mendemonstrasikan alur program. Untuk klaim ilmiah di poster/laporan akhir, hasil analisis sebenarnya menggunakan data tingkat responden (n=401) sesuai rujukan dosen pembimbing — bukan 7 observasi kota ini.")

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
    SETRUM — Prototipe untuk Statistics Infographic Competition, SATRIA DATA 2026.<br>
    Seluruh data dalam aplikasi ini bersifat ilustratif untuk keperluan demonstrasi alur program,
    kecuali disebutkan sebagai data final. Lihat dokumen rencana untuk daftar lengkap data yang perlu diverifikasi.
</div>
""", unsafe_allow_html=True)
