"""
SETRUM — Sistem Evaluasi Trips & Rute untuk Mobilitas EV
Aplikasi web Streamlit — prototipe untuk SIC SATRIA DATA 2026

REVISI 2: Fitur dipersempit ke yang 100% berbasis data riil/tertelusur sumbernya.
Fitur Trip Planner (koordinat estimasi) dan kalkulator baterai (spek belum terverifikasi
sumber resminya) DIHAPUS dari versi ini.

File pendamping wajib ada di folder yang sama:
- spklu_processed.csv      (3.050 lokasi SPKLU riil, sumber: data lapangan tim)
- kota_final.csv            (agregat SPKLU per kota, dihitung dari spklu_processed.csv)
- data_ev_valid.csv         (harga EV dengan sumber tertelusur: SEVA.id, AstraOtoshop, dst.)

Cara jalankan lokal: streamlit run app.py
Cara deploy: push folder ini (app.py + 3 file CSV + requirements.txt) ke GitHub,
lalu hubungkan di share.streamlit.io
"""

import numpy as np
import pandas as pd
import streamlit as st
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

GREEN = "#1B7A43"
GREEN_LIGHT = "#E8F5EC"
AMBER = "#E0A526"
AMBER_LIGHT = "#FFF6DA"
RED = "#C1502E"
RED_LIGHT = "#FBE7E2"
BLUE = "#1F5C8B"
GRAY = "#5A5A5A"

CUSTOM_CSS = f"""
<style>
    .main {{ background-color: #FAFBFA; }}
    h1, h2, h3 {{ color: {GREEN}; font-family: 'Segoe UI', sans-serif; }}
    .hero-box {{
        background: linear-gradient(135deg, {GREEN} 0%, #2E9E5E 100%);
        padding: 28px 32px; border-radius: 14px; color: white; margin-bottom: 24px;
    }}
    .hero-box h1 {{ color: white; margin-bottom: 4px; }}
    .hero-box p {{ color: #E8F5EC; font-size: 17px; margin-bottom: 0; }}
    .stat-card {{
        background-color: white; border: 1px solid #E0E0E0; border-radius: 12px;
        padding: 18px; text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }}
    .stat-card .big {{ font-size: 28px; font-weight: 700; color: {GREEN}; }}
    .stat-card .label {{ font-size: 13px; color: {GRAY}; }}
    .result-box {{ border-radius: 12px; padding: 20px 24px; margin-top: 12px; }}
    .badge {{ display: inline-block; padding: 4px 14px; border-radius: 20px; font-weight: 600; font-size: 14px; }}
    div[data-testid="stMetricValue"] {{ color: {GREEN}; }}
    .footer-note {{
        font-size: 12px; color: #999999; margin-top: 40px;
        border-top: 1px solid #EEEEEE; padding-top: 12px;
    }}
    .data-tag {{
        display: inline-block; background-color: {GREEN_LIGHT}; color: {GREEN};
        font-size: 11px; font-weight: 600; padding: 2px 10px; border-radius: 10px; margin-bottom: 8px;
    }}
    .formula-box {{
        background-color: #F7F7F9; border: 1px solid #E5E5EA; border-radius: 10px;
        padding: 14px 18px; margin: 10px 0; font-size: 14px;
    }}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# =========================================================
# DATA — SEMUA DARI FILE RIIL / SUMBER TERTELUSUR
# =========================================================

@st.cache_data
def load_spklu_semua():
    return pd.read_csv("spklu_processed.csv")

@st.cache_data
def load_data_kota():
    df = pd.read_csv("kota_final.csv")
    return df

@st.cache_data
def load_data_ev():
    return pd.read_csv("data_ev_valid.csv")

spklu_semua = load_spklu_semua()
kota_df = load_data_kota()
data_ev = load_data_ev()

# =========================================================
# PEMODELAN MATEMATIKA
# =========================================================
# Lihat penjelasan lengkap & alasan pemilihan formula di tab "Tentang Model" pada aplikasi.

# --- 1. Skor Infrastruktur (Min-Max Normalization) ---
# S_infra(kota) = (N_kota - N_min) / (N_max - N_min) * 100
N_MIN = kota_df["jumlah_spklu"].min()
N_MAX = kota_df["jumlah_spklu"].max()
kota_df["skor_infra"] = ((kota_df["jumlah_spklu"] - N_MIN) / (N_MAX - N_MIN) * 100).round(1)

# --- 2. Skor Kualitas Charging (rata-rata tertimbang kategori) ---
# Bobot proporsional terhadap kecepatan pengisian relatif (rasio kasar daya kW vs standard 7kW)
BOBOT_KATEGORI = {"standard": 1, "medium": 2, "fast": 4, "ultrafast": 8}
W_MAX = max(BOBOT_KATEGORI.values())

# --- 3. Skor Gabungan Kesiapan Infrastruktur ---
# S_kesiapan = alpha * S_infra + (1 - alpha) * S_kualitas
ALPHA = 0.6

# --- 4. Skor Kesesuaian Budget (fungsi peluruhan eksponensial) ---
# S_budget = 100                                  jika harga <= budget
# S_budget = 100 * exp(-k * (harga-budget)/budget) jika harga > budget
K_PELURUHAN = 2

def skor_budget(harga_juta, budget_juta):
    if harga_juta <= budget_juta:
        return 100.0
    selisih_relatif = (harga_juta - budget_juta) / budget_juta
    return round(100 * np.exp(-K_PELURUHAN * selisih_relatif), 1)

# --- 5. Skor Akhir Rekomendasi ---
# S_akhir = beta * S_kesiapan(kota) + (1 - beta) * S_budget(harga, budget)
BETA = 0.5

# --- 6. Analisis Klaster (GMM pada variabel kota yang 100% riil) ---
@st.cache_data
def jalankan_clustering(df_kota):
    fitur = ["jumlah_spklu", "rata2_daya_kw", "jumlah_ultrafast"]
    X = df_kota[fitur].values
    X_scaled = StandardScaler().fit_transform(X)
    gmm = GaussianMixture(n_components=3, random_state=42)
    labels = gmm.fit_predict(X_scaled)
    df = df_kota.copy()
    df["klaster_raw"] = labels
    urutan = df.groupby("klaster_raw")["jumlah_spklu"].mean().sort_values().index
    mapping = {urutan[0]: "Belum Layak", urutan[1]: "Bersyarat", urutan[2]: "Siap"}
    df["klaster"] = df["klaster_raw"].map(mapping)
    return df

kota_df = jalankan_clustering(kota_df)
WARNA_KLASTER = {"Siap": GREEN, "Bersyarat": AMBER, "Belum Layak": RED}

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
    st.markdown('<div class="stat-card"><div class="big">65%</div><div class="label">konsumen khawatir kehabisan baterai (Populix, 2024)</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div class="stat-card"><div class="big">{len(spklu_semua):,}</div><div class="label">SPKLU dalam dataset ini</div></div>'.replace(",", "."), unsafe_allow_html=True)
with col3:
    n_kota = kota_df["kota"].nunique()
    st.markdown(f'<div class="stat-card"><div class="big">{n_kota}</div><div class="label">kota teridentifikasi & dianalisis</div></div>', unsafe_allow_html=True)
with col4:
    st.markdown(f'<div class="stat-card"><div class="big">{len(data_ev)}</div><div class="label">model EV dengan harga tertelusur sumber</div></div>', unsafe_allow_html=True)

st.write("")

# =========================================================
# TAB UTAMA
# =========================================================

tab1, tab2, tab3 = st.tabs(["✅ Cek Kesiapan & Rekomendasi", "📊 Peta Kesiapan Kota", "🧮 Tentang Model"])

# ---------- TAB 1: FEASIBILITY CHECK ----------
with tab1:
    st.subheader("Apakah EV cocok untukmu di kotamu?")
    st.markdown('<span class="data-tag">📍 Skor infrastruktur dari 3.050 data SPKLU riil</span> <span class="data-tag">💰 Harga EV dengan sumber tertelusur</span>', unsafe_allow_html=True)
    st.write("Pilih kota domisili dan anggaranmu — sistem menghitung skor kesiapan infrastruktur kotamu dan mencocokkannya dengan model EV yang sesuai budget.")

    colA, colB = st.columns([1, 2])
    with colA:
        kota_pilih = st.selectbox("Kota domisili", kota_df["kota"].tolist())
        anggaran = st.slider("Anggaran (juta Rupiah)", 150, 900, 300, step=10)
        cek = st.button("✅ Cek Kelayakan", type="primary", use_container_width=True)

        row_kota = kota_df[kota_df["kota"] == kota_pilih].iloc[0]
        st.caption(f"📍 {kota_pilih}: **{int(row_kota['jumlah_spklu'])} SPKLU** riil tercatat, rata-rata daya **{row_kota['rata2_daya_kw']:.1f} kW**, **{int(row_kota['jumlah_ultrafast'])} unit ultrafast**.")

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
            label, warna, bg = "Sangat Layak", GREEN, GREEN_LIGHT
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
                <div style="font-size:14px; color:{GRAY};">Rp {harga_terbaik} juta (OTR Jakarta)</div>
            </div>
            """, unsafe_allow_html=True)

            m1, m2 = st.columns(2)
            m1.metric("Skor Infrastruktur Kota", f"{skor_infra}/100")
            m2.metric("Skor Kesiapan Gabungan", f"{skor_kesiapan}/100", help="0.6 x infrastruktur + 0.4 x kualitas charger")

        st.write("")
        st.write(f"**Rekomendasi lengkap model EV** (diurutkan dari paling sesuai — *{mobil_terbaik}* di posisi teratas):")
        st.dataframe(df_rekom_tampil, use_container_width=True, hide_index=True)

    st.caption("Bobot alpha=0.6 (infrastruktur) dan beta=0.5 (kesiapan vs budget) adalah asumsi awal tim, bukan hasil estimasi statistik dari survei. Lihat tab 'Tentang Model' untuk detail formula dan alasan pemilihan nilai ini.")

# ---------- TAB 2: PETA KESIAPAN KOTA ----------
with tab2:
    st.subheader("Hasil Analisis Klaster — Kesiapan Infrastruktur EV per Kota")
    st.markdown('<span class="data-tag">📍 Seluruh angka dihitung langsung dari 3.050 data SPKLU riil</span>', unsafe_allow_html=True)
    st.write("Pengelompokan kota berdasarkan **jumlah SPKLU**, **rata-rata daya charger**, dan **jumlah charger ultrafast** — variabel input GMM clustering, seluruhnya riil tanpa estimasi.")

    colA, colB = st.columns([3, 2])
    with colA:
        st.write("**Sebaran kota berdasarkan klaster:**")
        for klaster in ["Siap", "Bersyarat", "Belum Layak"]:
            subset = kota_df[kota_df["klaster"] == klaster]
            if subset.empty:
                continue
            st.markdown(f"<span class='badge' style='background-color:{WARNA_KLASTER[klaster]};color:white;'>{klaster}</span>", unsafe_allow_html=True)
            st.dataframe(
                subset[["kota", "jumlah_spklu", "rata2_daya_kw", "jumlah_ultrafast", "skor_infra"]]
                .rename(columns={"kota": "Kota", "jumlah_spklu": "Jumlah SPKLU", "rata2_daya_kw": "Rata2 Daya (kW)",
                                  "jumlah_ultrafast": "Jml Ultrafast", "skor_infra": "Skor Infrastruktur"}),
                use_container_width=True, hide_index=True
            )

    with colB:
        st.write("**Posisi kota: jumlah SPKLU vs rata-rata daya**")
        chart_data = kota_df[["kota", "jumlah_spklu", "rata2_daya_kw", "klaster"]].copy()
        st.scatter_chart(chart_data, x="jumlah_spklu", y="rata2_daya_kw", color="klaster", size=100)

    with st.expander("Lihat distribusi kategori charger seluruh dataset (3.050 lokasi)"):
        ringkasan_charger = spklu_semua["kategori_charger"].value_counts().reset_index()
        ringkasan_charger.columns = ["Kategori", "Jumlah"]
        st.dataframe(ringkasan_charger, hide_index=True, use_container_width=True)

    st.caption("15 kota dipilih berdasarkan jumlah SPKLU teridentifikasi terbanyak di dataset (bukan sampel acak/representatif secara statistik formal).")

# ---------- TAB 3: TENTANG MODEL ----------
with tab3:
    st.subheader("Pemodelan Matematika di Balik SETRUM")
    st.write("Transparansi penuh: berikut semua formula yang dipakai, beserta alasan pemilihannya.")

    st.markdown("#### 1. Skor Infrastruktur Kota — Min-Max Normalization")
    st.markdown(f"""
    <div class="formula-box">
    S_infra(kota) = (N_kota - N_min) / (N_max - N_min) x 100<br><br>
    N_kota = jumlah SPKLU riil di kota itu &nbsp;|&nbsp;
    N_min = {int(N_MIN)} &nbsp;|&nbsp; N_max = {int(N_MAX)} (Jakarta)
    </div>
    """, unsafe_allow_html=True)
    st.caption("Dipilih min-max (bukan z-score) karena tujuannya menunjukkan posisi kota dalam rentang realistis Indonesia (0 = paling minim, 100 = paling siap), bukan jarak dari rata-rata nasional.")

    st.markdown("#### 2. Skor Kualitas Charging — Rata-Rata Tertimbang")
    st.markdown(f"""
    <div class="formula-box">
    S_kualitas(kota) = (sum(w_i x n_i) / sum(n_i)) x (100 / w_max)<br><br>
    Bobot kategori: standard=1, medium=2, fast=4, ultrafast=8
    </div>
    """, unsafe_allow_html=True)
    st.caption("Bobot proporsional terhadap kecepatan pengisian relatif (rasio kasar daya kW terhadap charger standard 7kW). Ini asumsi yang bisa didiskusikan, bukan koefisien yang diturunkan dari studi waktu pengisian riil.")

    st.markdown("#### 3. Skor Kesiapan Gabungan")
    st.markdown(f"""
    <div class="formula-box">
    S_kesiapan = alpha x S_infra + (1-alpha) x S_kualitas, &nbsp; alpha = {ALPHA}
    </div>
    """, unsafe_allow_html=True)
    st.caption("alpha=0.6 berarti kuantitas SPKLU sedikit lebih diutamakan dari kualitas, asumsi tim bahwa range anxiety paling terkait 'ada/tidaknya' charger, baru soal kecepatan. Nilai ini bisa diuji ulang jika ada data persepsi konsumen.")

    st.markdown("#### 4. Skor Kesesuaian Budget — Fungsi Peluruhan Eksponensial")
    st.markdown(f"""
    <div class="formula-box">
    S_budget = 100, &nbsp; jika harga &le; budget<br>
    S_budget = 100 x exp(-k x (harga-budget)/budget), &nbsp; jika harga &gt; budget, &nbsp; k = {K_PELURUHAN}
    </div>
    """, unsafe_allow_html=True)
    st.caption("Dipilih fungsi eksponensial (bukan threshold kaku 0/100) agar selisih kecil di atas budget tidak langsung dihukum penuh, transisi skornya halus, mencerminkan toleransi konsumen yang riil.")

    st.markdown("#### 5. Skor Akhir Rekomendasi")
    st.markdown(f"""
    <div class="formula-box">
    S_akhir = beta x S_kesiapan(kota) + (1-beta) x S_budget(harga, budget), &nbsp; beta = {BETA}
    </div>
    """, unsafe_allow_html=True)
    st.caption("beta=0.5: infrastruktur dan kesesuaian budget diberi bobot seimbang sebagai dua pertimbangan utama calon konsumen.")

    st.markdown("#### 6. Analisis Klaster — Gaussian Mixture Model (GMM)")
    st.markdown("""
    <div class="formula-box">
    Variabel input (distandarisasi Z-score sebelum clustering):<br>
    x = [jumlah_SPKLU, rata-rata_daya_kW, jumlah_ultrafast]<br><br>
    GMM dipilih karena batas kesiapan EV antar kota cenderung gradasi (probabilistik),
    bukan kategori yang tegas seperti yang diasumsikan K-Means.
    </div>
    """, unsafe_allow_html=True)

    st.warning("Yang masih berupa asumsi (bukan hasil estimasi statistik): bobot kategori charger (1/2/4/8), alpha=0.6, beta=0.5, dan k=2. Semua nilai ini logis secara konsep tapi belum divalidasi dengan data survei konsumen riil. Jika ada data 401 responden dari dosen pembimbing, bobot-bobot ini sebaiknya diestimasi ulang (misal lewat regresi atau analisis faktor) agar bukan lagi asumsi peneliti.")

# =========================================================
# FOOTER
# =========================================================
st.markdown(f"""
<div class="footer-note">
    SETRUM — Prototipe untuk Statistics Infographic Competition, SATRIA DATA 2026.<br>
    Data SPKLU ({len(spklu_semua):,} lokasi) bersumber dari data riil yang dikumpulkan tim.
    Harga EV bersumber dari SEVA.id, AstraOtoshop, Moladin, Kompas (per Jan-Jun 2026), lihat kolom "Sumber Harga".
    Formula pemodelan dijelaskan lengkap di tab "Tentang Model".
</div>
""".replace(",", "."), unsafe_allow_html=True)
