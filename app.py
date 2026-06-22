import streamlit as st
import pickle
import numpy as np
import shap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from icons import ICONS, icon

# ── Konfigurasi ──
st.set_page_config(
    page_title="DiabetesAI — Prediksi Diabetes",
    page_icon="🩺",
    layout="wide"
)

# ── Load CSS ──
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── Helper ──
def section_title(icon_name, text):
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:8px;font-size:1rem;'
        f'font-weight:600;color:#1a73e8;border-left:4px solid #1a73e8;'
        f'padding-left:10px;margin-bottom:12px;">'
        f'{icon(icon_name)} {text}</div>',
        unsafe_allow_html=True
    )

def labeled_input(icon_name, label, **kwargs):
    st.markdown(
        f'<div class="input-label">{icon(icon_name)} {label}</div>',
        unsafe_allow_html=True
    )
    key = label.replace(" ", "_").replace("/", "_").replace("(","").replace(")","")
    return st.number_input("", key=key, label_visibility="collapsed", **kwargs)

# ── Load Model ──
@st.cache_resource
def load_model():
    with open('diabetes_model/rf_model.pkl', 'rb') as f:
        rf = pickle.load(f)
    with open('diabetes_model/xgb_model.pkl', 'rb') as f:
        xgb = pickle.load(f)
    with open('diabetes_model/calibrated_model.pkl', 'rb') as f:
        calibrated_model = pickle.load(f)
    with open('diabetes_model/scaler_meta.pkl', 'rb') as f:
        scaler_meta = pickle.load(f)
    with open('diabetes_model/threshold.pkl', 'rb') as f:
        threshold = pickle.load(f)
    return rf, xgb, calibrated_model, scaler_meta, threshold

rf, xgb, calibrated_model, scaler_meta, threshold = load_model()

# ── Header ──
st.markdown(f"""
    <div class="main-header">
        <h1>{icon("stethoscope", "margin-right:10px;color:white;")} DiabetesAI</h1>
        <p>Sistem Prediksi Diabetes Mellitus Tipe-2 berbasis Machine Learning dengan Explainable AI (SHAP)</p>
        <p style="opacity:0.65;font-size:0.8rem;margin-top:4px;">
            Model: Hybrid Stacking (Random Forest + XGBoost + SVM) &nbsp;|&nbsp; Dataset: PIMA Indian Diabetes
        </p>
    </div>
""", unsafe_allow_html=True)

# ── Layout ──
col_input, col_result = st.columns([1, 1.2], gap="large")

# ════════════════════════
# KOLOM KIRI — INPUT
# ════════════════════════
with col_input:
    section_title("clipboard", "Data Klinis Pasien")

    glucose        = labeled_input("glucose", "Kadar Gula Darah / Glukosa (mg/dL)",     min_value=0,   max_value=300,  value=120)
    bmi            = labeled_input("weight",  "Indeks Massa Tubuh / BMI (kg/m²)",        min_value=0.0, max_value=70.0, value=30.0)
    age            = labeled_input("user",    "Usia (tahun)",                             min_value=21,  max_value=100,  value=30)
    pregnancies    = labeled_input("baby",    "Jumlah Kehamilan",                         min_value=0,   max_value=20,   value=1)
    insulin        = labeled_input("syringe", "Kadar Insulin Serum (mu U/ml)",           min_value=0,   max_value=900,  value=80)
    blood_pressure = labeled_input("heart",   "Tekanan Darah Diastolik (mm Hg)",         min_value=0,   max_value=150,  value=70)
    skin_thickness = labeled_input("ruler",   "Ketebalan Lipatan Kulit Trisep (mm)",     min_value=0,   max_value=100,  value=20)
    dpf            = labeled_input("dna",     "Riwayat Genetik Diabetes (0.0 – 2.5)",   min_value=0.0, max_value=3.0,  value=0.5)
    st.markdown("<br>", unsafe_allow_html=True)
    prediksi = st.button("Prediksi Sekarang", type="primary", use_container_width=True)

# ════════════════════════
# KOLOM KANAN — HASIL
# ════════════════════════
with col_result:
    if prediksi:
        with st.spinner("Memproses prediksi..."):
            input_data  = np.array([[pregnancies, glucose, blood_pressure,
                                      skin_thickness, insulin, bmi, dpf, age]])
            rf_prob     = rf.predict_proba(input_data)[:, 1]
            xgb_prob    = xgb.predict_proba(input_data)[:, 1]
            meta_input  = np.column_stack((rf_prob, xgb_prob,
                                           rf_prob - xgb_prob,
                                           rf_prob * xgb_prob))
            meta_scaled = scaler_meta.transform(meta_input)
            prob        = calibrated_model.predict_proba(meta_scaled)[:, 1][0]
            hasil       = 1 if prob >= threshold else 0

        # ── Hasil ──
        section_title("bar_chart", "Hasil Prediksi")

        if hasil == 1:
            st.markdown(
                f'<div class="result-positive">{icon("warning_red")} POSITIF DIABETES</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="result-negative">{icon("check")} NEGATIF DIABETES</div>',
                unsafe_allow_html=True
            )

        m1, m2, m3 = st.columns(3)
        m1.metric("Probabilitas", f"{prob:.2%}")
        m2.metric("Threshold",    f"{threshold:.4f}")
        m3.metric("Status",       "Berisiko" if hasil == 1 else "Aman")

        st.divider()

        # ── SHAP ──
        section_title("search", "Kontribusi Fitur (SHAP)")

        feature_names = ['Pregnancies','Glucose','BloodPressure','SkinThickness',
                         'Insulin','BMI','DiabetesPedigreeFunction','Age']

        explainer  = shap.TreeExplainer(rf)
        shap_vals  = explainer.shap_values(input_data)
        shap_input = shap_vals[1][0] if isinstance(shap_vals, list) else shap_vals[0,:,1]

        sorted_idx = np.argsort(np.abs(shap_input))[::-1]
        colors     = ['#e74c3c' if v > 0 else '#2980b9' for v in shap_input[sorted_idx]]

        nama_indo_chart = {
            'Glucose':                  'Kadar Gula Darah',
            'BMI':                      'Indeks Massa Tubuh',
            'Age':                      'Usia',
            'Pregnancies':              'Jumlah Kehamilan',
            'Insulin':                  'Kadar Insulin',
            'BloodPressure':            'Tekanan Darah',
            'DiabetesPedigreeFunction': 'Riwayat Genetik',
            'SkinThickness':            'Ketebalan Kulit',
        }
        satuan_chart = {
            'Glucose': 'mg/dL', 'BMI': 'kg/m²', 'Age': 'thn',
            'Pregnancies': 'x', 'Insulin': 'mU/ml',
            'BloodPressure': 'mmHg', 'DiabetesPedigreeFunction': '',
            'SkinThickness': 'mm',
        }
        labels = [
            f"{input_data[0][i]:.1f} {satuan_chart.get(feature_names[i], '')} = {nama_indo_chart.get(feature_names[i], feature_names[i])}"
            for i in sorted_idx
        ]
        values = shap_input[sorted_idx]

        fig, ax = plt.subplots(figsize=(7, 4.5))
        fig.patch.set_facecolor('#fafafa')
        ax.set_facecolor('#fafafa')
        bars = ax.barh(labels, values, color=colors, edgecolor='white', height=0.6)

        thr_panjang = 0.03
        for bar, val in zip(bars, values):
            x_pos   = bar.get_width()
            bar_len = abs(x_pos)
            if val >= 0:
                if bar_len > thr_panjang:
                    ax.text(x_pos-0.003, bar.get_y()+bar.get_height()/2,
                            f'+{val:.3f}', va='center', ha='right',
                            fontsize=8, color='white', fontweight='bold')
                else:
                    ax.text(x_pos+0.003, bar.get_y()+bar.get_height()/2,
                            f'+{val:.3f}', va='center', ha='left',
                            fontsize=8, color='#e74c3c', fontweight='bold')
            else:
                if bar_len > thr_panjang:
                    ax.text(x_pos+0.003, bar.get_y()+bar.get_height()/2,
                            f'{val:.3f}', va='center', ha='left',
                            fontsize=8, color='white', fontweight='bold')
                else:
                    ax.text(x_pos-0.003, bar.get_y()+bar.get_height()/2,
                            f'{val:.3f}', va='center', ha='right',
                            fontsize=8, color='#2980b9', fontweight='bold')

        ax.text(0.98, 0.98, f'f(x) = {prob:.3f}',
                transform=ax.transAxes, ha='right', va='top',
                fontsize=9, color='gray', style='italic')
        ax.axvline(0, color='#333', linewidth=0.8)
        ax.set_xlabel('SHAP Value', fontsize=9, color='#444')
        ax.set_title('Kontribusi Fitur terhadap Prediksi',
                     fontsize=10, fontweight='bold', color='#222')
        ax.tick_params(axis='y', labelsize=8.5)
        ax.tick_params(axis='x', labelsize=8)
        ax.invert_yaxis()
        for spine in ['top','right']:
            ax.spines[spine].set_visible(False)
        red_patch  = mpatches.Patch(color='#e74c3c', label='Mendorong → Diabetes')
        blue_patch = mpatches.Patch(color='#2980b9', label='Menahan ← Tidak Diabetes')
        ax.legend(handles=[red_patch, blue_patch], fontsize=8,
                  loc='lower right', framealpha=0.7)
        plt.tight_layout()
        st.pyplot(fig)

        st.divider()
# ── Interpretasi Naratif ──
        section_title("file_text", "Interpretasi Hasil")

        nama_indo = {
            'Glucose':                  'kadar gula darah',
            'BMI':                      'indeks massa tubuh (BMI)',
            'Age':                      'usia',
            'Pregnancies':              'jumlah kehamilan',
            'Insulin':                  'kadar insulin',
            'BloodPressure':            'tekanan darah',
            'DiabetesPedigreeFunction': 'riwayat genetik diabetes',
            'SkinThickness':            'ketebalan lipatan kulit',
        }

        satuan = {
            'Glucose': 'mg/dL', 'BMI': 'kg/m²', 'Age': 'tahun',
            'Pregnancies': 'kali', 'Insulin': 'mu U/ml',
            'BloodPressure': 'mm Hg', 'DiabetesPedigreeFunction': '',
            'SkinThickness': 'mm',
        }

        normal_range = {
            'Glucose':                  (140, 'tinggi',   'normal'),   # post OGTT 2 jam: <140 normal
            'BMI':                      (25,  'tinggi',   'normal'),   # 
            'Age':                      (45,  'berisiko', 'muda'),     # ADA: ≥45 tahun mulai skrining
            'Pregnancies':              (5,   'banyak',   'sedikit'),  # 
            'Insulin':                  (166, 'tinggi',   'normal'),   # 
            'BloodPressure':            (80,  'tinggi',   'normal'),   # 
            'DiabetesPedigreeFunction': (0.6, 'tinggi',   'rendah'),   # >0.6 = tinggi
            'SkinThickness':            (35,  'tebal',    'normal'),   # wanita dewasa: >35 tebal
        }

        # Deskripsi singkat per kondisi (untuk paragraf)
        deskripsi_singkat = {
            'Glucose':  {'tinggi': 'kadar gula darah yang tinggi ({val} mg/dL)',
                         'normal': 'kadar gula darah yang normal ({val} mg/dL)'},
            'BMI':      {'tinggi': 'indeks massa tubuh yang tinggi ({val} kg/m²)',
                         'normal': 'indeks massa tubuh yang normal ({val} kg/m²)'},
            'Age':      {'berisiko': 'usia yang sudah memasuki kelompok berisiko ({val} tahun)',
                         'muda':     'usia yang masih tergolong muda ({val} tahun)'},
            'Pregnancies': {'banyak':  'riwayat kehamilan yang cukup banyak ({val} kali)',
                            'sedikit': 'jumlah kehamilan yang sedikit ({val} kali)'},
            'Insulin':  {'tinggi': 'kadar insulin yang tinggi ({val} mu U/ml)',
                         'normal': 'kadar insulin yang normal ({val} mu U/ml)'},
            'BloodPressure': {'tinggi': 'tekanan darah yang tinggi ({val} mm Hg)',
                              'normal': 'tekanan darah yang normal ({val} mm Hg)'},
            'DiabetesPedigreeFunction': {'tinggi': 'riwayat genetik diabetes yang kuat ({val})',
                                         'rendah': 'riwayat genetik diabetes yang rendah ({val})'},
            'SkinThickness': {'tebal':  'ketebalan lipatan kulit yang cukup tebal ({val} mm)',
                              'normal': 'ketebalan lipatan kulit yang normal ({val} mm)'},
        }

        top_positif, top_negatif = [], []
        for i in sorted_idx:
            fname = feature_names[i]
            sval  = shap_input[i]
            fval  = input_data[0][i]
            batas, lbl_t, lbl_n = normal_range.get(fname, (0, str(fval), str(fval)))
            kondisi = lbl_t if fval >= batas else lbl_n
            if sval >  0.01: top_positif.append((fname, sval, fval, kondisi))
            if sval < -0.01: top_negatif.append((fname, sval, fval, kondisi))

        # ── Bangun kalimat deskripsi faktor ──
        def buat_deskripsi(daftar, max_item=3):
            hasil_deskripsi = []
            for fname, sval, fval, kondisi in daftar[:max_item]:
                tmpl = deskripsi_singkat.get(fname, {}).get(kondisi, f"{nama_indo.get(fname, fname)} ({fval:.1f})")
                hasil_deskripsi.append(tmpl.format(val=f"{fval:.1f}"))
            if len(hasil_deskripsi) == 1:
                return hasil_deskripsi[0]
            elif len(hasil_deskripsi) == 2:
                return f"{hasil_deskripsi[0]} serta {hasil_deskripsi[1]}"
            else:
                return f"{hasil_deskripsi[0]}, {hasil_deskripsi[1]}, serta {hasil_deskripsi[2]}"

        desc_positif = buat_deskripsi(top_positif) if top_positif else None
        desc_negatif = buat_deskripsi(top_negatif) if top_negatif else None

        # ── Paragraf utama ──
        if hasil == 1:
            paragraf = f"Berdasarkan data klinis yang dimasukkan, model memprediksi bahwa pasien ini **berisiko mengalami Diabetes Mellitus Tipe-2** dengan probabilitas sebesar **{prob:.1%}**."

            if desc_positif and desc_negatif:
                paragraf += (
                    f" Prediksi ini terutama didorong oleh {desc_positif}, "
                    f"yang menjadi faktor utama peningkat risiko diabetes pada pasien ini. "
                    f"Di sisi lain, terdapat faktor yang sedikit menekan risiko tersebut, "
                    f"yaitu {desc_negatif}, namun pengaruhnya tidak cukup besar untuk mengubah hasil prediksi secara keseluruhan."
                )
            elif desc_positif:
                paragraf += (
                    f" Prediksi ini terutama didorong oleh {desc_positif}, "
                    f"yang menjadi faktor utama peningkat risiko diabetes pada pasien ini."
                )

        else:
            paragraf = f"Berdasarkan data klinis yang dimasukkan, model memprediksi bahwa pasien ini **tidak terindikasi berisiko tinggi terhadap Diabetes Mellitus Tipe-2** dengan probabilitas diabetes hanya sebesar **{prob:.1%}**."

            if desc_negatif and desc_positif:
                paragraf += (
                    f" Hasil ini didukung oleh {desc_negatif}, "
                    f"yang menjadi faktor penekan risiko diabetes pada pasien ini. "
                    f"Meskipun terdapat beberapa faktor yang sedikit meningkatkan risiko seperti {desc_positif}, "
                    f"faktor-faktor tersebut tidak cukup dominan untuk mengubah prediksi secara keseluruhan."
                )
            elif desc_negatif:
                paragraf += (
                    f" Hasil ini didukung oleh {desc_negatif}, "
                    f"yang menjadi faktor penekan risiko diabetes pada pasien ini."
                )

        st.markdown(paragraf)
        st.markdown("---")

        # ── Penjelasan detail per faktor ──
        penjelasan_detail = {
            'Glucose': {
                'tinggi': 'Kadar gula darah {val} mg/dL tergolong tinggi (normal post OGTT 2 jam: <140 mg/dL). Ini menunjukkan tubuh kesulitan memproses glukosa, yang merupakan indikator utama diabetes.',
                'normal': 'Kadar gula darah {val} mg/dL masih dalam batas normal (post OGTT 2 jam: <140 mg/dL). Faktor ini tidak berkontribusi pada peningkatan risiko.',
            },
            'BMI': {
                'tinggi': 'Indeks massa tubuh {val} kg/m² tergolong di atas normal (normal: 18,5–24,9). Kelebihan berat badan membuat sel tubuh kurang responsif terhadap insulin.',
                'normal': 'Indeks massa tubuh {val} kg/m² masih dalam kategori normal. Faktor berat badan tidak berkontribusi pada peningkatan risiko.',
            },
            'Age': {
                'berisiko': 'Usia {val} tahun termasuk kelompok berisiko. Semakin tua, fungsi pankreas dalam memproduksi insulin cenderung menurun.',
                'muda':     'Usia {val} tahun masih tergolong muda. Risiko diabetes umumnya baru meningkat signifikan setelah usia 40 tahun.',
            },
            'Pregnancies': {
                'banyak':  'Riwayat {val} kali kehamilan perlu diperhatikan. Kehamilan berulang dapat memengaruhi respons tubuh terhadap insulin.',
                'sedikit': 'Jumlah kehamilan {val} kali tidak menunjukkan risiko tambahan yang signifikan.',
            },
            'Insulin': {
                'tinggi': 'Kadar insulin {val} mu U/ml berada di atas normal (16–166 mu U/ml), mengindikasikan tubuh sedang melawan resistensi insulin.',
                'normal': 'Kadar insulin {val} mu U/ml masih dalam rentang normal, menunjukkan respons pankreas yang baik.',
            },
            'BloodPressure': {
                'tinggi': 'Tekanan darah {val} mm Hg tergolong tinggi (normal: <80 mm Hg). Hipertensi dan diabetes sering terjadi bersamaan karena akar masalah yang sama.',
                'normal': 'Tekanan darah {val} mm Hg masih dalam batas normal dan tidak berkontribusi pada risiko diabetes.',
            },
            'DiabetesPedigreeFunction': {
                'tinggi': 'Nilai riwayat genetik {val} tergolong tinggi (>0,5), mengindikasikan riwayat keluarga yang cukup kuat terhadap diabetes.',
                'rendah': 'Nilai riwayat genetik {val} tergolong rendah, faktor keturunan tidak banyak berkontribusi pada risiko ini.',
            },
            'SkinThickness': {
            'tebal':  'Ketebalan lipatan kulit {val} mm tergolong di atas normal (normal wanita dewasa: <35 mm), dapat mengindikasikan kadar lemak tubuh yang lebih tinggi dan berpotensi berkontribusi pada resistensi insulin.',
            'normal': 'Ketebalan lipatan kulit {val} mm masih dalam batas normal (<35 mm) dan tidak menunjukkan risiko tambahan.',
            },
        }

        if top_positif:
            st.markdown("**Faktor yang meningkatkan risiko:**")
            for fname, sval, fval, kondisi in top_positif[:3]:
                detail = penjelasan_detail.get(fname, {}).get(kondisi, "").format(val=f"{fval:.1f}")
                st.markdown(
                    f'<div class="interp-box">'
                    f'<span style="color:#e74c3c;font-weight:bold;">▲ {nama_indo.get(fname,"").capitalize()}</span><br>'
                    f'<span style="color:#444;font-size:0.9rem;">{detail}</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )

        if top_negatif:
            st.markdown("<br>**Faktor yang menekan risiko:**", unsafe_allow_html=True)
            for fname, sval, fval, kondisi in top_negatif[:3]:
                detail = penjelasan_detail.get(fname, {}).get(kondisi, "").format(val=f"{fval:.1f}")
                st.markdown(
                    f'<div class="interp-box" style="border-left-color:#2980b9;">'
                    f'<span style="color:#2980b9;font-weight:bold;">▼ {nama_indo.get(fname,"").capitalize()}</span><br>'
                    f'<span style="color:#444;font-size:0.9rem;">{detail}</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Saran ──
        if hasil == 1:
            saran = (
                "**Saran:** Mengingat hasil prediksi menunjukkan risiko diabetes, sangat disarankan "
                "untuk segera berkonsultasi dengan dokter atau tenaga medis untuk pemeriksaan lebih lanjut. "
                "Langkah pencegahan seperti menjaga pola makan, rutin berolahraga, dan memantau "
                "kadar gula darah secara berkala dapat membantu mengurangi risiko tersebut."
            )
        else:
            saran = (
                "**Saran:** Meski hasil prediksi menunjukkan risiko rendah, tetap disarankan untuk "
                "menerapkan pola hidup sehat — menjaga berat badan ideal, mengonsumsi makanan bergizi, "
                "dan rutin berolahraga — sebagai langkah pencegahan jangka panjang."
            )
        st.markdown(saran)
        st.markdown(
            f'<div class="disclaimer">{icon("alert")} '
            f'<span><b>Disclaimer:</b> Hasil prediksi ini bersifat sebagai alat bantu '
            f'skrining awal dan <b>tidak menggantikan diagnosis medis</b> dari tenaga '
            f'kesehatan profesional.</span></div>',
            unsafe_allow_html=True
        )

    else:
        st.markdown("""
            <div style="text-align:center;color:#bbb;padding:5rem 2rem;">
                <svg xmlns="http://www.w3.org/2000/svg" width="60" height="60"
                     viewBox="0 0 24 24" fill="none" stroke="#ccc" stroke-width="1.5">
                    <path d="M4.8 2.3A.3.3 0 1 0 5 2H4a2 2 0 0 0-2 2v5a6 6 0 0 0 6 6
                             6 6 0 0 0 6-6V4a2 2 0 0 0-2-2h-1a.2.2 0 1 0 .3.3"/>
                    <path d="M8 15v1a6 6 0 0 0 6 6h0a6 6 0 0 0 6-6v-4"/>
                    <circle cx="20" cy="10" r="2"/>
                </svg>
                <div style="font-size:1rem;margin-top:1.2rem;color:#aaa;">
                    Isi data klinis pasien di sebelah kiri,<br>
                    lalu klik <b>Prediksi Sekarang</b>
                </div>
            </div>
        """, unsafe_allow_html=True)