import json
from pathlib import Path

import joblib
import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title='Dashboard Kualitas Udara Dalam Ruangan', layout='wide')

BASE = Path(__file__).parent
RAW_CSV = BASE / 'data' / 'IoT_Indoor_Air_Quality_Dataset.csv'
CURATED = BASE / 'data' / 'iaqi_curated_zone.parquet'
SK_MODEL = BASE / 'models' / 'rf_sklearn.joblib'
SPARK_MODEL = BASE / 'models' / 'rf_model_spark'
META = BASE / 'models' / 'meta.json'

ORDER = ['Good', 'Moderate', 'Polluted']
COLORS = {'Good': '#2e9e5b', 'Moderate': '#e0a100', 'Polluted': '#d64545'}
# (batas Moderate, batas Polluted), sama dengan calculate_iaq_status di notebook
THRESH = {'pm25': (21, 51), 'pm10': (31, 76), 'co2': (600, 1000), 'co': (1.8, 8.8)}


# ---------- fungsi bantu ----------
def ribu(n):
    return f'{n:,}'.replace(',', '.')


def standardize(df):
    df = df.copy()
    df.columns = (df.columns.str.replace(r'[(].*[)]', '', regex=True)                     .str.strip().str.lower().str.replace(' ', '_'))
    return df.rename(columns={'pm2.5': 'pm25'})


def sub_levels(vals):
    return {k: ('Polluted' if vals[k] >= p else 'Moderate' if vals[k] >= m else 'Good')
            for k, (m, p) in THRESH.items()}


def rule_label(vals):
    return max(sub_levels(vals).values(), key=ORDER.index)


@st.cache_data
def load_raw():
    return pd.read_csv(RAW_CSV)


@st.cache_data
def load_clean():
    return pd.read_parquet(CURATED)


@st.cache_data
def load_meta():
    return json.loads(META.read_text(encoding='utf-8'))


@st.cache_resource
def load_sklearn():
    return joblib.load(SK_MODEL)


@st.cache_resource
def load_spark():
    from pyspark.sql import SparkSession
    from pyspark.ml.classification import RandomForestClassificationModel
    spark = (SparkSession.builder.master('local[1]').appName('IAQ_Dashboard')
             .config('spark.driver.memory', '1g').config('spark.ui.enabled', 'false')
             .config('spark.sql.shuffle.partitions', '1').getOrCreate())
    return spark, RandomForestClassificationModel.load(str(SPARK_MODEL))


def predict_sklearn(row, cols):
    model = load_sklearn()
    proba = model.predict_proba(pd.DataFrame([row], columns=cols))[0]
    return {str(c): float(p) for c, p in zip(model.classes_, proba)}


def predict_spark(row, cols, labels):
    from pyspark.ml.feature import VectorAssembler
    spark, model = load_spark()
    sdf = spark.createDataFrame([tuple(float(row[c]) for c in cols)], cols)
    sdf = VectorAssembler(inputCols=cols, outputCol='features').transform(sdf)
    proba = model.transform(sdf).select('probability').first()[0].toArray()
    return {lab: float(p) for lab, p in zip(labels, proba)}


@st.cache_data
def quality_scores(raw):
    d = standardize(raw)
    d['timestamp'] = pd.to_datetime(d['timestamp'], dayfirst=True, errors='coerce')
    selaras = ~((d['motion_detected'] == 0) & (d['occupancy_count'] > 0))
    ts = d['timestamp'].dropna().drop_duplicates().sort_values()
    gaps = int((ts.diff().dropna() != pd.Timedelta(minutes=5)).sum())
    scores = {
        'Completeness': 100 * d.notna().all(axis=1).mean(),
        'Uniqueness (timestamp)': 100 * (1 - d['timestamp'].duplicated().mean()),
        'Validity (humidity 30-70%)': 100 * d['humidity'].between(30, 70).mean(),
        'Consistency (PM2.5 ≤ PM10)': 100 * (d['pm25'] <= d['pm10']).mean(),
        'Consistency (motion–occupancy)': 100 * selaras.mean(),
    }
    return scores, gaps


@st.cache_data
def build_graph_tables(df):
    g = df.sort_values('timestamp').reset_index(drop=True)
    g['state'] = g['aq_status'] + '_' + g['ventilation_status']
    g['next_state'] = g['state'].shift(-1)
    edges = (g.dropna(subset=['next_state']).groupby(['state', 'next_state'])
             .size().reset_index(name='weight'))
    G = nx.DiGraph()
    for r in edges.itertuples():
        G.add_edge(r.state, r.next_state, weight=r.weight)
    gm = pd.DataFrame({'PageRank': nx.pagerank(G, weight='weight'),
                       'Weighted In-Degree': dict(G.in_degree(weight='weight'))})
    return edges, gm.sort_values('PageRank', ascending=False)


def graph_figure(edges, gm, min_weight):
    nodes = sorted(gm.index)
    pr = gm['PageRank'].to_dict()
    ang = np.linspace(0, 2 * np.pi, len(nodes), endpoint=False)
    pos = {n: (float(np.cos(a)), float(np.sin(a))) for n, a in zip(nodes, ang)}
    wmax = edges['weight'].max()
    fig = go.Figure()
    for r in edges.itertuples():
        if r.state == r.next_state or r.weight < min_weight:
            continue
        (x0, y0), (x1, y1) = pos[r.state], pos[r.next_state]
        d = float(np.hypot(x1 - x0, y1 - y0))
        ux, uy = (x1 - x0) / d, (y1 - y0) / d
        ox, oy = -uy * 0.05, ux * 0.05  # geser agar panah dua arah tidak menumpuk
        fig.add_annotation(
            x=x1 - ux * 0.13 + ox, y=y1 - uy * 0.13 + oy,
            ax=x0 + ux * 0.13 + ox, ay=y0 + uy * 0.13 + oy,
            xref='x', yref='y', axref='x', ayref='y', showarrow=True,
            arrowhead=3, arrowwidth=0.5 + 6 * r.weight / wmax,
            arrowcolor='rgba(90,90,90,0.55)')
    fig.add_trace(go.Scatter(
        x=[pos[n][0] for n in nodes], y=[pos[n][1] for n in nodes],
        mode='markers+text', text=nodes, textposition='top center',
        marker=dict(size=[20 + 120 * pr[n] for n in nodes],
                    color=[COLORS[n.split('_')[0]] for n in nodes]),
        hoverinfo='text',
        hovertext=[f'{n}<br>PageRank {pr[n]:.4f}' for n in nodes]))
    fig.update_xaxes(visible=False, range=[-1.5, 1.5])
    fig.update_yaxes(visible=False, range=[-1.4, 1.4], scaleanchor='x')
    fig.update_layout(height=620, showlegend=False, margin=dict(l=10, r=10, t=10, b=10))
    return fig


def plot_cm(block, title):
    labels = block['labels']
    cm = pd.DataFrame(block['confusion_matrix'], index=labels, columns=labels).loc[ORDER, ORDER]
    return px.imshow(cm, text_auto=True, color_continuous_scale='Blues', title=title,
                     labels=dict(x='Prediksi', y='Aktual', color='Jumlah'))


# ---------- data ----------
raw, clean, meta = load_raw(), load_clean(), load_meta()
cols = meta['feature_cols']
cmp = pd.DataFrame(meta['compare']).set_index('Metrik')

st.title('Dashboard Kualitas Udara Dalam Ruangan')
st.caption('Big Data Analytics · IoT Indoor Air Quality (Kaggle) · Curated Zone → Graph Analytics dan Random Forest')

t_ov, t_raw, t_clean, t_graph, t_rf, t_pred = st.tabs(
    ['Ringkasan', 'Dataset Utama', 'Dataset Bersih', 'Graph Analytics', 'Random Forest', 'Klasifikasi'])

# ---------- 1. Ringkasan ----------
with t_ov:
    m = st.columns(4)
    m[0].metric('Data mentah (baris)', ribu(len(raw)))
    m[1].metric('Data bersih (baris)', ribu(len(clean)), f'-{ribu(len(raw) - len(clean))}', delta_color='off')
    m[2].metric('Akurasi PySpark', cmp.loc['Accuracy', 'PySpark'])
    m[3].metric('Akurasi Scikit-Learn', cmp.loc['Accuracy', 'Scikit-Learn'])
    dist = clean['aq_status'].value_counts().reindex(ORDER).reset_index()
    dist.columns = ['Kelas', 'Jumlah']
    st.plotly_chart(px.bar(dist, x='Kelas', y='Jumlah', color='Kelas', text='Jumlah',
                           color_discrete_map=COLORS, title='Distribusi kelas aq_status'))

# ---------- 2. Dataset utama ----------
with t_raw:
    st.subheader('Dataset utama (Raw Zone)')
    st.write(f'{ribu(len(raw))} baris × {raw.shape[1]} kolom')
    st.dataframe(raw, height=420)
    miss = raw.isna().sum().rename('Jumlah kosong').to_frame()
    miss['Persen'] = (miss['Jumlah kosong'] / len(raw) * 100).round(2)
    st.markdown('**Nilai kosong per kolom**')
    st.dataframe(miss)
    scores, gaps = quality_scores(raw)
    st.markdown('**Skor kualitas data (Tabel 3.2 laporan)**')
    q = st.columns(len(scores))
    for col, (name, val) in zip(q, scores.items()):
        col.metric(name, f'{val:.2f}%')
    st.caption(f'Timeliness: {ribu(gaps)} selisih timestamp yang bukan tepat 5 menit.')

# ---------- 3. Dataset bersih ----------
with t_clean:
    st.subheader('Dataset bersih (Curated Zone)')
    pick = st.multiselect('Filter kelas kualitas udara', ORDER, default=ORDER)
    view = clean[clean['aq_status'].isin(pick)]
    st.write(f'{ribu(len(view))} baris ditampilkan')
    st.dataframe(view, height=420)
    st.download_button('Unduh CSV (hasil filter)', view.to_csv(index=False).encode('utf-8'),
                       'iaq_bersih.csv', 'text/csv')
    pol = st.selectbox('Distribusi polutan per kelas', ['pm25', 'pm10', 'co2', 'co', 'tvoc'])
    st.plotly_chart(px.histogram(clean, x=pol, color='aq_status', nbins=60, barmode='overlay',
                                 opacity=0.7, color_discrete_map=COLORS,
                                 category_orders={'aq_status': ORDER}))

# ---------- 4. Graph analytics ----------
with t_graph:
    st.subheader('Graph Analytics: transisi state (aq_status × ventilasi)')
    edges, gm = build_graph_tables(clean)
    c = st.columns(3)
    c[0].metric('Simpul (nodes)', len(gm))
    c[1].metric('Garis (edges)', len(edges))
    c[2].metric('Total transisi', ribu(int(edges['weight'].sum())))
    min_w = st.slider('Tampilkan edge dengan bobot minimal', 0, int(edges['weight'].max()), 0, step=50)
    st.plotly_chart(graph_figure(edges, gm, min_w))
    st.caption('Ukuran simpul = PageRank, warna = kelas kualitas udara, tebal panah = bobot. Self-loop tidak digambar, lihat tabel edge.')
    g1, g2 = st.columns(2)
    g1.markdown('**PageRank dan Weighted In-Degree**')
    g1.dataframe(gm.style.format({'PageRank': '{:.4f}', 'Weighted In-Degree': '{:,.0f}'}))
    heat = edges.pivot(index='state', columns='next_state', values='weight').fillna(0)
    g2.markdown('**Matriks transisi**')
    g2.plotly_chart(px.imshow(heat, text_auto=True, color_continuous_scale='Blues'))
    st.markdown('**Tabel edge**')
    st.dataframe(edges.sort_values('weight', ascending=False))

# ---------- 5. Random Forest ----------
with t_rf:
    st.subheader('Hasil evaluasi Random Forest')
    st.dataframe(cmp)
    a, b = st.columns(2)
    a.plotly_chart(plot_cm(meta['spark'], 'Confusion matrix: PySpark MLlib'))
    b.plotly_chart(plot_cm(meta['sklearn'], 'Confusion matrix: Scikit-Learn'))
    a.markdown('**Metrik per kelas, PySpark**')
    a.dataframe(pd.DataFrame(meta['spark']['per_class']).set_index('Class'))
    b.markdown('**Metrik per kelas, Scikit-Learn**')
    b.dataframe(pd.DataFrame(meta['sklearn']['per_class']).set_index('Class'))
    fi = pd.DataFrame({'PySpark': meta['spark']['feature_importance'],
                       'Scikit-Learn': meta['sklearn']['feature_importance']}) * 100
    fi.index.name = 'Fitur'
    fi = fi.sort_values('Scikit-Learn', ascending=False).reset_index().melt(
        id_vars='Fitur', var_name='Model', value_name='Importance (%)')
    log_y = st.toggle('Skala logaritmik (agar fitur kecil terlihat)', value=True)
    st.plotly_chart(px.bar(fi, x='Fitur', y='Importance (%)', color='Model',
                           barmode='group', log_y=log_y))

# ---------- 6. Klasifikasi ----------
with t_pred:
    st.subheader('Klasifikasi kualitas udara dengan Random Forest')
    choice = st.radio('Model', ['PySpark MLlib', 'Scikit-Learn'], horizontal=True)
    d, lo, hi = meta['defaults'], meta['min'], meta['max']
    with st.form('form_klasifikasi'):
        st.markdown('**Konsentrasi polutan**')
        c1, c2, c3, c4 = st.columns(4)
        pm25 = c1.number_input('PM2.5 (µg/m³)', 0.0, 1000.0, float(round(d['pm25'], 1)), 0.5)
        pm10 = c2.number_input('PM10 (µg/m³)', 0.0, 1000.0, float(round(d['pm10'], 1)), 0.5)
        co2 = c3.number_input('CO₂ (ppm)', 0.0, 10000.0, float(round(d['co2'], 0)), 10.0)
        co = c4.number_input('CO (ppm)', 0.0, 100.0, float(round(d['co'], 2)), 0.1)
        with st.expander('Parameter lain (default = median data bersih)'):
            e1, e2, e3, e4 = st.columns(4)
            temperature = e1.number_input('Suhu (°C)', -10.0, 60.0, float(round(d['temperature'], 1)), 0.5)
            humidity = e2.number_input('Kelembapan (%)', 0.0, 100.0, float(round(d['humidity'], 1)), 1.0)
            tvoc = e3.number_input('TVOC (ppb)', 0.0, 5000.0, float(round(d['tvoc'], 0)), 10.0)
            light = e4.number_input('Intensitas cahaya (lux)', 0.0, 100000.0, float(round(d['light_intensity'], 0)), 10.0)
            f1, f2, f3 = st.columns(3)
            motion = f1.selectbox('Motion detected', [0, 1], index=int(d['motion_detected']))
            occ = f2.number_input('Jumlah penghuni', 0, 500, int(d['occupancy_count']), 1)
            vent = f3.selectbox('Ventilasi', ['Closed', 'Open'], index=int(d['ventilation_encoded']))
        go_btn = st.form_submit_button('Klasifikasikan', type='primary')

    if go_btn:
        row = {'temperature': temperature, 'humidity': humidity, 'co2': co2, 'pm25': pm25,
               'pm10': pm10, 'tvoc': tvoc, 'co': co, 'light_intensity': light,
               'motion_detected': motion, 'occupancy_count': occ,
               'ventilation_encoded': 1 if vent == 'Open' else 0}
        outside = [f'{c} = {row[c]:g} (data: {lo[c]:g} sampai {hi[c]:g})'
                   for c in cols if not lo[c] <= row[c] <= hi[c]]
        try:
            with st.spinner('Menjalankan model...'):
                if choice == 'PySpark MLlib':
                    proba = predict_spark(row, cols, meta['spark']['labels'])
                else:
                    proba = predict_sklearn(row, cols)
        except Exception as exc:
            st.error(f'Model gagal dijalankan: {exc}')
            st.stop()
        pred = max(proba, key=proba.get)
        rule = rule_label(row)
        left, right = st.columns([1, 2])
        box = {'Good': left.success, 'Moderate': left.warning, 'Polluted': left.error}[pred]
        box(f'Prediksi {choice}: **{pred}**')
        left.write(f'Label menurut aturan ambang: **{rule}**')
        if rule == pred:
            left.caption('Prediksi model sama dengan aturan ambang.')
        else:
            left.warning('Prediksi model berbeda dari aturan ambang.')
        sub = pd.DataFrame({'Nilai': {k: row[k] for k in THRESH}, 'Level': sub_levels(row)})
        left.markdown('**Level per polutan**')
        left.dataframe(sub)
        pdf = pd.DataFrame({'Kelas': ORDER, 'Probabilitas': [proba[c] for c in ORDER]})
        right.plotly_chart(px.bar(pdf, x='Kelas', y='Probabilitas', color='Kelas',
                                  color_discrete_map=COLORS, range_y=[0, 1], text_auto='.1%'))
        if outside:
            st.warning('Input di luar rentang data latih: ' + '; '.join(outside) +
                       '. Prediksi pada kondisi ini tidak dapat diandalkan.')