import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from scipy.stats import spearmanr 

# =======================================================
# --- KONSTANTA & KONFIGURASI UMUM ---
# =======================================================

# Nama Kolom Likert
LIKERT_COLS = ['skala_kenyamanan(1-5)', 'skala_harga(1-5)', 
               'skala_rasa(1-5)', 'skala_metode_bayar(1-5)', 
               'skala_rekomendasi(1-5)', 'skala_antrean(1-5)']
# Pemetaan Nama Metrik untuk Tampilan
METRIC_MAPPING = {
    'skala_kenyamanan(1-5)': 'Kenyamanan Kantin',
    'skala_harga(1-5)': 'Harga Makanan',
    'skala_rasa(1-5)': 'Rasa Makanan',
    'skala_metode_bayar(1-5)': 'Metode Pembayaran',
    'skala_rekomendasi(1-5)': 'Rekomendasi (Overall)',
    'skala_antrean(1-5)': 'Waktu Antrean (Dibalik)' 
}

# Page config
st.set_page_config(
    page_title="Canteen Satisfaction Survey Dashboard",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    div[data-testid="metric-container"] {
        background-color: rgba(28, 131, 225, 0.1);
        border: 1px solid rgba(28, 131, 225, 0.2);
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    /* Memastikan warna delta kritis tetap Merah */
    .st-emotion-cache-1rq3e5w.e1ekc1qf1 {
        color: #DC3545 !important;
    }
    .footer {
        text-align: center;
        padding: 20px;
        color: #666;
        border-top: 1px solid #e0e0e0;
        margin-top: 50px;
    }
    /* Peningkatan Font Size untuk Narasi */
    .lead-text p {
        font-size: 17px;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)


# =======================================================
# --- FUNGSI DATA LOADING, FILTERING, & DESKRIPTIF ---
# =======================================================

@st.cache_data
def load_data(file_path=None): 
    """Load and prepare the dataset"""
    try:
        if file_path and Path(file_path).exists():
            df = pd.read_csv(file_path)
        else:
            st.error(f"Default file not found at {file_path}. Please ensure the file is in the correct directory.")
            return None
        
        # Define column types
        id_cols = ['Nama Lengkap', 'NPM', 'Fakultas', 'Prodi', 
                   'kategori_pengeluaran', 'kategori_frekuensi_makan']
        
        numeric_cols = ['rata_rata_uang', 'frekuensi_makan_seminggu', 
                        'waktu_tunggu_menit', 'jumlah_menu', 'frekuensi_metode_bayar', 'skor_kepuasan']
        
        likert_cols = LIKERT_COLS
        
        if 'NPM' in df.columns:
            df['NPM'] = df['NPM'].astype(str)
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        for col in likert_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    except Exception as e:
        st.error(f"Error loading or processing data: {str(e)}")
        return None

def apply_filters(df, filters):
    """Apply sidebar filters to dataframe"""
    df_filtered = df.copy()
    
    # Apply categorical filters
    for col, values in filters['categorical'].items():
        if values and col in df_filtered.columns:
            df_filtered = df_filtered[df_filtered[col].isin(values)]
    
    # Apply numeric filters
    for col, (min_val, max_val) in filters['numeric'].items():
        if col in df_filtered.columns:
            df_filtered = df_filtered[
                (df_filtered[col] >= min_val) & 
                (df_filtered[col] <= max_val)
            ]
    
    return df_filtered

@st.cache_data
def describe_numeric(df, exclude_cols=['NPM']):
    """Generate descriptive statistics for numeric columns"""
    numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                    if col not in exclude_cols]
    
    if not numeric_cols:
        return pd.DataFrame()
    
    stats = df[numeric_cols].describe().T
    stats['Q1'] = df[numeric_cols].quantile(0.25)
    stats['Q3'] = df[numeric_cols].quantile(0.75)
    
    # Reorder columns
    stats = stats[['count', 'mean', 'std', 'min', 'Q1', '50%', 'Q3', 'max']]
    stats.columns = ['Count', 'Mean', 'Std Dev', 'Min', 'Q1', 'Median', 'Q3', 'Max']
    
    return stats.round(2)


# =======================================================
# --- FUNGSI ANALISIS & KORELASI ---
# =======================================================

@st.cache_data
def compute_correlations(df, method='spearman'):
    """
    Menghitung korelasi Item-Total (RIT) sesuai logika pre-processing (termasuk reverse coding).
    """
    
    items = LIKERT_COLS
    valid_items = [c for c in items if c in df.columns]
    
    if len(valid_items) < 3:
        return pd.DataFrame(), pd.DataFrame([{'Variable 1': np.nan, 'Correlation': np.nan}])
        
    X_num = df[valid_items].copy(deep=True).apply(pd.to_numeric, errors="coerce")
    X_num = X_num.where((X_num >= 1) & (X_num <= 5), np.nan)

    Z = X_num.copy()
    if "skala_antrean(1-5)" in Z.columns:
        Z["skala_antrean(1-5)"] = 6 - Z["skala_antrean(1-5)"]

    rows_rit = []
    for col in Z.columns:
        others = [c for c in Z.columns if c != col]
        
        rest_score = Z[others].mean(axis=1)
        
        sub = pd.concat([Z[col], rest_score], axis=1).dropna()
        sub.columns = ['item_score', 'rest_score']

        if len(sub) >= 10:
            rho, p = spearmanr(sub['item_score'], sub['rest_score'])
            rows_rit.append({
                "Variable 1": col, 
                "Variable 2": "Rest Score",
                "Correlation": rho, 
                "p_value": p, 
                "n": len(sub)
            })
        else:
            rows_rit.append({
                "Variable 1": col, 
                "Variable 2": "Rest Score",
                "Correlation": np.nan, 
                "p_value": np.nan, 
                "n": len(sub)
            })

    rit_df = pd.DataFrame(rows_rit) 
    
    # --- Korelasi Penuh (Untuk Heatmap) ---
    # Hitung korelasi antar semua metrik yang sudah dibalik (Z)
    full_corr_matrix = Z.corr(method='spearman')
        
    return full_corr_matrix, rit_df

def analyze_likert(df):
    """Analyze Likert scale questions and prepare data for pie charts."""
    likert_cols = LIKERT_COLS
    valid_cols = [col for col in likert_cols if col in df.columns]
    
    if not valid_cols:
        return None, None
    
    stats = []
    pie_data = []
    
    for col in valid_cols:
        col_data = df[col].dropna()
        positive_percent = (col_data >= 4).mean() * 100
        
        question_title = col.replace('skala_', '').replace('(1-5)', '').title()
        
        stats.append({
            'Question': question_title,
            'Mean': col_data.mean(),
            'Median': col_data.median(),
            '% Positive (4-5)': positive_percent,
            '% Neutral/Negative (1-3)': 100 - positive_percent, 
            'N': len(col_data)
        })
        
        question_display = METRIC_MAPPING.get(col, question_title)

        pie_data.append({
            'Question': question_title,
            'Category': 'Positive (4-5)',
            'Percentage': positive_percent
        })
        pie_data.append({
            'Question': question_title,
            'Category': 'Neutral/Negative (1-3)',
            'Percentage': 100 - positive_percent
        })
    
    stats_df = pd.DataFrame(stats)
    pie_df = pd.DataFrame(pie_data)
    
    return stats_df, pie_df


# =======================================================
# --- FUNGSI CHART VISUALISASI ---
# =======================================================

def plot_numeric_distribution(df, columns, plot_type='histogram'):
    """
    Membuat histogram untuk variabel continuous (Menggunakan warna biru).
    """
    if not columns:
        return None
    
    # Logic untuk Histogram (disederhanakan untuk satu kolom)
    col = columns[0]
    fig = px.histogram(df, x=col, nbins=30,
                       title=f'Distribution of {col}',
                       # Warna biru dari custom CSS
                       color_discrete_sequence=['#667eea']) 
    
    fig.update_layout(height=400, showlegend=True)
    return fig

def plot_categorical(df, column, top_n=None):
    """
    Create bar chart for categorical column, improved for Likert scales (Menggunakan warna biru).
    """
    if '(1-5)' in column:
        series_data = df[column].dropna().astype(int)
        value_counts = series_data.value_counts().sort_index()

        target_categories = [i for i in range(1, 6)]
        
        value_counts_full = value_counts.reindex(target_categories, fill_value=0)
        
        df_plot = value_counts_full.reset_index()
        df_plot.columns = [column, 'Count']
        
        df_plot[column] = df_plot[column].astype(str)
        
        category_order = [str(i) for i in target_categories]

    else:
        value_counts = df[column].value_counts()
        if top_n:
            value_counts = value_counts.head(top_n)
            
        df_plot = value_counts.reset_index()
        df_plot.columns = [column, 'Count']
        category_order = 'total ascending'

    fig = px.bar(df_plot, 
                 x='Count', 
                 y=column,
                 orientation='h',
                 title=f'Distribution of {column}',
                 labels={'x': 'Count', 'y': column},
                 color_discrete_sequence=['#667eea']) 
    
    if '(1-5)' in column:
        fig.update_yaxes(type='category', categoryorder='category ascending', categoryarray=category_order)
        fig.update_layout(height=400)
    else:
        fig.update_layout(height=max(400, len(df_plot) * 30))
        
    return fig

def plot_correlation_heatmap(corr_matrix):
    """Create correlation heatmap (Hanya Item-to-Item)"""
    clean_cols = {}
    
    # Filter kolom agar hanya metrik LIKERT yang tersisa (menghapus skor_kepuasan)
    matrix_to_plot = corr_matrix.copy()
    
    # Menghapus skor_kepuasan_original jika ada di indeks/kolom
    cols_to_drop = [col for col in matrix_to_plot.columns if 'skor_kepuasan' in col]
    matrix_to_plot = matrix_to_plot.drop(columns=cols_to_drop, index=cols_to_drop, errors='ignore')
    
    for col in matrix_to_plot.columns:
        if '(1-5)' in col:
             clean_cols[col] = col.replace('skala_', '').replace('(1-5)', '').title()
        else:
            clean_cols[col] = col
            
    corr_matrix_clean = matrix_to_plot.rename(columns=clean_cols, index=clean_cols)
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix_clean.values,
        x=corr_matrix_clean.columns,
        y=corr_matrix_clean.columns,
        colorscale='RdBu',
        zmid=0,
        text=corr_matrix_clean.values.round(2),
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title="Correlation")
    ))
    
    fig.update_layout(
        title='Correlation Heatmap (Spearman Rank)',
        height=500,
        xaxis={'side': 'bottom'},
        yaxis={'side': 'left'}
    )
    
    return fig

def plot_impact_bar_chart(rit_df):
    """Creates a bar chart showing the correlation (impact) of metrics on 'Rest Score' (RIT)."""
    
    if rit_df.empty or rit_df['Correlation'].isnull().all():
        return px.bar(title="Impact Analysis Not Available (No valid data for RIT calculation).")

    impact_df = rit_df.copy()
    impact_df['Metric'] = impact_df['Variable 1'].map(METRIC_MAPPING)
    
    impact_df = impact_df.sort_values(by='Correlation', ascending=True) 
    
    fig = px.bar(
        impact_df,
        x='Correlation',
        y='Metric',
        orientation='h',
        color='Correlation',
        color_continuous_scale=px.colors.diverging.RdBu_r, 
        range_color=[-1, 1],
        text=impact_df['Correlation'].round(3),
        title='Item-Rest Correlation: Impact on Internal Consistency'
    )
    
    fig.update_traces(textposition='auto')
    fig.update_layout(
        height=400,
        xaxis_title='Spearman Correlation (RIT)', # FIX: Menghapus simbol LaTeX
        yaxis_title='Satisfaction Metric'
    )
    
    return fig

def plot_satisfaction_boxplot(df, stats_df):
    """
    Boxplot untuk memvisualisasikan distribusi skor kepuasan total berdasarkan kategori pengeluaran.
    """
    
    if 'skor_kepuasan' not in df.columns or 'kategori_pengeluaran' not in df.columns:
        return px.bar(title="Boxplot Not Available (Missing 'skor_kepuasan' or 'kategori_pengeluaran').")

    # Order kategori pengeluaran secara eksplisit
    category_order = ['Rendah', 'Sedang', 'Tinggi']
    
    df_plot = df.copy()
    
    # Gunakan median skor kepuasan untuk mengurutkan boxplot secara visual (optional)
    median_order = df_plot.groupby('kategori_pengeluaran')['skor_kepuasan'].median().sort_values(ascending=False).index.tolist()
    
    fig = px.box(df_plot, 
                 x='kategori_pengeluaran', 
                 y='skor_kepuasan', 
                 color='kategori_pengeluaran',
                 category_orders={'kategori_pengeluaran': category_order},
                 color_discrete_sequence=px.colors.qualitative.Plotly,
                 title='Distribusi Skor Kepuasan Berdasarkan Kategori Pengeluaran')
                 
    fig.update_traces(quartilemethod="exclusive")
    fig.update_layout(xaxis_title='Kategori Pengeluaran', 
                      yaxis_title='Skor Kepuasan Total (1-5)',
                      showlegend=False)

    return fig

def plot_single_pie_chart(pie_df, question):
    """Creates a single interactive pie chart for a selected question."""
    data_for_pie = pie_df[pie_df['Question'] == question]
    
    if data_for_pie['Percentage'].sum() == 0:
        return None

    fig_pie = px.pie(data_for_pie, 
                     values='Percentage', 
                     names='Category', 
                     title=f'Distribution for {question}',
                     color='Category',
                     # MODIFIED: Skema warna Biru/Merah
                     color_discrete_map={'Positive (4-5)': '#007BFF', # Biru
                                         'Neutral/Negative (1-3)': '#DC3545'}, # Merah
                     hole=0.4)
    
    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
    fig_pie.update_layout(showlegend=True, margin=dict(l=20, r=20, t=50, b=20), height=400)
    
    return fig_pie

def create_gauge_chart(value, min_val=1, max_val=5):
    """Create a gauge chart for overall satisfaction"""
    normalized_value = (value - min_val) / (max_val - min_val) * 100
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=normalized_value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Overall Satisfaction Score (%)"},
        delta={'reference': 60, 'increasing': {'color': "#4CAF50"}}, 
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 40], 'color': '#ff9999'}, 
                {'range': [40, 60], 'color': '#add8e6'}, 
                {'range': [60, 80], 'color': '#6495ed'}, 
                {'range': [80, 100], 'color': '#007BFF'} 
            ],
            'threshold': {
                'line': {'color': "#DC3545", 'width': 4}, 
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig

def plot_top_ranking_bar(stats_df):
    """Bar chart untuk Ranking Rata-rata Skor Kepuasan (Warna Biru Kontinu)"""
    ranking_df = stats_df.sort_values('Mean', ascending=True)
            
    fig_ranking = px.bar(ranking_df, x='Mean', y='Question',
                         orientation='h',
                         title='Average Satisfaction Metrics Score',
                         color='Mean',
                         color_continuous_scale='Blues',
                         range_color=[1, 5])
    fig_ranking.update_layout(height=400)
    return fig_ranking

def create_overview_kpis(df_filtered, stats_df):
    """Menghitung dan menampilkan KPI utama untuk Dashboard Overview (4 Kolom Rata)."""
    
    total_responses = len(df_filtered)
    avg_satisfaction = df_filtered['skor_kepuasan'].mean()
    
    best_metric = stats_df.loc[stats_df['Mean'].idxmax()]
    worst_metric = stats_df.loc[stats_df['Mean'].idxmin()]
    
    st.markdown("### 📊 Key Performance Indicators (KPIs)")
    
    # 4 Kolom dengan lebar yang sama (untuk KPI utama)
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(label="Total Responden", value=f"{total_responses}")
    
    with col2:
        st.metric(label="Skor Kepuasan Rata-rata (1-5)", value=f"{avg_satisfaction:.2f}",
                  delta_color="off")
    
    with col3:
        # Menampilkan Metrik Paling Puas
        st.metric(label=f"Metrik Paling Puas ({best_metric['Mean']:.2f})", 
                  value=best_metric['Question'],
                  delta=f"Positive: {best_metric['% Positive (4-5)']:.1f}%")
        
    with col4:
        # Menampilkan Metrik Paling Kritis
        st.metric(label=f"Metrik Paling Kritis ({worst_metric['Mean']:.2f})", 
                   value=worst_metric['Question'],
                   delta=f"Positive: {worst_metric['% Positive (4-5)']:.1f}%",
                   delta_color="inverse")
    
    st.markdown("---") # Garis pemisah setelah baris KPI


def create_overview_charts(df_filtered, stats_df, rit_df):
    """Menampilkan chart utama di Dashboard Overview."""
    st.markdown("### 📈 Strategic Insights")
    
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        st.markdown("#### Ranking Metrik Kepuasan")
        fig_ranking = plot_top_ranking_bar(stats_df)
        st.plotly_chart(fig_ranking, use_container_width=True, key="overview_ranking_chart")
        
    with col_right:
        st.markdown("#### Item Impact on Internal Consistency")
        fig_impact = plot_impact_bar_chart(rit_df)
        st.plotly_chart(fig_impact, use_container_width=True, key="overview_impact_chart")
        
    st.markdown("---")


# =======================================================
# --- MAIN APP EXECUTION ---
# =======================================================

def main():
    st.title("🍽️ Canteen Satisfaction Survey Dashboard - SUMBER MAKMUR")
    st.markdown("### Comprehensive analysis of canteen satisfaction survey data")
    
    default_path = "Survey_Kepuasan_Kantin_Cleaned.csv"
    df = load_data(default_path)
    
    if df is None:
        st.stop()
    
    # Inisialisasi Analisis Dasar (di luar filter sidebar untuk memastikan data dasar tersedia)
    corr_matrix_full, rit_df_full = compute_correlations(df, 'spearman')
    stats_df_full, pie_df_full = analyze_likert(df)
    
    # =======================================================
    # START FILTER FAKULTAS & PRODI DI SIDEBAR (Dipertahankan)
    # =======================================================
    
    st.sidebar.header("🔍 Filters")
    filters = {'categorical': {}, 'numeric': {}}
    categorical_filter_cols = ['Fakultas', 'Prodi'] 
    df_temp = df.copy()

    for col in categorical_filter_cols:
        # FIX: Tambahkan key unik untuk setiap multiselect
        if col in df.columns:
            unique_values = sorted(df_temp[col].dropna().unique())
            selected = st.sidebar.multiselect(f"Select {col}", unique_values, key=f"sidebar_select_{col}")
            if selected:
                filters['categorical'][col] = selected
                df_temp = apply_filters(df, filters) 

    if st.sidebar.button("🔄 Reset Filters", key="reset_button"):
        st.rerun()

    filters['numeric'] = {}
    df_filtered = apply_filters(df, filters)
    
    # Hitung ulang metrik untuk data yang difilter
    corr_matrix, rit_df = compute_correlations(df_filtered, 'spearman')
    stats_df, pie_df = analyze_likert(df_filtered)
    
    # =======================================================
    # END FILTER FAKULTAS & PRODI DI SIDEBAR
    # =======================================================
    
    if len(df_filtered) == 0:
        st.warning("No data found after applying filters. Please reset filters.")
        st.stop()
    
    st.info(f"📊 Showing {len(df_filtered)} of {len(df)} rows after filtering")
    
    # Main content tabs
    tab0, tab1, tab2, tab3, tab4 = st.tabs([
        "🏠 Dashboard Overview", 
        "📈 Descriptive Statistics",
        "🔗 Correlations", 
        "😊 Satisfaction Analysis",
        "📊 Raw Data"
    ])
    
    # =======================================================
    # TAB 0: DASHBOARD OVERVIEW
    # =======================================================
    with tab0:
        # Narasi awal yang menarik
        avg_score = stats_df['Mean'].mean()
        
        if not rit_df.empty and not rit_df['Correlation'].isnull().all():
             high_corr_metric = METRIC_MAPPING.get(rit_df.sort_values(by='Correlation', key=abs, ascending=False).iloc[0]['Variable 1'], "Metrik Utama")
        else:
             high_corr_metric = "Data tidak tersedia"

        st.markdown(f"""
        <div class="lead-text">
        ### 👋 Insight Awal: Kepuasan Kantin UPN Veteran Jatim
        <p>Kantin merupakan pusat vital bagi mahasiswa UPN Veteran Jatim, bukan hanya sebagai tempat makan, tetapi juga area interaksi sosial dan istirahat. Mengingat pentingnya kantin dalam pengalaman kampus, kualitas layanan dan produknya secara langsung memengaruhi kesejahteraan dan kepuasan harian mahasiswa.</p>
        <p>Dari **{len(df_filtered)} responden**, ditemukan bahwa tingkat kepuasan rata-rata kantin saat ini berada di skor **{avg_score:.2f}/5.0**. 
        Analisis Item-Rest Correlation menunjukkan bahwa metrik **{high_corr_metric}** memiliki dampak korelasi internal tertinggi, menjadikannya faktor utama dalam mendefinisikan kepuasan responden secara keseluruhan.</p>
        <p>Dashboard ini menyajikan analisis mendalam untuk mengidentifikasi area yang membutuhkan peningkatan strategis. Temukan faktor apa yang paling memengaruhi persepsi mahasiswa!</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("---")
        
        create_overview_kpis(df_filtered, stats_df)
        create_overview_charts(df_filtered, stats_df, rit_df)
        st.markdown(f"""
        ### 👨‍💻 About This Analysis (Kelompok SUMBER MAKMUR)
        Dashboard ini dibuat untuk menganalisis data survei kepuasan Kantin UPNV Jatim.
        <br>
        **Anggota Kelompok:**
        - Mohammad Alimun Hakim – 24083010017
        - Galih Zaky Tristanaya – 24083010088
        - Indra Maulana R.F.Y – 24083010105
        """, unsafe_allow_html=True)
    
    # =======================================================
    # SISA TABS (DESCRIPTIVE, CORRELATION, ANALYSIS, RAW DATA)
    # =======================================================
    
    categorical_filter_cols = ['Fakultas', 'Prodi', 'kategori_pengeluaran', 'kategori_frekuensi_makan']
    
    with tab1:
        st.header("Descriptive Statistics")
        
        # Numeric statistics
        st.subheader("Numeric Variables")
        numeric_stats = describe_numeric(df_filtered)
        
        if not numeric_stats.empty:
            st.dataframe(numeric_stats, use_container_width=True)
            
            # Download button
            csv = numeric_stats.to_csv()
            st.download_button(
                label="📥 Download Numeric Stats (CSV)",
                data=csv,
                file_name="numeric_statistics.csv",
                mime="text/csv"
            )
            
            # Mendefinisikan kolom untuk visualisasi yang berbeda
            likert_cols_viz = LIKERT_COLS
            all_numeric_cols = list(numeric_stats.index)
            continuous_cols = [col for col in all_numeric_cols if col not in likert_cols_viz]
            
            
            # =======================================================
            # 1. Continuous Variable Distributions (Histogram)
            # =======================================================
            
            st.subheader("Continuous Variable Distributions (Histogram)")
            
            selected_continuous = st.selectbox(
                "Select Continuous column",
                [col for col in continuous_cols if col in df_filtered.columns], key="cont_select_tab1"
            )

            if selected_continuous:
                fig = plot_numeric_distribution(df_filtered, [selected_continuous], 'histogram') 
                if fig:
                    st.plotly_chart(fig, use_container_width=True, key="hist_tab1")
                st.markdown(f"""
                    <p style='font-size: 15px;'>
                    Histogram menunjukkan frekuensi responden berdasarkan nilai **{selected_continuous}**. 
                    Ini penting untuk melihat apakah data terpusat, menyebar, atau memiliki pencilan (outliers).
                    </p>
                    """, unsafe_allow_html=True)


            # 2. Satisfaction Scale Distributions (Bar Chart)
            st.subheader("Satisfaction Scale Distributions (Bar Chart)")

            selected_likert = st.selectbox(
                "Select Satisfaction Scale variable",
                [col for col in likert_cols_viz if col in df_filtered.columns], key="likert_select_tab1"
            )

            if selected_likert:
                fig = plot_categorical(df_filtered, selected_likert)
                if fig:
                    st.plotly_chart(fig, use_container_width=True, key="bar_tab1")
                st.markdown(f"""
                    <p style='font-size: 15px;'>
                    Bar chart ini memvisualisasikan distribusi respons skor (1 sampai 5) untuk **{METRIC_MAPPING.get(selected_likert, selected_likert)}**. 
                    Mode (skor yang paling sering dipilih) menunjukkan konsensus atau kecenderungan sentimen responden terhadap metrik tersebut.
                    </p>
                    """, unsafe_allow_html=True)
            
            # =======================================================
            
        # Categorical statistics
        st.subheader("Categorical Variables")
        
        cat_options = ['Fakultas', 'Prodi', 'kategori_pengeluaran', 'kategori_frekuensi_makan']
        available_cats = [col for col in cat_options if col in df_filtered.columns]
        
        if available_cats:
            selected_cat = st.selectbox("Select categorical variable (Nominal/Ordinal)", available_cats, key="cat_select_tab1")
            
            if selected_cat:
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    freq_table = df_filtered[selected_cat].value_counts().reset_index()
                    freq_table.columns = [selected_cat, 'Count']
                    freq_table['Percentage'] = (freq_table['Count'] / len(df_filtered) * 100).round(2)
                    st.dataframe(freq_table, use_container_width=True)
                
                with col2:
                    fig = plot_categorical(df_filtered, selected_cat)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True, key="cat_bar_tab1")
                st.markdown(f"""
                    <p style='font-size: 15px;'>
                    Tabel dan bar chart ini menunjukkan komposisi responden berdasarkan variabel kategori (**{selected_cat}**). 
                    Ini membantu memahami segmentasi data yang sedang dianalisis.
                    </p>
                    """, unsafe_allow_html=True)
    
    with tab2:
        st.header("Correlation Analysis")
        st.markdown("---") 

        corr_matrix, rit_df = compute_correlations(df_filtered, 'spearman')
        
        if not corr_matrix.empty and not rit_df.empty:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("Correlation Heatmap (Spearman Rank)")
                fig = plot_correlation_heatmap(corr_matrix)
                st.plotly_chart(fig, use_container_width=True, key="heatmap_tab2")
                st.markdown("""
                    <p style='font-size: 15px;'>
                    Heatmap menunjukkan korelasi Spearman ($ \\rho $) antara semua metrik kepuasan (tanpa skor agregat). 
                    Warna Biru Kuat menunjukkan korelasi positif tinggi (misal: Jika Rasa naik, Kenyamanan juga cenderung naik).
                    </p>
                    """, unsafe_allow_html=True)
            
            with col2:
                # Menghitung Top 5 Correlation (item-to-item) untuk ditampilkan di tabel kanan
                corr_pairs = []
                # Ambil nama metrik yang sudah dibersihkan dari fungsi heatmap
                clean_cols_map = {c: c.replace('skala_', '').replace('(1-5)', '').title() for c in corr_matrix.columns}
                
                # Hanya masukkan item-item Likert (tanpa skor total)
                likert_item_cols = [c for c in corr_matrix.columns if c not in ['skor_kepuasan', 'skala_antrean(1-5)']]
                
                for i in range(len(likert_item_cols)):
                    for j in range(i+1, len(likert_item_cols)):
                        metric1 = likert_item_cols[i]
                        metric2 = likert_item_cols[j]
                        corr_pairs.append({
                            'Metric 1': clean_cols_map[metric1],
                            'Metric 2': clean_cols_map[metric2],
                            'Correlation': corr_matrix.loc[metric1, metric2]
                        })
                
                top_item_corr_display = pd.DataFrame(corr_pairs).sort_values('Correlation', key=abs, ascending=False).head(5)
                
                if not top_item_corr_display.empty:
                    top_item_corr_display['Pair'] = top_item_corr_display['Metric 1'] + " & " + top_item_corr_display['Metric 2']
                    top_item_corr_display = top_item_corr_display[['Pair', 'Correlation']].reset_index(drop=True)
                    
                    st.subheader("Top Inter-Item Correlations")
                    st.dataframe(top_item_corr_display.round(3), use_container_width=True)
                    st.markdown("""
                        <p style='font-size: 15px;'>
                        Tabel ini menampilkan 5 pasang metrik layanan yang memiliki korelasi terkuat. 
                        Korelasi tinggi di sini menunjukkan metrik-metrik tersebut cenderung dipersepsikan secara serupa oleh responden.
                        </p>
                        """, unsafe_allow_html=True)
                else:
                    st.warning("Tidak ada korelasi antar item yang ditemukan.")
            
            csv = corr_matrix.to_csv()
            st.download_button(
                label="📥 Download Correlation Matrix (CSV)",
                data=csv,
                file_name="correlation_matrix_satisfaction_spearman.csv",
                mime="text/csv"
            )
            
            st.subheader("Metric Impact on Overall Satisfaction")
            
            fig_impact = plot_impact_bar_chart(rit_df) 
            st.plotly_chart(fig_impact, use_container_width=True, key="impact_bar_tab2")
            
            top_rit_metric_name = METRIC_MAPPING.get(rit_df.sort_values(by='Correlation', key=abs, ascending=False).iloc[0]['Variable 1'], "Metrik Utama")
            
            st.markdown(f"""
                <p style='font-size: 15px;'>
                Bar chart Item-Rest Correlation (RIT) ini mengukur seberapa kuat masing-masing metrik berkontribusi 
                terhadap konsistensi internal metrik kepuasan total. Metrik dengan RIT tertinggi (**{top_rit_metric_name}**)
                adalah faktor **paling berpengaruh** terhadap sentimen kepuasan keseluruhan.
                </p>
                """, unsafe_allow_html=True)
            
            # Tabel Ringkasan RIT (Dipindah ke bawah barchart)
            st.markdown("#### Ringkasan Item-Rest Correlation (RIT)")
            
            rit_summary_df = rit_df.copy()
            rit_summary_df['Metric'] = rit_summary_df['Variable 1'].map(METRIC_MAPPING)
            rit_summary_df = rit_summary_df[['Metric', 'Correlation', 'p_value', 'n']].rename(columns={'Correlation': 'RIT Rho', 'p_value': 'P-Value', 'n': 'N'})
            
            st.dataframe(rit_summary_df.sort_values('RIT Rho', ascending=False).round(3), use_container_width=True)
            st.markdown("""
                <p style='font-size: 15px;'>
                Tabel ini menyajikan nilai Item-Rest Correlation ($\rho$) yang digunakan untuk menilai validitas dan reliabilitas internal item. 
                Nilai $\\rho$ yang lebih tinggi menunjukkan item tersebut berkorelasi kuat dengan sisa dari skala tersebut.
                </p>
                """, unsafe_allow_html=True)
            
            st.subheader("Boxplot Distributions of Satisfaction Score by Spending Category")
            fig_boxplot = plot_satisfaction_boxplot(df_filtered, stats_df)
            st.plotly_chart(fig_boxplot, use_container_width=True, key="boxplot_tab2")
            st.markdown("""
                <p style='font-size: 15px;'>
                Boxplot ini memvisualisasikan sebaran skor kepuasan total di antara berbagai kategori pengeluaran (Rendah, Sedang, Tinggi).
                Ini membantu mengidentifikasi apakah terdapat perbedaan signifikan dalam tingkat kepuasan berdasarkan profil pengeluaran mahasiswa.
                </p>
                """, unsafe_allow_html=True)
            
        else:
            st.warning("Insufficient or invalid numeric data for correlation analysis in the filtered dataset.")
    
    with tab3:
        st.header("Satisfaction Analysis (Likert Scales)")
        
        stats_df, pie_df = analyze_likert(df_filtered)
        
        if stats_df is not None:
            
            if 'skor_kepuasan' in df_filtered.columns:
                overall_score = df_filtered['skor_kepuasan'].mean()
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h2 style="text-align: center;">Overall Satisfaction</h2>
                        <h1 style="text-align: center; font-size: 48px;">{overall_score:.2f}/5.0</h1>
                        <p style="text-align: center;">Based on Average Satisfaction Score</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    fig_gauge = create_gauge_chart(overall_score)
                    st.plotly_chart(fig_gauge, use_container_width=True, key="gauge_tab3")
                st.markdown("""
                    <p style='font-size: 15px;'>
                    Skor Kepuasan Keseluruhan yang dihitung dari rata-rata semua dimensi layanan. 
                    Angka di atas 4.0 menunjukkan kepuasan tinggi secara umum (berdasarkan skala 1-5).
                    </p>
                    """, unsafe_allow_html=True)
            
            if 'skor_kepuasan' in df_filtered.columns:
                st.subheader("Distribution of Overall Satisfaction Score")
                fig_hist_score = plot_numeric_distribution(df_filtered, ['skor_kepuasan'], 'histogram')
                st.plotly_chart(fig_hist_score, use_container_width=True, key="hist_score_tab3")
                st.markdown("""
                    <p style='font-size: 15px;'>
                    Histogram ini memvisualisasikan sebaran skor kepuasan rata-rata yang dilaporkan oleh responden. 
                    Idealnya, distribusi terpusat pada skor tinggi (4 atau 5).
                    </p>
                    """, unsafe_allow_html=True)
            
            st.subheader("Satisfaction Metrics Summary")
            stats_df_display = stats_df.round(2)
            st.dataframe(stats_df_display, use_container_width=True)
            st.markdown("""
                <p style='font-size: 15px;'>
                Tabel ringkasan ini membandingkan Mean, Median, dan Persentase Positif (% 4-5) di semua metrik. 
                Perhatikan metrik dengan Mean rendah dan % Positive yang kecil untuk identifikasi area krisis.
                </p>
                """, unsafe_allow_html=True)

            st.subheader("Positive vs. Neutral/Negative Response Distribution")
            
            questions = stats_df['Question'].unique()
            
            selected_question = st.selectbox(
                "Select a Satisfaction Metric to view its Positive Response Distribution",
                questions
            )
            
            if selected_question:
                fig_pie = plot_single_pie_chart(pie_df, selected_question)
                if fig_pie:
                    st.plotly_chart(fig_pie, use_container_width=True, key="pie_tab3")
                st.markdown(f"""
                    <p style='font-size: 15px;'>
                    Pie chart menunjukkan proporsi responden yang memberikan skor Puas (Biru, 4-5) 
                    dibandingkan Netral/Tidak Puas (Merah, 1-3) untuk metrik **{selected_question}**. 
                    Target perbaikan harus fokus pada metrik dengan proporsi 'Neutral/Negative' (Merah) yang besar.
                    </p>
                    """, unsafe_allow_html=True)
            
            ranking_df = stats_df.sort_values('Mean', ascending=True)
            
            fig_ranking = plot_top_ranking_bar(stats_df)
            st.plotly_chart(fig_ranking, use_container_width=True, key="ranking_tab3")
            st.markdown("""
                <p style='font-size: 15px;'>
                Bar chart ranking menunjukkan metrik layanan mana yang secara rata-rata (Mean) 
                memiliki skor terendah (berada di kiri) dan tertinggi (berada di kanan). 
                Metrik terendah adalah prioritas utama untuk perbaikan operasional.
                </p>
                """, unsafe_allow_html=True)
        else:
            st.warning("No Likert scale columns found in the dataset")
    
    with tab4:
        st.header("Raw Data View")
        
        st.subheader(f"Filtered Dataset ({len(df_filtered)} rows)")
        st.dataframe(df_filtered, use_container_width=True)
        
        csv = df_filtered.to_csv(index=False)
        st.download_button(
            label="📥 Download Filtered Data (CSV)",
            data=csv,
            file_name="filtered_data.csv",
            mime="text/csv"
        )
    
    st.header("📖 Help & Documentation")
    
    with st.expander("Getting Started", expanded=True):
        st.markdown(f"""
        **Welcome to the Canteen Satisfaction Survey Dashboard!**
        
        This dashboard provides comprehensive analysis of survey data, automatically loaded from **`{default_path}`**.
        
        The main features are:
        
        1. **Data Loading**: Automatically loads from the default path.
        2. **Filtering**: Gunakan sidebar untuk **memfilter data berdasarkan Fakultas dan Prodi**.
        3. **Visualizations**: Interactive Plotly charts with hover, zoom, and export capabilities.
        """)
    
    with st.expander("Features Guide"):
        st.markdown("""
        **📈 Descriptive Statistics**
        - View summary statistics for all numeric variables
        - Visualize distributions with histograms or bar charts, depending on variable type.
        - Analyze categorical variable frequencies
        
        **🔗 Correlations**
        - Compute **Spearman Rank Correlation** untuk mengukur hubungan monotonik antara variabel skala.
        - **Lihat Metrik Dampak** (Item-Rest Correlation) pada konsistensi internal metrik (RIT).
        - Interactive heatmap visualization
        - Identify top correlated pairs
        - Create scatter plots with optional grouping
        
        **😊 Satisfaction Analysis**
        - Analyze Likert scale responses (1-5)
        - Overall satisfaction KPI with gauge chart
        - **Histogram Distribusi Skor Kepuasan**
        - **Lihat distribusi respons Positif vs. Netral/Negatif (Pie Chart) untuk setiap metrik.**
        - Rankings by average scores
        
        **📊 Raw Data**
        - View and download filtered dataset
        - Export functionality for all analyses
        """)
    
    with st.expander("Tips & Tricks"):
        st.markdown("""
        - 🎯 **Gunakan filter Fakultas dan Prodi di sidebar** untuk menganalisis sub-set data spesifik.
        - 📸 Plotly charts can be saved as images using the camera icon
        - 🔍 Hover over charts for detailed information
        - 📥 Download results for further analysis
        - 🔄 Gunakan tombol **'Reset Filters'** di sidebar untuk menampilkan data lengkap.
        """)
    
    # Footer
    st.markdown("""
    <div class="footer">
    <p>Canteen Satisfaction Survey Dashboard v1.0 | Built with Streamlit & Plotly</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()