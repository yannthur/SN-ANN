import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
from datetime import timedelta

# ==========================================
# 1. CONFIGURATION ET CONSTANTES
# ==========================================
st.set_page_config(
    page_title="Prédiction Netflix - Ensemble AI", 
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📈"
)

# CSS personnalisé pour un meilleur design
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #E50914 0%, #B20710 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .subtitle {
        text-align: center;
        color: #888;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 0.5rem 0;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #E50914 0%, #B20710 100%);
        color: white;
        font-weight: bold;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        border: none;
        font-size: 1.1rem;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(229, 9, 20, 0.4);
    }
    .prediction-card {
        background: rgba(255, 255, 255, 0.05);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #E50914;
        margin: 1rem 0;
    }
    .info-box {
        background: rgba(102, 126, 234, 0.1);
        padding: 1rem;
        border-radius: 8px;
        border-left: 3px solid #667eea;
        margin: 1rem 0;
    }
    .model-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.85rem;
        font-weight: bold;
        margin: 0.2rem;
    }
    .bilstm-badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    .gru-badge {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
    }
    .ensemble-badge {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# En-tête
st.markdown('<h1 class="main-header">📈 Prédiction Netflix (NFLX)</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Ensemble d\'IA - Modèles Bi-LSTM & GRU</p>', unsafe_allow_html=True)

# Paramètres (Doivent correspondre EXACTEMENT à ceux de l'entraînement)
SEQ_LENGTH = 60
HIDDEN_SIZE_1 = 128
HIDDEN_SIZE_2 = 211
NUM_LAYERS = 2
INPUT_SIZE = 1
OUTPUT_SIZE = 1
TICKER = "NFLX"

# ==========================================
# 2. DÉFINITION DES MODÈLES
# ==========================================
class BiLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(BiLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, 
            batch_first=True, bidirectional=True, dropout=dropout
        )
        self.fc = nn.Linear(hidden_size * 2, output_size)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout
        )
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out

# ==========================================
# 3. FONCTIONS UTILITAIRES
# ==========================================
@st.cache_resource
def load_trained_models(bilstm_path, gru_path, device):
    """Charge les deux modèles et leurs poids."""
    # Modèle Bi-LSTM
    bilstm_model = BiLSTMModel(INPUT_SIZE, HIDDEN_SIZE_1, NUM_LAYERS, OUTPUT_SIZE)
    bilstm_model.load_state_dict(torch.load(bilstm_path, map_location=torch.device('cpu')))
    bilstm_model.to(device)
    bilstm_model.eval()
    
    # Modèle GRU
    gru_model = GRUModel(INPUT_SIZE, HIDDEN_SIZE_2, NUM_LAYERS, OUTPUT_SIZE)
    gru_model.load_state_dict(torch.load(gru_path, map_location=torch.device('cpu')))
    gru_model.to(device)
    gru_model.eval()
    
    return bilstm_model, gru_model

@st.cache_data(ttl=3600)
def load_data():
    """Charge les données historiques via Yahoo Finance avec retry."""
    import time
    
    for attempt in range(3):
        try:
            data = yf.download(TICKER, period="2y", interval="1d", progress=False)
            
            if data.empty:
                if attempt < 2:
                    time.sleep(5 * (attempt + 1))  # wait 5s, then 10s
                    continue
                else:
                    return pd.DataFrame()  # return empty after 3 tries
            
            if isinstance(data.columns, pd.MultiIndex):
                data = data.xs('Close', level=0, axis=1) if 'Close' in data.columns.get_level_values(0) else data
            elif 'Close' in data.columns:
                data = data[['Close']]
            
            if isinstance(data, pd.DataFrame) and data.shape[1] > 1:
                data = data.iloc[:, 0].to_frame()
            
            data.columns = ['Close']
            return data
            
        except Exception as e:
            if attempt < 2:
                time.sleep(5 * (attempt + 1))
            else:
                st.error(f"Impossible de télécharger les données après 3 tentatives: {e}")
                return pd.DataFrame()
    
    return pd.DataFrame()

# ==========================================
# 4. SIDEBAR - PARAMÈTRES
# ==========================================
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/0/08/Netflix_2015_logo.svg", width=200)
    st.markdown("---")
    
    st.markdown("### ⚙️ Paramètres de Prédiction")
    
    # Sélection du nombre de jours
    prediction_days = st.slider(
        "Nombre de jours à prédire",
        min_value=1,
        max_value=30,
        value=15,
        help="Choisissez le nombre de jours ouvrés à prédire (1-30 jours)"
    )
    
    # Sélection de la période d'historique à afficher
    history_days = st.selectbox(
        "Historique à afficher",
        options=[30, 60, 90, 180, 365],
        index=2,
        help="Nombre de jours d'historique à afficher sur le graphique"
    )
    
    st.markdown("---")
    st.markdown("### 🤖 Modèles Utilisés")
    # Bi-LSTM
    st.markdown('<span class="model-badge bilstm-badge">🔵 Bi-LSTM</span>', unsafe_allow_html=True)
    st.caption("Réseau bidirectionnel à mémoire longue")

    # GRU
    st.markdown('<span class="model-badge gru-badge">🟣 GRU</span>', unsafe_allow_html=True)
    st.caption("Unité récurrente à portes")

    # Ensemble
    st.markdown('<span class="model-badge ensemble-badge">💎 Ensemble</span>', unsafe_allow_html=True)
    st.caption("Moyenne pondérée des deux modèles")
    
    st.markdown("---")
    st.markdown("### 📊 Architecture")
    st.info(f"""
    **Couches cachées:** {HIDDEN_SIZE_1}  
    **Nombre de couches:** {NUM_LAYERS}  
    **Séquence d'entrée:** {SEQ_LENGTH} jours
    """)
    
    st.markdown("---")
    st.markdown("### ℹ️ À propos")
    st.caption("""
    Cette application combine deux modèles d'IA (Bi-LSTM et GRU) pour améliorer 
    la précision des prédictions. L'approche "ensemble" réduit le risque d'erreur 
    en moyennant les prédictions des deux modèles.
    """)
    st.warning("⚠️ Ces prédictions sont à titre informatif uniquement et ne constituent pas des conseils financiers.")

# ==========================================
# 5. LOGIQUE PRINCIPALE
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Chargement
with st.spinner('🔄 Chargement des modèles et des données...'):
    try:
        bilstm_model, gru_model = load_trained_models(
            'best_bilstm_nflx.pth', 
            'best_gru_nflx.pth', 
            device
        )
        df = load_data()
        st.success("✅ Modèles Bi-LSTM et GRU chargés avec succès!")
    except FileNotFoundError as e:
        st.error(f"❌ Fichier manquant: {str(e)}")
        st.info("Assurez-vous que 'best_bilstm_nflx.pth' et 'best_gru_nflx.pth' sont présents.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement: {str(e)}")
        st.stop()

# Guard: stop cleanly if data failed to load
if df is None or df.empty:
    st.error("❌ Impossible de charger les données depuis Yahoo Finance (rate limit ou erreur réseau).")
    st.warning("💡 Streamlit Cloud partage des IPs avec d'autres apps — Yahoo Finance bloque parfois ces requêtes. Relancez dans quelques minutes.")
    st.stop()

# Safe to proceed
last_date = df.index[-1]
last_price = df['Close'].iloc[-1]

# Calcul de la variation sur 24h et 7 jours
price_change_1d = ((df['Close'].iloc[-1] - df['Close'].iloc[-2]) / df['Close'].iloc[-2] * 100) if len(df) > 1 else 0
price_change_7d = ((df['Close'].iloc[-1] - df['Close'].iloc[-7]) / df['Close'].iloc[-7] * 100) if len(df) > 7 else 0

st.markdown("### 📊 Données Actuelles du Marché")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Dernier Prix", 
        f"${last_price:.2f}",
        delta=f"{price_change_1d:.2f}% (24h)"
    )

with col2:
    st.metric(
        "Variation 7 jours",
        f"{price_change_7d:.2f}%",
        delta=None
    )

with col3:
    st.metric(
        "Volume",
        f"{df['Close'].iloc[-1]:.0f}",
        delta=None
    )

with col4:
    st.metric(
        "Dernière Mise à Jour",
        last_date.strftime('%d/%m/%Y')
    )

st.markdown("---")

# ==========================================
# 6. PRÉDICTION AVEC LES DEUX MODÈLES
# ==========================================
st.markdown(f"### 🔮 Lancer la Prédiction ({prediction_days} jours)")

col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    predict_button = st.button('🚀 Générer les Prédictions (Ensemble)', use_container_width=True)

if predict_button:
    
    with st.spinner(f'🤖 Génération des prédictions pour les {prediction_days} prochains jours...'):
        # 1. Normalisation
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaled_data = scaler.fit_transform(df[['Close']].values)

        # 2. Préparation de la dernière séquence
        last_sequence = scaled_data[-SEQ_LENGTH:]
        initial_seq = torch.tensor(last_sequence, dtype=torch.float32).view(1, SEQ_LENGTH, 1).to(device)
        
        # 3. Barre de progression
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # 4. Prédictions Bi-LSTM
        status_text.text("🔵 Prédiction avec Bi-LSTM...")
        progress_bar.progress(0.33)
        bilstm_predictions = predict_future(bilstm_model, initial_seq, prediction_days, device)
        
        # 5. Prédictions GRU
        status_text.text("🟣 Prédiction avec GRU...")
        progress_bar.progress(0.66)
        gru_predictions = predict_future(gru_model, initial_seq, prediction_days, device)
        
        # 6. Calcul de la moyenne (Ensemble)
        status_text.text("🔷 Calcul de l'ensemble (moyenne)...")
        progress_bar.progress(1.0)
        ensemble_predictions = [(b + g) / 2 for b, g in zip(bilstm_predictions, gru_predictions)]
        
        progress_bar.empty()
        status_text.empty()

        # 7. Inversion de la normalisation
        bilstm_predictions_inv = scaler.inverse_transform(np.array(bilstm_predictions).reshape(-1, 1))
        gru_predictions_inv = scaler.inverse_transform(np.array(gru_predictions).reshape(-1, 1))
        ensemble_predictions_inv = scaler.inverse_transform(np.array(ensemble_predictions).reshape(-1, 1))

        # 8. Création des dates futures
        future_dates = pd.bdate_range(start=last_date + timedelta(days=1), periods=prediction_days)
        
        # DataFrame avec les trois prédictions
        df_future = pd.DataFrame({
            'Bi-LSTM': bilstm_predictions_inv.flatten(),
            'GRU': gru_predictions_inv.flatten(),
            'Ensemble': ensemble_predictions_inv.flatten()
        }, index=future_dates)
        
        # ==========================================
        # 9. VISUALISATION COMPARATIVE
        # ==========================================
        st.markdown("---")
        st.markdown("### 📈 Résultats de la Prédiction - Comparaison des Modèles")
        
        # Métriques comparatives
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<span class="model-badge bilstm-badge">Bi-LSTM</span>', unsafe_allow_html=True)
            bilstm_variation = ((df_future['Bi-LSTM'].iloc[-1] - last_price) / last_price) * 100
            st.metric(
                f"Prix J+{prediction_days}", 
                f"${df_future['Bi-LSTM'].iloc[-1]:.2f}",
                delta=f"{bilstm_variation:.2f}%"
            )
        
        with col2:
            st.markdown('<span class="model-badge gru-badge">GRU</span>', unsafe_allow_html=True)
            gru_variation = ((df_future['GRU'].iloc[-1] - last_price) / last_price) * 100
            st.metric(
                f"Prix J+{prediction_days}", 
                f"${df_future['GRU'].iloc[-1]:.2f}",
                delta=f"{gru_variation:.2f}%"
            )
        
        with col3:
            st.markdown('<span class="model-badge ensemble-badge">Ensemble</span>', unsafe_allow_html=True)
            ensemble_variation = ((df_future['Ensemble'].iloc[-1] - last_price) / last_price) * 100
            st.metric(
                f"Prix J+{prediction_days}", 
                f"${df_future['Ensemble'].iloc[-1]:.2f}",
                delta=f"{ensemble_variation:.2f}%"
            )

        # Graphique interactif Plotly avec les 3 modèles
        st.markdown("### 📉 Visualisation Interactive - Comparaison des Modèles")
        
        fig = go.Figure()

        # Historique récent
        recent_df = df.iloc[-history_days:]
        fig.add_trace(go.Scatter(
            x=recent_df.index, 
            y=recent_df['Close'], 
            mode='lines', 
            name='Historique Réel',
            line=dict(color='#888888', width=2),
            fill='tozeroy',
            fillcolor='rgba(136, 136, 136, 0.1)'
        ))

        # Prédictions Bi-LSTM
        fig.add_trace(go.Scatter(
            x=df_future.index, 
            y=df_future['Bi-LSTM'], 
            mode='lines+markers', 
            name='Bi-LSTM',
            line=dict(color='#667eea', width=2, dash='dot'),
            marker=dict(size=5, symbol='circle')
        ))
        
        # Prédictions GRU
        fig.add_trace(go.Scatter(
            x=df_future.index, 
            y=df_future['GRU'], 
            mode='lines+markers', 
            name='GRU',
            line=dict(color='#f5576c', width=2, dash='dot'),
            marker=dict(size=5, symbol='square')
        ))
        
        # Prédictions Ensemble (Moyenne)
        fig.add_trace(go.Scatter(
            x=df_future.index, 
            y=df_future['Ensemble'], 
            mode='lines+markers', 
            name='Ensemble (Moyenne)',
            line=dict(color='#00f2fe', width=3),
            marker=dict(size=7, symbol='diamond')
        ))
        
        # Liens visuels
        fig.add_trace(go.Scatter(
            x=[recent_df.index[-1], df_future.index[0]],
            y=[recent_df['Close'].iloc[-1], df_future['Bi-LSTM'].iloc[0]],
            mode='lines',
            showlegend=False,
            line=dict(color='#667eea', width=1, dash='dot'),
            opacity=0.5
        ))
        
        fig.add_trace(go.Scatter(
            x=[recent_df.index[-1], df_future.index[0]],
            y=[recent_df['Close'].iloc[-1], df_future['GRU'].iloc[0]],
            mode='lines',
            showlegend=False,
            line=dict(color='#f5576c', width=1, dash='dot'),
            opacity=0.5
        ))
        
        fig.add_trace(go.Scatter(
            x=[recent_df.index[-1], df_future.index[0]],
            y=[recent_df['Close'].iloc[-1], df_future['Ensemble'].iloc[0]],
            mode='lines',
            showlegend=False,
            line=dict(color='#00f2fe', width=2),
            opacity=0.7
        ))

        fig.update_layout(
            title={
                'text': f"Projection du prix Netflix - Comparaison Bi-LSTM vs GRU vs Ensemble ({prediction_days} jours)",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18}
            },
            xaxis_title="Date",
            yaxis_title="Prix ($)",
            template="plotly_dark",
            hovermode="x unified",
            height=650,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                bgcolor="rgba(0,0,0,0.5)"
            )
        )

        st.plotly_chart(fig, use_container_width=True)

        # Analyse détaillée par modèle
        st.markdown("### 🎯 Analyse Comparative des Tendances")
        
        col1, col2, col3 = st.columns(3)
        
        def get_trend_info(variation):
            if variation > 0:
                return "📈", "haussière", "green"
            else:
                return "📉", "baissière", "red"
        
        with col1:
            emoji, tendance, color = get_trend_info(bilstm_variation)
            st.markdown(f"""
            <div class="info-box" style="border-left-color: #667eea;">
                <h4>{emoji} Bi-LSTM - Tendance {tendance.capitalize()}</h4>
                <p style='font-size: 1.3rem; color: {color}; font-weight: bold;'>
                    {bilstm_variation:+.2f}%
                </p>
                <p style='font-size: 0.9rem;'>Prix final: ${df_future['Bi-LSTM'].iloc[-1]:.2f}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            emoji, tendance, color = get_trend_info(gru_variation)
            st.markdown(f"""
            <div class="info-box" style="border-left-color: #f5576c;">
                <h4>{emoji} GRU - Tendance {tendance.capitalize()}</h4>
                <p style='font-size: 1.3rem; color: {color}; font-weight: bold;'>
                    {gru_variation:+.2f}%
                </p>
                <p style='font-size: 0.9rem;'>Prix final: ${df_future['GRU'].iloc[-1]:.2f}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            emoji, tendance, color = get_trend_info(ensemble_variation)
            st.markdown(f"""
            <div class="info-box" style="border-left-color: #00f2fe;">
                <h4>{emoji} Ensemble - Tendance {tendance.capitalize()}</h4>
                <p style='font-size: 1.3rem; color: {color}; font-weight: bold;'>
                    {ensemble_variation:+.2f}%
                </p>
                <p style='font-size: 0.9rem;'>Prix final: ${df_future['Ensemble'].iloc[-1]:.2f}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Convergence/Divergence des modèles
        st.markdown("### 🔍 Analyse de Convergence")
        
        divergence = abs(df_future['Bi-LSTM'].iloc[-1] - df_future['GRU'].iloc[-1])
        divergence_pct = (divergence / df_future['Ensemble'].iloc[-1]) * 100
        
        if divergence_pct < 2:
            convergence_status = "🟢 Forte convergence"
            convergence_msg = "Les deux modèles sont très alignés, renforçant la fiabilité de la prédiction."
        elif divergence_pct < 5:
            convergence_status = "🟡 Convergence modérée"
            convergence_msg = "Les modèles montrent un accord raisonnable avec quelques divergences."
        else:
            convergence_status = "🔴 Divergence significative"
            convergence_msg = "Les modèles présentent des prédictions divergentes, la prudence est recommandée."
        
        st.markdown(f"""
        <div class="prediction-card">
            <h4>{convergence_status}</h4>
            <p><b>Écart entre les modèles:</b> ${divergence:.2f} ({divergence_pct:.2f}%)</p>
            <p>{convergence_msg}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Tableau des valeurs comparatif
        st.markdown("### 📋 Données Détaillées - Comparaison Complète")
        with st.expander("📊 Voir le tableau complet des prédictions"):
            df_display = df_future.copy()
            df_display['Date'] = df_display.index.strftime('%d/%m/%Y')
            df_display['Bi-LSTM ($)'] = df_display['Bi-LSTM'].apply(lambda x: f"${x:.2f}")
            df_display['GRU ($)'] = df_display['GRU'].apply(lambda x: f"${x:.2f}")
            df_display['Ensemble ($)'] = df_display['Ensemble'].apply(lambda x: f"${x:.2f}")
            df_display['Écart Bi-LSTM/GRU'] = abs(df_future['Bi-LSTM'] - df_future['GRU']).apply(lambda x: f"${x:.2f}")
            
            st.dataframe(
                df_display[['Date', 'Bi-LSTM ($)', 'GRU ($)', 'Ensemble ($)', 'Écart Bi-LSTM/GRU']],
                use_container_width=True,
                height=400
            )
        
        # Statistiques supplémentaires
        st.markdown("### 📊 Statistiques Comparatives")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div class="info-box">
                <h4>📈 Statistiques de l'Ensemble</h4>
                <ul>
                    <li><b>Prix moyen prédit:</b> ${df_future['Ensemble'].mean():.2f}</li>
                    <li><b>Maximum prédit:</b> ${df_future['Ensemble'].max():.2f}</li>
                    <li><b>Minimum prédit:</b> ${df_future['Ensemble'].min():.2f}</li>
                    <li><b>Volatilité:</b> {(df_future['Ensemble'].std() / df_future['Ensemble'].mean() * 100):.2f}%</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="info-box">
                <h4>🎯 Concordance des Modèles</h4>
                <ul>
                    <li><b>Écart moyen:</b> ${abs(df_future['Bi-LSTM'] - df_future['GRU']).mean():.2f}</li>
                    <li><b>Écart maximum:</b> ${abs(df_future['Bi-LSTM'] - df_future['GRU']).max():.2f}</li>
                    <li><b>Corrélation:</b> {df_future['Bi-LSTM'].corr(df_future['GRU']):.4f}</li>
                    <li><b>Tendance commune:</b> {('Oui ✅' if bilstm_variation * gru_variation > 0 else 'Non ⚠️')}</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888; padding: 2rem 0;'>
    <p>🤖 Développé avec Streamlit & PyTorch | Modèles Bi-LSTM & GRU</p>
    <p style='font-size: 0.9rem;'>
        ⚠️ Avertissement: Les prédictions financières comportent des risques. 
        Cette application est destinée à des fins éducatives uniquement.
    </p>
</div>
""", unsafe_allow_html=True)
