import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
from datetime import timedelta
import time

# ==========================================
# 1. CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="NFLX · Prédiction IA",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📈"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500;600&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0A0A0A;
    color: #E5E5E5;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem; max-width: 1400px; }

[data-testid="stSidebar"] {
    background-color: #111111;
    border-right: 1px solid #1E1E1E;
}
[data-testid="stSidebar"] .block-container { padding: 2rem 1.5rem; }

.wordmark {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.6rem;
    letter-spacing: 0.12em;
    color: #E50914;
    margin-bottom: 0.25rem;
}
.wordmark-sub {
    font-size: 0.7rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #555;
    margin-bottom: 2rem;
}

.hero { border-bottom: 1px solid #1E1E1E; padding-bottom: 2rem; margin-bottom: 2.5rem; }
.hero-ticker {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 4.5rem;
    letter-spacing: 0.04em;
    line-height: 1;
    color: #FFFFFF;
    margin: 0;
}
.hero-ticker span { color: #E50914; }
.hero-desc {
    font-size: 0.85rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #555;
    margin-top: 0.4rem;
}

.kpi-row {
    display: flex;
    gap: 1px;
    margin-bottom: 2.5rem;
    background: #1E1E1E;
    border-radius: 4px;
    overflow: hidden;
}
.kpi-cell {
    flex: 1;
    background: #111111;
    padding: 1.2rem 1.5rem;
    display: flex;
    flex-direction: column;
    gap: 0.35rem;
}
.kpi-label {
    font-size: 0.68rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #555;
}
.kpi-value {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2rem;
    letter-spacing: 0.03em;
    color: #FFFFFF;
    line-height: 1;
}
.kpi-delta-up   { font-size: 0.78rem; color: #46D369; font-weight: 500; }
.kpi-delta-down { font-size: 0.78rem; color: #E50914; font-weight: 500; }
.kpi-neutral    { font-size: 0.78rem; color: #888;    font-weight: 400; }

.hr { border: none; border-top: 1px solid #1E1E1E; margin: 2rem 0; }

.section-label {
    font-size: 0.68rem;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: #555;
    margin-bottom: 1rem;
}

.pill-row { display: flex; gap: 0.5rem; flex-wrap: wrap; margin-bottom: 0.5rem; }
.pill {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 2px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}
.pill-bilstm  { background: rgba(70,211,105,0.12); color: #46D369; border: 1px solid rgba(70,211,105,0.25); }
.pill-gru     { background: rgba(229,9,20,0.12);   color: #E50914; border: 1px solid rgba(229,9,20,0.25); }
.pill-ensemble{ background: rgba(255,255,255,0.06); color: #CCCCCC; border: 1px solid rgba(255,255,255,0.12); }

.stButton > button {
    background: #E50914 !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 3px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.14em !important;
    text-transform: uppercase !important;
    padding: 0.7rem 1.6rem !important;
    width: 100% !important;
    transition: background 0.15s ease, transform 0.1s ease !important;
    box-shadow: none !important;
}
.stButton > button:hover {
    background: #F40612 !important;
    transform: translateY(-1px) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

[data-testid="stSlider"] > div > div > div { background: #E50914 !important; }
[data-testid="stSlider"] > div > div > div > div { background: #E50914 !important; }

[data-testid="stSelectbox"] > div > div {
    background: #111111 !important;
    border: 1px solid #2A2A2A !important;
    border-radius: 3px !important;
    color: #E5E5E5 !important;
}

.result-panel {
    background: #111111;
    border: 1px solid #1E1E1E;
    border-radius: 4px;
    padding: 1.5rem;
    height: 100%;
}
.result-panel-accent-green { border-top: 2px solid #46D369; }
.result-panel-accent-red   { border-top: 2px solid #E50914; }
.result-panel-accent-white { border-top: 2px solid #CCCCCC; }
.result-model-name {
    font-size: 0.65rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #555;
    margin-bottom: 0.6rem;
}
.result-price {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2.4rem;
    color: #FFFFFF;
    line-height: 1;
    margin-bottom: 0.4rem;
}
.result-delta-up   { font-size: 0.85rem; color: #46D369; font-weight: 600; }
.result-delta-down { font-size: 0.85rem; color: #E50914; font-weight: 600; }

.convergence-box {
    background: #111111;
    border: 1px solid #1E1E1E;
    border-radius: 4px;
    padding: 1.25rem 1.5rem;
    display: flex;
    align-items: center;
    gap: 1rem;
    margin: 1.5rem 0;
}
.conv-indicator { font-size: 1.4rem; }
.conv-title { font-size: 0.72rem; letter-spacing: 0.15em; text-transform: uppercase; color: #888; margin-bottom: 0.2rem; }
.conv-msg   { font-size: 0.85rem; color: #CCCCCC; }

.stats-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1px;
    background: #1E1E1E;
    border-radius: 4px;
    overflow: hidden;
    margin-top: 1.5rem;
}
.stats-cell { background: #111111; padding: 1rem 1.25rem; }
.stats-key  { font-size: 0.68rem; letter-spacing: 0.16em; text-transform: uppercase; color: #555; margin-bottom: 0.3rem; }
.stats-val  { font-size: 1rem; font-weight: 500; color: #E5E5E5; }

.trend-card {
    background: #111111;
    border: 1px solid #1E1E1E;
    border-radius: 4px;
    padding: 1.25rem 1.5rem;
    height: 100%;
}
.trend-model    { font-size: 0.65rem; letter-spacing: 0.2em; text-transform: uppercase; color: #555; margin-bottom: 0.5rem; }
.trend-pct-up   { font-family: 'Bebas Neue', sans-serif; font-size: 2rem; color: #46D369; line-height: 1; }
.trend-pct-down { font-family: 'Bebas Neue', sans-serif; font-size: 2rem; color: #E50914; line-height: 1; }
.trend-label    { font-size: 0.75rem; color: #666; margin-top: 0.3rem; }
.trend-final    { font-size: 0.8rem; color: #888; margin-top: 0.6rem; font-weight: 500; }

[data-testid="stDataFrame"] { border: 1px solid #1E1E1E; border-radius: 4px; overflow: hidden; }
[data-testid="stDataFrame"] table { background: #0A0A0A; }
[data-testid="stDataFrame"] th {
    background: #111111 !important; color: #555 !important;
    font-size: 0.7rem !important; letter-spacing: 0.15em !important;
    text-transform: uppercase !important; border-bottom: 1px solid #1E1E1E !important;
}
[data-testid="stDataFrame"] td {
    color: #CCC !important; font-size: 0.85rem !important;
    border-bottom: 1px solid #141414 !important;
}

[data-testid="stAlert"] {
    background: #111111 !important; border: 1px solid #2A2A2A !important;
    border-radius: 3px !important; color: #AAA !important; font-size: 0.82rem !important;
}
[data-testid="stExpander"] {
    background: #111111 !important; border: 1px solid #1E1E1E !important; border-radius: 4px !important;
}
.stSpinner > div { border-color: #E50914 !important; }
[data-testid="stProgressBar"] > div > div { background: #E50914 !important; }

.footer {
    margin-top: 4rem;
    padding-top: 2rem;
    border-top: 1px solid #1E1E1E;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.footer-brand   { font-family: 'Bebas Neue', sans-serif; font-size: 1rem; letter-spacing: 0.1em; color: #333; }
.footer-warning { font-size: 0.72rem; color: #444; max-width: 480px; text-align: right; }

.error-box {
    background: #1A0A0A;
    border: 1px solid #3A1A1A;
    border-left: 3px solid #E50914;
    border-radius: 4px;
    padding: 1.25rem 1.5rem;
    margin: 1rem 0;
}
.error-box-title { font-size: 0.72rem; letter-spacing: 0.18em; text-transform: uppercase; color: #E50914; margin-bottom: 0.5rem; }
.error-box-msg   { font-size: 0.85rem; color: #888; line-height: 1.6; }
</style>
""", unsafe_allow_html=True)


# ==========================================
# 2. CONSTANTES
# ==========================================
SEQ_LENGTH    = 60
HIDDEN_SIZE_1 = 128
HIDDEN_SIZE_2 = 211
NUM_LAYERS    = 2
INPUT_SIZE    = 1
OUTPUT_SIZE   = 1
TICKER        = "NFLX"

# Nombre max de tentatives pour yfinance
MAX_RETRIES   = 3
RETRY_DELAY   = 4   # secondes entre chaque tentative


# ==========================================
# 3. MODÈLES
# ==========================================
class BiLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, bidirectional=True, dropout=dropout
        )
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])


class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.gru = nn.GRU(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        return self.fc(out[:, -1, :])


# ==========================================
# 4. FONCTIONS
# ==========================================
@st.cache_resource
def load_trained_models(bilstm_path, gru_path, device):
    bilstm = BiLSTMModel(INPUT_SIZE, HIDDEN_SIZE_1, NUM_LAYERS, OUTPUT_SIZE)
    bilstm.load_state_dict(torch.load(bilstm_path, map_location='cpu'))
    bilstm.to(device).eval()
    gru = GRUModel(INPUT_SIZE, HIDDEN_SIZE_2, NUM_LAYERS, OUTPUT_SIZE)
    gru.load_state_dict(torch.load(gru_path, map_location='cpu'))
    gru.to(device).eval()
    return bilstm, gru


@st.cache_data(ttl=3600)
def load_data() -> pd.DataFrame:
    """
    Télécharge les données NFLX avec retry automatique en cas de rate-limit.
    Retourne un DataFrame avec colonne 'Close', ou un DataFrame VIDE si échec total.
    L'appelant DOIT vérifier df.empty avant d'utiliser les données.
    """
    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            data = yf.download(TICKER, period="2y", interval="1d", progress=False)

            # Nettoyage MultiIndex éventuel
            if isinstance(data.columns, pd.MultiIndex):
                if 'Close' in data.columns.get_level_values(0):
                    data = data.xs('Close', level=0, axis=1)
            elif 'Close' in data.columns:
                data = data[['Close']]

            if isinstance(data, pd.DataFrame) and data.shape[1] > 1:
                data = data.iloc[:, 0].to_frame()

            data.columns = ['Close']
            data = data.dropna()

            # Vérification : au moins SEQ_LENGTH + 30 lignes utiles
            if len(data) >= SEQ_LENGTH + 1:
                return data

            # Données insuffisantes même si téléchargement réussi
            last_exc = ValueError(f"Seulement {len(data)} lignes reçues (minimum requis : {SEQ_LENGTH + 1})")

        except Exception as exc:
            last_exc = exc

        # Attendre avant de réessayer (sauf dernière tentative)
        if attempt < MAX_RETRIES:
            time.sleep(RETRY_DELAY)

    # Toutes les tentatives ont échoué → retourner DataFrame vide avec le bon index
    st.session_state["data_error"] = str(last_exc)
    return pd.DataFrame(columns=['Close'])


def predict_future(model, initial_sequence, num_days, device):
    current_seq = initial_sequence.clone()
    predictions = []
    for _ in range(num_days):
        with torch.no_grad():
            pred = model(current_seq)
            predictions.append(pred.item())
            pred_tensor = pred.view(1, 1, 1)
            current_seq = torch.cat((current_seq[:, 1:, :], pred_tensor), dim=1)
    return predictions


# ==========================================
# 5. SIDEBAR
# ==========================================
with st.sidebar:
    st.markdown('<div class="wordmark">NFLX</div>', unsafe_allow_html=True)
    st.markdown('<div class="wordmark-sub">Prédiction IA · Ensemble</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-label">Prédiction</div>', unsafe_allow_html=True)
    prediction_days = st.slider(
        "Horizon (jours ouvrés)",
        min_value=1, max_value=30, value=15,
        help="Nombre de jours ouvrés à projeter"
    )
    st.markdown("")

    st.markdown('<div class="section-label">Historique affiché</div>', unsafe_allow_html=True)
    # FIX #1 : label non-vide pour éviter le warning d'accessibilité
    history_days = st.selectbox(
        "Nombre de jours d'historique",          # ← label visible requis par Streamlit
        options=[5, 15, 30, 60, 90, 180, 365],
        index=2,
        format_func=lambda x: f"{x} jours",
        label_visibility="collapsed"             # ← on le cache visuellement, pas sémantiquement
    )

    st.markdown('<hr class="hr">', unsafe_allow_html=True)

    st.markdown('<div class="section-label">Modèles</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="pill-row">
        <span class="pill pill-bilstm">Bi-LSTM</span>
        <span class="pill pill-gru">GRU</span>
        <span class="pill pill-ensemble">Ensemble</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("")
    st.markdown("""
    <div style="font-size:0.78rem; color:#555; line-height:1.7;">
        Le <b style="color:#888">Bi-LSTM</b> capture les dépendances temporelles dans les deux sens.
        Le <b style="color:#888">GRU</b> est plus compact et rapide à converger.
        L'<b style="color:#888">Ensemble</b> moyenne les deux pour réduire la variance.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="hr">', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="section-label">Architecture</div>
    <div style="font-size:0.78rem; color:#555; line-height:2;">
        Séquence d'entrée — <b style="color:#888">{SEQ_LENGTH} jours</b><br>
        Couches — <b style="color:#888">{NUM_LAYERS}</b><br>
        Taille cachée — <b style="color:#888">{HIDDEN_SIZE_1} / {HIDDEN_SIZE_2}</b>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="hr">', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.72rem; color:#3A3A3A; line-height:1.6;">
        ⚠ Ces projections sont à titre informatif uniquement.<br>
        Pas de conseil financier.
    </div>
    """, unsafe_allow_html=True)


# ==========================================
# 6. CHARGEMENT
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Réinitialise l'erreur de données à chaque run
if "data_error" not in st.session_state:
    st.session_state["data_error"] = None

with st.spinner('Chargement des modèles et données…'):
    try:
        bilstm_model, gru_model = load_trained_models(
            'best_bilstm_nflx.pth', 'best_gru_nflx.pth', device
        )
    except FileNotFoundError as e:
        st.error(f"Fichier manquant : {e}")
        st.info("Placez `best_bilstm_nflx.pth` et `best_gru_nflx.pth` dans le répertoire de l'app.")
        st.stop()
    except Exception as e:
        st.error(f"Erreur au chargement des modèles : {e}")
        st.stop()

    df = load_data()


# ==========================================
# FIX #2 : Guard central — df vide = erreur yfinance
# On affiche un message clair et on STOPPE proprement.
# ==========================================
if df.empty:
    err_detail = st.session_state.get("data_error", "Erreur inconnue")
    st.markdown(f"""
    <div class="hero">
        <p class="hero-ticker">NET<span>FLIX</span></p>
        <p class="hero-desc">NASDAQ · NFLX · Données indisponibles</p>
    </div>
    <div class="error-box">
        <div class="error-box-title">⚠ Impossible de récupérer les données de marché</div>
        <div class="error-box-msg">
            Yahoo Finance a temporairement bloqué la requête (rate-limit). L'app a réessayé {MAX_RETRIES} fois sans succès.<br><br>
            <b>Détail :</b> {err_detail}<br><br>
            <b>Solutions :</b><br>
            • Patientez 1 à 2 minutes puis rechargez la page<br>
            • Si l'erreur persiste, redémarrez l'app depuis le menu Streamlit Cloud<br>
            • Le cache se réinitialise automatiquement après 1 heure
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Bouton de rechargement forcé (vide le cache data)
    if st.button("🔄 Réessayer maintenant"):
        load_data.clear()
        st.rerun()

    st.stop()   # ← arrêt propre, aucune autre ligne ne s'exécute


# ==========================================
# 7. HERO + KPIs
# ==========================================
last_date  = df.index[-1]

last_price = float(df['Close'].iloc[-1])
price_1d   = float((df['Close'].iloc[-1] - df['Close'].iloc[-2])  / df['Close'].iloc[-2]  * 100) if len(df) > 1  else 0.0
price_7d   = float((df['Close'].iloc[-1] - df['Close'].iloc[-7])  / df['Close'].iloc[-7]  * 100) if len(df) > 7  else 0.0
price_30d  = float((df['Close'].iloc[-1] - df['Close'].iloc[-30]) / df['Close'].iloc[-30] * 100) if len(df) > 30 else 0.0

st.markdown(f"""
<div class="hero">
    <p class="hero-ticker">NET<span>FLIX</span></p>
    <p class="hero-desc">NASDAQ · NFLX · Mise à jour {last_date.strftime('%d %b %Y').upper()}</p>
</div>
""", unsafe_allow_html=True)


def delta_class(v):
    return "kpi-delta-up" if v >= 0 else "kpi-delta-down"

def delta_arrow(v):
    return f"↑ +{v:.2f}%" if v >= 0 else f"↓ {v:.2f}%"


st.markdown(f"""
<div class="kpi-row">
    <div class="kpi-cell">
        <div class="kpi-label">Dernier cours</div>
        <div class="kpi-value">${last_price:.2f}</div>
        <div class="{delta_class(price_1d)}">{delta_arrow(price_1d)} 24h</div>
    </div>
    <div class="kpi-cell">
        <div class="kpi-label">7 jours</div>
        <div class="kpi-value">{price_7d:+.2f}%</div>
        <div class="kpi-neutral">Performance hebdo</div>
    </div>
    <div class="kpi-cell">
        <div class="kpi-label">30 jours</div>
        <div class="kpi-value">{price_30d:+.2f}%</div>
        <div class="kpi-neutral">Performance mensuelle</div>
    </div>
    <div class="kpi-cell">
        <div class="kpi-label">Horizon prédit</div>
        <div class="kpi-value">{prediction_days}J</div>
        <div class="kpi-neutral">{prediction_days} jours ouvrés</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ==========================================
# 8. BOUTON DE PRÉDICTION
# ==========================================
st.markdown('<div class="section-label">Projection</div>', unsafe_allow_html=True)

col_pad1, col_btn, col_pad2 = st.columns([2, 1, 2])
with col_btn:
    predict_button = st.button(f"→ Générer ({prediction_days}j)")

if predict_button:
    with st.spinner(f'Calcul des projections sur {prediction_days} jours…'):

        scaler      = MinMaxScaler(feature_range=(-1, 1))
        scaled_data = scaler.fit_transform(df[['Close']].values)

        last_sequence = scaled_data[-SEQ_LENGTH:]
        initial_seq   = torch.tensor(last_sequence, dtype=torch.float32).view(1, SEQ_LENGTH, 1).to(device)

        progress_bar = st.progress(0)
        status_text  = st.empty()

        status_text.text("Bi-LSTM — calcul en cours…")
        progress_bar.progress(0.33)
        bilstm_preds = predict_future(bilstm_model, initial_seq, prediction_days, device)

        status_text.text("GRU — calcul en cours…")
        progress_bar.progress(0.66)
        gru_preds = predict_future(gru_model, initial_seq, prediction_days, device)

        status_text.text("Ensemble — agrégation…")
        progress_bar.progress(1.0)
        ensemble_preds = [(b + g) / 2 for b, g in zip(bilstm_preds, gru_preds)]

        progress_bar.empty()
        status_text.empty()

        bilstm_inv   = scaler.inverse_transform(np.array(bilstm_preds).reshape(-1, 1))
        gru_inv      = scaler.inverse_transform(np.array(gru_preds).reshape(-1, 1))
        ensemble_inv = scaler.inverse_transform(np.array(ensemble_preds).reshape(-1, 1))

        future_dates = pd.bdate_range(start=last_date + timedelta(days=1), periods=prediction_days)
        df_future = pd.DataFrame({
            'Bi-LSTM':  bilstm_inv.flatten(),
            'GRU':      gru_inv.flatten(),
            'Ensemble': ensemble_inv.flatten()
        }, index=future_dates)

        bilstm_var   = float((df_future['Bi-LSTM'].iloc[-1]  - last_price) / last_price * 100)
        gru_var      = float((df_future['GRU'].iloc[-1]      - last_price) / last_price * 100)
        ensemble_var = float((df_future['Ensemble'].iloc[-1] - last_price) / last_price * 100)

    # ---- RÉSULTATS ----
    st.markdown('<hr class="hr">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Résultats</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    panels = [
        (c1, "Bi-LSTM",  float(df_future['Bi-LSTM'].iloc[-1]),  bilstm_var,   "result-panel-accent-green"),
        (c2, "GRU",      float(df_future['GRU'].iloc[-1]),      gru_var,      "result-panel-accent-red"),
        (c3, "Ensemble", float(df_future['Ensemble'].iloc[-1]), ensemble_var, "result-panel-accent-white"),
    ]
    for col, name, price, var, accent in panels:
        delta_cls = "result-delta-up" if var >= 0 else "result-delta-down"
        arrow     = "↑" if var >= 0 else "↓"
        with col:
            st.markdown(f"""
            <div class="result-panel {accent}">
                <div class="result-model-name">{name} · J+{prediction_days}</div>
                <div class="result-price">${price:.2f}</div>
                <div class="{delta_cls}">{arrow} {var:+.2f}%</div>
            </div>
            """, unsafe_allow_html=True)

    # ---- GRAPHIQUE ----
    st.markdown("")
    # FIX #3 : s'assurer que history_days ne dépasse pas la longueur réelle du df
    safe_history = min(history_days, len(df))
    recent_df = df.iloc[-safe_history:]

    fig = go.Figure()

    # Historique
    fig.add_trace(go.Scatter(
        x=recent_df.index,
        y=recent_df['Close'],
        mode='lines',
        name='Historique',
        line=dict(color='#3A3A3A', width=1.5),
        fill='tozeroy',
        fillcolor='rgba(255,255,255,0.02)',
        hovertemplate='%{x|%d %b %Y}<br>$%{y:.2f}<extra>Historique</extra>'
    ))

    # Connexions historique → prédiction
    for clr, col_name in [('#46D369','Bi-LSTM'), ('#E50914','GRU'), ('#BBBBBB','Ensemble')]:
        fig.add_trace(go.Scatter(
            x=[recent_df.index[-1], df_future.index[0]],
            y=[float(recent_df['Close'].iloc[-1]), float(df_future[col_name].iloc[0])],
            mode='lines',
            showlegend=False,
            hoverinfo='skip',
            line=dict(color=clr, width=1, dash='dot'),
            opacity=0.4
        ))

    # Bi-LSTM
    fig.add_trace(go.Scatter(
        x=df_future.index, y=df_future['Bi-LSTM'],
        mode='lines', name='Bi-LSTM',
        line=dict(color='#46D369', width=1.5, dash='dot'),
        hovertemplate='%{x|%d %b %Y}<br>$%{y:.2f}<extra>Bi-LSTM</extra>'
    ))

    # GRU
    fig.add_trace(go.Scatter(
        x=df_future.index, y=df_future['GRU'],
        mode='lines', name='GRU',
        line=dict(color='#E50914', width=1.5, dash='dot'),
        hovertemplate='%{x|%d %b %Y}<br>$%{y:.2f}<extra>GRU</extra>'
    ))

    # Bande de confiance
    half_gap = (df_future['Bi-LSTM'] - df_future['GRU']).abs() / 2
    fig.add_trace(go.Scatter(
        x=list(df_future.index) + list(df_future.index[::-1]),
        y=list(df_future['Ensemble'] + half_gap) + list((df_future['Ensemble'] - half_gap)[::-1]),
        fill='toself',
        fillcolor='rgba(255,255,255,0.04)',
        line=dict(color='rgba(0,0,0,0)'),
        showlegend=False,
        hoverinfo='skip',
        name='Intervalle'
    ))

    # Ensemble
    fig.add_trace(go.Scatter(
        x=df_future.index, y=df_future['Ensemble'],
        mode='lines+markers', name='Ensemble',
        line=dict(color='#FFFFFF', width=2.5),
        marker=dict(size=5, color='#FFFFFF', symbol='circle'),
        hovertemplate='%{x|%d %b %Y}<br>$%{y:.2f}<extra>Ensemble</extra>'
    ))

    # Ligne verticale "Aujourd'hui" via Scatter (pas add_vline — buggé avec datetime)
    all_y_values = (
        [float(v) for v in recent_df['Close'].dropna()]
        + [float(v) for v in df_future['Bi-LSTM']]
        + [float(v) for v in df_future['GRU']]
        + [float(v) for v in df_future['Ensemble']]
    )
    y_min = min(all_y_values) * 0.97
    y_max = max(all_y_values) * 1.03

    fig.add_trace(go.Scatter(
        x=[last_date, last_date],
        y=[y_min, y_max],
        mode='lines',
        showlegend=False,
        hoverinfo='skip',
        line=dict(color='#2A2A2A', width=1),
    ))
    fig.add_annotation(
        x=last_date,
        y=y_max,
        text="Aujourd'hui",
        showarrow=False,
        xanchor='left',
        yanchor='top',
        font=dict(color='#444', size=10, family='DM Sans'),
        bgcolor='rgba(0,0,0,0)',
        bordercolor='rgba(0,0,0,0)',
    )

    fig.update_layout(
        paper_bgcolor='#0A0A0A',
        plot_bgcolor='#0A0A0A',
        margin=dict(l=0, r=0, t=20, b=0),
        height=420,
        hovermode='x unified',
        xaxis=dict(
            showgrid=False, zeroline=False, showline=False,
            tickfont=dict(family='DM Sans', size=10, color='#444'),
        ),
        yaxis=dict(
            showgrid=True, gridcolor='#141414', zeroline=False, showline=False,
            tickfont=dict(family='DM Sans', size=10, color='#444'),
            tickprefix='$',
        ),
        legend=dict(
            orientation='h', x=0, y=1.06,
            font=dict(family='DM Sans', size=10, color='#666'),
            bgcolor='rgba(0,0,0,0)', bordercolor='rgba(0,0,0,0)',
        ),
        hoverlabel=dict(
            bgcolor='#111111', bordercolor='#2A2A2A',
            font=dict(family='DM Sans', size=11, color='#CCC'),
        ),
    )

    st.plotly_chart(fig, use_container_width=True)

    # ---- CONVERGENCE ----
    divergence     = abs(float(df_future['Bi-LSTM'].iloc[-1]) - float(df_future['GRU'].iloc[-1]))
    divergence_pct = (divergence / float(df_future['Ensemble'].iloc[-1])) * 100

    if divergence_pct < 2:
        conv_icon, conv_title = "🟢", "Forte convergence"
        conv_msg = "Les deux modèles sont très alignés — la prédiction Ensemble est fiable."
    elif divergence_pct < 5:
        conv_icon, conv_title = "🟡", "Convergence modérée"
        conv_msg = "Accord raisonnable entre les modèles, quelques divergences à noter."
    else:
        conv_icon, conv_title = "🔴", "Divergence significative"
        conv_msg = "Les modèles divergent — interprétez la prédiction avec prudence."

    st.markdown(f"""
    <div class="convergence-box">
        <div class="conv-indicator">{conv_icon}</div>
        <div>
            <div class="conv-title">{conv_title} · Écart ${divergence:.2f} ({divergence_pct:.1f}%)</div>
            <div class="conv-msg">{conv_msg}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ---- TENDANCES ----
    st.markdown('<div class="section-label">Tendances</div>', unsafe_allow_html=True)
    tc1, tc2, tc3 = st.columns(3)

    def trend_html(col_obj, model_name, var, final_price):
        pct_class = "trend-pct-up" if var >= 0 else "trend-pct-down"
        label_txt = "Haussière"    if var >= 0 else "Baissière"
        with col_obj:
            st.markdown(f"""
            <div class="trend-card">
                <div class="trend-model">{model_name}</div>
                <div class="{pct_class}">{var:+.2f}%</div>
                <div class="trend-label">{label_txt}</div>
                <div class="trend-final">Cible · ${final_price:.2f}</div>
            </div>
            """, unsafe_allow_html=True)

    trend_html(tc1, "Bi-LSTM",  bilstm_var,   float(df_future['Bi-LSTM'].iloc[-1]))
    trend_html(tc2, "GRU",      gru_var,       float(df_future['GRU'].iloc[-1]))
    trend_html(tc3, "Ensemble", ensemble_var,  float(df_future['Ensemble'].iloc[-1]))

    # ---- STATISTIQUES ----
    st.markdown('<hr class="hr">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Statistiques</div>', unsafe_allow_html=True)

    s1, s2 = st.columns(2)

    raw_corr = df_future['Bi-LSTM'].corr(df_future['GRU'])
    corr_str = f"{raw_corr:.4f}" if (not np.isnan(raw_corr)) else "N/A (1 point)"

    tendance_commune = "Oui ✓" if bilstm_var * gru_var > 0 else "Non ✗"

    ens_mean  = float(df_future['Ensemble'].mean())
    ens_std   = float(df_future['Ensemble'].std()) if prediction_days > 1 else 0.0
    ens_max   = float(df_future['Ensemble'].max())
    ens_min   = float(df_future['Ensemble'].min())
    vol_str   = f"{(ens_std / ens_mean * 100):.2f}%" if ens_mean != 0 else "N/A"
    gap_mean  = float(abs(df_future['Bi-LSTM'] - df_future['GRU']).mean())
    gap_max   = float(abs(df_future['Bi-LSTM'] - df_future['GRU']).max())

    with s1:
        st.markdown(f"""
        <div class="stats-grid">
            <div class="stats-cell">
                <div class="stats-key">Prix moyen (Ensemble)</div>
                <div class="stats-val">${ens_mean:.2f}</div>
            </div>
            <div class="stats-cell">
                <div class="stats-key">Volatilité prédite</div>
                <div class="stats-val">{vol_str}</div>
            </div>
            <div class="stats-cell">
                <div class="stats-key">Maximum prédit</div>
                <div class="stats-val">${ens_max:.2f}</div>
            </div>
            <div class="stats-cell">
                <div class="stats-key">Minimum prédit</div>
                <div class="stats-val">${ens_min:.2f}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with s2:
        st.markdown(f"""
        <div class="stats-grid">
            <div class="stats-cell">
                <div class="stats-key">Corrélation des modèles</div>
                <div class="stats-val">{corr_str}</div>
            </div>
            <div class="stats-cell">
                <div class="stats-key">Tendance commune</div>
                <div class="stats-val">{tendance_commune}</div>
            </div>
            <div class="stats-cell">
                <div class="stats-key">Écart moyen</div>
                <div class="stats-val">${gap_mean:.2f}</div>
            </div>
            <div class="stats-cell">
                <div class="stats-key">Écart maximum</div>
                <div class="stats-val">${gap_max:.2f}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ---- TABLEAU DÉTAILLÉ ----
    st.markdown('<hr class="hr">', unsafe_allow_html=True)
    with st.expander("Données complètes — tableau de prédiction"):
        df_display = pd.DataFrame(
            {
                'Bi-LSTM ($)':       [f"${v:.2f}" for v in df_future['Bi-LSTM']],
                'GRU ($)':           [f"${v:.2f}" for v in df_future['GRU']],
                'Ensemble ($)':      [f"${v:.2f}" for v in df_future['Ensemble']],
                'Écart Bi-LSTM/GRU': [f"${abs(b-g):.2f}" for b, g in zip(df_future['Bi-LSTM'], df_future['GRU'])],
            },
            index=df_future.index.strftime('%d/%m/%Y')
        )
        df_display.index.name = 'Date'
        st.dataframe(df_display, use_container_width=True, height=380)


# ==========================================
# FOOTER
# ==========================================
st.markdown("""
<div class="footer">
    <div class="footer-brand">NFLX · IA Ensemble</div>
    <div class="footer-warning">
        Les projections présentées sont générées par des modèles d'apprentissage automatique
        à titre éducatif uniquement. Elles ne constituent pas un conseil financier.
    </div>
</div>
""", unsafe_allow_html=True)
