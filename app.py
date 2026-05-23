import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# ==========================================================
# CONFIGURAÇÃO DA PÁGINA
# ==========================================================

st.set_page_config(
    page_title="PCA e PLS de Corantes por Imagem Digital",
    layout="wide"
)

st.title("PCA e PLS para Misturas de Corantes por Imagem Digital")

st.markdown("""
Este aplicativo extrai informações de cor de imagens de soluções de corantes
e aplica **PCA** para visualizar separações e **PLS** para prever concentração
ou número de gotas adicionadas.

Fluxo geral:

Imagem → ROI → RGB / HSV / Grayscale / Histogramas → PCA → PLS
""")


# ==========================================================
# FUNÇÕES
# ==========================================================

def load_image(uploaded_file):
    """Carrega imagem como RGB."""
    image = Image.open(uploaded_file).convert("RGB")
    return np.array(image)


def apply_roi(img, use_roi, x, y, w, h):
    """Aplica ROI manual se habilitado."""
    if not use_roi:
        return img

    height, width = img.shape[:2]

    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))
    w = max(1, min(w, width - x))
    h = max(1, min(h, height - y))

    return img[y:y+h, x:x+w]


def extract_color_features(img, hist_bins=16, use_hist=True):
    """
    Extrai descritores RGB, HSV, grayscale e histogramas.
    A imagem deve estar em RGB.
    """

    # RGB
    rgb = img.copy()

    # HSV usando OpenCV
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

    # Grayscale
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    features = {}

    # Canais RGB
    channel_names_rgb = ["R", "G", "B"]
    for i, ch in enumerate(channel_names_rgb):
        canal = rgb[:, :, i]
        features[f"{ch}_mean"] = np.mean(canal)
        features[f"{ch}_std"] = np.std(canal)
        features[f"{ch}_min"] = np.min(canal)
        features[f"{ch}_max"] = np.max(canal)
        features[f"{ch}_median"] = np.median(canal)

    # Canais HSV
    channel_names_hsv = ["H", "S", "V"]
    for i, ch in enumerate(channel_names_hsv):
        canal = hsv[:, :, i]
        features[f"{ch}_mean"] = np.mean(canal)
        features[f"{ch}_std"] = np.std(canal)
        features[f"{ch}_min"] = np.min(canal)
        features[f"{ch}_max"] = np.max(canal)
        features[f"{ch}_median"] = np.median(canal)

    # Grayscale
    features["Gray_mean"] = np.mean(gray)
    features["Gray_std"] = np.std(gray)
    features["Gray_min"] = np.min(gray)
    features["Gray_max"] = np.max(gray)
    features["Gray_median"] = np.median(gray)

    # Histogramas normalizados
    if use_hist:
        for i, ch in enumerate(channel_names_rgb):
            hist, _ = np.histogram(
                rgb[:, :, i],
                bins=hist_bins,
                range=(0, 256),
                density=True
            )
            for j, value in enumerate(hist):
                features[f"{ch}_hist_{j+1}"] = value

        for i, ch in enumerate(channel_names_hsv):
            hist_range = (0, 180) if ch == "H" else (0, 256)
            hist, _ = np.histogram(
                hsv[:, :, i],
                bins=hist_bins,
                range=hist_range,
                density=True
            )
            for j, value in enumerate(hist):
                features[f"{ch}_hist_{j+1}"] = value

        hist_gray, _ = np.histogram(
            gray,
            bins=hist_bins,
            range=(0, 256),
            density=True
        )
        for j, value in enumerate(hist_gray):
            features[f"Gray_hist_{j+1}"] = value

    return features, rgb, hsv, gray


def plot_histograms(rgb, hsv, gray, sample_name, hist_bins=32):
    """Gera histogramas RGB, HSV e grayscale."""

    st.markdown(f"#### Histogramas - {sample_name}")

    fig1, ax1 = plt.subplots(figsize=(7, 4))
    ax1.hist(rgb[:, :, 0].ravel(), bins=hist_bins, alpha=0.5, label="R")
    ax1.hist(rgb[:, :, 1].ravel(), bins=hist_bins, alpha=0.5, label="G")
    ax1.hist(rgb[:, :, 2].ravel(), bins=hist_bins, alpha=0.5, label="B")
    ax1.set_xlabel("Intensidade")
    ax1.set_ylabel("Frequência")
    ax1.set_title("Histograma RGB")
    ax1.legend()
    ax1.grid(True)
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots(figsize=(7, 4))
    ax2.hist(hsv[:, :, 0].ravel(), bins=hist_bins, alpha=0.5, label="H")
    ax2.hist(hsv[:, :, 1].ravel(), bins=hist_bins, alpha=0.5, label="S")
    ax2.hist(hsv[:, :, 2].ravel(), bins=hist_bins, alpha=0.5, label="V")
    ax2.set_xlabel("Intensidade")
    ax2.set_ylabel("Frequência")
    ax2.set_title("Histograma HSV")
    ax2.legend()
    ax2.grid(True)
    st.pyplot(fig2)

    fig3, ax3 = plt.subplots(figsize=(7, 4))
    ax3.hist(gray.ravel(), bins=hist_bins, alpha=0.8)
    ax3.set_xlabel("Intensidade")
    ax3.set_ylabel("Frequência")
    ax3.set_title("Histograma Grayscale")
    ax3.grid(True)
    st.pyplot(fig3)


def safe_r2(y_true, y_pred):
    """Calcula R² com segurança."""
    if len(y_true) < 2:
        return np.nan
    return r2_score(y_true, y_pred)


def calculate_metrics(y_true, y_pred):
    """Calcula métricas de regressão."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = safe_r2(y_true, y_pred)
    return r2, rmse, mae


# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.header("Configurações")

use_roi = st.sidebar.checkbox("Usar ROI manual", value=False)

st.sidebar.markdown("### ROI manual")
st.sidebar.info(
    "Use a mesma posição de ROI para todas as imagens, se as fotos foram tiradas no mesmo enquadramento."
)

roi_x = st.sidebar.number_input("x inicial", min_value=0, value=0, step=1)
roi_y = st.sidebar.number_input("y inicial", min_value=0, value=0, step=1)
roi_w = st.sidebar.number_input("largura da ROI", min_value=1, value=300, step=1)
roi_h = st.sidebar.number_input("altura da ROI", min_value=1, value=300, step=1)

st.sidebar.markdown("### Histogramas")
use_hist = st.sidebar.checkbox("Usar histogramas como variáveis", value=True)
hist_bins = st.sidebar.slider("Número de bins dos histogramas", 8, 64, 16, step=8)

st.sidebar.markdown("### PCA")
ncomp_pca_user = st.sidebar.slider("Número de componentes PCA para cálculo", 2, 5, 2)

st.sidebar.markdown("### PLS")
response_name = st.sidebar.text_input(
    "Nome da resposta y",
    value="Gotas ou concentração"
)


# ==========================================================
# UPLOAD DAS IMAGENS
# ==========================================================

uploaded_files = st.file_uploader(
    "Carregue as imagens das amostras",
    type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
    accept_multiple_files=True
)

if not uploaded_files:
    st.warning("Carregue as imagens para iniciar a análise.")
    st.stop()


# ==========================================================
# ENTRADA DAS AMOSTRAS
# ==========================================================

st.subheader("1. Identificação das amostras")

st.markdown("""
Informe o nome de cada amostra e o valor de resposta.

Exemplo:

- A0 = 0 gotas
- A1 = 1 gota
- A2 = 2 gotas
- A3 = 3 gotas

Também pode usar concentração em ppm, mg/L ou outra unidade.
""")

all_rows = []
image_cache = {}

for idx, uploaded_file in enumerate(uploaded_files):
    img = load_image(uploaded_file)
    image_cache[uploaded_file.name] = img

    with st.expander(f"Amostra: {uploaded_file.name}", expanded=True):
        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            st.image(img, caption=f"Imagem original: {uploaded_file.name}", width=250)

        with col2:
            sample_name = st.text_input(
                f"Nome da amostra",
                value=uploaded_file.name.split(".")[0],
                key=f"name_{idx}"
            )

        with col3:
            y_value = st.number_input(
                f"Valor de y",
                value=float(idx),
                step=1.0,
                key=f"y_{idx}"
            )

        # Aplica ROI
        roi_img = apply_roi(img, use_roi, roi_x, roi_y, roi_w, roi_h)

        st.image(roi_img, caption="Imagem usada na análise", width=250)

        features, rgb, hsv, gray = extract_color_features(
            roi_img,
            hist_bins=hist_bins,
            use_hist=use_hist
        )

        row = {
            "Arquivo": uploaded_file.name,
            "Amostra": sample_name,
            "y": y_value
        }

        row.update(features)
        all_rows.append(row)

        image_cache[f"{uploaded_file.name}_roi"] = roi_img
        image_cache[f"{uploaded_file.name}_rgb"] = rgb
        image_cache[f"{uploaded_file.name}_hsv"] = hsv
        image_cache[f"{uploaded_file.name}_gray"] = gray


# ==========================================================
# TABELA DE DESCRITORES
# ==========================================================

df = pd.DataFrame(all_rows)

st.subheader("2. Tabela de descritores extraídos")

st.dataframe(df)

csv_descritores = df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Baixar descritores em CSV",
    data=csv_descritores,
    file_name="descritores_corantes.csv",
    mime="text/csv"
)


# ==========================================================
# VISUALIZAÇÃO DOS CANAIS E HISTOGRAMAS
# ==========================================================

st.subheader("3. Visualização dos canais de cor")

sample_to_view = st.selectbox(
    "Escolha uma amostra para visualizar canais e histogramas",
    options=df["Arquivo"].tolist()
)

if sample_to_view:
    rgb = image_cache[f"{sample_to_view}_rgb"]
    hsv = image_cache[f"{sample_to_view}_hsv"]
    gray = image_cache[f"{sample_to_view}_gray"]

    st.markdown(f"### Amostra selecionada: {sample_to_view}")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(rgb[:, :, 0], caption="Canal R", clamp=True)
        st.image(rgb[:, :, 1], caption="Canal G", clamp=True)
        st.image(rgb[:, :, 2], caption="Canal B", clamp=True)

    with col2:
        st.image(hsv[:, :, 0], caption="Canal H", clamp=True)
        st.image(hsv[:, :, 1], caption="Canal S", clamp=True)
        st.image(hsv[:, :, 2], caption="Canal V", clamp=True)

    with col3:
        st.image(gray, caption="Grayscale", clamp=True)

    plot_histograms(rgb, hsv, gray, sample_to_view, hist_bins=hist_bins)


# ==========================================================
# PREPARAÇÃO DA MATRIZ X E VETOR y
# ==========================================================

st.subheader("4. Preparação dos dados para PCA e PLS")

meta_cols = ["Arquivo", "Amostra", "y"]
feature_cols = [col for col in df.columns if col not in meta_cols]

X = df[feature_cols].values
y = df["y"].values.astype(float)

st.write(f"Número de amostras: **{X.shape[0]}**")
st.write(f"Número de variáveis de cor: **{X.shape[1]}**")

if X.shape[0] < 3:
    st.error("São necessárias pelo menos 3 amostras para PCA/PLS com interpretação mínima.")
    st.stop()

# Remove colunas com variância zero
std_cols = np.std(X, axis=0)
valid_cols = std_cols > 0

if np.sum(~valid_cols) > 0:
    st.warning(
        f"Foram removidas {np.sum(~valid_cols)} variáveis com variância zero."
    )

X = X[:, valid_cols]
feature_cols_valid = np.array(feature_cols)[valid_cols].tolist()

# Autoescalamento
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# ==========================================================
# PCA
# ==========================================================

st.subheader("5. PCA - Análise de Componentes Principais")

max_pca_components = min(ncomp_pca_user, X_scaled.shape[0] - 1, X_scaled.shape[1])

pca = PCA(n_components=max_pca_components)
scores = pca.fit_transform(X_scaled)

df_scores = pd.DataFrame()
df_scores["Amostra"] = df["Amostra"]
df_scores["Arquivo"] = df["Arquivo"]
df_scores["y"] = y

for i in range(max_pca_components):
    df_scores[f"PC{i+1}"] = scores[:, i]

st.markdown("### Variância explicada")

explained = pca.explained_variance_ratio_ * 100

df_explained = pd.DataFrame({
    "Componente": [f"PC{i+1}" for i in range(max_pca_components)],
    "Variância explicada (%)": explained,
    "Variância acumulada (%)": np.cumsum(explained)
})

st.dataframe(df_explained)

# Score plot PC1 x PC2
if max_pca_components >= 2:
    fig_pca, ax_pca = plt.subplots(figsize=(7, 5))

    scatter = ax_pca.scatter(
        df_scores["PC1"],
        df_scores["PC2"],
        c=df_scores["y"],
        s=90
    )

    for i, row in df_scores.iterrows():
        ax_pca.text(row["PC1"], row["PC2"], row["Amostra"])

    ax_pca.set_xlabel(f"PC1 ({explained[0]:.2f}%)")
    ax_pca.set_ylabel(f"PC2 ({explained[1]:.2f}%)")
    ax_pca.set_title("PCA - Score Plot")
    ax_pca.grid(True)

    cbar = plt.colorbar(scatter, ax=ax_pca)
    cbar.set_label(response_name)

    st.pyplot(fig_pca)

# Loadings
st.markdown("### Loadings do PCA")

loadings = pd.DataFrame(
    pca.components_.T,
    columns=[f"PC{i+1}" for i in range(max_pca_components)],
    index=feature_cols_valid
)

st.dataframe(loadings)

# Top variáveis PC1
st.markdown("### Variáveis mais importantes em PC1")

top_n = min(20, len(feature_cols_valid))

top_pc1 = loadings["PC1"].abs().sort_values(ascending=False).head(top_n)

fig_load, ax_load = plt.subplots(figsize=(8, 5))
ax_load.bar(top_pc1.index, top_pc1.values)
ax_load.set_ylabel("|Loading PC1|")
ax_load.set_title("Top variáveis que mais influenciam PC1")
ax_load.tick_params(axis="x", rotation=90)
ax_load.grid(True)

st.pyplot(fig_load)


# ==========================================================
# PLS
# ==========================================================

st.subheader("6. PLS - Regressão por Mínimos Quadrados Parciais")

st.markdown("""
O PLS usa os descritores de cor como matriz **X** e tenta prever o valor **y**,
que pode ser número de gotas ou concentração.
""")

max_pls_components = min(X_scaled.shape[0] - 2, X_scaled.shape[1])

if max_pls_components < 1:
    st.error("Número de amostras insuficiente para validação cruzada no PLS.")
    st.stop()

ncomp_pls = st.slider(
    "Número de componentes PLS",
    min_value=1,
    max_value=max_pls_components,
    value=min(2, max_pls_components)
)

pls = PLSRegression(n_components=ncomp_pls)

# Calibração
pls.fit(X_scaled, y)
y_cal = pls.predict(X_scaled).ravel()

# Validação cruzada Leave-One-Out
loo = LeaveOneOut()
y_cv = cross_val_predict(pls, X_scaled, y, cv=loo).ravel()

# Métricas
r2_cal, rmsec, mae_cal = calculate_metrics(y, y_cal)
r2_cv, rmsecv, mae_cv = calculate_metrics(y, y_cv)

col1, col2, col3 = st.columns(3)
col1.metric("R² calibração", f"{r2_cal:.3f}")
col2.metric("RMSEC", f"{rmsec:.3f}")
col3.metric("MAE calibração", f"{mae_cal:.3f}")

col4, col5, col6 = st.columns(3)
col4.metric("R² validação cruzada", f"{r2_cv:.3f}")
col5.metric("RMSECV", f"{rmsecv:.3f}")
col6.metric("MAE validação", f"{mae_cv:.3f}")

# Gráfico real vs previsto
fig_pls, ax_pls = plt.subplots(figsize=(7, 5))

ax_pls.scatter(y, y_cal, s=90, label="Calibração")
ax_pls.scatter(y, y_cv, s=90, label="Validação cruzada")

min_val = min(np.min(y), np.min(y_cal), np.min(y_cv))
max_val = max(np.max(y), np.max(y_cal), np.max(y_cv))

ax_pls.plot([min_val, max_val], [min_val, max_val], linestyle="--")

ax_pls.set_xlabel(f"{response_name} real")
ax_pls.set_ylabel(f"{response_name} previsto")
ax_pls.set_title("PLS - Real vs Previsto")
ax_pls.legend()
ax_pls.grid(True)

st.pyplot(fig_pls)

# Tabela PLS
df_pls = pd.DataFrame({
    "Arquivo": df["Arquivo"],
    "Amostra": df["Amostra"],
    f"{response_name}_real": y,
    f"{response_name}_previsto_calibracao": y_cal,
    f"{response_name}_previsto_validacao": y_cv,
    "Erro_validacao": y - y_cv
})

st.markdown("### Tabela de resultados do PLS")

st.dataframe(df_pls)

csv_pls = df_pls.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Baixar resultados do PLS em CSV",
    data=csv_pls,
    file_name="resultados_pls_corantes.csv",
    mime="text/csv"
)


# ==========================================================
# INTERPRETAÇÃO AUTOMÁTICA
# ==========================================================

st.subheader("7. Interpretação automática")

pc1_corr = np.corrcoef(df_scores["PC1"], y)[0, 1]

st.write(f"Correlação entre PC1 e {response_name}: **{pc1_corr:.3f}**")

if abs(pc1_corr) >= 0.85:
    st.success(
        "O PCA indica forte tendência entre PC1 e a adição/concentração do segundo corante. "
        "Isso sugere que a mudança de cor está sendo bem capturada."
    )
elif abs(pc1_corr) >= 0.60:
    st.warning(
        "O PCA indica tendência moderada. Há influência do segundo corante, "
        "mas o experimento pode melhorar com mais amostras, réplicas ou melhor controle de iluminação."
    )
else:
    st.info(
        "O PCA ainda não mostra uma tendência forte com a adição/concentração. "
        "Verifique iluminação, ROI, fundo, volume das gotas e número de amostras."
    )

if r2_cv >= 0.90:
    st.success(
        "O PLS apresentou boa capacidade preditiva em validação cruzada. "
        "Os descritores de cor conseguem prever bem a resposta informada."
    )
elif r2_cv >= 0.70:
    st.warning(
        "O PLS apresentou desempenho intermediário. "
        "Pode funcionar, mas é recomendável aumentar o número de amostras e réplicas."
    )
else:
    st.error(
        "O PLS apresentou baixa capacidade preditiva. "
        "Isso pode indicar poucas amostras, ruído experimental, iluminação variável ou baixa diferença visual entre as misturas."
    )


# ==========================================================
# RECOMENDAÇÕES EXPERIMENTAIS
# ==========================================================

st.subheader("8. Recomendações para melhorar o experimento")

st.markdown("""
Para obter melhor separação no PCA e melhor previsão no PLS:

1. Use sempre o mesmo recipiente.
2. Use fundo branco ou preto fixo.
3. Mantenha a mesma iluminação.
4. Mantenha a mesma distância da câmera.
5. Evite sombra e reflexo.
6. Tire pelo menos 3 fotos por ponto.
7. Faça réplicas reais de preparo.
8. Use sempre o mesmo volume da solução base.
9. Padronize o volume da gota.
10. Use ROI apenas na região da solução, evitando bordas e reflexos.
""")
