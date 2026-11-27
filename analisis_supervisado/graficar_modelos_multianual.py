import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# --------------------------------------------------
# CONFIGURACIÓN
# --------------------------------------------------

CARPETA_RESULTADOS = Path("resultados_multianual_bts")

ARCHIVO_METRICAS = CARPETA_RESULTADOS / "metricas_multianual.json"

CARPETA_GRAFICAS = CARPETA_RESULTADOS / "graficas"

CARPETA_GRAFICAS.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# CARGAR RESULTADOS
# --------------------------------------------------

if not ARCHIVO_METRICAS.exists():
    raise FileNotFoundError(
        f"No se encontró {ARCHIVO_METRICAS}. "
        "Ejecuta primero entrenar_modelos_multianual_bts.py"
    )

with open(ARCHIVO_METRICAS, "r", encoding="utf-8") as archivo:
    datos = json.load(archivo)

modelos = datos["modelos"]


# --------------------------------------------------
# ESTILO GENERAL
# --------------------------------------------------

sns.set_theme(style="whitegrid")

COLORES = ["#5271FF", "#FF8E72"]

NOMBRES_PROBLEMAS = {
    "retrasos": "Retrasos",
    "cancelaciones": "Cancelaciones",
    "desvios": "Desvíos"
}

NOMBRES_ALGORITMOS = {
    "regresion_logistica": "Regresión logística",
    "arbol_decision": "Árbol de decisión"
}


# --------------------------------------------------
# GRÁFICA 1: DISTRIBUCIÓN DE EVENTOS
# --------------------------------------------------

problemas = list(NOMBRES_PROBLEMAS.keys())

cantidades = []

for problema in problemas:

    # Tomamos un solo algoritmo por problema para
    # evitar contar dos veces los mismos vuelos.
    modelo = next(
        m for m in modelos
        if m["problema"] == problema
        and m["algoritmo"] == "regresion_logistica"
    )

    total = (
        modelo["entrenamiento"]["positivos"]
        + modelo["validacion"]["positivos_reales"]
        + modelo["prueba"]["positivos_reales"]
    )

    cantidades.append(total)


plt.figure(figsize=(9, 5))

barras = plt.bar(
    [NOMBRES_PROBLEMAS[p] for p in problemas],
    cantidades,
    color=["#5271FF", "#FFB347", "#9B72CF"]
)

plt.bar_label(
    barras,
    labels=[f"{valor:,}" for valor in cantidades],
    padding=5
)

plt.title(
    "Cantidad de retrasos, cancelaciones y desvíos\n"
    "Muestra de 2023 a 2025",
    fontsize=14,
    fontweight="bold"
)

plt.ylabel("Cantidad de vuelos")
plt.xlabel("Tipo de evento")

plt.tight_layout()

plt.savefig(
    CARPETA_GRAFICAS / "01_distribucion_eventos.png",
    dpi=300
)

plt.close()


# --------------------------------------------------
# GRÁFICA 2: COMPARACIÓN DE F1
# --------------------------------------------------

fig, ejes = plt.subplots(1, 3, figsize=(15, 5))

for indice, problema in enumerate(problemas):

    seleccionados = [
        m for m in modelos
        if m["problema"] == problema
    ]

    nombres = [
        NOMBRES_ALGORITMOS[m["algoritmo"]]
        for m in seleccionados
    ]

    valores = [
        m["prueba"]["f1"]
        for m in seleccionados
    ]

    barras = ejes[indice].bar(
        nombres,
        valores,
        color=COLORES
    )

    ejes[indice].set_title(
        NOMBRES_PROBLEMAS[problema],
        fontweight="bold"
    )

    ejes[indice].set_ylabel("F1")

    ejes[indice].tick_params(
        axis="x",
        labelrotation=15
    )

    limite = max(valores) * 1.25 if max(valores) > 0 else 1
    ejes[indice].set_ylim(0, limite)

    ejes[indice].bar_label(
        barras,
        labels=[f"{v:.4f}" for v in valores],
        padding=3
    )

fig.suptitle(
    "Comparación del desempeño de los algoritmos\n"
    "Prueba: julio a diciembre de 2025",
    fontsize=15,
    fontweight="bold"
)

plt.tight_layout()

plt.savefig(
    CARPETA_GRAFICAS / "02_comparacion_f1.png",
    dpi=300
)

plt.close()


# --------------------------------------------------
# GRÁFICA 3: PRECISION Y RECALL
# --------------------------------------------------

fig, ejes = plt.subplots(1, 3, figsize=(15, 5))

for indice, problema in enumerate(problemas):

    seleccionados = [
        m for m in modelos
        if m["problema"] == problema
    ]

    nombres = ["Regresión\nlogística", "Árbol de\ndecisión"]

    precision = [
        m["prueba"]["precision"]
        for m in seleccionados
    ]

    recall = [
        m["prueba"]["recall"]
        for m in seleccionados
    ]

    x = np.arange(len(nombres))

    ancho = 0.35

    ejes[indice].bar(
        x - ancho / 2,
        precision,
        ancho,
        label="Precision",
        color="#5271FF"
    )

    ejes[indice].bar(
        x + ancho / 2,
        recall,
        ancho,
        label="Recall",
        color="#FF8E72"
    )

    ejes[indice].set_title(
        NOMBRES_PROBLEMAS[problema],
        fontweight="bold"
    )

    ejes[indice].set_xticks(x)
    ejes[indice].set_xticklabels(nombres)

    ejes[indice].set_ylim(0, 1)

    ejes[indice].set_ylabel("Valor de la métrica")

    ejes[indice].legend()

fig.suptitle(
    "Precision y Recall de los modelos\n"
    "Prueba: julio a diciembre de 2025",
    fontsize=15,
    fontweight="bold"
)

plt.tight_layout()

plt.savefig(
    CARPETA_GRAFICAS / "03_precision_recall.png",
    dpi=300
)

plt.close()


# --------------------------------------------------
# GRÁFICA 4: MATRICES DE CONFUSIÓN
# --------------------------------------------------

fig, ejes = plt.subplots(3, 2, figsize=(12, 14))

for indice, modelo in enumerate(modelos):

    fila = indice // 2
    columna = indice % 2

    matriz = np.array(
        modelo["prueba"]["matriz_confusion"]
    )

    sns.heatmap(
        matriz,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        ax=ejes[fila, columna],
        xticklabels=["No", "Sí"],
        yticklabels=["No", "Sí"]
    )

    problema = NOMBRES_PROBLEMAS[modelo["problema"]]

    algoritmo = NOMBRES_ALGORITMOS[modelo["algoritmo"]]

    ejes[fila, columna].set_title(
        f"{problema} - {algoritmo}",
        fontweight="bold"
    )

    ejes[fila, columna].set_xlabel(
        "Predicción del modelo"
    )

    ejes[fila, columna].set_ylabel(
        "Resultado real"
    )

fig.suptitle(
    "Matrices de confusión de los modelos multianuales\n"
    "Prueba: julio a diciembre de 2025",
    fontsize=15,
    fontweight="bold"
)

plt.tight_layout(rect=[0, 0, 1, 0.96])

plt.savefig(
    CARPETA_GRAFICAS / "04_matrices_confusion.png",
    dpi=300
)

plt.close()


# --------------------------------------------------
# FINALIZACIÓN
# --------------------------------------------------

print("\nGráficas generadas correctamente:\n")

for archivo in sorted(CARPETA_GRAFICAS.glob("*.png")):
    print(f"- {archivo.name}")

print(
    "\nLas imágenes están guardadas en:",
    CARPETA_GRAFICAS.resolve()
)