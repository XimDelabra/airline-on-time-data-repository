"""Experimento multianual BTS: entrenar con 2023-24 y evaluar en 2025.

Ejecutar desde la raíz del repositorio:
    python entrenar_modelos_multianual_bts.py

Conserva los mismos siete predictores y algoritmos/hiperparámetros de V1.
Guarda los resultados en una carpeta distinta, sin alterar los modelos previos.
"""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

ENTRADA = Path("datos_procesados_bts/multianual/vuelos_2023_2025_limpios.csv")
DESTINO = Path("resultados_multianual_bts")
CATEGORICAS = ["Reporting_Airline", "Origin", "Dest"]
NUMERICAS = ["Month", "DayOfWeek", "DepartureHour", "Distance"]
ENTRADAS = CATEGORICAS + NUMERICAS
OBJETIVOS = {"retrasos": "ArrDel15", "cancelaciones": "Cancelled", "desvios": "Diverted"}


def crear_modelo(nombre):
    if nombre == "regresion_logistica":
        clasificador = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    else:
        clasificador = DecisionTreeClassifier(
            max_depth=12, min_samples_leaf=50, class_weight="balanced", random_state=42)
    return Pipeline([
        ("preparacion", ColumnTransformer([
            ("categoricas", OneHotEncoder(handle_unknown="ignore"), CATEGORICAS),
            ("numericas", StandardScaler(), NUMERICAS),
        ])),
        ("clasificador", clasificador),
    ])


def metricas(modelo, muestra, objetivo):
    y = muestra[objetivo].astype(int)
    pred = modelo.predict(muestra[ENTRADAS])
    return {
        "registros": int(len(y)), "positivos_reales": int(y.sum()),
        "positivos_predichos": int(pred.sum()),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "matriz_confusion": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
    }


def main():
    if not ENTRADA.is_file():
        raise SystemExit(f"No encuentro {ENTRADA}. Ejecuta preparar_datos_multianual_bts.py")
    datos = pd.read_csv(ENTRADA)
    esperados = set(ENTRADAS + list(OBJETIVOS.values()) + ["Year", "FlightDate"])
    if faltantes := (esperados - set(datos.columns)):
        raise SystemExit(f"Faltan columnas: {sorted(faltantes)}")
    if set(datos["Year"].unique()) != {2023, 2024, 2025}:
        raise SystemExit("Se esperaban datos de los tres años: 2023, 2024 y 2025")
    datos["FlightDate"] = pd.to_datetime(datos["FlightDate"], errors="raise")
    datos["Month"] = datos["FlightDate"].dt.month
    # 2025 no participa en el entrenamiento. La mitad final se mantiene aislada.
    divisiones = {
        "entrenamiento": (datos["Year"] <= 2024),
        "validacion": (datos["Year"] == 2025) & (datos["Month"] <= 6),
        "prueba": (datos["Year"] == 2025) & (datos["Month"] >= 7),
    }
    DESTINO.mkdir(exist_ok=True)
    resultados = []

    for problema, objetivo in OBJETIVOS.items():
        validos = datos[objetivo].notna()
        conjuntos = {nombre: datos.loc[validos & mascara].copy()
                    for nombre, mascara in divisiones.items()}
        for nombre, conjunto in conjuntos.items():
            if conjunto.empty or conjunto[objetivo].nunique() != 2:
                raise SystemExit(f"Faltan clases en {problema} - {nombre}")
            conjunto[objetivo] = conjunto[objetivo].astype(int)
            print(f"{problema} - {nombre}: {len(conjunto):,} vuelos; "
                  f"{int(conjunto[objetivo].sum()):,} positivos")

        for algoritmo in ("regresion_logistica", "arbol_decision"):
            print(f"Entrenando {problema}: {algoritmo}...", flush=True)
            modelo = crear_modelo(algoritmo)
            train = conjuntos["entrenamiento"]
            modelo.fit(train[ENTRADAS], train[objetivo])
            validacion = metricas(modelo, conjuntos["validacion"], objetivo)
            prueba = metricas(modelo, conjuntos["prueba"], objetivo)
            joblib.dump(modelo, DESTINO / f"modelo_{problema}_{algoritmo}_multianual.joblib")
            resultados.append({
                "problema": problema, "algoritmo": algoritmo,
                "objetivo": objetivo, "variables_entrada": ENTRADAS,
                "entrenamiento": {"registros": len(train),
                                  "positivos": int(train[objetivo].sum())},
                "validacion": validacion, "prueba": prueba,
            })
            print(f"  Validación 2025: F1={validacion['f1']:.4f}, "
                  f"precision={validacion['precision']:.4f}, recall={validacion['recall']:.4f}")
            print(f"  Prueba 2025: F1={prueba['f1']:.4f}, "
                  f"precision={prueba['precision']:.4f}, recall={prueba['recall']:.4f}")

    (DESTINO / "metricas_multianual.json").write_text(json.dumps({
        "division_temporal": {"entrenamiento": "2023-2024",
                              "validacion": "enero-junio 2025",
                              "prueba": "julio-diciembre 2025"},
        "nota": "No comparar directamente métricas de prueba con V1 como si "
                "provinieran de los mismos vuelos. Primero comparar en la "
                "misma partición o usar resultados como evidencia exploratoria.",
        "modelos": resultados,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame([{
        "problema": r["problema"], "algoritmo": r["algoritmo"],
        "entrenamiento": r["entrenamiento"]["registros"],
        "f1_validacion": r["validacion"]["f1"],
        "precision_validacion": r["validacion"]["precision"],
        "recall_validacion": r["validacion"]["recall"],
        "f1_prueba": r["prueba"]["f1"],
        "precision_prueba": r["prueba"]["precision"],
        "recall_prueba": r["prueba"]["recall"],
    } for r in resultados]).to_csv(DESTINO / "comparacion_multianual.csv", index=False)
    print(f"\nListo. Resultados guardados en {DESTINO.resolve()}")


if __name__ == "__main__":
    main()
