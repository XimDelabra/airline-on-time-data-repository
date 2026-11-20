"""Mejora 1: entrenar los mismos algoritmos con tres variables adicionales.

Ejecutar desde la raíz del repositorio:
    python entrenar_modelos_bts_v2.py

Mantiene las configuraciones de V1 para aislar el efecto de agregar variables.
No modifica resultados_bts/ ni los modelos originales.
"""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

ARCHIVO_V2 = Path("datos_procesados_bts/vuelos_2024_limpios_v2.csv")
ORIGINALES = Path("resultados_bts/metricas_modelos.json")
DESTINO = Path("resultados_bts_v2")
OBJETIVOS = {"retrasos": "ArrDel15", "cancelaciones": "Cancelled", "desvios": "Diverted"}
CATEGORICAS = ["Reporting_Airline", "Origin", "Dest"]
NUMERICAS_V1 = ["Month", "DayOfWeek", "DepartureHour", "Distance"]
NUMERICAS_NUEVAS = ["DayofMonth", "ArrivalHour", "CRSElapsedTime"]
ENTRADAS = CATEGORICAS + NUMERICAS_V1 + NUMERICAS_NUEVAS


def crear_modelo(algoritmo):
    preprocesador = ColumnTransformer([
        ("categoricas", OneHotEncoder(handle_unknown="ignore"), CATEGORICAS),
        ("numericas", StandardScaler(), NUMERICAS_V1 + NUMERICAS_NUEVAS),
    ])
    if algoritmo == "regresion_logistica":
        clasificador = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    else:
        clasificador = DecisionTreeClassifier(
            max_depth=12, min_samples_leaf=50, class_weight="balanced", random_state=42)
    return Pipeline([("preparacion", preprocesador), ("clasificador", clasificador)])


def evaluar(modelo, datos, objetivo):
    reales = datos[objetivo].astype(int)
    predichos = modelo.predict(datos[ENTRADAS])
    return {
        "registros": int(len(datos)), "positivos_reales": int(reales.sum()),
        "accuracy": float(accuracy_score(reales, predichos)),
        "balanced_accuracy": float(balanced_accuracy_score(reales, predichos)),
        "precision": float(precision_score(reales, predichos, zero_division=0)),
        "recall": float(recall_score(reales, predichos, zero_division=0)),
        "f1": float(f1_score(reales, predichos, zero_division=0)),
        "matriz_confusion": confusion_matrix(reales, predichos, labels=[0, 1]).tolist(),
    }


def main():
    if not ARCHIVO_V2.is_file():
        raise SystemExit(f"Falta {ARCHIVO_V2}; ejecuta preparar_datos_bts_v2.py")
    if not ORIGINALES.is_file():
        raise SystemExit(f"Falta {ORIGINALES}; se necesita para comparar con V1")
    datos = pd.read_csv(ARCHIVO_V2, parse_dates=["FlightDate"])
    esperadas = set(ENTRADAS + list(OBJETIVOS.values()))
    if faltantes := (esperadas - set(datos.columns)):
        raise SystemExit(f"Faltan columnas: {sorted(faltantes)}")
    datos["Month"] = datos["FlightDate"].dt.month
    previo = json.loads(ORIGINALES.read_text(encoding="utf-8"))["modelos"]
    DESTINO.mkdir(exist_ok=True)
    resultados = []

    for problema, objetivo in OBJETIVOS.items():
        subset = datos.loc[datos[objetivo].notna()].copy()
        subset[objetivo] = subset[objetivo].astype(int)
        train = subset.loc[subset["Month"] <= 9]
        valid = subset.loc[subset["Month"] == 10]
        test = subset.loc[subset["Month"] >= 11]
        for nombre in ("regresion_logistica", "arbol_decision"):
            anterior = next(a for a in previo if a["problema"] == problema and a["algoritmo"] == nombre)
            for grupo, etiqueta in ((valid, "validacion"), (test, "prueba")):
                if (len(grupo) != anterior[etiqueta]["registros"] or
                        int(grupo[objetivo].sum()) != anterior[etiqueta]["positivos_reales"]):
                    raise SystemExit(f"La muestra V2 no coincide con V1 ({problema}, {etiqueta})")
            modelo = crear_modelo(nombre)
            print(f"Entrenando {problema} - {nombre}...", flush=True)
            modelo.fit(train[ENTRADAS], train[objetivo])
            met_valid, met_test = evaluar(modelo, valid, objetivo), evaluar(modelo, test, objetivo)
            joblib.dump(modelo, DESTINO / f"modelo_{problema}_{nombre}_v2.joblib")
            resultados.append({
                "problema": problema, "algoritmo": nombre,
                "variable_objetivo": objetivo, "variables_entrada": ENTRADAS,
                "validacion": met_valid, "prueba": met_test,
                "f1_v1_validacion": anterior["validacion"]["f1"],
                "f1_v1_prueba": anterior["prueba"]["f1"],
            })
            print(f"  F1 octubre V1 {anterior['validacion']['f1']:.3f} -> "
                  f"V2 {met_valid['f1']:.3f}; "
                  f"F1 nov-dic V1 {anterior['prueba']['f1']:.3f} -> "
                  f"V2 {met_test['f1']:.3f}")

    (DESTINO / "metricas_modelos_v2.json").write_text(
        json.dumps({"cambio": "Se añadieron DayofMonth, ArrivalHour y CRSElapsedTime; "
                             "mismos algoritmos, parámetros, vuelos y particiones que V1.",
                    "nota": "Comparación exploratoria; las decisiones siguientes se tomarán "
                            "con octubre, no buscando maximizar el resultado de prueba.",
                    "modelos": resultados}, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame([{
        "problema": r["problema"], "algoritmo": r["algoritmo"],
        "f1_v1_validacion": r["f1_v1_validacion"],
        "f1_v2_validacion": r["validacion"]["f1"],
        "f1_v1_prueba": r["f1_v1_prueba"],
        "f1_v2_prueba": r["prueba"]["f1"],
        "precision_v2_prueba": r["prueba"]["precision"],
        "recall_v2_prueba": r["prueba"]["recall"],
    } for r in resultados]).to_csv(DESTINO / "comparacion_v1_vs_v2.csv", index=False)
    print(f"\nListo. Guardé seis modelos V2 y comparaciones en {DESTINO.resolve()}")


if __name__ == "__main__":
    main()
