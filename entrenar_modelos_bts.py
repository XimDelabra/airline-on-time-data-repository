"""Paso 3: entrenar y evaluar seis modelos BTS sin fuga de información.

Ejecuta en la carpeta principal del repositorio:
    python entrenar_modelos_bts.py

Requiere pandas y scikit-learn. Guarda métricas y modelos en resultados_bts/.
La evaluación final utiliza noviembre y diciembre, posteriores al entrenamiento.
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


ENTRADA = Path("datos_procesados_bts/vuelos_2024_limpios.csv")
SALIDA = Path("resultados_bts")
CATEGORICAS = ["Reporting_Airline", "Origin", "Dest"]
NUMERICAS = ["Month", "DayOfWeek", "DepartureHour", "Distance"]
VARIABLES = CATEGORICAS + NUMERICAS
OBJETIVOS = {"retrasos": "ArrDel15", "cancelaciones": "Cancelled",
             "desvios": "Diverted"}


def preparar():
    return ColumnTransformer([
        ("categoricas", OneHotEncoder(handle_unknown="ignore"), CATEGORICAS),
        ("numericas", StandardScaler(), NUMERICAS),
    ])


def evaluar(modelo, x, y):
    predicho = modelo.predict(x)
    matriz = confusion_matrix(y, predicho, labels=[0, 1]).tolist()
    return {
        "registros": int(len(y)), "positivos_reales": int(y.sum()),
        "accuracy": float(accuracy_score(y, predicho)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicho)),
        "precision": float(precision_score(y, predicho, zero_division=0)),
        "recall": float(recall_score(y, predicho, zero_division=0)),
        "f1": float(f1_score(y, predicho, zero_division=0)),
        "matriz_confusion": matriz,
    }


def main():
    if not ENTRADA.is_file():
        raise SystemExit(f"No existe {ENTRADA}; ejecuta primero limpiar_datos_bts.py")
    SALIDA.mkdir(parents=True, exist_ok=True)
    datos = pd.read_csv(ENTRADA, parse_dates=["FlightDate"])
    datos["Month"] = datos["FlightDate"].dt.month

    # Periodos cronológicos: enero-septiembre / octubre / noviembre-diciembre.
    periodos = {
        "entrenamiento": datos["Month"] <= 9,
        "validacion": datos["Month"] == 10,
        "prueba": datos["Month"] >= 11,
    }
    resultados = []
    for problema, objetivo in OBJETIVOS.items():
        # Solo retrasos exige llegada efectiva; los cancelados y desviados
        # permanecen en sus respectivos problemas de clasificación.
        sub = datos.loc[datos[objetivo].notna()].copy()
        sub[objetivo] = sub[objetivo].astype(int)
        grupos = {nombre: sub.loc[mascara.loc[sub.index]] for nombre, mascara in periodos.items()}
        if any(g.empty or g[objetivo].nunique() < 2 for g in grupos.values()):
            raise SystemExit(f"No hay ambas clases en todos los periodos de {problema}")
        print(f"\n{problema.upper()} ({objetivo})")
        for nombre, g in grupos.items():
            print(f"  {nombre}: {len(g):,} vuelos; {int(g[objetivo].sum()):,} positivos")

        algoritmos = {
            "regresion_logistica": LogisticRegression(
                max_iter=1000, class_weight="balanced", random_state=42
            ),
            "arbol_decision": DecisionTreeClassifier(
                max_depth=12, min_samples_leaf=50,
                class_weight="balanced", random_state=42
            ),
        }
        for nombre, algoritmo in algoritmos.items():
            print(f"  Entrenando {nombre}...", flush=True)
            modelo = Pipeline([("preparacion", preparar()), ("clasificador", algoritmo)])
            train = grupos["entrenamiento"]
            modelo.fit(train[VARIABLES], train[objetivo])
            met_val = evaluar(modelo, grupos["validacion"][VARIABLES], grupos["validacion"][objetivo])
            met_test = evaluar(modelo, grupos["prueba"][VARIABLES], grupos["prueba"][objetivo])
            joblib.dump(modelo, SALIDA / f"modelo_{problema}_{nombre}.joblib")
            resultados.append({"problema": problema, "algoritmo": nombre,
                               "variable_objetivo": objetivo,
                               "variables_entrada": VARIABLES,
                               "validacion": met_val, "prueba": met_test})
            print(f"    F1 prueba={met_test['f1']:.3f}, "
                  f"Recall={met_test['recall']:.3f}, "
                  f"Precision={met_test['precision']:.3f}")

    (SALIDA / "metricas_modelos.json").write_text(
        json.dumps({"metodologia": "Entrenamiento enero-septiembre, validación octubre, "
                                   "prueba noviembre-diciembre de 2024; sin usar "
                                   "resultados del vuelo como entradas.",
                    "modelos": resultados}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    pd.DataFrame([
        {"problema": item["problema"], "algoritmo": item["algoritmo"],
         "periodo": periodo, **{k: v for k, v in item[periodo].items()
                               if k != "matriz_confusion"}}
        for item in resultados for periodo in ("validacion", "prueba")
    ]).to_csv(SALIDA / "comparacion_modelos.csv", index=False)
    print(f"\nListo. Guardé seis modelos y métricas en {SALIDA.resolve()}")
    print("Estos son modelos iniciales; la optimización y las gráficas siguen después.")


if __name__ == "__main__":
    main()
