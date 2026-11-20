"""Experimento 3: Random Forest para retrasos, cancelaciones y desvíos.

Ejecutar desde la raíz del repositorio:
    python entrenar_random_forest_bts.py

Utiliza exactamente las siete variables originales y los mismos periodos de V1.
No modifica los modelos originales ni selecciona parámetros usando prueba.
"""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ENTRADA = Path("datos_procesados_bts/vuelos_2024_limpios.csv")
METRICAS_V1 = Path("resultados_bts/metricas_modelos.json")
DESTINO = Path("resultados_random_forest_bts")

CATEGORICAS = ["Reporting_Airline", "Origin", "Dest"]
NUMERICAS = ["Month", "DayOfWeek", "DepartureHour", "Distance"]
VARIABLES = CATEGORICAS + NUMERICAS
OBJETIVOS = {"retrasos": "ArrDel15", "cancelaciones": "Cancelled", "desvios": "Diverted"}


def evaluar(modelo, datos, objetivo):
    reales = datos[objetivo].astype(int)
    predicciones = modelo.predict(datos[VARIABLES])
    return {
        "registros": int(len(datos)),
        "positivos_reales": int(reales.sum()),
        "predicciones_positivas": int(predicciones.sum()),
        "accuracy": float(accuracy_score(reales, predicciones)),
        "balanced_accuracy": float(balanced_accuracy_score(reales, predicciones)),
        "precision": float(precision_score(reales, predicciones, zero_division=0)),
        "recall": float(recall_score(reales, predicciones, zero_division=0)),
        "f1": float(f1_score(reales, predicciones, zero_division=0)),
        "matriz_confusion": confusion_matrix(reales, predicciones, labels=[0, 1]).tolist(),
    }


def main():
    if not ENTRADA.exists() or not METRICAS_V1.exists():
        raise SystemExit("Faltan datos limpios V1 o sus métricas. Ejecuta primero los programas originales.")
    df = pd.read_csv(ENTRADA, parse_dates=["FlightDate"])
    df["Month"] = df["FlightDate"].dt.month
    anteriores = json.loads(METRICAS_V1.read_text(encoding="utf-8"))["modelos"]
    DESTINO.mkdir(exist_ok=True)
    resultados = []

    for problema, objetivo in OBJETIVOS.items():
        datos = df.loc[df[objetivo].notna()].copy()
        datos[objetivo] = datos[objetivo].astype(int)
        train = datos.loc[datos["Month"] <= 9]
        valid = datos.loc[datos["Month"] == 10]
        test = datos.loc[datos["Month"] >= 11]
        if any(g[objetivo].nunique() != 2 for g in [train, valid, test]):
            raise SystemExit(f"No están presentes las dos clases en todos los periodos: {problema}")

        # Las mismas siete características; solo cambia el tipo de algoritmo.
        modelo = Pipeline([
            ("preparacion", ColumnTransformer([
                ("categoricas", OneHotEncoder(handle_unknown="ignore"), CATEGORICAS),
                ("numericas", "passthrough", NUMERICAS),
            ])),
            ("clasificador", RandomForestClassifier(
                n_estimators=120,
                max_depth=14,
                min_samples_leaf=20,
                class_weight="balanced_subsample",
                random_state=42,
                n_jobs=2,
            )),
        ])
        print(f"Entrenando Random Forest para {problema}...", flush=True)
        modelo.fit(train[VARIABLES], train[objetivo])
        met_val = evaluar(modelo, valid, objetivo)
        met_test = evaluar(modelo, test, objetivo)
        joblib.dump(modelo, DESTINO / f"random_forest_{problema}.joblib")

        referencias = [r for r in anteriores if r["problema"] == problema]
        mejor_v1_val = max(referencias, key=lambda r: r["validacion"]["f1"])
        resultados.append({
            "problema": problema,
            "algoritmo": "random_forest",
            "configuracion": {"n_estimators": 120, "max_depth": 14,
                              "min_samples_leaf": 20, "class_weight": "balanced_subsample",
                              "random_state": 42},
            "variables_entrada": VARIABLES,
            "validacion": met_val,
            "prueba": met_test,
            "mejor_v1_en_validacion": {
                "algoritmo": mejor_v1_val["algoritmo"],
                "f1": mejor_v1_val["validacion"]["f1"],
            },
        })
        print(f"  Validación octubre: F1 RF={met_val['f1']:.4f}; "
              f"mejor V1={mejor_v1_val['validacion']['f1']:.4f} "
              f"({mejor_v1_val['algoritmo']})")
        print(f"  Prueba nov-dic (informativa): F1={met_test['f1']:.4f}; "
              f"precision={met_test['precision']:.4f}; recall={met_test['recall']:.4f}")

    salida = {
        "objetivo_experimento": "Comparar Random Forest con los dos algoritmos V1, "
                                "manteniendo muestra y variables originales.",
        "periodos": {"entrenamiento": "enero-septiembre 2024",
                     "validacion": "octubre 2024", "prueba": "noviembre-diciembre 2024"},
        "advertencia": "No seleccionar nuevas configuraciones observando el rendimiento de prueba.",
        "modelos": resultados,
    }
    (DESTINO / "metricas_random_forest.json").write_text(
        json.dumps(salida, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame([{
        "problema": r["problema"],
        "f1_mejor_v1_validacion": r["mejor_v1_en_validacion"]["f1"],
        "algoritmo_v1": r["mejor_v1_en_validacion"]["algoritmo"],
        "f1_random_forest_validacion": r["validacion"]["f1"],
        "f1_random_forest_prueba": r["prueba"]["f1"],
        "precision_random_forest_prueba": r["prueba"]["precision"],
        "recall_random_forest_prueba": r["prueba"]["recall"],
    } for r in resultados]).to_csv(DESTINO / "comparacion_random_forest_vs_v1.csv", index=False)

    print(f"\nListo: tres modelos nuevos y sus comparaciones en {DESTINO.resolve()}")


if __name__ == "__main__":
    main()
