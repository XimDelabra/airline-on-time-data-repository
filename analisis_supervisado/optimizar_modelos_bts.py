"""Paso 4: optimizar umbrales con octubre y evaluar en noviembre-diciembre.

Uso: python optimizar_modelos_bts.py
Requiere haber ejecutado entrenar_modelos_bts.py.
No modifica los seis modelos originales ni usa prueba para escoger umbrales.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

DATOS = Path("datos_procesados_bts/vuelos_2024_limpios.csv")
MODELOS = Path("resultados_bts")
OBJETIVOS = {"retrasos": "ArrDel15", "cancelaciones": "Cancelled", "desvios": "Diverted"}
ALGORITMOS = ["regresion_logistica", "arbol_decision"]
ENTRADAS = ["Reporting_Airline", "Origin", "Dest", "Month", "DayOfWeek",
            "DepartureHour", "Distance"]


def medir(reales, probabilidades, umbral):
    pred = (probabilidades >= umbral).astype(int)
    return {
        "f1": float(f1_score(reales, pred, zero_division=0)),
        "precision": float(precision_score(reales, pred, zero_division=0)),
        "recall": float(recall_score(reales, pred, zero_division=0)),
        "matriz_confusion": confusion_matrix(reales, pred, labels=[0, 1]).tolist(),
        "predicciones_positivas": int(pred.sum()),
    }


def main():
    if not DATOS.exists():
        raise SystemExit(f"No encuentro {DATOS}")
    MODELOS.mkdir(exist_ok=True)
    datos = pd.read_csv(DATOS)
    datos["Month"] = pd.to_datetime(datos["FlightDate"]).dt.month
    resultados = []

    # F1 en validación determina el umbral elegido. Prueba NO participa.
    umbrales = np.arange(0.05, 0.951, 0.025)
    for problema, objetivo in OBJETIVOS.items():
        muestra = datos.loc[datos[objetivo].notna()].copy()
        valida = muestra.loc[muestra["Month"] == 10]
        prueba = muestra.loc[muestra["Month"] >= 11]
        y_val = valida[objetivo].astype(int).to_numpy()
        y_test = prueba[objetivo].astype(int).to_numpy()
        for algoritmo in ALGORITMOS:
            archivo = MODELOS / f"modelo_{problema}_{algoritmo}.joblib"
            if not archivo.exists():
                raise SystemExit(f"No encuentro {archivo}. Ejecuta antes entrenar_modelos_bts.py")
            modelo = joblib.load(archivo)  # Solo cargar modelos creados por ustedes.
            p_val = modelo.predict_proba(valida[ENTRADAS])[:, 1]
            p_test = modelo.predict_proba(prueba[ENTRADAS])[:, 1]
            # Desempate: un umbral mayor tiende a producir menos falsas alarmas.
            mejor = max(umbrales, key=lambda u: (f1_score(y_val, p_val >= u, zero_division=0), u))
            antes_val = medir(y_val, p_val, 0.5)
            despues_val = medir(y_val, p_val, mejor)
            antes_test = medir(y_test, p_test, 0.5)
            despues_test = medir(y_test, p_test, mejor)
            resultados.append({
                "problema": problema, "algoritmo": algoritmo,
                "umbral_inicial": 0.5, "umbral_optimizado": round(float(mejor), 3),
                "validacion_inicial": antes_val, "validacion_optimizada": despues_val,
                "prueba_inicial": antes_test, "prueba_optimizada": despues_test,
            })
            print(f"{problema:15} {algoritmo:21} umbral {mejor:.3f}  "
                  f"F1 prueba {antes_test['f1']:.3f} -> {despues_test['f1']:.3f}  "
                  f"precision {despues_test['precision']:.3f}  "
                  f"recall {despues_test['recall']:.3f}")

    informe = {
        "metodo": "Elegir umbral por máximo F1 en octubre de 2024; "
                  "evaluar una sola vez en noviembre y diciembre, sin reentrenar.",
        "advertencia": "Optimizar F1 no garantiza mejor desempeño en prueba; "
                       "la baja frecuencia de cancelaciones/desvíos limita la predicción.",
        "resultados": resultados,
    }
    (MODELOS / "reporte_optimizacion.json").write_text(
        json.dumps(informe, ensure_ascii=False, indent=2), encoding="utf-8")
    filas = []
    for r in resultados:
        for fase in ("validacion", "prueba"):
            for variante in ("inicial", "optimizada"):
                met = r[f"{fase}_{variante}"]
                filas.append({"problema": r["problema"], "algoritmo": r["algoritmo"],
                              "conjunto": fase, "version": variante,
                              "umbral": r["umbral_inicial" if variante == "inicial" else "umbral_optimizado"],
                              "f1": met["f1"], "precision": met["precision"],
                              "recall": met["recall"],
                              "predicciones_positivas": met["predicciones_positivas"]})
    pd.DataFrame(filas).to_csv(MODELOS / "comparacion_antes_despues.csv", index=False)
    print("\nGuardados reporte_optimizacion.json y comparacion_antes_despues.csv")


if __name__ == "__main__":
    main()
