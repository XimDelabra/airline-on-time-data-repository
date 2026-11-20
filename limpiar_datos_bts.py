"""Paso 2: validar y limpiar la muestra BTS de 2024 sin tocar el CSV original.

Ejecutar desde la raíz del repositorio: python limpiar_datos_bts.py
Requiere solo Python (sin instalar bibliotecas).
"""

import csv
import json
from collections import Counter
from datetime import date
from pathlib import Path

BASE = Path("datos_procesados_bts")
ENTRADA = BASE / "muestra_bts_2024_sin_limpiar.csv"
SALIDA = BASE / "vuelos_2024_limpios.csv"
REPORTE = BASE / "reporte_limpieza_2024.json"

CAMPOS = ["FlightDate", "Month", "DayOfWeek", "Reporting_Airline",
          "Origin", "Dest", "CRSDepTime", "DepartureHour", "Distance",
          "ArrDel15", "Cancelled", "Diverted"]


def indicador(valor):
    if valor == "":
        return ""
    numero = float(valor)
    if numero not in (0, 1):
        raise ValueError("Indicador diferente de 0 o 1")
    return str(int(numero))


def limpiar(fila):
    fecha = date.fromisoformat(fila["FlightDate"].strip())
    if fecha.year != 2024:
        raise ValueError("Fecha fuera de 2024")
    dia_semana = int(fila["DayOfWeek"])
    if dia_semana != fecha.isoweekday():
        raise ValueError("Día de la semana no coincide con fecha")

    hora_original = fila["CRSDepTime"].strip()
    if not hora_original.isdigit() or len(hora_original) > 4:
        raise ValueError("Hora de salida inválida")
    hora_num = int(hora_original)
    horas, minutos = divmod(hora_num, 100)
    # En BTS, 2400 representa medianoche al final del día.
    if hora_num == 2400:
        horas, minutos = 0, 0
    if horas > 23 or minutos > 59:
        raise ValueError("Hora de salida fuera de rango")

    distancia = float(fila["Distance"].strip())
    if not 0 < distancia < 10000:
        raise ValueError("Distancia fuera de rango")

    aerolinea = fila["Reporting_Airline"].strip().upper()
    origen = fila["Origin"].strip().upper()
    destino = fila["Dest"].strip().upper()
    if not aerolinea or not origen or not destino or origen == destino:
        raise ValueError("Aerolínea o ruta inválida")

    retraso = indicador(fila["ArrDel15"].strip())
    cancelado = indicador(fila["Cancelled"].strip())
    desviado = indicador(fila["Diverted"].strip())
    if cancelado == "" or desviado == "":
        raise ValueError("Falta indicador de cancelación o desvío")
    if cancelado == "1" and desviado == "1":
        raise ValueError("Vuelo cancelado y desviado simultáneamente")
    if (cancelado == "1" or desviado == "1") and retraso != "":
        raise ValueError("Llegada puntualidad definida para vuelo cancelado/desviado")
    if cancelado == "0" and desviado == "0" and retraso == "":
        raise ValueError("Falta retraso para vuelo completado")

    return {
        "FlightDate": fecha.isoformat(), "Month": fecha.month,
        "DayOfWeek": dia_semana, "Reporting_Airline": aerolinea,
        "Origin": origen, "Dest": destino,
        "CRSDepTime": f"{horas:02d}:{minutos:02d}",
        "DepartureHour": horas, "Distance": distancia,
        "ArrDel15": retraso, "Cancelled": cancelado, "Diverted": desviado,
    }


def main():
    if not ENTRADA.is_file():
        raise SystemExit(f"No encontré {ENTRADA}. Ejecuta este programa desde la raíz del repositorio.")
    estadisticas = Counter()
    descartes = Counter()
    vistos = set()
    with ENTRADA.open(encoding="utf-8-sig", newline="") as origen, \
            SALIDA.open("w", encoding="utf-8", newline="") as destino:
        lector = csv.DictReader(origen)
        escritor = csv.DictWriter(destino, fieldnames=CAMPOS)
        escritor.writeheader()
        for fila in lector:
            estadisticas["filas_entrada"] += 1
            try:
                limpia = limpiar(fila)
            except (ValueError, KeyError, TypeError) as error:
                descartes[str(error)] += 1
                continue
            clave = tuple(limpia.values())
            if clave in vistos:
                descartes["Duplicado exacto"] += 1
                continue
            vistos.add(clave)
            escritor.writerow(limpia)
            estadisticas["filas_conservadas"] += 1
            for campo in ("ArrDel15", "Cancelled", "Diverted"):
                estadisticas[f"{campo}_{limpia[campo] or 'vacio'}"] += 1

    informe = {
        "entrada": str(ENTRADA), "salida": str(SALIDA),
        "estadisticas": dict(estadisticas), "descartes": dict(descartes),
        "notas": [
            "ArrDel15 vacío es válido para vuelos cancelados/desviados; no se interpreta como cero.",
            "No usar ArrDel15, Cancelled ni Diverted entre sí como características predictoras.",
            "Distance conserva la unidad original del BTS: millas.",
            "El CSV resultante es una muestra mensual; no representa automáticamente todos los años.",
            "Los registros eliminados por validaciones deben revisarse antes de entrenar si son numerosos.",
        ],
    }
    REPORTE.write_text(json.dumps(informe, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Registros leídos: {estadisticas['filas_entrada']:,}")
    print(f"Registros conservados: {estadisticas['filas_conservadas']:,}")
    print(f"Descartes: {dict(descartes)}")
    print(f"Datos limpios: {SALIDA.resolve()}")
    print(f"Reporte: {REPORTE.resolve()}")
    print("Siguiente paso: dividir datos y entrenar modelos; esto aún NO entrena modelos.")


if __name__ == "__main__":
    main()
