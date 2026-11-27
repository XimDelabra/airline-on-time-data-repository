"""Mejora 1: añadir variables a LOS MISMOS vuelos de la muestra anterior.

Ejecutar desde la raíz del repositorio:
    python preparar_datos_bts_v2.py

No modifica los ZIP, datasets, modelos ni reportes de la versión 1.
Requiere solo la biblioteca estándar de Python.
"""

import csv
import io
import json
import random
import zipfile
from collections import Counter
from datetime import date
from pathlib import Path

ENTRADA = Path("datos_crudos_bts/zip/2024")
SALIDA = Path("datos_procesados_bts")
ANTERIOR = SALIDA / "muestra_bts_2024_sin_limpiar.csv"
MUESTRA = SALIDA / "muestra_bts_2024_v2_sin_limpiar.csv"
LIMPIO = SALIDA / "vuelos_2024_limpios_v2.csv"
REPORTE = SALIDA / "reporte_preparacion_v2.json"

ORIGINALES = ["FlightDate", "DayOfWeek", "Reporting_Airline", "Origin", "Dest",
              "CRSDepTime", "Distance", "ArrDel15", "Cancelled", "Diverted"]
NUEVAS = ["CRSArrTime", "CRSElapsedTime", "DayofMonth"]
COLUMNAS = ORIGINALES + NUEVAS
LIMITE = 12000
SEMILLA = 42


def muestrear_zip(archivo, semilla):
    """Misma selección por reservorio, semillas y orden que preparar_datos_bts.py."""
    rng = random.Random(semilla)
    muestra = []
    total = 0
    with zipfile.ZipFile(archivo) as comprimido:
        csvs = [n for n in comprimido.namelist() if n.lower().endswith(".csv")]
        if len(csvs) != 1:
            raise ValueError(f"ZIP con {len(csvs)} CSV: {archivo.name}")
        with comprimido.open(csvs[0]) as archivo_csv:
            lector = csv.DictReader(io.TextIOWrapper(archivo_csv, encoding="utf-8-sig", newline=""))
            faltan = set(COLUMNAS) - set(lector.fieldnames or [])
            if faltan:
                raise ValueError(f"Faltan columnas en {archivo.name}: {sorted(faltan)}")
            for fila in lector:
                total += 1
                seleccion = {k: (fila[k] or "").strip() for k in COLUMNAS}
                if len(muestra) < LIMITE:
                    muestra.append(seleccion)
                else:
                    posicion = rng.randrange(total)
                    if posicion < LIMITE:
                        muestra[posicion] = seleccion
    return muestra, total


def hora_bts(valor):
    valor = valor.strip()
    if not valor.isdigit() or len(valor) > 4:
        raise ValueError("Hora inválida")
    numero = int(valor)
    if numero == 2400:
        return 0, "00:00"
    h, m = divmod(numero, 100)
    if h > 23 or m > 59:
        raise ValueError("Hora fuera de rango")
    return h, f"{h:02d}:{m:02d}"


def binario(valor, permite_vacio=False):
    if not valor and permite_vacio:
        return ""
    numero = float(valor)
    if numero not in (0, 1):
        raise ValueError("Indicador no binario")
    return int(numero)


def limpiar(f):
    fecha = date.fromisoformat(f["FlightDate"])
    dia = int(f["DayOfWeek"])
    if fecha.year != 2024 or fecha.isoweekday() != dia or fecha.day != int(f["DayofMonth"]):
        raise ValueError("Fecha o día incoherente")
    salida_hora, salida_texto = hora_bts(f["CRSDepTime"])
    llegada_hora, llegada_texto = hora_bts(f["CRSArrTime"])
    duracion = float(f["CRSElapsedTime"])
    distancia = float(f["Distance"])
    if not (0 < duracion <= 1800 and 0 < distancia < 10000):
        raise ValueError("Duración o distancia fuera de rango")
    compania, origen, destino = [f[k].upper() for k in ("Reporting_Airline", "Origin", "Dest")]
    if not compania or not origen or not destino or origen == destino:
        raise ValueError("Ruta o aerolínea inválida")
    retraso = binario(f["ArrDel15"], permite_vacio=True)
    cancelado = binario(f["Cancelled"])
    desviado = binario(f["Diverted"])
    if cancelado and desviado:
        raise ValueError("Cancelado y desviado simultáneamente")
    if (cancelado or desviado) and retraso != "":
        raise ValueError("Retraso indicado para vuelo cancelado/desviado")
    if not (cancelado or desviado) and retraso == "":
        raise ValueError("Falta retraso en vuelo completado")
    return {
        "FlightDate": fecha.isoformat(), "Month": fecha.month,
        "DayofMonth": fecha.day, "DayOfWeek": dia,
        "Reporting_Airline": compania, "Origin": origen, "Dest": destino,
        "CRSDepTime": salida_texto, "DepartureHour": salida_hora,
        "CRSArrTime": llegada_texto, "ArrivalHour": llegada_hora,
        "CRSElapsedTime": duracion, "Distance": distancia,
        "ArrDel15": retraso, "Cancelled": cancelado, "Diverted": desviado,
    }


def main():
    archivos = sorted(ENTRADA.glob("*.zip"))
    if len(archivos) != 12:
        raise SystemExit(f"Se esperaban 12 ZIP de 2024 en {ENTRADA}; encontré {len(archivos)}")
    SALIDA.mkdir(parents=True, exist_ok=True)
    # Se verifica fila por fila que realmente sean los vuelos anteriores.
    if not ANTERIOR.is_file():
        raise SystemExit(f"Falta {ANTERIOR}. Se necesita para verificar que la muestra no cambió.")

    columnas_limpias = [
        "FlightDate", "Month", "DayofMonth", "DayOfWeek", "Reporting_Airline",
        "Origin", "Dest", "CRSDepTime", "DepartureHour", "CRSArrTime",
        "ArrivalHour", "CRSElapsedTime", "Distance", "ArrDel15", "Cancelled", "Diverted"]
    conteos = Counter()
    descartes = Counter()
    with ANTERIOR.open(encoding="utf-8-sig", newline="") as origen, \
         MUESTRA.open("w", encoding="utf-8", newline="") as nueva_muestra, \
         LIMPIO.open("w", encoding="utf-8", newline="") as nuevo_limpio:
        anterior = csv.DictReader(origen)
        muestra_writer = csv.DictWriter(nueva_muestra, fieldnames=COLUMNAS)
        limpio_writer = csv.DictWriter(nuevo_limpio, fieldnames=columnas_limpias)
        muestra_writer.writeheader()
        limpio_writer.writeheader()
        for indice, archivo in enumerate(archivos):
            filas, total = muestrear_zip(archivo, SEMILLA + indice)
            print(f"{archivo.name}: {total:,} registros; {len(filas):,} seleccionados", flush=True)
            for fila in filas:
                previa = next(anterior, None)
                if previa is None or any(previa[k] != fila[k] for k in ORIGINALES):
                    raise SystemExit("ERROR: las muestras no coinciden. No entrenes con esta versión.")
                muestra_writer.writerow(fila)
                conteos["seleccionados"] += 1
                try:
                    limpio_writer.writerow(limpiar(fila))
                    conteos["limpios"] += 1
                except (ValueError, KeyError) as error:
                    descartes[str(error)] += 1
        if next(anterior, None) is not None:
            raise SystemExit("ERROR: sobraron registros en la muestra anterior")

    REPORTE.write_text(json.dumps({
        "metodo": "Mismo muestreo aleatorio reproducible de 12.000 por mes que versión 1",
        "verificacion_mismas_filas": True,
        "nuevas_columnas": NUEVAS, "conteos": dict(conteos),
        "descartes": dict(descartes),
        "nota": "En esta prueba las nuevas columnas deben conocerse antes del vuelo; "
                "la duración y los horarios son programados, no reales."},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nV2 lista: {LIMPIO.resolve()}")
    print(f"Registros seleccionados: {conteos['seleccionados']:,}; limpios: {conteos['limpios']:,}")
    print(f"Descartes: {dict(descartes)}")
    print("Archivos originales y modelos V1 conservados sin cambios.")


if __name__ == "__main__":
    main()
