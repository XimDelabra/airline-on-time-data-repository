"""Muestra BTS multianual: 2023-2025, sin modificar resultados anteriores.

Ejecutar desde la carpeta principal del repositorio:
    python preparar_datos_multianual_bts.py

Selecciona hasta 12.500 vuelos por mes (450.000 en 36 meses), valida los
campos y guarda un CSV aparte. NO entrena modelos.
"""

import csv
import io
import json
import random
import re
import zipfile
from collections import Counter
from datetime import date
from pathlib import Path

RAIZ = Path("datos_crudos_bts/zip")
DESTINO = Path("datos_procesados_bts/multianual")
SALIDA = DESTINO / "vuelos_2023_2025_limpios.csv"
REPORTE = DESTINO / "reporte_muestra_multianual.json"
ANIOS = (2023, 2024, 2025)
POR_MES = 12500
SEMILLA = 2026

ORIGEN = ["FlightDate", "DayOfWeek", "Reporting_Airline", "Origin", "Dest",
          "CRSDepTime", "Distance", "ArrDel15", "Cancelled", "Diverted"]
COLUMNAS = ["FlightDate", "Year", "Month", "DayOfWeek", "Reporting_Airline",
            "Origin", "Dest", "CRSDepTime", "DepartureHour", "Distance",
            "ArrDel15", "Cancelled", "Diverted"]


def archivos_por_anio(anio):
    carpeta = RAIZ / str(anio)
    archivos = list(carpeta.glob("*.zip"))
    meses = {}
    for archivo in archivos:
        encontrado = re.search(rf"_{anio}_(\d{{1,2}})\.zip$", archivo.name)
        if encontrado:
            mes = int(encontrado.group(1))
            if mes in meses:
                raise ValueError(f"Dos ZIP para {anio}-{mes:02d}")
            meses[mes] = archivo
    if set(meses) != set(range(1, 13)):
        raise ValueError(f"Faltan meses en {carpeta}: {sorted(set(range(1, 13)) - set(meses))}")
    return [(mes, meses[mes]) for mes in range(1, 13)]


def muestra_zip(archivo, semilla):
    randomizador = random.Random(semilla)
    muestra = []
    total = 0
    with zipfile.ZipFile(archivo) as z:
        csvs = [nombre for nombre in z.namelist() if nombre.lower().endswith(".csv")]
        if len(csvs) != 1:
            raise ValueError(f"Se esperaba un CSV en {archivo.name}")
        with z.open(csvs[0]) as contenido:
            lector = csv.DictReader(io.TextIOWrapper(contenido, encoding="utf-8-sig", newline=""))
            faltantes = set(ORIGEN) - set(lector.fieldnames or [])
            if faltantes:
                raise ValueError(f"Faltan columnas en {archivo.name}: {sorted(faltantes)}")
            for fila in lector:
                total += 1
                seleccion = {campo: (fila[campo] or "").strip() for campo in ORIGEN}
                if len(muestra) < POR_MES:
                    muestra.append(seleccion)
                else:
                    lugar = randomizador.randrange(total)
                    if lugar < POR_MES:
                        muestra[lugar] = seleccion
    return muestra, total


def bandera(texto, admite_vacio=False):
    if texto == "" and admite_vacio:
        return ""
    valor = float(texto)
    if valor not in (0, 1):
        raise ValueError("Indicador no binario")
    return int(valor)


def limpiar(fila, anio, mes):
    fecha = date.fromisoformat(fila["FlightDate"])
    if fecha.year != anio or fecha.month != mes or fecha.isoweekday() != int(fila["DayOfWeek"]):
        raise ValueError("Fecha incoherente")
    texto_hora = fila["CRSDepTime"]
    if not texto_hora.isdigit() or len(texto_hora) > 4:
        raise ValueError("Hora inválida")
    valor_hora = int(texto_hora)
    if valor_hora == 2400:
        hora, minuto = 0, 0
    else:
        hora, minuto = divmod(valor_hora, 100)
    if hora > 23 or minuto > 59:
        raise ValueError("Hora fuera de rango")
    distancia = float(fila["Distance"])
    if not 0 < distancia < 10000:
        raise ValueError("Distancia inválida")
    aerolinea, origen, destino = (fila[k].upper() for k in ("Reporting_Airline", "Origin", "Dest"))
    if not aerolinea or not origen or not destino or origen == destino:
        raise ValueError("Ruta o aerolínea inválida")
    retraso = bandera(fila["ArrDel15"], admite_vacio=True)
    cancelado = bandera(fila["Cancelled"])
    desviado = bandera(fila["Diverted"])
    if cancelado and desviado:
        raise ValueError("Cancelado y desviado simultáneamente")
    if (cancelado or desviado) and retraso != "":
        raise ValueError("Retraso no vacío en cancelado/desviado")
    if not (cancelado or desviado) and retraso == "":
        raise ValueError("Falta retraso de vuelo completado")
    return {"FlightDate": fecha.isoformat(), "Year": anio, "Month": mes,
            "DayOfWeek": fecha.isoweekday(), "Reporting_Airline": aerolinea,
            "Origin": origen, "Dest": destino,
            "CRSDepTime": f"{hora:02d}:{minuto:02d}", "DepartureHour": hora,
            "Distance": distancia, "ArrDel15": retraso,
            "Cancelled": cancelado, "Diverted": desviado}


def main():
    DESTINO.mkdir(parents=True, exist_ok=True)
    estadisticas = []
    descartes = Counter()
    with SALIDA.open("w", encoding="utf-8", newline="") as destino:
        escritor = csv.DictWriter(destino, fieldnames=COLUMNAS)
        escritor.writeheader()
        for anio in ANIOS:
            for mes, archivo in archivos_por_anio(anio):
                muestra, registros = muestra_zip(archivo, SEMILLA + anio * 100 + mes)
                guardados = 0
                positivos = Counter()
                for fila in muestra:
                    try:
                        limpio = limpiar(fila, anio, mes)
                    except (ValueError, KeyError) as error:
                        descartes[str(error)] += 1
                        continue
                    escritor.writerow(limpio)
                    guardados += 1
                    for campo in ("ArrDel15", "Cancelled", "Diverted"):
                        if limpio[campo] == 1:
                            positivos[campo] += 1
                estadisticas.append({"anio": anio, "mes": mes, "zip": archivo.name,
                                     "originales": registros, "muestreados": len(muestra),
                                     "limpios": guardados, "positivos": dict(positivos)})
                print(f"{anio}-{mes:02d}: {registros:,} originales, "
                      f"{len(muestra):,} en muestra, {guardados:,} limpios", flush=True)
    REPORTE.write_text(json.dumps({
        "periodos": {"entrenamiento": "2023-2024", "validacion": "enero-junio de 2025",
                     "prueba_no_utilizada": "julio-diciembre de 2025"},
        "muestreo": "reservorio aleatorio independiente de hasta 12500 vuelos por mes",
        "semilla": SEMILLA,
        "total_muestreado": sum(x["muestreados"] for x in estadisticas),
        "total_limpio": sum(x["limpios"] for x in estadisticas),
        "descartes": dict(descartes), "detalle": estadisticas,
        "nota": "Este CSV es una muestra de igual tamaño por mes, no todos los vuelos. "
                "Los vuelos cancelados/desviados mantienen ArrDel15 vacío."},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nDataset creado: {SALIDA.resolve()}")
    print(f"Reporte: {REPORTE.resolve()}")
    print("No se han entrenado modelos. Los datos anteriores no se modificaron.")


if __name__ == "__main__":
    main()
