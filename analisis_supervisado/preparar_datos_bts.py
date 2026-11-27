"""Paso 1 del proyecto BTS: inspeccionar ZIP y crear una muestra reproducible.

Uso desde la raíz del repositorio:
    python preparar_datos_bts.py

No modifica ni descomprime permanentemente los ZIP originales. Este script
NO entrena modelos ni constituye todavía la limpieza final de los datos.
"""

import argparse
import csv
import io
import json
import random
import zipfile
from collections import Counter
from pathlib import Path


COLUMNAS = [
    "FlightDate", "DayOfWeek", "Reporting_Airline", "Origin", "Dest",
    "CRSDepTime", "Distance", "ArrDel15", "Cancelled", "Diverted",
]
OBJETIVOS = ["ArrDel15", "Cancelled", "Diverted"]


def procesar_zip(archivo: Path, limite: int, semilla: int):
    """Selecciona hasta `limite` filas aleatorias por ZIP, con memoria acotada."""
    rng = random.Random(semilla)
    muestra = []
    total = 0
    conteos = {nombre: Counter() for nombre in OBJETIVOS}

    with zipfile.ZipFile(archivo) as comprimido:
        csvs = [n for n in comprimido.namelist() if n.lower().endswith(".csv")]
        if len(csvs) != 1:
            raise ValueError(f"Se esperaba un CSV en {archivo.name}; se encontraron {len(csvs)}")

        with comprimido.open(csvs[0]) as contenido:
            lector = csv.DictReader(io.TextIOWrapper(contenido, encoding="utf-8-sig", newline=""))
            faltantes = sorted(set(COLUMNAS) - set(lector.fieldnames or []))
            if faltantes:
                raise ValueError(f"En {archivo.name} faltan columnas: {', '.join(faltantes)}")

            for fila in lector:
                total += 1
                seleccion = {c: (fila[c] or "").strip() for c in COLUMNAS}
                for objetivo in OBJETIVOS:
                    conteos[objetivo][seleccion[objetivo] or "VACIO"] += 1
                # Muestreo de reservorio: cada fila tiene igual probabilidad de entrar.
                if len(muestra) < limite:
                    muestra.append(seleccion)
                else:
                    posicion = rng.randrange(total)
                    if posicion < limite:
                        muestra[posicion] = seleccion

    return muestra, {"archivo": archivo.name, "filas_originales": total,
                     "filas_muestra": len(muestra),
                     "valores_objetivo": {k: dict(v) for k, v in conteos.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path,
                        default=Path("datos_crudos_bts/zip/2024"),
                        help="Carpeta con los ZIP de 2024")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("datos_procesados_bts"),
                        help="Carpeta donde guardar la muestra y el reporte")
    parser.add_argument("--max-per-file", type=int, default=12000,
                        help="Máximo de vuelos seleccionados por archivo ZIP")
    parser.add_argument("--seed", type=int, default=42,
                        help="Semilla para reproducir la muestra")
    args = parser.parse_args()
    if args.max_per_file < 1:
        parser.error("--max-per-file debe ser mayor que cero")

    archivos = sorted(args.input_dir.glob("*.zip"))
    if not archivos:
        parser.error(f"No se encontraron ZIP en {args.input_dir.resolve()}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    salida = args.output_dir / "muestra_bts_2024_sin_limpiar.csv"
    reporte = args.output_dir / "reporte_exploracion_2024.json"
    detalles = []

    with salida.open("w", encoding="utf-8", newline="") as destino:
        escritor = csv.DictWriter(destino, fieldnames=COLUMNAS)
        escritor.writeheader()
        for indice, archivo in enumerate(archivos):
            print(f"Leyendo {archivo.name} ...", flush=True)
            muestra, resumen = procesar_zip(archivo, args.max_per_file, args.seed + indice)
            escritor.writerows(muestra)
            detalles.append(resumen)
            print(f"  {resumen['filas_originales']:,} originales; "
                  f"{resumen['filas_muestra']:,} en muestra")

    resumen_total = {
        "descripcion": "Muestra exploratoria SIN limpieza ni entrenamiento",
        "semilla": args.seed,
        "maximo_por_archivo": args.max_per_file,
        "columnas": COLUMNAS,
        "total_originales": sum(d["filas_originales"] for d in detalles),
        "total_muestra": sum(d["filas_muestra"] for d in detalles),
        "archivos": detalles,
    }
    reporte.write_text(json.dumps(resumen_total, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nMuestra creada: {salida.resolve()}")
    print(f"Reporte creado: {reporte.resolve()}")
    print("NOTA: no se han limpiado valores faltantes ni entrenado modelos.")


if __name__ == "__main__":
    main()
