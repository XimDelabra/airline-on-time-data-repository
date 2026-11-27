
from pathlib import Path
from collections import Counter
from zipfile import ZipFile

import json
import re
import time

import pandas as pd


# ==================================================
# CONFIGURACIÓN
# ==================================================

CARPETA_BASE = Path(__file__).resolve().parent
RAIZ_REPOSITORIO = CARPETA_BASE.parent

CARPETA_DATOS = RAIZ_REPOSITORIO / "datos_crudos_bts" / "zip"
CARPETA_RESULTADOS = CARPETA_BASE / "resultados_exploratorios"

CHUNK_SIZE = 100_000

UMBRALES = [
    5, 10, 20, 30, 50, 100,
    200, 500, 1000, 2000, 5000
]

COLUMNAS = [
    "Reporting_Airline",
    "Origin",
    "Dest",
    "DayOfWeek",
    "ArrDel15",
    "ArrDelayMinutes",
    "Cancelled",
    "Diverted"
]

COLUMNAS_BINARIAS = [
    "ArrDel15",
    "Cancelled",
    "Diverted"
]

COLUMNAS_CATEGORICAS = [
    "Reporting_Airline",
    "Origin",
    "Dest",
    "DayOfWeek"
]

GRUPOS = {
    "rutas": [
        "Origin",
        "Dest"
    ],
    "destinos_dias": [
        "Dest",
        "DayOfWeek"
    ],
    "rutas_aerolinea": [
        "Reporting_Airline",
        "Origin",
        "Dest"
    ],
    "destinos_dias_aerolinea": [
        "Reporting_Airline",
        "Dest",
        "DayOfWeek"
    ]
}


# ==================================================
# IDENTIFICACIÓN DE ARCHIVOS
# ==================================================

def identificar_archivos():

    if not CARPETA_DATOS.is_dir():
        raise FileNotFoundError(
            f"No existe la carpeta: {CARPETA_DATOS}"
        )

    archivos = []

    for archivo in CARPETA_DATOS.rglob("*.zip"):

        coincidencia = re.search(
            r"_(\d{4})_(\d{1,2})$",
            archivo.stem
        )

        if not coincidencia:
            raise ValueError(
                f"Nombre de archivo inesperado: {archivo}"
            )

        anio = int(coincidencia.group(1))
        mes = int(coincidencia.group(2))

        if not 1 <= mes <= 12:
            raise ValueError(
                f"Mes inválido en: {archivo}"
            )

        archivos.append((anio, mes, archivo))

    archivos.sort(key=lambda x: (x[0], x[1]))

    if not archivos:
        raise FileNotFoundError(
            "No se encontraron archivos ZIP."
        )

    periodos = [
        (anio, mes)
        for anio, mes, _ in archivos
    ]

    if len(periodos) != len(set(periodos)):
        raise ValueError(
            "Se encontraron archivos mensuales duplicados."
        )

    return archivos


# ==================================================
# ANÁLISIS DE DISTRIBUCIONES
# ==================================================

def analizar_distribucion(nombre, contador):

    serie = pd.Series(
        list(contador.values()),
        dtype="int64"
    )

    if serie.empty:

        print(f"Sin datos para {nombre}")

        return

    print(f"\n{'=' * 60}")
    print(nombre.upper())
    print("=" * 60)

    estadisticas = serie.describe(
        percentiles=[
            0.01,
            0.05,
            0.10,
            0.25,
            0.50,
            0.75,
            0.90,
            0.95,
            0.99
        ]
    )

    print("\nEstadísticas descriptivas:")
    print(estadisticas.to_string())

    total_grupos = len(serie)
    total_vuelos = int(serie.sum())

    resultados = []

    for umbral in UMBRALES:

        conservados = serie[serie >= umbral]
        descartados = serie[serie < umbral]

        resultados.append({

            "umbral": umbral,

            "grupos_conservados":
                len(conservados),

            "grupos_descartados":
                len(descartados),

            "porcentaje_grupos_conservados":
                len(conservados) / total_grupos * 100,

            "porcentaje_grupos_descartados":
                len(descartados) / total_grupos * 100,

            "vuelos_conservados":
                int(conservados.sum()),

            "vuelos_descartados":
                int(descartados.sum()),

            "porcentaje_vuelos_conservados":
                conservados.sum() / total_vuelos * 100,

            "porcentaje_vuelos_descartados":
                descartados.sum() / total_vuelos * 100

        })

    tabla_umbrales = pd.DataFrame(resultados).round(4)

    print("\nComparación de umbrales:")
    print(tabla_umbrales.to_string(index=False))

    tabla_umbrales.to_csv(
        CARPETA_RESULTADOS / f"umbrales_{nombre}.csv",
        index=False
    )

    estadisticas.to_csv(
        CARPETA_RESULTADOS / f"estadisticas_{nombre}.csv",
        header=["valor"]
    )

    registros = [
        (*clave, cantidad)
        for clave, cantidad in contador.items()
    ]

    tabla_grupos = pd.DataFrame(
        registros,
        columns=GRUPOS[nombre] + ["TotalFlights"]
    )

    tabla_grupos.sort_values(
        "TotalFlights",
        ascending=False
    ).to_csv(
        CARPETA_RESULTADOS / f"conteos_{nombre}.csv",
        index=False
    )


# ==================================================
# PROCESAMIENTO PRINCIPAL
# ==================================================

def main():

    tiempo_inicio = time.perf_counter()

    archivos = identificar_archivos()

    CARPETA_RESULTADOS.mkdir(
        parents=True,
        exist_ok=True
    )

    anios = sorted({
        anio for anio, _, _ in archivos
    })

    print("=" * 60)
    print("ANÁLISIS EXPLORATORIO NO SUPERVISADO - BTS")
    print("=" * 60)

    print(f"Carpeta de origen: {CARPETA_DATOS}")
    print(f"Archivos identificados: {len(archivos)}")

    print(
        f"Periodo inicial: "
        f"{archivos[0][0]}-{archivos[0][1]:02d}"
    )

    print(
        f"Periodo final: "
        f"{archivos[-1][0]}-{archivos[-1][1]:02d}"
    )

    print(f"Tamaño de bloque: {CHUNK_SIZE:,}")

    # ----------------------------------------------
    # INICIALIZACIÓN
    # ----------------------------------------------

    total_registros = 0

    registros_por_anio = Counter()
    registros_por_archivo = Counter()

    valores_nulos = Counter()
    valores_invalidos = Counter()

    estados_vuelos = Counter()

    valores_unicos = {
        columna: set()
        for columna in COLUMNAS_CATEGORICAS
    }

    agrupaciones = {
        nombre: Counter()
        for nombre in GRUPOS
    }

    registros_sin_agrupacion = Counter()

    archivos_procesados = []

    suma_minutos_retraso = 0.0
    cantidad_retrasos_con_minutos = 0

    # ----------------------------------------------
    # LECTURA DE ARCHIVOS ZIP
    # ----------------------------------------------

    for indice, (anio, mes, archivo) in enumerate(
        archivos,
        start=1
    ):

        inicio_archivo = time.perf_counter()

        registros_archivo = 0

        print(
            f"\n[{indice}/{len(archivos)}] "
            f"Procesando {anio}-{mes:02d}"
        )

        with ZipFile(archivo, "r") as zip_file:

            archivos_csv = [
                nombre
                for nombre in zip_file.namelist()
                if nombre.lower().endswith(".csv")
            ]

            if len(archivos_csv) != 1:

                raise ValueError(
                    f"Se esperaba exactamente un CSV en "
                    f"{archivo.name}. "
                    f"Encontrados: {len(archivos_csv)}"
                )

            nombre_csv = archivos_csv[0]

            # Verificar columnas disponibles
            with zip_file.open(nombre_csv) as csv_file:

                encabezados = pd.read_csv(
                    csv_file,
                    nrows=0
                ).columns.tolist()

            columnas_faltantes = [
                columna
                for columna in COLUMNAS
                if columna not in encabezados
            ]

            if columnas_faltantes:

                raise ValueError(
                    f"Columnas faltantes en {archivo.name}: "
                    f"{columnas_faltantes}"
                )

            # Leer el CSV por bloques
            with zip_file.open(nombre_csv) as csv_file:

                lector = pd.read_csv(
                    csv_file,
                    usecols=COLUMNAS,
                    chunksize=CHUNK_SIZE,
                    low_memory=False
                )

                for chunk in lector:

                    cantidad = len(chunk)

                    total_registros += cantidad
                    registros_archivo += cantidad

                    registros_por_anio[anio] += cantidad

                    registros_por_archivo[
                        archivo.name
                    ] += cantidad

                    # ------------------------------
                    # VALORES NULOS
                    # ------------------------------

                    valores_nulos.update(
                        chunk.isna().sum().to_dict()
                    )

                    # ------------------------------
                    # VALORES ÚNICOS
                    # ------------------------------

                    for columna in COLUMNAS_CATEGORICAS:

                        valores_unicos[columna].update(
                            chunk[columna].dropna().unique()
                        )

                    # ------------------------------
                    # VARIABLES BINARIAS
                    # ------------------------------

                    for columna in COLUMNAS_BINARIAS:

                        invalidos = (
                            chunk[columna].notna()
                            & ~chunk[columna].isin([0, 1])
                        )

                        valores_invalidos[columna] += int(
                            invalidos.sum()
                        )

                    # ------------------------------
                    # DÍAS DE LA SEMANA
                    # ------------------------------

                    dias_invalidos = (
                        chunk["DayOfWeek"].notna()
                        & ~chunk["DayOfWeek"].isin(
                            range(1, 8)
                        )
                    )

                    valores_invalidos["DayOfWeek"] += int(
                        dias_invalidos.sum()
                    )

                    # ------------------------------
                    # MINUTOS DE RETRASO
                    # ------------------------------

                    minutos = pd.to_numeric(
                        chunk["ArrDelayMinutes"],
                        errors="coerce"
                    )

                    minutos_invalidos = (
                        chunk["ArrDelayMinutes"].notna()
                        & (
                            minutos.isna()
                            | minutos.lt(0)
                        )
                    )

                    valores_invalidos[
                        "ArrDelayMinutes"
                    ] += int(minutos_invalidos.sum())

                    # ------------------------------
                    # ESTADOS OPERATIVOS
                    # ------------------------------

                    cancelados = chunk["Cancelled"].eq(1)

                    desviados = chunk["Diverted"].eq(1)

                    llegadas_evaluables = (
                        chunk["Cancelled"].eq(0)
                        & chunk["Diverted"].eq(0)
                        & chunk["ArrDel15"].isin([0, 1])
                    )

                    retrasados = (
                        llegadas_evaluables
                        & chunk["ArrDel15"].eq(1)
                    )

                    puntuales = (
                        llegadas_evaluables
                        & chunk["ArrDel15"].eq(0)
                    )

                    estados_vuelos["cancelados"] += int(
                        cancelados.sum()
                    )

                    estados_vuelos["desviados"] += int(
                        desviados.sum()
                    )

                    estados_vuelos[
                        "llegadas_evaluables"
                    ] += int(llegadas_evaluables.sum())

                    estados_vuelos["retrasados"] += int(
                        retrasados.sum()
                    )

                    estados_vuelos["puntuales"] += int(
                        puntuales.sum()
                    )

                    estados_vuelos[
                        "llegadas_no_evaluables"
                    ] += int(
                        (~llegadas_evaluables).sum()
                    )

                    # ------------------------------
                    # MAGNITUD DE LOS RETRASOS
                    # ------------------------------

                    retrasos_con_minutos = (
                        retrasados
                        & minutos.notna()
                        & minutos.ge(15)
                    )

                    cantidad_retrasos_con_minutos += int(
                        retrasos_con_minutos.sum()
                    )

                    suma_minutos_retraso += float(
                        minutos[retrasos_con_minutos].sum()
                    )

                    # Comprobar consistencia entre
                    # ArrDel15 y ArrDelayMinutes

                    inconsistencias_retraso = (
                        llegadas_evaluables
                        & minutos.notna()
                        & minutos.ge(0)
                        & (
                            chunk["ArrDel15"].eq(1)
                            != minutos.ge(15)
                        )
                    )

                    valores_invalidos[
                        "inconsistencia_retraso_15_minutos"
                    ] += int(
                        inconsistencias_retraso.sum()
                    )

                    # ------------------------------
                    # AGRUPACIONES
                    # ------------------------------

                    for nombre, columnas in GRUPOS.items():

                        mascara_valida = (
                            chunk[columnas]
                            .notna()
                            .all(axis=1)
                        )

                        if "DayOfWeek" in columnas:

                            mascara_valida &= (
                                chunk["DayOfWeek"]
                                .isin(range(1, 8))
                            )

                        registros_sin_agrupacion[
                            nombre
                        ] += int(
                            (~mascara_valida).sum()
                        )

                        conteos = (
                            chunk.loc[mascara_valida]
                            .groupby(columnas)
                            .size()
                            .to_dict()
                        )

                        agrupaciones[nombre].update(
                            conteos
                        )

        duracion_archivo = (
            time.perf_counter() - inicio_archivo
        )

        archivos_procesados.append({
            "anio": anio,
            "mes": mes,
            "archivo": archivo.name,
            "registros": registros_archivo,
            "tiempo_segundos": round(
                duracion_archivo,
                2
            )
        })

        print(
            f"Registros del archivo: "
            f"{registros_archivo:,}"
        )

        print(
            f"Registros acumulados: "
            f"{total_registros:,}"
        )

        print(
            f"Tiempo de procesamiento: "
            f"{duracion_archivo:.2f} segundos"
        )

    # ----------------------------------------------
    # VALIDACIÓN FINAL
    # ----------------------------------------------

    if total_registros == 0:

        raise ValueError(
            "No se encontraron registros."
        )

    # ----------------------------------------------
    # INDICADORES GENERALES
    # ----------------------------------------------

    llegadas_evaluables = estados_vuelos[
        "llegadas_evaluables"
    ]

    porcentaje_retrasados = (
        estados_vuelos["retrasados"]
        / llegadas_evaluables * 100
        if llegadas_evaluables > 0
        else None
    )

    promedio_minutos_retraso = (
        suma_minutos_retraso
        / cantidad_retrasos_con_minutos
        if cantidad_retrasos_con_minutos > 0
        else None
    )

    porcentaje_cancelados = (
        estados_vuelos["cancelados"]
        / total_registros * 100
    )

    porcentaje_desviados = (
        estados_vuelos["desviados"]
        / total_registros * 100
    )

    indicadores = {

        "porcentaje_vuelos_retrasados":
            porcentaje_retrasados,

        "promedio_minutos_retraso":
            promedio_minutos_retraso,

        "porcentaje_vuelos_cancelados":
            porcentaje_cancelados,

        "porcentaje_vuelos_desviados":
            porcentaje_desviados

    }

    indicadores = {
        nombre: round(valor, 4)
        if valor is not None
        else None

        for nombre, valor in indicadores.items()
    }

    # ----------------------------------------------
    # DISTRIBUCIONES POR AGRUPACIÓN
    # ----------------------------------------------

    for nombre, contador in agrupaciones.items():

        analizar_distribucion(
            nombre,
            contador
        )

    # ----------------------------------------------
    # RESUMEN GENERAL
    # ----------------------------------------------

    tiempo_total = (
        time.perf_counter() - tiempo_inicio
    )

    resumen = {

        "periodo": {

            "inicio":
                f"{archivos[0][0]}-{archivos[0][1]:02d}",

            "fin":
                f"{archivos[-1][0]}-{archivos[-1][1]:02d}",

            "anios": anios

        },

        "archivos_procesados": len(
            archivos_procesados
        ),

        "total_registros": total_registros,

        "registros_por_anio": dict(
            sorted(registros_por_anio.items())
        ),

        "registros_por_archivo": dict(
            registros_por_archivo
        ),

        "valores_nulos": {

            columna: {

                "cantidad": int(
                    valores_nulos[columna]
                ),

                "porcentaje": round(
                    valores_nulos[columna]
                    / total_registros * 100,
                    4
                )

            }

            for columna in COLUMNAS

        },

        "valores_invalidos": {

            columna: int(cantidad)

            for columna, cantidad
            in valores_invalidos.items()

        },

        "cantidad_valores_unicos": {

            columna: len(valores)

            for columna, valores
            in valores_unicos.items()

        },

        "estados_vuelos": dict(
            estados_vuelos
        ),

        "indicadores_generales": indicadores,

        "retrasos_con_minutos_validos":
            cantidad_retrasos_con_minutos,

        "cantidad_agrupaciones": {

            nombre: len(contador)

            for nombre, contador
            in agrupaciones.items()

        },

        "registros_sin_agrupacion": dict(
            registros_sin_agrupacion
        ),

        "tiempo_total_segundos": round(
            tiempo_total,
            2
        ),

        "tiempo_total_minutos": round(
            tiempo_total / 60,
            2
        )

    }

    # ----------------------------------------------
    # EXPORTACIÓN DE RESULTADOS
    # ----------------------------------------------

    with open(
        CARPETA_RESULTADOS / "resumen_exploratorio.json",
        "w",
        encoding="utf-8"
    ) as archivo:

        json.dump(
            resumen,
            archivo,
            indent=4,
            ensure_ascii=False
        )

    pd.DataFrame(
        archivos_procesados
    ).to_csv(
        CARPETA_RESULTADOS / "archivos_procesados.csv",
        index=False
    )

    pd.DataFrame([

        {
            "anio": anio,
            "total_vuelos": cantidad
        }

        for anio, cantidad
        in sorted(registros_por_anio.items())

    ]).to_csv(
        CARPETA_RESULTADOS / "registros_por_anio.csv",
        index=False
    )

    # ----------------------------------------------
    # RESULTADOS EN TERMINAL
    # ----------------------------------------------

    print("\n" + "=" * 60)
    print("RESUMEN GENERAL")
    print("=" * 60)

    print(f"Archivos procesados: {len(archivos_procesados)}")
    print(f"Total de vuelos: {total_registros:,}")

    print("\nIndicadores generales:")

    for nombre, valor in indicadores.items():

        print(f"{nombre}: {valor}")

    print("\nCantidad de agrupaciones:")

    for nombre, contador in agrupaciones.items():

        print(f"{nombre}: {len(contador):,}")

    print(
        f"\nTiempo total: "
        f"{tiempo_total / 60:.2f} minutos"
    )
    
    print(
        f"Resultados guardados en: "
        f"{CARPETA_RESULTADOS}"
    )

    print("\nAnálisis exploratorio finalizado.")


# ==================================================
# PUNTO DE ENTRADA
# ==================================================

if __name__ == "__main__":

    main()
