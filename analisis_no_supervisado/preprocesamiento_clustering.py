from pathlib import Path
from zipfile import ZipFile
from collections import Counter
import json
import re

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

BASE = Path(__file__).resolve().parent
DATOS = BASE.parent / 'datos_crudos_bts' / 'zip'
SALIDA = BASE / 'datos_transformados'
CHUNK_SIZE = 100_000
MIN_VUELOS = 200
COLUMNAS = ['Reporting_Airline', 'Origin', 'Dest', 'DayOfWeek',
            'ArrDel15', 'ArrDelayMinutes', 'Cancelled', 'Diverted']
GRUPOS = {
    'rutas': ['Origin', 'Dest'],
    'destinos_dias': ['Dest', 'DayOfWeek'],
    'rutas_aerolinea': ['Reporting_Airline', 'Origin', 'Dest'],
    'destinos_dias_aerolinea': ['Reporting_Airline', 'Dest', 'DayOfWeek'],
}
CONTEOS = ['total_vuelos', 'llegadas_evaluables', 'retrasados',
           'retrasos_con_minutos', 'suma_minutos_retraso', 'cancelados', 'desviados']
VARIABLES = ['porcentaje_retrasados', 'promedio_minutos_retraso',
             'porcentaje_cancelados', 'porcentaje_desviados']


def identificar_archivos():
    archivos = []
    for ruta in DATOS.rglob('*.zip'):
        m = re.search(r'_(\d{4})_(\d{1,2})$', ruta.stem)
        if not m or not 1 <= int(m[2]) <= 12:
            raise ValueError(f'Nombre inesperado: {ruta}')
        archivos.append((int(m[1]), int(m[2]), ruta))
    archivos.sort()
    if not archivos:
        raise FileNotFoundError(f'No se encontraron ZIP en {DATOS}')
    if len({(a, m) for a, m, _ in archivos}) != len(archivos):
        raise ValueError('Hay meses duplicados')
    return archivos


def preparar_bloque(df):
    # No se eliminan cancelaciones ni desvíos del total de operaciones.
    for c in ['ArrDel15', 'ArrDelayMinutes', 'Cancelled', 'Diverted', 'DayOfWeek']:
        df[c] = pd.to_numeric(df[c], errors='raise')
    if df[['Reporting_Airline', 'Origin', 'Dest']].isna().any().any():
        raise ValueError('Identificadores de agrupación nulos')
    if not df['DayOfWeek'].isin(range(1, 8)).all():
        raise ValueError('Día de la semana inválido')
    for c in ['Cancelled', 'Diverted']:
        if not df[c].isin([0, 1]).all():
            raise ValueError(f'Indicador inválido: {c}')
    if not df['ArrDel15'].dropna().isin([0, 1]).all():
        raise ValueError('ArrDel15 inválido')
    minutos = df['ArrDelayMinutes']
    if not np.isfinite(minutos.dropna()).all() or minutos.lt(0).any():
        raise ValueError('Minutos de retraso inválidos')
    cancelados = df['Cancelled'].eq(1)
    desviados = df['Diverted'].eq(1)
    evaluables = ~cancelados & ~desviados & df['ArrDel15'].isin([0, 1])
    retrasados = evaluables & df['ArrDel15'].eq(1)
    comparables = evaluables & minutos.notna()
    if (comparables & (df['ArrDel15'].eq(1) != minutos.ge(15))).any():
        raise ValueError('Inconsistencia entre ArrDel15 y ArrDelayMinutes')
    con_minutos = retrasados & minutos.ge(15)
    df = df[list(GRUPOS['rutas_aerolinea']) + ['DayOfWeek']].copy()
    df['total_vuelos'] = 1
    df['llegadas_evaluables'] = evaluables.astype('int64')
    df['retrasados'] = retrasados.astype('int64')
    df['retrasos_con_minutos'] = con_minutos.astype('int64')
    df['suma_minutos_retraso'] = minutos.where(con_minutos, 0)
    df['cancelados'] = cancelados.astype('int64')
    df['desviados'] = desviados.astype('int64')
    return df


def exportar(nombre, tabla):
    tabla = tabla.reset_index().sort_values(GRUPOS[nombre]).reset_index(drop=True)
    enteros = [c for c in CONTEOS if c != 'suma_minutos_retraso']
    tabla[enteros] = tabla[enteros].astype('int64')
    suficiente = tabla.total_vuelos.ge(MIN_VUELOS)
    evaluable = tabla.llegadas_evaluables.gt(0)
    magnitud_valida = tabla.retrasados.eq(tabla.retrasos_con_minutos)
    tabla['porcentaje_retrasados'] = 100 * tabla.retrasados / tabla.llegadas_evaluables.replace(0, np.nan)
    tabla['promedio_minutos_retraso'] = tabla.suma_minutos_retraso / tabla.retrasos_con_minutos.replace(0, np.nan)
    tabla.loc[evaluable & tabla.retrasados.eq(0), 'promedio_minutos_retraso'] = 0.0
    tabla['porcentaje_cancelados'] = 100 * tabla.cancelados / tabla.total_vuelos
    tabla['porcentaje_desviados'] = 100 * tabla.desviados / tabla.total_vuelos
    tabla['motivo_exclusion'] = np.select(
        [~suficiente, ~evaluable, ~magnitud_valida],
        ['menos_de_200_vuelos', 'sin_llegadas_evaluables', 'retrasos_sin_minutos'], default='')
    validos = tabla.loc[tabla.motivo_exclusion.eq('')].drop(columns='motivo_exclusion').copy()
    excluidos = tabla.loc[tabla.motivo_exclusion.ne('')].copy()
    if validos.empty:
        raise ValueError(f'No hay agrupaciones válidas para {nombre}')
    if not np.isfinite(validos[VARIABLES].to_numpy()).all():
        raise ValueError(f'Indicadores no finitos en {nombre}')
    validos.to_csv(SALIDA / f'{nombre}.csv', index=False)
    excluidos.to_csv(SALIDA / f'{nombre}_excluidos.csv', index=False)
    scaler = StandardScaler()
    escalados = validos.copy()
    escalados[VARIABLES] = scaler.fit_transform(validos[VARIABLES])
    escalados.to_csv(SALIDA / f'{nombre}_estandarizado.csv', index=False)
    joblib.dump(scaler, SALIDA / f'{nombre}_scaler.joblib')
    return {
        'grupos_totales': len(tabla), 'grupos_conservados': len(validos),
        'grupos_excluidos': len(excluidos),
        'vuelos_totales': int(tabla.total_vuelos.sum()),
        'vuelos_conservados': int(validos.total_vuelos.sum()),
        'motivos_exclusion': {k: int(v) for k, v in Counter(excluidos.motivo_exclusion).items()},
        'estadisticas': validos[VARIABLES].describe().astype(object).where(lambda x: x.notna(), None).to_dict(),
        'media_scaler': dict(zip(VARIABLES, scaler.mean_.tolist())),
        'escala_scaler': dict(zip(VARIABLES, scaler.scale_.tolist())),
    }


def main():
    archivos = identificar_archivos()
    SALIDA.mkdir(parents=True, exist_ok=True)
    acumulados = {nombre: None for nombre in GRUPOS}
    total = 0
    for i, (anio, mes, ruta) in enumerate(archivos, 1):
        print(f'[{i}/{len(archivos)}] {anio}-{mes:02d}', flush=True)
        with ZipFile(ruta) as z:
            csvs = [n for n in z.namelist() if n.lower().endswith('.csv')]
            if len(csvs) != 1:
                raise ValueError(f'Se esperaba un CSV en {ruta}')
            with z.open(csvs[0]) as f:
                for chunk in pd.read_csv(f, usecols=COLUMNAS, chunksize=CHUNK_SIZE,
                                         low_memory=False):
                    total += len(chunk)
                    try:
                        bloque = preparar_bloque(chunk)
                    except ValueError as e:
                        raise ValueError(f'{ruta.name}: {e}') from e
                    for nombre, claves in GRUPOS.items():
                        parcial = bloque.groupby(claves, sort=False)[CONTEOS].sum()
                        anterior = acumulados[nombre]
                        acumulados[nombre] = parcial if anterior is None else anterior.add(parcial, fill_value=0)
        print(f'  Vuelos acumulados: {total:,}', flush=True)
    if not total:
        raise ValueError('Dataset vacío')
    resumen = {'archivos_procesados': len(archivos), 'total_vuelos': total,
               'periodo_inicio': f'{archivos[0][0]}-{archivos[0][1]:02d}',
               'periodo_fin': f'{archivos[-1][0]}-{archivos[-1][1]:02d}',
               'umbral_minimo': MIN_VUELOS, 'variables_modelo': VARIABLES,
               'agrupaciones': {}}
    for nombre, tabla in acumulados.items():
        if int(tabla.total_vuelos.sum()) != total:
            raise ValueError(f'El total no coincide en {nombre}')
        resumen['agrupaciones'][nombre] = exportar(nombre, tabla)
        print(nombre, resumen['agrupaciones'][nombre]['grupos_conservados'], 'grupos válidos')
    (SALIDA / 'resumen_preprocesamiento.json').write_text(
        json.dumps(resumen, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    print(f'Resultados guardados en {SALIDA}')


if __name__ == '__main__':
    main()
