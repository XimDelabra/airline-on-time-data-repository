"""Implementación, evaluación y optimización de DBSCAN para BTS."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import (silhouette_score, calinski_harabasz_score,
                             davies_bouldin_score)
from sklearn.neighbors import NearestNeighbors
from threadpoolctl import threadpool_limits

BASE = Path(__file__).resolve().parent
DATOS = BASE / 'datos_transformados'
SALIDA = BASE / 'modelos' / 'dbscan'
VARIABLES = ['porcentaje_retrasados', 'promedio_minutos_retraso',
             'porcentaje_cancelados', 'porcentaje_desviados']
GRUPOS = {
    'rutas': ['Origin', 'Dest'],
    'destinos_dias': ['Dest', 'DayOfWeek'],
    'rutas_aerolinea': ['Reporting_Airline', 'Origin', 'Dest'],
    'destinos_dias_aerolinea': ['Reporting_Airline', 'Dest', 'DayOfWeek'],
}
INICIAL = (0.5, 8)
EPS = [i / 10 for i in range(1, 21)]
MIN_SAMPLES = [4, 8, 12, 16, 20]
CONFIG = dict(metric='euclidean', algorithm='auto', n_jobs=-1)


def cargar(nombre):
    claves = GRUPOS[nombre]
    original = pd.read_csv(DATOS / f'{nombre}.csv')
    escalado = pd.read_csv(DATOS / f'{nombre}_estandarizado.csv')
    scaler = joblib.load(DATOS / f'{nombre}_scaler.joblib')
    for tabla in (original, escalado):
        if tabla[claves].isna().any().any() or tabla.duplicated(claves).any():
            raise ValueError(f'Identificadores nulos o duplicados: {nombre}')
    if not original[claves].equals(escalado[claves]):
        raise ValueError(f'Identificadores no coinciden: {nombre}')
    X = escalado[VARIABLES].to_numpy(dtype=float)
    if len(X) < 21 or not np.isfinite(X).all():
        raise ValueError(f'Datos insuficientes o no finitos: {nombre}')
    if not np.allclose(X, scaler.transform(original[VARIABLES]), atol=1e-8):
        raise ValueError(f'Escalado inconsistente: {nombre}')
    return original, X


def ajustar(X, eps, min_samples):
    return DBSCAN(eps=eps, min_samples=min_samples, **CONFIG).fit(X)


def evaluar(X, modelo, eps, min_samples):
    etiquetas = modelo.labels_
    mascara = etiquetas != -1
    n = int(mascara.sum())
    k = len(np.unique(etiquetas[mascara]))
    valido = 2 <= k < n
    # El ruido no constituye un clúster. Las métricas son exactas y
    # utilizan únicamente los puntos agrupados, sin muestreo.
    return dict(eps=eps, min_samples=min_samples, clusters_obtenidos=k,
                puntos_agrupados=n, puntos_ruido=len(X)-n,
                porcentaje_ruido=100*(len(X)-n)/len(X),
                silhouette=float(silhouette_score(X[mascara], etiquetas[mascara])) if valido else None,
                calinski_harabasz=float(calinski_harabasz_score(X[mascara], etiquetas[mascara])) if valido else None,
                davies_bouldin=float(davies_bouldin_score(X[mascara], etiquetas[mascara])) if valido else None)


def exportar(carpeta, etapa, modelo, original):
    tabla = original.copy()
    tabla['cluster'] = modelo.labels_
    tabla.to_csv(carpeta / f'asignaciones_{etapa}.csv', index=False)
    tabla.loc[tabla.cluster == -1].to_csv(carpeta / f'ruido_{etapa}.csv', index=False)
    perfil = tabla.groupby('cluster')[VARIABLES].mean()
    perfil.insert(0, 'cantidad_grupos', tabla.groupby('cluster').size())
    perfil['total_vuelos'] = tabla.groupby('cluster')['total_vuelos'].sum()
    perfil.to_csv(carpeta / f'perfiles_{etapa}.csv')
    joblib.dump(modelo, carpeta / f'modelo_{etapa}.joblib')


def ejecutar(nombre, max_ruido):
    inicio = perf_counter()
    carpeta = SALIDA / nombre
    carpeta.mkdir(parents=True, exist_ok=True)
    original, X = cargar(nombre)
    # La distancia al vecino incluye el propio punto, como min_samples.
    vecinos = NearestNeighbors(n_neighbors=max(MIN_SAMPLES), **CONFIG).fit(X)
    distancias, _ = vecinos.kneighbors(X)
    pd.DataFrame({f'min_samples_{m}': np.sort(distancias[:, m-1])
                  for m in MIN_SAMPLES}).to_csv(carpeta / 'distancias_vecinos.csv', index=False)
    resultados = []
    configuraciones = [INICIAL] + [(e, m) for e in EPS for m in MIN_SAMPLES if (e, m) != INICIAL]
    inicial = ajustar(X, *INICIAL)
    for i, (eps, m) in enumerate(configuraciones, 1):
        modelo = inicial if (eps, m) == INICIAL else ajustar(X, eps, m)
        fila = evaluar(X, modelo, eps, m)
        resultados.append(fila)
        print(f'{nombre} [{i}/{len(configuraciones)}] eps={eps:.1f} min_samples={m} | clusters={fila["clusters_obtenidos"]} | ruido={fila["porcentaje_ruido"]:.2f}%', flush=True)
        del modelo
    tabla = pd.DataFrame(resultados).sort_values(['eps', 'min_samples'])
    metricas = ['silhouette', 'calinski_harabasz', 'davies_bouldin']
    candidatos = tabla.dropna(subset=metricas).copy()
    candidatos = candidatos[candidatos.porcentaje_ruido <= max_ruido].copy()
    tabla['elegible'] = tabla.index.isin(candidatos.index)
    tabla.to_csv(carpeta / 'metricas_busqueda.csv', index=False)
    # Igual peso para las tres métricas y la cobertura de puntos.
    for columna, asc in [('silhouette', False), ('calinski_harabasz', False),
                         ('davies_bouldin', True), ('porcentaje_ruido', True)]:
        candidatos[f'rango_{columna}'] = candidatos[columna].rank(ascending=asc, method='min')
    candidatos['rango_total'] = candidatos[[f'rango_{c}' for c in metricas + ['porcentaje_ruido']]].sum(axis=1)
    candidatos = candidatos.sort_values(['rango_total', 'silhouette', 'porcentaje_ruido', 'eps', 'min_samples'],
                                        ascending=[True, False, True, True, True])
    candidatos.to_csv(carpeta / 'ranking_configuraciones.csv', index=False)
    exportar(carpeta, 'inicial', inicial, original)
    comparacion = [{'etapa': 'inicial', **resultados[0]}]
    mejor = None
    if not candidatos.empty:
        mejor = (float(candidatos.iloc[0].eps), int(candidatos.iloc[0].min_samples))
        optimizado = inicial if mejor == INICIAL else ajustar(X, *mejor)
        exportar(carpeta, 'optimizado', optimizado, original)
        comparacion.append({'etapa': 'optimizado', **next(r for r in resultados if (r['eps'], r['min_samples']) == mejor)})
    else:
        # No publicar una configuración inválida ni reutilizar resultados antiguos.
        for archivo in carpeta.glob('*_optimizado.*'):
            archivo.unlink()
        print(f'AVISO: {nombre}: ninguna configuración cumple los criterios de selección.', flush=True)
    pd.DataFrame(comparacion).to_csv(carpeta / 'comparacion.csv', index=False)
    return dict(dataset=nombre, agrupaciones=len(X), vuelos=int(original.total_vuelos.sum()),
                configuracion_inicial=dict(eps=INICIAL[0], min_samples=INICIAL[1]),
                configuracion_optimizada=dict(eps=mejor[0], min_samples=mejor[1]) if mejor else None,
                comparacion=comparacion, tiempo_segundos=round(perf_counter()-inicio, 2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--incluir-aerolineas', action='store_true')
    parser.add_argument('--max-ruido', type=float, default=30.0,
                        help='Porcentaje máximo de ruido admitido para seleccionar el modelo (predeterminado: 30)')
    args = parser.parse_args()
    if not 0 <= args.max_ruido < 100:
        parser.error('--max-ruido debe estar entre 0 y menos de 100')
    nombres = list(GRUPOS) if args.incluir_aerolineas else ['rutas', 'destinos_dias']
    SALIDA.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=2):
        resultados = [ejecutar(n, args.max_ruido) for n in nombres]
    resumen = dict(variables=VARIABLES, configuracion=CONFIG, rango_eps=EPS,
                   valores_min_samples=MIN_SAMPLES, maximo_porcentaje_ruido=args.max_ruido,
                   evaluacion='Métricas exactas sobre puntos agrupados; ruido reportado por separado',
                   seleccion='Suma de rangos de tres métricas y porcentaje de ruido, con igual peso; al menos dos clústeres y límite de ruido',
                   resultados=resultados)
    (SALIDA / 'resumen_dbscan.json').write_text(json.dumps(resumen, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    print(json.dumps(resumen, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
