"""Implementación, evaluación y optimización de K-Means para BTS."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (silhouette_score, calinski_harabasz_score,
                             davies_bouldin_score)
from threadpoolctl import threadpool_limits

BASE = Path(__file__).resolve().parent
DATOS = BASE / 'datos_transformados'
SALIDA = BASE / 'modelos' / 'kmeans'
VARIABLES = ['porcentaje_retrasados', 'promedio_minutos_retraso',
             'porcentaje_cancelados', 'porcentaje_desviados']
GRUPOS = {
    'rutas': ['Origin', 'Dest'],
    'destinos_dias': ['Dest', 'DayOfWeek'],
    'rutas_aerolinea': ['Reporting_Airline', 'Origin', 'Dest'],
    'destinos_dias_aerolinea': ['Reporting_Airline', 'Dest', 'DayOfWeek'],
}
K_INICIAL = 3
SEMILLA = 42
CONFIG = dict(init='k-means++', n_init=10, max_iter=300,
              random_state=SEMILLA, tol=0.0001)


def cargar(nombre):
    claves = GRUPOS[nombre]
    original = pd.read_csv(DATOS / f'{nombre}.csv')
    escalado = pd.read_csv(DATOS / f'{nombre}_estandarizado.csv')
    scaler = joblib.load(DATOS / f'{nombre}_scaler.joblib')
    for tabla in (original, escalado):
        if tabla[claves].isna().any().any() or tabla.duplicated(claves).any():
            raise ValueError(f'Identificadores nulos o duplicados: {nombre}')
    if not original[claves].equals(escalado[claves]):
        raise ValueError(f'Los identificadores no coinciden: {nombre}')
    X = escalado[VARIABLES].to_numpy(dtype=float)
    if len(X) < 16 or not np.isfinite(X).all():
        raise ValueError(f'Datos insuficientes o no finitos: {nombre}')
    if not np.allclose(X, scaler.transform(original[VARIABLES]), atol=1e-8):
        raise ValueError(f'El escalado no coincide con los originales: {nombre}')
    return original, X, scaler


def evaluar(X, modelo, k):
    etiquetas = modelo.labels_
    cantidad = len(np.unique(etiquetas))
    valido = 2 <= cantidad < len(X) and cantidad == k
    return {
        'k': k, 'clusters_obtenidos': cantidad,
        'silhouette': float(silhouette_score(X, etiquetas)) if valido else None,
        'calinski_harabasz': float(calinski_harabasz_score(X, etiquetas)) if valido else None,
        'davies_bouldin': float(davies_bouldin_score(X, etiquetas)) if valido else None,
        'inercia': float(modelo.inertia_), 'iteraciones': int(modelo.n_iter_),
    }


def exportar(carpeta, etapa, modelo, original, scaler):
    tabla = original.copy()
    tabla['cluster'] = modelo.labels_
    tabla.to_csv(carpeta / f'asignaciones_{etapa}.csv', index=False)
    perfil = tabla.groupby('cluster')[VARIABLES].mean()
    perfil.insert(0, 'cantidad_grupos', tabla.groupby('cluster').size())
    perfil['total_vuelos'] = tabla.groupby('cluster')['total_vuelos'].sum()
    perfil.to_csv(carpeta / f'perfiles_{etapa}.csv')
    centros = pd.DataFrame(scaler.inverse_transform(modelo.cluster_centers_), columns=VARIABLES)
    centros.index.name = 'cluster'
    centros.to_csv(carpeta / f'centroides_{etapa}.csv')
    joblib.dump(modelo, carpeta / f'modelo_{etapa}.joblib')


def ejecutar(nombre):
    inicio = perf_counter()
    carpeta = SALIDA / nombre
    carpeta.mkdir(parents=True, exist_ok=True)
    original, X, scaler = cargar(nombre)
    resultados, modelos = [], {}
    # Primero se evalúa la línea base, luego las demás configuraciones.
    for k in [K_INICIAL] + [k for k in range(2, 16) if k != K_INICIAL]:
        modelo = KMeans(n_clusters=k, **CONFIG).fit(X)
        fila = evaluar(X, modelo, k)
        resultados.append(fila)
        modelos[k] = modelo
        print(f'{nombre}: K={k:2d} | silhouette={fila["silhouette"]}', flush=True)
    tabla = pd.DataFrame(resultados).sort_values('k')
    tabla.to_csv(carpeta / 'metricas_busqueda.csv', index=False)
    candidatos = tabla.dropna(subset=['silhouette', 'calinski_harabasz', 'davies_bouldin']).copy()
    if candidatos.empty or K_INICIAL not in candidatos.k.values:
        raise ValueError(f'No hay configuraciones válidas o línea base válida: {nombre}')
    # Combinar las tres métricas con igual peso mediante rangos ordinales.
    # En empate: mayor silhouette y después menor K.
    candidatos['rango_silhouette'] = candidatos.silhouette.rank(ascending=False, method='min')
    candidatos['rango_ch'] = candidatos.calinski_harabasz.rank(ascending=False, method='min')
    candidatos['rango_db'] = candidatos.davies_bouldin.rank(ascending=True, method='min')
    candidatos['rango_total'] = candidatos[['rango_silhouette', 'rango_ch', 'rango_db']].sum(axis=1)
    candidatos = candidatos.sort_values(['rango_total', 'silhouette', 'k'], ascending=[True, False, True])
    candidatos.to_csv(carpeta / 'ranking_configuraciones.csv', index=False)
    mejor_k = int(candidatos.iloc[0].k)
    comparacion = []
    for etapa, k in [('inicial', K_INICIAL), ('optimizado', mejor_k)]:
        exportar(carpeta, etapa, modelos[k], original, scaler)
        comparacion.append({'etapa': etapa, **next(r for r in resultados if r['k'] == k)})
    pd.DataFrame(comparacion).to_csv(carpeta / 'comparacion.csv', index=False)
    return {'dataset': nombre, 'agrupaciones': len(X), 'vuelos': int(original.total_vuelos.sum()),
            'k_inicial': K_INICIAL, 'k_optimizado': mejor_k,
            'comparacion': comparacion, 'tiempo_segundos': round(perf_counter()-inicio, 2)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--incluir-aerolineas', action='store_true', help='Ejecutar también los datasets segmentados por aerolínea')
    args = parser.parse_args()
    nombres = list(GRUPOS) if args.incluir_aerolineas else ['rutas', 'destinos_dias']
    SALIDA.mkdir(parents=True, exist_ok=True)
    # Limita los hilos BLAS para evitar sobrecarga; Silhouette usa todos los puntos.
    with threadpool_limits(limits=2):
        resultados = [ejecutar(n) for n in nombres]
    resumen = {'variables': VARIABLES, 'configuracion': CONFIG, 'rango_k': [2, 15],
               'evaluacion': 'Silhouette exacto, Calinski-Harabasz, Davies-Bouldin e inercia',
               'seleccion': 'Suma de rangos de las tres métricas; desempate por Silhouette y menor K',
               'resultados': resultados}
    (SALIDA / 'resumen_kmeans.json').write_text(json.dumps(resumen, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    print(json.dumps(resumen, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
