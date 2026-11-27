"""Genera gráficas de los resultados existentes de K-Means y DBSCAN."""
from pathlib import Path
from itertools import combinations
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import seaborn as sns

BASE = Path(__file__).resolve().parent
MODELOS = BASE / 'modelos'
SALIDA = BASE / 'graficas'
VARIABLES = {
    'porcentaje_retrasados': 'Llegadas retrasadas (%)',
    'promedio_minutos_retraso': 'Retraso promedio (min)',
    'porcentaje_cancelados': 'Cancelaciones (%)',
    'porcentaje_desviados': 'Desvíos (%)',
}
DATASETS = {'rutas': 'Rutas', 'destinos_dias': 'Destinos y días',
            'rutas_aerolinea': 'Rutas por aerolínea',
            'destinos_dias_aerolinea': 'Destinos y días por aerolínea'}
METRICAS = {'silhouette': 'Silhouette (↑)', 'calinski_harabasz': 'Calinski-Harabasz (↑)',
            'davies_bouldin': 'Davies-Bouldin (↓)'}
ETAPAS = ['inicial', 'optimizado']


def guardar(fig, ruta):
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, dpi=200, bbox_inches='tight')
    plt.close(fig)


def leer(ruta):
    if not ruta.is_file():
        raise FileNotFoundError(f'Falta el archivo: {ruta}')
    return pd.read_csv(ruta)


def etiqueta(c):
    return 'Ruido (-1)' if c == -1 else f'Clúster {c}'


def dispersion(tabla, centro, titulo, salida, limites):
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    grupos = sorted(tabla.cluster.unique(), key=lambda c: (c != -1, c))
    colores = {c: ('#888888' if c == -1 else plt.get_cmap('tab20')(int(c) % 20)) for c in grupos}
    for ax, (x, y) in zip(axes.flat, combinations(VARIABLES, 2)):
        for c in grupos:
            t = tabla[tabla.cluster == c]
            ax.scatter(t[x], t[y], s=7, alpha=.38, color=colores[c], label=etiqueta(c), rasterized=True)
        if centro is not None:
            ax.scatter(centro[x], centro[y], marker='X', s=105, c='black', edgecolors='white', label='Centroides', zorder=5)
        ax.set(xlabel=VARIABLES[x], ylabel=VARIABLES[y], xlim=limites[x], ylim=limites[y])
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=min(8, len(labels)), fontsize=9)
    fig.suptitle(titulo + '\nProyecciones de las cuatro dimensiones (sin PCA)')
    fig.tight_layout(rect=(0, .08, 1, .93))
    guardar(fig, salida)


def distribucion(tabla, titulo, salida):
    conteo = tabla.groupby('cluster').size()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    nombres = [etiqueta(c) for c in conteo.index]
    colores = ['#888888' if c == -1 else '#4477aa' for c in conteo.index]
    for ax, valores, unidad in [(axes[0], conteo, 'Agrupaciones'), (axes[1], 100*conteo/len(tabla), 'Porcentaje de agrupaciones')]:
        barras = ax.bar(nombres, valores, color=colores)
        ax.set_ylabel(unidad)
        ax.tick_params(axis='x', rotation=45)
        ax.bar_label(barras, fmt='%.1f' if ax == axes[1] else '%d', fontsize=8, padding=2)
        ax.margins(y=.18)
    axes[1].yaxis.set_major_formatter(PercentFormatter(100))
    fig.suptitle(titulo + ' — tamaño de clústeres')
    fig.tight_layout()
    guardar(fig, salida)


def cajas(tabla, titulo, salida):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    t = tabla.assign(grupo=tabla.cluster.map(etiqueta))
    orden = [etiqueta(c) for c in sorted(tabla.cluster.unique())]
    for ax, (v, nombre) in zip(axes.flat, VARIABLES.items()):
        sns.boxplot(data=t, x='grupo', y=v, order=orden, ax=ax, color='#7aa6c2', fliersize=2)
        ax.set(xlabel='', ylabel=nombre)
        ax.tick_params(axis='x', rotation=45)
    fig.suptitle(titulo + ' — perfiles operativos')
    fig.tight_layout()
    guardar(fig, salida)


def comparar(tabla, columnas, titulo, salida):
    fig, axes = plt.subplots(1, len(columnas), figsize=(4.4*len(columnas), 4.8), squeeze=False)
    for ax, (col, label) in zip(axes.flat, columnas.items()):
        valores = tabla[col]
        barras = ax.bar(tabla.etapa, valores, color=['#94a3b8' if e == 'inicial' else '#4477aa' for e in tabla.etapa])
        ax.axhline(0, color='gray', lw=.7)
        ax.bar_label(barras, labels=[f'{v:.3f}' if pd.notna(v) else 'N/D' for v in valores], padding=4)
        ax.set_title(label)
        ax.margins(y=.2)
    fig.suptitle(titulo)
    fig.tight_layout()
    guardar(fig, salida)


def busqueda_kmeans(t, comp, titulo, salida):
    t = t.sort_values('k')
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (c, nombre) in zip(axes.flat, {'inercia': 'Inercia — método del codo', **METRICAS}.items()):
        ax.plot(t.k, t[c], 'o-', markersize=4)
        for etapa, estilo in [('inicial', '--'), ('optimizado', ':')]:
            fila = comp[comp.etapa == etapa]
            if not fila.empty:
                ax.axvline(fila.k.iloc[0], linestyle=estilo, color='black' if etapa == 'inicial' else '#cc6677', label=etapa)
        ax.set(xlabel='K', ylabel=nombre, xticks=t.k)
        ax.legend()
    fig.suptitle(titulo + ' — búsqueda de K')
    fig.tight_layout()
    guardar(fig, salida)


def busqueda_dbscan(t, comp, titulo, salida):
    for col, label in {**METRICAS, 'porcentaje_ruido': 'Ruido (%)', 'clusters_obtenidos': 'Clústeres'}.items():
        tabla = t.pivot(index='min_samples', columns='eps', values=col).sort_index()
        fig, ax = plt.subplots(figsize=(17, 4.5))
        sns.heatmap(tabla, ax=ax, cmap='viridis_r' if col in ['davies_bouldin', 'porcentaje_ruido'] else 'viridis', annot=True, fmt='.2f' if col != 'clusters_obtenidos' else '.0f', annot_kws={'size': 7}, mask=tabla.isna(), cbar_kws={'label': label})
        for _, fila in comp.iterrows():
            i = np.where(np.isclose(tabla.index.astype(float), fila.min_samples))[0]
            j = np.where(np.isclose(tabla.columns.astype(float), fila.eps))[0]
            if len(i) and len(j):
                from matplotlib.patches import Rectangle
                ax.add_patch(Rectangle((j[0], i[0]), 1, 1, fill=False, edgecolor='red' if fila.etapa == 'optimizado' else 'white', linewidth=2.5, linestyle='-' if fila.etapa == 'optimizado' else '--'))
        ax.set_title(titulo + ' — ' + label + '\nBlanco discontinuo: inicial; rojo: optimizado. Celdas vacías: métrica no definida.')
        guardar(fig, salida / f'mapa_{col}.png')


def vecinos(t, comp, titulo, salida):
    fig, ax = plt.subplots(figsize=(10, 6))
    for c in t.columns:
        ax.plot(np.arange(1, len(t)+1), t[c], label=c.replace('_', ' '))
    for _, fila in comp.iterrows():
        ax.axhline(fila.eps, linestyle='--' if fila.etapa == 'inicial' else ':', label=f'{fila.etapa}: eps={fila.eps:g}, min_samples={int(fila.min_samples)}')
    ax.set(xlabel='Agrupaciones ordenadas por distancia', ylabel='Distancia al vecino (espacio estandarizado)', title=titulo + ' — k-distance plot')
    ax.legend()
    guardar(fig, salida)


def ejecutar(algoritmo, nombre, etapas):
    carpeta = MODELOS / algoritmo / nombre
    salida = SALIDA / algoritmo / nombre
    titulo = f'{algoritmo.upper()} | {DATASETS[nombre]}'
    comp = leer(carpeta / 'comparacion.csv')
    busqueda = leer(carpeta / 'metricas_busqueda.csv')
    tablas = {e: leer(carpeta / f'asignaciones_{e}.csv') for e in etapas if e in comp.etapa.values}
    if not tablas:
        print(f'AVISO: no hay etapas solicitadas en {carpeta}')
        return None
    # Los límites se calculan con todos los datos de la primera etapa:
    # los modelos utilizan las mismas agrupaciones y no eliminan el ruido.
    limites = {}
    for v in VARIABLES:
        minimo = min(t[v].min() for t in tablas.values())
        maximo = max(t[v].max() for t in tablas.values())
        margen = max((maximo-minimo)*.04, .001)
        limites[v] = (minimo-margen, maximo+margen)
    for etapa, tabla in tablas.items():
        destino = salida / etapa
        centro = leer(carpeta / f'centroides_{etapa}.csv') if algoritmo == 'kmeans' else None
        dispersion(tabla, centro, titulo + ' — ' + etapa, destino / 'dispersion.png', limites)
        distribucion(tabla, titulo + ' — ' + etapa, destino / 'distribucion.png')
        cajas(tabla, titulo + ' — ' + etapa, destino / 'boxplots.png')
        if algoritmo == 'dbscan':
            ruido = tabla[tabla.cluster == -1]
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.bar(['Agrupados', 'Ruido'], [len(tabla)-len(ruido), len(ruido)], color=['#4477aa', '#888888'])
            ax.set(ylabel='Agrupaciones', title=f'{titulo} — {etapa}: ruido {100*len(ruido)/len(tabla):.2f}%')
            guardar(fig, destino / 'ruido.png')
    if len(etapas) == 2:
        extra = {'inercia': 'Inercia (↓)'} if algoritmo == 'kmeans' else {'porcentaje_ruido': 'Ruido (%)'}
        comparar(comp, {**METRICAS, **extra}, titulo + ' — inicial vs. optimizado', salida / 'comparacion_metricas.png')
        if algoritmo == 'kmeans':
            busqueda_kmeans(busqueda, comp, titulo, salida / 'busqueda_k.png')
        else:
            busqueda_dbscan(busqueda, comp, titulo, salida)
            vecinos(leer(carpeta / 'distancias_vecinos.csv'), comp, titulo, salida / 'distancias_vecinos.png')
    print(f'Generado: {salida}', flush=True)
    return comp.assign(algoritmo=algoritmo, dataset=nombre)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--etapa', choices=['inicial', 'optimizado', 'ambas'], default='ambas')
    parser.add_argument('--solo-principales', action='store_true')
    args = parser.parse_args()
    etapas = ETAPAS if args.etapa == 'ambas' else [args.etapa]
    sns.set_theme(style='whitegrid', context='notebook')
    resultados = []
    for nombre in list(DATASETS)[:2] if args.solo_principales else DATASETS:
        for algoritmo in ['kmeans', 'dbscan']:
            r = ejecutar(algoritmo, nombre, etapas)
            if r is not None:
                resultados.append(r)
    if args.etapa == 'ambas' and resultados:
        tabla = pd.concat(resultados, ignore_index=True)
        carpeta = SALIDA / 'comparacion_algoritmos'
        carpeta.mkdir(parents=True, exist_ok=True)
        tabla.to_csv(carpeta / 'metricas_consolidadas.csv', index=False)
        for nombre, t in tabla.groupby('dataset'):
            fig, axes = plt.subplots(1, 3, figsize=(13, 5))
            for ax, (c, label) in zip(axes, METRICAS.items()):
                sns.barplot(data=t, x='algoritmo', y=c, hue='etapa', ax=ax, errorbar=None)
                ax.set(xlabel='', ylabel=label)
            fig.suptitle(DATASETS[nombre] + '\nDBSCAN: métricas sin ruido; K-Means: todos los puntos. Comparación descriptiva.')
            fig.tight_layout()
            guardar(fig, carpeta / f'{nombre}.png')
    print('Visualización finalizada.')


if __name__ == '__main__':
    main()
