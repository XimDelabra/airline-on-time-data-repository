#!/usr/bin/env python3
import argparse
import re
import sys
import time
from pathlib import Path
from urllib.parse import urljoin
import requests

BASE_URL = "https://transtats.bts.gov/PREZIP/"
PREFIX = "On_Time_Reporting_Carrier_On_Time_Performance_1987_present"
FILE_RE = re.compile(
    rf'({re.escape(PREFIX)}_(\d{{4}})_(\d{{1,2}})\.zip)',
    re.IGNORECASE
)
HEADERS = {"User-Agent": "Mozilla/5.0 (AcademicDataDownloader/1.0)"}

def get_session():
    s = requests.Session()
    s.headers.update(HEADERS)
    return s

def discover_from_index(session, start_year, end_year):
    try:
        r = session.get(BASE_URL, timeout=60)
        r.raise_for_status()
    except Exception as e:
        print(f"[Aviso] No pude leer el índice PREZIP: {e}")
        return []
    found = {}
    for filename, year_s, month_s in FILE_RE.findall(r.text):
        year = int(year_s)
        month = int(month_s)
        if start_year <= year <= end_year and 1 <= month <= 12:
            found[(year, month)] = filename
    return [(y, m, found[(y, m)]) for y, m in sorted(found)]

def probe_files(session, start_year, end_year):
    files = []
    print("[Info] Probando archivos por año/mes...")
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            filename = f"{PREFIX}_{year}_{month}.zip"
            url = urljoin(BASE_URL, filename)
            try:
                r = session.head(url, allow_redirects=True, timeout=30)
                if r.status_code == 200:
                    files.append((year, month, filename))
                elif r.status_code in (403, 405):
                    r = session.get(url, stream=True, timeout=30)
                    if r.status_code == 200:
                        files.append((year, month, filename))
                    r.close()
            except requests.RequestException:
                pass
        print(f"  Revisado {year}")
    return files

def human_size(num):
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.1f} {unit}"
        size /= 1024

def download_file(session, url, dest, retries=4):
    temp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            with session.get(url, stream=True, timeout=(30, 180)) as r:
                r.raise_for_status()
                total = int(r.headers.get("Content-Length", 0))
                downloaded = 0
                with open(temp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if not chunk:
                            continue
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total:
                            pct = downloaded * 100 / total
                            print(
                                f"\r    {pct:6.2f}%  {human_size(downloaded)} / {human_size(total)}",
                                end="",
                                flush=True,
                            )
                if total:
                    print()
            temp.replace(dest)
            return True
        except Exception as e:
            print(f"\n    Intento {attempt}/{retries} falló: {e}")
            if temp.exists():
                temp.unlink(missing_ok=True)
            if attempt < retries:
                time.sleep(3 * attempt)
    return False

def main():
    parser = argparse.ArgumentParser(description="Descarga histórica BTS On-Time Performance")
    parser.add_argument("--inicio", type=int, default=1995, help="Año inicial")
    parser.add_argument("--fin", type=int, default=2026, help="Año final")
    parser.add_argument("--destino", default="datos_crudos_bts", help="Carpeta destino")
    parser.add_argument("--solo-listar", action="store_true", help="Solo mostrar archivos")
    args = parser.parse_args()

    if args.inicio > args.fin:
        print("Error: --inicio no puede ser mayor que --fin.")
        sys.exit(1)

    root = Path(args.destino)
    root.mkdir(parents=True, exist_ok=True)
    session = get_session()

    print("Buscando archivos disponibles en BTS...")
    files = discover_from_index(session, args.inicio, args.fin)

    if not files:
        print("[Aviso] No se pudieron descubrir desde el índice.")
        print("Usando detección directa por año y mes...")
        files = probe_files(session, args.inicio, args.fin)

    if not files:
        print("No se encontró ningún archivo.")
        print("Es posible que BTS haya cambiado la ruta o el patrón de nombres.")
        sys.exit(2)

    print(f"\nSe encontraron {len(files)} archivos disponibles.\n")

    if args.solo_listar:
        for year, month, filename in files:
            print(f"{year}-{month:02d}  {filename}")
        return

    ok = 0
    skipped = 0
    failed = []

    for i, (year, month, filename) in enumerate(files, start=1):
        year_dir = root / str(year)
        year_dir.mkdir(parents=True, exist_ok=True)
        dest = year_dir / filename
        url = urljoin(BASE_URL, filename)

        print(f"[{i}/{len(files)}] {year}-{month:02d}")

        if dest.exists() and dest.stat().st_size > 0:
            print(f"    Ya existe: {dest}")
            skipped += 1
            continue

        print(f"    Descargando {filename}")
        if download_file(session, url, dest):
            print(f"    OK -> {dest}")
            ok += 1
        else:
            print(f"    ERROR -> {filename}")
            failed.append((year, month, filename))

    print("\n========== RESUMEN ==========")
    print(f"Descargados: {ok}")
    print(f"Ya existentes: {skipped}")
    print(f"Fallidos: {len(failed)}")
    print(f"Carpeta: {root.resolve()}")

    if failed:
        print("\nArchivos fallidos:")
        for year, month, filename in failed:
            print(f"  - {year}-{month:02d}: {filename}")
        print("\nPuedes ejecutar el script de nuevo; saltará los archivos ya descargados.")

if __name__ == "__main__":
    main()
