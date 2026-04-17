import os
import re
import unicodedata
import numpy as np
import pandas as pd


INPUT_DIR = r"."
OUTPUT_XLSX = os.path.join(INPUT_DIR, "All_ER_Dataset.xlsx")

EXCLUDE_START = pd.Timestamp("2020-03-18")
EXCLUDE_END = pd.Timestamp("2021-09-30")

EXCLUDE_FILES = {
    os.path.basename(OUTPUT_XLSX).lower(),
    "all_er_dataset.xlsx",
    "total_er.xlsx",
}

TARGET_ROWS = {
    "TOTAL DE ATENCIONES DE URGENCIA": "TOTAL_ATENCIONES_URGENCIA",
    "TOTAL DEMANDA": "TOTAL_DEMANDA",
    "SECCION 1 TOTAL ATENCIONES DE URGENCIA": "SECCION_1_TOTAL_ATENCIONES_URGENCIA",
    "TOTAL CAUSA SISTEMA RESPIRATORIO J00 J98": "RESP_TOTAL",
    "IRA ALTA J00 J06": "IRA_ALTA",
    "INFLUENZA J09 J11": "INFLUENZA",
    "NEUMONIA J12 J18": "NEUMONIA",
    "BRONQUITIS BRONQUIOLITIS AGUDA J20 J21": "BRONQUITIS_BRONQUIOLITIS_AGUDA",
    "CRISIS OBSTRUCTIVA BRONQUIAL J40 J46": "CRISIS_OBSTRUCTIVA_BRONQUIAL",
    "OTRA CAUSA RESPIRATORIA NO CONTENIDAS EN LAS CATEGORIAS ANTERIORES J22 J30 J39 J47 J60 J98": "OTRAS_CAUSAS_RESPIRATORIAS",
    "TOTAL ATENCIONES POR COVID 19 VIRUS NO IDENTIFICADO U07 2": "COVID_U07_2",
    "TOTAL ATENCIONES POR COVID 19 VIRUS IDENTIFICADO U07 1": "COVID_U07_1",
    "TOTAL ATENCIONES POR CAUSA SISTEMA CIRCULATORIO I00 I99": "TOTAL_CIRCULATORIO",
    "INFARTO AGUDO MIOCARDIO I21 I22": "INFARTO_AGUDO_MIOCARDIO",
    "ACCIDENTE VASCULAR ENCEFALICO I60 I66 I67 8 I67 9 I69": "ACCIDENTE_VASCULAR_ENCEFALICO",
    "CRISIS HIPERTENSIVA I10 X": "CRISIS_HIPERTENSIVA",
    "ARRITMIA GRAVE I44 I46 0 I46 9 I49": "ARRITMIA_GRAVE",
    "OTRAS CAUSAS CIRCULATORIAS NO CONTENIDAS EN LAS CATEGORIAS ANTERIORES I00 I09 I11 I15 I20 I23 I28 I30 I42 I50 I52 I67 0 I67 7 E I70 I99": "OTRAS_CAUSAS_CIRCULATORIAS",
    "TOTAL ATENCIONES POR TRAUMATISMOS Y ENVENENAMIENTOS S00 T98": "TOTAL_TRAUMATISMOS_ENVENENAMIENTOS",
    "LESIONES POR ACCIDENTES DEL TRANSITO CAUSA EXTERNA V01 V89": "LESIONES_TRANSITO",
    "LESIONES AUTOINFLINGIDAS INTENCIONALMENTE CAUSA EXTERNA X60 X84": "LESIONES_AUTOINFLIGIDAS",
    "LESIONES POR QUEMADURAS EXPOSICION AL HUMO FUEGO LLAMAS CONTACTO CON CALOR Y SUSTANCIAS CALIENTES CAUSA EXTERNA X00 X19": "LESIONES_QUEMADURAS",
    "LESIONES POR OTRAS CAUSAS EXTERNAS NO CONTENIDAS EN LAS CATEGORIAS ANTERIORES CAUSA EXTERNA V90 W99 X20 X59 X85 Y98": "LESIONES_OTRAS_CAUSAS_EXTERNAS",
    "TOTAL ATENCIONES POR CAUSA DE TRASTORNOS MENTALES F00 F99": "TOTAL_TRASTORNOS_MENTALES",
    "IDEACION SUICIDA R45 8": "IDEACION_SUICIDA",
    "TRASTORNOS MENTALES Y DEL COMPORTAMIENTO DEBIDOS AL USO DE SUSTANCIAS PSICOACTIVAS F10 F19": "TRASTORNOS_SUSTANCIAS_PSICOACTIVAS",
    "TRASTORNOS DEL HUMOR AFECTIVOS F30 F39": "TRASTORNOS_HUMOR",
    "TRASTORNOS NEUROTICOS TRASTORNOS RELACIONADOS CON EL ESTRES Y TRASTORNOS SOMATOMORFOS F40 F48": "TRASTORNOS_NEUROTICOS_ESTRES_SOMATOMORFOS",
    "OTROS TRASTORNOS MENTALES NO CONTENIDOS EN LAS CATEGORIAS ANTERIORES": "OTROS_TRASTORNOS_MENTALES",
    "TOTAL ATENCIONES POR DIARREA AGUDA A00 A09": "DIARREA_AGUDA",
    "TOTAL ATENCIONES POR OTRAS CAUSAS NO CONTENIDAS EN LAS CAUSAS ANTERIORES": "OTRAS_CAUSAS",
    "SECCION 2 TOTAL HOSPITALIZACIONES POR GRUPO DE CAUSA": "SECCION_2_TOTAL_HOSPITALIZACIONES",
    "CAUSAS SISTEMA RESPIRATORIO J00 J98": "HOSP_RESPIRATORIO",
    "POR COVID 19 VIRUS NO IDENTIFICADO U07 2": "HOSP_COVID_U07_2",
    "POR COVID 19 VIRUS IDENTIFICADO U07 1": "HOSP_COVID_U07_1",
    "CAUSAS SISTEMA CIRCULATORIO I00 I99": "HOSP_CIRCULATORIO",
    "CAUSAS POR TRAUMATISMOS Y ENVENENAMIENTOS S00 T98": "HOSP_TRAUMATISMOS_ENVENENAMIENTOS",
    "CAUSA POR TRASTORNOS MENTALES F00 F99": "HOSP_TRASTORNOS_MENTALES",
    "POR OTRAS CAUSAS NO CONTENIDAS EN LAS CAUSAS ANTERIORES": "HOSP_OTRAS_CAUSAS",
    "CIRUGIAS DE URGENCIA": "CIRUGIAS_URGENCIA",
}


def extract_year_from_filename(filename: str) -> str:
    base = os.path.splitext(os.path.basename(filename))[0]
    m = re.search(r"(20\d{2})", base)
    return m.group(1) if m else ""


def normalize_text(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def normalize_key(x):
    s = normalize_text(x)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("utf-8")
    s = s.upper()
    s = s.replace("–", "-").replace("—", "-")
    s = re.sub(r"[()]", " ", s)
    s = re.sub(r"[/;,:]", " ", s)
    s = re.sub(r"[^A-Z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def to_numeric_clean(series: pd.Series) -> pd.Series:
    def clean_value(x):
        if pd.isna(x):
            return np.nan
        s = str(x).strip()
        if s in {"", "-", "nan", "None"}:
            return np.nan
        s = s.replace(",", ".")
        return s

    return pd.to_numeric(series.map(clean_value), errors="coerce")


def parse_mixed_excel_dates(series: pd.Series) -> pd.Series:
    s = series.copy()

    is_num = pd.to_numeric(s, errors="coerce").notna()
    s_num = pd.to_numeric(s.where(is_num), errors="coerce")
    s_txt = s.where(~is_num)

    out_num = pd.to_datetime("1899-12-30") + pd.to_timedelta(s_num, unit="D")
    out_txt = pd.to_datetime(s_txt, errors="coerce")

    out = out_txt.copy()
    out[is_num] = out_num[is_num]
    return pd.to_datetime(out, errors="coerce").dt.normalize()


def find_anchor_row(df: pd.DataFrame) -> int:
    first_col = df.iloc[:, 0].map(normalize_text)
    keys = first_col.map(normalize_key)

    matches = keys[keys.eq("TOTAL DEMANDA")]
    if len(matches) == 0:
        raise ValueError("No se encontró la fila 'TOTAL DEMANDA'.")

    return matches.index[0]


def transform_one_file(path: str, debug=False) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=0, header=None)
    df = df.dropna(how="all").dropna(axis=1, how="all").reset_index(drop=True)
    df = df.map(normalize_text)

    anchor_row = find_anchor_row(df)
    row_fechas = anchor_row - 1
    fechas = parse_mixed_excel_dates(df.iloc[row_fechas, 1:])

    if debug:
        print(f"\n--- DEBUG {os.path.basename(path)} ---")
        print("shape:", df.shape)
        print("row_fechas:", row_fechas, df.iloc[row_fechas, :8].tolist())
        print("anchor_row:", anchor_row, df.iloc[anchor_row, :8].tolist())

    rows_out = []

    for ridx in range(row_fechas, len(df)):
        raw_label = normalize_text(df.iloc[ridx, 0])
        if raw_label == "":
            continue

        label_key = normalize_key(raw_label)
        var_name = TARGET_ROWS.get(label_key)

        if var_name is None:
            continue

        valores = to_numeric_clean(df.iloc[ridx, 1:])

        tmp = pd.DataFrame({
            "FECHA": fechas.values,
            var_name: valores.values
        })

        tmp = tmp.dropna(subset=["FECHA"]).copy()
        tmp = tmp[
            ~tmp["FECHA"].between(EXCLUDE_START, EXCLUDE_END, inclusive="both")
        ].copy()

        rows_out.append(tmp)

    if not rows_out:
        raise ValueError(f"No se extrajeron filas objetivo desde {os.path.basename(path)}")

    merged = rows_out[0].copy()

    for extra in rows_out[1:]:
        merged = merged.merge(extra, on="FECHA", how="outer")

    merged["source_file"] = os.path.basename(path)
    merged["source_year"] = extract_year_from_filename(path)

    merged = merged.sort_values("FECHA").reset_index(drop=True)
    return merged


def main():
    files = sorted(
        f for f in os.listdir(INPUT_DIR)
        if f.lower().endswith(".xlsx")
        and not f.startswith("~$")
        and f.lower() not in EXCLUDE_FILES
    )

    if not files:
        raise FileNotFoundError(f"No se encontraron archivos .xlsx en {INPUT_DIR}")

    all_frames = []

    for i, f in enumerate(files):
        path = os.path.join(INPUT_DIR, f)
        tmp = transform_one_file(path, debug=(i == 0))
        all_frames.append(tmp)
        print(f"{f}: {len(tmp)} filas")

    final_df = pd.concat(all_frames, axis=0, ignore_index=True)

    final_df = (
        final_df
        .sort_values(["FECHA", "source_file"])
        .drop_duplicates(subset=["FECHA"], keep="first")
        .sort_values("FECHA")
        .reset_index(drop=True)
    )

    ordered_cols = ["FECHA"] + [
        c for c in final_df.columns
        if c not in ["FECHA", "source_file", "source_year"]
    ] + ["source_file", "source_year"]

    final_df = final_df[ordered_cols]

    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        final_df.to_excel(writer, sheet_name="ER_Consolidado", index=False)

    print(f"\nDone:\n{OUTPUT_XLSX}")
    print(f"Filas finales: {len(final_df)}")
    print(f"Rango final: {final_df['FECHA'].min().date()} -> {final_df['FECHA'].max().date()}")
    print("\nColumnas finales:")
    print(final_df.columns.tolist())


if __name__ == "__main__":
    main()
