"""Obtención y control de calidad del Bank Marketing (UCI, variante additional-full)."""
from pathlib import Path
from urllib.request import urlretrieve
import hashlib
import zipfile
import io
import pandas as pd

SOURCE_URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
EXPECTED_COLUMNS = {"age", "job", "marital", "education", "default", "housing", "loan", "contact", "month", "day_of_week", "duration", "campaign", "pdays", "previous", "poutcome", "emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed", "y"}

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()

def obtain_data(root):
    """Conserva el ZIP oficial y extrae únicamente la versión additional-full."""
    raw=Path(root)/'datos'/'datos_ini'
    raw.mkdir(parents=True,exist_ok=True)
    output=raw/'bank-additional-full.csv'
    if output.exists(): return output
    archive=raw/'uci_bank_marketing.zip'
    if not archive.exists(): urlretrieve(SOURCE_URL,archive)
    with zipfile.ZipFile(archive) as outer:
        nested=next((n for n in outer.namelist() if n.endswith('bank-additional.zip')),None)
        direct=next((n for n in outer.namelist() if n.endswith('bank-additional-full.csv')),None)
        if direct: content=outer.read(direct)
        elif nested:
            with zipfile.ZipFile(io.BytesIO(outer.read(nested))) as inner:
                member=next(n for n in inner.namelist() if n.endswith('bank-additional-full.csv'))
                content=inner.read(member)
        else: raise ValueError('El paquete UCI no contiene bank-additional-full.csv')
    output.write_bytes(content)
    return output

def quality_report(df):
    return {'rows':int(len(df)), 'columns':int(df.shape[1]),
            'exact_duplicates':int(df.duplicated().sum()),
            'missing_cells':int(df.isna().sum().sum()),
            'unknown_cells':int((df.select_dtypes(include='object')=='unknown').sum().sum())}

def prepare_dataset(input_path, output_path):
    original=pd.read_csv(input_path,sep=';')
    if set(original.columns)!=EXPECTED_COLUMNS:
        raise ValueError(f'Esquema inesperado. Faltan: {EXPECTED_COLUMNS-set(original.columns)}')
    if not set(original['y'].dropna().unique()).issubset({'yes','no'}):
        raise ValueError('Etiquetas de clasificación no reconocidas')
    before=quality_report(original)
    df=original.drop_duplicates().copy()
    # 'duration' sólo existe tras la llamada: impedir fuga de información de desenlace.
    df=df.drop(columns=['duration'])
    # 999 en pdays significa que no hubo contacto anterior; se conserva como categoría.
    df['pdays']=df['pdays'].astype('string').fillna('missing')
    # 'unknown' es una categoría informativa, no un nulo demostrable.
    df['y']=df['y'].map({'no':0,'yes':1}).astype('int8')
    output_path=Path(output_path)
    output_path.parent.mkdir(parents=True,exist_ok=True)
    df.to_csv(output_path,index=False)
    return df, {'original':before,'clean':quality_report(df),'removed_duplicate_rows':int(len(original)-len(df)), 'excluded_leakage_variable':'duration', 'raw_sha256':sha256(input_path),'clean_sha256':sha256(output_path)}
