# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 18:17:53 2025

@author: river
"""
'''
NOTA PRINCIPAL: Hace una correcta extracción característica al dentificar patrones de interés a partir de la
función "buscar_pagina_con_tablas" sin embargo para la extracción de los datos hace saltos incorrectos:
    - Cuando llega a los miles de casos o decenas de miles de casos, en el PDF están registrados como:
        Ej, 1 390 (1390) y la función extraer_tablas_de_pagina tiene un fallo en la lógica de separación de datos
        de la línea 216 a 240, el error se ve como 139 en la columna 1 y 0 en la columna 2. 
    Te adjunto fotos para que visualices mejor. 

'''



# -*- coding: utf-8 -*-
"""
Script para extraer datos epidemiológicos de múltiples boletines semanales
y combinarlos en un único DataFrame anual.
"""

#Librerías
import zipfile
from PyPDF2 import PdfReader
import pandas as pd
import io
import re
import os
import re
import numpy as np

#%% Función para extraer el archivo PDF del ZIP
def extraer_pdf_desde_zip(ruta_zip, nombre_pdf):
    """
    Extrae un archivo PDF específico de un archivo ZIP.

    Parámetros:
    - ruta_zip: Ruta del archivo ZIP.
    - nombre_pdf: Nombre del archivo PDF a extraer.

    Retorna:
    - Un objeto BytesIO con el contenido del PDF.
    """
    with zipfile.ZipFile(ruta_zip, 'r') as zip_ref:
        with zip_ref.open(nombre_pdf) as pdf_file:
            pdf_bytes = pdf_file.read()
    return io.BytesIO(pdf_bytes)

#%% Función para obtener la lista de archivos PDF en el ZIP
def obtener_pdfs_del_zip(ruta_zip):
    """
    Obtiene la lista de archivos PDF dentro del ZIP.
    
    Parámetros:
    - ruta_zip: Ruta del archivo ZIP.
    
    Retorna:
    - Lista de nombres de archivos PDF.
    """
    pdf_files = []
    with zipfile.ZipFile(ruta_zip, 'r') as zip_ref:
        for file in zip_ref.namelist():
            if file.lower().endswith('.pdf') and file.startswith('sem'):
                pdf_files.append(file)
    return sorted(pdf_files)

#%% Función para extraer el número de semana del nombre del archivo
def extraer_numero_semana(nombre_pdf):
    """
    Extrae el número de semana del nombre del archivo PDF.
    
    Parámetros:
    - nombre_pdf: Nombre del archivo PDF (ej: 'sem02.pdf')
    
    Retorna:
    - Número de semana como entero
    """
    match = re.search(r'sem(\d+)', nombre_pdf, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None

#%% Función para buscar la página que contiene las tablas de interés
def buscar_pagina_con_tablas(pdf_reader):
    """
    Busca la página que contiene tablas de enfermedades neurológicas.
    Intenta varias frases y patrones de búsqueda para mayor robustez.

    Parámetros:
    - pdf_reader: Objeto PdfReader con el contenido del PDF.

    Retorna:
    - El número de la página que contiene las tablas o -1 si no se encuentra.
    """
    # Patrones de búsqueda (de más específico a más general)
    patrones = [
        "Casos por entidad federativa de Enfermedades Neurológicas y de Salud",
        "Enfermedades Neurológicas",
        "Neurológicas",
        "Depresión",
        "Enfermedad de Parkinson",
        "Enfermedad de Alzheimer"
    ]
    
    # Buscar en todas las páginas del PDF
    for pagina_num in range(len(pdf_reader.pages)):
        pagina = pdf_reader.pages[pagina_num]
        texto = pagina.extract_text()
        
        # Verificar cada patrón
        for patron in patrones:
            if patron in texto:
                # Verificar si parece una página con tablas de datos 
                # (contiene al menos dos de las enfermedades o tiene estructura tabular)
                if (("Depresión" in texto and "Parkinson" in texto) or 
                    ("Depresión" in texto and "Alzheimer" in texto) or
                    ("Parkinson" in texto and "Alzheimer" in texto) or
                    (texto.count("M") > 5 and texto.count("F") > 5)):
                    print(f"  Página encontrada ({pagina_num+1}): Patrón '{patron}'")
                    return pagina_num
    
    # Si no encuentra nada, buscar de forma más general
    for pagina_num in range(len(pdf_reader.pages)):
        pagina = pdf_reader.pages[pagina_num]
        texto = pagina.extract_text()
        
        # Verificar si hay al menos dos entidades federativas y algunas cifras
        entidades_encontradas = 0
        entidades = ["Aguascalientes", "Baja California", "Chiapas", "Jalisco", "México", "Veracruz"]
        for entidad in entidades:
            if entidad in texto:
                entidades_encontradas += 1
        
        # Si hay al menos 3 entidades y contiene dígitos, es probable que sea una tabla
        if entidades_encontradas >= 3 and any(c.isdigit() for c in texto):
            print(f"  Página encontrada ({pagina_num+1}): Detección por entidades")
            return pagina_num
            
    print("  No se encontró ninguna página con tablas usando los patrones de búsqueda")
    return -1  # Retorna -1 si no se encuentra la página

#%% Helper functions
def parse_first_item(item):
    """
    Separa el nombre del estado (parte de texto) de cualquier número que
    esté pegado al final.
    """
    # Reemplaza múltiples espacios por uno solo (por seguridad)
    item = re.sub(r'\s+', ' ', item).strip()
    
    # Busca una parte sin dígitos seguida opcionalmente de algo numérico
    match = re.match(r'([^\d]+)(.*)', item)
    if match:
        state_part = match.group(1).strip()  # texto (nombre estado)
        number_part = match.group(2).strip() # lo que quede (puede ser dígitos u otro)
    else:
        # Si no macha, consideramos todo como nombre
        state_part = item
        number_part = ""
    return state_part, number_part

def split_and_clean(items):
    """
    - Si el string contiene guiones, lo parte por espacios ('25 -' -> ['25','-']).
    - Si NO contiene guiones ('1 047'), elimina espacios internos para leerlo como un solo número ('1047').
    - Convierte dígitos a int, guiones a NaN.
    """
    output = []
    for piece in items:
        # Caso 1: Si hay un guion, partimos por espacio
        if '-' in piece:
            tokens = piece.split()
            for t in tokens:
                t = t.strip().replace(' ', '')  # elimina espacios internos
                if t.isdigit():
                    output.append(int(t))
                elif t == '-':
                    output.append(np.nan)
                else:
                    output.append(np.nan)
        else:
            # Caso 2: No hay guion => eliminar todos los espacios internos y tratarlo como un único valor
            piece_no_spaces = piece.replace(' ', '')
            if piece_no_spaces.isdigit():
                output.append(int(piece_no_spaces))
            else:
                # Si no es dígito limpio, lo convertimos a NaN
                output.append(np.nan)
    return output

#%% Función para extraer las tablas de la página específica
def extraer_tablas_de_pagina(pdf_reader, pagina_num):
    """
    Extrae las tablas de enfermedades neurológicas de la página especificada.
    Intenta ser robusto frente a diferentes formatos de presentación.
    
    Parámetros:
    - pdf_reader: Objeto PdfReader con el contenido del PDF.
    - pagina_num: Número de la página que contiene las tablas.
    
    Retorna:
    - Una lista con un DataFrame que contiene la tabla estructurada.
    """
    # Verificar que la página existe
    if pagina_num >= len(pdf_reader.pages):
        print(f"Error: La página {pagina_num} no existe en el PDF.")
        return []
    
    # Extraer el texto de la página
    page = pdf_reader.pages[pagina_num]
    
    def preprocess(line):
        return re.split(r'\s{2,}', line.replace(',',''))

    lines = map(preprocess, page.extract_text().split('\n'))

    entidades = [
        "Aguascalientes", "Baja California", "Baja California Sur", "Campeche", 
        "Coahuila", "Colima", "Chiapas", "Chihuahua", "Distrito Federal", 
        "Durango", "Guanajuato", "Guerrero", "Hidalgo", "Jalisco", "México", 
        "Michoacán", "Morelos", "Nayarit", "Nuevo León", "Oaxaca", "Puebla", 
        "Querétaro", "Quintana Roo", "San Luis Potosí", "Sinaloa", "Sonora", 
        "Tabasco", "Tamaulipas", "Tlaxcala", "Veracruz", "Yucatán", "Zacatecas", "TOTAL"
    ]

    def filter_by_entity(line):
        if ' ' not in line[0] and line[0] in entidades:
            return True
        
        return " ".join(line[0].split(' ')[:-1]) in entidades

    filtered_lines = list(filter(filter_by_entity, lines))
    cleaned_data = []
    for row in filtered_lines:
        # 1) Separar el primer item (nombre estado + posible número)
        state, possible_num = parse_first_item(row[0])

        # 2) Construir lista con "possible_num" (si lo hay) y el resto de la fila
        #    a partir del segundo elemento.
        rest = []
        if possible_num:
            rest.append(possible_num)
        rest.extend(row[1:])  # Agregar el resto

        # 3) Dividir cada sub-elemento y limpiar
        numeric_values = split_and_clean(rest)

        # -- En este paso numeric_values puede tener más o menos de 9 elementos. --
        # Si quieres forzar a que cada fila tenga EXACTAMENTE 9 valores:
        if len(numeric_values) < 9:
            # Rellenar con NaN si faltan
            numeric_values += [np.nan]*(9 - len(numeric_values))
        elif len(numeric_values) > 9:
            # Recortar si sobran
            numeric_values = numeric_values[:9]

        # 4) Agregar la fila completa: [Estado] + 9 valores
        cleaned_data.append([state] + numeric_values)

    # Definir columnas
    columns = [
        'ENTIDAD_FEDERATIVA',
        'Depresion_Sem','Depresion_M','Depresion_F',
        'Parkinson_Sem','Parkinson_M','Parkinson_F',
        'Alzheimer_Sem','Alzheimer_M','Alzheimer_F'
    ]

    # Crear DataFrame
    df = pd.DataFrame(cleaned_data, columns=columns)

    # FORMATEAR NÚMEROS MAYORES DE 999
    # (aunque con estos datos quizá no sea muy común)
    def formato_miles(x):
        if pd.isna(x):
            return np.nan
        # Si es entero y > 999, poner formato "1,234"
        if isinstance(x, int) and x > 999:
            return int(f"{x:,}".replace(",", ""))
        return int(x)

    df.iloc[:, 1:] = df.iloc[:, 1:].applymap(formato_miles)
    
    return [df]

#%% Función para procesar un archivo PDF del ZIP
def procesar_pdf_desde_zip(ruta_zip, nombre_pdf):
    """
    Procesa un archivo PDF dentro del ZIP para extraer las tablas de interés.
    Intenta varios enfoques para maximizar la extracción de datos.

    Parámetros:
    - ruta_zip: Ruta del archivo ZIP.
    - nombre_pdf: Nombre del archivo PDF dentro del ZIP.

    Retorna:
    - Una lista de DataFrames con las tablas extraídas y el número de semana.
    """
    try:
        print(f"Procesando {nombre_pdf}...")
        pdf_bytes = extraer_pdf_desde_zip(ruta_zip, nombre_pdf)
        pdf_reader = PdfReader(pdf_bytes)
        
        # Intentar buscar la página con las tablas
        pagina_num = buscar_pagina_con_tablas(pdf_reader)
        
        if pagina_num == -1:
            # Si no encontramos la página específica, intentar buscar en todas las páginas
            print(f"  Realizando búsqueda exhaustiva en todas las páginas...")
            for p in range(len(pdf_reader.pages)):
                # Extraer tablas de la página actual y ver si contienen datos relevantes
                tablas_intento = extraer_tablas_de_pagina(pdf_reader, p)
                if tablas_intento and not tablas_intento[0].empty:
                    # Verificar si parece tener datos de enfermedades neurológicas
                    cols = tablas_intento[0].columns
                    if any('Depresion' in col or 'Parkinson' in col or 'Alzheimer' in col for col in cols):
                        print(f"  Encontrada posible tabla en página {p+1}")
                        pagina_num = p
                        break
        
        if pagina_num == -1:
            print(f"  No se encontró ninguna página con tablas en {nombre_pdf}.")
            return [], None
        
        tablas = extraer_tablas_de_pagina(pdf_reader, pagina_num)
        semana = extraer_numero_semana(nombre_pdf)
        
        # Verificar si se obtuvieron datos
        if tablas and not tablas[0].empty:
            print(f"  ✓ Extracción exitosa: {len(tablas[0])} registros")
            return tablas, semana
        else:
            print(f"  ✗ No se pudieron extraer datos de {nombre_pdf}")
            return [], None
    except Exception as e:
        print(f"  ✗ Error al procesar {nombre_pdf}: {e}")
        return [], None

#%% Función principal para procesar un archivo ZIP completo
def procesar_zip_completo(ruta_zip, anio):
    """
    Procesa todos los archivos PDF en un ZIP para extraer y combinar las tablas de interés.

    Parámetros:
    - ruta_zip: Ruta del archivo ZIP.
    - anio: Año de los boletines (para agregar como columna).

    Retorna:
    - Un DataFrame combinado con todos los datos de las semanas.
    """
    # Obtener lista de PDFs en el ZIP
    pdfs = obtener_pdfs_del_zip(ruta_zip)
    print(f"Se encontraron {len(pdfs)} archivos PDF en el ZIP.")
    
    # DataFrame para almacenar todos los datos
    df_completo = pd.DataFrame()
    
    # Procesar cada PDF
    exitos = 0
    fallos = 0
    
    for pdf in pdfs:
        print(f"Procesando {pdf}...")
        tablas, semana = procesar_pdf_desde_zip(ruta_zip, pdf)
        
        if tablas and semana is not None and not tablas[0].empty:
            for tabla in tablas:
                # Verificar si la tabla parece tener la estructura esperada
                if ('Depresion_Sem' in tabla.columns and 
                    'Parkinson_Sem' in tabla.columns and 
                    'Alzheimer_Sem' in tabla.columns):
                    
                    # Agregar columnas de año y semana
                    tabla['ANIO'] = anio
                    tabla['SEMANA'] = semana
                    
                    # Concatenar con el DataFrame completo
                    if df_completo.empty:
                        df_completo = tabla
                    else:
                        df_completo = pd.concat([df_completo, tabla], ignore_index=True)
                    
                    exitos += 1
                else:
                    print(f"  ⚠️ La tabla extraída no tiene la estructura esperada. Columnas: {tabla.columns.tolist()}")
                    fallos += 1
        else:
            fallos += 1
    
    print(f"\nResumen de extracción: {exitos} PDFs procesados con éxito, {fallos} PDFs con problemas")
    
    return df_completo

#%% Función para guardar el DataFrame combinado
def guardar_dataframe(df, ruta_salida, anio):
    """
    Guarda el DataFrame en un archivo CSV.
    Crea la carpeta de destino si no existe.
    
    Parámetros:
    - df: DataFrame a guardar
    - ruta_salida: Carpeta donde guardar el archivo
    - anio: Año para el nombre del archivo
    """
    # Crear la carpeta de destino si no existe
    if not os.path.exists(ruta_salida):
        try:
            os.makedirs(ruta_salida)
            print(f"Se creó la carpeta: {ruta_salida}")
        except Exception as e:
            print(f"Error al crear la carpeta {ruta_salida}: {e}")
            # Si no se puede crear la carpeta, guardar en el directorio actual
            ruta_salida = os.path.dirname(os.path.abspath(__file__))
            print(f"Se guardará en el directorio actual: {ruta_salida}")
    
    nombre_archivo = f"datos_epidemiologicos_{anio}.csv"
    ruta_completa = os.path.join(ruta_salida, nombre_archivo)
    
    try:
        df.to_csv(ruta_completa, index=False, encoding='utf-8')
        print(f"Datos guardados en {ruta_completa}")
    except Exception as e:
        # Si falla, intentar guardar en el directorio actual con un nombre temporal
        ruta_alternativa = f"datos_epidemiologicos_{anio}_temp.csv"
        df.to_csv(ruta_alternativa, index=False, encoding='utf-8')
        print(f"Error al guardar en la ruta original: {e}")
        print(f"Datos guardados en ruta alternativa: {ruta_alternativa}")

#%% Función para ejecutar el proceso completo
def ejecutar_extraccion_anual(ruta_zip, anio, ruta_salida=None):
    """
    Ejecuta el proceso completo de extracción y combinación de datos para un año.
    
    Parámetros:
    - ruta_zip: Ruta del archivo ZIP del año
    - anio: Año de los datos
    - ruta_salida: Carpeta donde guardar el resultado (opcional)
    
    Retorna:
    - DataFrame combinado con todos los datos del año
    """
    print(f"Iniciando extracción de datos para el año {anio}...")
    
    # Procesar el ZIP completo
    df_anual = procesar_zip_completo(ruta_zip, anio)
    
    # Mostrar resumen
    print(f"\nResumen de datos extraídos para {anio}:")
    print(f"Total de registros: {len(df_anual)}")
    print(f"Semanas procesadas: {df_anual['SEMANA'].nunique()}")
    print(f"Entidades federativas: {df_anual['ENTIDAD_FEDERATIVA'].nunique()}")
    
    # Guardar si se proporciona ruta
    if ruta_salida and not df_anual.empty:
        guardar_dataframe(df_anual, ruta_salida, anio)
    
    return df_anual

#%% Función para realizar un análisis detallado de los datos
def analizar_datos(df):
    """
    Realiza un análisis detallado de los datos extraídos.
    
    Parámetros:
    - df: DataFrame con los datos extraídos
    """
    if df.empty:
        print("No hay datos para analizar.")
        return
    
    # Análisis básico
    print("\n===== ANÁLISIS DE LOS DATOS EXTRAÍDOS =====")
    print(f"Total de registros: {len(df)}")
    
    # Análisis por semanas
    semanas = df['SEMANA'].unique()
    print(f"\n• Semanas procesadas: {len(semanas)}")
    print(f"  Rango: {min(semanas)} - {max(semanas)}")
    
    # Verificar semanas faltantes
    semanas_esperadas = set(range(1, 54))  # Algunos años tienen 53 semanas
    semanas_faltantes = semanas_esperadas - set(semanas)
    if semanas_faltantes:
        print(f"  Semanas faltantes: {sorted(semanas_faltantes)}")
    
    # Análisis por entidades
    entidades = df['ENTIDAD_FEDERATIVA'].unique()
    print(f"\n• Entidades federativas: {len(entidades)}")
    
    # Análisis por enfermedades
    print("\n• Estadísticas por enfermedad:")
    
    for enfermedad, prefijo in [
        ("Depresión", "Depresion"), 
        ("Parkinson", "Parkinson"), 
        ("Alzheimer", "Alzheimer")
    ]:
        total = df[f'{prefijo}_Sem'].sum()
        hombres = df[f'{prefijo}_M'].sum()
        mujeres = df[f'{prefijo}_F'].sum()
        
        print(f"  {enfermedad}:")
        print(f"    - Total de casos: {total}")
        print(f"    - Hombres: {hombres} ({hombres/total*100:.1f}%)")
        print(f"    - Mujeres: {mujeres} ({mujeres/total*100:.1f}%)")
    
    # Verificar la consistencia de los datos
    inconsistencias = 0
    for prefijo in ["Depresion", "Parkinson", "Alzheimer"]:
        # Verificar si la suma de M + F = Sem
        df['suma_mf'] = df[f'{prefijo}_M'] + df[f'{prefijo}_F']
        df['diferencia'] = df['suma_mf'] - df[f'{prefijo}_Sem']
        
        inconsistentes = df[df['diferencia'] != 0]
        if not inconsistentes.empty:
            inconsistencias += len(inconsistentes)
            print(f"\n• Se encontraron {len(inconsistentes)} registros donde M+F ≠ Total en {prefijo}")
        
        # Eliminar columnas temporales
        df.drop(['suma_mf', 'diferencia'], axis=1, inplace=True)
    
    if inconsistencias == 0:
        print("\n• Los datos son consistentes (M+F = Total en todas las enfermedades)")
    
    print("\n=========================================")

#%% Ejemplo de uso
if __name__ == "__main__":
    # Parámetros
    ruta_zip = r"./2014.zip"
    anio = 2014
    ruta_salida = r"./Datos Procesados"
    
    # Ejecutar extracción con manejo de excepciones
    try:
        df_2014 = ejecutar_extraccion_anual(ruta_zip, anio, ruta_salida)
        
        # Mostrar las primeras filas del resultado
        print("\nPrimeras filas del DataFrame resultante:")
        print(df_2014.head())
        
        # Realizar análisis detallado
        analizar_datos(df_2014)
        
    except Exception as e:
        print(f"\nError durante la ejecución: {e}")
        print("Intenta verificar las rutas y que tienes permisos de escritura.")
        
    print("\n¡Proceso completado!")
    print(f"Nota: Los datos están disponibles en el DataFrame 'df_{anio}' para análisis adicionales.")