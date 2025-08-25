"""
Utilidades centralizadas para procesamiento de LaTeX
Funciones para manipulación de documentos LaTeX, tablas, imágenes y compilación
"""

import os
import subprocess
import csv
import re
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Union
from PIL import Image
import shutil


def escape_for_latex(text: str) -> str:
    """
    Escapa caracteres especiales para LaTeX manteniendo comandos existentes
    
    Parameters
    ----------
    text : str
        Texto a escapar
        
    Returns
    -------
    str
        Texto escapado para LaTeX
    """
    # Primero escapar dobles backslashes
    text = re.sub(r'\\\\(?=\s*(?:\n|$))', r'\\\\\\\\', text)
    # Luego escapar backslashes seguidos de letras (comandos LaTeX)
    text = re.sub(r'(?<!\\)\\([a-zA-Z])', r'\\\\\1', text)
    return text


def escape_text(text: str) -> str:
    """
    Escapa caracteres reservados de LaTeX sin modificar escapes ya existentes
    ni comandos LaTeX
    
    Parameters
    ----------
    text : str
        Texto a escapar
        
    Returns
    -------
    str
        Texto escapado
    """
    replacements = {
        '{': r'\{',
        '}': r'\}',
        '$': r'\$',
        '&': r'\&',
        '#': r'\#',
        '_': r'\_',
        '^': r'\^{}',
        '%': r'\%',
        '~': r'\textasciitilde{}',
        '|': r'\textbar{}',
        '\\': r'\textbackslash{}'
    }

    # Escapar caracteres excepto barra invertida
    pattern = re.compile(r'(?<!\\)([{}$&#_^%~|])')
    text = pattern.sub(lambda m: replacements[m.group(1)], text)

    # Escapar barra invertida SOLO si no va seguida de letras ni de comandos
    backslash_pattern = re.compile(r'\\(?![a-zA-Z]|\\|\{|%|#|_|\^|~|\|)')
    text = backslash_pattern.sub(replacements['\\'], text)

    return text


def extract_table(content: str, caption: str) -> Optional[str]:
    """
    Extrae una tabla de un documento LaTeX basándose en su caption
    
    Parameters
    ----------
    content : str
        Contenido del documento LaTeX
    caption : str
        Caption de la tabla a extraer
        
    Returns
    -------
    Optional[str]
        Código LaTeX de la tabla o None si no se encuentra
    """
    # Buscar todas las tablas en el documento
    tables = re.findall(r'(\\begin{table}(.|\n)*?\\end{table})', content)
    
    for table, _ in tables:
        if f'\\caption{{{caption}}}' in table:
            return table
    
    return None


def extract_table_from_file(file_path: Union[str, Path], caption: str) -> Optional[str]:
    """
    Extrae una tabla de un archivo LaTeX
    
    Parameters
    ----------
    file_path : str or Path
        Ruta al archivo LaTeX
    caption : str
        Caption de la tabla a extraer
        
    Returns
    -------
    Optional[str]
        Código LaTeX de la tabla o None si no se encuentra
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        return extract_table(content, caption)
    except (FileNotFoundError, UnicodeDecodeError) as e:
        print(f"Error leyendo archivo {file_path}: {e}")
        return None


def get_table_rows(latex_table: str) -> List[str]:
    """
    Extrae las filas de una tabla LaTeX
    
    Parameters
    ----------
    latex_table : str
        Código LaTeX de la tabla
        
    Returns
    -------
    List[str]
        Lista de filas de la tabla
    """
    # Extraer solo el contenido del entorno tabular
    tabular_pattern = re.compile(r'\\begin{tabular}.*?\\end{tabular}', re.DOTALL)
    match = tabular_pattern.search(latex_table)
    
    if not match:
        return []
    
    table_content = match.group()
    
    # Remover begin y end
    table_content = re.sub(r'^.*\\begin{tabular}\{.*?\}.*$\n?', '', table_content, flags=re.MULTILINE)
    table_content = re.sub(r'^.*\\end{tabular}.*$\n?', '', table_content, flags=re.MULTILINE)
    
    # Remover comentarios
    table_content = re.sub(r'%.*', '', table_content)
    
    # Remover líneas horizontales y saltos innecesarios
    table_content = re.sub(r'\\hline', '', table_content)
    table_content = re.sub(r'\\cline\{[^}]*\}', '', table_content)
    table_content = re.sub(r'\n', '', table_content)
    table_content = table_content.rstrip('\n').rstrip(r'\\')
    
    # Separar filas
    rows = table_content.split(r'\\')
    
    return [row.strip() for row in rows if row.strip()]


def highlight_cell(latex_table: str,
                   row_key: Union[str, int],
                   column_key: Union[str, int],
                   cellcolor: str = '[rgb]{{.949,0.949,0.949}}',
                   text_color: str = '[rgb]{{1,0,0}}',
                   text_bf: bool = True,
                   row_index: bool = False,
                   column_index: bool = False,
                   highlight_column: bool = True,
                   highlight_row: bool = False,
                   first_row: int = 2) -> str:
    """
    Resalta una celda específica en una tabla LaTeX
    
    Parameters
    ----------
    latex_table : str
        Código LaTeX de la tabla
    row_key : str or int
        Identificador de la fila
    column_key : str or int
        Identificador de la columna
    cellcolor : str
        Color de fondo de la celda
    text_color : str
        Color del texto
    text_bf : bool
        Si el texto debe ser negrita
    row_index : bool
        Si row_key es un índice numérico
    column_index : bool
        Si column_key es un índice numérico
    highlight_column : bool
        Si resaltar toda la columna
    highlight_row : bool
        Si resaltar toda la fila
    first_row : int
        Índice de la primera fila de datos
        
    Returns
    -------
    str
        Tabla LaTeX con la celda resaltada
    """
    table_rows = get_table_rows(latex_table)
    
    if not table_rows:
        return latex_table
    
    # Determinar índice de fila
    if not row_index:
        row_key = str(row_key)
        row_idx = None
        for i, row in enumerate(table_rows[first_row:]):
            if row.split('&')[0].strip() == row_key.strip():
                row_idx = i
                break
        if row_idx is None:
            return latex_table
    else:
        row_idx = row_key
    
    # Determinar índice de columna
    if not column_index and column_key is not None:
        column_key = str(column_key)
        col_idx = None
        if first_row > 0 and len(table_rows) > first_row - 1:
            header_row = table_rows[first_row - 1]
            for i, cell in enumerate(header_row.split('&')):
                # Extraer texto de comandos LaTeX como \textbf{}
                cell_text = re.sub(r'\\textbf\{(.*?)\}', r'\1', cell).strip()
                if cell_text == column_key.strip():
                    col_idx = i
                    break
        if col_idx is None:
            return latex_table
    else:
        col_idx = column_key
    
    # Crear patrones de formato
    base_format = r'{cell} \\cellcolor' + cellcolor
    cell_pattern = r'\\textbf{{{cell}}}' if text_bf else r'{cell}'
    
    if text_color:
        cell_pattern = r'\\textcolor' + text_color + '{{' + cell_pattern + '}}'
    if cellcolor:
        cell_pattern = cell_pattern + r' \\cellcolor' + cellcolor
    
    # Aplicar formato
    modified_table = latex_table
    
    for i, row in enumerate(table_rows[first_row:]):
        cells = row.split('&')
        
        if len(cells) <= col_idx:
            continue
            
        if i == row_idx:
            # Celda principal
            if col_idx < len(cells):
                original_cell = cells[col_idx].strip()
                formatted_cell = cell_pattern.format(cell=original_cell)
                cells[col_idx] = formatted_cell
                
                if highlight_row:
                    # Aplicar color de fila a todas las celdas
                    for j in range(len(cells)):
                        if j != col_idx:
                            cells[j] = base_format.format(cell=cells[j].strip())
                
                new_row = '&'.join(cells)
                modified_table = modified_table.replace(row, new_row)
        
        elif highlight_column and col_idx < len(cells):
            # Otras filas de la columna
            original_cell = cells[col_idx].strip()
            formatted_cell = base_format.format(cell=original_cell)
            cells[col_idx] = formatted_cell
            new_row = '&'.join(cells)
            modified_table = modified_table.replace(row, new_row)
    
    return modified_table


def highlight_row(latex_table: str,
                  key: Union[str, int],
                  cellcolor: str = '[rgb]{{.949,0.949,0.949}}',
                  text_color: Optional[str] = None,
                  text_bf: bool = False,
                  row_index: bool = False,
                  first_row: int = 2) -> str:
    """
    Resalta una fila completa en una tabla LaTeX
    
    Parameters
    ----------
    latex_table : str
        Código LaTeX de la tabla
    key : str or int
        Identificador de la fila
    cellcolor : str
        Color de fondo
    text_color : str, optional
        Color del texto
    text_bf : bool
        Si el texto debe ser negrita
    row_index : bool
        Si key es un índice numérico
    first_row : int
        Índice de la primera fila de datos
        
    Returns
    -------
    str
        Tabla LaTeX con la fila resaltada
    """
    table_rows = get_table_rows(latex_table)
    
    if not table_rows:
        return latex_table
    
    # Determinar índice de fila
    if not row_index:
        key = str(key)
        row_idx = None
        for i, row in enumerate(table_rows):
            if row.split('&')[0].strip() == key.strip():
                row_idx = i
                break
        if row_idx is None:
            return latex_table
    else:
        row_idx = first_row + key
    
    if row_idx >= len(table_rows):
        return latex_table
    
    row = table_rows[row_idx]
    
    # Crear patrón de formato
    pattern = r'\\textbf{{{cell}}}' if text_bf else r'{cell}'
    if text_color:
        pattern = r'\\textcolor' + text_color + '{{' + pattern + '}}'
    if cellcolor:
        pattern = pattern + r' \\cellcolor' + cellcolor
    
    # Aplicar formato a todas las celdas de la fila
    cells = row.split('&')
    new_cells = [pattern.format(cell=cell.strip()) for cell in cells]
    new_row = '&'.join(new_cells)
    
    return latex_table.replace(row, new_row)


def highlight_column(latex_table: str,
                     key: Union[str, int],
                     cellcolor: str = '[rgb]{{.949,0.949,0.949}}',
                     text_color: Optional[str] = None,
                     text_bf: bool = False,
                     column_index: bool = False,
                     first_row: int = 2) -> str:
    """
    Resalta una columna completa en una tabla LaTeX
    
    Parameters
    ----------
    latex_table : str
        Código LaTeX de la tabla
    key : str or int
        Identificador de la columna
    cellcolor : str
        Color de fondo
    text_color : str, optional
        Color del texto
    text_bf : bool
        Si el texto debe ser negrita
    column_index : bool
        Si key es un índice numérico
    first_row : int
        Índice de la primera fila de datos
        
    Returns
    -------
    str
        Tabla LaTeX con la columna resaltada
    """
    table_rows = get_table_rows(latex_table)
    
    if not table_rows:
        return latex_table
    
    # Determinar índice de columna
    if not column_index:
        key = str(key)
        col_idx = None
        if first_row > 0 and len(table_rows) > first_row - 1:
            header_row = table_rows[first_row - 1]
            for i, cell in enumerate(header_row.split('&')):
                cell_text = re.sub(r'\\textbf\{(.*?)\}', r'\1', cell).strip()
                if cell_text == key.strip():
                    col_idx = i
                    break
        if col_idx is None:
            return latex_table
    else:
        col_idx = key
    
    # Crear patrón de formato
    pattern = r'\\textbf{{{cell}}}' if text_bf else r'{cell}'
    if text_color:
        pattern = r'\\textcolor' + text_color + '{{' + pattern + '}}'
    if cellcolor:
        pattern = pattern + r' \\cellcolor' + cellcolor
    
    # Aplicar formato a todas las celdas de la columna
    modified_table = latex_table
    
    for row in table_rows[first_row:]:
        cells = row.split('&')
        if col_idx < len(cells):
            original_cell = cells[col_idx].strip()
            formatted_cell = pattern.format(cell=original_cell)
            
            # Reemplazar en el contexto de la fila completa
            new_row = row
            if '&' + original_cell in new_row:
                new_row = new_row.replace('&' + original_cell, '&' + formatted_cell)
            elif original_cell + '&' in new_row:
                new_row = new_row.replace(original_cell + '&', formatted_cell + '&')
            
            modified_table = modified_table.replace(row, new_row)
    
    return modified_table


def dataframe_to_latex(df: pd.DataFrame,
                      columns: Optional[List[str]] = None,
                      decimals: int = 2,
                      escape: bool = True,
                      caption: Optional[str] = None,
                      label: Optional[str] = None,
                      position: str = 'H') -> str:
    """
    Convierte un DataFrame de pandas a una tabla LaTeX
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame a convertir
    columns : List[str], optional
        Nombres personalizados para las columnas
    decimals : int
        Número de decimales para valores numéricos
    escape : bool
        Si escapar caracteres especiales de LaTeX
    caption : str, optional
        Caption de la tabla
    label : str, optional
        Label para referencias cruzadas
    position : str
        Posición de la tabla (H, t, b, etc.)
        
    Returns
    -------
    str
        Código LaTeX de la tabla
    """
    df_copy = df.copy()
    
    if columns is not None:
        df_copy.columns = columns
    
    column_length = len(df_copy.columns)
    
    # Usar Styler para formateo
    styler = df_copy.style.format(precision=decimals).hide(axis="index")
    
    # Convertir a LaTeX
    latex_table = styler.to_latex(
        hrules=True, 
        column_format='c' * column_length,
        caption=caption,
        label=label,
        position=position
    )
    
    if escape:
        # Reemplazar _ por \_ en LaTeX
        latex_table = latex_table.replace('_', r'\_')
    
    return latex_table


def distribute_images(image_1_path: Union[str, Path], 
                     image_2_path: Union[str, Path]) -> Tuple[str, str]:
    """
    Calcula el ancho proporcional para dos imágenes en LaTeX
    
    Parameters
    ----------
    image_1_path : str or Path
        Ruta a la primera imagen
    image_2_path : str or Path
        Ruta a la segunda imagen
        
    Returns
    -------
    Tuple[str, str]
        Anchos proporcionales como strings LaTeX (ej: "0.60\\textwidth")
    """
    try:
        # Cargar imágenes y obtener dimensiones
        im_1 = Image.open(image_1_path)
        width_1, height_1 = im_1.size
        relation_1 = width_1 / height_1
        
        im_2 = Image.open(image_2_path)
        width_2, height_2 = im_2.size
        relation_2 = width_2 / height_2
        
        # Calcular proporciones
        total_relation = relation_1 + relation_2
        width_1_prop = 0.95 * relation_1 / total_relation
        width_2_prop = 0.95 * relation_2 / total_relation
        
        return f'{width_1_prop:.2f}\\textwidth', f'{width_2_prop:.2f}\\textwidth'
        
    except Exception as e:
        print(f"Error calculando distribución de imágenes: {e}")
        return '0.45\\textwidth', '0.45\\textwidth'


def save_variable_to_csv(key: str, value: str, file_path: Union[str, Path]) -> None:
    """
    Guarda una variable clave-valor en un archivo CSV
    
    Parameters
    ----------
    key : str
        Nombre de la variable
    value : str
        Valor de la variable
    file_path : str or Path
        Ruta del archivo CSV
    """
    dict_var = {}
    
    # Leer variables existentes
    try:
        with open(file_path, newline='', encoding='utf-8') as file:
            reader = csv.reader(file)
            for row in reader:
                if len(row) >= 2:
                    dict_var[row[0]] = row[1]
    except FileNotFoundError:
        pass
    
    # Agregar nueva variable
    dict_var[key] = value
    
    # Escribir archivo actualizado
    with open(file_path, 'w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        for k, v in dict_var.items():
            writer.writerow([k, v])


def read_variables_from_csv(file_path: Union[str, Path]) -> Dict[str, str]:
    """
    Lee variables de un archivo CSV
    
    Parameters
    ----------
    file_path : str or Path
        Ruta del archivo CSV
        
    Returns
    -------
    Dict[str, str]
        Diccionario con las variables
    """
    dict_var = {}
    
    try:
        with open(file_path, encoding='utf-8', newline='') as file:
            reader = csv.reader(file)
            for row in reader:
                if len(row) >= 2:
                    dict_var[row[0]] = row[1]
    except FileNotFoundError:
        pass
    
    return dict_var


def compile_latex(filename: Union[str, Path], 
                 clean_aux: bool = True,
                 runs: int = 1) -> bool:
    """
    Compila un archivo LaTeX a PDF
    
    Parameters
    ----------
    filename : str or Path
        Nombre del archivo LaTeX (con o sin extensión)
    clean_aux : bool
        Si eliminar archivos auxiliares después de compilar
    runs : int
        Número de pasadas de compilación (útil para referencias cruzadas)
        
    Returns
    -------
    bool
        True si la compilación fue exitosa
    """
    filename = str(filename)
    
    # Remover extensión si está presente
    if filename.endswith('.tex'):
        base_name = filename[:-4]
    else:
        base_name = filename
        filename = filename + '.tex'
    
    try:
        # Ejecutar pdflatex
        for _ in range(runs):
            result = subprocess.run(
                ['pdflatex', '-interaction=nonstopmode', filename],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                print(f"Error compilando {filename}:")
                print(result.stdout)
                print(result.stderr)
                return False
        
        # Limpiar archivos auxiliares
        if clean_aux:
            aux_extensions = ['.log', '.aux', '.fdb_latexmk', '.fls', '.toc', '.lof', '.lot']
            for ext in aux_extensions:
                aux_file = base_name + ext
                try:
                    os.remove(aux_file)
                except FileNotFoundError:
                    pass
        
        return True
        
    except FileNotFoundError:
        print("Error: pdflatex no encontrado. Asegúrate de tener LaTeX instalado.")
        return False
    except Exception as e:
        print(f"Error compilando LaTeX: {e}")
        return False


def create_table_wrapper(caption: Optional[str] = None, 
                        textwidth: bool = False,
                        position: str = 'H') -> str:
    """
    Crea un wrapper para tablas LaTeX
    
    Parameters
    ----------
    caption : str, optional
        Caption de la tabla
    textwidth : bool
        Si ajustar al ancho del texto
    position : str
        Posición de la tabla
        
    Returns
    -------
    str
        Template del wrapper
    """
    wrapper = r'{tabular_code}'
    
    if textwidth:
        wrapper = r'\\resizebox{{\\textwidth}}{{!}}{{' + wrapper + '}}'
    
    if caption:
        wrapper = f"""\\begin{{table}}[{position}]
\\centering
\\caption{{{caption}}}
{wrapper}
\\end{{table}}"""
    
    return wrapper


def process_latex_variables(content: str, 
                          variables: Dict[str, any],
                          unit_dict: Optional[Dict[str, float]] = None) -> str:
    """
    Procesa variables en un template LaTeX
    
    Busca patrones como @variable.2f1 y los reemplaza con valores formateados
    
    Parameters
    ----------
    content : str
        Contenido del template LaTeX
    variables : Dict[str, any]
        Diccionario con variables y valores
    unit_dict : Dict[str, float], optional
        Diccionario de unidades para conversión
        
    Returns
    -------
    str
        Contenido con variables reemplazadas
    """
    # Buscar patrones @variable.decimals+unit
    matches = set(re.findall(r'@([a-zA-Z_][a-zA-Z0-9_\\]*)\.(\d)([a-zA-Z0-9_]+)', content))
    
    for match in matches:
        variable = match[0]
        n_decimals = int(match[1])
        unit = match[2]
        
        if variable not in variables:
            continue
        
        value = variables[variable]
        
        # Aplicar conversión de unidades si está disponible
        if unit_dict and unit in unit_dict:
            converted_value = value / unit_dict[unit]
            replacement = f'{converted_value:.{n_decimals}f}'
        elif unit == 'nn':  # Sin formato numérico
            replacement = str(value)
        else:
            # Formato numérico sin conversión
            try:
                replacement = f'{float(value):.{n_decimals}f}'
            except (ValueError, TypeError):
                replacement = str(value)
        
        # Reemplazar en el contenido
        pattern = f'@{variable}.{n_decimals}{unit}'
        content = content.replace(pattern, replacement)
    
    return content


# Funciones de utilidad para compatibilidad
def dataframe_latex(table: pd.DataFrame, **kwargs) -> str:
    """Función de compatibilidad para dataframe_to_latex"""
    return dataframe_to_latex(table, **kwargs)


if __name__ == '__main__':
    # Ejemplos de uso
    print("=== LaTeX Utils - Ejemplos ===\n")
    
    # Ejemplo de escape de texto
    text = "Texto con $símbolos especiales$ & caracteres {especiales}"
    escaped = escape_text(text)
    print(f"Original: {text}")
    print(f"Escapado: {escaped}\n")
    
    # Ejemplo de DataFrame a LaTeX
    df = pd.DataFrame({
        'Piso': ['1', '2', '3'],
        'Altura': [3.0, 6.0, 9.0],
        'Deriva': [0.005, 0.006, 0.004]
    })
    
    latex_table = dataframe_to_latex(
        df, 
        caption="Tabla de ejemplo",
        label="tab:ejemplo",
        decimals=3
    )
    print("=== Tabla LaTeX generada ===")
    print(latex_table[:200] + "...")
    
    # Ejemplo de distribución de imágenes (simulado)
    print(f"\n=== Distribución de imágenes ===")
    print("Si tuviéramos dos imágenes con relaciones 16:9 y 4:3:")
    print("Anchos calculados: 0.64\\textwidth, 0.31\\textwidth")
    
    # Ejemplo de procesamiento de variables
    template = "El factor Z es @Z.2f1 y el periodo es @T.3nn segundos."
    variables = {'Z': 0.35, 'T': 1.25}
    processed = process_latex_variables(template, variables)
    print(f"\n=== Procesamiento de variables ===")
    print(f"Template: {template}")
    print(f"Procesado: {processed}")