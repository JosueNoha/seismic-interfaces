"""
Procesador de tablas para memorias de cálculo sísmico
Centraliza la creación y procesamiento de tablas LaTeX para memorias sísmicas
"""

import re
from typing import Dict, Any, Optional, Union, List
import pandas as pd
from ...core.latex_utils import (
    dataframe_to_latex, create_table_wrapper, escape_text
)


class TableProcessor:
    """
    Procesador centralizado de tablas LaTeX para memorias sísmicas
    
    Proporciona métodos para crear y formatear tablas comunes 
    en análisis sísmico según diferentes normativas
    """
    
    def __init__(self):
        """Inicializa el procesador de tablas"""
        self.default_decimals = 3
        self.default_position = 'H'
    
    def create_table_wrapper(self, caption: Optional[str] = None, 
                           textwidth: bool = False,
                           position: str = None) -> str:
        """
        Crea un wrapper para tablas LaTeX
        
        Parameters
        ----------
        caption : str, optional
            Caption de la tabla
        textwidth : bool
            Si ajustar al ancho del texto
        position : str, optional
            Posición de la tabla
            
        Returns
        -------
        str
            Template del wrapper con placeholder {tabular_code}
        """
        pos = position or self.default_position
        wrapper = r'{tabular_code}'
        
        if textwidth:
            wrapper = r'\\resizebox{{\\textwidth}}{{!}}{{' + '\n' + wrapper + r'}}'
        
        if caption:
            wrapper = r'\\caption{{' + caption + '}} \n' + wrapper
        
        wrapper = (r'\\begin{{table}}[' + pos + ']' + '\n' + 
                  r'\\centering' + '\n' + wrapper + '\n' + 
                  r'\\end{{table}}')
        
        return wrapper
    
    def process_dataframe_to_latex(self, df: pd.DataFrame,
                                 columns: Optional[List[str]] = None,
                                 decimals: int = None,
                                 escape: bool = True) -> str:
        """
        Convierte un DataFrame a código LaTeX tabular
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame a procesar
        columns : List[str], optional
            Nombres personalizados de columnas
        decimals : int, optional
            Número de decimales
        escape : bool
            Si escapar caracteres especiales
            
        Returns
        -------
        str
            Código LaTeX del tabular
        """
        dec = decimals if decimals is not None else self.default_decimals
        return dataframe_to_latex(df, columns=columns, decimals=dec, escape=escape)
    
    def insert_table_in_content(self, content: str, placeholder: str, 
                              table_latex: str) -> str:
        """
        Inserta una tabla en el contenido reemplazando un placeholder
        
        Parameters
        ----------
        content : str
            Contenido de la plantilla
        placeholder : str
            Placeholder a reemplazar (ej: @table_modal)
        table_latex : str
            Código LaTeX de la tabla completa
            
        Returns
        -------
        str
            Contenido con tabla insertada
        """
        escaped_table = escape_text(table_latex)
        return re.sub(re.escape(placeholder), escaped_table, content)
    
    def create_modal_table(self, modal_data: pd.DataFrame, content: str) -> str:
        """
        Crea tabla de períodos y porcentajes de masa participativa
        
        Parameters
        ----------
        modal_data : pd.DataFrame
            Datos modales con columnas: Period, UX, UY, RZ, SumUX, SumUY, SumRZ
        content : str
            Contenido de la plantilla
            
        Returns
        -------
        str
            Contenido con tabla modal insertada
        """
        # Preparar datos
        modal_df = modal_data.copy()
        
        # Asegurar tipos numéricos
        numeric_cols = ['Period', 'UX', 'UY', 'RZ', 'SumUX', 'SumUY', 'SumRZ']
        for col in numeric_cols:
            if col in modal_df.columns:
                modal_df[col] = pd.to_numeric(modal_df[col], errors='coerce')
        
        # Crear wrapper y tabla
        wrapper = self.create_table_wrapper('Periodos y porcentajes de masa participativa')
        table_latex = self.process_dataframe_to_latex(modal_df, decimals=2)
        complete_table = wrapper.format(tabular_code=table_latex)
        
        # Insertar en contenido
        return self.insert_table_in_content(content, r'@table\_modal', complete_table)
    
    def create_torsion_table(self, torsion_x: pd.DataFrame, torsion_y: pd.DataFrame,
                           content: str) -> str:
        """
        Crea tabla de irregularidad torsional
        
        Parameters
        ----------
        torsion_x : pd.DataFrame
            Datos de torsión en dirección X
        torsion_y : pd.DataFrame
            Datos de torsión en dirección Y
        content : str
            Contenido de la plantilla
            
        Returns
        -------
        str
            Contenido con tablas de torsión insertadas
        """
        # Procesar datos X
        if 'Ratio' in torsion_x.columns and 'Story' in torsion_x.columns:
            idx_x = torsion_x.groupby(['Story'])['Ratio'].idxmax()
            data_x = torsion_x.loc[idx_x].reset_index(drop=True)
        else:
            data_x = torsion_x.copy()
        
        # Procesar datos Y
        if 'Ratio' in torsion_y.columns and 'Story' in torsion_y.columns:
            idx_y = torsion_y.groupby(['Story'])['Ratio'].idxmax()
            data_y = torsion_y.loc[idx_y].reset_index(drop=True)
        else:
            data_y = torsion_y.copy()
        
        # Seleccionar columnas relevantes
        torsion_columns = ['Story', 'Max Drift', 'Avg Drift', 'Ratio']
        if 'tor_reg' in data_x.columns:
            torsion_columns.append('tor_reg')
        
        # Filtrar columnas existentes
        available_cols_x = [col for col in torsion_columns if col in data_x.columns]
        available_cols_y = [col for col in torsion_columns if col in data_y.columns]
        
        data_x = data_x[available_cols_x] if available_cols_x else data_x
        data_y = data_y[available_cols_y] if available_cols_y else data_y
        
        # Nombres de columnas para LaTeX
        display_columns = ['Piso', 'Deriva Máxima', 'Deriva Media', 'Ratio']
        if len(available_cols_x) > 4:
            display_columns.append('Regularidad')
        
        # Crear tablas
        wrapper_x = self.create_table_wrapper(
            'Irregularidad Torsional en la dirección X', textwidth=False
        )
        wrapper_y = self.create_table_wrapper(
            'Irregularidad Torsional en la dirección Y', textwidth=False
        )
        
        table_x = self.process_dataframe_to_latex(
            data_x, columns=display_columns[:len(available_cols_x)], decimals=3
        )
        table_y = self.process_dataframe_to_latex(
            data_y, columns=display_columns[:len(available_cols_y)], decimals=3
        )
        
        complete_table_x = wrapper_x.format(tabular_code=table_x)
        complete_table_y = wrapper_y.format(tabular_code=table_y)
        
        # Insertar en contenido
        content = self.insert_table_in_content(content, r'@table\_torsion\_x', complete_table_x)
        content = self.insert_table_in_content(content, r'@table\_torsion\_y', complete_table_y)
        
        return content
    
    def create_mass_table(self, mass_data: pd.DataFrame, content: str) -> str:
        """
        Crea tabla de irregularidad de masa
        
        Parameters
        ----------
        mass_data : pd.DataFrame
            Datos de masa por piso
        content : str
            Contenido de la plantilla
            
        Returns
        -------
        str
            Contenido con tabla de masa insertada
        """
        # Preparar datos
        data = mass_data.copy()
        
        # Reemplazar valores vacíos por 0
        data = data.replace('', 0.0)
        data = data.fillna(0.0)
        
        # Convertir columnas numéricas
        numeric_cols = ['Mass', '1.5 Mass']
        for col in numeric_cols:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
        
        # Seleccionar columnas
        mass_columns = ['Story', 'Mass', '1.5 Mass']
        if 'Tipo de Piso' in data.columns:
            mass_columns.append('Tipo de Piso')
        if 'is_reg' in data.columns:
            mass_columns.append('is_reg')
        
        # Filtrar columnas existentes
        available_cols = [col for col in mass_columns if col in data.columns]
        data = data[available_cols]
        
        # Nombres de columnas para LaTeX
        display_columns = ['Piso', 'Masa', '1.5 Masa']
        if 'Tipo de Piso' in available_cols:
            display_columns.append('Tipo de Piso')
        if 'is_reg' in available_cols:
            display_columns.append('Regularidad')
        
        # Crear tabla
        wrapper = self.create_table_wrapper('Irregularidad de Masa o Peso')
        table_latex = self.process_dataframe_to_latex(
            data, columns=display_columns[:len(available_cols)], decimals=2
        )
        complete_table = wrapper.format(tabular_code=table_latex)
        
        # Insertar en contenido
        return self.insert_table_in_content(content, r'@table\_mass', complete_table)
    
    def create_stiffness_table(self, stiffness_x: pd.DataFrame, stiffness_y: pd.DataFrame,
                             content: str, force_unit: str = 'tonf', 
                             displacement_unit: str = 'mm') -> str:
        """
        Crea tabla de irregularidad de rigidez
        
        Parameters
        ----------
        stiffness_x : pd.DataFrame
            Datos de rigidez en dirección X
        stiffness_y : pd.DataFrame
            Datos de rigidez en dirección Y
        content : str
            Contenido de la plantilla
        force_unit : str
            Unidad de fuerza para las columnas
        displacement_unit : str
            Unidad de desplazamiento para las columnas
            
        Returns
        -------
        str
            Contenido con tablas de rigidez insertadas
        """
        # Procesar datos si es necesario
        data_x = stiffness_x.copy()
        data_y = stiffness_y.copy()
        
        # Seleccionar columnas relevantes
        stiffness_columns = ['Story', 'OutputCase', 'V', 'drift', 'stiff', 
                           '70%k_prev', '80%k_3']
        if 'is_reg' in data_x.columns:
            stiffness_columns.append('is_reg')
        
        # Filtrar columnas existentes
        available_cols_x = [col for col in stiffness_columns if col in data_x.columns]
        available_cols_y = [col for col in stiffness_columns if col in data_y.columns]
        
        data_x = data_x[available_cols_x] if available_cols_x else data_x
        data_y = data_y[available_cols_y] if available_cols_y else data_y
        
        # Nombres de columnas para LaTeX
        display_columns = [
            'Piso', 'Caso de Carga', 
            f'V ({force_unit})',
            rf'\\makecell{{Desplazamiento\\\\Relativo ({displacement_unit})}}',
            rf'\\makecell{{Rigidez\\\\Lateral(k) ({force_unit}/{displacement_unit})}}',
            r'70\%k previo',
            r'80\%Prom(k)'
        ]
        
        if len(available_cols_x) > 7:
            display_columns.append('Regularidad')
        
        # Crear tablas
        wrapper = self.create_table_wrapper('Irregularidad de rigidez', textwidth=True)
        
        table_x = self.process_dataframe_to_latex(
            data_x, 
            columns=display_columns[:len(available_cols_x)], 
            decimals=3,
            escape=False  # Para permitir comandos LaTeX en los headers
        )
        table_y = self.process_dataframe_to_latex(
            data_y, 
            columns=display_columns[:len(available_cols_y)], 
            decimals=3,
            escape=False
        )
        
        complete_table_x = wrapper.format(tabular_code=table_x)
        complete_table_y = wrapper.format(tabular_code=table_y)
        
        # Insertar en contenido
        content = self.insert_table_in_content(content, r'@table\_stiffness\_x', complete_table_x)
        content = self.insert_table_in_content(content, r'@table\_stiffness\_y', complete_table_y)
        
        return content
    
    def create_shear_table(self, static_data: pd.DataFrame, dynamic_data: pd.DataFrame,
                         content: str, force_unit: str = 'tonf') -> str:
        """
        Crea tablas de cortante estático y dinámico
        
        Parameters
        ----------
        static_data : pd.DataFrame
            Datos de cortante estático
        dynamic_data : pd.DataFrame  
            Datos de cortante dinámico
        content : str
            Contenido de la plantilla
        force_unit : str
            Unidad de fuerza
            
        Returns
        -------
        str
            Contenido con tablas de cortante insertadas
        """
        # Preparar datos
        data_static = static_data.copy()
        data_dynamic = dynamic_data.copy()
        
        # Seleccionar columnas de cortante
        shear_columns = ['Story', 'V_x', 'V_y']
        
        # Filtrar columnas existentes
        available_static = [col for col in shear_columns if col in data_static.columns]
        available_dynamic = [col for col in shear_columns if col in data_dynamic.columns]
        
        if available_static:
            data_static = data_static[available_static]
        if available_dynamic:
            data_dynamic = data_dynamic[available_dynamic]
        
        # Nombres de columnas para LaTeX
        display_columns = ['Piso', f'Vx ({force_unit})', f'Vy ({force_unit})']
        
        # Crear tablas
        wrapper_static = self.create_table_wrapper(
            'Cortante Estático Sísmico', textwidth=False
        )
        wrapper_dynamic = self.create_table_wrapper(
            'Cortante Dinámico Sísmico', textwidth=False
        )
        
        table_static = self.process_dataframe_to_latex(
            data_static, columns=display_columns[:len(available_static)], decimals=3
        )
        table_dynamic = self.process_dataframe_to_latex(
            data_dynamic, columns=display_columns[:len(available_dynamic)], decimals=3
        )
        
        complete_static = wrapper_static.format(tabular_code=table_static)
        complete_dynamic = wrapper_dynamic.format(tabular_code=table_dynamic)
        
        # Insertar en contenido
        content = self.insert_table_in_content(content, r'@table\_shear\_static', complete_static)
        content = self.insert_table_in_content(content, r'@table\_shear\_dynamic', complete_dynamic)
        
        return content
    
    def create_drift_table(self, drift_data: pd.DataFrame, content: str) -> str:
        """
        Crea tabla de desplazamientos relativos (derivas)
        
        Parameters
        ----------
        drift_data : pd.DataFrame
            Datos de derivas
        content : str
            Contenido de la plantilla
            
        Returns
        -------
        str
            Contenido con tabla de derivas insertada
        """
        # Preparar datos
        data = drift_data.copy()
        
        # Seleccionar columnas
        drift_columns = ['Story', 'drift x', 'drift y']
        
        # Si las columnas tienen nombres diferentes, mapear
        column_mapping = {
            'drift_x': 'drift x',
            'drift_y': 'drift y',
            'Drift_X': 'drift x',
            'Drift_Y': 'drift y'
        }
        
        for old_name, new_name in column_mapping.items():
            if old_name in data.columns and new_name not in data.columns:
                data = data.rename(columns={old_name: new_name})
        
        # Filtrar columnas disponibles
        available_cols = [col for col in drift_columns if col in data.columns]
        data = data[available_cols] if available_cols else data
        
        # Nombres de columnas para LaTeX
        display_columns = ['Piso', 'Deriva X', 'Deriva Y']
        
        # Crear tabla
        wrapper = self.create_table_wrapper('Desplazamientos Relativos')
        table_latex = self.process_dataframe_to_latex(
            data, columns=display_columns[:len(available_cols)], decimals=3
        )
        complete_table = wrapper.format(tabular_code=table_latex)
        
        # Insertar en contenido
        return self.insert_table_in_content(content, r'@table\_drifts', complete_table)
    
    def create_displacement_table(self, displacement_data: pd.DataFrame, content: str,
                                displacement_unit: str = 'mm') -> str:
        """
        Crea tabla de desplazamientos
        
        Parameters
        ----------
        displacement_data : pd.DataFrame
            Datos de desplazamientos
        content : str
            Contenido de la plantilla
        displacement_unit : str
            Unidad de desplazamiento
            
        Returns
        -------
        str
            Contenido con tabla de desplazamientos insertada
        """
        # Preparar datos
        data = displacement_data.copy()
        
        # Nombres de columnas para LaTeX
        display_columns = [
            'Piso', 
            f'Desplazamiento X ({displacement_unit})',
            f'Desplazamiento Y ({displacement_unit})'
        ]
        
        # Crear tabla
        wrapper = self.create_table_wrapper('Desplazamientos Inelásticos')
        table_latex = self.process_dataframe_to_latex(
            data, columns=display_columns, decimals=3
        )
        complete_table = wrapper.format(tabular_code=table_latex)
        
        # Insertar en contenido
        return self.insert_table_in_content(content, r'@table\_disp', complete_table)
    
    def create_static_analysis_table(self, static_data: pd.DataFrame, content: str) -> str:
        """
        Crea tabla de análisis sísmico estático por pisos
        
        Parameters
        ----------
        static_data : pd.DataFrame
            Datos del análisis estático
        content : str
            Contenido de la plantilla
            
        Returns
        -------
        str
            Contenido con tabla de análisis estático insertada
        """
        # Nombres de columnas para LaTeX (con comandos LaTeX)
        display_columns = [
            'Piso', 'Peso', 'Altura', '$H^k_x$', '$H^k_y$', 
            '$P~H_x$', '$P~H_y$', 'ax', 'ay', 'vx', 'vy'
        ]
        
        # Crear tabla
        wrapper = self.create_table_wrapper('Análisis sísmico estático por pisos')
        table_latex = self.process_dataframe_to_latex(
            static_data, columns=display_columns, decimals=3, escape=False
        )
        complete_table = wrapper.format(tabular_code=table_latex)
        
        # Insertar en contenido
        return self.insert_table_in_content(content, r'@table\_static', complete_table)


# Funciones de utilidad para compatibilidad con código existente
def create_modal_table_legacy(table_data: pd.DataFrame, content: str) -> str:
    """
    Función de compatibilidad para crear tabla modal
    Compatible con el código existente de Bolivia/Perú
    """
    processor = TableProcessor()
    return processor.create_modal_table(table_data, content)


def create_torsion_table_legacy(data_x: pd.DataFrame, data_y: pd.DataFrame, 
                               content: str) -> str:
    """
    Función de compatibilidad para crear tabla de torsión
    Compatible con el código existente de Bolivia/Perú
    """
    processor = TableProcessor()
    return processor.create_torsion_table(data_x, data_y, content)


def create_mass_table_legacy(mass_data: pd.DataFrame, content: str) -> str:
    """
    Función de compatibilidad para crear tabla de masa
    Compatible con el código existente de Bolivia/Perú
    """
    processor = TableProcessor()
    return processor.create_mass_table(mass_data, content)


def create_stiffness_table_legacy(data_x: pd.DataFrame, data_y: pd.DataFrame,
                                content: str, u_f: str = 'tonf', u_d: str = 'mm') -> str:
    """
    Función de compatibilidad para crear tabla de rigidez
    Compatible con el código existente de Bolivia/Perú
    """
    processor = TableProcessor()
    return processor.create_stiffness_table(data_x, data_y, content, u_f, u_d)