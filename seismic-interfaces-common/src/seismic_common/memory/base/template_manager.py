"""
Gestor de plantillas LaTeX para memorias de cálculo sísmico
Maneja la carga, procesamiento y manipulación de plantillas LaTeX
"""

import os
import re
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from ...core.latex_utils import extract_table, highlight_cell, escape_text


class TemplateManager:
    """
    Gestor centralizado de plantillas LaTeX para memorias sísmicas
    
    Proporciona funcionalidades para cargar, procesar y manipular
    plantillas LaTeX de manera uniforme
    """
    
    def __init__(self):
        """Inicializa el gestor de plantillas"""
        self.template_cache = {}
        self._base_templates_path = self._get_base_templates_path()
    
    def _get_base_templates_path(self) -> Path:
        """
        Obtiene la ruta base de las plantillas comunes
        
        Returns
        -------
        Path
            Ruta al directorio de plantillas base
        """
        # Desde memory/base/template_manager.py ir a memory/templates/
        return Path(__file__).parent.parent / "templates"
    
    def load_template(self, template_path: Union[str, Path]) -> str:
        """
        Carga una plantilla LaTeX desde archivo
        
        Parameters
        ----------
        template_path : str or Path
            Ruta al archivo de plantilla
            
        Returns
        -------
        str
            Contenido de la plantilla
            
        Raises
        ------
        FileNotFoundError
            Si la plantilla no existe
        """
        template_path = Path(template_path)
        
        # Usar cache si está disponible
        cache_key = str(template_path.resolve())
        if cache_key in self.template_cache:
            return self.template_cache[cache_key]
        
        if not template_path.exists():
            raise FileNotFoundError(f"Plantilla no encontrada: {template_path}")
        
        try:
            with open(template_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            # Guardar en cache
            self.template_cache[cache_key] = content
            return content
            
        except Exception as e:
            raise IOError(f"Error leyendo plantilla {template_path}: {e}")
    
    def get_base_template(self, template_name: str = "base_template.ltx") -> str:
        """
        Obtiene una plantilla base común
        
        Parameters
        ----------
        template_name : str
            Nombre de la plantilla base
            
        Returns
        -------
        str
            Contenido de la plantilla base
        """
        template_path = self._base_templates_path / template_name
        return self.load_template(template_path)
    
    def get_common_sections(self, section_name: str = "common_sections.ltx") -> str:
        """
        Obtiene secciones comunes LaTeX
        
        Parameters
        ----------
        section_name : str
            Nombre del archivo de secciones comunes
            
        Returns
        -------
        str
            Contenido de las secciones comunes
        """
        sections_path = self._base_templates_path / section_name
        return self.load_template(sections_path)
    
    def extract_table_from_template(self, content: str, caption: str) -> str:
        """
        Extrae una tabla específica de una plantilla por su caption
        
        Parameters
        ----------
        content : str
            Contenido de la plantilla
        caption : str
            Caption de la tabla a extraer
            
        Returns
        -------
        str
            Código LaTeX de la tabla
        """
        return extract_table(content, caption)
    
    def highlight_table_cell(self, latex_table: str, 
                            row_key: Union[str, int],
                            column_key: Union[str, int],
                            **kwargs) -> str:
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
        **kwargs
            Argumentos adicionales para personalizar el resaltado
            
        Returns
        -------
        str
            Tabla LaTeX con la celda resaltada
        """
        return highlight_cell(latex_table, row_key, column_key, **kwargs)
    
    def replace_table_in_content(self, content: str, 
                               original_table: str, 
                               new_table: str) -> str:
        """
        Reemplaza una tabla en el contenido de manera segura
        
        Parameters
        ----------
        content : str
            Contenido original
        original_table : str
            Tabla original a reemplazar
        new_table : str
            Nueva tabla
            
        Returns
        -------
        str
            Contenido con la tabla reemplazada
        """
        # Escapar caracteres especiales para el reemplazo
        escaped_original = re.escape(original_table)
        escaped_new = escape_text(new_table)
        
        return re.sub(escaped_original, escaped_new, content)
    
    def process_factor_table(self, content: str, 
                           table_caption: str,
                           highlight_value: Union[str, int],
                           highlight_column: str,
                           **highlight_options) -> str:
        """
        Procesa una tabla de factores sísmicos (zona, suelo, etc.)
        
        Función genérica para procesar tablas de factores sísmicos
        comunes como factor de zona, factor de suelo, etc.
        
        Parameters
        ----------
        content : str
            Contenido de la plantilla
        table_caption : str
            Caption de la tabla a procesar
        highlight_value : str or int
            Valor a resaltar en la tabla
        highlight_column : str
            Columna donde está el valor
        **highlight_options
            Opciones adicionales de resaltado
            
        Returns
        -------
        str
            Contenido con la tabla procesada
        """
        # Extraer tabla original
        original_table = self.extract_table_from_template(content, table_caption)
        
        if not original_table:
            return content  # Si no se encuentra la tabla, devolver contenido sin cambios
        
        # Resaltar celda específica
        highlighted_table = self.highlight_table_cell(
            original_table, 
            highlight_value, 
            highlight_column,
            **highlight_options
        )
        
        # Reemplazar en el contenido
        return self.replace_table_in_content(content, original_table, highlighted_table)
    
    def process_zone_factor_table(self, content: str, 
                                zone_value: Union[str, int],
                                caption: Optional[str] = None) -> str:
        """
        Procesa tabla de factor de zona sísmica
        
        Parameters
        ----------
        content : str
            Contenido de la plantilla
        zone_value : str or int
            Valor de zona sísmica
        caption : str, optional
            Caption personalizado de la tabla
            
        Returns
        -------
        str
            Contenido con tabla de zona procesada
        """
        # Caption por defecto común
        default_caption = 'Factor de zona'
        table_caption = caption or default_caption
        
        return self.process_factor_table(
            content=content,
            table_caption=table_caption,
            highlight_value=zone_value,
            highlight_column='Z',
            highlight_column=False  # Solo resaltar la celda específica
        )
    
    def process_soil_factor_table(self, content: str,
                                zone_value: Union[str, int],
                                soil_value: Union[str, int],
                                caption: Optional[str] = None) -> str:
        """
        Procesa tabla de factor de suelo
        
        Parameters
        ----------
        content : str
            Contenido de la plantilla
        zone_value : str or int
            Valor de zona sísmica
        soil_value : str or int
            Tipo de suelo
        caption : str, optional
            Caption personalizado de la tabla
            
        Returns
        -------
        str
            Contenido con tabla de suelo procesada
        """
        # Caption por defecto común
        default_caption = 'Factor de suelo'
        table_caption = caption or default_caption
        
        return self.process_factor_table(
            content=content,
            table_caption=table_caption,
            highlight_value=zone_value,
            highlight_column=soil_value
        )
    
    def process_usage_factor_table(self, content: str,
                                 usage_category: Union[str, int],
                                 caption: Optional[str] = None) -> str:
        """
        Procesa tabla de factor de uso o importancia
        
        Parameters
        ----------
        content : str
            Contenido de la plantilla
        usage_category : str or int
            Categoría de uso de la edificación
        caption : str, optional
            Caption personalizado de la tabla
            
        Returns
        -------
        str
            Contenido con tabla de uso procesada
        """
        # Caption por defecto común
        default_caption = 'Factor de Uso o Importancia'
        table_caption = caption or default_caption
        
        return self.process_factor_table(
            content=content,
            table_caption=table_caption,
            highlight_value=usage_category,
            highlight_column='FACTOR U',
            highlight_column=False,
            row_index=True,
            cellcolor='[rgb]{{1,0.949,0.8}}'
        )
    
    def process_structural_system_table(self, content: str,
                                      system_x: str,
                                      system_y: str,
                                      direction: str = 'both',
                                      caption: Optional[str] = None) -> str:
        """
        Procesa tabla de sistemas estructurales
        
        Parameters
        ----------
        content : str
            Contenido de la plantilla
        system_x : str
            Sistema estructural en dirección X
        system_y : str
            Sistema estructural en dirección Y
        direction : str
            Dirección a procesar ('x', 'y', 'both')
        caption : str, optional
            Caption personalizado de la tabla
            
        Returns
        -------
        str
            Contenido con tabla de sistemas procesada
        """
        # Caption por defecto común
        default_caption = 'Sistemas estructurales'
        table_caption = caption or default_caption
        
        if direction in ['x', 'both']:
            content = self.process_factor_table(
                content=content,
                table_caption=table_caption,
                highlight_value=system_x,
                highlight_column='Rox' if 'Rox' in content else 'R',
                highlight_column=False
            )
        
        if direction in ['y', 'both'] and system_y != system_x:
            content = self.process_factor_table(
                content=content,
                table_caption=table_caption,
                highlight_value=system_y,
                highlight_column='Roy' if 'Roy' in content else 'R',
                highlight_column=False
            )
        
        return content
    
    def process_soil_periods_table(self, content: str,
                                 soil_type: Union[str, int],
                                 caption: Optional[str] = None) -> str:
        """
        Procesa tabla de períodos del suelo
        
        Parameters
        ----------
        content : str
            Contenido de la plantilla
        soil_type : str or int
            Tipo de suelo
        caption : str, optional
            Caption personalizado de la tabla
            
        Returns
        -------
        str
            Contenido con tabla de períodos procesada
        """
        # Caption por defecto común  
        default_caption = 'Períodos del suelo'
        table_caption = caption or default_caption
        
        return self.process_factor_table(
            content=content,
            table_caption=table_caption,
            highlight_value=soil_type,
            highlight_column='Tipo',
            highlight_column=False
        )
    
    def insert_variable_placeholders(self, content: str, 
                                   variables: Dict[str, str]) -> str:
        """
        Inserta placeholders de variables en el contenido
        
        Parameters
        ----------
        content : str
            Contenido de la plantilla
        variables : Dict[str, str]
            Diccionario de variables y sus placeholders
            
        Returns
        -------
        str
            Contenido con placeholders insertados
        """
        for var_name, placeholder in variables.items():
            # Buscar y reemplazar marcadores específicos
            pattern = f"@{var_name}@"
            content = content.replace(pattern, placeholder)
        
        return content
    
    def validate_template(self, content: str) -> Dict[str, Any]:
        """
        Valida la estructura básica de una plantilla LaTeX
        
        Parameters
        ----------
        content : str
            Contenido de la plantilla a validar
            
        Returns
        -------
        Dict[str, Any]
            Resultado de la validación con errores y advertencias
        """
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'info': {}
        }
        
        # Verificar estructura básica de documento
        if '\\documentclass' not in content:
            validation_result['errors'].append("Falta declaración \\documentclass")
        
        if '\\begin{document}' not in content:
            validation_result['errors'].append("Falta \\begin{document}")
        
        if '\\end{document}' not in content:
            validation_result['errors'].append("Falta \\end{document}")
        
        # Verificar balance de llaves
        open_braces = content.count('{')
        close_braces = content.count('}')
        if open_braces != close_braces:
            validation_result['warnings'].append(
                f"Desbalance de llaves: {open_braces} abiertas, {close_braces} cerradas"
            )
        
        # Contar tablas
        table_count = len(re.findall(r'\\begin{table}', content))
        validation_result['info']['tables_found'] = table_count
        
        # Contar figuras
        figure_count = len(re.findall(r'\\begin{figure}', content))
        validation_result['info']['figures_found'] = figure_count
        
        # Marcar como inválido si hay errores
        if validation_result['errors']:
            validation_result['valid'] = False
        
        return validation_result
    
    def clear_cache(self):
        """Limpia el cache de plantillas"""
        self.template_cache.clear()
    
    def get_cached_templates(self) -> List[str]:
        """
        Obtiene la lista de plantillas en cache
        
        Returns
        -------
        List[str]
            Lista de rutas de plantillas en cache
        """
        return list(self.template_cache.keys())


# Funciones de utilidad para compatibilidad con código existente
def load_template_content(template_path: Union[str, Path]) -> str:
    """
    Función de utilidad para cargar una plantilla
    
    Parameters
    ----------
    template_path : str or Path
        Ruta de la plantilla
        
    Returns
    -------
    str
        Contenido de la plantilla
    """
    manager = TemplateManager()
    return manager.load_template(template_path)


def process_zone_table(content: str, zone_value: Union[str, int], 
                      template_path: Optional[Union[str, Path]] = None) -> str:
    """
    Función de utilidad para procesar tabla de zona
    Compatible con el código existente
    
    Parameters
    ----------
    content : str, optional
        Contenido de la plantilla
    zone_value : str or int
        Valor de zona sísmica
    template_path : str or Path, optional
        Ruta de la plantilla si content es None
        
    Returns
    -------
    str
        Contenido procesado
    """
    if content is None and template_path:
        manager = TemplateManager()
        content = manager.load_template(template_path)
    elif content is None:
        raise ValueError("Debe proporcionar 'content' o 'template_path'")
    
    manager = TemplateManager()
    return manager.process_zone_factor_table(content, zone_value)


def process_soil_table(content: str, zone_value: Union[str, int], 
                      soil_value: Union[str, int],
                      template_path: Optional[Union[str, Path]] = None) -> str:
    """
    Función de utilidad para procesar tabla de suelo
    Compatible con el código existente
    
    Parameters
    ----------
    content : str, optional
        Contenido de la plantilla
    zone_value : str or int
        Valor de zona sísmica
    soil_value : str or int
        Tipo de suelo
    template_path : str or Path, optional
        Ruta de la plantilla si content es None
        
    Returns
    -------
    str
        Contenido procesado
    """
    if content is None and template_path:
        manager = TemplateManager()
        content = manager.load_template(template_path)
    elif content is None:
        raise ValueError("Debe proporcionar 'content' o 'template_path'")
    
    manager = TemplateManager()
    return manager.process_soil_factor_table(content, zone_value, soil_value)