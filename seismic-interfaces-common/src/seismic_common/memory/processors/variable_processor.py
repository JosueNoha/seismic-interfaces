"""
Procesador de variables para memorias de cálculo sísmico
Centraliza el procesamiento, formateo y conversión de variables para templates LaTeX
"""

import re
import csv
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple, Callable
from ...core.latex_utils import process_latex_variables
from ...core.unit_tool import Units, create_unit_dict, UnitSystem


class VariableProcessor:
    """
    Procesador centralizado de variables para memorias sísmicas
    
    Proporciona métodos para procesar, formatear y convertir variables
    en templates LaTeX con soporte completo para unidades de ingeniería
    """
    
    def __init__(self, unit_system: Union[str, UnitSystem] = 'SI'):
        """
        Inicializa el procesador de variables
        
        Parameters
        ----------
        unit_system : str or UnitSystem
            Sistema de unidades ('SI', 'FPS', 'MKS')
        """
        # Configurar sistema de unidades
        if isinstance(unit_system, str):
            self.units = Units(UnitSystem(unit_system))
        else:
            self.units = Units(unit_system)
        
        # Crear diccionario de unidades para compatibilidad
        self.unit_dict = create_unit_dict(self.units)
        
        # Configuración del procesador
        self.default_decimals = 3
        self.variable_registry = {}
        self.conversion_history = []
        
        # Patrón regex para variables LaTeX
        # Formato: @variable.decimales+unidad (ej: @Z.2f1, @T.3nn)
        self.variable_pattern = re.compile(
            r'@([a-zA-Z_][a-zA-Z0-9_\\]*)\.(\d)([a-zA-Z0-9_]+)'
        )
    
    def set_unit_system(self, unit_system: Union[str, UnitSystem]) -> None:
        """
        Cambia el sistema de unidades
        
        Parameters
        ----------
        unit_system : str or UnitSystem
            Nuevo sistema de unidades
        """
        if isinstance(unit_system, str):
            self.units = Units(UnitSystem(unit_system))
        else:
            self.units = Units(unit_system)
        
        self.unit_dict = create_unit_dict(self.units)
    
    def register_variable(self, name: str, value: Any, unit: str = None, 
                         description: str = None) -> None:
        """
        Registra una variable en el sistema
        
        Parameters
        ----------
        name : str
            Nombre de la variable
        value : Any
            Valor de la variable
        unit : str, optional
            Unidad de la variable
        description : str, optional
            Descripción de la variable
        """
        self.variable_registry[name] = {
            'value': value,
            'unit': unit,
            'description': description,
            'type': type(value).__name__
        }
    
    def get_registered_variables(self) -> Dict[str, Dict[str, Any]]:
        """
        Obtiene todas las variables registradas
        
        Returns
        -------
        Dict[str, Dict[str, Any]]
            Diccionario con variables registradas y su información
        """
        return self.variable_registry.copy()
    
    def find_variables_in_content(self, content: str) -> List[Dict[str, str]]:
        """
        Encuentra todas las variables en un contenido LaTeX
        
        Parameters
        ----------
        content : str
            Contenido del template LaTeX
            
        Returns
        -------
        List[Dict[str, str]]
            Lista de variables encontradas con su información
        """
        matches = self.variable_pattern.findall(content)
        variables = []
        
        for match in matches:
            variable_name, decimals, unit = match
            variables.append({
                'name': variable_name,
                'decimals': int(decimals),
                'unit': unit,
                'pattern': f'@{variable_name}.{decimals}{unit}'
            })
        
        return variables
    
    def process_variables(self, variables: Dict[str, Any], content: str,
                         custom_unit_dict: Optional[Dict[str, float]] = None) -> str:
        """
        Procesa variables en un template LaTeX
        
        Parameters
        ----------
        variables : Dict[str, Any]
            Diccionario con variables y valores
        content : str
            Contenido del template LaTeX
        custom_unit_dict : Dict[str, float], optional
            Diccionario personalizado de unidades
            
        Returns
        -------
        str
            Contenido con variables procesadas
        """
        # Usar diccionario personalizado o el por defecto
        unit_dict = custom_unit_dict or self.unit_dict
        
        # Encontrar todas las variables en el contenido
        found_variables = self.find_variables_in_content(content)
        
        # Procesar cada variable encontrada
        for var_info in found_variables:
            var_name = var_info['name']
            decimals = var_info['decimals']
            unit = var_info['unit']
            pattern = var_info['pattern']
            
            if var_name not in variables:
                continue
            
            value = variables[var_name]
            
            # Formatear valor según el tipo de unidad
            if unit == 'nn':
                # Sin formato numérico, usar valor como string
                replacement = str(value)
            elif unit in unit_dict:
                # Conversión con unidades
                try:
                    converted_value = float(value) / unit_dict[unit]
                    replacement = f'{converted_value:.{decimals}f}'
                    
                    # Registrar conversión
                    self.conversion_history.append({
                        'variable': var_name,
                        'original_value': value,
                        'converted_value': converted_value,
                        'unit': unit,
                        'factor': unit_dict[unit]
                    })
                    
                except (ValueError, TypeError, ZeroDivisionError) as e:
                    replacement = f'ERROR({str(e)})'
            else:
                # Formato numérico sin conversión
                try:
                    replacement = f'{float(value):.{decimals}f}'
                except (ValueError, TypeError):
                    replacement = str(value)
            
            # Reemplazar en el contenido
            content = content.replace(pattern, replacement)
        
        return content
    
    def create_extended_unit_dict(self, additional_units: Dict[str, float] = None) -> Dict[str, float]:
        """
        Crea un diccionario extendido de unidades
        
        Parameters
        ----------
        additional_units : Dict[str, float], optional
            Unidades adicionales específicas del proyecto
            
        Returns
        -------
        Dict[str, float]
            Diccionario completo de unidades
        """
        extended_dict = self.unit_dict.copy()
        
        # Agregar unidades derivadas comunes
        derived_units = {
            # Unidades compuestas comunes
            'kgf_m': self.units.kgf * self.units.m,     # Momento
            'tonf_m': self.units.tonf * self.units.m,   # Momento
            'kgf_cm2': self.units.kgf / (self.units.cm ** 2),  # Esfuerzo
            'tonf_m2': self.units.tonf / (self.units.m ** 2),   # Carga distribuida
            'kgf_m2': self.units.kgf / (self.units.m ** 2),     # Carga distribuida
            
            # Unidades específicas de ingeniería sísmica
            'g_accel': 9.81,  # Aceleración gravitacional (m/s²)
            
            # Sin unidad (adimensional)
            'nu': 1,
            'adim': 1,
            
            # Factores de conversión específicos
            'kip': 4.4482 * self.units.kN if hasattr(self.units, 'kN') else 4448.2,
        }
        
        extended_dict.update(derived_units)
        
        # Agregar unidades adicionales si se proporcionan
        if additional_units:
            extended_dict.update(additional_units)
        
        return extended_dict
    
    def format_variable_with_unit(self, value: float, unit: str, 
                                decimals: int = None) -> str:
        """
        Formatea una variable con su unidad
        
        Parameters
        ----------
        value : float
            Valor de la variable
        unit : str
            Unidad de la variable
        decimals : int, optional
            Número de decimales
            
        Returns
        -------
        str
            Valor formateado
        """
        dec = decimals if decimals is not None else self.default_decimals
        
        if unit == 'nn':
            return str(value)
        elif unit in self.unit_dict:
            converted_value = value / self.unit_dict[unit]
            return f'{converted_value:.{dec}f}'
        else:
            try:
                return f'{float(value):.{dec}f}'
            except (ValueError, TypeError):
                return str(value)
    
    def validate_variables(self, variables: Dict[str, Any], 
                         content: str) -> Dict[str, List[str]]:
        """
        Valida variables antes del procesamiento
        
        Parameters
        ----------
        variables : Dict[str, Any]
            Variables a validar
        content : str
            Contenido del template
            
        Returns
        -------
        Dict[str, List[str]]
            Reporte de validación con errores y advertencias
        """
        validation_report = {
            'errors': [],
            'warnings': [],
            'missing_variables': [],
            'unused_variables': []
        }
        
        # Encontrar variables requeridas en el contenido
        found_variables = self.find_variables_in_content(content)
        required_vars = {var['name'] for var in found_variables}
        provided_vars = set(variables.keys())
        
        # Variables faltantes
        missing = required_vars - provided_vars
        validation_report['missing_variables'] = list(missing)
        
        # Variables no utilizadas
        unused = provided_vars - required_vars
        validation_report['unused_variables'] = list(unused)
        
        # Validar tipos de datos y valores
        for var_info in found_variables:
            var_name = var_info['name']
            unit = var_info['unit']
            
            if var_name not in variables:
                continue
            
            value = variables[var_name]
            
            # Validar valores numéricos
            if unit != 'nn':
                try:
                    float(value)
                except (ValueError, TypeError):
                    validation_report['errors'].append(
                        f"Variable '{var_name}' debe ser numérica (valor: {value})"
                    )
            
            # Validar unidades conocidas
            if unit not in ['nn'] and unit not in self.unit_dict:
                validation_report['warnings'].append(
                    f"Unidad '{unit}' no reconocida para variable '{var_name}'"
                )
        
        return validation_report
    
    def export_variables_to_csv(self, variables: Dict[str, Any], 
                              file_path: Union[str, Path]) -> None:
        """
        Exporta variables a archivo CSV
        
        Parameters
        ----------
        variables : Dict[str, Any]
            Variables a exportar
        file_path : str or Path
            Ruta del archivo CSV
        """
        with open(file_path, 'w', encoding='utf-8', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Variable', 'Valor', 'Tipo', 'Descripcion'])
            
            for name, value in variables.items():
                var_type = type(value).__name__
                description = (self.variable_registry.get(name, {})
                             .get('description', ''))
                
                writer.writerow([name, str(value), var_type, description])
    
    def import_variables_from_csv(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Importa variables desde archivo CSV
        
        Parameters
        ----------
        file_path : str or Path
            Ruta del archivo CSV
            
        Returns
        -------
        Dict[str, Any]
            Variables importadas
        """
        variables = {}
        
        try:
            with open(file_path, encoding='utf-8', newline='') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    name = row['Variable']
                    value_str = row['Valor']
                    var_type = row.get('Tipo', 'str')
                    
                    # Convertir valor según el tipo
                    try:
                        if var_type == 'int':
                            value = int(value_str)
                        elif var_type == 'float':
                            value = float(value_str)
                        elif var_type == 'bool':
                            value = value_str.lower() in ('true', '1', 'yes')
                        else:
                            value = value_str
                        
                        variables[name] = value
                        
                        # Registrar variable si hay descripción
                        if 'Descripcion' in row and row['Descripcion']:
                            self.register_variable(name, value, 
                                                 description=row['Descripcion'])
                    
                    except ValueError:
                        # Si falla la conversión, usar como string
                        variables[name] = value_str
                        
        except FileNotFoundError:
            pass
        
        return variables
    
    def create_variable_summary(self, variables: Dict[str, Any], 
                              content: str) -> str:
        """
        Crea un resumen de las variables procesadas
        
        Parameters
        ----------
        variables : Dict[str, Any]
            Variables procesadas
        content : str
            Contenido del template
            
        Returns
        -------
        str
            Resumen en formato texto
        """
        summary = ["=== RESUMEN DE VARIABLES ===\n"]
        
        found_variables = self.find_variables_in_content(content)
        
        summary.append(f"Variables encontradas en template: {len(found_variables)}")
        summary.append(f"Variables proporcionadas: {len(variables)}")
        summary.append(f"Sistema de unidades: {self.units.system.value}\n")
        
        # Listar variables procesadas
        summary.append("VARIABLES PROCESADAS:")
        for var_info in found_variables:
            name = var_info['name']
            if name in variables:
                value = variables[name]
                unit = var_info['unit']
                decimals = var_info['decimals']
                
                formatted_value = self.format_variable_with_unit(
                    value, unit, decimals
                )
                
                summary.append(f"  {name}: {value} → {formatted_value} ({unit})")
        
        # Conversiones realizadas
        if self.conversion_history:
            summary.append(f"\nCONVERSIONES REALIZADAS ({len(self.conversion_history)}):")
            for conv in self.conversion_history[-10:]:  # Últimas 10
                summary.append(
                    f"  {conv['variable']}: {conv['original_value']} → "
                    f"{conv['converted_value']:.3f} (factor: {conv['factor']:.6f})"
                )
        
        return "\n".join(summary)
    
    def clear_conversion_history(self) -> None:
        """Limpia el historial de conversiones"""
        self.conversion_history.clear()


# Funciones de utilidad para compatibilidad con código existente
def save_variables_legacy(var_dict: Dict[str, Any], content: str, 
                        template_path: Optional[str] = None,
                        unit_system: str = 'SI') -> str:
    """
    Función de compatibilidad para save_variables del código existente
    
    Parameters
    ----------
    var_dict : Dict[str, Any]
        Diccionario de variables
    content : str, optional
        Contenido del template
    template_path : str, optional
        Ruta del template si content es None
    unit_system : str
        Sistema de unidades
        
    Returns
    -------
    str
        Contenido con variables procesadas
    """
    if content is None and template_path:
        with open(template_path, 'r', encoding='utf-8') as file:
            content = file.read()
    elif content is None:
        raise ValueError("Debe proporcionar 'content' o 'template_path'")
    
    processor = VariableProcessor(unit_system)
    return processor.process_variables(var_dict, content)


def create_bolivia_unit_dict() -> Dict[str, float]:
    """
    Crea diccionario de unidades para Bolivia (compatibilidad)
    
    Returns
    -------
    Dict[str, float]
        Diccionario de unidades para Bolivia
    """
    units = Units(UnitSystem.SI)
    return {
        'mm': units.mm,
        'm': units.m,
        'cm': units.cm,
        'pies': units.ft,
        'pulg': units.inch,
        'tonf': units.tonf,
        'kN': units.kN,
        'kip': 4.4482 * units.kN,
        'nn': 1  # Sin unidad
    }


def create_peru_unit_dict() -> Dict[str, float]:
    """
    Crea diccionario de unidades para Perú (compatibilidad)
    
    Returns
    -------
    Dict[str, float]
        Diccionario de unidades para Perú
    """
    units = Units(UnitSystem.SI)
    return {
        'mm': units.mm,
        'm': units.m,
        'cm': units.cm,
        'pies': units.ft,
        'pulg': units.inch,
        'tonf': units.tonf,
        'kN': units.kN,
        'kgf': units.kgf,
        'kip': 4.4482 * units.kN,
        'nn': 1  # Sin unidad
    }


def create_variable_processor_for_country(country_code: str) -> VariableProcessor:
    """
    Crea un procesador de variables configurado para un país específico
    
    Parameters
    ----------
    country_code : str
        Código del país ('PE', 'BO', 'US', etc.)
        
    Returns
    -------
    VariableProcessor
        Procesador configurado
    """
    # Mapeo de países a sistemas de unidades
    country_systems = {
        'PE': 'SI',   # Perú
        'BO': 'SI',   # Bolivia  
        'EC': 'SI',   # Ecuador
        'CO': 'SI',   # Colombia
        'US': 'FPS',  # Estados Unidos
        'UK': 'FPS',  # Reino Unido
    }
    
    unit_system = country_systems.get(country_code, 'SI')
    processor = VariableProcessor(unit_system)
    
    # Configuraciones específicas por país
    if country_code in ['PE', 'BO']:
        # Registrar variables típicas para normativas sudamericanas
        processor.register_variable('Z', 0.0, 'adim', 'Factor de zona sísmica')
        processor.register_variable('U', 0.0, 'adim', 'Factor de uso')
        processor.register_variable('S', 0.0, 'adim', 'Factor de suelo')
        processor.register_variable('R', 0.0, 'adim', 'Factor de reducción')
    
    return processor


# Clase para manejo de templates específicos
class TemplateVariableExtractor:
    """Extractor especializado de variables desde templates"""
    
    def __init__(self):
        self.variable_pattern = re.compile(
            r'@([a-zA-Z_][a-zA-Z0-9_\\]*)\.(\d)([a-zA-Z0-9_]+)'
        )
    
    def extract_from_file(self, file_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """
        Extrae variables de un archivo template
        
        Parameters
        ----------
        file_path : str or Path
            Ruta del archivo template
            
        Returns
        -------
        List[Dict[str, Any]]
            Lista de variables encontradas
        """
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        matches = self.variable_pattern.findall(content)
        
        variables = []
        for match in matches:
            variables.append({
                'name': match[0],
                'decimals': int(match[1]),
                'unit': match[2],
                'pattern': f'@{match[0]}.{match[1]}{match[2]}'
            })
        
        return variables
    
    def create_variable_template(self, variables: List[Dict[str, Any]], 
                               output_path: Union[str, Path]) -> None:
        """
        Crea un template CSV con las variables encontradas
        
        Parameters
        ----------
        variables : List[Dict[str, Any]]
            Variables a incluir en el template
        output_path : str or Path
            Ruta del archivo CSV de salida
        """
        with open(output_path, 'w', encoding='utf-8', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Variable', 'Valor', 'Tipo', 'Unidad', 'Descripcion'])
            
            for var in variables:
                writer.writerow([
                    var['name'],
                    '',  # Valor vacío para llenar
                    'float' if var['unit'] != 'nn' else 'str',
                    var['unit'],
                    ''  # Descripción vacía para llenar
                ])