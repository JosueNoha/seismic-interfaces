"""
Módulo de validadores centralizados para interfaces sísmicas
============================================================

Este módulo centraliza todas las funciones de validación utilizadas
en el proyecto de interfaces sísmicas, evitando duplicación de código
entre las diferentes normativas.

Tipos de validadores incluidos:
- Validadores de parámetros sísmicos básicos
- Validadores de estructura de datos
- Validadores de ubicación geográfica
- Validadores de modelos sísmicos
- Validadores de consistencia de datos
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import asdict
import logging
from datetime import datetime

# Configurar logger
logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Excepción personalizada para errores de validación"""
    def __init__(self, message: str, field: str = None, value: Any = None):
        self.message = message
        self.field = field
        self.value = value
        super().__init__(self.message)


class ValidationReport:
    """Clase para reportes de validación estructurados"""
    
    def __init__(self):
        self.is_valid = True
        self.errors = []
        self.warnings = []
        self.field_validations = {}
        
    def add_error(self, message: str, field: str = None, value: Any = None):
        """Añade un error al reporte"""
        self.is_valid = False
        error_info = {'message': message, 'field': field, 'value': value}
        self.errors.append(error_info)
        
        if field:
            self.field_validations[field] = {'status': 'error', 'message': message}
    
    def add_warning(self, message: str, field: str = None, value: Any = None):
        """Añade una advertencia al reporte"""
        warning_info = {'message': message, 'field': field, 'value': value}
        self.warnings.append(warning_info)
        
        if field and field not in self.field_validations:
            self.field_validations[field] = {'status': 'warning', 'message': message}
    
    def add_success(self, field: str, message: str = "Válido"):
        """Marca un campo como válido"""
        self.field_validations[field] = {'status': 'valid', 'message': message}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte el reporte a diccionario"""
        return {
            'is_valid': self.is_valid,
            'errors': self.errors,
            'warnings': self.warnings,
            'field_validations': self.field_validations,
            'summary': {
                'total_errors': len(self.errors),
                'total_warnings': len(self.warnings),
                'total_fields': len(self.field_validations)
            }
        }


# ============================================================================
# VALIDADORES DE PARÁMETROS SÍSMICOS BÁSICOS
# ============================================================================

def validate_reduction_factors(Rx: float, Ry: float) -> ValidationReport:
    """
    Valida factores de reducción sísmica
    
    Parameters
    ----------
    Rx : float
        Factor de reducción en dirección X
    Ry : float
        Factor de reducción en dirección Y
        
    Returns
    -------
    ValidationReport
        Reporte de validación
    """
    report = ValidationReport()
    
    # Validar Rx
    if Rx <= 0:
        report.add_error("Factor de reducción Rx debe ser mayor a 0", "Rx", Rx)
    elif Rx > 8.0:
        report.add_warning("Factor Rx muy alto (>8.0), verificar", "Rx", Rx)
    else:
        report.add_success("Rx", f"Factor Rx válido: {Rx}")
    
    # Validar Ry
    if Ry <= 0:
        report.add_error("Factor de reducción Ry debe ser mayor a 0", "Ry", Ry)
    elif Ry > 8.0:
        report.add_warning("Factor Ry muy alto (>8.0), verificar", "Ry", Ry)
    else:
        report.add_success("Ry", f"Factor Ry válido: {Ry}")
    
    return report


def validate_irregularity_factors(Ia: float, Ip: float) -> ValidationReport:
    """
    Valida factores de irregularidad
    
    Parameters
    ----------
    Ia : float
        Factor de irregularidad en altura
    Ip : float
        Factor de irregularidad en planta
        
    Returns
    -------
    ValidationReport
        Reporte de validación
    """
    report = ValidationReport()
    
    # Validar Ia
    if not (0.5 <= Ia <= 1.0):
        report.add_error("Factor Ia debe estar entre 0.5 y 1.0", "Ia", Ia)
    else:
        if Ia < 1.0:
            report.add_warning("Estructura irregular en altura", "Ia", Ia)
        report.add_success("Ia", f"Factor Ia válido: {Ia}")
    
    # Validar Ip
    if not (0.5 <= Ip <= 1.0):
        report.add_error("Factor Ip debe estar entre 0.5 y 1.0", "Ip", Ip)
    else:
        if Ip < 1.0:
            report.add_warning("Estructura irregular en planta", "Ip", Ip)
        report.add_success("Ip", f"Factor Ip válido: {Ip}")
    
    return report


def validate_fundamental_periods(Tx: float, Ty: float) -> ValidationReport:
    """
    Valida periodos fundamentales
    
    Parameters
    ----------
    Tx : float
        Periodo fundamental en dirección X
    Ty : float
        Periodo fundamental en dirección Y
        
    Returns
    -------
    ValidationReport
        Reporte de validación
    """
    report = ValidationReport()
    
    # Validar Tx
    if Tx <= 0:
        report.add_error("Periodo Tx debe ser mayor a 0", "Tx", Tx)
    elif Tx > 4.0:
        report.add_warning("Periodo Tx muy alto (>4.0s), verificar", "Tx", Tx)
    else:
        report.add_success("Tx", f"Periodo Tx válido: {Tx}s")
    
    # Validar Ty
    if Ty <= 0:
        report.add_error("Periodo Ty debe ser mayor a 0", "Ty", Ty)
    elif Ty > 4.0:
        report.add_warning("Periodo Ty muy alto (>4.0s), verificar", "Ty", Ty)
    else:
        report.add_success("Ty", f"Periodo Ty válido: {Ty}s")
    
    # Validar relación entre periodos
    if Tx > 0 and Ty > 0:
        ratio = max(Tx, Ty) / min(Tx, Ty)
        if ratio > 3.0:
            report.add_warning(f"Gran diferencia entre periodos (ratio: {ratio:.2f})", "periods_ratio", ratio)
    
    return report


def validate_mass_participation(MP_x: float, MP_y: float, min_participation: float = 0.9) -> ValidationReport:
    """
    Valida participación de masa modal
    
    Parameters
    ----------
    MP_x : float
        Participación de masa en dirección X
    MP_y : float
        Participación de masa en dirección Y
    min_participation : float, optional
        Participación mínima requerida (default: 0.9)
        
    Returns
    -------
    ValidationReport
        Reporte de validación
    """
    report = ValidationReport()
    
    # Validar MP_x
    if MP_x < min_participation:
        report.add_error(f"Participación de masa en X ({MP_x:.1%}) menor al {min_participation:.0%}", "MP_x", MP_x)
    elif MP_x > 1.0:
        report.add_error("Participación de masa en X no puede ser mayor al 100%", "MP_x", MP_x)
    else:
        report.add_success("MP_x", f"Participación en X válida: {MP_x:.1%}")
    
    # Validar MP_y
    if MP_y < min_participation:
        report.add_error(f"Participación de masa en Y ({MP_y:.1%}) menor al {min_participation:.0%}", "MP_y", MP_y)
    elif MP_y > 1.0:
        report.add_error("Participación de masa en Y no puede ser mayor al 100%", "MP_y", MP_y)
    else:
        report.add_success("MP_y", f"Participación en Y válida: {MP_y:.1%}")
    
    return report


def validate_drift_limits(max_drift_x: float, max_drift_y: float, 
                         normative_limit: float = 0.007) -> ValidationReport:
    """
    Valida límites de deriva
    
    Parameters
    ----------
    max_drift_x : float
        Deriva máxima en dirección X
    max_drift_y : float
        Deriva máxima en dirección Y
    normative_limit : float, optional
        Límite normativo (default: 0.007)
        
    Returns
    -------
    ValidationReport
        Reporte de validación
    """
    report = ValidationReport()
    
    # Validar deriva en X
    if max_drift_x <= 0:
        report.add_error("Deriva máxima en X debe ser mayor a 0", "max_drift_x", max_drift_x)
    elif max_drift_x > normative_limit:
        report.add_error(f"Deriva en X ({max_drift_x:.4f}) excede límite normativo ({normative_limit:.4f})", 
                        "max_drift_x", max_drift_x)
    elif max_drift_x > normative_limit * 0.8:
        report.add_warning(f"Deriva en X cerca del límite ({max_drift_x:.4f})", "max_drift_x", max_drift_x)
    else:
        report.add_success("max_drift_x", f"Deriva X válida: {max_drift_x:.4f}")
    
    # Validar deriva en Y
    if max_drift_y <= 0:
        report.add_error("Deriva máxima en Y debe ser mayor a 0", "max_drift_y", max_drift_y)
    elif max_drift_y > normative_limit:
        report.add_error(f"Deriva en Y ({max_drift_y:.4f}) excede límite normativo ({normative_limit:.4f})", 
                        "max_drift_y", max_drift_y)
    elif max_drift_y > normative_limit * 0.8:
        report.add_warning(f"Deriva en Y cerca del límite ({max_drift_y:.4f})", "max_drift_y", max_drift_y)
    else:
        report.add_success("max_drift_y", f"Deriva Y válida: {max_drift_y:.4f}")
    
    return report


# ============================================================================
# VALIDADORES DE UBICACIÓN GEOGRÁFICA
# ============================================================================

def validate_location(location_data: Dict[str, str], country_code: str = 'PE') -> ValidationReport:
    """
    Valida datos de ubicación geográfica
    
    Parameters
    ----------
    location_data : dict
        Diccionario con datos de ubicación
    country_code : str, optional
        Código del país (default: 'PE')
        
    Returns
    -------
    ValidationReport
        Reporte de validación
    """
    report = ValidationReport()
    
    # Definir jerarquías por país
    country_hierarchies = {
        'PE': ['DEPARTAMENTO', 'PROVINCIA', 'DISTRITO'],
        'BO': ['DEPARTAMENTO', 'PROVINCIA', 'MUNICIPIO'],
        'EC': ['PROVINCIA', 'CANTON', 'PARROQUIA'],
        'CO': ['DEPARTAMENTO', 'MUNICIPIO']
    }
    
    required_fields = country_hierarchies.get(country_code, ['REGION', 'SUBREGION'])
    
    # Validar campos requeridos
    for field in required_fields:
        if field not in location_data or not location_data[field]:
            report.add_error(f"Campo requerido faltante: {field}", field, None)
        else:
            # Validar formato (solo letras, espacios y algunos caracteres especiales)
            value = location_data[field].strip().upper()
            if not value:
                report.add_error(f"Campo {field} no puede estar vacío", field, value)
            elif not all(c.isalpha() or c.isspace() or c in "ÑÁÉÍÓÚÜñáéíóúü.-'" for c in value):
                report.add_warning(f"Campo {field} contiene caracteres especiales", field, value)
            else:
                report.add_success(field, f"{field}: {value}")
    
    return report


def validate_country_data_structure(data: Union[pd.DataFrame, Dict], country_code: str) -> Tuple[bool, List[str]]:
    """
    Valida estructura de datos por país
    
    Parameters
    ----------
    data : DataFrame or dict
        Datos a validar
    country_code : str
        Código del país
        
    Returns
    -------
    Tuple[bool, List[str]]
        (es_válido, lista_errores)
    """
    errors = []
    
    # Convertir a DataFrame si es necesario
    if isinstance(data, dict):
        try:
            df = pd.DataFrame(data)
        except Exception as e:
            errors.append(f"Error al convertir datos a DataFrame: {e}")
            return False, errors
    else:
        df = data
    
    # Definir campos requeridos por país
    required_fields_by_country = {
        'PE': ['DEPARTAMENTO', 'PROVINCIA', 'DISTRITO'],
        'BO': ['DEPARTAMENTO', 'PROVINCIA', 'MUNICIPIO'],
        'EC': ['PROVINCIA', 'CANTON', 'PARROQUIA'],
        'CO': ['DEPARTAMENTO', 'MUNICIPIO']
    }
    
    required_fields = required_fields_by_country.get(country_code, [])
    
    # Verificar campos requeridos
    missing_fields = [field for field in required_fields if field not in df.columns]
    if missing_fields:
        errors.append(f"Campos faltantes para {country_code}: {missing_fields}")
    
    # Verificar datos no vacíos
    for field in required_fields:
        if field in df.columns:
            empty_count = df[field].isna().sum() + (df[field] == '').sum()
            if empty_count > 0:
                errors.append(f"Campo {field} tiene {empty_count} valores vacíos")
    
    is_valid = len(errors) == 0
    return is_valid, errors


# ============================================================================
# VALIDADORES DE MODELOS SÍSMICOS
# ============================================================================

def validate_seismic_data_model(seismic_data: Any) -> ValidationReport:
    """
    Valida modelo completo de datos sísmicos
    
    Parameters
    ----------
    seismic_data : Any
        Modelo de datos sísmicos a validar
        
    Returns
    -------
    ValidationReport
        Reporte de validación completa
    """
    report = ValidationReport()
    
    try:
        # Validar información básica del proyecto
        if hasattr(seismic_data, 'project'):
            project = seismic_data.project
            if not hasattr(project, 'name') or not project.name:
                report.add_error("Nombre del proyecto requerido", "project_name")
            if not hasattr(project, 'location') or not project.location:
                report.add_warning("Ubicación del proyecto no especificada", "project_location")
        else:
            report.add_error("Información del proyecto faltante", "project")
        
        # Validar factores de reducción
        if hasattr(seismic_data, 'Rx') and hasattr(seismic_data, 'Ry'):
            reduction_report = validate_reduction_factors(seismic_data.Rx, seismic_data.Ry)
            _merge_reports(report, reduction_report)
        else:
            report.add_error("Factores de reducción faltantes", "reduction_factors")
        
        # Validar factores de irregularidad
        if hasattr(seismic_data, 'Ia') and hasattr(seismic_data, 'Ip'):
            irregularity_report = validate_irregularity_factors(seismic_data.Ia, seismic_data.Ip)
            _merge_reports(report, irregularity_report)
        else:
            report.add_error("Factores de irregularidad faltantes", "irregularity_factors")
        
        # Validar periodos fundamentales
        if hasattr(seismic_data, 'Tx') and hasattr(seismic_data, 'Ty'):
            periods_report = validate_fundamental_periods(seismic_data.Tx, seismic_data.Ty)
            _merge_reports(report, periods_report)
        else:
            report.add_error("Periodos fundamentales faltantes", "periods")
        
        # Validar participación de masa
        if hasattr(seismic_data, 'MP_x') and hasattr(seismic_data, 'MP_y'):
            mass_report = validate_mass_participation(seismic_data.MP_x, seismic_data.MP_y)
            _merge_reports(report, mass_report)
        else:
            report.add_error("Participación de masa faltante", "mass_participation")
        
        # Validar límites de deriva
        if hasattr(seismic_data, 'max_drift_x') and hasattr(seismic_data, 'max_drift_y'):
            drift_report = validate_drift_limits(seismic_data.max_drift_x, seismic_data.max_drift_y)
            _merge_reports(report, drift_report)
        else:
            report.add_error("Límites de deriva faltantes", "drift_limits")
        
        # Validar cargas sísmicas
        if hasattr(seismic_data, 'loads') and hasattr(seismic_data.loads, 'seism_loads'):
            loads = seismic_data.loads.seism_loads
            if not loads:
                report.add_error("Cargas sísmicas no definidas", "seismic_loads")
            else:
                # Verificar casos en ambas direcciones
                x_cases = [key for key in loads.keys() if 'X' in key.upper()]
                y_cases = [key for key in loads.keys() if 'Y' in key.upper()]
                
                if not x_cases:
                    report.add_error("No se encontraron casos sísmicos en dirección X", "seismic_loads_x")
                if not y_cases:
                    report.add_error("No se encontraron casos sísmicos en dirección Y", "seismic_loads_y")
                
                if x_cases and y_cases:
                    report.add_success("seismic_loads", f"Cargas sísmicas válidas: {len(loads)} casos")
        else:
            report.add_error("Definición de cargas faltante", "loads")
        
    except Exception as e:
        report.add_error(f"Error durante validación del modelo: {str(e)}", "model_validation")
        logger.error(f"Error en validación del modelo sísmico: {e}")
    
    return report


# ============================================================================
# VALIDADORES DE CONSISTENCIA DE DATOS
# ============================================================================

def validate_location_consistency(location_model: Any) -> ValidationReport:
    """
    Valida consistencia en modelo de ubicaciones
    
    Parameters
    ----------
    location_model : Any
        Modelo de ubicaciones a validar
        
    Returns
    -------
    ValidationReport
        Reporte de validación de consistencia
    """
    report = ValidationReport()
    
    try:
        if not hasattr(location_model, 'data') or location_model.data is None:
            report.add_error("Modelo de ubicación sin datos", "location_data")
            return report
        
        data = location_model.data
        
        # Verificar estructura jerárquica
        country_code = getattr(location_model, 'country_code', 'PE')
        is_valid, errors = validate_country_data_structure(data, country_code)
        
        for error in errors:
            report.add_error(error, "data_structure")
        
        if is_valid:
            report.add_success("data_structure", "Estructura de datos válida")
        
        # Verificar duplicados
        if hasattr(location_model, 'country_code'):
            hierarchy = _get_country_hierarchy(location_model.country_code)
            duplicates = data.duplicated(subset=hierarchy)
            duplicate_count = duplicates.sum()
            
            if duplicate_count > 0:
                report.add_warning(f"Se encontraron {duplicate_count} ubicaciones duplicadas", "duplicates")
            else:
                report.add_success("duplicates", "Sin ubicaciones duplicadas")
        
        # Verificar completitud de datos
        total_records = len(data)
        if total_records == 0:
            report.add_error("No hay registros en el modelo de ubicación", "record_count")
        else:
            report.add_success("record_count", f"Modelo con {total_records} ubicaciones")
        
    except Exception as e:
        report.add_error(f"Error en validación de consistencia: {str(e)}", "consistency_validation")
        logger.error(f"Error en validación de consistencia: {e}")
    
    return report


def validate_template_variables(variables: Dict[str, Any], template_content: str) -> ValidationReport:
    """
    Valida variables contra contenido de template
    
    Parameters
    ----------
    variables : dict
        Diccionario de variables
    template_content : str
        Contenido del template LaTeX
        
    Returns
    -------
    ValidationReport
        Reporte de validación de variables
    """
    report = ValidationReport()
    
    try:
        import re
        
        # Extraer variables requeridas del template (formato @variable.format)
        variable_pattern = r'@([a-zA-Z_][a-zA-Z0-9_]*)\.[0-9]*[a-z]*'
        required_vars = set(re.findall(variable_pattern, template_content))
        
        # Verificar variables requeridas
        missing_vars = []
        available_vars = []
        
        for var in required_vars:
            if var not in variables:
                missing_vars.append(var)
                report.add_error(f"Variable requerida faltante: @{var}", var)
            else:
                available_vars.append(var)
                report.add_success(var, f"Variable disponible: @{var}")
        
        # Verificar variables no utilizadas
        unused_vars = set(variables.keys()) - required_vars
        for var in unused_vars:
            report.add_warning(f"Variable definida pero no utilizada: {var}", var)
        
        # Resumen
        total_required = len(required_vars)
        total_available = len(available_vars)
        
        if total_required == total_available:
            report.add_success("template_variables", 
                             f"Todas las variables requeridas están disponibles ({total_available}/{total_required})")
        else:
            report.add_error(f"Variables faltantes: {len(missing_vars)}/{total_required}", "template_variables")
    
    except Exception as e:
        report.add_error(f"Error en validación de variables: {str(e)}", "variable_validation")
        logger.error(f"Error en validación de variables del template: {e}")
    
    return report


# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def _merge_reports(main_report: ValidationReport, sub_report: ValidationReport):
    """Fusiona reportes de validación"""
    main_report.errors.extend(sub_report.errors)
    main_report.warnings.extend(sub_report.warnings)
    main_report.field_validations.update(sub_report.field_validations)
    
    if not sub_report.is_valid:
        main_report.is_valid = False


def _get_country_hierarchy(country_code: str) -> List[str]:
    """Obtiene jerarquía de ubicación por país"""
    hierarchies = {
        'PE': ['DEPARTAMENTO', 'PROVINCIA', 'DISTRITO'],
        'BO': ['DEPARTAMENTO', 'PROVINCIA', 'MUNICIPIO'],
        'EC': ['PROVINCIA', 'CANTON', 'PARROQUIA'],
        'CO': ['DEPARTAMENTO', 'MUNICIPIO']
    }
    return hierarchies.get(country_code, ['REGION', 'SUBREGION'])


# ============================================================================
# FUNCIONES DE CONVENIENCIA
# ============================================================================

def validate_all_seismic_parameters(seismic_data: Any, normative_limit: float = 0.007) -> ValidationReport:
    """
    Función de conveniencia para validar todos los parámetros sísmicos básicos
    
    Parameters
    ----------
    seismic_data : Any
        Datos sísmicos a validar
    normative_limit : float, optional
        Límite normativo de deriva (default: 0.007)
        
    Returns
    -------
    ValidationReport
        Reporte completo de validación
    """
    main_report = ValidationReport()
    
    # Validar cada grupo de parámetros
    validation_functions = [
        ('reduction_factors', lambda: validate_reduction_factors(seismic_data.Rx, seismic_data.Ry)),
        ('irregularity_factors', lambda: validate_irregularity_factors(seismic_data.Ia, seismic_data.Ip)),
        ('periods', lambda: validate_fundamental_periods(seismic_data.Tx, seismic_data.Ty)),
        ('mass_participation', lambda: validate_mass_participation(seismic_data.MP_x, seismic_data.MP_y)),
        ('drift_limits', lambda: validate_drift_limits(seismic_data.max_drift_x, seismic_data.max_drift_y, normative_limit))
    ]
    
    for validation_name, validation_func in validation_functions:
        try:
            sub_report = validation_func()
            _merge_reports(main_report, sub_report)
        except AttributeError as e:
            main_report.add_error(f"Parámetro faltante para {validation_name}: {str(e)}", validation_name)
        except Exception as e:
            main_report.add_error(f"Error en validación {validation_name}: {str(e)}", validation_name)
    
    return main_report


def quick_validate(value: float, min_val: float = None, max_val: float = None, 
                  field_name: str = "value") -> Tuple[bool, str]:
    """
    Validación rápida de un valor numérico
    
    Parameters
    ----------
    value : float
        Valor a validar
    min_val : float, optional
        Valor mínimo permitido
    max_val : float, optional
        Valor máximo permitido
    field_name : str, optional
        Nombre del campo para mensajes de error
        
    Returns
    -------
    Tuple[bool, str]
        (es_válido, mensaje)
    """
    if not isinstance(value, (int, float)):
        return False, f"{field_name} debe ser numérico"
    
    if np.isnan(value) or np.isinf(value):
        return False, f"{field_name} debe ser un número válido"
    
    if min_val is not None and value < min_val:
        return False, f"{field_name} debe ser >= {min_val}"
    
    if max_val is not None and value > max_val:
        return False, f"{field_name} debe ser <= {max_val}"
    
    return True, f"{field_name} válido: {value}"


# ============================================================================
# EXPORTACIÓN DE FUNCIONES PRINCIPALES
# ============================================================================

__all__ = [
    'ValidationError',
    'ValidationReport',
    'validate_reduction_factors',
    'validate_irregularity_factors', 
    'validate_fundamental_periods',
    'validate_mass_participation',
    'validate_drift_limits',
    'validate_location',
    'validate_country_data_structure',
    'validate_seismic_data_model',
    'validate_location_consistency',
    'validate_template_variables',
    'validate_all_seismic_parameters',
    'quick_validate'
]