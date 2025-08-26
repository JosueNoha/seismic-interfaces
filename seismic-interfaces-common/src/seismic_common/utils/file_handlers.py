"""
Módulo de manejadores de archivos centralizados para interfaces sísmicas
========================================================================

Este módulo centraliza todas las funciones de manejo de archivos utilizadas
en el proyecto de interfaces sísmicas, evitando duplicación de código
entre las diferentes normativas.

Tipos de manejadores incluidos:
- Importación/exportación Excel (.xlsx, .xls)
- Importación/exportación CSV
- Manejo de archivos JSON
- Procesamiento de archivos LaTeX
- Importación de archivos de software (SAP2000, ETABS, etc.)
- Compresión y archivado de proyectos
- Validación y backup de archivos
"""

import pandas as pd
import numpy as np
import json
import pickle
import csv
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from pathlib import Path
from datetime import datetime
import logging
import shutil
import zipfile
import tempfile
from dataclasses import dataclass, asdict
from enum import Enum
import warnings

# Configurar logger
logger = logging.getLogger(__name__)


class FileFormat(Enum):
    """Enumeración de formatos de archivo soportados"""
    EXCEL = "excel"
    CSV = "csv"
    JSON = "json"
    PICKLE = "pickle"
    LATEX = "latex"
    TXT = "txt"
    XML = "xml"
    SAP2000 = "sap2000"
    ETABS = "etabs"
    ZIP = "zip"


@dataclass
class FileInfo:
    """Información de archivo con metadatos"""
    filepath: Path
    format: FileFormat
    size_bytes: int
    modified_date: datetime
    encoding: str = 'utf-8'
    sheets: List[str] = None  # Para archivos Excel
    columns: List[str] = None  # Para archivos tabulares
    
    def __post_init__(self):
        if isinstance(self.filepath, str):
            self.filepath = Path(self.filepath)
        
        if self.sheets is None:
            self.sheets = []
        
        if self.columns is None:
            self.columns = []


class FileOperationResult:
    """Resultado de operaciones de archivo"""
    
    def __init__(self, success: bool = True, message: str = "", 
                 data: Any = None, errors: List[str] = None):
        self.success = success
        self.message = message
        self.data = data
        self.errors = errors or []
        self.timestamp = datetime.now()
    
    def add_error(self, error: str):
        """Añade un error al resultado"""
        self.errors.append(error)
        self.success = False
    
    def add_warning(self, warning: str):
        """Añade una advertencia al resultado"""
        if not hasattr(self, 'warnings'):
            self.warnings = []
        self.warnings.append(warning)


class ExcelHandler:
    """
    Manejador especializado para archivos Excel
    """
    
    def __init__(self, default_engine: str = 'openpyxl'):
        """
        Inicializa el manejador Excel
        
        Parameters
        ----------
        default_engine : str
            Motor por defecto ('openpyxl', 'xlsxwriter')
        """
        self.default_engine = default_engine
        self.supported_extensions = {'.xlsx', '.xls', '.xlsm'}
    
    def read_excel(self, filepath: Union[str, Path], 
                   sheet_name: Union[str, int, List[str]] = None,
                   **kwargs) -> FileOperationResult:
        """
        Lee archivo Excel con manejo robusto de errores
        
        Parameters
        ----------
        filepath : str or Path
            Ruta del archivo Excel
        sheet_name : str, int, list, optional
            Nombre(s) de hoja(s) a leer
        **kwargs
            Argumentos adicionales para pandas.read_excel()
            
        Returns
        -------
        FileOperationResult
            Resultado con datos o errores
        """
        filepath = Path(filepath)
        result = FileOperationResult()
        
        try:
            if not filepath.exists():
                result.add_error(f"Archivo no encontrado: {filepath}")
                return result
            
            if filepath.suffix.lower() not in self.supported_extensions:
                result.add_error(f"Extensión no soportada: {filepath.suffix}")
                return result
            
            # Leer archivo Excel
            data = pd.read_excel(
                filepath, 
                sheet_name=sheet_name, 
                engine=self.default_engine,
                **kwargs
            )
            
            result.data = data
            result.message = f"Archivo Excel leído exitosamente: {filepath.name}"
            
            # Obtener información de hojas si es necesario
            if isinstance(data, dict):  # Múltiples hojas
                result.message += f" ({len(data)} hojas)"
            
        except FileNotFoundError:
            result.add_error(f"Archivo no encontrado: {filepath}")
        except PermissionError:
            result.add_error(f"Sin permisos para leer: {filepath}")
        except pd.errors.EmptyDataError:
            result.add_error("El archivo Excel está vacío")
        except Exception as e:
            result.add_error(f"Error leyendo Excel: {str(e)}")
            logger.error(f"Error en read_excel: {e}")
        
        return result
    
    def write_excel(self, data: Union[pd.DataFrame, Dict[str, pd.DataFrame]], 
                    filepath: Union[str, Path],
                    **kwargs) -> FileOperationResult:
        """
        Escribe datos a archivo Excel
        
        Parameters
        ----------
        data : DataFrame or dict
            Datos a escribir (DataFrame o dict de DataFrames)
        filepath : str or Path
            Ruta del archivo de destino
        **kwargs
            Argumentos adicionales para pandas.to_excel()
            
        Returns
        -------
        FileOperationResult
            Resultado de la operación
        """
        filepath = Path(filepath)
        result = FileOperationResult()
        
        try:
            # Crear directorio si no existe
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            if isinstance(data, pd.DataFrame):
                # Escribir DataFrame único
                data.to_excel(
                    filepath, 
                    index=False, 
                    engine=self.default_engine,
                    **kwargs
                )
                result.message = f"DataFrame exportado a: {filepath.name}"
            
            elif isinstance(data, dict):
                # Escribir múltiples hojas
                with pd.ExcelWriter(filepath, engine=self.default_engine, **kwargs) as writer:
                    for sheet_name, df in data.items():
                        if isinstance(df, pd.DataFrame):
                            df.to_excel(writer, sheet_name=sheet_name, index=False)
                
                result.message = f"Libro Excel con {len(data)} hojas exportado a: {filepath.name}"
            
            else:
                result.add_error("Tipo de datos no soportado para Excel")
                return result
            
        except PermissionError:
            result.add_error(f"Sin permisos para escribir: {filepath}")
        except Exception as e:
            result.add_error(f"Error escribiendo Excel: {str(e)}")
            logger.error(f"Error en write_excel: {e}")
        
        return result
    
    def get_sheet_names(self, filepath: Union[str, Path]) -> List[str]:
        """
        Obtiene nombres de hojas de un archivo Excel
        
        Parameters
        ----------
        filepath : str or Path
            Ruta del archivo Excel
            
        Returns
        -------
        List[str]
            Lista de nombres de hojas
        """
        try:
            excel_file = pd.ExcelFile(filepath, engine=self.default_engine)
            return excel_file.sheet_names
        except Exception as e:
            logger.error(f"Error obteniendo hojas de {filepath}: {e}")
            return []


class CSVHandler:
    """
    Manejador especializado para archivos CSV
    """
    
    def __init__(self, default_encoding: str = 'utf-8'):
        """
        Inicializa el manejador CSV
        
        Parameters
        ----------
        default_encoding : str
            Codificación por defecto
        """
        self.default_encoding = default_encoding
        self.common_encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        self.common_separators = [',', ';', '\t', '|']
    
    def read_csv(self, filepath: Union[str, Path], 
                 encoding: str = None, separator: str = None,
                 **kwargs) -> FileOperationResult:
        """
        Lee archivo CSV con detección automática de parámetros
        
        Parameters
        ----------
        filepath : str or Path
            Ruta del archivo CSV
        encoding : str, optional
            Codificación específica
        separator : str, optional
            Separador específico
        **kwargs
            Argumentos adicionales para pandas.read_csv()
            
        Returns
        -------
        FileOperationResult
            Resultado con datos o errores
        """
        filepath = Path(filepath)
        result = FileOperationResult()
        
        try:
            if not filepath.exists():
                result.add_error(f"Archivo no encontrado: {filepath}")
                return result
            
            # Detectar parámetros si no se especifican
            if encoding is None:
                encoding = self._detect_encoding(filepath)
            
            if separator is None:
                separator = self._detect_separator(filepath, encoding)
            
            # Leer CSV
            data = pd.read_csv(
                filepath,
                encoding=encoding,
                sep=separator,
                **kwargs
            )
            
            result.data = data
            result.message = f"CSV leído exitosamente: {filepath.name} (encoding: {encoding}, sep: '{separator}')"
            
        except UnicodeDecodeError:
            result.add_error(f"Error de codificación. Probado: {encoding}")
        except pd.errors.EmptyDataError:
            result.add_error("El archivo CSV está vacío")
        except Exception as e:
            result.add_error(f"Error leyendo CSV: {str(e)}")
            logger.error(f"Error en read_csv: {e}")
        
        return result
    
    def write_csv(self, data: pd.DataFrame, filepath: Union[str, Path],
                  encoding: str = None, **kwargs) -> FileOperationResult:
        """
        Escribe DataFrame a archivo CSV
        
        Parameters
        ----------
        data : DataFrame
            Datos a escribir
        filepath : str or Path
            Ruta del archivo de destino
        encoding : str, optional
            Codificación a usar
        **kwargs
            Argumentos adicionales para pandas.to_csv()
            
        Returns
        -------
        FileOperationResult
            Resultado de la operación
        """
        filepath = Path(filepath)
        result = FileOperationResult()
        
        try:
            # Crear directorio si no existe
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            encoding = encoding or self.default_encoding
            
            data.to_csv(
                filepath,
                index=False,
                encoding=encoding,
                **kwargs
            )
            
            result.message = f"CSV exportado a: {filepath.name}"
            
        except PermissionError:
            result.add_error(f"Sin permisos para escribir: {filepath}")
        except Exception as e:
            result.add_error(f"Error escribiendo CSV: {str(e)}")
            logger.error(f"Error en write_csv: {e}")
        
        return result
    
    def _detect_encoding(self, filepath: Path) -> str:
        """Detecta codificación del archivo CSV"""
        for encoding in self.common_encodings:
            try:
                with open(filepath, 'r', encoding=encoding) as f:
                    f.read(1024)  # Leer muestra
                return encoding
            except UnicodeDecodeError:
                continue
        return self.default_encoding
    
    def _detect_separator(self, filepath: Path, encoding: str) -> str:
        """Detecta separador del archivo CSV"""
        try:
            with open(filepath, 'r', encoding=encoding) as f:
                sample = f.read(1024)
            
            # Contar ocurrencias de cada separador
            separator_counts = {}
            for sep in self.common_separators:
                separator_counts[sep] = sample.count(sep)
            
            # Retornar el más frecuente
            return max(separator_counts, key=separator_counts.get)
        
        except Exception:
            return ','  # Por defecto


class JSONHandler:
    """
    Manejador especializado para archivos JSON
    """
    
    def __init__(self, default_encoding: str = 'utf-8'):
        """
        Inicializa el manejador JSON
        
        Parameters
        ----------
        default_encoding : str
            Codificación por defecto
        """
        self.default_encoding = default_encoding
    
    def read_json(self, filepath: Union[str, Path], **kwargs) -> FileOperationResult:
        """
        Lee archivo JSON
        
        Parameters
        ----------
        filepath : str or Path
            Ruta del archivo JSON
        **kwargs
            Argumentos adicionales para json.load()
            
        Returns
        -------
        FileOperationResult
            Resultado con datos o errores
        """
        filepath = Path(filepath)
        result = FileOperationResult()
        
        try:
            if not filepath.exists():
                result.add_error(f"Archivo no encontrado: {filepath}")
                return result
            
            with open(filepath, 'r', encoding=self.default_encoding) as f:
                data = json.load(f, **kwargs)
            
            result.data = data
            result.message = f"JSON leído exitosamente: {filepath.name}"
            
        except json.JSONDecodeError as e:
            result.add_error(f"Error de formato JSON: {str(e)}")
        except UnicodeDecodeError:
            result.add_error(f"Error de codificación en: {filepath}")
        except Exception as e:
            result.add_error(f"Error leyendo JSON: {str(e)}")
            logger.error(f"Error en read_json: {e}")
        
        return result
    
    def write_json(self, data: Any, filepath: Union[str, Path], 
                   **kwargs) -> FileOperationResult:
        """
        Escribe datos a archivo JSON
        
        Parameters
        ----------
        data : Any
            Datos a escribir (debe ser serializable a JSON)
        filepath : str or Path
            Ruta del archivo de destino
        **kwargs
            Argumentos adicionales para json.dump()
            
        Returns
        -------
        FileOperationResult
            Resultado de la operación
        """
        filepath = Path(filepath)
        result = FileOperationResult()
        
        try:
            # Crear directorio si no existe
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            # Configurar parámetros por defecto
            default_kwargs = {
                'indent': 2,
                'ensure_ascii': False,
                'default': self._json_serializer
            }
            default_kwargs.update(kwargs)
            
            with open(filepath, 'w', encoding=self.default_encoding) as f:
                json.dump(data, f, **default_kwargs)
            
            result.message = f"JSON exportado a: {filepath.name}"
            
        except TypeError as e:
            result.add_error(f"Datos no serializables a JSON: {str(e)}")
        except PermissionError:
            result.add_error(f"Sin permisos para escribir: {filepath}")
        except Exception as e:
            result.add_error(f"Error escribiendo JSON: {str(e)}")
            logger.error(f"Error en write_json: {e}")
        
        return result
    
    def _json_serializer(self, obj):
        """Serializador personalizado para tipos no estándar"""
        if isinstance(obj, (datetime, pd.Timestamp)):
            return obj.isoformat()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif hasattr(obj, 'to_dict'):
            return obj.to_dict()
        else:
            raise TypeError(f"Objeto no serializable: {type(obj)}")


class LaTeXHandler:
    """
    Manejador especializado para archivos LaTeX
    """
    
    def __init__(self, template_dir: Union[str, Path] = None):
        """
        Inicializa el manejador LaTeX
        
        Parameters
        ----------
        template_dir : str or Path, optional
            Directorio de plantillas LaTeX
        """
        self.template_dir = Path(template_dir) if template_dir else None
    
    def read_template(self, template_name: str) -> FileOperationResult:
        """
        Lee plantilla LaTeX
        
        Parameters
        ----------
        template_name : str
            Nombre del archivo de plantilla
            
        Returns
        -------
        FileOperationResult
            Resultado con contenido de plantilla
        """
        result = FileOperationResult()
        
        try:
            if self.template_dir:
                template_path = self.template_dir / template_name
            else:
                template_path = Path(template_name)
            
            if not template_path.exists():
                result.add_error(f"Plantilla no encontrada: {template_path}")
                return result
            
            with open(template_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            result.data = content
            result.message = f"Plantilla LaTeX leída: {template_name}"
            
        except Exception as e:
            result.add_error(f"Error leyendo plantilla: {str(e)}")
            logger.error(f"Error en read_template: {e}")
        
        return result
    
    def write_latex(self, content: str, filepath: Union[str, Path]) -> FileOperationResult:
        """
        Escribe contenido a archivo LaTeX
        
        Parameters
        ----------
        content : str
            Contenido LaTeX
        filepath : str or Path
            Ruta del archivo de destino
            
        Returns
        -------
        FileOperationResult
            Resultado de la operación
        """
        filepath = Path(filepath)
        result = FileOperationResult()
        
        try:
            # Crear directorio si no existe
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            
            result.message = f"Archivo LaTeX guardado: {filepath.name}"
            
        except PermissionError:
            result.add_error(f"Sin permisos para escribir: {filepath}")
        except Exception as e:
            result.add_error(f"Error escribiendo LaTeX: {str(e)}")
            logger.error(f"Error en write_latex: {e}")
        
        return result
    
    def dataframe_to_latex(self, df: pd.DataFrame, 
                          caption: str = "", decimals: int = 3,
                          **kwargs) -> str:
        """
        Convierte DataFrame a tabla LaTeX
        
        Parameters
        ----------
        df : DataFrame
            DataFrame a convertir
        caption : str
            Título de la tabla
        decimals : int
            Número de decimales para números
        **kwargs
            Argumentos adicionales para DataFrame.to_latex()
            
        Returns
        -------
        str
            Código LaTeX de la tabla
        """
        # Configurar parámetros por defecto
        default_kwargs = {
            'index': False,
            'escape': False,
            'column_format': 'l' + 'c' * (len(df.columns) - 1),
            'caption': caption,
            'float_format': f'{{:.{decimals}f}}'.format
        }
        default_kwargs.update(kwargs)
        
        # Generar tabla LaTeX
        latex_table = df.to_latex(**default_kwargs)
        
        # Mejorar formato
        latex_table = latex_table.replace('\\toprule', '\\hline')
        latex_table = latex_table.replace('\\midrule', '\\hline')
        latex_table = latex_table.replace('\\bottomrule', '\\hline')
        
        return latex_table


class ProjectArchiveHandler:
    """
    Manejador para archivado y compresión de proyectos
    """
    
    def __init__(self):
        """Inicializa el manejador de archivos"""
        self.temp_dir = Path(tempfile.gettempdir()) / "seismic_projects"
        self.temp_dir.mkdir(exist_ok=True)
    
    def create_project_archive(self, project_data: Dict[str, Any], 
                              output_path: Union[str, Path],
                              include_files: List[Path] = None) -> FileOperationResult:
        """
        Crea archivo comprimido del proyecto
        
        Parameters
        ----------
        project_data : dict
            Datos del proyecto a archivar
        output_path : str or Path
            Ruta del archivo ZIP de salida
        include_files : list, optional
            Archivos adicionales a incluir
            
        Returns
        -------
        FileOperationResult
            Resultado de la operación
        """
        output_path = Path(output_path)
        result = FileOperationResult()
        
        try:
            with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Guardar datos del proyecto como JSON
                project_json = json.dumps(
                    project_data, 
                    indent=2, 
                    ensure_ascii=False,
                    default=self._archive_serializer
                )
                zipf.writestr("project_data.json", project_json)
                
                # Incluir archivos adicionales
                if include_files:
                    for file_path in include_files:
                        if Path(file_path).exists():
                            zipf.write(file_path, Path(file_path).name)
                
                # Crear archivo de metadatos
                metadata = {
                    'created_date': datetime.now().isoformat(),
                    'version': '1.0',
                    'project_name': project_data.get('project', {}).get('name', 'Sin nombre'),
                    'files_included': len(include_files) if include_files else 0
                }
                
                zipf.writestr("metadata.json", json.dumps(metadata, indent=2))
            
            result.message = f"Proyecto archivado en: {output_path.name}"
            
        except Exception as e:
            result.add_error(f"Error creando archivo: {str(e)}")
            logger.error(f"Error en create_project_archive: {e}")
        
        return result
    
    def extract_project_archive(self, archive_path: Union[str, Path],
                               extract_to: Union[str, Path] = None) -> FileOperationResult:
        """
        Extrae archivo de proyecto
        
        Parameters
        ----------
        archive_path : str or Path
            Ruta del archivo ZIP
        extract_to : str or Path, optional
            Directorio de extracción
            
        Returns
        -------
        FileOperationResult
            Resultado con datos del proyecto
        """
        archive_path = Path(archive_path)
        result = FileOperationResult()
        
        try:
            if not archive_path.exists():
                result.add_error(f"Archivo no encontrado: {archive_path}")
                return result
            
            # Directorio de extracción
            if extract_to is None:
                extract_to = self.temp_dir / f"extracted_{archive_path.stem}"
            else:
                extract_to = Path(extract_to)
            
            extract_to.mkdir(parents=True, exist_ok=True)
            
            with zipfile.ZipFile(archive_path, 'r') as zipf:
                # Extraer archivos
                zipf.extractall(extract_to)
                
                # Leer datos del proyecto
                project_json_path = extract_to / "project_data.json"
                if project_json_path.exists():
                    with open(project_json_path, 'r', encoding='utf-8') as f:
                        project_data = json.load(f)
                    result.data = project_data
                
                # Leer metadatos
                metadata_path = extract_to / "metadata.json"
                if metadata_path.exists():
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                    result.data = result.data or {}
                    result.data['_metadata'] = metadata
            
            result.message = f"Proyecto extraído a: {extract_to}"
            
        except Exception as e:
            result.add_error(f"Error extrayendo archivo: {str(e)}")
            logger.error(f"Error en extract_project_archive: {e}")
        
        return result
    
    def _archive_serializer(self, obj):
        """Serializador para archivado"""
        if isinstance(obj, (datetime, pd.Timestamp)):
            return obj.isoformat()
        elif isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        else:
            return str(obj)


class UnifiedFileManager:
    """
    Gestor unificado de archivos que combina todos los manejadores
    """
    
    def __init__(self, template_dir: Union[str, Path] = None):
        """
        Inicializa el gestor unificado
        
        Parameters
        ----------
        template_dir : str or Path, optional
            Directorio de plantillas LaTeX
        """
        self.excel_handler = ExcelHandler()
        self.csv_handler = CSVHandler()
        self.json_handler = JSONHandler()
        self.latex_handler = LaTeXHandler(template_dir)
        self.archive_handler = ProjectArchiveHandler()
        
        self.operation_history = []
    
    def read_file(self, filepath: Union[str, Path], 
                  file_format: Union[str, FileFormat] = None,
                  **kwargs) -> FileOperationResult:
        """
        Lee archivo detectando formato automáticamente
        
        Parameters
        ----------
        filepath : str or Path
            Ruta del archivo
        file_format : str or FileFormat, optional
            Formato específico (si no se proporciona, se detecta)
        **kwargs
            Argumentos adicionales para el lector específico
            
        Returns
        -------
        FileOperationResult
            Resultado con datos o errores
        """
        filepath = Path(filepath)
        
        # Detectar formato si no se especifica
        if file_format is None:
            file_format = self._detect_format(filepath)
        elif isinstance(file_format, str):
            file_format = FileFormat(file_format.lower())
        
        # Delegar a manejador específico
        if file_format == FileFormat.EXCEL:
            result = self.excel_handler.read_excel(filepath, **kwargs)
        elif file_format == FileFormat.CSV:
            result = self.csv_handler.read_csv(filepath, **kwargs)
        elif file_format == FileFormat.JSON:
            result = self.json_handler.read_json(filepath, **kwargs)
        elif file_format == FileFormat.LATEX:
            result = self.latex_handler.read_template(filepath.name)
        else:
            result = FileOperationResult(
                success=False, 
                message=f"Formato no soportado: {file_format}"
            )
        
        # Registrar operación
        self.operation_history.append({
            'operation': 'read',
            'filepath': filepath,
            'format': file_format,
            'success': result.success,
            'timestamp': datetime.now()
        })
        
        return result
    
    def write_file(self, data: Any, filepath: Union[str, Path],
                   file_format: Union[str, FileFormat] = None,
                   **kwargs) -> FileOperationResult:
        """
        Escribe datos a archivo
        
        Parameters
        ----------
        data : Any
            Datos a escribir
        filepath : str or Path
            Ruta del archivo
        file_format : str or FileFormat, optional
            Formato específico (si no se proporciona, se detecta)
        **kwargs
            Argumentos adicionales para el escritor específico
            
        Returns
        -------
        FileOperationResult
            Resultado de la operación
        """
        filepath = Path(filepath)
        
        # Detectar formato si no se especifica
        if file_format is None:
            file_format = self._detect_format(filepath)
        elif isinstance(file_format, str):
            file_format = FileFormat(file_format.lower())
        
        # Delegar a manejador específico
        if file_format == FileFormat.EXCEL:
            result = self.excel_handler.write_excel(data, filepath, **kwargs)
        elif file_format == FileFormat.CSV:
            if isinstance(data, pd.DataFrame):
                result = self.csv_handler.write_csv(data, filepath, **kwargs)
            else:
                result = FileOperationResult(False, "CSV requiere DataFrame")
        elif file_format == FileFormat.JSON:
            result = self.json_handler.write_json(data, filepath, **kwargs)
        elif file_format == FileFormat.LATEX:
            if isinstance(data, str):
                result = self.latex_handler.write_latex(data, filepath)
            else:
                result = FileOperationResult(False, "LaTeX requiere string")
        else:
            result = FileOperationResult(
                success=False,
                message=f"Formato no soportado para escritura: {file_format}"
            )
        
        # Registrar operación
        self.operation_history.append({
            'operation': 'write',
            'filepath': filepath,
            'format': file_format,
            'success': result.success,
            'timestamp': datetime.now()
        })
        
        return result
    
    def _detect_format(self, filepath: Path) -> FileFormat:
        """
        Detecta formato de archivo por extensión
        
        Parameters
        ----------
        filepath : Path
            Ruta del archivo
            
        Returns
        -------
        FileFormat
            Formato detectado
        """
        extension = filepath.suffix.lower()
        
        format_map = {
            '.xlsx': FileFormat.EXCEL,
            '.xls': FileFormat.EXCEL,
            '.xlsm': FileFormat.EXCEL,
            '.csv': FileFormat.CSV,
            '.json': FileFormat.JSON,
            '.tex': FileFormat.LATEX,
            '.txt': FileFormat.TXT,
            '.zip': FileFormat.ZIP,
            '.pkl': FileFormat.PICKLE,
            '.pickle': FileFormat.PICKLE
        }
        
        return format_map.get(extension, FileFormat.TXT)
    
    def get_file_info(self, filepath: Union[str, Path]) -> FileInfo:
        """
        Obtiene información detallada de un archivo
        
        Parameters
        ----------
        filepath : str or Path
            Ruta del archivo
            
        Returns
        -------
        FileInfo
            Información del archivo
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {filepath}")
        
        # Información básica
        stat = filepath.stat()
        file_format = self._detect_format(filepath)
        
        info = FileInfo(
            filepath=filepath,
            format=file_format,
            size_bytes=stat.st_size,
            modified_date=datetime.fromtimestamp(stat.st_mtime)
        )
        
        # Información específica por tipo
        try:
            if file_format == FileFormat.EXCEL:
                info.sheets = self.excel_handler.get_sheet_names(filepath)
            elif file_format == FileFormat.CSV:
                # Leer primera fila para obtener columnas
                result = self.csv_handler.read_csv(filepath, nrows=0)
                if result.success:
                    info.columns = list(result.data.columns)
            elif file_format == FileFormat.JSON:
                result = self.json_handler.read_json(filepath)
                if result.success and isinstance(result.data, dict):
                    info.columns = list(result.data.keys())
        except Exception as e:
            logger.warning(f"No se pudo obtener información adicional de {filepath}: {e}")
        
        return info
    
    def backup_file(self, filepath: Union[str, Path], 
                   backup_dir: Union[str, Path] = None) -> FileOperationResult:
        """
        Crea backup de un archivo
        
        Parameters
        ----------
        filepath : str or Path
            Archivo a respaldar
        backup_dir : str or Path, optional
            Directorio de backup
            
        Returns
        -------
        FileOperationResult
            Resultado de la operación
        """
        filepath = Path(filepath)
        result = FileOperationResult()
        
        try:
            if not filepath.exists():
                result.add_error(f"Archivo no encontrado: {filepath}")
                return result
            
            # Directorio de backup
            if backup_dir is None:
                backup_dir = filepath.parent / "backups"
            else:
                backup_dir = Path(backup_dir)
            
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # Nombre del backup con timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"{filepath.stem}_{timestamp}{filepath.suffix}"
            backup_path = backup_dir / backup_name
            
            # Copiar archivo
            shutil.copy2(filepath, backup_path)
            
            result.message = f"Backup creado: {backup_path}"
            result.data = backup_path
            
        except Exception as e:
            result.add_error(f"Error creando backup: {str(e)}")
            logger.error(f"Error en backup_file: {e}")
        
        return result
    
    def validate_file(self, filepath: Union[str, Path]) -> FileOperationResult:
        """
        Valida integridad de un archivo
        
        Parameters
        ----------
        filepath : str or Path
            Archivo a validar
            
        Returns
        -------
        FileOperationResult
            Resultado de la validación
        """
        filepath = Path(filepath)
        result = FileOperationResult()
        
        try:
            if not filepath.exists():
                result.add_error("Archivo no existe")
                return result
            
            file_format = self._detect_format(filepath)
            
            # Validaciones específicas por formato
            if file_format == FileFormat.EXCEL:
                read_result = self.excel_handler.read_excel(filepath, nrows=1)
                if not read_result.success:
                    result.add_error("Archivo Excel corrupto o inválido")
                else:
                    result.message = "Archivo Excel válido"
            
            elif file_format == FileFormat.CSV:
                read_result = self.csv_handler.read_csv(filepath, nrows=1)
                if not read_result.success:
                    result.add_error("Archivo CSV inválido")
                else:
                    result.message = "Archivo CSV válido"
            
            elif file_format == FileFormat.JSON:
                read_result = self.json_handler.read_json(filepath)
                if not read_result.success:
                    result.add_error("Archivo JSON inválido")
                else:
                    result.message = "Archivo JSON válido"
            
            else:
                result.message = f"Archivo {file_format.value} presente"
        
        except Exception as e:
            result.add_error(f"Error validando archivo: {str(e)}")
        
        return result
    
    def get_operation_history(self) -> List[Dict[str, Any]]:
        """
        Obtiene historial de operaciones de archivo
        
        Returns
        -------
        List[Dict[str, Any]]
            Lista con historial de operaciones
        """
        return self.operation_history.copy()
    
    def clear_operation_history(self):
        """Limpia el historial de operaciones"""
        self.operation_history.clear()


class SeismicDataImporter:
    """
    Importador especializado para datos sísmicos desde diferentes software
    """
    
    def __init__(self):
        """Inicializa el importador de datos sísmicos"""
        self.file_manager = UnifiedFileManager()
        self.supported_software = ['sap2000', 'etabs', 'generic_excel', 'generic_csv']
    
    def import_from_software(self, filepath: Union[str, Path], 
                           software: str, **kwargs) -> FileOperationResult:
        """
        Importa datos sísmicos desde software específico
        
        Parameters
        ----------
        filepath : str or Path
            Ruta del archivo de datos
        software : str
            Software origen ('sap2000', 'etabs', 'generic_excel', etc.)
        **kwargs
            Argumentos adicionales específicos del software
            
        Returns
        -------
        FileOperationResult
            Resultado con datos sísmicos estructurados
        """
        result = FileOperationResult()
        
        try:
            if software.lower() not in self.supported_software:
                result.add_error(f"Software no soportado: {software}")
                return result
            
            # Delegar a método específico
            if software.lower() == 'sap2000':
                return self._import_from_sap2000(filepath, **kwargs)
            elif software.lower() == 'etabs':
                return self._import_from_etabs(filepath, **kwargs)
            elif software.lower() == 'generic_excel':
                return self._import_from_generic_excel(filepath, **kwargs)
            elif software.lower() == 'generic_csv':
                return self._import_from_generic_csv(filepath, **kwargs)
            
        except Exception as e:
            result.add_error(f"Error importando desde {software}: {str(e)}")
            logger.error(f"Error en import_from_software: {e}")
        
        return result
    
    def _import_from_sap2000(self, filepath: Path, **kwargs) -> FileOperationResult:
        """Importa datos desde SAP2000"""
        result = FileOperationResult()
        
        try:
            # Leer archivo Excel de SAP2000
            excel_result = self.file_manager.read_file(filepath, 'excel')
            
            if not excel_result.success:
                return excel_result
            
            # Estructurar datos según formato SAP2000
            sap_data = excel_result.data
            structured_data = {}
            
            # Procesar hojas específicas de SAP2000
            if isinstance(sap_data, dict):
                for sheet_name, df in sap_data.items():
                    if 'modal' in sheet_name.lower():
                        structured_data['modal_data'] = self._process_modal_data(df)
                    elif 'drift' in sheet_name.lower():
                        structured_data['drift_data'] = self._process_drift_data(df)
                    elif 'displ' in sheet_name.lower():
                        structured_data['displacement_data'] = self._process_displacement_data(df)
            
            result.data = structured_data
            result.message = f"Datos SAP2000 importados exitosamente"
            
        except Exception as e:
            result.add_error(f"Error procesando datos SAP2000: {str(e)}")
        
        return result
    
    def _import_from_etabs(self, filepath: Path, **kwargs) -> FileOperationResult:
        """Importa datos desde ETABS"""
        result = FileOperationResult()
        
        try:
            # Similar a SAP2000 pero con variaciones en formato
            excel_result = self.file_manager.read_file(filepath, 'excel')
            
            if not excel_result.success:
                return excel_result
            
            etabs_data = excel_result.data
            structured_data = {}
            
            # Procesar según formato ETABS
            if isinstance(etabs_data, dict):
                for sheet_name, df in etabs_data.items():
                    if 'mode' in sheet_name.lower():
                        structured_data['modal_data'] = self._process_modal_data(df)
                    elif 'story' in sheet_name.lower():
                        if 'drift' in sheet_name.lower():
                            structured_data['drift_data'] = self._process_drift_data(df)
                        elif 'displ' in sheet_name.lower():
                            structured_data['displacement_data'] = self._process_displacement_data(df)
            
            result.data = structured_data
            result.message = f"Datos ETABS importados exitosamente"
            
        except Exception as e:
            result.add_error(f"Error procesando datos ETABS: {str(e)}")
        
        return result
    
    def _import_from_generic_excel(self, filepath: Path, **kwargs) -> FileOperationResult:
        """Importa datos desde Excel genérico"""
        result = FileOperationResult()
        
        try:
            excel_result = self.file_manager.read_file(filepath, 'excel')
            
            if not excel_result.success:
                return excel_result
            
            # Procesamiento genérico de Excel
            data = excel_result.data
            
            if isinstance(data, pd.DataFrame):
                # Un solo DataFrame
                result.data = {'main_data': data}
            elif isinstance(data, dict):
                # Múltiples hojas
                result.data = data
            
            result.message = "Datos Excel genéricos importados"
            
        except Exception as e:
            result.add_error(f"Error importando Excel genérico: {str(e)}")
        
        return result
    
    def _import_from_generic_csv(self, filepath: Path, **kwargs) -> FileOperationResult:
        """Importa datos desde CSV genérico"""
        result = FileOperationResult()
        
        try:
            csv_result = self.file_manager.read_file(filepath, 'csv', **kwargs)
            
            if not csv_result.success:
                return csv_result
            
            result.data = {'main_data': csv_result.data}
            result.message = "Datos CSV genéricos importados"
            
        except Exception as e:
            result.add_error(f"Error importando CSV genérico: {str(e)}")
        
        return result
    
    def _process_modal_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Procesa datos modales"""
        processed = {
            'periods': [],
            'frequencies': [],
            'mass_participation': []
        }
        
        try:
            # Buscar columnas relevantes
            period_cols = [col for col in df.columns if 'period' in col.lower() or 'periodo' in col.lower()]
            freq_cols = [col for col in df.columns if 'freq' in col.lower()]
            mass_cols = [col for col in df.columns if 'mass' in col.lower() or 'masa' in col.lower()]
            
            if period_cols:
                processed['periods'] = df[period_cols[0]].dropna().tolist()
            if freq_cols:
                processed['frequencies'] = df[freq_cols[0]].dropna().tolist()
            if mass_cols:
                processed['mass_participation'] = df[mass_cols[0]].dropna().tolist()
        
        except Exception as e:
            logger.warning(f"Error procesando datos modales: {e}")
        
        return processed
    
    def _process_drift_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Procesa datos de deriva"""
        processed = {
            'story_names': [],
            'drifts_x': [],
            'drifts_y': []
        }
        
        try:
            # Buscar columnas relevantes
            story_cols = [col for col in df.columns if 'story' in col.lower() or 'piso' in col.lower()]
            drift_x_cols = [col for col in df.columns if ('drift' in col.lower() and 'x' in col.lower()) or 'deriva' in col.lower()]
            drift_y_cols = [col for col in df.columns if ('drift' in col.lower() and 'y' in col.lower()) or 'deriva' in col.lower()]
            
            if story_cols:
                processed['story_names'] = df[story_cols[0]].dropna().tolist()
            if drift_x_cols:
                processed['drifts_x'] = df[drift_x_cols[0]].dropna().tolist()
            if drift_y_cols and len(drift_y_cols) > len(drift_x_cols):
                processed['drifts_y'] = df[drift_y_cols[-1]].dropna().tolist()
        
        except Exception as e:
            logger.warning(f"Error procesando datos de deriva: {e}")
        
        return processed
    
    def _process_displacement_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Procesa datos de desplazamientos"""
        processed = {
            'story_names': [],
            'displacements_x': [],
            'displacements_y': []
        }
        
        try:
            # Similar al procesamiento de derivas
            story_cols = [col for col in df.columns if 'story' in col.lower() or 'piso' in col.lower()]
            disp_x_cols = [col for col in df.columns if 'x' in col.lower() and ('disp' in col.lower() or 'desp' in col.lower())]
            disp_y_cols = [col for col in df.columns if 'y' in col.lower() and ('disp' in col.lower() or 'desp' in col.lower())]
            
            if story_cols:
                processed['story_names'] = df[story_cols[0]].dropna().tolist()
            if disp_x_cols:
                processed['displacements_x'] = df[disp_x_cols[0]].dropna().tolist()
            if disp_y_cols:
                processed['displacements_y'] = df[disp_y_cols[0]].dropna().tolist()
        
        except Exception as e:
            logger.warning(f"Error procesando datos de desplazamiento: {e}")
        
        return processed


# ============================================================================
# FUNCIONES DE CONVENIENCIA
# ============================================================================

def quick_read_excel(filepath: Union[str, Path], **kwargs) -> pd.DataFrame:
    """
    Función de conveniencia para leer Excel rápidamente
    
    Parameters
    ----------
    filepath : str or Path
        Ruta del archivo Excel
    **kwargs
        Argumentos adicionales
        
    Returns
    -------
    DataFrame
        Datos leídos (None si hay error)
    """
    handler = ExcelHandler()
    result = handler.read_excel(filepath, **kwargs)
    
    if result.success:
        return result.data
    else:
        logger.error(f"Error leyendo {filepath}: {result.errors}")
        return None


def quick_write_excel(data: Union[pd.DataFrame, Dict], 
                     filepath: Union[str, Path], **kwargs) -> bool:
    """
    Función de conveniencia para escribir Excel rápidamente
    
    Parameters
    ----------
    data : DataFrame or dict
        Datos a escribir
    filepath : str or Path
        Ruta del archivo
    **kwargs
        Argumentos adicionales
        
    Returns
    -------
    bool
        True si fue exitoso
    """
    handler = ExcelHandler()
    result = handler.write_excel(data, filepath, **kwargs)
    
    if not result.success:
        logger.error(f"Error escribiendo {filepath}: {result.errors}")
    
    return result.success


def batch_export_tables(tables_data: Dict[str, pd.DataFrame], 
                       output_dir: Union[str, Path],
                       formats: List[str] = ['excel', 'csv']) -> Dict[str, bool]:
    """
    Exporta múltiples tablas en diferentes formatos
    
    Parameters
    ----------
    tables_data : dict
        Diccionario {nombre_tabla: DataFrame}
    output_dir : str or Path
        Directorio de salida
    formats : list
        Formatos a exportar
        
    Returns
    -------
    dict
        Resultados {formato: éxito}
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    manager = UnifiedFileManager()
    results = {}
    
    for format_name in formats:
        try:
            if format_name.lower() == 'excel':
                # Exportar como un archivo Excel con múltiples hojas
                output_file = output_dir / "tablas_completas.xlsx"
                result = manager.write_file(tables_data, output_file, 'excel')
                results['excel'] = result.success
                
            elif format_name.lower() == 'csv':
                # Exportar cada tabla como CSV separado
                all_success = True
                for table_name, df in tables_data.items():
                    output_file = output_dir / f"{table_name}.csv"
                    result = manager.write_file(df, output_file, 'csv')
                    if not result.success:
                        all_success = False
                results['csv'] = all_success
                
        except Exception as e:
            logger.error(f"Error exportando formato {format_name}: {e}")
            results[format_name] = False
    
    return results


def create_project_backup(project_data: Dict[str, Any], 
                         backup_dir: Union[str, Path],
                         include_files: List[Path] = None) -> bool:
    """
    Crea backup completo de proyecto
    
    Parameters
    ----------
    project_data : dict
        Datos del proyecto
    backup_dir : str or Path
        Directorio de backup
    include_files : list, optional
        Archivos adicionales a incluir
        
    Returns
    -------
    bool
        True si fue exitoso
    """
    try:
        archive_handler = ProjectArchiveHandler()
        
        # Crear nombre de archivo con timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        project_name = project_data.get('project', {}).get('name', 'proyecto')
        archive_name = f"{project_name}_{timestamp}.zip"
        
        backup_path = Path(backup_dir) / archive_name
        
        result = archive_handler.create_project_archive(
            project_data, backup_path, include_files
        )
        
        return result.success
        
    except Exception as e:
        logger.error(f"Error creando backup: {e}")
        return False


# ============================================================================
# EXPORTACIÓN DE FUNCIONES PRINCIPALES
# ============================================================================

__all__ = [
    'FileFormat',
    'FileInfo', 
    'FileOperationResult',
    'ExcelHandler',
    'CSVHandler',
    'JSONHandler',
    'LaTeXHandler',
    'ProjectArchiveHandler',
    'UnifiedFileManager',
    'SeismicDataImporter',
    'quick_read_excel',
    'quick_write_excel',
    'batch_export_tables',
    'create_project_backup'
]