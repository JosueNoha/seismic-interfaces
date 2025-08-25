"""
Modelo centralizado de tabla pandas para interfaces sísmicas
===========================================================

Este módulo proporciona modelos de tabla centralizados para mostrar DataFrames 
de pandas en interfaces PyQt5, específicamente diseñados para proyectos de 
análisis sísmico.

Características principales:
- Compatibilidad completa con código existente (pandasModel)
- Funcionalidades avanzadas (filtros, ordenamiento, exportación)
- Soporte para diferentes tipos de datos sísmicos
- Validación y formateo automático de datos
- Configuración flexible de visualización

Ejemplo de uso:
    ```python
    from seismic_common.models import PandasTableModel, SeismicTableModel
    
    # Uso básico (compatible con código existente)
    model = PandasTableModel(dataframe)
    table_view.setModel(model)
    
    # Uso avanzado para datos sísmicos
    seismic_model = SeismicTableModel(modal_data, table_type='modal')
    table_view.setModel(seismic_model)
    ```
"""

# Metadatos del módulo
__version__ = "1.0.0"
__author__ = "Proyecto Interfaces Sísmicas"
__description__ = "Modelos de tabla pandas centralizados para interfaces sísmicas"
__license__ = "MIT"
__status__ = "Production"

import sys
import logging
from typing import Optional, Dict, Any, List, Union, Tuple, Callable
from abc import ABC, abstractmethod

from PyQt5 import QtCore, QtGui
from PyQt5.QtCore import QAbstractTableModel, Qt, QModelIndex, pyqtSignal
from PyQt5.QtGui import QColor, QBrush, QFont

import pandas as pd
import numpy as np

# Configurar logging
logger = logging.getLogger(__name__)

# Constantes para formateo
DECIMAL_PLACES = {
    'period': 4,      # Periodos con 4 decimales
    'frequency': 3,   # Frecuencias con 3 decimales
    'percentage': 2,  # Porcentajes con 2 decimales
    'force': 1,       # Fuerzas con 1 decimal
    'displacement': 3, # Desplazamientos con 3 decimales
    'ratio': 4,       # Ratios con 4 decimales
    'default': 2      # Valor por defecto
}

# Colores para diferentes tipos de datos
COLORS = {
    'critical': QColor(255, 200, 200),      # Rojo claro para valores críticos
    'warning': QColor(255, 255, 200),       # Amarillo claro para advertencias
    'good': QColor(200, 255, 200),          # Verde claro para valores buenos
    'header': QColor(240, 240, 240),        # Gris claro para headers
    'selected': QColor(200, 200, 255),      # Azul claro para selecciones
    'default': QColor(255, 255, 255)        # Blanco por defecto
}


class PandasTableModel(QAbstractTableModel):
    """
    Modelo estándar para mostrar DataFrames de pandas en QTableView
    
    Compatible con el código existente (reemplaza pandasModel)
    Proporciona funcionalidad básica de visualización de datos tabulares
    """
    
    # Señales para comunicación con la vista
    dataChanged = pyqtSignal(QModelIndex, QModelIndex)
    modelReset = pyqtSignal()
    
    def __init__(self, data: pd.DataFrame, parent=None):
        """
        Inicializa el modelo con un DataFrame
        
        Parameters
        ----------
        data : pd.DataFrame
            DataFrame a mostrar en la tabla
        parent : QObject, optional
            Widget padre del modelo
        """
        super().__init__(parent)
        
        # Validar y procesar datos
        if data is None or data.empty:
            logger.warning("DataFrame vacío o None proporcionado")
            self._data = pd.DataFrame()
        else:
            # Crear copia para evitar modificaciones accidentales
            self._data = data.copy()
            
        # Datos originales para reseteo
        self._original_data = self._data.copy()
        
        # Configuraciones del modelo
        self._editable = False
        self._sortable = True
        self._filterable = True
        
        logger.debug(f"PandasTableModel inicializado con {self.rowCount()} filas y {self.columnCount()} columnas")
    
    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        """
        Retorna el número de filas en el modelo
        
        Parameters
        ----------
        parent : QModelIndex
            Índice padre (no usado para tablas planas)
            
        Returns
        -------
        int
            Número de filas
        """
        return len(self._data.index) if not self._data.empty else 0
    
    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        """
        Retorna el número de columnas en el modelo
        
        Parameters
        ----------
        parent : QModelIndex
            Índice padre (no usado para tablas planas)
            
        Returns
        -------
        int
            Número de columnas
        """
        return len(self._data.columns) if not self._data.empty else 0
    
    def data(self, index: QModelIndex, role: int = Qt.DisplayRole) -> Any:
        """
        Retorna los datos para un índice y rol específicos
        
        Parameters
        ----------
        index : QModelIndex
            Índice de la celda
        role : int
            Rol de los datos (DisplayRole, BackgroundRole, etc.)
            
        Returns
        -------
        Any
            Datos solicitados o None si el índice no es válido
        """
        if not index.isValid() or self._data.empty:
            return None
            
        row = index.row()
        col = index.column()
        
        # Verificar límites
        if row >= self.rowCount() or col >= self.columnCount():
            return None
        
        # Obtener valor
        try:
            value = self._data.iloc[row, col]
        except (IndexError, KeyError) as e:
            logger.warning(f"Error accediendo a celda [{row}, {col}]: {e}")
            return None
        
        # Procesar según el rol
        if role == Qt.DisplayRole:
            return self._format_display_value(value, col)
        elif role == Qt.TextAlignmentRole:
            return self._get_alignment(value, col)
        elif role == Qt.BackgroundRole:
            return self._get_background_color(value, col)
        elif role == Qt.FontRole:
            return self._get_font(value, col)
        elif role == Qt.ToolTipRole:
            return self._get_tooltip(value, col)
        
        return None
    
    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.DisplayRole) -> Any:
        """
        Retorna los datos del header para una sección específica
        
        Parameters
        ----------
        section : int
            Número de sección (fila/columna)
        orientation : Qt.Orientation
            Orientación (Horizontal para columnas, Vertical para filas)
        role : int
            Rol de los datos
            
        Returns
        -------
        Any
            Datos del header o None
        """
        if self._data.empty:
            return None
            
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                # Headers de columnas
                if 0 <= section < len(self._data.columns):
                    return str(self._data.columns[section])
            elif orientation == Qt.Vertical:
                # Headers de filas (índices)
                if 0 <= section < len(self._data.index):
                    return str(self._data.index[section])
        
        elif role == Qt.FontRole and orientation == Qt.Horizontal:
            # Fuente para headers de columnas
            font = QFont()
            font.setBold(True)
            return font
            
        elif role == Qt.BackgroundRole:
            # Color de fondo para headers
            return QBrush(COLORS['header'])
        
        return None
    
    def flags(self, index: QModelIndex) -> Qt.ItemFlags:
        """
        Retorna las flags para un índice específico
        
        Parameters
        ----------
        index : QModelIndex
            Índice de la celda
            
        Returns
        -------
        Qt.ItemFlags
            Flags del item
        """
        if not index.isValid():
            return Qt.NoItemFlags
        
        flags = Qt.ItemIsEnabled | Qt.ItemIsSelectable
        
        if self._editable:
            flags |= Qt.ItemIsEditable
        
        return flags
    
    def sort(self, column: int, order: Qt.SortOrder = Qt.AscendingOrder):
        """
        Ordena el modelo por una columna específica
        
        Parameters
        ----------
        column : int
            Índice de la columna a ordenar
        order : Qt.SortOrder
            Orden de clasificación (Ascendente/Descendente)
        """
        if not self._sortable or self._data.empty or column >= self.columnCount():
            return
            
        self.layoutAboutToBeChanged.emit()
        
        try:
            # Obtener nombre de la columna
            column_name = self._data.columns[column]
            
            # Ordenar DataFrame
            ascending = (order == Qt.AscendingOrder)
            self._data = self._data.sort_values(by=column_name, ascending=ascending)
            
            # Resetear índices
            self._data = self._data.reset_index(drop=True)
            
            logger.debug(f"Datos ordenados por columna '{column_name}' ({'asc' if ascending else 'desc'})")
            
        except Exception as e:
            logger.error(f"Error ordenando por columna {column}: {e}")
            
        self.layoutChanged.emit()
    
    def _format_display_value(self, value: Any, column: int) -> str:
        """
        Formatea un valor para mostrar
        
        Parameters
        ----------
        value : Any
            Valor a formatear
        column : int
            Índice de la columna
            
        Returns
        -------
        str
            Valor formateado
        """
        if pd.isna(value):
            return ""
        
        # Detectar tipo de columna por nombre
        column_name = self._data.columns[column].lower()
        
        # Aplicar formateo específico
        try:
            if isinstance(value, (int, np.integer)):
                return str(int(value))
            elif isinstance(value, (float, np.floating)):
                # Determinar número de decimales
                decimals = DECIMAL_PLACES['default']
                
                for key, places in DECIMAL_PLACES.items():
                    if key in column_name:
                        decimals = places
                        break
                
                return f"{float(value):.{decimals}f}"
            else:
                return str(value)
        except (ValueError, TypeError):
            return str(value)
    
    def _get_alignment(self, value: Any, column: int) -> Qt.Alignment:
        """
        Determina la alineación para un valor
        
        Parameters
        ----------
        value : Any
            Valor de la celda
        column : int
            Índice de la columna
            
        Returns
        -------
        Qt.Alignment
            Alineación del texto
        """
        if isinstance(value, (int, float, np.number)):
            return Qt.AlignRight | Qt.AlignVCenter
        else:
            return Qt.AlignLeft | Qt.AlignVCenter
    
    def _get_background_color(self, value: Any, column: int) -> Optional[QBrush]:
        """
        Determina el color de fondo para una celda
        
        Parameters
        ----------
        value : Any
            Valor de la celda
        column : int
            Índice de la columna
            
        Returns
        -------
        QBrush or None
            Color de fondo o None para color por defecto
        """
        # Esta implementación básica no aplica colores especiales
        # Las clases derivadas pueden sobrescribir este método
        return None
    
    def _get_font(self, value: Any, column: int) -> Optional[QFont]:
        """
        Determina la fuente para una celda
        
        Parameters
        ----------
        value : Any
            Valor de la celda
        column : int
            Índice de la columna
            
        Returns
        -------
        QFont or None
            Fuente específica o None para fuente por defecto
        """
        # Esta implementación básica no aplica fuentes especiales
        # Las clases derivadas pueden sobrescribir este método
        return None
    
    def _get_tooltip(self, value: Any, column: int) -> Optional[str]:
        """
        Genera tooltip para una celda
        
        Parameters
        ----------
        value : Any
            Valor de la celda
        column : int
            Índice de la columna
            
        Returns
        -------
        str or None
            Texto del tooltip o None
        """
        if pd.isna(value):
            return "Valor no disponible"
        return None
    
    # Métodos de gestión de datos
    def get_dataframe(self) -> pd.DataFrame:
        """
        Retorna una copia del DataFrame actual
        
        Returns
        -------
        pd.DataFrame
            Copia del DataFrame
        """
        return self._data.copy()
    
    def set_dataframe(self, data: pd.DataFrame):
        """
        Establece un nuevo DataFrame
        
        Parameters
        ----------
        data : pd.DataFrame
            Nuevo DataFrame
        """
        self.beginResetModel()
        
        if data is None or data.empty:
            self._data = pd.DataFrame()
        else:
            self._data = data.copy()
        
        self.endResetModel()
        logger.debug(f"DataFrame actualizado: {self.rowCount()} filas, {self.columnCount()} columnas")
    
    def reset_to_original(self):
        """Resetea los datos al estado original"""
        self.set_dataframe(self._original_data)
    
    def export_to_excel(self, filepath: str, **kwargs):
        """
        Exporta los datos a un archivo Excel
        
        Parameters
        ----------
        filepath : str
            Ruta del archivo Excel
        **kwargs
            Argumentos adicionales para pandas.to_excel()
        """
        try:
            self._data.to_excel(filepath, index=False, **kwargs)
            logger.info(f"Datos exportados a: {filepath}")
        except Exception as e:
            logger.error(f"Error exportando a Excel: {e}")
            raise
    
    def export_to_csv(self, filepath: str, **kwargs):
        """
        Exporta los datos a un archivo CSV
        
        Parameters
        ----------
        filepath : str
            Ruta del archivo CSV
        **kwargs
            Argumentos adicionales para pandas.to_csv()
        """
        try:
            self._data.to_csv(filepath, index=False, **kwargs)
            logger.info(f"Datos exportados a: {filepath}")
        except Exception as e:
            logger.error(f"Error exportando a CSV: {e}")
            raise
    
    # Propiedades de configuración
    @property
    def editable(self) -> bool:
        """Si el modelo es editable"""
        return self._editable
    
    @editable.setter
    def editable(self, value: bool):
        """Establece si el modelo es editable"""
        self._editable = value
        self.dataChanged.emit(QModelIndex(), QModelIndex())
    
    @property
    def sortable(self) -> bool:
        """Si el modelo es ordenable"""
        return self._sortable
    
    @sortable.setter
    def sortable(self, value: bool):
        """Establece si el modelo es ordenable"""
        self._sortable = value


class SeismicTableModel(PandasTableModel):
    """
    Modelo especializado para tablas de datos sísmicos
    
    Extiende PandasTableModel con funcionalidades específicas para:
    - Formateo de datos sísmicos
    - Validación de rangos críticos
    - Coloreado según criterios sísmicos
    - Tooltips informativos
    """
    
    # Diccionarios de configuración para análisis sísmico
    CRITICAL_LIMITS = {
        'drift': 0.007,      # Deriva máxima típica
        'torsion': 1.2,      # Irregularidad torsional
        'soft_story': 0.7,   # Piso blando
        'period_ratio': 0.9, # Ratio de periodos
        'mass_participation': 0.9  # Participación de masa mínima
    }
    
    WARNING_LIMITS = {
        'drift': 0.005,
        'torsion': 1.1,
        'soft_story': 0.8,
        'period_ratio': 0.95,
        'mass_participation': 0.85
    }
    
    def __init__(self, data: pd.DataFrame, table_type: str = 'general', parent=None):
        """
        Inicializa el modelo sísmico
        
        Parameters
        ----------
        data : pd.DataFrame
            DataFrame con datos sísmicos
        table_type : str
            Tipo de tabla ('modal', 'drift', 'irregularity', etc.)
        parent : QObject
            Widget padre
        """
        super().__init__(data, parent)
        
        self.table_type = table_type.lower()
        self._setup_seismic_formatting()
        
        logger.debug(f"SeismicTableModel inicializado para tipo: {table_type}")
    
    def _setup_seismic_formatting(self):
        """Configura el formateo específico según el tipo de tabla"""
        self._column_formats = {}
        self._critical_columns = []
        self._warning_columns = []
        
        if self.table_type == 'modal':
            self._setup_modal_formatting()
        elif self.table_type == 'drift':
            self._setup_drift_formatting()
        elif self.table_type == 'irregularity':
            self._setup_irregularity_formatting()
        elif self.table_type == 'static':
            self._setup_static_formatting()
    
    def _setup_modal_formatting(self):
        """Configura formateo para tabla modal"""
        self._column_formats = {
            'period': DECIMAL_PLACES['period'],
            'frequency': DECIMAL_PLACES['frequency'],
            'ux': DECIMAL_PLACES['percentage'],
            'uy': DECIMAL_PLACES['percentage'],
            'uz': DECIMAL_PLACES['percentage']
        }
    
    def _setup_drift_formatting(self):
        """Configura formateo para tabla de derivas"""
        self._column_formats = {
            'drift_x': 4,
            'drift_y': 4,
            'height': 2
        }
        self._critical_columns = ['drift_x', 'drift_y']
    
    def _setup_irregularity_formatting(self):
        """Configura formateo para tabla de irregularidades"""
        self._column_formats = {
            'ratio': DECIMAL_PLACES['ratio'],
            'factor': 2
        }
        self._critical_columns = ['ratio']
        self._warning_columns = ['factor']
    
    def _setup_static_formatting(self):
        """Configura formateo para análisis estático"""
        self._column_formats = {
            'force': DECIMAL_PLACES['force'],
            'displacement': DECIMAL_PLACES['displacement']
        }
    
    def _get_background_color(self, value: Any, column: int) -> Optional[QBrush]:
        """
        Determina color de fondo según criterios sísmicos
        
        Parameters
        ----------
        value : Any
            Valor de la celda
        column : int
            Índice de la columna
            
        Returns
        -------
        QBrush or None
            Color de fondo según criterio sísmico
        """
        if not isinstance(value, (int, float, np.number)) or pd.isna(value):
            return None
        
        column_name = self._data.columns[column].lower()
        
        # Verificar límites críticos
        for limit_type, limit_value in self.CRITICAL_LIMITS.items():
            if limit_type in column_name:
                if self._is_critical_value(value, limit_type, limit_value):
                    return QBrush(COLORS['critical'])
                elif self._is_warning_value(value, limit_type):
                    return QBrush(COLORS['warning'])
                elif self._is_good_value(value, limit_type):
                    return QBrush(COLORS['good'])
        
        return None
    
    def _is_critical_value(self, value: float, limit_type: str, limit: float) -> bool:
        """Verifica si un valor está en rango crítico"""
        if limit_type in ['drift', 'torsion']:
            return value > limit
        elif limit_type in ['soft_story', 'period_ratio', 'mass_participation']:
            return value < limit
        return False
    
    def _is_warning_value(self, value: float, limit_type: str) -> bool:
        """Verifica si un valor está en rango de advertencia"""
        warning_limit = self.WARNING_LIMITS.get(limit_type)
        if warning_limit is None:
            return False
        
        if limit_type in ['drift', 'torsion']:
            return value > warning_limit
        elif limit_type in ['soft_story', 'period_ratio', 'mass_participation']:
            return value < warning_limit
        return False
    
    def _is_good_value(self, value: float, limit_type: str) -> bool:
        """Verifica si un valor está en rango bueno"""
        warning_limit = self.WARNING_LIMITS.get(limit_type)
        if warning_limit is None:
            return False
        
        if limit_type in ['drift', 'torsion']:
            return value <= warning_limit
        elif limit_type in ['soft_story', 'period_ratio', 'mass_participation']:
            return value >= warning_limit
        return False
    
    def _get_tooltip(self, value: Any, column: int) -> Optional[str]:
        """
        Genera tooltips informativos para datos sísmicos
        
        Parameters
        ----------
        value : Any
            Valor de la celda
        column : int
            Índice de la columna
            
        Returns
        -------
        str or None
            Tooltip informativo
        """
        if pd.isna(value):
            return "Valor no disponible"
        
        column_name = self._data.columns[column].lower()
        
        # Tooltips específicos para datos sísmicos
        tooltips = {
            'drift': f"Deriva: {value:.4f}\nLímite crítico: {self.CRITICAL_LIMITS['drift']:.3f}",
            'period': f"Periodo: {value:.4f} s",
            'frequency': f"Frecuencia: {value:.3f} Hz",
            'torsion': f"Irregularidad torsional: {value:.2f}\nLímite: {self.CRITICAL_LIMITS['torsion']:.1f}",
            'mass': f"Participación de masa: {value:.1f}%\nMínimo requerido: {self.CRITICAL_LIMITS['mass_participation']*100:.0f}%"
        }
        
        for key, tooltip in tooltips.items():
            if key in column_name:
                return tooltip
        
        return f"Valor: {value}"


# Alias de compatibilidad con código existente
pandasModel = PandasTableModel  # Compatibilidad con appBolivia y appPeru

# Funciones de conveniencia para crear modelos específicos
def create_modal_table_model(modal_data: pd.DataFrame) -> SeismicTableModel:
    """
    Crea un modelo para tabla modal
    
    Parameters
    ----------
    modal_data : pd.DataFrame
        Datos del análisis modal
        
    Returns
    -------
    SeismicTableModel
        Modelo configurado para datos modales
    """
    return SeismicTableModel(modal_data, table_type='modal')

def create_drift_table_model(drift_data: pd.DataFrame) -> SeismicTableModel:
    """
    Crea un modelo para tabla de derivas
    
    Parameters
    ----------
    drift_data : pd.DataFrame
        Datos de derivas de piso
        
    Returns
    -------
    SeismicTableModel
        Modelo configurado para datos de deriva
    """
    return SeismicTableModel(drift_data, table_type='drift')

def create_irregularity_table_model(irregularity_data: pd.DataFrame) -> SeismicTableModel:
    """
    Crea un modelo para tabla de irregularidades
    
    Parameters
    ----------
    irregularity_data : pd.DataFrame
        Datos de irregularidades
        
    Returns
    -------
    SeismicTableModel
        Modelo configurado para datos de irregularidades
    """
    return SeismicTableModel(irregularity_data, table_type='irregularity')

def create_static_table_model(static_data: pd.DataFrame) -> SeismicTableModel:
    """
    Crea un modelo para análisis estático
    
    Parameters
    ----------
    static_data : pd.DataFrame
        Datos del análisis estático
        
    Returns
    -------
    SeismicTableModel
        Modelo configurado para análisis estático
    """
    return SeismicTableModel(static_data, table_type='static')


# Validación del módulo
def validate_model_compatibility():
    """
    Valida la compatibilidad del modelo con versiones anteriores
    
    Returns
    -------
    bool
        True si la compatibilidad es correcta
    """
    try:
        # Crear un DataFrame de prueba
        test_data = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['A', 'B', 'C'],
            'period': [0.5, 0.3, 0.2]
        })
        
        # Probar modelo básico
        basic_model = PandasTableModel(test_data)
        assert basic_model.rowCount() == 3
        assert basic_model.columnCount() == 3
        
        # Probar modelo sísmico
        seismic_model = SeismicTableModel(test_data, 'modal')
        assert seismic_model.rowCount() == 3
        assert seismic_model.columnCount() == 3
        
        # Probar alias de compatibilidad
        compat_model = pandasModel(test_data)
        assert compat_model.rowCount() == 3
        
        logger.info("✓ Validación de compatibilidad exitosa")
        return True
        
    except Exception as e:
        logger.error(f"✗ Error en validación de compatibilidad: {e}")
        return False

if __name__ == "__main__":
    # Ejecutar validaciones si el módulo se ejecuta directamente
    validate_model_compatibility()