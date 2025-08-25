"""
Selectores de ubicación reutilizables para interfaces sísmicas
Widgets genéricos para selección jerárquica de ubicaciones (país, estado/departamento, provincia, distrito/ciudad)
"""

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                            QComboBox, QCheckBox, QGroupBox, QFormLayout,
                            QSizePolicy, QPushButton, QMessageBox)
from PyQt5.QtCore import pyqtSignal, Qt, QObject
from typing import Dict, List, Optional, Any, Callable, Union
import pandas as pd


class LocationDatabase:
    """Clase base para manejo de base de datos de ubicaciones"""
    
    def __init__(self, data_source: Union[pd.DataFrame, Dict, str, None] = None):
        """
        Inicializa la base de datos de ubicaciones
        
        Parameters
        ----------
        data_source : Union[pd.DataFrame, Dict, str, None]
            Fuente de datos: DataFrame, diccionario, ruta de archivo o None
        """
        self.data = pd.DataFrame()
        self.location_hierarchy = []  # Lista de nombres de columnas en orden jerárquico
        
        if data_source is not None:
            self.load_data(data_source)
    
    def load_data(self, data_source: Union[pd.DataFrame, Dict, str]):
        """Carga datos desde diferentes fuentes"""
        if isinstance(data_source, pd.DataFrame):
            self.data = data_source.copy()
        elif isinstance(data_source, dict):
            self.data = pd.DataFrame(data_source)
        elif isinstance(data_source, str):
            # Asume que es un archivo CSV
            try:
                self.data = pd.read_csv(data_source)
            except Exception as e:
                raise ValueError(f"No se pudo cargar el archivo: {e}")
        else:
            raise TypeError("Tipo de fuente de datos no soportado")
    
    def set_hierarchy(self, columns: List[str]):
        """
        Establece la jerarquía de ubicaciones
        
        Parameters
        ----------
        columns : List[str]
            Lista de nombres de columnas en orden jerárquico
            Ejemplo: ['DEPARTAMENTO', 'PROVINCIA', 'DISTRITO']
        """
        self.location_hierarchy = columns
        
        # Verificar que las columnas existen
        missing_cols = [col for col in columns if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"Columnas no encontradas: {missing_cols}")
    
    def get_options(self, level: int, filters: Dict[str, str] = None) -> List[str]:
        """
        Obtiene las opciones disponibles para un nivel específico
        
        Parameters
        ----------
        level : int
            Nivel de la jerarquía (0 = primer nivel, 1 = segundo nivel, etc.)
        filters : Dict[str, str], optional
            Filtros basados en niveles superiores
            Ejemplo: {'DEPARTAMENTO': 'CUSCO'}
        
        Returns
        -------
        List[str]
            Lista de opciones disponibles
        """
        if level >= len(self.location_hierarchy):
            return []
        
        column = self.location_hierarchy[level]
        filtered_data = self.data.copy()
        
        # Aplicar filtros de niveles superiores
        if filters:
            for filter_col, filter_val in filters.items():
                if filter_col in self.data.columns:
                    filtered_data = filtered_data[filtered_data[filter_col] == filter_val]
        
        # Obtener valores únicos y ordenados
        options = sorted(filtered_data[column].dropna().unique().tolist())
        return options
    
    def get_location_data(self, location: Dict[str, str]) -> pd.DataFrame:
        """
        Obtiene datos específicos de una ubicación
        
        Parameters
        ----------
        location : Dict[str, str]
            Diccionario con la ubicación específica
            Ejemplo: {'DEPARTAMENTO': 'CUSCO', 'PROVINCIA': 'CUSCO', 'DISTRITO': 'CUSCO'}
        
        Returns
        -------
        pd.DataFrame
            Datos filtrados para la ubicación especificada
        """
        filtered_data = self.data.copy()
        
        for col, val in location.items():
            if col in self.data.columns:
                filtered_data = filtered_data[filtered_data[col] == val]
        
        return filtered_data
    
    def get_additional_data(self, location: Dict[str, str], columns: List[str]) -> Dict[str, Any]:
        """
        Obtiene datos adicionales para una ubicación específica
        
        Parameters
        ----------
        location : Dict[str, str]
            Ubicación específica
        columns : List[str]
            Columnas adicionales a obtener
        
        Returns
        -------
        Dict[str, Any]
            Datos adicionales
        """
        location_data = self.get_location_data(location)
        
        if location_data.empty:
            return {col: None for col in columns}
        
        # Tomar la primera fila si hay múltiples coincidencias
        row = location_data.iloc[0]
        return {col: row.get(col, None) for col in columns if col in row.index}


class HierarchicalLocationSelector(QWidget):
    """Selector jerárquico de ubicaciones genérico"""
    
    locationChanged = pyqtSignal(dict)  # Emite la ubicación completa seleccionada
    selectionCompleted = pyqtSignal(dict)  # Emite cuando se completa la selección
    dataUpdated = pyqtSignal(dict)  # Emite datos adicionales actualizados
    
    def __init__(self, database: LocationDatabase, parent=None):
        super().__init__(parent)
        self.database = database
        self.selectors = []  # Lista de QComboBox
        self.labels = []     # Lista de QLabel
        self.current_location = {}
        self.enabled = True
        
        self.setupUI()
        self.connectSignals()
    
    def setupUI(self):
        """Configura la interfaz del selector"""
        self.main_layout = QVBoxLayout(self)
        self.form_layout = QFormLayout()
        
        # Crear selectores para cada nivel de la jerarquía
        for i, column in enumerate(self.database.location_hierarchy):
            # Crear label y combobox
            label = QLabel(f"{column.title()}:")
            combo = QComboBox()
            combo.setEnabled(i == 0)  # Solo el primer nivel habilitado inicialmente
            
            # Conectar señal de cambio
            combo.currentTextChanged.connect(lambda text, level=i: self._on_selection_changed(level, text))
            
            # Agregar al layout
            self.form_layout.addRow(label, combo)
            
            # Guardar referencias
            self.labels.append(label)
            self.selectors.append(combo)
        
        self.main_layout.addLayout(self.form_layout)
        
        # Inicializar primer nivel
        if self.selectors:
            self._update_level_options(0)
    
    def connectSignals(self):
        """Conecta las señales internas"""
        pass  # Las conexiones se hacen en setupUI
    
    def _on_selection_changed(self, level: int, selected_value: str):
        """Maneja el cambio de selección en un nivel específico"""
        if not selected_value or not self.enabled:
            return
        
        # Actualizar ubicación actual
        column = self.database.location_hierarchy[level]
        self.current_location[column] = selected_value
        
        # Limpiar niveles inferiores
        self._clear_lower_levels(level + 1)
        
        # Actualizar siguiente nivel si existe
        if level + 1 < len(self.selectors):
            self._update_level_options(level + 1)
            self.selectors[level + 1].setEnabled(True)
        
        # Emitir señales
        self.locationChanged.emit(self.current_location.copy())
        
        # Si se completó toda la jerarquía, emitir selección completa
        if len(self.current_location) == len(self.database.location_hierarchy):
            self.selectionCompleted.emit(self.current_location.copy())
            self._emit_additional_data()
    
    def _update_level_options(self, level: int):
        """Actualiza las opciones de un nivel específico"""
        if level >= len(self.selectors):
            return
        
        # Obtener filtros de niveles superiores
        filters = {}
        for i in range(level):
            if i < len(self.database.location_hierarchy):
                column = self.database.location_hierarchy[i]
                if column in self.current_location:
                    filters[column] = self.current_location[column]
        
        # Obtener opciones y actualizar combobox
        options = self.database.get_options(level, filters)
        combo = self.selectors[level]
        
        combo.blockSignals(True)
        combo.clear()
        combo.addItems(options)
        combo.blockSignals(False)
    
    def _clear_lower_levels(self, start_level: int):
        """Limpia los niveles inferiores a partir del nivel especificado"""
        for i in range(start_level, len(self.selectors)):
            # Limpiar combobox
            self.selectors[i].blockSignals(True)
            self.selectors[i].clear()
            self.selectors[i].setEnabled(False)
            self.selectors[i].blockSignals(False)
            
            # Remover de ubicación actual
            if i < len(self.database.location_hierarchy):
                column = self.database.location_hierarchy[i]
                self.current_location.pop(column, None)
    
    def _emit_additional_data(self):
        """Emite datos adicionales de la ubicación seleccionada"""
        if len(self.current_location) == len(self.database.location_hierarchy):
            # Obtener todas las columnas adicionales disponibles
            additional_cols = [col for col in self.database.data.columns 
                             if col not in self.database.location_hierarchy]
            
            if additional_cols:
                additional_data = self.database.get_additional_data(
                    self.current_location, additional_cols
                )
                self.dataUpdated.emit(additional_data)
    
    def getCurrentLocation(self) -> Dict[str, str]:
        """Obtiene la ubicación actualmente seleccionada"""
        return self.current_location.copy()
    
    def setCurrentLocation(self, location: Dict[str, str]):
        """
        Establece la ubicación actual programáticamente
        
        Parameters
        ----------
        location : Dict[str, str]
            Ubicación a establecer
        """
        self.enabled = False  # Temporalmente deshabilitar señales
        
        try:
            # Limpiar selección actual
            self._clear_lower_levels(0)
            self.current_location = {}
            
            # Establecer cada nivel secuencialmente
            for i, column in enumerate(self.database.location_hierarchy):
                if column in location:
                    # Actualizar opciones del nivel actual
                    self._update_level_options(i)
                    
                    # Establecer valor
                    combo = self.selectors[i]
                    value = location[column]
                    
                    index = combo.findText(value)
                    if index >= 0:
                        combo.setCurrentIndex(index)
                        self.current_location[column] = value
                        
                        # Habilitar siguiente nivel
                        if i + 1 < len(self.selectors):
                            self.selectors[i + 1].setEnabled(True)
                    else:
                        break  # Valor no encontrado, detener
                else:
                    break  # Columna no especificada, detener
        
        finally:
            self.enabled = True  # Rehabilitar señales
            
            # Emitir señales finales
            if self.current_location:
                self.locationChanged.emit(self.current_location.copy())
                if len(self.current_location) == len(self.database.location_hierarchy):
                    self.selectionCompleted.emit(self.current_location.copy())
                    self._emit_additional_data()
    
    def setEnabled(self, enabled: bool):
        """Habilita o deshabilita todo el selector"""
        super().setEnabled(enabled)
        for selector in self.selectors:
            # Solo habilitar selectores que deben estar habilitados según el estado actual
            if enabled:
                # Habilitar según la lógica de selección actual
                level = self.selectors.index(selector)
                should_enable = (level == 0 or 
                               level < len(self.current_location) + 1)
                selector.setEnabled(should_enable)
            else:
                selector.setEnabled(False)


class LocationSelectorWithToggle(QWidget):
    """Selector de ubicación con opción de habilitación/deshabilitación"""
    
    locationChanged = pyqtSignal(dict)
    enabledChanged = pyqtSignal(bool)
    alternativeValueChanged = pyqtSignal(object)  # Para valor alternativo cuando está deshabilitado
    
    def __init__(self, database: LocationDatabase, 
                 toggle_text: str = "Usar ubicación específica",
                 alternative_widget: QWidget = None,
                 parent=None):
        super().__init__(parent)
        self.database = database
        self.toggle_text = toggle_text
        self.alternative_widget = alternative_widget
        
        self.setupUI()
        self.connectSignals()
    
    def setupUI(self):
        """Configura la interfaz con toggle"""
        layout = QVBoxLayout(self)
        
        # Checkbox para habilitar/deshabilitar
        self.enable_checkbox = QCheckBox(self.toggle_text)
        layout.addWidget(self.enable_checkbox)
        
        # Selector jerárquico
        self.location_selector = HierarchicalLocationSelector(self.database)
        layout.addWidget(self.location_selector)
        
        # Widget alternativo (si existe)
        if self.alternative_widget:
            layout.addWidget(self.alternative_widget)
        
        # Estado inicial
        self.location_selector.setEnabled(False)
        if self.alternative_widget:
            self.alternative_widget.setEnabled(True)
    
    def connectSignals(self):
        """Conecta las señales"""
        self.enable_checkbox.toggled.connect(self._on_toggle_changed)
        self.location_selector.locationChanged.connect(self.locationChanged.emit)
        self.location_selector.selectionCompleted.connect(self.locationChanged.emit)
        
        # Conectar señales del widget alternativo si tiene
        if self.alternative_widget and hasattr(self.alternative_widget, 'valueChanged'):
            self.alternative_widget.valueChanged.connect(self.alternativeValueChanged.emit)
        elif self.alternative_widget and hasattr(self.alternative_widget, 'currentTextChanged'):
            self.alternative_widget.currentTextChanged.connect(self.alternativeValueChanged.emit)
    
    def _on_toggle_changed(self, checked: bool):
        """Maneja el cambio del toggle"""
        self.location_selector.setEnabled(checked)
        if self.alternative_widget:
            self.alternative_widget.setEnabled(not checked)
        
        self.enabledChanged.emit(checked)
        
        # Emitir valor actual
        if checked:
            self.locationChanged.emit(self.location_selector.getCurrentLocation())
        elif self.alternative_widget:
            # Emitir valor del widget alternativo
            if hasattr(self.alternative_widget, 'currentText'):
                self.alternativeValueChanged.emit(self.alternative_widget.currentText())
            elif hasattr(self.alternative_widget, 'value'):
                self.alternativeValueChanged.emit(self.alternative_widget.value())
    
    def isLocationEnabled(self) -> bool:
        """Retorna si el selector de ubicación está habilitado"""
        return self.enable_checkbox.isChecked()
    
    def setLocationEnabled(self, enabled: bool):
        """Establece si el selector de ubicación está habilitado"""
        self.enable_checkbox.setChecked(enabled)
    
    def getCurrentLocation(self) -> Dict[str, str]:
        """Obtiene la ubicación actual (solo si está habilitado)"""
        if self.isLocationEnabled():
            return self.location_selector.getCurrentLocation()
        return {}
    
    def setCurrentLocation(self, location: Dict[str, str]):
        """Establece la ubicación actual"""
        self.location_selector.setCurrentLocation(location)


class CountryLocationSelector(HierarchicalLocationSelector):
    """Selector de ubicación específico para países con jerarquía estándar"""
    
    def __init__(self, country_data: pd.DataFrame, country_code: str = 'PE', parent=None):
        """
        Inicializa selector para países específicos
        
        Parameters
        ----------
        country_data : pd.DataFrame
            DataFrame con datos de ubicaciones del país
        country_code : str
            Código del país ('PE' para Perú, 'BO' para Bolivia, etc.)
        """
        # Configurar jerarquías por país
        hierarchies = {
            'PE': ['DEPARTAMENTO', 'PROVINCIA', 'DISTRITO'],
            'BO': ['DEPARTAMENTO', 'PROVINCIA', 'MUNICIPIO'],
            'CO': ['DEPARTAMENTO', 'MUNICIPIO'],
            'EC': ['PROVINCIA', 'CANTON', 'PARROQUIA'],
            'US': ['STATE', 'COUNTY', 'CITY'],
            'MX': ['ESTADO', 'MUNICIPIO']
        }
        
        database = LocationDatabase(country_data)
        hierarchy = hierarchies.get(country_code.upper(), ['LEVEL1', 'LEVEL2', 'LEVEL3'])
        
        # Usar solo las columnas que existen en los datos
        available_hierarchy = [col for col in hierarchy if col in country_data.columns]
        database.set_hierarchy(available_hierarchy)
        
        super().__init__(database, parent)


# Funciones de utilidad
def create_simple_location_selector(locations_dict: Dict[str, List[str]], 
                                   hierarchy_names: List[str]) -> HierarchicalLocationSelector:
    """
    Crea un selector simple desde un diccionario anidado
    
    Parameters
    ----------
    locations_dict : Dict[str, List[str]]
        Diccionario con estructura jerárquica
    hierarchy_names : List[str]
        Nombres de los niveles de jerarquía
    
    Returns
    -------
    HierarchicalLocationSelector
        Selector configurado
    """
    # Convertir diccionario a DataFrame
    rows = []
    for level1, level2_list in locations_dict.items():
        if isinstance(level2_list, list):
            for level2 in level2_list:
                row = {hierarchy_names[0]: level1}
                if len(hierarchy_names) > 1:
                    row[hierarchy_names[1]] = level2
                rows.append(row)
    
    df = pd.DataFrame(rows)
    database = LocationDatabase(df)
    database.set_hierarchy(hierarchy_names)
    
    return HierarchicalLocationSelector(database)


def create_csv_location_selector(csv_path: str, 
                                hierarchy_columns: List[str]) -> HierarchicalLocationSelector:
    """
    Crea un selector desde un archivo CSV
    
    Parameters
    ----------
    csv_path : str
        Ruta al archivo CSV
    hierarchy_columns : List[str]
        Columnas que forman la jerarquía
    
    Returns
    -------
    HierarchicalLocationSelector
        Selector configurado
    """
    database = LocationDatabase(csv_path)
    database.set_hierarchy(hierarchy_columns)
    
    return HierarchicalLocationSelector(database)