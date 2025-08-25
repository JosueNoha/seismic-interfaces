"""
Selectores de sistemas estructurales reutilizables para interfaces sísmicas
Widgets genéricos para selección de sistemas estructurales por dirección
"""

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                            QComboBox, QGroupBox, QFormLayout, QPushButton,
                            QTableWidget, QTableWidgetItem, QHeaderView,
                            QSizePolicy, QCheckBox, QSpinBox, QDoubleSpinBox)
from PyQt5.QtCore import pyqtSignal, Qt, QObject
from typing import Dict, List, Optional, Any, Union, Tuple
import pandas as pd
from .parameter_widgets import ParameterGroup, LabeledParameter, ParameterComboBox


class StructuralSystemDatabase:
    """Base de datos genérica para sistemas estructurales"""
    
    def __init__(self, systems_data: Union[Dict, List, pd.DataFrame, None] = None):
        """
        Inicializa la base de datos de sistemas estructurales
        
        Parameters
        ----------
        systems_data : Union[Dict, List, pd.DataFrame, None]
            Datos de sistemas estructurales
        """
        self.systems = {}  # {system_id: {name, category, properties}}
        self.categories = {}  # {category: [system_ids]}
        
        if systems_data is not None:
            self.load_systems(systems_data)
    
    def load_systems(self, systems_data: Union[Dict, List, pd.DataFrame]):
        """Carga sistemas desde diferentes fuentes"""
        if isinstance(systems_data, dict):
            self._load_from_dict(systems_data)
        elif isinstance(systems_data, list):
            self._load_from_list(systems_data)
        elif isinstance(systems_data, pd.DataFrame):
            self._load_from_dataframe(systems_data)
        else:
            raise TypeError("Tipo de datos no soportado")
    
    def _load_from_dict(self, data: Dict):
        """Carga desde diccionario estructurado"""
        # Formato: {category: {system_name: properties}}
        for category, systems in data.items():
            self.categories[category] = []
            for system_name, properties in systems.items():
                system_id = f"{category}_{len(self.systems)}"
                self.systems[system_id] = {
                    'name': system_name,
                    'category': category,
                    'properties': properties if isinstance(properties, dict) else {}
                }
                self.categories[category].append(system_id)
    
    def _load_from_list(self, data: List):
        """Carga desde lista simple de nombres"""
        category = "General"
        self.categories[category] = []
        
        for i, system_name in enumerate(data):
            system_id = f"system_{i}"
            self.systems[system_id] = {
                'name': system_name,
                'category': category,
                'properties': {}
            }
            self.categories[category].append(system_id)
    
    def _load_from_dataframe(self, df: pd.DataFrame):
        """Carga desde DataFrame"""
        # Espera columnas: 'name', 'category' (opcional), otras propiedades
        for index, row in df.iterrows():
            system_id = f"df_system_{index}"
            category = row.get('category', 'General')
            
            # Preparar propiedades (todas las columnas excepto name y category)
            properties = {}
            for col, val in row.items():
                if col not in ['name', 'category']:
                    properties[col] = val
            
            self.systems[system_id] = {
                'name': row['name'],
                'category': category,
                'properties': properties
            }
            
            # Agregar a categoría
            if category not in self.categories:
                self.categories[category] = []
            self.categories[category].append(system_id)
    
    def get_system_names(self, category: str = None) -> List[str]:
        """Obtiene lista de nombres de sistemas"""
        if category and category in self.categories:
            system_ids = self.categories[category]
            return [self.systems[sid]['name'] for sid in system_ids]
        else:
            return [system['name'] for system in self.systems.values()]
    
    def get_categories(self) -> List[str]:
        """Obtiene lista de categorías disponibles"""
        return list(self.categories.keys())
    
    def get_system_by_name(self, name: str) -> Optional[Dict]:
        """Obtiene sistema por nombre"""
        for system in self.systems.values():
            if system['name'] == name:
                return system
        return None
    
    def get_system_property(self, name: str, property_name: str) -> Any:
        """Obtiene propiedad específica de un sistema"""
        system = self.get_system_by_name(name)
        if system and 'properties' in system:
            return system['properties'].get(property_name, None)
        return None
    
    def get_all_properties(self, name: str) -> Dict[str, Any]:
        """Obtiene todas las propiedades de un sistema"""
        system = self.get_system_by_name(name)
        if system:
            return system.get('properties', {})
        return {}


class StructuralSystemSelector(QWidget):
    """Selector básico de sistema estructural"""
    
    systemChanged = pyqtSignal(str)  # Emite nombre del sistema seleccionado
    propertiesChanged = pyqtSignal(dict)  # Emite propiedades del sistema
    
    def __init__(self, database: StructuralSystemDatabase, 
                 label: str = "Sistema Estructural:", parent=None):
        super().__init__(parent)
        self.database = database
        self.label_text = label
        self.current_system = None
        
        self.setupUI()
        self.connectSignals()
    
    def setupUI(self):
        """Configura la interfaz del selector"""
        layout = QHBoxLayout(self)
        
        # Etiqueta
        self.label = QLabel(self.label_text)
        layout.addWidget(self.label)
        
        # ComboBox
        self.combo = ParameterComboBox()
        self.combo.setMinimumWidth(300)
        layout.addWidget(self.combo)
        
        # Cargar sistemas
        self.updateSystems()
        
        layout.addStretch()
    
    def connectSignals(self):
        """Conecta las señales"""
        self.combo.currentTextChanged.connect(self._on_system_changed)
    
    def updateSystems(self, category: str = None):
        """Actualiza la lista de sistemas disponibles"""
        systems = self.database.get_system_names(category)
        self.combo.setOptions(systems)
    
    def _on_system_changed(self, system_name: str):
        """Maneja el cambio de sistema"""
        if system_name and system_name != self.current_system:
            self.current_system = system_name
            self.systemChanged.emit(system_name)
            
            # Emitir propiedades
            properties = self.database.get_all_properties(system_name)
            self.propertiesChanged.emit(properties)
    
    def getCurrentSystem(self) -> Optional[str]:
        """Obtiene el sistema actualmente seleccionado"""
        return self.current_system
    
    def setCurrentSystem(self, system_name: str):
        """Establece el sistema actual"""
        if system_name in self.database.get_system_names():
            self.combo.setCurrentText(system_name)
    
    def getSystemProperty(self, property_name: str) -> Any:
        """Obtiene propiedad del sistema actual"""
        if self.current_system:
            return self.database.get_system_property(self.current_system, property_name)
        return None


class DualDirectionSystemSelector(QWidget):
    """Selector de sistemas estructurales para dos direcciones (X e Y)"""
    
    systemsChanged = pyqtSignal(dict)  # Emite {'X': system_x, 'Y': system_y}
    propertiesChanged = pyqtSignal(dict)  # Emite {'X': props_x, 'Y': props_y}
    
    def __init__(self, database: StructuralSystemDatabase, parent=None):
        super().__init__(parent)
        self.database = database
        self.current_systems = {'X': None, 'Y': None}
        
        self.setupUI()
        self.connectSignals()
    
    def setupUI(self):
        """Configura la interfaz para ambas direcciones"""
        layout = QVBoxLayout(self)
        
        # Grupo de sistemas estructurales
        self.systems_group = QGroupBox("Sistemas Estructurales")
        systems_layout = QFormLayout(self.systems_group)
        
        # Selector para dirección X
        self.selector_x = StructuralSystemSelector(
            self.database, "Sistema Estructural X:"
        )
        systems_layout.addRow(self.selector_x)
        
        # Selector para dirección Y  
        self.selector_y = StructuralSystemSelector(
            self.database, "Sistema Estructural Y:"
        )
        systems_layout.addRow(self.selector_y)
        
        # Botón para sincronizar (mismo sistema en ambas direcciones)
        sync_layout = QHBoxLayout()
        self.sync_button = QPushButton("Usar mismo sistema en X e Y")
        self.sync_button.setCheckable(True)
        sync_layout.addWidget(self.sync_button)
        sync_layout.addStretch()
        systems_layout.addRow(sync_layout)
        
        layout.addWidget(self.systems_group)
    
    def connectSignals(self):
        """Conecta las señales"""
        self.selector_x.systemChanged.connect(lambda system: self._on_system_changed('X', system))
        self.selector_y.systemChanged.connect(lambda system: self._on_system_changed('Y', system))
        self.sync_button.toggled.connect(self._on_sync_toggled)
    
    def _on_system_changed(self, direction: str, system_name: str):
        """Maneja el cambio de sistema en una dirección"""
        self.current_systems[direction] = system_name
        
        # Si está sincronizado, actualizar la otra dirección
        if self.sync_button.isChecked():
            other_direction = 'Y' if direction == 'X' else 'X'
            other_selector = self.selector_y if direction == 'X' else self.selector_x
            other_selector.setCurrentSystem(system_name)
            self.current_systems[other_direction] = system_name
        
        # Emitir señales
        self.systemsChanged.emit(self.current_systems.copy())
        self._emit_properties()
    
    def _on_sync_toggled(self, checked: bool):
        """Maneja la activación/desactivación de sincronización"""
        if checked:
            # Sincronizar Y con X
            system_x = self.current_systems['X']
            if system_x:
                self.selector_y.setCurrentSystem(system_x)
                self.current_systems['Y'] = system_x
                self.systemsChanged.emit(self.current_systems.copy())
                self._emit_properties()
        
        # Habilitar/deshabilitar selector Y
        self.selector_y.setEnabled(not checked)
    
    def _emit_properties(self):
        """Emite las propiedades de ambos sistemas"""
        properties = {}
        for direction, system in self.current_systems.items():
            if system:
                properties[direction] = self.database.get_all_properties(system)
            else:
                properties[direction] = {}
        
        self.propertiesChanged.emit(properties)
    
    def getCurrentSystems(self) -> Dict[str, Optional[str]]:
        """Obtiene los sistemas actualmente seleccionados"""
        return self.current_systems.copy()
    
    def setCurrentSystems(self, systems: Dict[str, str]):
        """Establece los sistemas actuales"""
        if 'X' in systems:
            self.selector_x.setCurrentSystem(systems['X'])
        if 'Y' in systems:
            self.selector_y.setCurrentSystem(systems['Y'])
    
    def getSystemProperty(self, direction: str, property_name: str) -> Any:
        """Obtiene propiedad de un sistema específico"""
        system = self.current_systems.get(direction)
        if system:
            return self.database.get_system_property(system, property_name)
        return None


class SystemPropertiesWidget(QWidget):
    """Widget que muestra propiedades de sistemas estructurales"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_properties = {}
        self.setupUI()
    
    def setupUI(self):
        """Configura la interfaz de propiedades"""
        layout = QVBoxLayout(self)
        
        # Tabla de propiedades
        self.properties_table = QTableWidget()
        self.properties_table.setColumnCount(3)
        self.properties_table.setHorizontalHeaderLabels(['Propiedad', 'Dirección X', 'Dirección Y'])
        self.properties_table.horizontalHeader().setStretchLastSection(True)
        
        layout.addWidget(self.properties_table)
    
    def updateProperties(self, properties: Dict[str, Dict[str, Any]]):
        """
        Actualiza la tabla de propiedades
        
        Parameters
        ----------
        properties : Dict[str, Dict[str, Any]]
            Propiedades por dirección: {'X': {...}, 'Y': {...}}
        """
        self.current_properties = properties
        
        # Obtener todas las propiedades únicas
        all_props = set()
        for direction_props in properties.values():
            all_props.update(direction_props.keys())
        
        # Configurar tabla
        self.properties_table.setRowCount(len(all_props))
        
        # Llenar tabla
        for row, prop_name in enumerate(sorted(all_props)):
            # Nombre de la propiedad
            self.properties_table.setItem(row, 0, QTableWidgetItem(prop_name))
            
            # Valor en X
            x_value = properties.get('X', {}).get(prop_name, '-')
            self.properties_table.setItem(row, 1, QTableWidgetItem(str(x_value)))
            
            # Valor en Y
            y_value = properties.get('Y', {}).get(prop_name, '-')
            self.properties_table.setItem(row, 2, QTableWidgetItem(str(y_value)))
        
        # Ajustar columnas
        self.properties_table.resizeColumnsToContents()


class CategoryBasedSystemSelector(QWidget):
    """Selector de sistemas organizados por categorías"""
    
    systemChanged = pyqtSignal(str)
    categoryChanged = pyqtSignal(str)
    propertiesChanged = pyqtSignal(dict)
    
    def __init__(self, database: StructuralSystemDatabase, parent=None):
        super().__init__(parent)
        self.database = database
        self.current_category = None
        self.current_system = None
        
        self.setupUI()
        self.connectSignals()
    
    def setupUI(self):
        """Configura la interfaz con categorías"""
        layout = QVBoxLayout(self)
        
        # Grupo de selección
        selection_group = QGroupBox("Selección de Sistema Estructural")
        selection_layout = QFormLayout(selection_group)
        
        # Selector de categoría
        self.category_combo = ParameterComboBox()
        categories = self.database.get_categories()
        self.category_combo.addItems(categories)
        selection_layout.addRow("Categoría:", self.category_combo)
        
        # Selector de sistema
        self.system_combo = ParameterComboBox()
        selection_layout.addRow("Sistema:", self.system_combo)
        
        layout.addWidget(selection_group)
        
        # Inicializar primera categoría
        if categories:
            self.category_combo.setCurrentText(categories[0])
            self._update_systems(categories[0])
    
    def connectSignals(self):
        """Conecta las señales"""
        self.category_combo.currentTextChanged.connect(self._on_category_changed)
        self.system_combo.currentTextChanged.connect(self._on_system_changed)
    
    def _on_category_changed(self, category: str):
        """Maneja el cambio de categoría"""
        self.current_category = category
        self._update_systems(category)
        self.categoryChanged.emit(category)
    
    def _update_systems(self, category: str):
        """Actualiza los sistemas disponibles para la categoría"""
        systems = self.database.get_system_names(category)
        self.system_combo.blockSignals(True)
        self.system_combo.clear()
        self.system_combo.addItems(systems)
        self.system_combo.blockSignals(False)
        
        # Seleccionar primer sistema si existe
        if systems:
            self.system_combo.setCurrentText(systems[0])
    
    def _on_system_changed(self, system_name: str):
        """Maneja el cambio de sistema"""
        if system_name != self.current_system:
            self.current_system = system_name
            self.systemChanged.emit(system_name)
            
            # Emitir propiedades
            properties = self.database.get_all_properties(system_name)
            self.propertiesChanged.emit(properties)
    
    def getCurrentSystem(self) -> Optional[str]:
        """Obtiene el sistema actual"""
        return self.current_system
    
    def getCurrentCategory(self) -> Optional[str]:
        """Obtiene la categoría actual"""
        return self.current_category


# Funciones de utilidad para crear bases de datos comunes
def create_basic_structural_systems() -> StructuralSystemDatabase:
    """Crea base de datos básica de sistemas estructurales"""
    systems = [
        "Pórticos de Concreto Armado",
        "Sistema Dual de Concreto Armado", 
        "Muros Estructurales de Concreto Armado",
        "Pórticos de Acero",
        "Sistema Dual de Acero",
        "Muros de Albañilería Confinada",
        "Pórticos de Madera"
    ]
    
    return StructuralSystemDatabase(systems)


def create_detailed_structural_systems() -> StructuralSystemDatabase:
    """Crea base de datos detallada con categorías y propiedades"""
    systems_data = {
        "Acero": {
            "Pórticos Especiales de Acero Resistentes a Momentos": {"R": 8, "code": "SMF"},
            "Pórticos Intermedios de Acero Resistentes a Momentos": {"R": 5, "code": "IMF"},
            "Pórticos Ordinarios de Acero Resistentes a Momentos": {"R": 4, "code": "OMF"},
            "Pórticos Especiales de Acero Concénticamente Arriostrados": {"R": 7, "code": "SCBF"},
            "Pórticos Ordinarios de Acero Concénticamente Arriostrados": {"R": 4, "code": "OCBF"},
            "Pórticos de Acero Excéntricamente Arriostrados": {"R": 8, "code": "EBF"}
        },
        "Concreto Armado": {
            "Pórticos de Concreto Armado": {"R": 8, "drift_limit": 0.007},
            "Sistema Dual de Concreto Armado": {"R": 7, "drift_limit": 0.007},
            "Muros Estructurales de Concreto Armado": {"R": 6, "drift_limit": 0.007},
            "Muros de Ductilidad Limitada de Concreto Armado": {"R": 4, "drift_limit": 0.005}
        },
        "Otros": {
            "Albañilería Armada o Confinada": {"R": 3, "drift_limit": 0.005},
            "Madera": {"R": 7, "drift_limit": 0.010}
        }
    }
    
    return StructuralSystemDatabase(systems_data)


def create_systems_from_dataframe(df: pd.DataFrame) -> StructuralSystemDatabase:
    """Crea base de datos desde DataFrame"""
    return StructuralSystemDatabase(df)