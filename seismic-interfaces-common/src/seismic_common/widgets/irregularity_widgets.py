"""
Widgets de irregularidades reutilizables para interfaces sísmicas
Widgets genéricos para selección y manejo de irregularidades estructurales
"""

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                            QCheckBox, QGroupBox, QFormLayout, QPushButton,
                            QTableWidget, QTableWidgetItem, QHeaderView,
                            QScrollArea, QFrame, QGridLayout, QTabWidget,
                            QSizePolicy, QSpinBox, QDoubleSpinBox, QTextEdit)
from PyQt5.QtCore import pyqtSignal, Qt, QObject
from PyQt5.QtGui import QFont, QPalette
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
import pandas as pd
from .parameter_widgets import ParameterGroup, LabeledParameter, ParameterCheckBox


class IrregularityDatabase:
    """Base de datos genérica para tipos de irregularidades"""
    
    def __init__(self, irregularities_data: Union[Dict, List, pd.DataFrame, None] = None):
        """
        Inicializa la base de datos de irregularidades
        
        Parameters
        ----------
        irregularities_data : Union[Dict, List, pd.DataFrame, None]
            Datos de irregularidades organizados por tipo
        """
        self.irregularities = {}  # {category: {irr_id: {name, description, properties}}}
        
        if irregularities_data is not None:
            self.load_irregularities(irregularities_data)
    
    def load_irregularities(self, data: Union[Dict, List, pd.DataFrame]):
        """Carga irregularidades desde diferentes fuentes"""
        if isinstance(data, dict):
            self._load_from_dict(data)
        elif isinstance(data, list):
            self._load_from_list(data)
        elif isinstance(data, pd.DataFrame):
            self._load_from_dataframe(data)
        else:
            raise TypeError("Tipo de datos no soportado")
    
    def _load_from_dict(self, data: Dict):
        """
        Carga desde diccionario estructurado
        Formato: {category: {irr_name: {description, properties}}}
        """
        for category, irregularities in data.items():
            self.irregularities[category] = {}
            for irr_name, irr_data in irregularities.items():
                irr_id = f"{category}_{len(self.irregularities[category])}"
                
                if isinstance(irr_data, dict):
                    description = irr_data.get('description', '')
                    properties = irr_data.get('properties', {})
                elif isinstance(irr_data, str):
                    description = irr_data
                    properties = {}
                else:
                    description = str(irr_data)
                    properties = {}
                
                self.irregularities[category][irr_id] = {
                    'name': irr_name,
                    'description': description,
                    'properties': properties
                }
    
    def _load_from_list(self, data: List):
        """Carga desde lista simple de nombres"""
        category = "General"
        self.irregularities[category] = {}
        
        for i, irr_name in enumerate(data):
            irr_id = f"irr_{i}"
            self.irregularities[category][irr_id] = {
                'name': irr_name,
                'description': '',
                'properties': {}
            }
    
    def _load_from_dataframe(self, df: pd.DataFrame):
        """Carga desde DataFrame con columnas: category, name, description"""
        for index, row in df.iterrows():
            category = row.get('category', 'General')
            name = row['name']
            description = row.get('description', '')
            
            if category not in self.irregularities:
                self.irregularities[category] = {}
            
            irr_id = f"df_irr_{category}_{len(self.irregularities[category])}"
            
            # Propiedades adicionales
            properties = {}
            for col, val in row.items():
                if col not in ['category', 'name', 'description']:
                    properties[col] = val
            
            self.irregularities[category][irr_id] = {
                'name': name,
                'description': description,
                'properties': properties
            }
    
    def get_categories(self) -> List[str]:
        """Obtiene las categorías de irregularidades"""
        return list(self.irregularities.keys())
    
    def get_irregularities(self, category: str) -> Dict[str, Dict]:
        """Obtiene irregularidades de una categoría específica"""
        return self.irregularities.get(category, {})
    
    def get_irregularity_names(self, category: str) -> List[str]:
        """Obtiene nombres de irregularidades de una categoría"""
        irregularities = self.get_irregularities(category)
        return [irr['name'] for irr in irregularities.values()]


class IrregularityCheckBox(QCheckBox):
    """CheckBox especializado para irregularidades con información adicional"""
    
    def __init__(self, irregularity_name: str, description: str = '', 
                 properties: Dict = None, parent=None):
        super().__init__(irregularity_name, parent)
        self.irregularity_name = irregularity_name
        self.description = description
        self.properties = properties or {}
        
        # Configurar tooltip si hay descripción
        if description:
            self.setToolTip(description)
        
        # Estilo para irregularidades activas
        self.stateChanged.connect(self._on_state_changed)
    
    def _on_state_changed(self, state):
        """Cambia el estilo según el estado"""
        if state == Qt.Checked:
            self.setStyleSheet("""
                QCheckBox {
                    font-weight: bold;
                    color: #D32F2F;
                }
            """)
        else:
            self.setStyleSheet("")


class IrregularityCategoryWidget(QWidget):
    """Widget para una categoría específica de irregularidades"""
    
    irregularitiesChanged = pyqtSignal(dict)  # Emite {irr_name: checked}
    
    def __init__(self, category_name: str, irregularities: Dict[str, Dict], 
                 columns: int = 1, parent=None):
        super().__init__(parent)
        self.category_name = category_name
        self.irregularities_data = irregularities
        self.columns = columns
        self.checkboxes = {}
        
        self.setupUI()
        self.connectSignals()
    
    def setupUI(self):
        """Configura la interfaz de la categoría"""
        layout = QVBoxLayout(self)
        
        # Título de la categoría
        title = QLabel(self.category_name)
        title.setFont(QFont("", 10, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Grid de checkboxes
        if self.columns > 1:
            grid_widget = QWidget()
            grid_layout = QGridLayout(grid_widget)
            
            irregularities = list(self.irregularities_data.values())
            row, col = 0, 0
            
            for irr_data in irregularities:
                checkbox = IrregularityCheckBox(
                    irr_data['name'],
                    irr_data.get('description', ''),
                    irr_data.get('properties', {})
                )
                self.checkboxes[irr_data['name']] = checkbox
                
                grid_layout.addWidget(checkbox, row, col)
                
                col += 1
                if col >= self.columns:
                    col = 0
                    row += 1
            
            layout.addWidget(grid_widget)
        
        else:
            # Layout vertical simple
            for irr_data in self.irregularities_data.values():
                checkbox = IrregularityCheckBox(
                    irr_data['name'],
                    irr_data.get('description', ''),
                    irr_data.get('properties', {})
                )
                self.checkboxes[irr_data['name']] = checkbox
                layout.addWidget(checkbox)
        
        layout.addStretch()
    
    def connectSignals(self):
        """Conecta las señales de los checkboxes"""
        for checkbox in self.checkboxes.values():
            checkbox.stateChanged.connect(self._on_irregularity_changed)
    
    def _on_irregularity_changed(self):
        """Maneja el cambio en cualquier irregularidad"""
        current_state = self.getSelectedIrregularities()
        self.irregularitiesChanged.emit(current_state)
    
    def getSelectedIrregularities(self) -> Dict[str, bool]:
        """Obtiene el estado de todas las irregularidades"""
        return {name: checkbox.isChecked() 
                for name, checkbox in self.checkboxes.items()}
    
    def setSelectedIrregularities(self, irregularities: Dict[str, bool]):
        """Establece el estado de las irregularidades"""
        for name, checked in irregularities.items():
            if name in self.checkboxes:
                self.checkboxes[name].setChecked(checked)
    
    def hasSelectedIrregularities(self) -> bool:
        """Verifica si hay irregularidades seleccionadas"""
        return any(self.getSelectedIrregularities().values())


class IrregularitySelector(QWidget):
    """Selector completo de irregularidades con múltiples categorías"""
    
    irregularitiesChanged = pyqtSignal(dict)  # Emite {category: {irr_name: checked}}
    irregularityFactorChanged = pyqtSignal(dict)  # Emite factores calculados
    
    def __init__(self, database: IrregularityDatabase, 
                 layout_type: str = 'tabs',  # 'tabs' o 'horizontal'
                 columns_per_category: int = 1,
                 parent=None):
        super().__init__(parent)
        self.database = database
        self.layout_type = layout_type
        self.columns_per_category = columns_per_category
        self.category_widgets = {}
        self.current_irregularities = {}
        
        self.setupUI()
        self.connectSignals()
    
    def setupUI(self):
        """Configura la interfaz principal"""
        layout = QVBoxLayout(self)
        
        # Título principal
        title = QLabel("Irregularidades Estructurales")
        title.setFont(QFont("", 12, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Contenedor según tipo de layout
        if self.layout_type == 'tabs':
            self._setup_tabs_layout(layout)
        else:
            self._setup_horizontal_layout(layout)
        
        # Resumen de irregularidades seleccionadas
        self.summary_widget = IrregularitySummaryWidget()
        layout.addWidget(self.summary_widget)
    
    def _setup_tabs_layout(self, layout):
        """Configura layout con tabs para cada categoría"""
        self.tab_widget = QTabWidget()
        
        for category in self.database.get_categories():
            irregularities = self.database.get_irregularities(category)
            
            if irregularities:
                category_widget = IrregularityCategoryWidget(
                    category, irregularities, self.columns_per_category
                )
                self.category_widgets[category] = category_widget
                self.tab_widget.addTab(category_widget, category)
        
        layout.addWidget(self.tab_widget)
    
    def _setup_horizontal_layout(self, layout):
        """Configura layout horizontal con todas las categorías"""
        scroll_area = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QHBoxLayout(scroll_widget)
        
        for category in self.database.get_categories():
            irregularities = self.database.get_irregularities(category)
            
            if irregularities:
                # Crear grupo para la categoría
                group_box = QGroupBox(category)
                group_layout = QVBoxLayout(group_box)
                
                category_widget = IrregularityCategoryWidget(
                    category, irregularities, self.columns_per_category
                )
                category_widget.setParent(None)  # Remover layout automático
                
                group_layout.addWidget(category_widget)
                self.category_widgets[category] = category_widget
                scroll_layout.addWidget(group_box)
        
        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)
        layout.addWidget(scroll_area)
    
    def connectSignals(self):
        """Conecta las señales de todas las categorías"""
        for category_widget in self.category_widgets.values():
            category_widget.irregularitiesChanged.connect(self._on_category_changed)
    
    def _on_category_changed(self, category_irregularities):
        """Maneja el cambio en una categoría"""
        # Actualizar irregularidades actuales
        self._update_current_irregularities()
        
        # Emitir señales
        self.irregularitiesChanged.emit(self.current_irregularities.copy())
        
        # Calcular y emitir factores
        factors = self._calculate_irregularity_factors()
        self.irregularityFactorChanged.emit(factors)
        
        # Actualizar resumen
        self.summary_widget.updateSummary(self.current_irregularities)
    
    def _update_current_irregularities(self):
        """Actualiza el estado actual de todas las irregularidades"""
        self.current_irregularities = {}
        
        for category, widget in self.category_widgets.items():
            self.current_irregularities[category] = widget.getSelectedIrregularities()
    
    def _calculate_irregularity_factors(self) -> Dict[str, float]:
        """
        Calcula factores de irregularidad según normativas
        (Implementación base, debe ser sobrescrita por normativas específicas)
        """
        factors = {'Ia': 1.0, 'Ip': 1.0}  # Factores de altura y planta
        
        # Verificar si hay irregularidades en altura
        height_irregularities = self.current_irregularities.get('Altura', {})
        if any(height_irregularities.values()):
            factors['Ia'] = 0.75  # Valor típico
        
        # Verificar si hay irregularidades en planta
        plan_irregularities = self.current_irregularities.get('Planta', {})
        if any(plan_irregularities.values()):
            factors['Ip'] = 0.75  # Valor típico
        
        return factors
    
    def getAllIrregularities(self) -> Dict[str, Dict[str, bool]]:
        """Obtiene todas las irregularidades seleccionadas"""
        return self.current_irregularities.copy()
    
    def setAllIrregularities(self, irregularities: Dict[str, Dict[str, bool]]):
        """Establece todas las irregularidades"""
        for category, category_irregularities in irregularities.items():
            if category in self.category_widgets:
                self.category_widgets[category].setSelectedIrregularities(category_irregularities)
        
        self._update_current_irregularities()
        self.summary_widget.updateSummary(self.current_irregularities)
    
    def hasAnyIrregularities(self) -> bool:
        """Verifica si hay alguna irregularidad seleccionada"""
        for category_irregularities in self.current_irregularities.values():
            if any(category_irregularities.values()):
                return True
        return False
    
    def getIrregularityCount(self) -> Dict[str, int]:
        """Obtiene el conteo de irregularidades por categoría"""
        count = {}
        for category, category_irregularities in self.current_irregularities.items():
            count[category] = sum(1 for checked in category_irregularities.values() if checked)
        return count


class IrregularitySummaryWidget(QWidget):
    """Widget que muestra un resumen de las irregularidades seleccionadas"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUI()
    
    def setupUI(self):
        """Configura la interfaz del resumen"""
        layout = QVBoxLayout(self)
        
        # Título
        title = QLabel("Resumen de Irregularidades")
        title.setFont(QFont("", 10, QFont.Bold))
        layout.addWidget(title)
        
        # Área de texto para el resumen
        self.summary_text = QTextEdit()
        self.summary_text.setMaximumHeight(80)
        self.summary_text.setReadOnly(True)
        layout.addWidget(self.summary_text)
    
    def updateSummary(self, irregularities: Dict[str, Dict[str, bool]]):
        """Actualiza el resumen de irregularidades"""
        summary_lines = []
        total_irregularities = 0
        
        for category, category_irregularities in irregularities.items():
            selected = [name for name, checked in category_irregularities.items() if checked]
            if selected:
                total_irregularities += len(selected)
                summary_lines.append(f"• {category}: {', '.join(selected)}")
        
        if summary_lines:
            summary_text = f"Total de irregularidades: {total_irregularities}\n\n"
            summary_text += "\n".join(summary_lines)
        else:
            summary_text = "No se han seleccionado irregularidades.\nLa estructura se considera regular."
        
        self.summary_text.setText(summary_text)


class IrregularityAnalysisWidget(QWidget):
    """Widget para análisis detallado de irregularidades con datos"""
    
    analysisRequested = pyqtSignal(str, str)  # irregularity_type, direction
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.analysis_data = {}
        self.setupUI()
    
    def setupUI(self):
        """Configura la interfaz de análisis"""
        layout = QVBoxLayout(self)
        
        # Título
        title = QLabel("Análisis de Irregularidades")
        title.setFont(QFont("", 12, QFont.Bold))
        layout.addWidget(title)
        
        # Botones de análisis
        buttons_layout = QGridLayout()
        
        self.analysis_buttons = {
            'rigidez': QPushButton("Análisis de Rigidez\n(Piso Blando)"),
            'torsion': QPushButton("Análisis Torsional"),
            'masa': QPushButton("Análisis de Masa"),
            'deriva': QPushButton("Verificación de Derivas")
        }
        
        row, col = 0, 0
        for analysis_type, button in self.analysis_buttons.items():
            button.clicked.connect(lambda checked, t=analysis_type: self._request_analysis(t))
            buttons_layout.addWidget(button, row, col)
            
            col += 1
            if col >= 2:
                col = 0
                row += 1
        
        layout.addLayout(buttons_layout)
        
        # Área de resultados
        self.results_area = QTextEdit()
        self.results_area.setReadOnly(True)
        self.results_area.setPlaceholderText("Los resultados del análisis aparecerán aquí...")
        layout.addWidget(self.results_area)
    
    def _request_analysis(self, analysis_type: str):
        """Solicita análisis específico"""
        # Por defecto analizar ambas direcciones
        for direction in ['X', 'Y']:
            self.analysisRequested.emit(analysis_type, direction)
    
    def setAnalysisData(self, analysis_type: str, direction: str, data: pd.DataFrame):
        """Establece datos de análisis"""
        key = f"{analysis_type}_{direction}"
        self.analysis_data[key] = data
        self._update_results_display()
    
    def _update_results_display(self):
        """Actualiza la visualización de resultados"""
        if not self.analysis_data:
            return
        
        results_text = "Resultados de Análisis:\n\n"
        
        for key, data in self.analysis_data.items():
            analysis_type, direction = key.rsplit('_', 1)
            results_text += f"=== {analysis_type.title()} - Dirección {direction} ===\n"
            
            if not data.empty:
                # Mostrar resumen de los datos
                results_text += f"Número de pisos analizados: {len(data)}\n"
                
                # Agregar información específica según tipo
                if 'is_regular' in data.columns:
                    irregular_count = data['is_regular'].str.contains('NO', na=False).sum()
                    if irregular_count > 0:
                        results_text += f"Pisos con irregularidades: {irregular_count}\n"
                    else:
                        results_text += "Todos los pisos son regulares\n"
            
            results_text += "\n"
        
        self.results_area.setText(results_text)


# Funciones de utilidad para crear bases de datos comunes
def create_basic_irregularities() -> IrregularityDatabase:
    """Crea base de datos básica de irregularidades"""
    irregularities = {
        "Altura": [
            "Irregularidad de Rigidez (Piso Blando)",
            "Irregularidad de Masa",
            "Irregularidad Geométrica Vertical",
            "Discontinuidad Vertical"
        ],
        "Planta": [
            "Irregularidad Torsional",
            "Esquinas Entrantes", 
            "Discontinuidad del Diafragma",
            "Sistemas No Paralelos"
        ]
    }
    
    return IrregularityDatabase(irregularities)


def create_detailed_irregularities() -> IrregularityDatabase:
    """Crea base de datos detallada con descripciones"""
    irregularities = {
        "Altura": {
            "Irregularidad de Rigidez (Piso Blando)": {
                "description": "Rigidez lateral menor al 70% del piso superior o 80% del promedio de tres pisos superiores",
                "properties": {"factor_limit": 0.7, "extreme_limit": 0.6}
            },
            "Irregularidad de Rigidez Extrema (Piso Blando)": {
                "description": "Rigidez lateral menor al 60% del piso superior o 70% del promedio de tres pisos superiores",
                "properties": {"factor_limit": 0.6, "extreme_limit": 0.5}
            },
            "Irregularidad de Masa": {
                "description": "Masa de un piso mayor a 1.5 veces la masa de un piso adyacente",
                "properties": {"factor_limit": 1.5}
            },
            "Irregularidad Geométrica Vertical": {
                "description": "Dimensión en planta del piso difiere en más del 30% del piso adyacente",
                "properties": {"factor_limit": 0.3}
            },
            "Discontinuidad Vertical": {
                "description": "Discontinuidad en el plano de elementos resistentes verticales",
                "properties": {}
            },
            "Discontinuidad Vertical Extrema": {
                "description": "Discontinuidad extrema en el plano de elementos resistentes",
                "properties": {}
            }
        },
        "Planta": {
            "Irregularidad Torsional": {
                "description": "Desplazamiento máximo mayor a 1.3 veces el promedio de extremos del entrepiso",
                "properties": {"factor_limit": 1.3}
            },
            "Irregularidad Torsional Extrema": {
                "description": "Desplazamiento máximo mayor a 1.5 veces el promedio de extremos del entrepiso", 
                "properties": {"factor_limit": 1.5}
            },
            "Esquinas Entrantes": {
                "description": "Proyecciones de la estructura en planta exceden 15% de la dimensión en esa dirección",
                "properties": {"factor_limit": 0.15}
            },
            "Discontinuidad del Diafragma": {
                "description": "Diafragma con discontinuidades abruptas o aberturas mayores al 50% del área bruta",
                "properties": {"factor_limit": 0.5}
            },
            "Sistemas No Paralelos": {
                "description": "Elementos resistentes no son paralelos a los ejes principales",
                "properties": {}
            }
        }
    }
    
    return IrregularityDatabase(irregularities)


def create_irregularities_from_dataframe(df: pd.DataFrame) -> IrregularityDatabase:
    """Crea base de datos desde DataFrame"""
    return IrregularityDatabase(df)