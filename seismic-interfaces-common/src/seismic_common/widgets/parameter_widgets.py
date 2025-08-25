"""
Widgets de parámetros reutilizables para interfaces sísmicas
Contiene solo widgets genéricos sin contenido específico de normativas
"""

from PyQt5.QtWidgets import (QWidget, QHBoxLayout, QVBoxLayout, QLabel, 
                            QComboBox, QLineEdit, QCheckBox, QPushButton, 
                            QSpinBox, QDoubleSpinBox, QGroupBox, QGridLayout,
                            QFrame, QSizePolicy)
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtGui import QFont, QDoubleValidator, QIntValidator
from typing import Dict, List, Optional, Any, Union


class ParameterLineEdit(QLineEdit):
    """LineEdit personalizado para parámetros numéricos"""
    
    valueChanged = pyqtSignal(float)
    
    def __init__(self, value: Union[int, float] = 0.0, 
                 decimals: int = 3, parent=None):
        super().__init__(parent)
        self.setValidator(QDoubleValidator())
        self.setValue(value)
        self.decimals = decimals
        self.editingFinished.connect(self._on_editing_finished)
    
    def setValue(self, value: Union[int, float]):
        """Establece el valor numérico"""
        self.setText(f"{float(value):.{self.decimals}f}")
    
    def value(self) -> float:
        """Retorna el valor numérico actual"""
        try:
            return float(self.text())
        except ValueError:
            return 0.0
    
    def _on_editing_finished(self):
        """Emite señal cuando se termina la edición"""
        self.valueChanged.emit(self.value())


class ParameterSpinBox(QSpinBox):
    """SpinBox personalizado para parámetros enteros"""
    
    def __init__(self, value: int = 0, minimum: int = 0, 
                 maximum: int = 999, parent=None):
        super().__init__(parent)
        self.setMinimum(minimum)
        self.setMaximum(maximum)
        self.setValue(value)


class ParameterDoubleSpinBox(QDoubleSpinBox):
    """DoubleSpinBox personalizado para parámetros decimales"""
    
    def __init__(self, value: float = 0.0, minimum: float = 0.0, 
                 maximum: float = 999.0, decimals: int = 3, 
                 step: float = 0.1, parent=None):
        super().__init__(parent)
        self.setMinimum(minimum)
        self.setMaximum(maximum)
        self.setValue(value)
        self.setDecimals(decimals)
        self.setSingleStep(step)


class ParameterComboBox(QComboBox):
    """ComboBox personalizado para selección de parámetros"""
    
    def __init__(self, options: List[str] = None, default_value: str = '', parent=None):
        super().__init__(parent)
        if options:
            self.addItems(options)
        if default_value and default_value in (options or []):
            self.setCurrentText(default_value)
    
    def setOptions(self, options: List[str], default_value: str = ''):
        """Actualiza las opciones del ComboBox"""
        self.clear()
        self.addItems(options)
        if default_value and default_value in options:
            self.setCurrentText(default_value)


class ParameterCheckBox(QCheckBox):
    """CheckBox personalizado para parámetros booleanos"""
    
    def __init__(self, text: str = '', checked: bool = False, parent=None):
        super().__init__(text, parent)
        self.setChecked(checked)


class LabeledParameter(QWidget):
    """Widget que combina etiqueta con control de parámetro"""
    
    valueChanged = pyqtSignal(object)
    
    def __init__(self, label_text: str, control_widget: QWidget, 
                 label_width: int = 150, parent=None):
        super().__init__(parent)
        self.control_widget = control_widget
        self._setup_ui(label_text, label_width)
        self._connect_signals()
    
    def _setup_ui(self, label_text: str, label_width: int):
        """Configura la interfaz del widget"""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Etiqueta
        self.label = QLabel(label_text)
        self.label.setMinimumWidth(label_width)
        self.label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        
        # Control
        layout.addWidget(self.label)
        layout.addWidget(self.control_widget)
        layout.addStretch()
    
    def _connect_signals(self):
        """Conecta las señales del widget de control"""
        if hasattr(self.control_widget, 'valueChanged'):
            self.control_widget.valueChanged.connect(self.valueChanged.emit)
        elif hasattr(self.control_widget, 'currentTextChanged'):
            self.control_widget.currentTextChanged.connect(self.valueChanged.emit)
        elif hasattr(self.control_widget, 'toggled'):
            self.control_widget.toggled.connect(self.valueChanged.emit)
        elif hasattr(self.control_widget, 'textChanged'):
            self.control_widget.textChanged.connect(self.valueChanged.emit)
    
    def value(self):
        """Retorna el valor del control"""
        if hasattr(self.control_widget, 'value'):
            return self.control_widget.value()
        elif hasattr(self.control_widget, 'currentText'):
            return self.control_widget.currentText()
        elif hasattr(self.control_widget, 'isChecked'):
            return self.control_widget.isChecked()
        elif hasattr(self.control_widget, 'text'):
            return self.control_widget.text()
        return None
    
    def setValue(self, value):
        """Establece el valor del control"""
        if hasattr(self.control_widget, 'setValue'):
            self.control_widget.setValue(value)
        elif hasattr(self.control_widget, 'setCurrentText'):
            self.control_widget.setCurrentText(value)
        elif hasattr(self.control_widget, 'setChecked'):
            self.control_widget.setChecked(value)
        elif hasattr(self.control_widget, 'setText'):
            self.control_widget.setText(str(value))
    
    def setEnabled(self, enabled: bool):
        """Habilita o deshabilita el control"""
        super().setEnabled(enabled)
        self.control_widget.setEnabled(enabled)
        self.label.setEnabled(enabled)


class ParameterGroup(QGroupBox):
    """Grupo de parámetros con título"""
    
    parametersChanged = pyqtSignal(dict)
    
    def __init__(self, title: str, parent=None):
        super().__init__(title, parent)
        self.parameters = {}
        self.layout = QVBoxLayout(self)
    
    def addParameter(self, name: str, parameter_widget: LabeledParameter):
        """Añade un parámetro al grupo"""
        self.parameters[name] = parameter_widget
        parameter_widget.valueChanged.connect(self._on_parameter_changed)
        self.layout.addWidget(parameter_widget)
    
    def removeParameter(self, name: str):
        """Remueve un parámetro del grupo"""
        if name in self.parameters:
            widget = self.parameters.pop(name)
            self.layout.removeWidget(widget)
            widget.deleteLater()
    
    def getParameter(self, name: str) -> Optional[LabeledParameter]:
        """Obtiene un parámetro por nombre"""
        return self.parameters.get(name)
    
    def getParameterValue(self, name: str):
        """Obtiene el valor de un parámetro por nombre"""
        param = self.getParameter(name)
        return param.value() if param else None
    
    def setParameterValue(self, name: str, value):
        """Establece el valor de un parámetro por nombre"""
        param = self.getParameter(name)
        if param:
            param.setValue(value)
    
    def getAllValues(self) -> Dict[str, Any]:
        """Obtiene todos los valores de los parámetros"""
        return {name: param.value() for name, param in self.parameters.items()}
    
    def setAllValues(self, values: Dict[str, Any]):
        """Establece todos los valores de los parámetros"""
        for name, value in values.items():
            self.setParameterValue(name, value)
    
    def clearParameters(self):
        """Limpia todos los parámetros del grupo"""
        for name in list(self.parameters.keys()):
            self.removeParameter(name)
    
    def _on_parameter_changed(self, value):
        """Maneja el cambio de cualquier parámetro en el grupo"""
        self.parametersChanged.emit(self.getAllValues())


class GridParameterGroup(QGroupBox):
    """Grupo de parámetros organizados en grid"""
    
    parametersChanged = pyqtSignal(dict)
    
    def __init__(self, title: str, columns: int = 2, parent=None):
        super().__init__(title, parent)
        self.parameters = {}
        self.columns = columns
        self.layout = QGridLayout(self)
        self.current_row = 0
        self.current_col = 0
    
    def addParameter(self, name: str, parameter_widget: LabeledParameter):
        """Añade un parámetro al grid"""
        self.parameters[name] = parameter_widget
        parameter_widget.valueChanged.connect(self._on_parameter_changed)
        
        # Añadir al grid
        self.layout.addWidget(parameter_widget, self.current_row, self.current_col)
        
        # Actualizar posición
        self.current_col += 1
        if self.current_col >= self.columns:
            self.current_col = 0
            self.current_row += 1
    
    def getParameter(self, name: str) -> Optional[LabeledParameter]:
        """Obtiene un parámetro por nombre"""
        return self.parameters.get(name)
    
    def getAllValues(self) -> Dict[str, Any]:
        """Obtiene todos los valores de los parámetros"""
        return {name: param.value() for name, param in self.parameters.items()}
    
    def setAllValues(self, values: Dict[str, Any]):
        """Establece todos los valores de los parámetros"""
        for name, value in values.items():
            if name in self.parameters:
                self.parameters[name].setValue(value)
    
    def _on_parameter_changed(self, value):
        """Maneja el cambio de cualquier parámetro en el grupo"""
        self.parametersChanged.emit(self.getAllValues())


class LoadSelectionWidget(QWidget):
    """Widget genérico para selección de cargas"""
    
    loadsChanged = pyqtSignal(dict)
    
    def __init__(self, load_cases: List[str] = None, parent=None):
        super().__init__(parent)
        self.load_cases = load_cases or []
        self.load_parameters = {}
        self.setupUI()
    
    def setupUI(self):
        """Configura la interfaz básica"""
        self.main_layout = QVBoxLayout(self)
        self.loads_group = ParameterGroup("Selección de Cargas")
        self.loads_group.parametersChanged.connect(self.loadsChanged.emit)
        self.main_layout.addWidget(self.loads_group)
    
    def addLoadParameter(self, name: str, label: str, default_value: str = ''):
        """Añade un parámetro de carga al widget"""
        combo = ParameterComboBox(self.load_cases, default_value)
        param = LabeledParameter(label, combo)
        self.loads_group.addParameter(name, param)
        self.load_parameters[name] = param
    
    def removeLoadParameter(self, name: str):
        """Remueve un parámetro de carga"""
        if name in self.load_parameters:
            self.loads_group.removeParameter(name)
            del self.load_parameters[name]
    
    def updateLoadCases(self, load_cases: List[str]):
        """Actualiza la lista de casos de carga disponibles"""
        self.load_cases = load_cases
        for param in self.load_parameters.values():
            if hasattr(param.control_widget, 'setOptions'):
                param.control_widget.setOptions(load_cases)
    
    def getAllLoads(self) -> Dict[str, str]:
        """Obtiene todas las cargas seleccionadas"""
        return self.loads_group.getAllValues()
    
    def setAllLoads(self, loads: Dict[str, str]):
        """Establece todas las cargas seleccionadas"""
        self.loads_group.setAllValues(loads)


class DynamicParameterWidget(QWidget):
    """Widget que permite agregar/quitar parámetros dinámicamente"""
    
    parametersChanged = pyqtSignal(dict)
    
    def __init__(self, title: str = "Parámetros", parent=None):
        super().__init__(parent)
        self.title = title
        self.parameter_groups = {}
        self.setupUI()
    
    def setupUI(self):
        """Configura la interfaz básica"""
        self.main_layout = QVBoxLayout(self)
        self.scroll_area_content = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_area_content)
    
    def addParameterGroup(self, group_name: str, group_title: str) -> ParameterGroup:
        """Añade un grupo de parámetros"""
        group = ParameterGroup(group_title)
        group.parametersChanged.connect(self._on_parameters_changed)
        self.parameter_groups[group_name] = group
        self.scroll_layout.addWidget(group)
        return group
    
    def removeParameterGroup(self, group_name: str):
        """Remueve un grupo de parámetros"""
        if group_name in self.parameter_groups:
            group = self.parameter_groups.pop(group_name)
            self.scroll_layout.removeWidget(group)
            group.deleteLater()
    
    def getParameterGroup(self, group_name: str) -> Optional[ParameterGroup]:
        """Obtiene un grupo de parámetros por nombre"""
        return self.parameter_groups.get(group_name)
    
    def getAllParameters(self) -> Dict[str, Dict[str, Any]]:
        """Obtiene todos los parámetros organizados por grupo"""
        return {
            name: group.getAllValues() 
            for name, group in self.parameter_groups.items()
        }
    
    def setAllParameters(self, parameters: Dict[str, Dict[str, Any]]):
        """Establece todos los parámetros organizados por grupo"""
        for group_name, group_params in parameters.items():
            if group_name in self.parameter_groups:
                self.parameter_groups[group_name].setAllValues(group_params)
    
    def _on_parameters_changed(self, values):
        """Maneja el cambio de parámetros en cualquier grupo"""
        self.parametersChanged.emit(self.getAllParameters())


# Funciones de utilidad para crear widgets comunes
def create_dropdown(options: List[str], description: str, 
                   default_value: str = '', label_width: int = 150) -> LabeledParameter:
    """Crea un dropdown con etiqueta"""
    combo = ParameterComboBox(options, default_value)
    return LabeledParameter(description, combo, label_width)


def create_input_box(description: str, value: Union[int, float] = 0.0, 
                    disabled: bool = False, decimals: int = 3, 
                    label_width: int = 150) -> LabeledParameter:
    """Crea un input box numérico con etiqueta"""
    if isinstance(value, int) and decimals == 0:
        input_widget = ParameterSpinBox(value)
    else:
        input_widget = ParameterLineEdit(value, decimals)
    
    input_widget.setDisabled(disabled)
    return LabeledParameter(description, input_widget, label_width)


def create_spinbox(description: str, value: int = 0, minimum: int = 0, 
                   maximum: int = 999, label_width: int = 150) -> LabeledParameter:
    """Crea un spinbox con etiqueta"""
    spinbox = ParameterSpinBox(value, minimum, maximum)
    return LabeledParameter(description, spinbox, label_width)


def create_double_spinbox(description: str, value: float = 0.0, 
                         minimum: float = 0.0, maximum: float = 999.0, 
                         decimals: int = 3, step: float = 0.1, 
                         label_width: int = 150) -> LabeledParameter:
    """Crea un double spinbox con etiqueta"""
    spinbox = ParameterDoubleSpinBox(value, minimum, maximum, decimals, step)
    return LabeledParameter(description, spinbox, label_width)


def create_check_box(description: str, checked: bool = False, 
                    label_width: int = 150) -> LabeledParameter:
    """Crea un checkbox con etiqueta"""
    checkbox = ParameterCheckBox(description, checked)
    return LabeledParameter("", checkbox, label_width)


def create_button(description: str) -> QPushButton:
    """Crea un botón con descripción"""
    return QPushButton(description)


def create_parameter_from_config(param_config: Dict[str, Any]) -> LabeledParameter:
    """Crea un parámetro basado en configuración"""
    param_type = param_config.get('type', 'text')
    label = param_config.get('label', '')
    default_value = param_config.get('default', '')
    
    if param_type == 'dropdown':
        options = param_config.get('options', [])
        return create_dropdown(options, label, default_value)
    elif param_type == 'number':
        decimals = param_config.get('decimals', 3)
        return create_input_box(label, default_value, decimals=decimals)
    elif param_type == 'integer':
        minimum = param_config.get('minimum', 0)
        maximum = param_config.get('maximum', 999)
        return create_spinbox(label, default_value, minimum, maximum)
    elif param_type == 'checkbox':
        return create_check_box(label, default_value)
    else:
        # Por defecto, campo de texto
        input_widget = QLineEdit(str(default_value))
        return LabeledParameter(label, input_widget)