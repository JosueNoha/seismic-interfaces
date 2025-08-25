"""
Diálogo centralizado para manejo de descripciones de análisis sísmicos
Centraliza la funcionalidad común de editar y gestionar textos descriptivos
"""

from typing import Optional, Dict, Any, List, Union, Callable
import os
from pathlib import Path

from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import Qt, pyqtSignal, QSize
from PyQt5.QtWidgets import (
    QDialog, QMainWindow, QVBoxLayout, QHBoxLayout, QLabel, 
    QPlainTextEdit, QPushButton, QFileDialog, QMessageBox, 
    QApplication, QSizePolicy, QSpacerItem, QFrame, QComboBox,
    QCheckBox, QGroupBox, QGridLayout, QSpinBox, QScrollArea,
    QTabWidget, QTextEdit, QSplitter
)
from PyQt5.QtGui import QFont, QIcon, QColor, QPalette, QTextCursor

import re


class SeismicDescriptionDialog(QMainWindow):
    """
    Diálogo centralizado para editar descripciones de análisis sísmico
    Reemplaza las clases Descriptions y similares de los proyectos existentes
    """
    
    # Señales
    description_saved = pyqtSignal(str, str)    # name, text
    description_cancelled = pyqtSignal(str)     # name
    description_changed = pyqtSignal(str, str)  # name, text
    window_closed = pyqtSignal()                # ventana cerrada
    
    def __init__(self, title: str = "Descripción de la Estructura",
                 description_text: str = "",
                 description_name: str = "",
                 parent=None,
                 show_templates: bool = True):
        """
        Inicializa el diálogo de descripción
        
        Parameters
        ----------
        title : str
            Título de la ventana
        description_text : str
            Texto inicial de la descripción
        description_name : str
            Nombre/tipo de la descripción
        parent : QWidget, optional
            Widget padre
        show_templates : bool
            Si mostrar panel de plantillas
        """
        super().__init__(parent)
        
        self.description_name = description_name
        self.show_templates = show_templates
        self._original_text = description_text
        self._has_changes = False
        
        # Configuración de la ventana
        self.setWindowTitle(title)
        self.resize(800, 500)
        
        # Plantillas por defecto para diferentes tipos de descripción
        self.default_templates = self._load_default_templates()
        
        # Configurar interfaz
        self.setup_ui()
        self.setup_connections()
        
        # Establecer texto inicial
        if description_text:
            self.pt_description.setPlainText(description_text)
    
    def setup_ui(self):
        """Configura la interfaz de usuario"""
        # Widget central
        self.central_widget = QFrame(self)
        self.setCentralWidget(self.central_widget)
        
        if self.show_templates:
            # Layout con panel de plantillas
            main_layout = QHBoxLayout(self.central_widget)
            
            # Splitter para editor y plantillas
            splitter = QSplitter(Qt.Horizontal)
            main_layout.addWidget(splitter)
            
            # Widget del editor
            editor_widget = self.create_editor_widget()
            splitter.addWidget(editor_widget)
            
            # Panel de plantillas
            templates_widget = self.create_templates_panel()
            splitter.addWidget(templates_widget)
            
            # Configurar proporción del splitter
            splitter.setSizes([600, 200])
            
        else:
            # Layout simple
            main_layout = QVBoxLayout(self.central_widget)
            editor_widget = self.create_editor_widget()
            main_layout.addWidget(editor_widget)
    
    def create_editor_widget(self):
        """Crea el widget principal del editor"""
        editor_widget = QFrame()
        layout = QVBoxLayout(editor_widget)
        
        # Label de título
        self.title_label = QLabel("Descripción de la Estructura")
        self.setup_title_font()
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setMaximumSize(QSize(16777215, 50))
        self.title_label.setObjectName("title_label")
        layout.addWidget(self.title_label)
        
        # Editor de texto principal
        self.pt_description = QPlainTextEdit()
        self.setup_editor_font()
        self.pt_description.setObjectName("pt_description")  # Compatible con código existente
        layout.addWidget(self.pt_description)
        
        # Layout de botones
        buttons_layout = QHBoxLayout()
        
        # Spacer izquierdo
        left_spacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        buttons_layout.addItem(left_spacer)
        
        # Botones de acción
        self.setup_buttons(buttons_layout)
        
        # Spacer derecho
        right_spacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        buttons_layout.addItem(right_spacer)
        
        layout.addLayout(buttons_layout)
        
        return editor_widget
    
    def create_templates_panel(self):
        """Crea el panel de plantillas y herramientas"""
        templates_widget = QFrame()
        layout = QVBoxLayout(templates_widget)
        
        # Título del panel
        panel_title = QLabel("Herramientas")
        panel_title.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(panel_title)
        
        # Scroll area para contenido
        scroll = QScrollArea()
        scroll_widget = QFrame()
        scroll_layout = QVBoxLayout(scroll_widget)
        
        # Grupo: Plantillas
        templates_group = QGroupBox("Plantillas")
        templates_layout = QVBoxLayout(templates_group)
        
        self.template_combo = QComboBox()
        self.template_combo.addItems(list(self.default_templates.keys()))
        templates_layout.addWidget(self.template_combo)
        
        self.load_template_btn = QPushButton("Cargar Plantilla")
        templates_layout.addWidget(self.load_template_btn)
        
        scroll_layout.addWidget(templates_group)
        
        # Grupo: Formato
        format_group = QGroupBox("Formato")
        format_layout = QVBoxLayout(format_group)
        
        self.word_wrap_check = QCheckBox("Ajuste de línea")
        self.word_wrap_check.setChecked(True)
        format_layout.addWidget(self.word_wrap_check)
        
        # Tamaño de fuente
        font_layout = QHBoxLayout()
        font_layout.addWidget(QLabel("Tamaño:"))
        self.font_size_spin = QSpinBox()
        self.font_size_spin.setRange(8, 24)
        self.font_size_spin.setValue(12)
        font_layout.addWidget(self.font_size_spin)
        format_layout.addLayout(font_layout)
        
        scroll_layout.addWidget(format_group)
        
        # Grupo: Estadísticas
        stats_group = QGroupBox("Estadísticas")
        stats_layout = QVBoxLayout(stats_group)
        
        self.char_count_label = QLabel("Caracteres: 0")
        self.word_count_label = QLabel("Palabras: 0")
        self.line_count_label = QLabel("Líneas: 0")
        
        stats_layout.addWidget(self.char_count_label)
        stats_layout.addWidget(self.word_count_label)
        stats_layout.addWidget(self.line_count_label)
        
        scroll_layout.addWidget(stats_group)
        
        # Grupo: Acciones
        actions_group = QGroupBox("Acciones")
        actions_layout = QVBoxLayout(actions_group)
        
        self.clear_btn = QPushButton("Limpiar")
        self.reset_btn = QPushButton("Restaurar")
        
        actions_layout.addWidget(self.clear_btn)
        actions_layout.addWidget(self.reset_btn)
        
        scroll_layout.addWidget(actions_group)
        
        scroll_layout.addStretch()
        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)
        
        return templates_widget
    
    def setup_title_font(self):
        """Configura la fuente del título"""
        font = QFont()
        font.setFamily("MS Shell Dlg 2")
        font.setPointSize(24)
        font.setBold(False)
        font.setWeight(50)
        self.title_label.setFont(font)
    
    def setup_editor_font(self):
        """Configura la fuente del editor"""
        font = QFont()
        font.setPointSize(12)
        self.pt_description.setFont(font)
    
    def setup_buttons(self, layout):
        """Configura los botones principales"""
        # Botón agregar/guardar
        self.b_add = QPushButton("Agregar Descripción")
        self.b_add.setObjectName("b_add")  # Compatible con código existente
        layout.addWidget(self.b_add)
        
        # Botón cancelar
        self.cancel_btn = QPushButton("Cancelar")
        layout.addWidget(self.cancel_btn)
        
        # Botón exportar (opcional)
        self.export_btn = QPushButton("Exportar")
        layout.addWidget(self.export_btn)
    
    def setup_connections(self):
        """Configura las conexiones de señales y slots"""
        # Botones principales
        self.b_add.clicked.connect(self.save_description)
        self.cancel_btn.clicked.connect(self.cancel_description)
        self.export_btn.clicked.connect(self.export_description)
        
        # Controles avanzados si existen
        if self.show_templates:
            self.load_template_btn.clicked.connect(self.load_template)
            self.word_wrap_check.toggled.connect(self.toggle_word_wrap)
            self.font_size_spin.valueChanged.connect(self.change_font_size)
            self.clear_btn.clicked.connect(self.clear_text)
            self.reset_btn.clicked.connect(self.reset_text)
            
            # Actualizar estadísticas cuando cambie el texto
            self.pt_description.textChanged.connect(self.update_statistics)
        
        # Detectar cambios en el texto
        self.pt_description.textChanged.connect(self.on_text_changed)
        
        # Señal de cierre de ventana
        # Note: QMainWindow no tiene finished signal, usamos closeEvent
    
    def _load_default_templates(self) -> Dict[str, str]:
        """Carga las plantillas por defecto"""
        return {
            "Descripción General": """La edificación de concreto armado tiene 13 niveles y 1 sótano, con un área en planta aproximada de 400 m2, el sistema estructural en la dirección X es de muros estructurales especiales con pórticos especiales que toman el 0.25 de la cortante en la base y en la dirección Y tenemos unicamente pórticos especiales.""",
            
            "Criterios de Modelamiento": """Para el modelamiento del edificio en el programa ETABS se considero la rigidez efectiva de los elementos estructurales y el diafragma rígido se considero infinitamente flexible fuera del plano, en el caso de las vigas se considero el ancho efectivo de las vigas T debido a que se tiene un vaciado monolítico con el sistema de piso. El edificio se considera empotrado en la base, la planta y el 3D del edificio se muestra en la siguiente figura.""",
            
            "Descripción de Cargas": """Se considero 220 kgf/m2 de sobrecarga muerta (tabiquería y piso terminado) y 250 kgf/m2 de sobrecarga viva aplicado al área en planta del edificio.""",
            
            "Plantilla Vacía": ""
        }
    
    def set_title(self, title: str):
        """
        Establece el título del diálogo y del label
        
        Parameters
        ----------
        title : str
            Nuevo título
        """
        self.setWindowTitle(title)
        self.title_label.setText(title)
    
    def set_description_name(self, name: str):
        """
        Establece el nombre de la descripción
        
        Parameters
        ----------
        name : str
            Nombre de la descripción
        """
        self.description_name = name
    
    def get_description_text(self) -> str:
        """
        Obtiene el texto actual de la descripción
        
        Returns
        -------
        str
            Texto de la descripción
        """
        return self.pt_description.toPlainText()
    
    def set_description_text(self, text: str):
        """
        Establece el texto de la descripción
        
        Parameters
        ----------
        text : str
            Nuevo texto
        """
        self.pt_description.setPlainText(text)
        self._original_text = text
        self._has_changes = False
    
    def load_template(self):
        """Carga la plantilla seleccionada"""
        if not hasattr(self, 'template_combo'):
            return
        
        template_name = self.template_combo.currentText()
        template_text = self.default_templates.get(template_name, "")
        
        if template_text and self._has_changes:
            reply = QMessageBox.question(
                self, "Confirmar",
                "¿Desea reemplazar el texto actual con la plantilla?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                return
        
        self.pt_description.setPlainText(template_text)
    
    def toggle_word_wrap(self, enabled: bool):
        """Alterna el ajuste de línea"""
        if enabled:
            self.pt_description.setLineWrapMode(QPlainTextEdit.WidgetWidth)
        else:
            self.pt_description.setLineWrapMode(QPlainTextEdit.NoWrap)
    
    def change_font_size(self, size: int):
        """Cambia el tamaño de fuente"""
        font = self.pt_description.font()
        font.setPointSize(size)
        self.pt_description.setFont(font)
    
    def update_statistics(self):
        """Actualiza las estadísticas del texto"""
        if not self.show_templates:
            return
        
        text = self.pt_description.toPlainText()
        
        # Contar caracteres, palabras y líneas
        char_count = len(text)
        word_count = len(text.split()) if text.strip() else 0
        line_count = text.count('\n') + 1 if text else 1
        
        self.char_count_label.setText(f"Caracteres: {char_count}")
        self.word_count_label.setText(f"Palabras: {word_count}")
        self.line_count_label.setText(f"Líneas: {line_count}")
    
    def clear_text(self):
        """Limpia el texto del editor"""
        reply = QMessageBox.question(
            self, "Confirmar",
            "¿Desea limpiar todo el texto?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.pt_description.clear()
    
    def reset_text(self):
        """Restaura el texto original"""
        if self._has_changes:
            reply = QMessageBox.question(
                self, "Confirmar",
                "¿Desea restaurar el texto original?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self.pt_description.setPlainText(self._original_text)
    
    def on_text_changed(self):
        """Maneja el evento de cambio de texto"""
        current_text = self.get_description_text()
        self._has_changes = (current_text != self._original_text)
        
        # Emitir señal de cambio
        self.description_changed.emit(self.description_name, current_text)
        
        # Actualizar estadísticas si están habilitadas
        if self.show_templates:
            self.update_statistics()
    
    def save_description(self):
        """Guarda la descripción"""
        text = self.get_description_text().strip()
        
        if not text:
            QMessageBox.warning(
                self, "Advertencia",
                "La descripción no puede estar vacía."
            )
            return
        
        # Emitir señal de guardado
        self.description_saved.emit(self.description_name, text)
        
        # Cerrar ventana
        self.close()
    
    def cancel_description(self):
        """Cancela la edición"""
        if self._has_changes:
            reply = QMessageBox.question(
                self, "Confirmar",
                "¿Desea cerrar sin guardar los cambios?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                return
        
        # Emitir señal de cancelación
        self.description_cancelled.emit(self.description_name)
        
        # Cerrar ventana
        self.close()
    
    def export_description(self):
        """Exporta la descripción a un archivo de texto"""
        text = self.get_description_text()
        
        if not text.strip():
            QMessageBox.warning(
                self, "Advertencia", 
                "No hay texto para exportar."
            )
            return
        
        # Diálogo para seleccionar archivo
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Exportar descripción",
            f"descripcion_{self.description_name}.txt",
            "Text Files (*.txt);;All Files (*)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(text)
                
                QMessageBox.information(
                    self, "Éxito",
                    f"Descripción exportada correctamente a:\n{file_path}"
                )
                
            except Exception as e:
                QMessageBox.critical(
                    self, "Error",
                    f"Error al exportar la descripción:\n{str(e)}"
                )
    
    def closeEvent(self, event):
        """Maneja el evento de cierre de ventana"""
        if self._has_changes:
            reply = QMessageBox.question(
                self, "Confirmar",
                "¿Desea cerrar sin guardar los cambios?",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel
            )
            
            if reply == QMessageBox.Save:
                self.save_description()
                event.accept()
            elif reply == QMessageBox.Discard:
                self.window_closed.emit()
                event.accept()
            else:
                event.ignore()
        else:
            self.window_closed.emit()
            event.accept()
    
    # Métodos de compatibilidad con código existente
    @property
    def ui(self):
        """
        Propiedad de compatibilidad para acceder a elementos UI
        Permite usar dialog.ui.pt_description como en el código existente
        """
        return self
    
    @property
    def name(self):
        """Compatibilidad: nombre de la descripción"""
        return self.description_name
    
    @name.setter
    def name(self, value):
        """Compatibilidad: establecer nombre de la descripción"""
        self.description_name = value


# Funciones de conveniencia para crear diálogos específicos
def create_structure_description_dialog(text: str = "", parent=None) -> SeismicDescriptionDialog:
    """
    Crea un diálogo para descripción general de la estructura
    
    Parameters
    ----------
    text : str
        Texto inicial
    parent : QWidget, optional
        Widget padre
        
    Returns
    -------
    SeismicDescriptionDialog
        Diálogo configurado para descripción general
    """
    dialog = SeismicDescriptionDialog(
        title="Descripción de la Estructura",
        description_text=text,
        description_name="descripcion",
        parent=parent,
        show_templates=True
    )
    
    return dialog


def create_modeling_description_dialog(text: str = "", parent=None) -> SeismicDescriptionDialog:
    """
    Crea un diálogo para criterios de modelamiento
    
    Parameters
    ----------
    text : str
        Texto inicial
    parent : QWidget, optional
        Widget padre
        
    Returns
    -------
    SeismicDescriptionDialog
        Diálogo configurado para modelamiento
    """
    dialog = SeismicDescriptionDialog(
        title="Criterios de Modelamiento",
        description_text=text,
        description_name="modelamiento",
        parent=parent,
        show_templates=True
    )
    
    return dialog


def create_loads_description_dialog(text: str = "", parent=None) -> SeismicDescriptionDialog:
    """
    Crea un diálogo para descripción de cargas
    
    Parameters
    ----------
    text : str
        Texto inicial
    parent : QWidget, optional
        Widget padre
        
    Returns
    -------
    SeismicDescriptionDialog
        Diálogo configurado para cargas
    """
    dialog = SeismicDescriptionDialog(
        title="Descripción de Cargas Consideradas",
        description_text=text,
        description_name="cargas",
        parent=parent,
        show_templates=True
    )
    
    return dialog


def create_simple_description_dialog(title: str = "Descripción", text: str = "", parent=None) -> SeismicDescriptionDialog:
    """
    Crea un diálogo simple sin plantillas
    
    Parameters
    ----------
    title : str
        Título del diálogo
    text : str
        Texto inicial
    parent : QWidget, optional
        Widget padre
        
    Returns
    -------
    SeismicDescriptionDialog
        Diálogo simple
    """
    dialog = SeismicDescriptionDialog(
        title=title,
        description_text=text,
        description_name="custom",
        parent=parent,
        show_templates=False
    )
    
    return dialog


# Clase de compatibilidad para código existente
class Descriptions(SeismicDescriptionDialog):
    """
    Clase de compatibilidad para el código existente
    Permite usar Descriptions() como antes sin cambios
    """
    
    def __init__(self, parent=None):
        super().__init__(
            title="Descripción de la Estructura",
            description_text="",
            description_name="",
            parent=parent,
            show_templates=False  # Comportamiento original más simple
        )


# Alias para compatibilidad adicional  
DescriptionDialog = SeismicDescriptionDialog
SeismicDescriptions = Descriptions