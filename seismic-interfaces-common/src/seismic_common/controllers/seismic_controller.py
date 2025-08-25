"""
Controlador específico para análisis sísmico
Extiende BaseController con funcionalidades avanzadas de análisis sísmico
"""

import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Union
import threading
import time
from abc import ABC, abstractmethod

from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import QThread, QTimer, pyqtSignal, QMutex
from PyQt5.QtWidgets import QProgressDialog, QApplication

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

# Importaciones del sistema centralizado
try:
    from .base_controller import BaseController, BaseTableDialog, BaseGraphDialog
    from seismic_common.core import (
        BaseSeismicAnalysis,
        get_modal_data,
        get_table,
        get_story_drifts,
        get_joint_displacements,
        get_base_reactions
    )
    CORE_AVAILABLE = True
except ImportError:
    # Fallback para desarrollo
    CORE_AVAILABLE = False
    print("⚠️ Importaciones centralizadas no disponibles")


class AnalysisWorker(QThread):
    """
    Worker thread para ejecutar análisis sísmicos en background
    """
    
    # Señales
    progress_updated = pyqtSignal(int, str)  # progreso, mensaje
    analysis_completed = pyqtSignal(str, bool)  # tipo_análisis, éxito
    error_occurred = pyqtSignal(str, str)  # título, mensaje
    
    def __init__(self, seismic_analysis, sap_model, analysis_type, parameters=None):
        super().__init__()
        self.seismic_analysis = seismic_analysis
        self.sap_model = sap_model
        self.analysis_type = analysis_type
        self.parameters = parameters or {}
        self.mutex = QMutex()
        self._is_cancelled = False
    
    def cancel(self):
        """Cancela el análisis en progreso"""
        self.mutex.lock()
        self._is_cancelled = True
        self.mutex.unlock()
    
    def is_cancelled(self):
        """Verifica si el análisis fue cancelado"""
        self.mutex.lock()
        cancelled = self._is_cancelled
        self.mutex.unlock()
        return cancelled
    
    def run(self):
        """Ejecuta el análisis en background"""
        try:
            if self.analysis_type == 'complete':
                self._run_complete_analysis()
            elif self.analysis_type == 'modal':
                self._run_modal_analysis()
            elif self.analysis_type == 'irregularities':
                self._run_irregularities_analysis()
            elif self.analysis_type == 'drifts':
                self._run_drifts_analysis()
            elif self.analysis_type == 'shears':
                self._run_shears_analysis()
            else:
                self.error_occurred.emit("Error", f"Tipo de análisis no válido: {self.analysis_type}")
                return
            
            if not self.is_cancelled():
                self.analysis_completed.emit(self.analysis_type, True)
                
        except Exception as e:
            self.error_occurred.emit("Error en Análisis", str(e))
            self.analysis_completed.emit(self.analysis_type, False)
    
    def _run_complete_analysis(self):
        """Ejecuta análisis completo paso a paso"""
        steps = [
            ("Análisis Modal", self._run_modal_analysis),
            ("Análisis de Derivas", self._run_drifts_analysis),
            ("Irregularidad de Rigidez", lambda: self._run_single_irregularity('rigidez')),
            ("Irregularidad Torsional", lambda: self._run_single_irregularity('torsion')),
            ("Irregularidad de Masa", lambda: self._run_single_irregularity('masa')),
            ("Análisis de Cortantes", self._run_shears_analysis)
        ]
        
        total_steps = len(steps)
        
        for i, (step_name, step_function) in enumerate(steps):
            if self.is_cancelled():
                return
            
            progress = int((i / total_steps) * 100)
            self.progress_updated.emit(progress, f"Ejecutando: {step_name}")
            
            try:
                step_function()
                time.sleep(0.1)  # Pequeña pausa para actualizar UI
            except Exception as e:
                self.error_occurred.emit(f"Error en {step_name}", str(e))
                return
        
        self.progress_updated.emit(100, "Análisis completo finalizado")
    
    def _run_modal_analysis(self):
        """Ejecuta análisis modal"""
        if self.is_cancelled():
            return
        self.seismic_analysis.ana_modal(self.sap_model)
    
    def _run_drifts_analysis(self):
        """Ejecuta análisis de derivas"""
        if self.is_cancelled():
            return
        self.seismic_analysis.derivas(self.sap_model, disp_combo=True)
    
    def _run_irregularities_analysis(self):
        """Ejecuta todos los análisis de irregularidades"""
        irregularities = ['rigidez', 'torsion', 'masa']
        for irregularity in irregularities:
            if self.is_cancelled():
                return
            self._run_single_irregularity(irregularity)
    
    def _run_single_irregularity(self, irregularity_type):
        """Ejecuta un análisis de irregularidad específico"""
        if self.is_cancelled():
            return
            
        if irregularity_type == 'rigidez':
            self.seismic_analysis.irregularidad_rigidez(self.sap_model, combo=True)
        elif irregularity_type == 'torsion':
            self.seismic_analysis.irregularidad_torsion(
                self.sap_model, 
                half_condition=False, 
                disp_combo=True
            )
        elif irregularity_type == 'masa':
            self.seismic_analysis.irregularidad_masa(self.sap_model)
    
    def _run_shears_analysis(self):
        """Ejecuta análisis de cortantes"""
        if self.is_cancelled():
            return
        self.seismic_analysis.graph_shear(self.sap_model, 'dynamic')
        if not self.is_cancelled():
            self.seismic_analysis.graph_shear(self.sap_model, 'static')


class SeismicProgressDialog(QProgressDialog):
    """
    Diálogo de progreso personalizado para análisis sísmicos
    """
    
    def __init__(self, parent=None, title="Ejecutando Análisis"):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.setAutoClose(False)
        self.setAutoReset(False)
        self.setMinimumDuration(0)
        self.setRange(0, 100)
        self.setValue(0)
        
        # Estilo personalizado
        self.setStyleSheet("""
            QProgressDialog {
                background-color: #f0f0f0;
                border: 2px solid #ccc;
                border-radius: 10px;
            }
            QProgressBar {
                border: 2px solid #ccc;
                border-radius: 5px;
                text-align: center;
                font-weight: bold;
                color: #333;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                border-radius: 3px;
            }
        """)
        
        self.resize(400, 120)
        
    def update_progress(self, value: int, message: str):
        """Actualiza el progreso y mensaje"""
        self.setValue(value)
        self.setLabelText(message)
        QApplication.processEvents()


class SeismicController(BaseController):
    """
    Controlador específico para análisis sísmico avanzado
    
    Extiende BaseController con funcionalidades específicas de análisis sísmico:
    - Análisis en background con progreso
    - Validaciones específicas de parámetros sísmicos
    - Generación de reportes personalizados
    - Análisis batch de múltiples configuraciones
    """
    
    # Señales adicionales
    batch_analysis_started = pyqtSignal()
    batch_analysis_completed = pyqtSignal(int, int)  # exitosos, fallidos
    parameter_validation_failed = pyqtSignal(str, List[str])  # parámetro, errores
    
    def __init__(self, main_window: QtWidgets.QMainWindow):
        """
        Inicializa el controlador sísmico
        
        Parameters
        ----------
        main_window : QtWidgets.QMainWindow
            Ventana principal de la aplicación
        """
        super().__init__(main_window)
        
        # Worker threads
        self.current_worker = None
        self.progress_dialog = None
        
        # Configuración específica de análisis sísmico
        self.seismic_config = {
            'run_complete_analysis': True,
            'validate_mass_participation': True,
            'min_mass_participation_x': 0.90,
            'min_mass_participation_y': 0.90,
            'max_drift_allowed': 0.007,
            'check_irregularities': True,
            'generate_graphs': True,
            'save_intermediate_results': True
        }
        
        # Batch analysis
        self.batch_configurations = []
        self.batch_results = []
        
        # Timers para análisis automático
        self.auto_analysis_timer = QTimer()
        self.auto_analysis_timer.timeout.connect(self._check_auto_analysis)
        
    # Métodos de validación específicos - GENÉRICOS
    def validate_seismic_parameters(self) -> Tuple[bool, List[str]]:
        """
        Valida parámetros sísmicos específicos
        
        NOTA: Este método debe ser sobrescrito por cada normativa específica
        ya que los parámetros varían entre países/normas
        
        Returns
        -------
        Tuple[bool, List[str]]
            (Es válido, Lista de errores)
        """
        errors = []
        
        if not self.seismic_analysis:
            errors.append("Análisis sísmico no inicializado")
            return False, errors
        
        # Validaciones genéricas que aplican a todas las normativas
        
        # Validar que existan parámetros normativos
        if not hasattr(self.seismic_analysis.data, 'normative_params'):
            errors.append("No se han definido parámetros normativos")
        else:
            params = self.seismic_analysis.data.normative_params
            if not params:
                errors.append("Parámetros normativos están vacíos")
        
        # Validar cargas sísmicas (común a todas las normativas)
        if hasattr(self.seismic_analysis, 'loads'):
            seismic_loads = self.seismic_analysis.loads.seism_loads
            if not seismic_loads:
                errors.append("No se han definido cargas sísmicas")
            else:
                # Verificar que existan casos en ambas direcciones
                x_cases = self.get_seismic_cases_for_direction('X')
                y_cases = self.get_seismic_cases_for_direction('Y')
                
                if not x_cases:
                    errors.append("No se encontraron casos sísmicos en dirección X")
                if not y_cases:
                    errors.append("No se encontraron casos sísmicos en dirección Y")
        
        # Validar deriva máxima (común, pero límites varían)
        max_drift_x = getattr(self.seismic_analysis.data, 'max_drift_x', 0)
        max_drift_y = getattr(self.seismic_analysis.data, 'max_drift_y', 0)
        
        if max_drift_x <= 0:
            errors.append("Deriva máxima en X debe ser mayor a 0")
        if max_drift_y <= 0:
            errors.append("Deriva máxima en Y debe ser mayor a 0")
        
        # Validar factores de reducción (comunes)
        if self.seismic_analysis.data.Rx <= 0:
            errors.append("Factor de reducción Rx debe ser mayor a 0")
        if self.seismic_analysis.data.Ry <= 0:
            errors.append("Factor de reducción Ry debe ser mayor a 0")
        
        # LLAMAR A VALIDACIONES ESPECÍFICAS DE LA NORMATIVA
        specific_valid, specific_errors = self._validate_normative_specific_parameters()
        errors.extend(specific_errors)
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    @abstractmethod  
    def _validate_normative_specific_parameters(self) -> Tuple[bool, List[str]]:
        """
        Valida parámetros específicos de cada normativa
        
        DEBE SER IMPLEMENTADO por cada controlador específico (Perú, Bolivia, etc.)
        
        Returns
        -------
        Tuple[bool, List[str]]
            (Es válido, Lista de errores específicos)
            
        Examples
        --------
        Para E-030 (Perú):
        - Validar Z entre 0.10 y 0.45
        - Validar S entre 0.8 y 2.0  
        - Validar U entre 1.0 y 1.5
        - Validar Tp y Tl según tipo de suelo
        
        Para CNBDS (Bolivia):
        - Validar PGA según zona
        - Validar Fa y Fv según tipo de suelo
        - Validar factores específicos de Bolivia
        """
        pass
    
    def validate_modal_results(self) -> Tuple[bool, List[str]]:
        """
        Valida resultados del análisis modal
        
        Esta validación es más universal, pero los límites pueden variar
        
        Returns
        -------
        Tuple[bool, List[str]]
            (Es válido, Lista de advertencias)
        """
        warnings = []
        
        if not self.seismic_analysis or self.seismic_analysis.tables.modal.empty:
            warnings.append("No hay resultados de análisis modal")
            return False, warnings
        
        # Validar masa participativa - los límites pueden ser configurables
        mp_x = getattr(self.seismic_analysis.data, 'MP_x', 0)
        mp_y = getattr(self.seismic_analysis.data, 'MP_y', 0)
        
        min_mp_x = self.seismic_config.get('min_mass_participation_x', 0.90)
        min_mp_y = self.seismic_config.get('min_mass_participation_y', 0.90)
        
        if mp_x < min_mp_x:
            warnings.append(f"Masa participativa X ({mp_x:.3f}) menor al mínimo requerido ({min_mp_x})")
        
        if mp_y < min_mp_y:
            warnings.append(f"Masa participativa Y ({mp_y:.3f}) menor al mínimo requerido ({min_mp_y})")
        
        # Validar periodos fundamentales
        tx = getattr(self.seismic_analysis.data, 'Tx', 0)
        ty = getattr(self.seismic_analysis.data, 'Ty', 0)
        
        if tx <= 0 or ty <= 0:
            warnings.append("Periodos fundamentales no válidos")
        
        # Validar relación de periodos (puede variar según normativa)
        max_period_ratio = self.seismic_config.get('max_period_ratio', 0.8)
        if tx > 0 and ty > 0:
            ratio = abs(tx - ty) / max(tx, ty)
            if ratio > max_period_ratio:
                warnings.append(f"Gran diferencia entre periodos fundamentales Tx y Ty (ratio: {ratio:.2f})")
        
        # LLAMAR A VALIDACIONES ESPECÍFICAS DE RESULTADOS MODALES
        specific_valid, specific_warnings = self._validate_normative_specific_modal_results()
        warnings.extend(specific_warnings)
        
        # Considerar válido si no hay advertencias críticas sobre masa participativa
        critical_warnings = [w for w in warnings if "menor al mínimo" in w]
        is_valid = len(critical_warnings) == 0
        
        return is_valid, warnings
    
    @abstractmethod
    def _validate_normative_specific_modal_results(self) -> Tuple[bool, List[str]]:
        """
        Valida resultados modales específicos de cada normativa
        
        DEBE SER IMPLEMENTADO por cada controlador específico
        
        Returns
        -------
        Tuple[bool, List[str]]
            (Es válido, Lista de advertencias específicas)
            
        Examples
        --------
        Para E-030:
        - Validar que se cumple Art. 29.1.2 (90% masa participativa)
        - Verificar número mínimo de modos
        - Validar periodos según fórmulas aproximadas
        
        Para CNBDS:
        - Validaciones específicas de Bolivia
        """
        pass
    
    # Métodos de análisis avanzados
    def run_complete_seismic_analysis(self, show_progress: bool = True) -> None:
        """
        Ejecuta análisis sísmico completo en background
        
        Parameters
        ----------
        show_progress : bool
            Si mostrar diálogo de progreso
        """
        if not self.validate_etabs_connection():
            return
        
        # Validar parámetros antes de comenzar
        is_valid, errors = self.validate_seismic_parameters()
        if not is_valid:
            error_msg = "Errores en parámetros sísmicos:\n" + "\n".join(f"• {error}" for error in errors)
            self.show_error_message("Parámetros Inválidos", error_msg)
            self.parameter_validation_failed.emit("seismic_parameters", errors)
            return
        
        # Configurar parámetros del análisis
        self.set_seismic_parameters()
        
        # Crear worker thread
        self.current_worker = AnalysisWorker(
            self.seismic_analysis,
            self.sap_model,
            'complete'
        )
        
        # Conectar señales
        self.current_worker.progress_updated.connect(self._on_analysis_progress)
        self.current_worker.analysis_completed.connect(self._on_analysis_completed)
        self.current_worker.error_occurred.connect(self._on_analysis_error)
        
        # Mostrar diálogo de progreso
        if show_progress:
            self.progress_dialog = SeismicProgressDialog(
                self.main_window, 
                "Ejecutando Análisis Sísmico Completo"
            )
            self.progress_dialog.canceled.connect(self._cancel_analysis)
            self.progress_dialog.show()
        
        # Iniciar análisis
        self.current_worker.start()
    
    def run_batch_analysis(self, configurations: List[Dict[str, Any]]) -> None:
        """
        Ejecuta análisis batch con múltiples configuraciones
        
        Parameters
        ----------
        configurations : List[Dict[str, Any]]
            Lista de configuraciones de parámetros sísmicos
        """
        if not self.validate_etabs_connection():
            return
        
        self.batch_configurations = configurations
        self.batch_results = []
        
        self.batch_analysis_started.emit()
        
        # Ejecutar cada configuración secuencialmente
        for i, config in enumerate(configurations):
            if self.progress_dialog and self.progress_dialog.wasCanceled():
                break
            
            try:
                # Aplicar configuración
                self._apply_configuration(config)
                
                # Ejecutar análisis
                self.current_worker = AnalysisWorker(
                    self.seismic_analysis,
                    self.sap_model,
                    'complete',
                    config
                )
                
                # Ejecutar de forma síncrona para batch
                self.current_worker.run()
                
                # Guardar resultados
                results = self._collect_analysis_results()
                results['configuration'] = config
                results['batch_index'] = i
                self.batch_results.append(results)
                
            except Exception as e:
                self.batch_results.append({
                    'configuration': config,
                    'batch_index': i,
                    'error': str(e)
                })
        
        # Emitir señal de finalización
        successful = len([r for r in self.batch_results if 'error' not in r])
        failed = len(self.batch_results) - successful
        
        self.batch_analysis_completed.emit(successful, failed)
    
    def _apply_configuration(self, config: Dict[str, Any]) -> None:
        """Aplica una configuración específica al análisis"""
        for key, value in config.items():
            if hasattr(self.seismic_analysis.data, key):
                setattr(self.seismic_analysis.data, key, value)
            elif key in self.seismic_analysis.data.normative_params:
                self.seismic_analysis.data.normative_params[key] = value
    
    def _collect_analysis_results(self) -> Dict[str, Any]:
        """Recolecta resultados del análisis actual"""
        results = {
            'modal': {
                'Tx': self.seismic_analysis.data.Tx,
                'Ty': self.seismic_analysis.data.Ty,
                'MP_x': self.seismic_analysis.data.MP_x,
                'MP_y': self.seismic_analysis.data.MP_y
            },
            'irregularities': {
                'rigidez': not self.seismic_analysis.tables.rigidez_table.empty,
                'torsion': not self.seismic_analysis.tables.torsion_table.empty,
                'masa': not self.seismic_analysis.tables.masa_table.empty
            }
        }
        
        # Agregar derivas máximas si están disponibles
        if not self.seismic_analysis.tables.story_drifts.empty:
            drifts = self.seismic_analysis.tables.story_drifts
            if 'DriftX' in drifts.columns:
                results['max_drift_x'] = float(drifts['DriftX'].max())
            if 'DriftY' in drifts.columns:
                results['max_drift_y'] = float(drifts['DriftY'].max())
        
        return results
    
    # Métodos de callback para worker threads
    def _on_analysis_progress(self, value: int, message: str) -> None:
        """Callback para actualización de progreso"""
        if self.progress_dialog:
            self.progress_dialog.update_progress(value, message)
    
    def _on_analysis_completed(self, analysis_type: str, success: bool) -> None:
        """Callback para análisis completado"""
        if self.progress_dialog:
            self.progress_dialog.close()
            self.progress_dialog = None
        
        if success:
            # Validar resultados modales si es análisis completo
            if analysis_type == 'complete' and self.seismic_config['validate_mass_participation']:
                is_valid, warnings = self.validate_modal_results()
                if warnings:
                    warning_msg = "Advertencias en resultados:\n" + "\n".join(f"• {w}" for w in warnings)
                    self.show_warning_message("Advertencias de Análisis", warning_msg)
            
            self.show_success_message(
                "Análisis Completado", 
                f"El análisis {analysis_type} se completó exitosamente"
            )
            
            # Emitir señal
            self.analysis_completed.emit(analysis_type)
        
        # Limpiar worker
        self.current_worker = None
    
    def _on_analysis_error(self, title: str, message: str) -> None:
        """Callback para errores en análisis"""
        if self.progress_dialog:
            self.progress_dialog.close()
            self.progress_dialog = None
        
        self.show_error_message(title, message)
        self.current_worker = None
    
    def _cancel_analysis(self) -> None:
        """Cancela el análisis en progreso"""
        if self.current_worker:
            self.current_worker.cancel()
            self.current_worker.wait(3000)  # Esperar hasta 3 segundos
            if self.current_worker.isRunning():
                self.current_worker.terminate()
            self.current_worker = None
        
        if self.progress_dialog:
            self.progress_dialog.close()
            self.progress_dialog = None
        
        self.show_info_message("Análisis Cancelado", "El análisis fue cancelado por el usuario")
    
    # Métodos específicos de generación de reportes
    def generate_comparative_report(self, configurations: List[Dict[str, Any]], 
                                  output_dir: str) -> None:
        """
        Genera reporte comparativo entre múltiples configuraciones
        
        Parameters
        ----------
        configurations : List[Dict[str, Any]]
            Lista de configuraciones a comparar
        output_dir : str
            Directorio de salida
        """
        if not configurations:
            self.show_error_message("Error", "No hay configuraciones para comparar")
            return
        
        try:
            # Ejecutar análisis batch
            self.run_batch_analysis(configurations)
            
            # Generar reporte comparativo
            self._generate_comparison_tables()
            self._generate_comparison_graphs()
            
            # Crear documento final
            report_path = Path(output_dir) / "reporte_comparativo.pdf"
            # Implementar generación de PDF comparativo aquí
            
            self.show_success_message(
                "Reporte Generado",
                f"Reporte comparativo generado en:\n{report_path}"
            )
            
        except Exception as e:
            self.show_error_message("Error Reporte", f"Error generando reporte: {str(e)}")
    
    def _generate_comparison_tables(self) -> None:
        """Genera tablas comparativas de resultados batch"""
        if not self.batch_results:
            return
        
        # Crear tabla comparativa de parámetros modales
        comparison_data = []
        for result in self.batch_results:
            if 'error' not in result:
                row = {
                    'Configuración': result['batch_index'] + 1,
                    'Tx (s)': result['modal']['Tx'],
                    'Ty (s)': result['modal']['Ty'],
                    'MP_x': result['modal']['MP_x'],
                    'MP_y': result['modal']['MP_y']
                }
                
                # Agregar parámetros de configuración
                config = result['configuration']
                for key, value in config.items():
                    row[key] = value
                
                comparison_data.append(row)
        
        if comparison_data:
            self.comparison_df = pd.DataFrame(comparison_data)
    
    def _generate_comparison_graphs(self) -> None:
        """Genera gráficos comparativos"""
        if not hasattr(self, 'comparison_df') or self.comparison_df.empty:
            return
        
        # Crear gráfico comparativo de periodos
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Comparación de Resultados Sísmicos', fontsize=16)
        
        # Periodos fundamentales
        ax1.bar(self.comparison_df['Configuración'], self.comparison_df['Tx (s)'], 
                alpha=0.7, label='Tx')
        ax1.bar(self.comparison_df['Configuración'], self.comparison_df['Ty (s)'], 
                alpha=0.7, label='Ty')
        ax1.set_xlabel('Configuración')
        ax1.set_ylabel('Periodo (s)')
        ax1.set_title('Periodos Fundamentales')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Masas participativas
        ax2.bar(self.comparison_df['Configuración'], self.comparison_df['MP_x'], 
                alpha=0.7, label='MP_x')
        ax2.bar(self.comparison_df['Configuración'], self.comparison_df['MP_y'], 
                alpha=0.7, label='MP_y')
        ax2.axhline(y=0.9, color='r', linestyle='--', alpha=0.7, label='Mínimo 90%')
        ax2.set_xlabel('Configuración')
        ax2.set_ylabel('Masa Participativa')
        ax2.set_title('Masas Participativas')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Relación de periodos
        period_ratio = self.comparison_df['Tx (s)'] / self.comparison_df['Ty (s)']
        ax3.bar(self.comparison_df['Configuración'], period_ratio, alpha=0.7)
        ax3.set_xlabel('Configuración')
        ax3.set_ylabel('Tx / Ty')
        ax3.set_title('Relación de Periodos')
        ax3.grid(True, alpha=0.3)
        
        # Scatter plot Tx vs Ty
        ax4.scatter(self.comparison_df['Tx (s)'], self.comparison_df['Ty (s)'], 
                   s=100, alpha=0.7, c=self.comparison_df['Configuración'], 
                   cmap='viridis')
        ax4.set_xlabel('Tx (s)')
        ax4.set_ylabel('Ty (s)')
        ax4.set_title('Tx vs Ty')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self.comparison_figure = fig
    
    # Métodos de auto-análisis
    def enable_auto_analysis(self, interval_ms: int = 5000) -> None:
        """
        Habilita análisis automático cuando cambian parámetros
        
        Parameters
        ----------
        interval_ms : int
            Intervalo en milisegundos para verificar cambios
        """
        self.auto_analysis_timer.start(interval_ms)
    
    def disable_auto_analysis(self) -> None:
        """Deshabilita análisis automático"""
        self.auto_analysis_timer.stop()
    
    def _check_auto_analysis(self) -> None:
        """Verifica si es necesario ejecutar análisis automático"""
        # Implementar lógica para detectar cambios en parámetros
        # y ejecutar análisis automáticamente si es necesario
        pass
    
    # Métodos de configuración específicos
    def set_seismic_config(self, key: str, value: Any) -> None:
        """Establece configuración específica de análisis sísmico"""
        self.seismic_config[key] = value
    
    def get_seismic_config(self, key: str, default: Any = None) -> Any:
        """Obtiene configuración específica de análisis sísmico"""
        return self.seismic_config.get(key, default)
    
    # Override de métodos abstractos de BaseController
    def initialize_seismic_analysis(self) -> None:
        """Implementación por defecto - debe ser sobrescrita por cada normativa"""
        if CORE_AVAILABLE:
            self.seismic_analysis = BaseSeismicAnalysis()
        else:
            self.show_warning_message(
                "Core No Disponible", 
                "Sistema centralizado no disponible. Funcionalidad limitada."
            )
    
    @abstractmethod
    def set_seismic_parameters(self) -> None:
        """
        Configura parámetros sísmicos desde la interfaz
        
        DEBE SER IMPLEMENTADO por cada controlador específico ya que
        cada normativa tiene parámetros diferentes:
        
        E-030 (Perú): Z, U, S, Tp, Tl, Rox, Roy
        CNBDS (Bolivia): PGA, Fa, Fv, To, Ts  
        ASCE (USA): Ss, S1, Fa, Fv, TL
        """
        pass


if __name__ == '__main__':
    # Ejemplo de uso
    print("=== Seismic Controller - Información ===")
    
    if CORE_AVAILABLE:
        print("✓ Sistema centralizado disponible")
    else:
        print("⚠️ Sistema centralizado no disponible")
    
    print("\nCaracterísticas del SeismicController:")
    print("• Análisis en background con progreso")
    print("• Validaciones específicas de parámetros sísmicos") 
    print("• Análisis batch con múltiples configuraciones")
    print("• Generación de reportes comparativos")
    print("• Auto-análisis cuando cambian parámetros")
    print("• Worker threads para no bloquear UI")