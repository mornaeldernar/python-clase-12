"""
LABORATORIO 2: RESAMPLING Y AGREGACIÓN TEMPORAL
Sesión 12: Series Temporales en Pandas

Objetivo: Aplicar técnicas de resampling para agregación y frecuencias temporales
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Configuración para mejor visualización
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 50)

print("=" * 80)
print("LABORATORIO 2: RESAMPLING Y AGREGACIÓN TEMPORAL")
print("=" * 80)

# =============================================================================
# EJERCICIO 1: CARGAR Y PREPARAR DATOS
# =============================================================================

print("\n1. CARGANDO DATOS DE SENSORES TEMPORALES...")
print("-" * 50)

# Cargar datos de sensores
df_sensores = pd.read_csv('../datos/sensores_temporales.csv')

# Convertir timestamp a datetime y establecer como índice
df_sensores['timestamp'] = pd.to_datetime(df_sensores['timestamp'])
df_sensores.set_index('timestamp', inplace=True)

print(f"Forma del dataset: {df_sensores.shape}")
print(f"Rango temporal: {df_sensores.index.min()} a {df_sensores.index.max()}")
print(f"Frecuencia original: cada 15 minutos")

# Mostrar las primeras filas
print("\nPrimeras 10 filas:")
print(df_sensores.head(10))

# =============================================================================
# EJERCICIO 2: RESAMPLING BÁSICO - CAMBIO DE FRECUENCIAS
# =============================================================================

print("\n\n2. RESAMPLING BÁSICO - CAMBIO DE FRECUENCIAS...")
print("-" * 50)

# Resampling a diferentes frecuencias
print("Resampling a diferentes frecuencias temporales:")

columnas_numericas = df_sensores.select_dtypes(include=[np.number]).columns.tolist()
columnas_categoricas = df_sensores.select_dtypes(exclude=[np.number]).columns.tolist()

# Resampling a 1 hora (promedio)
df_hora = df_sensores[columnas_numericas].resample('H').mean()
print(f"\n• Resampling a 1 hora (promedio): {len(df_hora)} registros")
print(df_hora.head())

# Resampling a 6 horas (promedio)
df_6h = df_sensores[columnas_numericas].resample('6H').mean()
print(f"\n• Resampling a 6 horas (promedio): {len(df_6h)} registros")
print(df_6h.head())

# Resampling a 1 día (promedio)
df_dia = df_sensores[columnas_numericas].resample('D').mean()
print(f"\n• Resampling a 1 día (promedio): {len(df_dia)} registros")
print(df_dia.head())

# Resampling a 1 semana (promedio)
df_semana = df_sensores[columnas_numericas].resample('W').mean()
print(f"\n• Resampling a 1 semana (promedio): {len(df_semana)} registros")
print(df_semana.head())

# =============================================================================
# EJERCICIO 3: DIFERENTES MÉTODOS DE AGREGACIÓN
# =============================================================================

print("\n\n3. DIFERENTES MÉTODOS DE AGREGACIÓN...")
print("-" * 50)

# Agregación con múltiples métodos
print("Agregación diaria con múltiples métodos:")

agregacion_diaria = df_sensores.resample('D').agg({
    'valor': ['mean', 'std', 'min', 'max', 'count'],
    'pozo_id': 'first',
    'tipo_sensor': 'first',
    'unidad': 'first',
    'calidad_dato': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'N/A'
})

print(agregacion_diaria.head())

# Agregación semanal con métodos específicos
print("\nAgregación semanal con métodos específicos:")

agregacion_semanal = df_sensores.resample('W')['valor'].agg({
        'mean',
        'std',
         'min',
        'max',
        lambda x: x.max() - x.min(),
        'median'
})
agregacion_semanal.columns = [
    'promedio', 'desviacion', 'minimo', 'maximo', 'rango', 'mediana'
]
print(agregacion_semanal.head())

# =============================================================================
# EJERCICIO 4: RESAMPLING CON DIFERENTES FUNCIONES
# =============================================================================

print("\n\n4. RESAMPLING CON DIFERENTES FUNCIONES...")
print("-" * 50)

# Resampling con diferentes funciones para el valor
print("Resampling diario con diferentes funciones:")

resampling_funciones = df_sensores.resample('D')['valor'].agg({
        'sum',
        'mean',
        'median',
        'std',
        'min',
        'max',
        'first',
        'last',
        'count'
    
})
resampling_funciones.columns = [
    'suma', 'promedio', 'mediana', 'desviacion', 'minimo', 'maximo', 'primero', 'ultimo', 'conteo'
]

print(resampling_funciones.head())

# =============================================================================
# EJERCICIO 5: RESAMPLING CONDICIONAL
# =============================================================================

print("\n\n5. RESAMPLING CONDICIONAL...")
print("-" * 50)

# Resampling solo para datos de calidad excelente
df_excelente = df_sensores[df_sensores['calidad_dato'] == 'Excelente']
resampling_excelente = df_excelente.resample('D')['valor'].agg(['mean', 'std', 'count'])

print("Resampling diario solo para datos de calidad excelente:")
print(resampling_excelente.head())

# Comparar con resampling de todos los datos
resampling_todos = df_sensores.resample('D')['valor'].agg(['mean', 'std', 'count'])

print("\nComparación - Resampling de todos los datos:")
print(resampling_todos.head())

# =============================================================================
# EJERCICIO 6: RESAMPLING CON RELLENO DE VALORES FALTANTES
# =============================================================================

print("\n\n6. RESAMPLING CON RELLENO DE VALORES FALTANTES...")
print("-" * 50)

# Crear un dataset con valores faltantes para demostrar
df_con_faltantes = df_sensores.copy()
# Simular algunos valores faltantes
df_con_faltantes.loc[df_con_faltantes.sample(frac=0.1).index, 'valor'] = np.nan

print(f"Valores faltantes en el dataset: {df_con_faltantes['valor'].isna().sum()}")

# Resampling con diferentes métodos de relleno
print("\nResampling con diferentes métodos de relleno:")

# Forward fill (ffill)
columnas_numericas = df_con_faltantes.select_dtypes(include=[np.number]).columns.tolist()

resampling_ffill = df_con_faltantes[columnas_numericas].resample('H').mean().fillna(method='ffill')
print(f"• Forward fill: {resampling_ffill['valor'].isna().sum()} valores faltantes")

# Backward fill (bfill)
resampling_bfill = df_con_faltantes[columnas_numericas].resample('H').mean().fillna(method='bfill')
print(f"• Backward fill: {resampling_bfill['valor'].isna().sum()} valores faltantes")

# Interpolación lineal
resampling_interp = df_con_faltantes[columnas_numericas].resample('H').mean().interpolate(method='linear')
print(f"• Interpolación lineal: {resampling_interp['valor'].isna().sum()} valores faltantes")

# =============================================================================
# EJERCICIO 7: RESAMPLING AVANZADO - MÚLTIPLES FRECUENCIAS
# =============================================================================

print("\n\n7. RESAMPLING AVANZADO - MÚLTIPLES FRECUENCIAS...")
print("-" * 50)

# Crear múltiples resamplings para análisis comparativo
frecuencias = ['15T', 'H', '6H', 'D', 'W']
resamplings = {}

for freq in frecuencias:
    resamplings[freq] = df_sensores.resample(freq)['valor'].agg(['mean', 'std', 'count'])

print("Comparación de resamplings a diferentes frecuencias:")
for freq, data in resamplings.items():
    print(f"\nFrecuencia {freq}:")
    print(f"  • Registros: {len(data)}")
    print(f"  • Promedio: {data['mean'].mean():.2f}")
    print(f"  • Desviación estándar: {data['std'].mean():.2f}")

# =============================================================================
# EJERCICIO 8: VISUALIZACIÓN DE RESAMPLING
# =============================================================================

print("\n\n8. VISUALIZACIÓN DE RESAMPLING...")
print("-" * 50)

# Crear figura con subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Análisis de Resampling Temporal', fontsize=16)

# Gráfico 1: Datos originales (15 minutos)
axes[0, 0].plot(df_sensores.index, df_sensores['valor'], 'b-', linewidth=0.5, alpha=0.7)
axes[0, 0].set_title('Datos Originales (15 minutos)')
axes[0, 0].set_xlabel('Fecha')
axes[0, 0].set_ylabel('Valor del Sensor')
axes[0, 0].grid(True, alpha=0.3)

# Gráfico 2: Resampling a 1 hora
axes[0, 1].plot(df_hora.index, df_hora['valor'], 'r-', linewidth=1.5)
axes[0, 1].set_title('Resampling a 1 Hora (Promedio)')
axes[0, 1].set_xlabel('Fecha')
axes[0, 1].set_ylabel('Valor Promedio')
axes[0, 1].grid(True, alpha=0.3)

# Gráfico 3: Resampling a 6 horas
axes[1, 0].plot(df_6h.index, df_6h['valor'], 'g-', linewidth=2)
axes[1, 0].set_title('Resampling a 6 Horas (Promedio)')
axes[1, 0].set_xlabel('Fecha')
axes[1, 0].set_ylabel('Valor Promedio')
axes[1, 0].grid(True, alpha=0.3)

# Gráfico 4: Resampling a 1 día
axes[1, 1].plot(df_dia.index, df_dia['valor'], 'purple', linewidth=2.5)
axes[1, 1].set_title('Resampling a 1 Día (Promedio)')
axes[1, 1].set_xlabel('Fecha')
axes[1, 1].set_ylabel('Valor Promedio')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# =============================================================================
# EJERCICIO 9: ANÁLISIS DE ESTADÍSTICAS POR FRECUENCIA
# =============================================================================

print("\n\n9. ANÁLISIS DE ESTADÍSTICAS POR FRECUENCIA...")
print("-" * 50)

# Crear DataFrame comparativo
comparacion_frecuencias = pd.DataFrame({
    'Frecuencia': frecuencias,
    'Registros': [len(resamplings[freq]) for freq in frecuencias],
    'Promedio_Valor': [resamplings[freq]['mean'].mean() for freq in frecuencias],
    'Desv_Estandar': [resamplings[freq]['std'].mean() for freq in frecuencias],
    'Min_Valor': [resamplings[freq]['mean'].min() for freq in frecuencias],
    'Max_Valor': [resamplings[freq]['mean'].max() for freq in frecuencias]
})

print("Comparación de estadísticas por frecuencia de resampling:")
print(comparacion_frecuencias)

# =============================================================================
# EJERCICIO 10: RESAMPLING CON EVENTOS OPERACIONALES
# =============================================================================

print("\n\n10. RESAMPLING CON EVENTOS OPERACIONALES...")
print("-" * 50)

# Cargar datos de eventos operacionales
df_eventos = pd.read_csv('../datos/eventos_operacionales.csv')
df_eventos['fecha_evento'] = pd.to_datetime(df_eventos['fecha_evento'])
df_eventos.set_index('fecha_evento', inplace=True)

print("Datos de eventos operacionales:")
print(df_eventos.head())

# Resampling de eventos por día
eventos_diarios = df_eventos.resample('D').agg({
    'duracion_horas': 'sum',
    'impacto_produccion': 'sum',
    'tipo_evento': 'count'
}).rename(columns={'tipo_evento': 'num_eventos'})

print("\nAgregación diaria de eventos:")
print(eventos_diarios.head())

# Resampling de eventos por semana
eventos_semanales = df_eventos.resample('W').agg({
    'duracion_horas': 'sum',
    'impacto_produccion': 'sum',
    'tipo_evento': 'count'
}).rename(columns={'tipo_evento': 'num_eventos'})

print("\nAgregación semanal de eventos:")
print(eventos_semanales.head())

# =============================================================================
# EJERCICIO 11: ANÁLISIS INTEGRADO - SENSORES Y EVENTOS
# =============================================================================

print("\n\n11. ANÁLISIS INTEGRADO - SENSORES Y EVENTOS...")
print("-" * 50)

# Resampling de sensores a frecuencia diaria para comparar con eventos
sensores_diarios = df_sensores.resample('D')['valor'].agg(['mean', 'std', 'min', 'max'])

# Combinar datos de sensores y eventos
analisis_integrado = pd.concat([sensores_diarios, eventos_diarios], axis=1)
analisis_integrado = analisis_integrado.fillna(0)  # Rellenar valores faltantes con 0

print("Análisis integrado - Sensores y eventos diarios:")
print(analisis_integrado.head(10))

# Calcular correlaciones
correlaciones = analisis_integrado.corr()
print("\nCorrelaciones entre variables:")
print(correlaciones)

# =============================================================================
# EJERCICIO 12: RESUMEN Y CONCLUSIONES
# =============================================================================

print("\n\n12. RESUMEN Y CONCLUSIONES...")
print("-" * 50)

print("RESUMEN DEL ANÁLISIS DE RESAMPLING:")
print(f"• Datos originales: {len(df_sensores)} registros cada 15 minutos")
print(f"• Período analizado: {df_sensores.index.min()} a {df_sensores.index.max()}")

print("\nCOMPARACIÓN DE FRECUENCIAS:")
for freq in frecuencias:
    data = resamplings[freq]
    print(f"• {freq}: {len(data)} registros, promedio: {data['mean'].mean():.2f}")

print("\nBENEFICIOS DEL RESAMPLING:")
print("• Reducción de ruido en datos de alta frecuencia")
print("• Facilitación del análisis de tendencias")
print("• Comparabilidad entre diferentes escalas temporales")
print("• Optimización del almacenamiento y procesamiento")

print("\nAPLICACIONES EN EL SECTOR PETROLERO:")
print("• Análisis de tendencias de producción")
print("• Monitoreo de parámetros operativos")
print("• Correlación entre eventos y rendimiento")
print("• Generación de reportes ejecutivos")

print("\n" + "=" * 80)
print("LABORATORIO 2 COMPLETADO")
print("=" * 80) 