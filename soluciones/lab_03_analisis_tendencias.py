"""
LABORATORIO 3: ANÁLISIS DE TENDENCIAS Y FUNCIONES MÓVILES
Sesión 12: Series Temporales en Pandas

Objetivo: Implementar rolling windows y funciones móviles para análisis de tendencias
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
print("LABORATORIO 3: ANÁLISIS DE TENDENCIAS Y FUNCIONES MÓVILES")
print("=" * 80)

# =============================================================================
# EJERCICIO 1: CARGAR Y PREPARAR DATOS
# =============================================================================

print("\n1. CARGANDO DATOS DE PARÁMETROS DE POZOS...")
print("-" * 50)

# Cargar datos de parámetros de pozos
df_parametros = pd.read_csv('../datos/parametros_pozos.csv')

# Convertir fecha a datetime y establecer como índice
df_parametros['fecha'] = pd.to_datetime(df_parametros['fecha'])
df_parametros.set_index('fecha', inplace=True)

print(f"Forma del dataset: {df_parametros.shape}")
print(f"Rango temporal: {df_parametros.index.min()} a {df_parametros.index.max()}")
print(f"Frecuencia: cada 6 horas")

# Mostrar las primeras filas
print("\nPrimeras 10 filas:")
print(df_parametros.head(10))

# =============================================================================
# EJERCICIO 2: ROLLING WINDOWS BÁSICOS
# =============================================================================

print("\n\n2. ROLLING WINDOWS BÁSICOS...")
print("-" * 50)

# Rolling window de 7 días (28 registros de 6 horas)
df_parametros['produccion_ma_7d'] = df_parametros['caudal_bpd'].rolling(window=28).mean()
df_parametros['presion_ma_7d'] = df_parametros['presion_cabeza_psi'].rolling(window=28).mean()
df_parametros['temperatura_ma_7d'] = df_parametros['temperatura_cabeza_f'].rolling(window=28).mean()

print("Medias móviles de 7 días calculadas:")
print(df_parametros[['caudal_bpd', 'produccion_ma_7d', 'presion_cabeza_psi', 'presion_ma_7d']].head(15))

# Rolling window de 30 días (120 registros de 6 horas)
df_parametros['produccion_ma_30d'] = df_parametros['caudal_bpd'].rolling(window=120).mean()
df_parametros['presion_ma_30d'] = df_parametros['presion_cabeza_psi'].rolling(window=120).mean()

print("\nMedias móviles de 30 días calculadas:")
print(df_parametros[['caudal_bpd', 'produccion_ma_30d', 'presion_cabeza_psi', 'presion_ma_30d']].head(15))

# =============================================================================
# EJERCICIO 3: DIFERENTES FUNCIONES DE ROLLING
# =============================================================================

print("\n\n3. DIFERENTES FUNCIONES DE ROLLING...")
print("-" * 50)

# Múltiples funciones de rolling para producción
df_parametros['produccion_ma_7d'] = df_parametros['caudal_bpd'].rolling(window=28).mean()
df_parametros['produccion_mediana_7d'] = df_parametros['caudal_bpd'].rolling(window=28).median()
df_parametros['produccion_std_7d'] = df_parametros['caudal_bpd'].rolling(window=28).std()
df_parametros['produccion_min_7d'] = df_parametros['caudal_bpd'].rolling(window=28).min()
df_parametros['produccion_max_7d'] = df_parametros['caudal_bpd'].rolling(window=28).max()

print("Estadísticas de rolling window de 7 días para producción:")
print(df_parametros[['caudal_bpd', 'produccion_ma_7d', 'produccion_mediana_7d', 
                     'produccion_std_7d', 'produccion_min_7d', 'produccion_max_7d']].head(15))

# =============================================================================
# EJERCICIO 4: ROLLING WINDOWS CON DIFERENTES TAMAÑOS
# =============================================================================

print("\n\n4. ROLLING WINDOWS CON DIFERENTES TAMAÑOS...")
print("-" * 50)

# Rolling windows de diferentes tamaños
ventanas = [4, 28, 120, 240]  # 1 día, 1 semana, 1 mes, 2 meses

for ventana in ventanas:
    df_parametros[f'produccion_ma_{ventana}'] = df_parametros['caudal_bpd'].rolling(window=ventana).mean()
    df_parametros[f'presion_ma_{ventana}'] = df_parametros['presion_cabeza_psi'].rolling(window=ventana).mean()

print("Medias móviles con diferentes tamaños de ventana:")
columnas_rolling = ['caudal_bpd'] + [f'produccion_ma_{v}' for v in ventanas]
print(df_parametros[columnas_rolling].head(15))

# =============================================================================
# EJERCICIO 5: ANÁLISIS DE TENDENCIAS CON ROLLING
# =============================================================================

print("\n\n5. ANÁLISIS DE TENDENCIAS CON ROLLING...")
print("-" * 50)

# Calcular tendencia móvil usando regresión lineal
def calcular_tendencia(serie):
    """Calcula la pendiente de la línea de tendencia en una ventana móvil"""
    if len(serie) < 2:
        return np.nan
    x = np.arange(len(serie))
    slope = np.polyfit(x, serie, 1)[0]
    return slope

# Aplicar función de tendencia en ventana móvil de 7 días
df_parametros['tendencia_7d'] = df_parametros['caudal_bpd'].rolling(window=28).apply(calcular_tendencia)
df_parametros['tendencia_30d'] = df_parametros['caudal_bpd'].rolling(window=120).apply(calcular_tendencia)

print("Análisis de tendencias móviles:")
print(df_parametros[['caudal_bpd', 'tendencia_7d', 'tendencia_30d']].head(15))

# Interpretar tendencias
print("\nInterpretación de tendencias:")
print("• Valores positivos: Tendencia creciente")
print("• Valores negativos: Tendencia decreciente")
print("• Valores cercanos a cero: Sin tendencia clara")

# =============================================================================
# EJERCICIO 6: DETECCIÓN DE ANOMALÍAS CON ROLLING
# =============================================================================

print("\n\n6. DETECCIÓN DE ANOMALÍAS CON ROLLING...")
print("-" * 50)

# Calcular límites de control usando rolling statistics
df_parametros['produccion_ma_7d'] = df_parametros['caudal_bpd'].rolling(window=28).mean()
df_parametros['produccion_std_7d'] = df_parametros['caudal_bpd'].rolling(window=28).std()

# Límites de control (3 desviaciones estándar)
df_parametros['limite_superior'] = df_parametros['produccion_ma_7d'] + (3 * df_parametros['produccion_std_7d'])
df_parametros['limite_inferior'] = df_parametros['produccion_ma_7d'] - (3 * df_parametros['produccion_std_7d'])

# Detectar anomalías
df_parametros['es_anomalia'] = (
    (df_parametros['caudal_bpd'] > df_parametros['limite_superior']) |
    (df_parametros['caudal_bpd'] < df_parametros['limite_inferior'])
)

print("Detección de anomalías:")
print(df_parametros[['caudal_bpd', 'produccion_ma_7d', 'limite_superior', 
                     'limite_inferior', 'es_anomalia']].head(15))

# Contar anomalías
num_anomalias = df_parametros['es_anomalia'].sum()
print(f"\nTotal de anomalías detectadas: {num_anomalias}")

# =============================================================================
# EJERCICIO 7: ROLLING WINDOWS CON CENTERING
# =============================================================================

print("\n\n7. ROLLING WINDOWS CON CENTERING...")
print("-" * 50)

# Rolling window centrado (promedio de valores antes y después del punto actual)
df_parametros['produccion_ma_centrada'] = df_parametros['caudal_bpd'].rolling(window=28, center=True).mean()
df_parametros['presion_ma_centrada'] = df_parametros['presion_cabeza_psi'].rolling(window=28, center=True).mean()

print("Medias móviles centradas:")
print(df_parametros[['caudal_bpd', 'produccion_ma_7d', 'produccion_ma_centrada']].head(15))

# =============================================================================
# EJERCICIO 8: ROLLING WINDOWS CON MIN_PERIODS
# =============================================================================

print("\n\n8. ROLLING WINDOWS CON MIN_PERIODS...")
print("-" * 50)

# Rolling window con mínimo de períodos requeridos
df_parametros['produccion_ma_min_periods'] = df_parametros['caudal_bpd'].rolling(
    window=28, min_periods=14).mean()  # Requiere al menos 14 períodos

print("Medias móviles con mínimo de períodos:")
print(df_parametros[['caudal_bpd', 'produccion_ma_7d', 'produccion_ma_min_periods']].head(15))

# =============================================================================
# EJERCICIO 9: ANÁLISIS DE ESTACIONALIDAD
# =============================================================================

print("\n\n9. ANÁLISIS DE ESTACIONALIDAD...")
print("-" * 50)

# Calcular estacionalidad usando rolling window de 1 año (aproximadamente)
# Como tenemos datos de 1 mes, usaremos un período más corto para demostración
df_parametros['estacionalidad_7d'] = df_parametros['caudal_bpd'].rolling(window=28).mean()

# Calcular componente de tendencia (rolling window más largo)
df_parametros['tendencia_30d'] = df_parametros['caudal_bpd'].rolling(window=120).mean()

# Calcular residuos (datos - tendencia - estacionalidad)
df_parametros['residuos'] = df_parametros['caudal_bpd'] - df_parametros['tendencia_30d'] - df_parametros['estacionalidad_7d']

print("Descomposición de la serie temporal:")
print(df_parametros[['caudal_bpd', 'tendencia_30d', 'estacionalidad_7d', 'residuos']].head(15))

# =============================================================================
# EJERCICIO 10: VISUALIZACIÓN DE ROLLING WINDOWS
# =============================================================================

print("\n\n10. VISUALIZACIÓN DE ROLLING WINDOWS...")
print("-" * 50)

# Crear figura con subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Análisis de Rolling Windows y Tendencias', fontsize=16)

# Gráfico 1: Producción con medias móviles
axes[0, 0].plot(df_parametros.index, df_parametros['caudal_bpd'], 'b-', linewidth=1, alpha=0.7, label='Producción')
axes[0, 0].plot(df_parametros.index, df_parametros['produccion_ma_7d'], 'r-', linewidth=2, label='MA 7 días')
axes[0, 0].plot(df_parametros.index, df_parametros['produccion_ma_30d'], 'g-', linewidth=2, label='MA 30 días')
axes[0, 0].set_title('Producción con Medias Móviles')
axes[0, 0].set_xlabel('Fecha')
axes[0, 0].set_ylabel('Producción (BPD)')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Gráfico 2: Límites de control y anomalías
axes[0, 1].plot(df_parametros.index, df_parametros['caudal_bpd'], 'b-', linewidth=1, alpha=0.7, label='Producción')
axes[0, 1].plot(df_parametros.index, df_parametros['limite_superior'], 'r--', linewidth=1, label='Límite Superior')
axes[0, 1].plot(df_parametros.index, df_parametros['limite_inferior'], 'r--', linewidth=1, label='Límite Inferior')
axes[0, 1].scatter(df_parametros[df_parametros['es_anomalia']].index, 
                   df_parametros[df_parametros['es_anomalia']]['caudal_bpd'], 
                   color='red', s=50, label='Anomalías')
axes[0, 1].set_title('Detección de Anomalías')
axes[0, 1].set_xlabel('Fecha')
axes[0, 1].set_ylabel('Producción (BPD)')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Gráfico 3: Descomposición de la serie temporal
axes[1, 0].plot(df_parametros.index, df_parametros['caudal_bpd'], 'b-', linewidth=1, alpha=0.7, label='Original')
axes[1, 0].plot(df_parametros.index, df_parametros['tendencia_30d'], 'r-', linewidth=2, label='Tendencia')
axes[1, 0].plot(df_parametros.index, df_parametros['estacionalidad_7d'], 'g-', linewidth=2, label='Estacionalidad')
axes[1, 0].set_title('Descomposición de Serie Temporal')
axes[1, 0].set_xlabel('Fecha')
axes[1, 0].set_ylabel('Producción (BPD)')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Gráfico 4: Análisis de tendencias
axes[1, 1].plot(df_parametros.index, df_parametros['tendencia_7d'], 'b-', linewidth=1, label='Tendencia 7 días')
axes[1, 1].plot(df_parametros.index, df_parametros['tendencia_30d'], 'r-', linewidth=1, label='Tendencia 30 días')
axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
axes[1, 1].set_title('Análisis de Tendencias')
axes[1, 1].set_xlabel('Fecha')
axes[1, 1].set_ylabel('Pendiente de Tendencia')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# =============================================================================
# EJERCICIO 11: ANÁLISIS DE CORRELACIÓN MÓVIL
# =============================================================================

print("\n\n11. ANÁLISIS DE CORRELACIÓN MÓVIL...")
print("-" * 50)

# Calcular correlación móvil entre producción y presión
def correlacion_movil(serie1, serie2, ventana=28):
    """Calcula la correlación móvil entre dos series"""
    correlaciones = []
    for i in range(len(serie1)):
        if i < ventana - 1:
            correlaciones.append(np.nan)
        else:
            corr = np.corrcoef(serie1.iloc[i-ventana+1:i+1], 
                              serie2.iloc[i-ventana+1:i+1])[0, 1]
            correlaciones.append(corr)
    return pd.Series(correlaciones, index=serie1.index)

df_parametros['correlacion_prod_presion'] = correlacion_movil(
    df_parametros['caudal_bpd'], df_parametros['presion_cabeza_psi'], ventana=28)

print("Correlación móvil entre producción y presión:")
print(df_parametros[['caudal_bpd', 'presion_cabeza_psi', 'correlacion_prod_presion']].head(15))

# =============================================================================
# EJERCICIO 12: RESUMEN Y CONCLUSIONES
# =============================================================================

print("\n\n12. RESUMEN Y CONCLUSIONES...")
print("-" * 50)

print("RESUMEN DEL ANÁLISIS DE TENDENCIAS:")
print(f"• Período analizado: {df_parametros.index.min()} a {df_parametros.index.max()}")
print(f"• Total de registros: {len(df_parametros)}")
print(f"• Anomalías detectadas: {num_anomalias}")

# Análisis de tendencias
tendencia_promedio_7d = df_parametros['tendencia_7d'].mean()
tendencia_promedio_30d = df_parametros['tendencia_30d'].mean()

print(f"\nANÁLISIS DE TENDENCIAS:")
print(f"• Tendencia promedio 7 días: {tendencia_promedio_7d:.4f}")
print(f"• Tendencia promedio 30 días: {tendencia_promedio_30d:.4f}")

if tendencia_promedio_7d > 0:
    print("  → Tendencia POSITIVA a corto plazo")
else:
    print("  → Tendencia NEGATIVA a corto plazo")

if tendencia_promedio_30d > 0:
    print("  → Tendencia POSITIVA a largo plazo")
else:
    print("  → Tendencia NEGATIVA a largo plazo")

print("\nPATRONES IDENTIFICADOS:")
print("• Variabilidad en producción: Detectada mediante rolling windows")
print("• Anomalías: Identificadas usando límites de control")
print("• Correlaciones: Relaciones dinámicas entre variables")
print("• Estacionalidad: Patrones cíclicos en los datos")

print("\nAPLICACIONES EN EL SECTOR PETROLERO:")
print("• Monitoreo de tendencias de producción")
print("• Detección temprana de anomalías")
print("• Análisis de correlaciones operacionales")
print("• Predicción de comportamientos futuros")

print("\n" + "=" * 80)
print("LABORATORIO 3 COMPLETADO")
print("=" * 80) 