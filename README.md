# SESIÓN 12: SERIES TEMPORALES EN PANDAS

## Bienvenidos a la Sesión 12

Este laboratorio está diseñado para consultores, analistas de datos y cualquier persona en Meridian Consulting que haya completado las Sesiones 7, 8, 9 y 11 o tenga conocimientos sólidos de Pandas y manipulación de datos.

### 📋 Información del Laboratorio

- **Duración:** 2 horas
- **Nivel:** Avanzado
- **Requisitos previos:** Sesiones 7, 8, 9 y 11 completadas o conocimientos equivalentes en Pandas

### 🎯 Objetivos de Aprendizaje

Al finalizar este laboratorio, serás capaz de:

1. **Dominar** la manipulación de datos temporales con DatetimeIndex
2. **Aplicar** técnicas de resampling para agregación y frecuencias temporales
3. **Implementar** rolling windows y funciones móviles para análisis de tendencias
4. **Realizar** operaciones específicas para series temporales (diferencias, cambios porcentuales)
5. **Detectar** estacionalidad y tendencias en datos de producción petrolera
6. **Normalizar** datos en diferentes escalas temporales para análisis comparativos

### 🛠️ Preparación del Entorno

#### Verificación de Dependencias

```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

print(f"Versión de Pandas: {pd.__version__}")
print(f"Versión de NumPy: {np.__version__}")

# Configuración para mejor visualización
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 50)

# Configuración de matplotlib para series temporales
plt.style.use('seaborn-v0_8')
```

### 📚 Estructura del Laboratorio

| Tiempo | Tema | Descripción |
|--------|------|-------------|
| 0:00-0:15 | **Introducción y Repaso** | Conceptos de series temporales y casos de uso petroleros |
| 0:15-0:45 | **DatetimeIndex y Manipulación Temporal** | Creación, indexación y operaciones básicas |
| 0:45-1:15 | **Resampling y Agregación** | Cambio de frecuencias y agregación temporal |
| 1:15-1:45 | **Rolling Windows y Funciones Móviles** | Análisis de tendencias y patrones |
| 1:45-2:00 | **Proyecto Integrado** | Análisis completo de series temporales de producción |

### 💼 Casos de Uso en el Sector Petrolero

Durante este laboratorio trabajaremos con escenarios reales del sector:

- **Análisis de Tendencias de Producción**: Identificación de patrones en producción de pozos
- **Detección de Patrones Cíclicos**: Análisis de estacionalidad en operaciones
- **Normalización Temporal**: Comparación de datos en diferentes escalas de tiempo
- **Análisis de Comportamiento**: Seguimiento de cambios en rendimiento de pozos
- **Previsión Operativa**: Base para modelos predictivos basados en tendencias históricas

### 🚀 Nuevas Capacidades que Desarrollarás

#### **Manipulación de DatetimeIndex**
```python
# Creación de índice temporal
df['fecha'] = pd.to_datetime(df['fecha'])
df.set_index('fecha', inplace=True)

# Selección temporal
df['2023-01-01':'2023-12-31']
df.loc['2023-06']  # Junio 2023
```

#### **Resampling y Agregación**
```python
# Cambio de frecuencia
df_diario = df.resample('D').mean()  # Promedio diario
df_mensual = df.resample('M').sum()  # Suma mensual

# Agregación personalizada
df_trimestral = df.resample('Q').agg({
    'produccion': 'sum',
    'presion': 'mean',
    'temperatura': ['min', 'max', 'mean']
})
```

#### **Rolling Windows y Funciones Móviles**
```python
# Media móvil de 7 días
df['produccion_ma_7d'] = df['produccion'].rolling(window=7).mean()

# Tendencia móvil
df['tendencia_30d'] = df['produccion'].rolling(window=30).apply(
    lambda x: np.polyfit(range(len(x)), x, 1)[0]
)
```

#### **Operaciones Específicas para Series Temporales**
```python
# Diferencias temporales
df['cambio_produccion'] = df['produccion'].diff()
df['cambio_porcentual'] = df['produccion'].pct_change()

# Detección de estacionalidad
df['estacionalidad'] = df['produccion'].rolling(window=365).mean()
```

### 📁 Estructura de Archivos

```
sesion-12/
├── datos/                     # Datasets de series temporales del sector petrolero
│   ├── produccion_historica.csv   # Datos históricos de producción con timestamps
│   ├── sensores_temporales.csv    # Datos de sensores con series temporales
│   ├── eventos_operacionales.csv  # Eventos y mantenimientos con fechas
│   └── parametros_pozos.csv       # Parámetros operativos con series temporales
├── ejercicios/               # Ejercicios progresivos de series temporales
│   ├── lab_01_manipulacion_temporal.py
│   ├── lab_02_resampling_agregacion.py
│   └── lab_03_analisis_tendencias.py
├── soluciones/              # Soluciones completas
├── demos/                   # Scripts de demostración
└── docs/                   # Documentación completa
```

### 🎓 Metodología de Aprendizaje Avanzada

1. **Análisis temporal progresivo**: Desde operaciones básicas hasta análisis complejos
2. **Práctica con datos reales**: Series temporales de producción petrolera
3. **Visualización temporal**: Gráficos específicos para series temporales
4. **Detección de patrones**: Identificación de tendencias y estacionalidad

### 💡 Valor para Meridian Consulting

Al completar esta sesión, contribuirás a:

- **Identificación precisa de tendencias**: Análisis de patrones en producción petrolera
- **Normalización temporal**: Capacidad para comparar datos en diferentes escalas
- **Análisis de estacionalidad**: Comprensión de comportamientos cíclicos
- **Mejora en previsiones**: Base sólida para modelos predictivos

### 🔗 Enlaces de Referencia

- [Pandas Time Series Guide](https://pandas.pydata.org/docs/user_guide/timeseries.html)
- [Pandas Resampling](https://pandas.pydata.org/docs/user_guide/timeseries.html#resampling)
- [Time Series Analysis with Pandas](https://pandas.pydata.org/docs/user_guide/timeseries.html#time-series-analysis)

---

¡Estás listo para dominar el análisis de series temporales en el contexto petrolero! Comienza con los ejercicios para desarrollar habilidades avanzadas en análisis temporal. 