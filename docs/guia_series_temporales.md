# Guía Completa: Series Temporales en Pandas
## Sesión 12 - AdP-Meridian

---

## 📋 Índice

1. [Introducción a Series Temporales](#introducción-a-series-temporales)
2. [Manipulación de DatetimeIndex](#manipulación-de-datetimeindex)
3. [Resampling y Agregación](#resampling-y-agregación)
4. [Rolling Windows y Funciones Móviles](#rolling-windows-y-funciones-móviles)
5. [Análisis de Tendencias](#análisis-de-tendencias)
6. [Detección de Anomalías](#detección-de-anomalías)
7. [Aplicaciones en el Sector Petrolero](#aplicaciones-en-el-sector-petrolero)
8. [Mejores Prácticas](#mejores-prácticas)
9. [Referencias y Recursos](#referencias-y-recursos)

---

## 🎯 Introducción a Series Temporales

### ¿Qué son las Series Temporales?

Una serie temporal es una secuencia de observaciones ordenadas cronológicamente. En el contexto petrolero, esto incluye:
- Datos de producción por hora/día
- Lecturas de sensores en tiempo real
- Eventos operacionales con timestamps
- Parámetros de pozos a lo largo del tiempo

### Importancia en el Sector Petrolero

- **Monitoreo Continuo**: Seguimiento de parámetros críticos
- **Detección de Tendencias**: Identificación de patrones de producción
- **Análisis Predictivo**: Base para modelos de forecasting
- **Optimización Operacional**: Mejora de eficiencia basada en datos históricos

---

## 🕒 Manipulación de DatetimeIndex

### Creación de DatetimeIndex

```python
import pandas as pd

# Convertir columna de fecha a datetime
df['fecha'] = pd.to_datetime(df['fecha'])

# Establecer como índice
df.set_index('fecha', inplace=True)
```

### Selección Temporal

```python
# Selección por fecha específica
df.loc['2023-01-15']

# Selección por rango
df.loc['2023-01-01':'2023-01-31']

# Selección por mes
df.loc['2023-01']

# Selección por año
df.loc['2023']
```

### Componentes Temporales

```python
# Extraer componentes
df['año'] = df.index.year
df['mes'] = df.index.month
df['dia'] = df.index.day
df['hora'] = df.index.hour
df['dia_semana'] = df.index.dayofweek
```

---

## 📊 Resampling y Agregación

### Conceptos Básicos

El resampling permite cambiar la frecuencia de los datos temporales:
- **Downsampling**: Reducir frecuencia (ej: de minutos a horas)
- **Upsampling**: Aumentar frecuencia (ej: de días a horas)

### Frecuencias Comunes

```python
# Frecuencias de tiempo
'15T'    # 15 minutos
'H'      # 1 hora
'6H'     # 6 horas
'D'      # 1 día
'W'      # 1 semana
'M'      # 1 mes
'Q'      # 1 trimestre
'Y'      # 1 año
```

### Ejemplos de Resampling

```python
# Resampling básico
df_hora = df.resample('H').mean()
df_dia = df.resample('D').sum()
df_semana = df.resample('W').mean()

# Resampling con múltiples funciones
df_agregado = df.resample('D').agg({
    'produccion': ['mean', 'std', 'min', 'max'],
    'presion': 'mean',
    'temperatura': ['min', 'max']
})
```

### Manejo de Valores Faltantes

```python
# Forward fill
df_resampled = df.resample('H').mean().fillna(method='ffill')

# Backward fill
df_resampled = df.resample('H').mean().fillna(method='bfill')

# Interpolación
df_resampled = df.resample('H').mean().interpolate(method='linear')
```

---

## 📈 Rolling Windows y Funciones Móviles

### Conceptos Básicos

Las ventanas móviles (rolling windows) permiten calcular estadísticas sobre un número fijo de observaciones consecutivas.

### Rolling Windows Básicos

```python
# Media móvil de 7 días
df['ma_7d'] = df['produccion'].rolling(window=28).mean()

# Media móvil de 30 días
df['ma_30d'] = df['produccion'].rolling(window=120).mean()

# Múltiples estadísticas
df['rolling_stats'] = df['produccion'].rolling(window=28).agg(['mean', 'std', 'min', 'max'])
```

### Parámetros Avanzados

```python
# Rolling window centrado
df['ma_centrada'] = df['produccion'].rolling(window=28, center=True).mean()

# Mínimo de períodos requeridos
df['ma_min_periods'] = df['produccion'].rolling(window=28, min_periods=14).mean()
```

### Funciones Personalizadas

```python
def calcular_tendencia(serie):
    """Calcula la pendiente de la línea de tendencia"""
    if len(serie) < 2:
        return np.nan
    x = np.arange(len(serie))
    slope = np.polyfit(x, serie, 1)[0]
    return slope

# Aplicar función personalizada
df['tendencia'] = df['produccion'].rolling(window=28).apply(calcular_tendencia)
```

---

## 🔍 Análisis de Tendencias

### Identificación de Tendencias

```python
# Cambios porcentuales
df['cambio_diario'] = df['produccion'].pct_change() * 100
df['cambio_semanal'] = df['produccion'].pct_change(periods=28) * 100

# Diferencias absolutas
df['diferencia_diaria'] = df['produccion'].diff()
```

### Análisis de Estacionalidad

```python
# Componente de tendencia (rolling window largo)
df['tendencia'] = df['produccion'].rolling(window=120).mean()

# Componente estacional (rolling window corto)
df['estacionalidad'] = df['produccion'].rolling(window=28).mean()

# Residuos
df['residuos'] = df['produccion'] - df['tendencia'] - df['estacionalidad']
```

### Correlación Móvil

```python
def correlacion_movil(serie1, serie2, ventana=28):
    """Calcula correlación móvil entre dos series"""
    correlaciones = []
    for i in range(len(serie1)):
        if i < ventana - 1:
            correlaciones.append(np.nan)
        else:
            corr = np.corrcoef(serie1.iloc[i-ventana+1:i+1], 
                              serie2.iloc[i-ventana+1:i+1])[0, 1]
            correlaciones.append(corr)
    return pd.Series(correlaciones, index=serie1.index)

df['correlacion'] = correlacion_movil(df['produccion'], df['presion'])
```

---

## ⚠️ Detección de Anomalías

### Método de Límites de Control

```python
# Calcular estadísticas móviles
df['ma_7d'] = df['produccion'].rolling(window=28).mean()
df['std_7d'] = df['produccion'].rolling(window=28).std()

# Límites de control (3 desviaciones estándar)
df['limite_superior'] = df['ma_7d'] + (3 * df['std_7d'])
df['limite_inferior'] = df['ma_7d'] - (3 * df['std_7d'])

# Detectar anomalías
df['es_anomalia'] = (
    (df['produccion'] > df['limite_superior']) |
    (df['produccion'] < df['limite_inferior'])
)
```

### Método de Percentiles

```python
# Anomalías basadas en percentiles
percentil_95 = df['produccion'].rolling(window=28).quantile(0.95)
percentil_05 = df['produccion'].rolling(window=28).quantile(0.05)

df['es_anomalia_percentil'] = (
    (df['produccion'] > percentil_95) |
    (df['produccion'] < percentil_05)
)
```

---

## 🛢️ Aplicaciones en el Sector Petrolero

### Monitoreo de Producción

```python
# Análisis de tendencias de producción
def analizar_tendencia_produccion(df):
    """Analiza tendencias en producción petrolera"""
    
    # Calcular medias móviles
    df['ma_7d'] = df['produccion'].rolling(window=28).mean()
    df['ma_30d'] = df['produccion'].rolling(window=120).mean()
    
    # Calcular cambios porcentuales
    df['cambio_diario'] = df['produccion'].pct_change() * 100
    
    # Detectar anomalías
    df['anomalia'] = detectar_anomalias(df['produccion'])
    
    return df
```

### Análisis de Sensores

```python
# Resampling de datos de sensores
def procesar_datos_sensores(df_sensores):
    """Procesa datos de sensores con resampling"""
    
    # Resampling a diferentes frecuencias
    df_hora = df_sensores.resample('H').mean()
    df_dia = df_sensores.resample('D').agg(['mean', 'std', 'min', 'max'])
    
    # Detectar valores fuera de rango
    df_hora['fuera_rango'] = (
        (df_hora['valor'] > df_hora['valor'].quantile(0.99)) |
        (df_hora['valor'] < df_hora['valor'].quantile(0.01))
    )
    
    return df_hora, df_dia
```

### Correlación con Eventos

```python
# Análisis de impacto de eventos
def analizar_impacto_eventos(df_produccion, df_eventos):
    """Analiza el impacto de eventos en la producción"""
    
    # Resampling de producción a diario
    produccion_diaria = df_produccion.resample('D')['produccion'].mean()
    
    # Agregar eventos por día
    eventos_diarios = df_eventos.resample('D').agg({
        'duracion_horas': 'sum',
        'impacto_produccion': 'sum',
        'tipo_evento': 'count'
    })
    
    # Combinar datos
    analisis = pd.concat([produccion_diaria, eventos_diarios], axis=1)
    
    # Calcular correlaciones
    correlaciones = analisis.corr()
    
    return analisis, correlaciones
```

---

## ✅ Mejores Prácticas

### 1. Preparación de Datos

- **Validar fechas**: Asegurar que las fechas estén en formato correcto
- **Ordenar datos**: Los datos deben estar ordenados cronológicamente
- **Manejar valores faltantes**: Decidir estrategia (interpolación, forward fill, etc.)

### 2. Selección de Ventanas

- **Ventanas cortas**: Para detectar cambios rápidos (1-7 días)
- **Ventanas medias**: Para tendencias (7-30 días)
- **Ventanas largas**: Para patrones estacionales (30+ días)

### 3. Interpretación de Resultados

- **Contexto operacional**: Considerar eventos que afectan los datos
- **Estacionalidad**: Identificar patrones cíclicos
- **Tendencias**: Distinguir entre tendencias reales y ruido

### 4. Visualización

```python
import matplotlib.pyplot as plt

# Gráfico de series temporales con múltiples líneas
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Producción original
axes[0, 0].plot(df.index, df['produccion'], 'b-', alpha=0.7)
axes[0, 0].plot(df.index, df['ma_7d'], 'r-', linewidth=2)
axes[0, 0].set_title('Producción con Media Móvil')

# Cambios porcentuales
axes[0, 1].plot(df.index, df['cambio_diario'], 'g-')
axes[0, 1].axhline(y=0, color='black', linestyle='--')
axes[0, 1].set_title('Cambios Porcentuales')

# Anomalías
axes[1, 0].plot(df.index, df['produccion'], 'b-', alpha=0.7)
axes[1, 0].scatter(df[df['es_anomalia']].index, 
                   df[df['es_anomalia']]['produccion'], 
                   color='red', s=50)
axes[1, 0].set_title('Detección de Anomalías')

# Correlación móvil
axes[1, 1].plot(df.index, df['correlacion'], 'purple')
axes[1, 1].set_title('Correlación Móvil')

plt.tight_layout()
plt.show()
```

---

## 📚 Referencias y Recursos

### Documentación Oficial

- [Pandas Time Series Guide](https://pandas.pydata.org/docs/user_guide/timeseries.html)
- [Pandas Resampling](https://pandas.pydata.org/docs/user_guide/timeseries.html#resampling)
- [Pandas Rolling Windows](https://pandas.pydata.org/docs/user_guide/window.html)

### Libros Recomendados

- "Time Series Analysis: Forecasting and Control" - Box, Jenkins, Reinsel
- "Practical Time Series Analysis" - Aileen Nielsen
- "Python for Data Analysis" - Wes McKinney

### Artículos Técnicos

- "Time Series Analysis in the Oil and Gas Industry"
- "Anomaly Detection in Industrial Time Series Data"
- "Rolling Window Analysis for Production Optimization"

### Herramientas Adicionales

- **statsmodels**: Para análisis estadístico avanzado
- **scikit-learn**: Para detección de anomalías con ML
- **prophet**: Para forecasting de series temporales
- **tslearn**: Para análisis de series temporales con ML

---

## 🎓 Ejercicios Prácticos

### Ejercicio 1: Análisis Básico
1. Cargar datos de producción
2. Convertir a DatetimeIndex
3. Calcular medias móviles de 7 y 30 días
4. Visualizar tendencias

### Ejercicio 2: Resampling
1. Cargar datos de sensores
2. Realizar resampling a diferentes frecuencias
3. Comparar estadísticas por frecuencia
4. Analizar impacto en la información

### Ejercicio 3: Detección de Anomalías
1. Implementar límites de control
2. Detectar anomalías en producción
3. Analizar correlación con eventos
4. Generar reporte de anomalías

### Ejercicio 4: Análisis Integrado
1. Combinar datos de producción, sensores y eventos
2. Calcular correlaciones móviles
3. Identificar patrones temporales
4. Generar dashboard de monitoreo

---

## 🚀 Próximos Pasos

### Temas Avanzados

- **Forecasting**: Modelos predictivos para series temporales
- **Machine Learning**: Detección de anomalías con algoritmos ML
- **Deep Learning**: Redes neuronales para series temporales
- **Optimización**: Aplicación de análisis temporal para optimización operacional

### Implementación en Producción

- **Automatización**: Scripts automáticos para análisis diario
- **Dashboards**: Visualizaciones interactivas en tiempo real
- **Alertas**: Sistema de notificaciones para anomalías
- **Integración**: Conexión con sistemas SCADA y bases de datos

---

*Esta guía proporciona una base sólida para el análisis de series temporales en el contexto petrolero. Recuerda siempre considerar el contexto operacional y validar los resultados con expertos del dominio.* 