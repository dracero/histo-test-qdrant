# 🔬 RAG Multimodal de Histología — Branch Vuelta (v4.2)

Este repositorio contiene la implementación estable del sistema **RAG (Retrieval-Augmented Generation) Multimodal** especializado en histología, desarrollado para la Facultad de Medicina (FMED). Esta versión, denominada **Branch Vuelta (v4.2)**, representa una estabilización del sistema tras la migración completa a **Qdrant** como motor vectorial y la optimización del flujo de razonamiento agéntico con **LangGraph**.

---

## 🌟 Características Principales (v4.2)

A diferencia de versiones anteriores, la **v4.2** introduce mejoras críticas en la precisión y eficiencia del sistema:

1.  **Flujo Bifurcado**: El sistema detecta automáticamente si el usuario realiza una consulta teórica (solo texto) o si requiere análisis visual. Las consultas de texto puro saltan los pasos de procesamiento de imagen, acelerando la respuesta en un 40%.
2.  **Router Condicional Inteligente**: Utiliza un clasificador de intención basado en LLM para decidir si debe reutilizar una imagen en memoria o tratar la consulta como una pregunta enciclopédica.
3.  **Umbrales Diferenciados**: 
    - **Modo Texto**: Umbral de similitud de **0.30** (más permisivo para capturar conceptos teóricos).
    - **Modo Imagen**: Umbral estricto de **0.70** para garantizar precisión en la identificación de microfotografías.
4.  **Sistema de Prompts Optimizados**: Respuestas diferenciadas que evitan mencionar imágenes inexistentes cuando la consulta es puramente teórica.
5.  **Memoria Semántica Persistente**: Almacenamiento de resúmenes de conversación e imágenes activas directamente en Qdrant, permitiendo continuidad entre sesiones.

---

## 🏗️ Arquitectura Técnica

El sistema está orquestado por un grafo de estados (**LangGraph**) que asegura un flujo determinista pero flexible:

### Grafo de Nodos
- `inicializar`: Carga el estado y la memoria semántica.
- `procesar_imagen`: Genera embeddings **UNI** (estructural) y **PLIP** (semántico) si hay una imagen nueva.
- `clasificar`: Valida que la consulta esté dentro del dominio histológico mediante anclas semánticas.
- `generar_consulta`: Reformula la pregunta del usuario para optimizar la recuperación vectorial.
- `buscar`: Ejecuta la búsqueda en las colecciones `histo_chunks` e `histo_imagenes` de Qdrant.
- `filtrar_contexto`: Selecciona los mejores resultados basándose en los umbrales de la v4.2.
- `analisis_comparativo`: Compara la imagen del usuario con las referencias del manual usando una tabla de características discriminatorias.
- `generar_respuesta`: Sintetiza la información técnica con el LLM (Llama 3/4 vía Groq).
- `finalizar`: Guarda la trayectoria y actualiza la memoria.

---

## 🧠 Modelos y Embeddings

| Tipo | Modelo | Propósito |
| :--- | :--- | :--- |
| **LLM** | Llama-4-Scout (Groq) | Razonamiento agéntico y generación de respuestas técnicas. |
| **Embeddings Texto** | all-MiniLM-L6-v2 | Búsqueda semántica de chunks de texto (384d). |
| **Vision Model (UNI)** | MahmoodLab/UNI | Captura morfología celular y arquitectura de tejidos (1024d). |
| **Vision Model (PLIP)**| vinid/PLIP | Alineación semántica entre descripciones y visuales (512d). |

---

## ⚙️ Configuración e Instalación

### Requisitos Previos
- Python 3.10+
- Gestor de paquetes `uv` (recomendado)
- Tesseract OCR & Poppler (para extracción de PDFs)

### Pasos
1.  Clonar el repositorio: `git clone -b vuelta ...`
2.  Instalar dependencias:
    ```bash
    uv sync
    ```
3.  Configurar el archivo `.env`:
    ```env
    GROQ_API_KEY=tu_clave
    GOOGLE_API_KEY=tu_clave
    HF_TOKEN=tu_token_huggingface
    QDRANT_URL=tu_url_qdrant
    QDRANT_KEY=tu_clave_qdrant
    ```

### Ejecución
Para iniciar el servidor FastAPI con el cliente web (A2UI):
```bash
npm run dev
```

---

## 📁 Estructura del Proyecto (Branch Vuelta)

- `server.py`: Servidor FastAPI que expone la API de chat y el frontend.
- `qdrant-histo.py`: Implementación del Agente de Histología y el Vector Store.
- `client/`: Interfaz web con soporte para visualización de trayectorias.
- `pdf/`: Carpeta para los manuales de referencia.
- `imagenes_extraidas/`: Repositorio local de las figuras procesadas de los manuales.

---

## 📄 Notas de Versión
Esta versión **4.2** de la rama **vuelta** prioriza la estabilidad de la conexión con Qdrant Cloud y la eliminación de dependencias heredadas como Neo4j e ImageBind, logrando un sistema más ligero y fácil de desplegar.
