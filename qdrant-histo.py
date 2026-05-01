# =============================================================================
# RAG Multimodal de Histología con ImageBind + Neo4j — VERSIÓN 4.2
# =============================================================================
# Cambios sobre v4.1:
#   1. REFACTORING CONSULTAS DE TEXTO: Flujo bifurcado en LangGraph.
#      Consultas sin imagen saltan procesar_imagen y analisis_comparativo.
#   2. ROUTER CONDICIONAL: _route_por_modo decide el camino del grafo.
#   3. UMBRALES DIFERENCIADOS: texto puro usa umbral 0.45 (más permisivo),
#      modo imagen mantiene 0.6 para texto y 0.45 para imágenes.
#   4. SYSTEM PROMPT DIFERENCIADO: modo texto tiene prompt optimizado para
#      respuestas enciclopédicas sin referencias a imágenes del usuario.
#   5. MEJOR MANEJO DE NO-CONTEXTO: en modo texto, mensaje amable indicando
#      que no se encontró info, en vez de "fuera de dominio".
# =============================================================================
# Cambios v4.1:
#   1. RAGAS eliminado completamente.
#   2. MEMORIA DE IMAGEN PERSISTENTE.
#   3. CLASIFICADOR SEMÁNTICO.
#   4. MEMORIA SIEMPRE ACTUALIZADA.
#   5. MODO INTERACTIVO MEJORADO.
# =============================================================================

import os
import json
import time
import asyncio
import nest_asyncio
import torch
import numpy as np
import re
from typing import TypedDict, Annotated, List, Dict, Any, Optional, Tuple
from PIL import Image
import base64
import glob
from dotenv import load_dotenv

# Cargar variables de entorno desde .env
load_dotenv()

# UNI & PLIP
import timm
from transformers import CLIPProcessor, CLIPModel
from huggingface_hub import login

# Verificar HF_TOKEN
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN:
    try:
        login(token=HF_TOKEN)
        print("✅ Logueado en Hugging Face")
    except Exception as e:
        print(f"⚠️ Error login HF: {e}")
else:
    print("⚠️ HF_TOKEN no encontrado en .env (necesario para UNI)")

# Neo4j
from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchAny, MatchValue

import fitz # PyMuPDF
from pdf2image import convert_from_path
import pytesseract

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
import operator

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import uuid

# Wrapper para leer variables de entorno (compatible con .env)
class userdata:
    @staticmethod
    def get(key):
        return os.environ.get(key)

nest_asyncio.apply()

# =============================================================================
# CONFIGURACIÓN GLOBAL
# =============================================================================
# CONFIGURACIÓN GLOBAL
SIMILARITY_THRESHOLD  = 0.70  # Aumentado de 0.45 a 0.70 para mayor precisión en imágenes médicas
# Dimensiones de embeddings
DIM_TEXTO = 384
DIM_TEXTO_GEMINI = 3072
DIM_IMG_UNI      = 1024
DIM_IMG_PLIP     = 512

DIRECTORIO_IMAGENES   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "imagenes_extraidas")
DIRECTORIO_PDFS       = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pdf")

# Colecciones Qdrant
COLLECTION_CHUNKS   = "histo_chunks"
COLLECTION_IMAGENES = "histo_imagenes"

# Índices Neo4j
INDEX_TEXTO = "histo_text"      # Text Vector
INDEX_UNI   = "histo_img_uni"   # UNI Image
INDEX_PLIP  = "histo_img_plip"  # PLIP Image

NEO4J_GRAPH_DEPTH     = 2
SIMILAR_IMG_THRESHOLD = 0.85 # Más alto para modelos especializados

FEATURES_DISCRIMINATORIAS = [
    "presencia/ausencia de lumen central",
    "estratificación celular (capas concéntricas vs difusa)",
    "tipo de queratinización (parakeratosis, ortoqueratosis, ninguna)",
    "aspecto del núcleo (picnótico, fantasma, ausente, vesicular)",
    "células fantasma (sí/no)",
    "material amorfo central (sí/no y aspecto)",
    "patrón de tinción H&E (eosinofilia, basofilia)",
    "tamaño estimado de la estructura",
    "tejido circundante (estroma, epitelio, piel, otro)",
    "reacción inflamatoria perilesional (sí/no, tipo)",
    # Cartílago vs Hueso — features críticas
    "tipo de matriz extracelular: homogénea/vítrea (cartílago) vs calcificada con canalículos (hueso)",
    "células en lagunas: condrocitos (redondeados, en nidos/isógenos) vs osteocitos (estrellados, aislados con prolongaciones)",
    "presencia de pericondrio (cartílago) vs periostio (hueso)",
    "sistemas de Havers / osteonas (exclusivo de hueso compacto)",
    "nidos isógenos / grupos celulares (exclusivo de cartílago)",
    "tinción de la matriz: basófila-azulada (cartílago hialino) vs eosinófila-rosada (hueso)",
    "fibras visibles en la matriz: ausentes (hialino), elásticas (elástico), colágenas gruesas (fibroso)",
    "vascularización: avascular (cartílago, excepto fibrocartílago) vs muy vascularizado (hueso)",
]

# Anclas semánticas para el clasificador de dominio
ANCLAS_SEMANTICAS_HISTOLOGIA = [
    "histología tejido celular microscopía",
    "tipos de tejido epitelial conectivo muscular nervioso",
    "coloración hematoxilina eosina H&E tinción histológica",
    "estructuras celulares núcleo citoplasma membrana",
    "diagnóstico diferencial patología biopsia",
    "glándulas epitelio estratificado cilíndrico simple",
    "identificar tejido muestra microscópica",
    "¿qué tipo de tejido es este?",
    "¿cuál es la estructura observada en la imagen?",
    "clasificar célula estructura histológica",
    "tumor quiste folículo cuerpo lúteo albicans",
    "corte histológico preparación muestra lámina",
    "tejido epitelial simple cilíndrico estratificado pseudoestratificado",
    "tejido conectivo laxo denso adiposo cartilaginoso óseo",
    "tejido muscular liso estriado cardíaco esquelético",
    "tejido nervioso neurona glía axón dendrita",
]

def _safe(value, default: str = "") -> str:
    return value if isinstance(value, str) and value else default

async def invoke_con_reintento(llm, messages, max_retries=5):
    import asyncio
    for attempt in range(max_retries):
        try:
            return await llm.ainvoke(messages)
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "503" in err_str:
                if attempt < max_retries - 1:
                    espera = 15 * (attempt + 1)
                    print(f"   ⚠️ Límite de cuota API/Servidor Ocupado (429/503) - reintentando en {espera}s... (Intento {attempt+1}/{max_retries})")
                    await asyncio.sleep(espera)
                else:
                    raise e
            else:
                raise e

def invoke_con_reintento_sync(llm, messages, max_retries=5):
    import time
    for attempt in range(max_retries):
        try:
            return llm.invoke(messages)
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "503" in err_str:
                if attempt < max_retries - 1:
                    espera = 15 * (attempt + 1)
                    print(f"   ⚠️ Límite de cuota API/Servidor Ocupado (429/503) - reintentando en {espera}s... (Intento {attempt+1}/{max_retries})")
                    time.sleep(espera)
                else:
                    raise e
            else:
                raise e

def embed_query_con_reintento(embeddings, texto: str, max_retries=5):
    import time
    for attempt in range(max_retries):
        try:
            return embeddings.embed_query(texto)
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "503" in err_str:
                if attempt < max_retries - 1:
                    espera = 15 * (attempt + 1)
                    print(f"   ⚠️ Límite de cuota API en embeddings (429/503) - reintentando en {espera}s... (Intento {attempt+1}/{max_retries})")
                    time.sleep(espera)
                else:
                    raise e
            else:
                raise e

def embed_documents_con_reintento(embeddings, textos: list, max_retries=5):
    import time
    for attempt in range(max_retries):
        try:
            return embeddings.embed_documents(textos)
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "503" in err_str:
                if attempt < max_retries - 1:
                    espera = 15 * (attempt + 1)
                    print(f"   ⚠️ Límite de cuota API en embeddings (429/503) - reintentando en {espera}s... (Intento {attempt+1}/{max_retries})")
                    time.sleep(espera)
                else:
                    raise e
            else:
                raise e


# =============================================================================
# LANGSMITH
# =============================================================================

def setup_langsmith_environment():
    config = {
        "LANGCHAIN_TRACING_V2": "true",
        "LANGCHAIN_API_KEY":    userdata.get("LANGSMITH_API_KEY"),
        "LANGCHAIN_ENDPOINT":   "https://api.smith.langchain.com",
        "LANGCHAIN_PROJECT":    "rag_histologia_neo4j_v4"
    }
    for key, value in config.items():
        if value:
            os.environ[key] = value
    try:
        from langsmith import traceable, Client
        client = Client()
        print(f"✅ LangSmith — Proyecto: {os.environ.get('LANGCHAIN_PROJECT')}")
        return True, traceable, client
    except Exception as e:
        print(f"⚠️ LangSmith no disponible: {e}")
        def dummy_traceable(*args, **kwargs):
            def decorator(func): return func
            if len(args) == 1 and callable(args[0]): return args[0]
            return decorator
        return False, dummy_traceable, None

LANGSMITH_ENABLED, traceable, langsmith_client = setup_langsmith_environment()


# =============================================================================
# WRAPPERS DE MODELOS (PLIP & UNI)
# =============================================================================

def preprocess_image_for_embedding(image_path: str) -> Image.Image:
    """
    Preprocess USER image before generating embeddings.
    Applies only contrast/brightness enhancement — NO magnification.
    
    Las imágenes del usuario NO se magnifican porque los embeddings indexados
    provienen de imágenes ya magnificadas durante la extracción del PDF.
    Magnificar la imagen del usuario distorsionaría la comparación de similitud.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Preprocessed PIL Image (same resolution, enhanced contrast/brightness)
    """
    try:
        from PIL import ImageEnhance
        
        # Load image
        img = Image.open(image_path).convert('RGB')
        
        # Apply contrast enhancement (factor 1.2)
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.2)
        
        # Apply brightness enhancement (factor 1.1)
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(1.1)
        
        # NO magnification — user images keep their original resolution
        # to maintain comparability with indexed (already magnified) images
        
        return img
        
    except Exception as e:
        print(f"⚠️ Error preprocessing image: {e}, using original")
        return Image.open(image_path).convert('RGB')


class PlipWrapper:
    def __init__(self, device):
        self.device = device
        self.model = None
        self.processor = None

    def load(self):
        print("🔄 Cargando PLIP (vinid/plip)...")
        try:
            self.model = CLIPModel.from_pretrained("vinid/plip").to(self.device).eval()
            self.processor = CLIPProcessor.from_pretrained("vinid/plip")
            print("✅ PLIP cargado")
        except Exception as e:
            print(f"❌ Error cargando PLIP: {e}")

    def embed_image(self, image_path: str, preprocess: bool = True) -> np.ndarray:
        if not self.model: return np.zeros(DIM_IMG_PLIP)
        try:
            # Apply preprocessing for user images
            if preprocess:
                image = preprocess_image_for_embedding(image_path)
            else:
                image = Image.open(image_path).convert('RGB')
            
            inputs = self.processor(images=image, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(self.device)
            with torch.inference_mode():
                vision_out = self.model.vision_model(pixel_values=pixel_values)
                pooled = vision_out.pooler_output
                image_features = self.model.visual_projection(pooled)  # [1, 512]
            return image_features.cpu().numpy().flatten()
        except Exception as e:
            print(f"⚠️ Error embedding PLIP: {e}")
            return np.zeros(DIM_IMG_PLIP)

class UniWrapper:
    def __init__(self, device):
        self.device = device
        self.model = None
        self.transform = None

    def load(self):
        print("🔄 Cargando UNI (MahmoodLab)...")
        try:
            # UNI usa timm con hf_hub
            self.model = timm.create_model(
                "hf_hub:MahmoodLab/UNI", 
                pretrained=True, 
                init_values=1e-5, 
                dynamic_img_size=True
            )
            self.model.to(self.device).eval()
            
            # Transformación estándar de UNI (ViT-L/16)
            from timm.data import resolve_data_config
            from timm.data.transforms_factory import create_transform
            config = resolve_data_config(self.model.pretrained_cfg, model=self.model)
            self.transform = create_transform(**config)
            print("✅ UNI cargado")
        except Exception as e:
            print(f"❌ Error cargando UNI: {e}")

    def embed_image(self, image_path: str, preprocess: bool = True) -> np.ndarray:
        if not self.model: return np.zeros(DIM_IMG_UNI)
        try:
            # Apply preprocessing for user images
            if preprocess:
                image = preprocess_image_for_embedding(image_path)
            else:
                image = Image.open(image_path).convert('RGB')
            
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            with torch.inference_mode():
                emb = self.model(image_tensor) # UNI returns raw features [1, 1024]
            return emb.cpu().numpy().flatten()
        except Exception as e:
            print(f"⚠️ Error embedding UNI: {e}")
            return np.zeros(DIM_IMG_UNI)


# =============================================================================
# CLIENTE NEO4J (Adaptado para 3 índices)
# =============================================================================

class QdrantVectorStore:
    """
    Wrapper para Qdrant que encapsula:
    - Conexión (Qdrant Cloud o local)
    - Creación del esquema (colecciones para chunks e imágenes)
    - Indexación (texto e imágenes)
    - Búsqueda (híbrida, vectorial, semántica, etc)
    """

    def __init__(self, url: str, api_key: str):
        self.url = url
        self.api_key = api_key
        self.client = QdrantClient(url=url, api_key=api_key, timeout=60)

    async def connect(self):
        try:
            self.client.get_collections()
            print(f"✅ Qdrant conectado: {self.url}")
        except Exception as e:
            raise ConnectionError(f"No se pudo conectar a Qdrant: {e}")

    async def close(self):
        try:
            self.client.close()
        except Exception:
            pass

    async def run(self, query: str, params: dict = None):
        # Stub for compatibility with neo4j client interface where used minimally
        pass

    async def crear_esquema(self):
        print("🏗️ Creando esquema Qdrant (chunks e imágenes)...")
        
        # Colección de chunks (Texto 384d)
        try:
            info = self.client.get_collection(COLLECTION_CHUNKS)
            config = info.config.params.vectors
            if isinstance(config, dict):
                vec_config = config.get("texto", config.get("texto_emb"))
                dims = getattr(vec_config, "size", 0) if vec_config else 0
            else:
                dims = getattr(config, "size", 0)
                
            if dims != DIM_TEXTO:
                print(f"   ⚠️ Dimensión incorrecta en {COLLECTION_CHUNKS}. Recreando...")
                self.client.delete_collection(COLLECTION_CHUNKS)
                raise Exception("Recreate")
            print(f"   ✅ Colección '{COLLECTION_CHUNKS}' ya existe")
        except Exception as e:
            if "Recreate" not in str(e) and "Not found" not in str(e):
                # Ignore the exception if it's just 'Not found', else might be a real issue but we will try to create anyway
                pass
            try:
                self.client.create_collection(
                    collection_name=COLLECTION_CHUNKS,
                    vectors_config=VectorParams(size=DIM_TEXTO, distance=Distance.COSINE),
                )
                print(f"   ✅ Colección '{COLLECTION_CHUNKS}' creada")
            except Exception as e2:
                if "already exists" in str(e2):
                    print(f"   ✅ Colección '{COLLECTION_CHUNKS}' ya existe")
                else:
                    raise e2

        # Colección de imágenes (UNI 1024d + PLIP 512d + Texto 384d)
        try:
            col_info = self.client.get_collection(COLLECTION_IMAGENES)
            print(f"   ✅ Colección '{COLLECTION_IMAGENES}' ya existe")
        except Exception:
            try:
                self.client.create_collection(
                    collection_name=COLLECTION_IMAGENES,
                    vectors_config={
                        "uni":       VectorParams(size=DIM_IMG_UNI,  distance=Distance.COSINE),
                        "plip":      VectorParams(size=DIM_IMG_PLIP, distance=Distance.COSINE),
                        "texto_emb": VectorParams(size=DIM_TEXTO,    distance=Distance.COSINE),
                    },
                )
                print(f"   ✅ Colección '{COLLECTION_IMAGENES}' creada")
            except Exception as e2:
                if "already exists" in str(e2):
                    print(f"   ✅ Colección '{COLLECTION_IMAGENES}' ya existe")
                else:
                    raise e2

        # Índices de Payload
        for field in ["tejidos", "estructuras", "tinciones", "fuente"]:
            try:
                self.client.create_payload_index(
                    collection_name=COLLECTION_CHUNKS,
                    field_name=field,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
            except Exception: pass

        for field in ["fuente", "pagina_str"]:
            try:
                self.client.create_payload_index(
                    collection_name=COLLECTION_IMAGENES,
                    field_name=field,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
            except Exception: pass

        try:
            self.client.create_payload_index(
                collection_name=COLLECTION_CHUNKS,
                field_name="pagina",
                field_schema=models.PayloadSchemaType.INTEGER,
            )
        except Exception: pass

        print("✅ Esquema Qdrant listo (2 colecciones + payload index)")

    async def upsert_pdf(self, nombre: str):
        pass # Metadata implicit in points

    async def upsert_chunk(self, chunk_id: str, texto: str, fuente: str,
                            chunk_idx: int, embedding: list,
                            entidades: dict, pagina: int = 0, imagenes_pagina: list = None):
        import uuid
        point = PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id)),
            vector=embedding,
            payload={
                "id":               chunk_id,
                "texto":            texto,
                "fuente":           fuente,
                "chunk_idx":        chunk_idx,
                "pagina":           pagina,
                "imagenes_pagina":  [p for p in (imagenes_pagina or []) if __import__('os').path.exists(p)],
                "tipo":             "texto",
                "tejidos":          entidades.get("tejidos", []),
                "estructuras":      entidades.get("estructuras", []),
                "tinciones":        entidades.get("tinciones", []),
            },
        )
        self.client.upsert(collection_name=COLLECTION_CHUNKS, points=[point])

    async def upsert_imagen(self, imagen_id: str, path: str, fuente: str,
                             pagina: int, ocr_text: str, texto_pagina: str,
                             emb_uni: list, emb_plip: list, emb_texto: list = None,
                             caption: str = "", nombre_archivo: str = "", etiqueta: str = ""):
        import uuid
        vectors = {
            "uni":  emb_uni,
            "plip": emb_plip,
        }
        if emb_texto and any(v != 0.0 for v in emb_texto):
            vectors["texto_emb"] = emb_texto
        else:
            vectors["texto_emb"] = [0.0] * DIM_TEXTO

        point = PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_DNS, imagen_id)),
            vector=vectors,
            payload={
                "id":             imagen_id,
                "path":           path,
                "fuente":         fuente,
                "pagina":         pagina,
                "pagina_str":     str(pagina),
                "ocr_text":       ocr_text,
                "texto_pagina":   texto_pagina[:3000],
                "caption":        caption,
                "nombre_archivo": nombre_archivo,
                "etiqueta":       etiqueta,
                "tipo":           "imagen",
            },
        )
        self.client.upsert(collection_name=COLLECTION_IMAGENES, points=[point])

    async def crear_relaciones_similitud(self, umbral: float = SIMILAR_IMG_THRESHOLD):
        # Qdrant realiza búsqueda vectorial al vuelo, no requiere pre-computar relaciones
        pass

    async def busqueda_vectorial(self, embedding: list,
                                  index_name: str, top_k: int = 10) -> list:
        is_text_index = index_name == INDEX_TEXTO
        try:
            if is_text_index:
                results = self.client.query_points(
                    collection_name=COLLECTION_CHUNKS, query=embedding, limit=top_k
                ).points
                return [{
                    "id": str(r.id), "texto": r.payload.get("texto", ""), "fuente": r.payload.get("fuente", ""),
                    "tipo": "texto", "imagen_path": None, "similitud": r.score,
                    "nombre_archivo": "", "etiqueta": "",
                    "imagenes_pagina": r.payload.get("imagenes_pagina", []),
                    "pagina": r.payload.get("pagina"),
                } for r in results]
            else:
                # index_name represents UNI or PLIP using vector
                using_vector = "uni" if index_name == INDEX_UNI else "plip"
                results = self.client.query_points(
                    collection_name=COLLECTION_IMAGENES, query=embedding, using=using_vector, limit=top_k
                ).points
                out = []
                for r in results:
                    nombre_archivo = r.payload.get("nombre_archivo", "").lower()
                    if "_full." in nombre_archivo:
                        continue
                    caption = r.payload.get("caption", "")
                    texto_pag = r.payload.get("texto_pagina", "")
                    ocr = r.payload.get("ocr_text", "")
                    texto = caption if caption else (texto_pag if texto_pag else ocr)
                    out.append({
                        "id": str(r.id), "texto": texto, "fuente": r.payload.get("fuente", ""),
                        "tipo": "imagen", "imagen_path": r.payload.get("path"), "similitud": r.score,
                        "nombre_archivo": r.payload.get("nombre_archivo", ""),
                        "etiqueta": r.payload.get("etiqueta", ""), "texto_pagina": texto_pag,
                        "pagina": r.payload.get("pagina")
                    })
                return out
        except Exception as e:
            print(f"⚠️ Error búsqueda vectorial Qdrant {index_name}: {e}")
            return []

    async def busqueda_chunks_por_pagina(self, fuente: str, pagina: int, top_k: int = 3) -> list:
        try:
            results, _ = self.client.scroll(
                collection_name=COLLECTION_CHUNKS,
                scroll_filter=Filter(must=[
                    FieldCondition(key="fuente", match=MatchValue(value=fuente)),
                    FieldCondition(key="pagina", match=MatchValue(value=pagina)),
                ]),
                limit=top_k,
            )
            return [{
                "id": str(r.id), "texto": r.payload.get("texto", ""), "fuente": r.payload.get("fuente", ""),
                "tipo": "texto", "imagen_path": None, "similitud": 0.80,
                "nombre_archivo": "", "etiqueta": "",
                "imagenes_pagina": r.payload.get("imagenes_pagina", []),
                "pagina": r.payload.get("pagina"),
            } for r in results]
        except Exception as e:
            print(f"⚠️ Error búsqueda chunks por página: {e}")
            return []

    async def busqueda_por_entidades(self, entidades: dict, top_k: int = 10) -> list:
        tejidos = entidades.get("tejidos", [])
        estructuras = entidades.get("estructuras", [])
        tinciones = entidades.get("tinciones", [])
        if not any([tejidos, estructuras, tinciones]): return []
        conditions = []
        if tejidos: conditions.append(FieldCondition(key="tejidos", match=MatchAny(any=tejidos)))
        if estructuras: conditions.append(FieldCondition(key="estructuras", match=MatchAny(any=estructuras)))
        if tinciones: conditions.append(FieldCondition(key="tinciones", match=MatchAny(any=tinciones)))
        try:
            results, _ = self.client.scroll(
                collection_name=COLLECTION_CHUNKS,
                scroll_filter=Filter(should=conditions),
                limit=top_k,
            )
            return [{
                "id": str(r.id), "texto": r.payload.get("texto", ""), "fuente": r.payload.get("fuente", ""),
                "tipo": "texto", "imagen_path": None, "similitud": 0.49,
                "nombre_archivo": "", "etiqueta": "",
                "imagenes_pagina": r.payload.get("imagenes_pagina", []),
                "pagina": r.payload.get("pagina"),
            } for r in results]
        except Exception as e:
            print(f"⚠️ Error búsqueda entidades: {e}")
            return []

    async def expandir_vecindad(self, node_ids: list, depth: int = 1) -> list:
        # Simplificación de vecindad para Qdrant: si no hay ids, retornar vacio.
        # Podríamos buscar chunks de la misma página.
        if not node_ids: return []
        # Fallback simple para compatibilidad: (la implementación detallada en Qdrant es muy compleja)
        # Devolveremos chunks de los mismos documentos (fuente)
        return []

    async def busqueda_camino_semantico(self, tejido_origen: str, tejido_destino: str) -> list:
        # En Qdrant no hay caminos, hacemos un query textual
        return []

    async def busqueda_imagenes_por_texto(self, entidades: dict, top_k: int = 5) -> list:
        """Busca imágenes cuyo caption/texto_pagina contenga los términos de la consulta.
        Equivalente a la búsqueda CONTAINS de Neo4j."""
        import unicodedata
        tejidos = entidades.get("tejidos", [])
        estructuras = entidades.get("estructuras", [])
        consulta = entidades.get("_consulta", [])
        terminos = tejidos + estructuras + consulta
        if not terminos: return []

        def sin_tildes(s):
            return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

        terminos_lower = list(set([t.lower() for t in terminos] + [sin_tildes(t.lower()) for t in terminos]))

        # Scroll todas las imágenes y filtrar por texto en Python
        # (Qdrant no tiene full-text search nativo, pero la colección de imágenes es pequeña)
        try:
            all_imgs, _ = self.client.scroll(
                collection_name=COLLECTION_IMAGENES,
                limit=200,
                with_payload=True,
                with_vectors=False,
            )
            resultados = []
            for r in all_imgs:
                caption = (r.payload.get("caption", "") or "").lower()
                texto_pag = (r.payload.get("texto_pagina", "") or "").lower()
                ocr = (r.payload.get("ocr_text", "") or "").lower()
                combined_text = f"{caption} {texto_pag} {ocr}"
                combined_text_sin_tildes = sin_tildes(combined_text)

                if any(t in combined_text or t in combined_text_sin_tildes for t in terminos_lower):
                    texto_desc = caption if caption.strip() else (texto_pag if texto_pag.strip() else ocr)
                    img_path = r.payload.get("path", "")
                    nombre_archivo = r.payload.get("nombre_archivo", "").lower()
                    if img_path and os.path.exists(img_path) and "_full." not in nombre_archivo:
                        resultados.append({
                            "id": str(r.id), "texto": texto_desc, "fuente": r.payload.get("fuente", ""),
                            "tipo": "imagen", "imagen_path": img_path, "similitud": 0.50,
                            "nombre_archivo": r.payload.get("nombre_archivo", ""),
                            "etiqueta": r.payload.get("etiqueta", ""),
                            "caption_raw": r.payload.get("caption", "")
                        })
            if resultados:
                print(f"   🖼️ {len(resultados)} imágenes encontradas (búsqueda texto directo)")
            return resultados[:top_k]
        except Exception as e:
            print(f"⚠️ Error búsqueda imágenes por texto: {e}")
            return []

    async def busqueda_chunks_por_texto(self, terminos: list, top_k: int = 10) -> list:
        """Búsqueda de chunks por keywords en el texto del payload.
        Fallback para cuando la búsqueda vectorial no da buenos resultados."""
        import unicodedata
        if not terminos: return []

        def sin_tildes(s):
            return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

        terminos_lower = list(set([t.lower() for t in terminos] + [sin_tildes(t.lower()) for t in terminos]))

        try:
            all_chunks, _ = self.client.scroll(
                collection_name=COLLECTION_CHUNKS,
                limit=500,
                with_payload=True,
                with_vectors=False,
            )
            resultados = []
            for r in all_chunks:
                texto = (r.payload.get("texto", "") or "").lower()
                texto_sin_tildes = sin_tildes(texto)
                if any(t in texto or t in texto_sin_tildes for t in terminos_lower):
                    resultados.append({
                        "id": str(r.id), "texto": r.payload.get("texto", ""), "fuente": r.payload.get("fuente", ""),
                        "tipo": "texto", "imagen_path": None, "similitud": 0.50,
                        "nombre_archivo": "", "etiqueta": "",
                        "imagenes_pagina": r.payload.get("imagenes_pagina", []),
                        "pagina": r.payload.get("pagina"),
                    })
            if resultados:
                print(f"   📝 {len(resultados)} chunks encontrados (búsqueda texto directo)")
            return resultados[:top_k]
        except Exception as e:
            print(f"⚠️ Error búsqueda chunks por texto: {e}")
            return []

    async def buscar_imagenes_por_referencia(self, patrones: list, top_k: int = 10) -> list:
        """Busca imágenes cuya etiqueta, caption o nombre_archivo contenga alguno de los patrones.
        Equivalente a la query Cypher del repo de referencia (histo-test) que busca
        nodos :Imagen por etiqueta/caption/nombre_archivo."""
        import unicodedata
        if not patrones:
            return []

        def sin_tildes(s):
            return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

        patrones_lower = [p.lower() for p in patrones]
        patrones_sin_tildes = [sin_tildes(p) for p in patrones_lower]

        try:
            all_imgs, _ = self.client.scroll(
                collection_name=COLLECTION_IMAGENES,
                limit=200,
                with_payload=True,
                with_vectors=False,
            )
            resultados = []
            vistas = set()
            for r in all_imgs:
                etiqueta = (r.payload.get("etiqueta", "") or "").lower()
                caption = (r.payload.get("caption", "") or "").lower()
                nombre = (r.payload.get("nombre_archivo", "") or "").lower()
                combined = f"{etiqueta} {caption} {nombre}"
                combined_sin_tildes = sin_tildes(combined)

                if "_full." in nombre:
                    continue

                matched = any(
                    p in combined or p_st in combined_sin_tildes
                    for p, p_st in zip(patrones_lower, patrones_sin_tildes)
                )
                if not matched:
                    continue

                img_path = r.payload.get("path", "")
                if not img_path or not os.path.exists(img_path):
                    continue
                if img_path in vistas:
                    continue
                vistas.add(img_path)

                texto_desc = caption if caption.strip() else r.payload.get("texto_pagina", "") or ""
                resultados.append({
                    "id": str(r.id),
                    "texto": texto_desc,
                    "fuente": r.payload.get("fuente", ""),
                    "tipo": "imagen",
                    "imagen_path": img_path,
                    "similitud": 0.95,
                    "nombre_archivo": r.payload.get("nombre_archivo", ""),
                    "etiqueta": r.payload.get("etiqueta", ""),
                    "caption_raw": r.payload.get("caption", ""),
                    "origen": "referencia_respuesta",
                })

            if resultados:
                print(f"   🖼️ {len(resultados)} imágenes encontradas por referencia en respuesta")
            return resultados[:top_k]
        except Exception as e:
            print(f"⚠️ Error búsqueda imágenes por referencia: {e}")
            return []

    def extraer_imagenes_de_resultados(self, resultados: list, top_k: int = 5) -> list:
        """Extrae y valida imágenes desde resultados de búsqueda.
        Filtra resultados de tipo 'imagen', valida path en disco,
        transforma formato y limita a top_k."""
        imagenes = [r for r in resultados if r.get("tipo") == "imagen"]
        if not imagenes:
            return []

        imagenes_validas = []
        vistas = set()
        for img in imagenes:
            img_path = img.get("imagen_path", "")
            if not img_path or not os.path.exists(img_path):
                continue
            nombre = os.path.basename(img_path).lower()
            if "_full." in nombre:
                continue
            if img_path in vistas:
                continue
            vistas.add(img_path)

            caption = img.get("caption_raw") or img.get("texto", "") or ""
            imagenes_validas.append({
                "id": img.get("id", ""),
                "path": img_path,
                "caption": caption[:500],
                "nombre_archivo": img.get("nombre_archivo", os.path.basename(img_path)),
                "etiqueta": img.get("etiqueta", ""),
                "fuente": img.get("fuente", ""),
                "similitud_semantica": img.get("similitud", 0),
            })

        return imagenes_validas[:top_k]

    async def busqueda_imagenes_semantica(self, texto_embedding: list, entidades: dict,
                                          embeddings_model, top_k: int = 5) -> list:
        import numpy as np
        import os
        print("   🔎 Búsqueda semántica de imágenes en Qdrant...")
        if not texto_embedding: return []
        
        candidatas = {}
        # Usar vector texto_emb en COLLECTION_IMAGENES
        try:
            res_img_texto = self.client.query_points(
                collection_name=COLLECTION_IMAGENES, query=texto_embedding, using="texto_emb", limit=top_k * 3
            ).points
            for r in res_img_texto:
                caption = r.payload.get("caption", "")
                texto_pag = r.payload.get("texto_pagina", "")
                ocr = r.payload.get("ocr_text", "")
                texto = caption if caption else (texto_pag if texto_pag else ocr)
                candidatas[str(r.id)] = {
                    "id": str(r.id), "texto": texto, "fuente": r.payload.get("fuente", ""),
                    "imagen_path": r.payload.get("path"), "similitud": r.score,
                    "nombre_archivo": r.payload.get("nombre_archivo", ""),
                    "etiqueta": r.payload.get("etiqueta", ""),
                    "caption_raw": caption
                }
        except Exception as e:
            print(f"⚠️ Error en texto_emb de Qdrant: {e}")

        if not candidatas:
            print("   ℹ️ Sin imágenes candidatas")
            return []
            
        print(f"   📋 {len(candidatas)} imágenes candidatas encontradas, re-ranking semántico...")
        resultados_rankeados = []
        q_emb = np.array(texto_embedding)
        
        for img_id, img_dict in candidatas.items():
            caption_full = img_dict.get("texto", "") or img_dict.get("caption_raw", "")
            if not caption_full or len(caption_full.strip()) < 10: continue
            img_path = img_dict.get("imagen_path", "")
            nombre_archivo = img_dict.get("nombre_archivo", "").lower()
            if not img_path or not os.path.exists(img_path) or "_full." in nombre_archivo: continue
            
            try:
                # Similaridad con caption completo
                caption_emb = np.array(embed_query_con_reintento(embeddings_model, caption_full[:500]))
                sim_caption = float(q_emb @ caption_emb / (np.linalg.norm(q_emb) * np.linalg.norm(caption_emb) + 1e-10))
                
                # Similaridad con título (primera línea del caption = señal más fuerte)
                titulo = caption_full.split('\n')[0].strip() if caption_full else ""
                sim_titulo = 0.0
                if titulo and len(titulo) >= 5:
                    titulo_emb = np.array(embed_query_con_reintento(embeddings_model, titulo))
                    sim_titulo = float(q_emb @ titulo_emb / (np.linalg.norm(q_emb) * np.linalg.norm(titulo_emb) + 1e-10))
                
                # Score final: priorizar la señal más fuerte (título o caption)
                sim = max(sim_titulo, sim_caption) * 0.7 + min(sim_titulo, sim_caption) * 0.3
                
                resultados_rankeados.append({
                    "id": img_id, "path": img_path, "caption": caption_full[:500],
                    "nombre_archivo": img_dict.get("nombre_archivo", os.path.basename(img_path)),
                    "etiqueta": img_dict.get("etiqueta", ""), "fuente": img_dict.get("fuente", ""),
                    "similitud_semantica": sim
                })
            except Exception as e:
                continue
                
        resultados_rankeados.sort(key=lambda x: x["similitud_semantica"], reverse=True)
        if resultados_rankeados:
            top3 = [(r["nombre_archivo"], round(r["similitud_semantica"], 3)) for r in resultados_rankeados[:3]]
            print(f"   📊 Re-ranking top-3 sim: {top3}")

        # Umbral de similitud semántica: 0.55 para garantizar relevancia.
        UMBRAL_SIMILITUD_IMG = 0.55
        filtrados = [r for r in resultados_rankeados if r["similitud_semantica"] >= UMBRAL_SIMILITUD_IMG]

        # Filtro de coherencia de fuente: si el top resultado tiene una fuente clara,
        # solo incluir imágenes de otras fuentes si su similitud es prácticamente
        # igual al top (margen 0.01). Esto evita mezclar imágenes de temas distintos
        # (ej: arterias de arch3 con espermatogénesis de arch4) cuando el modelo
        # de embeddings 384d no discrimina bien entre temas histológicos.
        if filtrados and len(filtrados) > 1:
            top_sim = filtrados[0]["similitud_semantica"]
            top_fuente = filtrados[0].get("fuente", "")
            if top_fuente:
                coherentes = []
                for r in filtrados:
                    misma_fuente = r.get("fuente", "") == top_fuente
                    sim_casi_igual = r["similitud_semantica"] >= top_sim - 0.01
                    if misma_fuente or sim_casi_igual:
                        coherentes.append(r)
                if coherentes:
                    descartadas = len(filtrados) - len(coherentes)
                    if descartadas > 0:
                        print(f"   🔍 Filtro de fuente: descartadas {descartadas} imágenes de otras fuentes")
                    filtrados = coherentes

        if filtrados:
            print(f"   ✅ {len(filtrados)} imágenes con similitud semántica >= {UMBRAL_SIMILITUD_IMG}")
        else:
            print(f"   ⚠️ Ninguna imagen superó el umbral de similitud semántica ({UMBRAL_SIMILITUD_IMG})")
        return filtrados[:top_k]

    async def busqueda_hibrida(self, texto_embedding, imagen_embedding_uni, imagen_embedding_plip, entidades, top_k=10):
        res_texto = []
        res_uni = []
        res_plip = []
        res_ent = []
        res_pag_chunks = []
        res_vec = []
        res_img_texto = []
        res_keyword = []  # búsqueda de texto por keywords (fallback)
        
        if texto_embedding:
            res_texto = await self.busqueda_vectorial(texto_embedding, INDEX_TEXTO, top_k)
        if imagen_embedding_uni:
            res_uni = await self.busqueda_vectorial(imagen_embedding_uni, INDEX_UNI, top_k)
            res_uni = [r for r in res_uni if r.get("similitud", 0) >= 0.80]
        if imagen_embedding_plip:
            res_plip = await self.busqueda_vectorial(imagen_embedding_plip, INDEX_PLIP, top_k)
            res_plip = [r for r in res_plip if r.get("similitud", 0) >= 0.80]
            
        res_ent = await self.busqueda_por_entidades(entidades, top_k)
        
        tiene_imagen = imagen_embedding_uni is not None or imagen_embedding_plip is not None
        if tiene_imagen:
            top_img_results = [r for r in (res_uni + res_plip) if r.get("similitud", 0) > 0.75]
            for img_r in top_img_results[:3]:
                fuente = img_r.get("fuente", "")
                pagina = img_r.get("pagina")
                if fuente and pagina:
                    chunks_pag = await self.busqueda_chunks_por_pagina(fuente, pagina)
                    res_pag_chunks.extend(chunks_pag)

        if texto_embedding:
            try:
                raw_img = self.client.query_points(collection_name=COLLECTION_IMAGENES, query=texto_embedding, using="texto_emb", limit=top_k).points
                for r in raw_img:
                    nombre_archivo = r.payload.get("nombre_archivo", "").lower()
                    if "_full." in nombre_archivo:
                        continue
                    caption = r.payload.get("caption", "")
                    texto_pag = r.payload.get("texto_pagina", "")
                    ocr = r.payload.get("ocr_text", "")
                    texto = caption if caption else (texto_pag if texto_pag else ocr)
                    if r.score > 0.1:
                        res_img_texto.append({
                            "id": str(r.id), "texto": texto, "fuente": r.payload.get("fuente", ""),
                            "tipo": "imagen", "imagen_path": r.payload.get("path"), "similitud": r.score,
                            "nombre_archivo": r.payload.get("nombre_archivo", ""),
                            "etiqueta": r.payload.get("etiqueta", ""),
                            "pagina": r.payload.get("pagina")
                        })
            except Exception: pass

        # Fallback: búsqueda por keywords en el texto de los chunks
        # Si la búsqueda vectorial no da buenos resultados, buscamos por texto
        consulta_palabras = entidades.get("_consulta", [])
        tejidos = entidades.get("tejidos", [])
        estructuras = entidades.get("estructuras", [])
        terminos_keyword = tejidos + estructuras + consulta_palabras
        if not tiene_imagen and terminos_keyword:
            # Solo hacer fallback en modo texto cuando los resultados vectoriales son débiles
            top_vector_sim = max((r.get("similitud", 0) for r in res_texto), default=0)
            if top_vector_sim < 0.50:
                res_keyword = await self.busqueda_chunks_por_texto(terminos_keyword, top_k)

        combined = {}
        def agregar(resultados, peso, es_visual=False):
            for r in resultados:
                key = r.get("id") or f"{r.get('fuente')}_{str(r.get('texto',''))[:40]}"
                if not r.get("texto") and not r.get("imagen_path"): continue
                sim = r.get("similitud", 0)
                sim_ponderada = sim * peso
                if es_visual and sim > 0.95:
                    sim_ponderada += 2.0
                if key not in combined:
                    combined[key] = {**r, "similitud": sim_ponderada}
                else:
                    combined[key]["similitud"] += sim_ponderada

        if tiene_imagen:
            # Prioridad 1: lo que el usuario está escribiendo explícitamente en el chat
            agregar(res_texto, 0.80)
            
            # Prioridad 2: los matches visuales (en caso de que la consulta sea sobre la imagen)
            agregar(res_uni, 0.60, es_visual=True)
            agregar(res_plip, 0.60, es_visual=True)
            agregar(res_ent, 0.50)
            
            # Fallback contextual: los textos de la página de la imagen (solo útiles si no hay un buen match en res_texto)
            agregar(res_pag_chunks, 0.15)
        else:
            agregar(res_texto, 0.80)
            agregar(res_uni, 0.20, es_visual=True)
            agregar(res_plip, 0.20, es_visual=True)
            agregar(res_img_texto, 0.40)
            agregar(res_ent, 0.60)
            agregar(res_keyword, 0.70)  # keyword matches son confiables

        final = sorted(combined.values(), key=lambda x: x["similitud"], reverse=True)

        print(f"   📊 Híbrida: Txt={len(res_texto)} | "
              f"UNI={len(res_uni)} | PLIP={len(res_plip)} | Ent={len(res_ent)} | "
              f"ImgTxt={len(res_img_texto)} | Keyword={len(res_keyword)} -> {len(final)}")

        return final[:15]

# =============================================================================
# MEMORIA SEMÁNTICA CON PERSISTENCIA DE IMAGEN
# =============================================================================

class SemanticMemory:
    """
    Mantiene historial de conversación y la última imagen activa.
    La imagen persiste entre turnos hasta que el usuario la cambie
    explícitamente con el comando 'nueva imagen'.
    Añade persistencia en Qdrant con vectores texto, UNI y PLIP.
    """

    def __init__(self, llm, embeddings=None, uni=None, plip=None, max_entries: int = 10):
        self.llm            = llm
        self.embeddings     = embeddings
        self.uni            = uni
        self.plip           = plip
        self.conversations  = []
        self.max_entries    = max_entries
        self.summary        = ""
        self.direct_history = ""

        # Persistencia de imagen entre turnos
        self.imagen_activa_path: Optional[str] = None
        self.imagen_turno_subida: int = 0
        self.analisis_visual_activo: Optional[str] = None
        self.turno_actual: int = 0
        
        # Qdrant init
        self.collection_name = "memoria_histo"
        self.qdrant = QdrantClient(path="./qdrant_memoria")
        
        # --- Validación de Colección Qdrant (Dimensionalidad) ---
        try:
            info = self.qdrant.get_collection(self.collection_name)
            config = info.config.params.vectors
            
            # Verificar dimensión de 'texto'
            dims_actual = 0
            if isinstance(config, dict) and "texto" in config:
                dims_actual = config["texto"].size
            
            if dims_actual != DIM_TEXTO:
                print(f"   ⚠️ Dimensión de Qdrant incorrecta ({dims_actual} != {DIM_TEXTO}). Recreando...")
                self.qdrant.delete_collection(self.collection_name)
                raise Exception("Mismatch forcing recreation")
        except Exception:
            print(f"   🗂️ Configurando colección Qdrant '{self.collection_name}'...")
            self.qdrant.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "texto": VectorParams(size=DIM_TEXTO, distance=Distance.COSINE),
                    "uni": VectorParams(size=DIM_IMG_UNI, distance=Distance.COSINE),
                    "plip": VectorParams(size=DIM_IMG_PLIP, distance=Distance.COSINE),
                }
            )

    def set_imagen(self, path: Optional[str], analisis_visual: Optional[str] = None):
        """Registra una nueva imagen activa. None = limpiar."""
        if path and os.path.exists(path):
            self.imagen_activa_path  = path
            self.imagen_turno_subida = self.turno_actual
            self.analisis_visual_activo = analisis_visual
            print(f"   📌 Imagen activa registrada (turno {self.turno_actual}): {path}")
        elif path is None:
            self.imagen_activa_path = None
            self.analisis_visual_activo = None
            print("   🗑️  Imagen activa limpiada")

    def get_imagen_activa(self) -> Optional[str]:
        if self.imagen_activa_path and os.path.exists(self.imagen_activa_path):
            return self.imagen_activa_path
        return None

    def tiene_imagen_previa(self) -> bool:
        return self.get_imagen_activa() is not None

    def add_interaction(self, query: str, response: str):
        """Guarda siempre la interacción, independientemente del resultado RAG."""
        self.turno_actual += 1
        self.conversations.append({
            "query":    query,
            "response": response,
            "turno":    self.turno_actual,
            "imagen":   self.imagen_activa_path
        })
        if len(self.conversations) > self.max_entries:
            self.conversations.pop(0)

        self.direct_history += f"\nUsuario: {query}\nAsistente: {response}\n"
        if len(self.conversations) > 3:
            recent = self.conversations[-3:]
            self.direct_history = ""
            for conv in recent:
                img_nota = (f" [con imagen: {os.path.basename(conv['imagen'])}]"
                            if conv.get("imagen") else "")
                self.direct_history += (
                    f"\nUsuario{img_nota}: {conv['query']}\n"
                    f"Asistente: {conv['response']}\n"
                )
        self._update_summary()
        
        # Guardar en Qdrant cada 5 interacciones
        if self.turno_actual % 5 == 0 and len(self.conversations) > 0 and self.embeddings:
            self._guardar_memoria_qdrant()

    def _guardar_memoria_qdrant(self):
        print("   🧠 Generando resumen profundo para guardar en memoria (Qdrant)...")
        try:
            # Resumir contexto reciente
            resp = invoke_con_reintento_sync(self.llm, [
                SystemMessage(content="Genera un resumen detallado y técnico del siguiente historial de conversación sobre histología, destacando las entidades mencionadas y las conclusiones."),
                HumanMessage(content=self.direct_history)
            ])
            resumen = resp.content
            
            # Embeddings del resumen textual
            emb_texto = embed_query_con_reintento(self.embeddings, resumen)
            
            # Embeddings visuales de la imagen activa en este bloque (si la hay)
            emb_uni = [0.0] * DIM_IMG_UNI
            emb_plip = [0.0] * DIM_IMG_PLIP
            if self.imagen_activa_path and os.path.exists(self.imagen_activa_path):
                if self.uni:
                    # Memory images: apply preprocessing for consistency
                    emb_uni = self.uni.embed_image(self.imagen_activa_path, preprocess=True).tolist()
                if self.plip:
                    emb_plip = self.plip.embed_image(self.imagen_activa_path, preprocess=True).tolist()

            point = PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    "texto": emb_texto,
                    "uni": emb_uni,
                    "plip": emb_plip
                },
                payload={
                    "resumen": resumen,
                    "turno_fin": self.turno_actual,
                    "tiene_imagen": self.imagen_activa_path is not None,
                    "imagen_path": self.imagen_activa_path
                }
            )
            self.qdrant.upsert(
                collection_name=self.collection_name,
                points=[point]
            )
            print("   ✅ Resumen guardado en Qdrant (memoria_histo)")
        except Exception as e:
            print(f"   ⚠️ Error guardando memoria en Qdrant: {e}")

    def _update_summary(self):
        try:
            if len(self.conversations) > 6:
                resp = invoke_con_reintento_sync(self.llm, [
                    SystemMessage(content="Resume estas consultas de histología manteniendo términos técnicos:"),
                    HumanMessage(content=self.direct_history)
                ])
                self.summary = f"Resumen: {resp.content}\n\nRecientes:{self.direct_history}"
            else:
                self.summary = f"Recientes:{self.direct_history}"
        except Exception as e:
            self.summary = f"Recientes:{self.direct_history}"

    def get_context(self, query: str = "") -> str:
        ctx = self.summary.strip() or "No hay consultas previas."
        
        # Recuperar de Qdrant
        if query and self.embeddings:
            try:
                emb_query = embed_query_con_reintento(self.embeddings, query)
                resultados = self.qdrant.query_points(
                    collection_name=self.collection_name,
                    query=emb_query,
                    using="texto",
                    limit=2
                ).points
                
                memorias_recuperadas = [r.payload['resumen'] for r in resultados if r.score > 0.4]
                if memorias_recuperadas:
                    ctx += "\n\n[Memorias históricas recuperadas:]\n" + "\n- ".join(memorias_recuperadas)
            except Exception as e:
                print(f"   ⚠️ Error recuperando memoria Qdrant: {e}")

        if self.imagen_activa_path:
            ctx += (f"\n\n[Imagen activa en el chat: "
                    f"{os.path.basename(self.imagen_activa_path)}]")
        return ctx

    def get_history_for_prompt(self, max_turns: int = 5) -> str:
        """Retorna los últimos max_turns intercambios formateados para inyectar en el system prompt."""
        recent = self.conversations[-max_turns:]
        if not recent:
            return ""
        lines = []
        for conv in recent:
            lines.append(f"Usuario: {conv['query']}")
            resp = conv['response']
            resp_truncada = resp[:200] + "..." if len(resp) > 200 else resp
            lines.append(f"Asistente: {resp_truncada}")
        return "\n".join(lines)


# =============================================================================
# CLASIFICADOR SEMÁNTICO — reemplaza verificación por keywords
# =============================================================================

class ClasificadorSemantico:
    """
    Determina si una consulta pertenece al dominio histológico combinando:
      1. Similitud coseno contra la ontología extraída del contenido (temario).
         Fallback a anclas semánticas hardcodeadas si el temario está vacío.
      2. Razonamiento LLM con contexto de imagen disponible.
    """

    UMBRAL_SIMILITUD = 0.45
    UMBRAL_LLM       = 0.49

    def __init__(self, llm, embeddings, device: str, temario: List[str]):
        self.llm        = llm
        self.embeddings = embeddings
        self.device     = device
        self._temario: List[str]              = temario
        self._anclas_emb: Optional[np.ndarray] = None
        self._temario_emb: Optional[np.ndarray] = None

    @property
    def temario(self) -> List[str]:
        return self._temario

    @temario.setter
    def temario(self, value: List[str]):
        self._temario = value
        self._temario_emb = None  # Invalidar cache al actualizar ontología
        print(f"   🔄 Ontología actualizada ({len(value)} temas) — cache de embeddings invalidado")

    def _embed_textos(self, textos: List[str]) -> np.ndarray:
        return np.array(embed_documents_con_reintento(self.embeddings, textos))

    def _get_anclas_emb(self) -> np.ndarray:
        """Embeddings de anclas hardcodeadas (fallback cuando no hay temario)."""
        if self._anclas_emb is None:
            print("   🔄 Precalculando embeddings de anclas semánticas (fallback)...")
            self._anclas_emb = self._embed_textos(ANCLAS_SEMANTICAS_HISTOLOGIA)
        return self._anclas_emb

    def _get_temario_emb(self) -> Optional[np.ndarray]:
        """Embeddings de la ontología real extraída del contenido."""
        if not self._temario:
            return None
        if self._temario_emb is None:
            print(f"   🔄 Precalculando embeddings de ontología ({len(self._temario)} temas)...")
            self._temario_emb = self._embed_textos(self._temario)
        return self._temario_emb

    def similitud_con_dominio(self, consulta: str) -> float:
        try:
            q_emb = np.array(embed_query_con_reintento(self.embeddings, consulta))

            # Priorizar ontología extraída del contenido real
            temario_emb = self._get_temario_emb()
            if temario_emb is not None and len(temario_emb) > 0:
                sims = (q_emb @ temario_emb.T).flatten()
                return float(np.max(sims))

            # Fallback a anclas hardcodeadas (solo antes de indexar)
            a_emb = self._get_anclas_emb()
            sims  = (q_emb @ a_emb.T).flatten()
            return float(np.max(sims))
        except Exception as e:
            print(f"   ⚠️ Error similitud semántica: {e}")
            return 0.0

    async def clasificar(
        self,
        consulta: str,
        analisis_visual: Optional[str] = None,
        imagen_activa: bool = False,
        temario_muestra: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        # Paso 1: Similitud ImageBind
        sim = self.similitud_con_dominio(consulta)
        print(f"   📐 Similitud semántica con dominio histológico: {sim:.4f}")

        umbral_efectivo = self.UMBRAL_SIMILITUD * (0.6 if imagen_activa else 1.0)

        if sim >= umbral_efectivo:
            return {
                "valido":            True,
                "tema_encontrado":   None,
                "motivo":            f"Similitud ImageBind {sim:.3f} ≥ umbral {umbral_efectivo:.3f}",
                "similitud_dominio": sim,
                "metodo":            "semantico_imagebind"
            }

        # Paso 2: LLM como árbitro
        muestra_temas = (temario_muestra or self.temario)[:60]
        temario_txt   = "\n".join(f"- {t}" for t in muestra_temas)

        context_extra = ""
        if analisis_visual:
            context_extra = f"\n\nANÁLISIS DE IMAGEN DISPONIBLE:\n{analisis_visual[:600]}"
        if imagen_activa:
            context_extra += "\n\n[El usuario tiene una imagen histológica activa en el chat]"

        system = f"""Eres un clasificador de intención para un sistema RAG de histología médica.

Tu tarea: determinar si la consulta es una pregunta relacionada con histología,
patología, anatomía microscópica o morfología celular/tisular.

IMPORTANTE:
- "¿de qué tipo de tejido se trata?" SÍ es histológica.
- "¿qué ves en la imagen?" en contexto histológico SÍ es histológica.
- No es necesario que mencione palabras técnicas si el contexto lo indica.
- Si hay imagen histológica activa, dar beneficio de la duda.

TEMARIO DISPONIBLE (muestra):
{temario_txt}
{context_extra}

Responde ÚNICAMENTE en JSON válido (sin backticks):
{{"valido": true/false, "tema_encontrado": "tema más cercano o null", "confianza": 0.0-1.0, "motivo": "explicación breve"}}"""

        try:
            resp      = await invoke_con_reintento(self.llm, [
                SystemMessage(content=system),
                HumanMessage(content=f"CONSULTA: {consulta}")
            ])
            texto     = re.sub(r"```json\s*|\s*```", "", resp.content.strip())
            data      = json.loads(texto)
            confianza = float(data.get("confianza", 0.49))
            valido    = bool(data.get("valido", True))

            if not valido and imagen_activa:
                valido = True
                data["motivo"] += " [aceptado por imagen activa]"

            return {
                "valido":            valido,
                "tema_encontrado":   data.get("tema_encontrado"),
                "motivo":            data.get("motivo", ""),
                "similitud_dominio": sim,
                "metodo":            "llm" if sim < umbral_efectivo * 0.49 else "combinado"
            }
        except Exception as e:
            print(f"   ⚠️ Error clasificador LLM: {e}")
            return {
                "valido":            imagen_activa or sim > 0.10,
                "tema_encontrado":   None,
                "motivo":            f"Fallback: {e}",
                "similitud_dominio": sim,
                "metodo":            "fallback"
            }


# =============================================================================
# EXTRACTOR DE IMÁGENES DE PDF
# =============================================================================

class ImageExtractionConfig:
    """Configuration for image extraction and preprocessing."""
    
    # Size thresholds
    MIN_WIDTH: int = 150
    MIN_HEIGHT: int = 150
    TARGET_MAGNIFICATION_SIZE: int = 868
    
    # Enhancement settings
    ENHANCE_CONTRAST: bool = True
    ENHANCE_BRIGHTNESS: bool = True
    CONTRAST_FACTOR: float = 1.2
    BRIGHTNESS_FACTOR: float = 1.1
    
    # Fallback settings
    FALLBACK_DPI: int = 150
    
    # Resampling
    RESAMPLING_ALGORITHM = Image.Resampling.LANCZOS


class ExtractorImagenesPDF:
    def __init__(self, directorio_salida: str = DIRECTORIO_IMAGENES, config: Optional[ImageExtractionConfig] = None):
        self.directorio_salida = directorio_salida
        self.config = config if config is not None else ImageExtractionConfig()
        os.makedirs(directorio_salida, exist_ok=True)

    def _apply_preprocessing(self, image: Image.Image) -> Image.Image:
        """
        Apply preprocessing enhancements to image.
        
        Applies contrast enhancement followed by brightness enhancement
        based on configuration settings.
        
        Args:
            image: PIL Image object
            
        Returns:
            Enhanced PIL Image object, or original if preprocessing fails
        """
        try:
            from PIL import ImageEnhance
            
            enhanced = image
            
            # Apply contrast enhancement first
            if self.config.ENHANCE_CONTRAST:
                enhancer = ImageEnhance.Contrast(enhanced)
                enhanced = enhancer.enhance(self.config.CONTRAST_FACTOR)
                
            # Apply brightness enhancement second
            if self.config.ENHANCE_BRIGHTNESS:
                enhancer = ImageEnhance.Brightness(enhanced)
                enhanced = enhancer.enhance(self.config.BRIGHTNESS_FACTOR)
                
            return enhanced
            
        except Exception as e:
            print(f"  ⚠️ Preprocessing failed: {e}, using original image")
            return image

    def _apply_magnification(self, image: Image.Image) -> Image.Image:
        """
        Magnify image to target size if below threshold.
        
        Magnification rules:
        - If width < 868: scale to width=868, preserve aspect ratio
        - Else if height < 868: scale to height=868, preserve aspect ratio
        - Else: no magnification needed
        
        Args:
            image: PIL Image object
            
        Returns:
            Magnified PIL Image object (or original if no magnification needed)
        """
        w, h = image.size
        target = self.config.TARGET_MAGNIFICATION_SIZE
        
        # Check if magnification is needed
        if w >= target and h >= target:
            return image
            
        # Calculate new dimensions
        if w < target:
            # Scale based on width
            new_w = target
            new_h = int(h * (target / w))
            print(f"  🔍 Magnifying: {w}x{h} → {new_w}x{new_h} (width-based)")
        else:
            # Scale based on height (w >= target but h < target)
            new_h = target
            new_w = int(w * (target / h))
            print(f"  🔍 Magnifying: {w}x{h} → {new_w}x{new_h} (height-based)")
            
        # Apply magnification with LANCZOS resampling
        return image.resize((new_w, new_h), self.config.RESAMPLING_ALGORITHM)

    def _fallback_render_page(self, pdf_path: str, page_num: int) -> Optional[Image.Image]:
        """
        Render PDF page as image using pdf2image fallback.
        
        Args:
            pdf_path: Path to PDF file
            page_num: Page number (1-indexed)
            
        Returns:
            PIL Image object or None if rendering fails
        """
        try:
            from pdf2image import convert_from_path
            
            print(f"  🔄 Fallback: rendering page {page_num} with pdf2image")
            
            pag_imgs = convert_from_path(
                pdf_path,
                first_page=page_num,
                last_page=page_num,
                dpi=self.config.FALLBACK_DPI
            )
            
            if pag_imgs:
                return pag_imgs[0]
            return None
            
        except Exception as e:
            print(f"  ❌ Fallback rendering failed for page {page_num}: {e}")
            return None

    @staticmethod
    def _extraer_etiqueta_imagen(texto: str) -> str:
        """Busca la etiqueta formal de la imagen (Imagen X.X, Fig X.X, Figura X.X)
        en el texto y la devuelve. Devuelve cadena vacía si no encuentra."""
        if not texto:
            return ""
        patrones = [
            r'(Imagen\s+\d+[\.]?\d*[A-Za-z]?)',
            r'(Fig(?:ura)?\.?\s*\d+[\.]?\d*[A-Za-z]?)',
            r'(Lámina\s+\d+[\.]?\d*[A-Za-z]?)',
            r'(Foto(?:grafía)?\s+\d+[\.]?\d*[A-Za-z]?)',
        ]
        for patron in patrones:
            match = re.search(patron, texto, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return ""

    @staticmethod
    def extraer_caption_imagen(page_fitz, img_bbox, texto_pagina_completo: str) -> str:
        """Extrae la etiqueta 'Imagen X.X' / 'Fig X.X' + TODO el texto debajo
        de la imagen hasta el final de la página.
        
        Estrategia:
        1. Buscar texto desde ligeramente ARRIBA del borde inferior de la imagen
           (para capturar etiquetas que se solapan con el bbox) hasta el final.
        2. Si no se encontró etiqueta, buscar desde justo debajo.
        3. Siempre devolver el caption completo (etiqueta + descripción).
        """
        caption = ""
        try:
            page_rect = page_fitz.rect
            
            # Paso 1: Buscar con margen hacia arriba (10px) para capturar
            # etiquetas que solapan con el borde inferior de la imagen
            margen_overlap = 10  # pixels de margen
            area_expandida = fitz.Rect(
                0,
                max(0, img_bbox[3] - margen_overlap),
                page_rect.width,
                page_rect.height
            )
            texto_expandido = page_fitz.get_text("text", clip=area_expandida).strip()
            
            if texto_expandido:
                caption = texto_expandido
            else:
                # Paso 2: Buscar desde justo debajo sin margen
                area_abajo = fitz.Rect(
                    0,
                    img_bbox[3],
                    page_rect.width,
                    page_rect.height
                )
                caption = page_fitz.get_text("text", clip=area_abajo).strip()
        except Exception:
            pass
        
        if caption:
            # Limpiar: remover números de página sueltos al final
            caption = re.sub(r'\n\s*\d{1,3}\s*$', '', caption).strip()
            return caption
        # Fallback: primeros 500 chars del texto de la página
        return texto_pagina_completo[:500] if texto_pagina_completo else ""

    def extraer_de_pdf(self, pdf_path: str) -> List[Dict[str, str]]:
        imagenes_extraidas = []
        nombre_pdf = os.path.splitext(os.path.basename(pdf_path))[0]
        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            print(f"❌ Error abriendo {pdf_path}: {e}")
            return []

        def _texto_pagina_con_contexto(doc, idx_0based: int) -> str:
            """Devuelve texto SOLO de la página actual.
            El caption se extrae por proximidad espacial (bbox),
            no por concatenación de páginas adyacentes."""
            try:
                return doc[idx_0based].get_text().strip()
            except Exception:
                return ""

        for num_pagina, pagina in enumerate(doc, start=1):
            idx_0 = num_pagina - 1  # índice base-0 para acceder a doc[idx]
            valid_images_this_page = []
            
            # Método preciso: solo imágenes realmente dibujadas en esta página
            try:
                img_info_list = pagina.get_image_info(xrefs=True)
                # Mapear xref -> bbox para extracción de caption (Fix #1)
                xref_to_bbox = {info["xref"]: info.get("bbox") for info in img_info_list if info.get("xref")}
                page_xrefs = list(xref_to_bbox.keys())
                
                if page_xrefs:
                    for xref in page_xrefs:
                        try:
                            base_image = doc.extract_image(xref)
                            if not base_image: continue
                            
                            image_bytes = base_image["image"]
                            from io import BytesIO
                            pil_temp = Image.open(BytesIO(image_bytes))
                            w, h = pil_temp.size
                            
                            if w >= self.config.MIN_WIDTH and h >= self.config.MIN_HEIGHT:
                                valid_images_this_page.append({
                                    "pil": pil_temp,
                                    "area": w * h,
                                    "bbox": xref_to_bbox.get(xref)
                                })
                        except Exception:
                            continue
            except Exception:
                pass
            
            if valid_images_this_page:
                # Seleccionar la imagen más grande (histológica)
                mejor = max(valid_images_this_page, key=lambda x: x["area"])
                pil_img = mejor["pil"]
                img_bbox = mejor.get("bbox")
                
                # Apply preprocessing pipeline
                pil_img = self._apply_preprocessing(pil_img)
                
                # Apply magnification logic
                pil_img = self._apply_magnification(pil_img)
                
                nombre_archivo = f"{nombre_pdf}_pag{num_pagina}.png"
                ruta_completa  = os.path.join(self.directorio_salida, nombre_archivo)
                
                try:
                    pil_img.save(ruta_completa, format="PNG")
                    try:
                        ocr_text = pytesseract.image_to_string(pil_img).strip()[:300]
                    except Exception:
                        ocr_text = ""
                    
                    texto_completo_pagina = _texto_pagina_con_contexto(doc, idx_0)
                    # Extraer caption por proximidad espacial (Fix #1)
                    caption = ""
                    if img_bbox:
                        caption = self.extraer_caption_imagen(pagina, img_bbox, texto_completo_pagina)

                    # Extraer etiqueta formal ("Imagen 15.5", "Fig 3A")
                    etiqueta = self._extraer_etiqueta_imagen(caption)
                    if not etiqueta:
                        etiqueta = self._extraer_etiqueta_imagen(texto_completo_pagina)

                    imagenes_extraidas.append({
                        "path": ruta_completa, "fuente_pdf": os.path.basename(pdf_path),
                        "pagina": num_pagina, "indice": 1, "ocr_text": ocr_text,
                        "texto_pagina": texto_completo_pagina,
                        "caption": caption,
                        "nombre_archivo": nombre_archivo,
                        "etiqueta": etiqueta
                    })
                except Exception as e:
                    print(f"  ⚠️ Error guardando pág {num_pagina}: {e}")
            else:
                # FALLBACK: No hay imágenes o son muy chicas -> Renderizar página completa
                pil_full = self._fallback_render_page(pdf_path, num_pagina)
                
                if pil_full:
                    # Apply preprocessing pipeline to fallback image
                    pil_full = self._apply_preprocessing(pil_full)
                    
                    # Apply magnification logic to fallback image
                    pil_full = self._apply_magnification(pil_full)
                    
                    nombre_archivo = f"{nombre_pdf}_pag{num_pagina}_full.png"
                    ruta_completa  = os.path.join(self.directorio_salida, nombre_archivo)
                    
                    try:
                        pil_full.save(ruta_completa, format="PNG")
                        
                        try:
                            ocr_text = pytesseract.image_to_string(pil_full).strip()[:300]
                        except Exception:
                            ocr_text = ""

                        texto_completo_pagina = _texto_pagina_con_contexto(doc, idx_0)
                        # Fallback: sin bbox, buscar solo refs "Fig./Imagen"
                        refs = re.findall(
                            r'(?:Fig(?:ura)?\.?\s*\d+[A-Za-z]?[^.]*\.)|(?:Imagen\s+\d+[\.\d]*[^.]*\.)',
                            texto_completo_pagina, re.IGNORECASE
                        )
                        caption_fb = "\n".join(refs[:3]) if refs else ""

                        etiqueta_fb = self._extraer_etiqueta_imagen(caption_fb)
                        if not etiqueta_fb:
                            etiqueta_fb = self._extraer_etiqueta_imagen(texto_completo_pagina)

                        imagenes_extraidas.append({
                            "path": ruta_completa, "fuente_pdf": os.path.basename(pdf_path),
                            "pagina": num_pagina, "indice": 1, "ocr_text": ocr_text,
                            "texto_pagina": texto_completo_pagina,
                            "caption": caption_fb,
                            "nombre_archivo": nombre_archivo,
                            "etiqueta": etiqueta_fb
                        })
                    except Exception as e:
                        print(f"  ⚠️ Error guardando fallback pág {num_pagina}: {e}")

        doc.close()
        print(f"  📸 {len(imagenes_extraidas)} imágenes procesadas de {os.path.basename(pdf_path)}")
        return imagenes_extraidas

    def extraer_de_directorio(self, directorio: str) -> List[Dict[str, str]]:
        todas = []
        pdfs  = glob.glob(os.path.join(directorio, "*.pdf"))
        print(f"📂 Extrayendo {len(pdfs)} PDFs...")
        for pdf_path in pdfs:
            todas.extend(self.extraer_de_pdf(pdf_path))
        print(f"✅ Total imágenes: {len(todas)}")
        return todas


# =============================================================================
# EXTRACTOR DE TEMARIO
# =============================================================================

class ExtractorTemario:
    def __init__(self, llm):
        self.llm   = llm
        self.temas: List[str] = []

    async def extraer_temario(self, texto_completo: str) -> List[str]:
        print("📋 Extrayendo temario...")
        muestra = texto_completo[:8000]
        system = (
            "Eres un experto en histología. Genera una lista EXHAUSTIVA de temas, "
            "estructuras, tejidos, células, tinciones del manual.\n"
            "Un tema por línea, sin bullets. Solo la lista."
        )
        try:
            resp = await invoke_con_reintento(self.llm, [
                SystemMessage(content=system),
                HumanMessage(content=f"TEXTO:\n{muestra}")
            ])
            temas_raw  = resp.content.strip().split("\n")
            self.temas = [t.strip() for t in temas_raw if t.strip() and len(t.strip()) > 2]
            print(f"✅ Temario: {len(self.temas)} temas")
            with open("temario_histologia.json", "w", encoding="utf-8") as f:
                json.dump(self.temas, f, ensure_ascii=False, indent=2)
            return self.temas
        except Exception as e:
            print(f"❌ Error: {e}")
            return []

    def get_temario_texto(self) -> str:
        return "\n".join(f"- {t}" for t in self.temas[:100]) if self.temas else "No disponible."


# =============================================================================
# EXTRACTOR DE ENTIDADES HISTOLÓGICAS
# =============================================================================

class ExtractorEntidades:
    def __init__(self, llm):
        self.llm = llm

    async def extraer_de_texto(self, texto: str) -> Dict[str, List[str]]:
        system = (
            "Extrae entidades histológicas del texto. "
            'Responde SOLO en JSON: {"tejidos": [...], "estructuras": [...], "tinciones": [...]}\n'
            "Máximo 3 items por categoría. Si no hay, lista vacía."
        )
        try:
            resp = await invoke_con_reintento(self.llm, [
                SystemMessage(content=system),
                HumanMessage(content=texto[:500])
            ])
            texto_resp = re.sub(r"```json\s*|\s*```", "", resp.content.strip())
            resultado  = json.loads(texto_resp)
            return {
                "tejidos":     [t.lower() for t in resultado.get("tejidos", [])[:3]],
                "estructuras": [e.lower() for e in resultado.get("estructuras", [])[:3]],
                "tinciones":   [t.lower() for t in resultado.get("tinciones", [])[:3]],
            }
        except Exception:
            return {"tejidos": [], "estructuras": [], "tinciones": []}

    def extraer_de_texto_sync(self, texto: str) -> Dict[str, List[str]]:
        entidades: Dict[str, List[str]] = {"tejidos": [], "estructuras": [], "tinciones": []}
        TEJIDOS = [
            "epitelio", "conectivo", "muscular", "nervioso", "cartílago", "hueso",
            "sangre", "linfoide", "hepático", "renal", "pulmonar", "dérmico",
            "epitelial", "estroma", "mucosa", "serosa"
        ]
        ESTRUCTURAS = [
            "célula", "núcleo", "citoplasma", "membrana", "gránulo", "fibra",
            "canalículo", "vellosidad", "cripta", "glomérulo", "túbulo", "alvéolo",
            "folículo", "sinusoide", "perla córnea", "cuerpo de albicans",
            "cuerpo de councilman", "queratina", "colágeno"
        ]
        TINCIONES = [
            "h&e", "hematoxilina", "eosina", "pas", "tricrómico", "grocott",
            "ziehl", "giemsa", "reticulina", "alcian blue", "von kossa"
        ]
        texto_lower = texto.lower()
        entidades["tejidos"]     = [t for t in TEJIDOS     if t in texto_lower][:3]
        entidades["estructuras"] = [e for e in ESTRUCTURAS if e in texto_lower][:3]
        entidades["tinciones"]   = [t for t in TINCIONES   if t in texto_lower][:3]
        return entidades


# =============================================================================
# ESTADO DEL GRAFO LANGGRAPH
# =============================================================================

class AgentState(TypedDict):
    messages:                    Annotated[list, operator.add]
    consulta_texto:              str
    imagen_path:                 Optional[str]
    imagen_embedding_uni:        Optional[List[float]]
    imagen_embedding_plip:       Optional[List[float]]
    texto_embedding:             Optional[List[float]]
    contexto_memoria:            str
    contenido_base:              str
    terminos_busqueda:           str
    entidades_consulta:          Dict[str, List[str]]
    consulta_busqueda_texto:     str
    consulta_busqueda_visual:    str
    resultados_busqueda:         List[Dict[str, Any]]
    resultados_validos:          List[Dict[str, Any]]
    contexto_documentos:         str
    respuesta_final:             str
    trayectoria:                 List[Dict[str, Any]]
    user_id:                     str
    tiempo_inicio:               float
    analisis_visual:             Optional[str]
    tiene_imagen:                bool
    imagen_es_nueva:             bool           # True si se subió en ESTE turno
    contexto_suficiente:         bool
    temario:                     List[str]
    tema_valido:                 bool
    tema_encontrado:             Optional[str]
    imagenes_recuperadas:        List[str]
    imagenes_texto_map:          Dict[str, str]   # path -> texto descriptivo del manual
    analisis_comparativo:        Optional[str]
    estructura_identificada:     Optional[str]
    similitud_semantica_dominio: float
    confianza_baja:              bool
    mostrar_imagenes:            bool
    imagenes_para_mostrar:       List[Dict[str, str]]  # [{path, caption, nombre_archivo, etiqueta}]
    historial_conversacional:    str


# =============================================================================
# ASISTENTE PRINCIPAL v4.1
# =============================================================================

class AsistenteHistologiaQdrant:

    SIMILARITY_THRESHOLD = SIMILARITY_THRESHOLD

    def __init__(self):
        self._setup_apis()
        self.llm             = None
        self.memoria         = None
        self.graph           = None
        self.compiled_graph  = None
        self.memory_saver    = None
        self.contenido_base  = ""

        self.uni   = None
        self.plip  = None
        self.embeddings = None
        self.embed_dim = DIM_TEXTO

        self.neo4j: Optional[QdrantVectorStore] = None

        self.extractor_imagenes       = ExtractorImagenesPDF(DIRECTORIO_IMAGENES)
        self.extractor_temario        = None
        self.extractor_entidades      = None
        self.clasificador_semantico: Optional[ClasificadorSemantico] = None

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda":
            try:
                cap = torch.cuda.get_device_capability(0)
                if cap[0] < 7:
                    print(f"⚠️ GPU incompatible detectada (sm_{cap[0]}{cap[1]}). Forzando CPU para evitar fallback_error.")
                    self.device = "cpu"
            except:
                pass
        print(f"✅ AsistenteHistologiaQdrant v4.2 inicializado en {self.device}")

    def _setup_apis(self):
        os.environ["GOOGLE_API_KEY"] = userdata.get("GOOGLE_API_KEY") or ""
        print("✅ APIs configuradas")

    # ------------------------------------------------------------------
    # Inicialización
    # ------------------------------------------------------------------

    async def inicializar_componentes(self):
        self._init_modelos()
        self.memoria             = SemanticMemory(
            llm=self.llm,
            embeddings=self.embeddings,
            uni=self.uni,
            plip=self.plip
        )
        self.extractor_temario   = ExtractorTemario(llm=self.llm)
        self.extractor_entidades = ExtractorEntidades(llm=self.llm)
        self.clasificador_semantico = ClasificadorSemantico(
            llm=self.llm,
            embeddings=self.embeddings, # Gemini
            device=self.device,
            temario=[]   # se actualizará tras extraer el temario
        )

        self.neo4j = QdrantVectorStore(
            url     = userdata.get("QDRANT_URL")      or os.getenv("QDRANT_URL"),
            api_key = userdata.get("QDRANT_KEY")      or os.getenv("QDRANT_KEY"),
        )
        await self.neo4j.connect()
        await self.neo4j.crear_esquema()

        self.memory_saver   = MemorySaver()
        self._crear_grafo()
        self.compiled_graph = self.graph.compile(checkpointer=self.memory_saver)
        print("✅ Todos los componentes inicializados")

    def _init_modelos(self):
        self.llm = ChatGroq(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            api_key=userdata.get("GROQ_API_KEY"),
            temperature=0, max_retries=1
        )
        print("✅ Groq inicializado")
        
        # Inicializar Embeddings (HuggingFace)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': self.device}
        )
        print("✅ Embeddings HuggingFace inicializados")

        # Cargar Modelos (UNI + PLIP)
        self.plip = PlipWrapper(self.device)
        self.plip.load()
        
        self.uni = UniWrapper(self.device)
        self.uni.load()

    def _init_imagebind(self):
        pass # ImageBind eliminado

    # ------------------------------------------------------------------
    # Grafo LangGraph v4.1
    # ------------------------------------------------------------------

    def _crear_grafo(self):
        g = StateGraph(AgentState)

        g.add_node("inicializar",          self._nodo_inicializar)
        g.add_node("procesar_imagen",      self._nodo_procesar_imagen)
        g.add_node("clasificar",           self._nodo_clasificar)
        g.add_node("generar_consulta",     self._nodo_generar_consulta)
        g.add_node("buscar_neo4j",         self._nodo_buscar_neo4j)
        g.add_node("filtrar_contexto",     self._nodo_filtrar_contexto)
        g.add_node("analisis_comparativo", self._nodo_analisis_comparativo)
        g.add_node("generar_respuesta",    self._nodo_generar_respuesta)
        g.add_node("finalizar",            self._nodo_finalizar)
        g.add_node("fuera_temario",        self._nodo_fuera_temario)

        g.add_edge(START,                  "inicializar")

        # Router: si hay imagen → procesar_imagen, si solo texto → clasificar
        g.add_conditional_edges(
            "inicializar",
            self._route_por_modo,
            {"con_imagen": "procesar_imagen", "solo_texto": "clasificar"}
        )
        g.add_edge("procesar_imagen",      "clasificar")

        g.add_conditional_edges(
            "clasificar",
            self._route_por_temario,
            {"en_temario": "generar_consulta", "fuera_temario": "fuera_temario"}
        )
        g.add_edge("fuera_temario",        "finalizar")
        g.add_edge("generar_consulta",     "buscar_neo4j")
        g.add_edge("buscar_neo4j",         "filtrar_contexto")

        # Router: si hay imagen → analisis_comparativo, si no → generar_respuesta
        g.add_conditional_edges(
            "filtrar_contexto",
            self._route_analisis_comparativo,
            {"con_imagen": "analisis_comparativo", "sin_imagen": "generar_respuesta"}
        )
        g.add_edge("analisis_comparativo", "generar_respuesta")
        g.add_edge("generar_respuesta",    "finalizar")
        g.add_edge("finalizar",            END)

        self.graph = g

    # ------------------------------------------------------------------
    # Detección y reescritura
    # ------------------------------------------------------------------

    async def _detectar_solicitud_imagen(self, consulta: str) -> bool:
        """Detecta si la consulta solicita EXPLÍCITAMENTE ver una imagen del manual/base de datos
        usando el LLM para entender la intención del usuario."""
        try:
            resp_check = await invoke_con_reintento(self.llm, [
                SystemMessage(content=(
                    "Tu tarea es determinar si la consulta del usuario solicita EXPLÍCITAMENTE "
                    "ver, mostrar o buscar imágenes, fotos o microfotografías en la base de datos "
                    "o manual.\n"
                    "Solo debes responder 'SI' si la intención del usuario es pedir que el sistema "
                    "le muestre una imagen (ej: 'mostrame una imagen de...', 'quiero ver fotos de...', "
                    "'podes mostrarme imagenes').\n"
                    "Debes responder 'NO' si la palabra 'imagen' se usa en otro contexto (ej: 'qué se "
                    "ve en esta imagen', 'describe la imagen', 'analiza esta foto').\n"
                    "Respondé SOLO con 'SI' o 'NO'."
                )),
                HumanMessage(content=f"CONSULTA: {consulta}")
            ])
            return resp_check.content.strip().upper().startswith("SI")
        except Exception as e:
            print(f"⚠️ Error en detección de intención de imagen por LLM: {e}")
            return False

    async def _reescribir_consulta_con_contexto(self, consulta: str, historial: str) -> str:
        """Reescribe la consulta resolviendo referencias anafóricas usando el historial.
        Usa el LLM para detectar si hay referencias y reescribir."""
        if not historial:
            return consulta
        try:
            # Paso 1: Detectar si necesita reescritura
            resp_check = await invoke_con_reintento(self.llm, [
                SystemMessage(content=(
                    "Determiná si la consulta del usuario hace referencia a temas de la "
                    "conversación previa (usa pronombres como 'eso', 'esto', o pide 'más sobre' "
                    "algo mencionado antes). Respondé SOLO con 'SI' o 'NO'."
                )),
                HumanMessage(content=f"HISTORIAL:\n{historial}\n\nCONSULTA: {consulta}")
            ])
            necesita = "SI" in resp_check.content.strip().upper() or "SÍ" in resp_check.content.strip().upper()
            if not necesita:
                return consulta
            
            # Paso 2: Reescribir — SOLO resolver anáforas, no generar contenido
            resp = await invoke_con_reintento(self.llm, [
                SystemMessage(content=(
                    "Reescribí la siguiente consulta del usuario para que sea autocontenida, "
                    "resolviendo cualquier referencia a temas previos usando el historial.\n\n"
                    "REGLAS ESTRICTAS:\n"
                    "- Devolvé SOLO la consulta reescrita, sin explicaciones.\n"
                    "- La consulta reescrita debe ser UNA SOLA ORACIÓN CORTA (máximo 30 palabras).\n"
                    "- NO generes respuestas, NO describas imágenes, NO inventes contenido.\n"
                    "- Solo reemplazá pronombres y referencias por los términos concretos del historial.\n"
                    "- Si la consulta ya es clara por sí sola, devolvela tal cual."
                )),
                HumanMessage(content=f"HISTORIAL:\n{historial}\n\nCONSULTA ACTUAL: {consulta}")
            ])
            reescrita = resp.content.strip()
            # Validación: si la reescritura es demasiado larga, es una alucinación
            if reescrita and len(reescrita) < len(consulta) * 3 and len(reescrita) < 200:
                print(f"   🔄 Consulta reescrita: '{consulta}' → '{reescrita}'")
                return reescrita
            elif reescrita and len(reescrita) >= 200:
                print(f"   ⚠️ Reescritura descartada (demasiado larga: {len(reescrita)} chars)")
            return consulta
        except Exception as e:
            print(f"   ⚠️ Error reescribiendo consulta: {e}")
            return consulta

    def _route_por_modo(self, state: AgentState) -> str:
        """Decide si procesar imagen o ir directo a clasificar (texto puro).
        Usa el LLM como orquestador cuando hay imagen en memoria."""
        imagen_path = state.get("imagen_path")
        tiene_imagen_nueva = imagen_path and os.path.exists(imagen_path)
        tiene_imagen_memoria = self.memoria and self.memoria.tiene_imagen_previa()
        
        hace_referencia_imagen = False
        
        # Si hay imagen nueva, siempre modo multimodal (no necesita LLM)
        if tiene_imagen_nueva:
            modo = "con_imagen"
            hace_referencia_imagen = True
            print("🖼️ Modo multimodal detectado (imagen nueva)")
        elif tiene_imagen_memoria:
            # Usar LLM para decidir si la consulta hace referencia a la imagen
            consulta = state.get("consulta_texto", "")
            try:
                resp = invoke_con_reintento_sync(self.llm, [
                    SystemMessage(content=(
                        "Sos un clasificador estricto. El usuario tiene una imagen histológica "
                        "subida previamente en el chat. Determiná si su consulta ACTUAL necesita "
                        "analizar esa imagen, o si es una pregunta teórica que NO requiere la imagen.\n\n"
                        "REGLA PRINCIPAL: En caso de duda, respondé TEXTO. Solo respondé IMAGEN "
                        "si la consulta CLARAMENTE necesita ver/analizar la imagen subida.\n\n"
                        "Respondé SOLO con 'IMAGEN' o 'TEXTO'.\n\n"
                        "IMAGEN (necesita la imagen subida):\n"
                        "- 'qué es esto?' → IMAGEN\n"
                        "- 'describí lo que ves' → IMAGEN\n"
                        "- 'analizá la muestra' → IMAGEN\n"
                        "- 'qué se observa en la imagen?' → IMAGEN\n"
                        "- 'qué tejido es el de la foto?' → IMAGEN\n\n"
                        "TEXTO (pregunta teórica, NO necesita la imagen):\n"
                        "- 'qué es el tejido epitelial?' → TEXTO\n"
                        "- 'explicame la sarcomera' → TEXTO\n"
                        "- 'cuáles son los tipos de tejido conectivo?' → TEXTO\n"
                        "- 'qué es el tejido muscular estriado?' → TEXTO\n"
                        "- 'mostrame una imagen de cartílago' → TEXTO\n"
                        "- 'qué características tiene el tejido óseo?' → TEXTO\n"
                        "- 'hablame del tejido nervioso' → TEXTO"
                    )),
                    HumanMessage(content=f"CONSULTA: {consulta}")
                ])
                resultado = resp.content.strip().upper()
                hace_referencia_imagen = resultado.startswith("IMAGEN")
                print(f"   🤖 Clasificador de modo: '{consulta}' → {resultado}")
            except Exception as e:
                print(f"   ⚠️ Error en clasificador de modo, fallback a texto: {e}")
                hace_referencia_imagen = False
            
            if hace_referencia_imagen:
                modo = "con_imagen"
                print("🖼️ Modo multimodal detectado (imagen en memoria + referencia en consulta)")
            else:
                modo = "solo_texto"
                print("📝 Modo solo texto — consulta teórica (ignorando imagen en memoria)")
        else:
            modo = "solo_texto"
            print("📝 Modo solo texto — sin imagen disponible")
        
        # Registrar modo en trayectoria
        state["trayectoria"].append({
            "nodo": "Router:_route_por_modo",
            "modo_detectado": modo,
            "tiene_imagen_nueva": tiene_imagen_nueva,
            "tiene_imagen_memoria": tiene_imagen_memoria,
            "hace_referencia_imagen": hace_referencia_imagen
        })
        
        return modo

    def _route_analisis_comparativo(self, state: AgentState) -> str:
        """Salta análisis comparativo si no hay imagen."""
        if state.get("tiene_imagen") and state.get("imagen_path"):
            return "con_imagen"
        return "sin_imagen"

    def _route_por_temario(self, state: AgentState) -> str:
        return "en_temario"

    # ------------------------------------------------------------------
    # Nodos
    # ------------------------------------------------------------------

    async def _nodo_inicializar(self, state: AgentState) -> AgentState:
        print("📝 Inicializando flujo v4.2 (Neo4j + Conversacional)")
        
        consulta_original = state.get("consulta_texto", "")
        
        # Obtener historial conversacional
        historial = self.memoria.get_history_for_prompt(5)
        state["historial_conversacional"]    = historial
        
        # Reescribir consulta si tiene referencias anafóricas
        consulta_reescrita = await self._reescribir_consulta_con_contexto(consulta_original, historial)
        if consulta_reescrita != consulta_original:
            state["consulta_texto"] = consulta_reescrita
        
        # Detectar si el usuario pide imágenes del contenido
        state["mostrar_imagenes"]            = await self._detectar_solicitud_imagen(consulta_original)
        if state["mostrar_imagenes"]:
            print("   🖼️ Solicitud de imagen detectada")
        
        state["contexto_memoria"]            = self.memoria.get_context(state.get("consulta_texto", ""))
        state["contenido_base"]              = self.contenido_base
        state["tiempo_inicio"]               = time.time()
        state["tiene_imagen"]                = False
        state["imagen_es_nueva"]             = False
        state["contexto_suficiente"]         = False
        state["resultados_validos"]          = []
        state["terminos_busqueda"]           = ""
        state["entidades_consulta"]          = {"tejidos": [], "estructuras": [], "tinciones": []}
        state["imagenes_recuperadas"]        = []
        state["tema_valido"]                 = True
        state["tema_encontrado"]             = None
        state["temario"]                     = self.extractor_temario.temas if self.extractor_temario else []
        state["analisis_comparativo"]        = None
        state["estructura_identificada"]     = None
        state["texto_embedding"]             = None
        state["similitud_semantica_dominio"] = 0.0
        state["trayectoria"] = [{"nodo": "Inicializar", "tiempo": 0}]
        return state

    async def _nodo_procesar_imagen(self, state: AgentState) -> AgentState:
        """
        Tres casos:
          1. Imagen nueva este turno → procesarla y registrarla en memoria.
          2. Sin imagen nueva pero hay activa en memoria → reutilizarla.
          3. Sin imagen en ningún lado → modo texto puro.
        """
        t0 = time.time()
        print("🖼️ Procesando imagen...")

        imagen_path_nuevo = state.get("imagen_path")
        imagen_es_nueva   = False

        if imagen_path_nuevo and os.path.exists(imagen_path_nuevo):
            imagen_path_activo = imagen_path_nuevo
            imagen_es_nueva    = True
            # Se registrará en la memoria con su análisis después de generarlo
            print(f"   🆕 Nueva imagen: {imagen_path_activo}")

        elif self.memoria.tiene_imagen_previa():
            imagen_path_activo = self.memoria.get_imagen_activa()
            state["imagen_path"] = imagen_path_activo
            state["analisis_visual"] = self.memoria.analisis_visual_activo
            print(f"   ♻️  Reutilizando imagen del turno "
                  f"{self.memoria.imagen_turno_subida}: "
                  f"{os.path.basename(imagen_path_activo)}")

        else:
            imagen_path_activo = None

        if imagen_path_activo and os.path.exists(imagen_path_activo):
            try:
                # User images: apply preprocessing for consistency with indexed images
                emb_u = self.uni.embed_image(imagen_path_activo, preprocess=True)
                emb_p = self.plip.embed_image(imagen_path_activo, preprocess=True)
                
                state["imagen_embedding_uni"]   = emb_u.tolist()
                state["imagen_embedding_plip"]  = emb_p.tolist()
                
                state["tiene_imagen"]     = True
                state["imagen_es_nueva"]  = imagen_es_nueva

                if imagen_es_nueva or not state.get("analisis_visual"):
                    state["analisis_visual"] = await self._describir_imagen_histologica(
                        imagen_path_activo
                    )
                    # Guardar en memoria para que persista entre turnos
                    self.memoria.set_imagen(imagen_path_activo, state["analisis_visual"])
                    print(f"   🔬 Análisis visual generado ({len(state['analisis_visual'])} chars)")
                else:
                    print("   ♻️  Reutilizando análisis visual previo del contexto")

                print(f"✅ Imagen lista | nueva={imagen_es_nueva}")
            except Exception as e:
                print(f"❌ Error imagen: {e}")
                import traceback; traceback.print_exc()
                state["imagen_embedding_uni"]   = None
                state["imagen_embedding_plip"]  = None
                state["analisis_visual"]  = None
                state["tiene_imagen"]     = False
        else:
            print("ℹ️ Sin imagen — modo texto")
            state["imagen_embedding"] = None
            state["analisis_visual"]  = None
            state["tiene_imagen"]     = False
            state["imagen_es_nueva"]  = False

        state["trayectoria"].append({
            "nodo":            "ProcesarImagen",
            "tiene_imagen":    state["tiene_imagen"],
            "imagen_es_nueva": imagen_es_nueva,
            "tiempo":          round(time.time()-t0, 2)
        })
        return state

    async def _describir_imagen_histologica(self, imagen_path: str) -> str:
        try:
            with open(imagen_path, "rb") as f:
                data = base64.b64encode(f.read()).decode("utf-8")
            ext  = os.path.splitext(imagen_path)[1].lower()
            mime = "image/png" if ext == ".png" else "image/jpeg"
            features_lista = "\n".join(
                f"  {i+1}. {f}" for i, f in enumerate(FEATURES_DISCRIMINATORIAS)
            )
            msg = HumanMessage(content=[
                {"type": "text", "text": (
                    "Describe esta imagen histológica con máximo rigor.\n\n"
                    "PARTE 1 — DESCRIPCIÓN OBJETIVA (sin nombrar el tejido aún):\n"
                    "Describí SOLO lo que ves: coloración de la matriz, forma y disposición "
                    "de las células, presencia/ausencia de canalículos, vasos sanguíneos, "
                    "fibras visibles, capas envolventes. NO clasifiques todavía.\n\n"
                    f"PARTE 2 — FEATURES DISCRIMINATORIAS:\n{features_lista}\n\n"
                    "PARTE 3 — DIAGNÓSTICO DIFERENCIAL (mínimo 3 opciones):\n"
                    "⚠️ ERRORES COMUNES A EVITAR:\n"
                    "- Cartílago hialino vs Tejido óseo: ambos tienen células en lagunas, "
                    "pero el cartílago tiene matriz VÍTREA/HOMOGÉNEA basófila y condrocitos "
                    "en NIDOS ISÓGENOS, mientras que el hueso tiene matriz CALCIFICADA con "
                    "CANALÍCULOS y osteocitos AISLADOS con prolongaciones.\n"
                    "- Cartílago hialino vs elástico vs fibroso: diferenciar por fibras en la matriz.\n"
                    "- Si ves lagunas con células redondeadas en grupos y matriz homogénea "
                    "sin canalículos ni osteonas, es CARTÍLAGO, no hueso.\n\n"
                    "Para cada opción, listá las evidencias a favor y en contra."
                )},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}}
            ])
            resp = await invoke_con_reintento(self.llm, [msg])
            return resp.content
        except Exception as e:
            print(f"⚠️ Error describiendo imagen: {e}")
            return ""

    async def _nodo_clasificar(self, state: AgentState) -> AgentState:
        """
        1. Extrae términos histológicos y entidades para la búsqueda.
        2. Usa ClasificadorSemantico (ImageBind + LLM) para verificar dominio.
           Reemplaza la verificación por keywords de v4.0.
        """
        t0 = time.time()
        print("🔍 Clasificando consulta (semántico v4.1)...")

        # ── Extracción de términos ─────────────────────────────────────
        # SOLO extraer de la consulta del usuario y contexto de memoria.
        # NO incluir analisis_visual: es la interpretación del LLM sobre la imagen
        # y pre-clasificaría el tejido antes de buscarlo (sesga la búsqueda textual).
        # La búsqueda visual se hace por embeddings UNI/PLIP, no por texto.
        system = (
            "Extrae términos técnicos histológicos de la consulta.\n"
            "Devuelve:\nTEJIDO: [...]\nESTRUCTURA: [...]\nCONCEPTO: [...]\n"
            "TINCIÓN: [...]\nTÉRMINOS_CLAVE: [...]"
        )
        partes = [f"CONSULTA:\n{state['consulta_texto']}"]
        contexto_mem = _safe(state.get("contexto_memoria"))
        if contexto_mem and contexto_mem != "No hay consultas previas.":
            partes.append(f"CONTEXTO:\n{contexto_mem[:300]}")

        try:
            resp = await invoke_con_reintento(self.llm, [
                SystemMessage(content=system),
                HumanMessage(content="\n\n".join(partes))
            ])
            state["terminos_busqueda"] = resp.content
        except Exception as e:
            state["terminos_busqueda"] = state["consulta_texto"]

        # ── Entidades para búsqueda por grafo ──────────────────────────
        # SOLO extraer entidades de la consulta del usuario.
        # NO incluir analisis_visual porque es una interpretación del LLM
        # que puede alucinar (ej: confundir cartílago con hueso) y
        # contaminar la búsqueda con entidades incorrectas.
        state["entidades_consulta"] = await self.extractor_entidades.extraer_de_texto(
            state["consulta_texto"]
        )
        print(f"   🏷️ Entidades: {state['entidades_consulta']}")

        # ── Embedding del texto de consulta (Gemini) ───────────────────
        try:
            emb_texto = self._embed_texto(state["consulta_texto"])
            state["texto_embedding"] = emb_texto
        except Exception as e:
            print(f"⚠️ Error embedding texto: {e}")
            state["texto_embedding"] = None

        # ── Clasificación semántica de dominio ─────────────────────────
        verificacion = await self.clasificador_semantico.clasificar(
            consulta       = state["consulta_texto"],
            analisis_visual= state.get("analisis_visual"),
            imagen_activa  = state.get("tiene_imagen", False),
            temario_muestra= state.get("temario", [])[:60],
        )

        state["tema_valido"]                 = verificacion.get("valido", True)
        state["tema_encontrado"]             = verificacion.get("tema_encontrado")
        state["similitud_semantica_dominio"] = verificacion.get("similitud_dominio", 0.0)

        print(f"   📚 Válido: {state['tema_valido']} | "
              f"Tema: {state['tema_encontrado'] or 'N/A'} | "
              f"Sim: {state['similitud_semantica_dominio']:.3f} | "
              f"Método: {verificacion.get('metodo')}")

        state["trayectoria"].append({
            "nodo":                  "Clasificar",
            "tema_valido":           state["tema_valido"],
            "tema_encontrado":       state["tema_encontrado"],
            "entidades":             state["entidades_consulta"],
            "similitud_dominio":     state["similitud_semantica_dominio"],
            "metodo_clasificacion":  verificacion.get("metodo"),
            "tiempo":                round(time.time()-t0, 2)
        })
        return state

    async def _nodo_fuera_temario(self, state: AgentState) -> AgentState:
        t0 = time.time()
        print("🚫 Consulta fuera del dominio histológico")
        temario = state.get("temario") or []
        muestra = "\n".join(f"  • {t}" for t in temario[:20])
        if len(temario) > 20:
            muestra += f"\n  ... y {len(temario)-20} más"
        state["respuesta_final"] = (
            "⚠️ **Consulta fuera del dominio disponible**\n\n"
            "Tu consulta no parece estar relacionada con histología, patología "
            "o morfología tisular/celular.\n\n"
            f"**Temas disponibles (muestra):**\n{muestra}\n\n"
            "Si tenés una imagen histológica, subila y reformulá tu pregunta. "
            "Ejemplos válidos: '¿qué tipo de tejido es este?', "
            "'describe la estructura observada', 'diagnóstico diferencial'."
        )
        state["contexto_suficiente"] = False
        state["trayectoria"].append({"nodo": "FueraTemario", "tiempo": round(time.time()-t0, 2)})
        return state

    async def _nodo_generar_consulta(self, state: AgentState) -> AgentState:
        t0 = time.time()
        tema_extra = f"\nTEMA: {state['tema_encontrado']}" if state.get("tema_encontrado") else ""
        system = (
            "Genera consultas cortas (≤8 palabras) para histología.\n"
            "Formato:\nCONSULTA_TEXTO: <texto>\n"
            + ("CONSULTA_VISUAL: <visual>" if state.get("tiene_imagen") else "")
        )
        try:
            resp = await invoke_con_reintento(self.llm, [
                SystemMessage(content=system),
                HumanMessage(content=(
                    f"TÉRMINOS:\n{_safe(state.get('terminos_busqueda'))}"
                    f"{tema_extra}\nCONSULTA: {state['consulta_texto']}"
                ))
            ])
            contenido = resp.content
            ct = state["consulta_texto"][:77]
            cv = ""
            if "CONSULTA_TEXTO:" in contenido:
                after = contenido.split("CONSULTA_TEXTO:")[1]
                if "CONSULTA_VISUAL:" in after:
                    ct = after.split("CONSULTA_VISUAL:")[0].strip()[:77]
                    cv = after.split("CONSULTA_VISUAL:")[1].strip()[:77]
                else:
                    ct = after.strip()[:77]
            state["consulta_busqueda_texto"]  = ct
            state["consulta_busqueda_visual"] = cv
        except Exception as e:
            state["consulta_busqueda_texto"]  = state["consulta_texto"][:77]
            state["consulta_busqueda_visual"] = ""

        print(f"   📝 query='{state['consulta_busqueda_texto']}'")
        state["trayectoria"].append({
            "nodo": "GenerarConsulta", "query": state["consulta_busqueda_texto"],
            "tiempo": round(time.time()-t0, 2)
        })
        return state

    async def _nodo_buscar_neo4j(self, state: AgentState) -> AgentState:
        t0 = time.time()
        print("📚 Búsqueda híbrida Neo4j...")

        # Inyectar palabras clave de la consulta en entidades para búsqueda por keywords
        entidades = dict(state.get("entidades_consulta", {}))
        consulta_texto = state.get("consulta_busqueda_texto", "") or state.get("consulta_texto", "")
        # Extraer palabras significativas (>3 chars) como keywords de búsqueda
        stopwords = {"para", "como", "sobre", "este", "esta", "estos", "estas", "podes",
                     "puede", "mostrar", "mostrame", "dame", "quiero", "tipo", "tipos",
                     "explicar", "describir", "decir", "imagen", "imágenes", "imagenes",
                     "ejemplo", "ejemplos", "favor", "hablar", "habla", "cuál", "cual",
                     "cuáles", "cuales", "qué", "que", "cómo", "donde", "dónde", "tiene"}
        palabras_consulta = [
            w.strip("¿?¡!.,;:()\"'") for w in consulta_texto.lower().split()
            if len(w.strip("¿?¡!.,;:()\"'")) > 3 and w.strip("¿?¡!.,;:()\"'") not in stopwords
        ]
        entidades["_consulta"] = palabras_consulta

        resultados = await self.neo4j.busqueda_hibrida(
            texto_embedding        = state.get("texto_embedding"),
            imagen_embedding_uni   = state.get("imagen_embedding_uni"),
            imagen_embedding_plip  = state.get("imagen_embedding_plip"),
            entidades              = entidades,
            top_k                  = 10
        )

        tejidos = state.get("entidades_consulta", {}).get("tejidos", [])
        if len(tejidos) >= 2:
            camino = await self.neo4j.busqueda_camino_semantico(tejidos[0], tejidos[1])
            if camino:
                print(f"   🗺️ Camino semántico: {len(camino)} nodos")
                resultados.extend(camino)

        state["resultados_busqueda"] = resultados
        print(f"✅ {len(resultados)} resultados")

        # --- Imágenes para mostrar: búsqueda semántica contra captions ---
        # Cuando el usuario pide imágenes, buscamos las que mejor coincidan
        # semánticamente con su consulta. Máximo 3, con umbral de similitud.
        if state.get("mostrar_imagenes", False):
            print("   🖼️ Búsqueda semántica de imágenes para mostrar...")
            imgs_para_mostrar = await self.neo4j.busqueda_imagenes_semantica(
                texto_embedding=state.get("texto_embedding", []),
                entidades=entidades,
                embeddings_model=self.embeddings,
                top_k=3
            )
            if imgs_para_mostrar:
                print(f"   ✅ {len(imgs_para_mostrar)} imágenes encontradas por similitud semántica")
                for img in imgs_para_mostrar:
                    print(f"      → {img.get('etiqueta', '')} | {img.get('nombre_archivo', '')} | sim={img.get('similitud_semantica', 0):.3f}")
            else:
                print("   ⚠️ No se encontraron imágenes con similitud semántica suficiente")

            # Fallback: búsqueda por keywords en etiquetas/captions
            # Si la búsqueda semántica no encontró suficientes imágenes, buscar por texto exacto
            if not imgs_para_mostrar or len(imgs_para_mostrar) < 2:
                print("   🔤 Fallback: búsqueda por keywords en etiquetas/captions...")
                imgs_keyword = await self.neo4j.buscar_imagenes_por_referencia(
                    patrones=palabras_consulta, top_k=3
                )
                if imgs_keyword:
                    paths_ya = {img.get("path", "") for img in imgs_para_mostrar}
                    for img_kw in imgs_keyword:
                        kw_path = img_kw.get("imagen_path", "")
                        if kw_path and kw_path not in paths_ya:
                            imgs_para_mostrar.append({
                                "id": img_kw.get("id", ""),
                                "path": kw_path,
                                "caption": img_kw.get("caption_raw", img_kw.get("texto", ""))[:500],
                                "nombre_archivo": img_kw.get("nombre_archivo", os.path.basename(kw_path)),
                                "etiqueta": img_kw.get("etiqueta", ""),
                                "fuente": img_kw.get("fuente", ""),
                                "similitud_semantica": 0.90,
                            })
                            paths_ya.add(kw_path)
                    if len(imgs_para_mostrar) > 3:
                        imgs_para_mostrar = imgs_para_mostrar[:3]
                    print(f"   📋 Total imágenes tras fallback keywords: {len(imgs_para_mostrar)}")

            state["imagenes_para_mostrar"] = imgs_para_mostrar
            if imgs_para_mostrar and not state.get("contexto_suficiente"):
                state["contexto_suficiente"] = True

        state["trayectoria"].append({
            "nodo": "BuscarNeo4j", "hits": len(resultados),
            "imagenes_para_mostrar": len(state.get("imagenes_para_mostrar", [])),
            "tiempo": round(time.time()-t0, 2)
        })
        return state

    async def _nodo_filtrar_contexto(self, state: AgentState) -> AgentState:
        t0     = time.time()
        umbral_imagen = self.SIMILARITY_THRESHOLD
        es_solo_texto = not state.get("tiene_imagen", False)

        # Umbral de texto más permisivo en modo solo texto
        # Qdrant cosine scores tienden a ser más bajos que los de Neo4j vector index
        umbral_texto = 0.30 if es_solo_texto else 0.5

        validos = []
        for r in state["resultados_busqueda"]:
            current_sim = r.get("similitud", 0)
            if r.get("tipo") == "texto" and current_sim < umbral_texto:
                continue
            if r.get("tipo") == "imagen" and current_sim < umbral_imagen:
                continue

            # Si es imagen pero no existe el archivo en disco, lo rechazamos
            if r.get("tipo") == "imagen":
                img_p = r.get("imagen_path")
                if not img_p or not os.path.exists(img_p):
                    continue
            validos.append(r)

        state["resultados_validos"]  = validos
        # Contexto suficiente si hay resultados válidos O si se encontraron imágenes para mostrar
        tiene_imgs_para_mostrar = len(state.get("imagenes_para_mostrar", [])) > 0
        state["contexto_suficiente"] = len(validos) > 0 or tiene_imgs_para_mostrar

        vistas: set = set()
        imagenes_unicas: List[str] = []
        imagenes_texto: Dict[str, str] = {}
        for r in validos:
            img_path = r.get("imagen_path")
            if img_path and os.path.exists(img_path) and img_path not in vistas:
                vistas.add(img_path)
                imagenes_unicas.append(img_path)
                imagenes_texto[img_path] = _safe(r.get('texto', ''))[:500]
        state["imagenes_recuperadas"] = imagenes_unicas
        state["imagenes_texto_map"]   = imagenes_texto

        if validos:
            validos_sorted = sorted(validos, key=lambda x: x.get("similitud", 0), reverse=True)
            bloques = []
            for i, r in enumerate(validos_sorted, 1):
                # Marcar el resultado #1 de imagen como MEJOR MATCH VISUAL
                es_top_imagen = (i == 1 and r.get('tipo') == 'imagen')
                marcador = " ⭐ MEJOR MATCH VISUAL" if es_top_imagen else ""
                enc = (f"[Sección {i}{marcador} | Fuente: {r.get('fuente','N/A')} | "
                       f"Tipo: {r.get('tipo','?')} | Sim: {r.get('similitud',0):.3f}")
                if r.get("imagen_path"):
                    enc += f" | Imagen: {os.path.basename(r['imagen_path'])}"
                enc += "]"
                bloques.append(f"{enc}\n{_safe(r.get('texto',''))[:700]}")
            state["contexto_documentos"] = "\n\n".join(bloques)
            modo_str = "TEXTO" if es_solo_texto else "IMAGEN+TEXTO"
            print(f"✅ {len(validos)} válidos | {len(imagenes_unicas)} imgs | Modo: {modo_str}")
        else:
            state["contexto_documentos"] = ""
            print(f"⚠️ Ningún resultado supera umbral (texto={umbral_texto}, img={umbral_imagen})")

        state["trayectoria"].append({
            "nodo": "FiltrarContexto", "hits_validos": len(validos),
            "imgs": len(state["imagenes_recuperadas"]),
            "modo": "solo_texto" if es_solo_texto else "multimodal",
            "tiempo": round(time.time()-t0, 2)
        })
        return state

    async def _nodo_analisis_comparativo(self, state: AgentState) -> AgentState:
        t0 = time.time()

        if not state.get("tiene_imagen") or not state.get("imagen_path"):
            print("ℹ️ Sin imagen — análisis comparativo omitido")
            state["trayectoria"].append({
                "nodo": "AnalisisComparativo", "motivo": "sin imagen",
                "tiempo": round(time.time()-t0, 2)
            })
            return state

        imagenes_ref = [
            p for p in state.get("imagenes_recuperadas", [])[:3] if os.path.exists(p)
        ]
        if not imagenes_ref:
            print("ℹ️ Sin referencias — análisis comparativo omitido")
            state["analisis_comparativo"] = None
            # Evitar alucinaciones: si hay imagen pero no matches, invalidar contexto
            state["contexto_documentos"] = ""
            state["contexto_suficiente"] = False
            state["imagenes_recuperadas"] = []
            state["trayectoria"].append({
                "nodo": "AnalisisComparativo", "motivo": "sin referencias",
                "tiempo": round(time.time()-t0, 2)
            })
            return state

        print(f"🔬 Análisis comparativo vs {len(imagenes_ref)} referencias...")
        content_parts = [{"type": "text", "text": (
            "Compara la imagen de consulta con las referencias del manual para "
            "determinar si corresponden a la misma estructura histológica.\n\n"
            "=== IMAGEN DE CONSULTA ==="
        )}]

        try:
            with open(state["imagen_path"], "rb") as f:
                data_u = base64.b64encode(f.read()).decode("utf-8")
            ext  = os.path.splitext(state["imagen_path"])[1].lower()
            mime = "image/png" if ext == ".png" else "image/jpeg"
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{data_u}"}
            })
        except Exception as e:
            print(f"⚠️ No se pudo cargar imagen usuario: {e}")
            state["analisis_comparativo"] = None
            return state

        analisis_previo = _safe(state.get("analisis_visual"))
        if analisis_previo:
            content_parts.append({"type": "text", "text": (
                f"\nAnálisis previo:\n{analisis_previo[:600]}\n"
            )})

        imagenes_texto = state.get("imagenes_texto_map", {})
        for i, ref_path in enumerate(imagenes_ref, 1):
            texto_ref = imagenes_texto.get(ref_path, "Sin descripción disponible")
            content_parts.append({"type": "text", "text": (
                f"\n=== REFERENCIA #{i} ({os.path.basename(ref_path)}) ===\n"
                f"DESCRIPCIÓN DEL MANUAL: {texto_ref}"
            )})
            try:
                with open(ref_path, "rb") as f:
                    data_r = base64.b64encode(f.read()).decode("utf-8")
                ext  = os.path.splitext(ref_path)[1].lower()
                mime = "image/png" if ext == ".png" else "image/jpeg"
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{data_r}"}
                })
            except Exception as e:
                print(f"  ⚠️ No se pudo cargar {ref_path}: {e}")

        features_lista = "\n".join(f"  - {f}" for f in FEATURES_DISCRIMINATORIAS)
        content_parts.append({"type": "text", "text": (
            "\n=== INSTRUCCIONES ESTRICTAS ===\n"
            f"Compara rigurosamente basándote en:\n{features_lista}\n\n"
            "IMPORTANTE: Cada referencia incluye la DESCRIPCIÓN DEL MANUAL. "
            "Usa esa descripción como fuente de verdad para identificar la estructura, "
            "NO tu propia interpretación visual.\n\n"
            "TU ROL ES SER UN EVALUADOR OBJETIVO. Tu objetivo principal es determinar verdaderamente si la imagen de consulta y las referencias muestran la misma estructura histológica.\n\n"
            "1. TABLA COMPARATIVA (Markdown): | Feature | Consulta | Ref#1 | Ref#2 |\n"
            "2. VEREDICTO DE IDENTIDAD: Evalúa las similitudes morfológicas y declara si son el MISMO TEJIDO o TEJIDOS DIFERENTES.\n"
            "3. CONCLUSIÓN FINAL: Debes terminar con una de estas dos frases EXACTAS:\n"
            "   - Si coinciden: 'CONCLUSIÓN: SÍ son la misma estructura histológica'\n"
            "   - Si NO coinciden: 'CONCLUSIÓN: TEJIDOS DIFERENTES'\n"
            "   Si hay dudas pero las similitudes son significativas, considera que SÍ coinciden."
        )})

        try:
            resp = await invoke_con_reintento(self.llm, [HumanMessage(content=content_parts)])
            state["analisis_comparativo"] = resp.content
            print(f"✅ Análisis comparativo: {len(resp.content)} chars")
        except Exception as e:
            print(f"❌ Error análisis comparativo: {e}")
            state["analisis_comparativo"] = None

        estructura_db = None
        son_diferentes = False
        analisis_str = state.get("analisis_comparativo", "")
        if analisis_str and ("TEJIDOS DIFERENTES" in analisis_str or "NO son la misma" in analisis_str):
            son_diferentes = True

        if not son_diferentes:
            imagenes_texto = state.get("imagenes_texto_map", {})
            imagenes_rec = state.get("imagenes_recuperadas", [])
            if imagenes_rec and imagenes_texto:
                top_match_path = imagenes_rec[0]
                top_caption = imagenes_texto.get(top_match_path, "")
                if top_caption:
                    primera_linea = top_caption.split('\n')[0].strip()
                    match = re.match(r'Imagen\s+[\d.]+[A-Za-z]?\.\s*(.*)', primera_linea)
                    if match:
                        estructura_db = match.group(1).rstrip('.')
                    else:
                        estructura_db = primera_linea
                    print(f"   → Estructura (DB): {estructura_db} [de {os.path.basename(top_match_path)}]")
        else:
            print("   → Análisis comparativo concluyó que son TEJIDOS DIFERENTES. Imagen no en BD.")
            # Limpiar contexto para evitar alucinaciones
            state["contexto_documentos"] = ""
            state["contexto_suficiente"] = False
            state["imagenes_recuperadas"] = []
        
        state["estructura_identificada"] = estructura_db

        state["trayectoria"].append({
            "nodo": "AnalisisComparativo", "refs": len(imagenes_ref),
            "estructura": state.get("estructura_identificada"),
            "tiempo": round(time.time()-t0, 2)
        })
        return state

    async def _extraer_estructura(self, analisis: str) -> Optional[str]:
        try:
            resp = await invoke_con_reintento(self.llm, [
                SystemMessage(content=(
                    "Extrae el nombre de la estructura histológica más probable. Solo el nombre."
                )),
                HumanMessage(content=analisis[-1000:])
            ])
            return resp.content.strip()
        except Exception:
            return None

    async def _buscar_metadata_imagen(self, nombre_archivo: str) -> dict:
        """Busca metadata (etiqueta, caption, fuente) de una imagen en Qdrant por nombre de archivo."""
        try:
            all_imgs, _ = self.neo4j.client.scroll(
                collection_name=COLLECTION_IMAGENES,
                limit=200,
                with_payload=True,
                with_vectors=False,
            )
            for r in all_imgs:
                nombre_en_bd = (r.payload.get("nombre_archivo", "") or "").lower()
                if nombre_archivo.lower() in nombre_en_bd or nombre_en_bd in nombre_archivo.lower():
                    return {
                        "etiqueta": r.payload.get("etiqueta", ""),
                        "caption": r.payload.get("caption", ""),
                        "fuente": r.payload.get("fuente", ""),
                    }
        except Exception as e:
            print(f"   ⚠️ Error buscando metadata de {nombre_archivo}: {e}")
        return {"etiqueta": "", "caption": "", "fuente": ""}

    async def _nodo_generar_respuesta(self, state: AgentState) -> AgentState:
        t0 = time.time()
        es_solo_texto = not state.get("tiene_imagen", False)
        modo_str = "TEXTO" if es_solo_texto else "MULTIMODAL"
        print(f"💭 Generando respuesta v4.2 [{modo_str}]...")

        # Si no hay contexto textual PERO sí hay imágenes para mostrar,
        # no cortamos: dejamos que el LLM genere la respuesta con las imágenes.
        imagenes_para_mostrar_check = state.get("imagenes_para_mostrar", [])
        tiene_imgs_mostrar = state.get("mostrar_imagenes", False) and len(imagenes_para_mostrar_check) > 0

        if not state["contexto_suficiente"] and not tiene_imgs_mostrar:
            if es_solo_texto:
                # Modo texto sin contexto: respuesta más amable
                print("   ⚠️ Sin contexto RAG para consulta de texto")
                temario = state.get("temario") or []
                muestra = "\n".join(f"  • {t}" for t in temario[:15])
                if len(temario) > 15:
                    muestra += f"\n  ... y {len(temario)-15} más"
                state["respuesta_final"] = (
                    "⚠️ **No encontré información específica sobre eso en el manual**\n\n"
                    "La consulta es válida pero no encontré contenido suficiente en la "
                    "base de datos del manual para responderla con precisión.\n\n"
                    f"**Temas disponibles en el manual (muestra):**\n{muestra}\n\n"
                    "Podés intentar:\n"
                    "- Reformular la pregunta con términos más específicos\n"
                    "- Preguntar sobre alguno de los temas listados arriba\n"
                    "- Subir una imagen histológica para análisis visual"
                )
            else:
                # Modo imagen sin contexto: rechazo estricto
                print("   ⚠️ Sin contexto RAG — rechazo (imagen no encontrada en DB)")
                temario = state.get("temario") or []
                muestra = "\n".join(f"  • {t}" for t in temario[:20])
                if len(temario) > 20:
                    muestra += f"\n  ... y {len(temario)-20} más"
                state["respuesta_final"] = (
                    "⚠️ **Imagen no encontrada en la base de datos**\n\n"
                    "La imagen que proporcionaste no parece coincidir con ninguna "
                    "estructura histológica documentada en el manual actual.\n\n"
                    f"**Temas disponibles (muestra):**\n{muestra}\n\n"
                    "Nota: El sistema está diseñado para identificar únicamente las imágenes "
                    "y variantes presentes en la bibliografía indexada. Si es una imagen "
                    "externa (como una página entera o de otro libro), es posible que no haya coincidencias."
                )
            state["trayectoria"].append({
                "nodo": "GenerarRespuesta", "contexto_suficiente": False,
                "modo": "solo_texto" if es_solo_texto else "multimodal",
                "tiempo": round(time.time()-t0, 2)
            })
            return state

        # Si llegamos aquí con imágenes para mostrar pero sin contexto textual,
        # generamos contexto mínimo a partir de las imágenes encontradas
        if tiene_imgs_mostrar and not state.get("contexto_documentos"):
            print("   📸 Usando imágenes encontradas como contexto para la respuesta")
            bloques_img = []
            for i, img_info in enumerate(imagenes_para_mostrar_check, 1):
                etiqueta = img_info.get('etiqueta', '')
                caption = img_info.get('caption', '')[:500]
                nombre = img_info.get('nombre_archivo', '')
                fuente = img_info.get('fuente', '')
                sim = img_info.get('similitud_semantica', 0)
                bloques_img.append(
                    f"[Sección {i} | Fuente: {fuente} | Tipo: imagen | Sim: {sim:.3f} | "
                    f"Imagen: {nombre}]\n"
                    f"Etiqueta: {etiqueta}\nDescripción: {caption}"
                )
            state["contexto_documentos"] = "\n\n".join(bloques_img)

        tiene_comparativo = bool(_safe(state.get("analisis_comparativo")))
        historial_str = _safe(state.get("historial_conversacional"))

        # ── Instrucción de prosa y continuidad conversacional ──────────
        instruccion_prosa = (
            "ESTILO DE RESPUESTA:\n"
            "- Respondé en prosa, como un profesor explicando en texto corrido.\n"
            "- Evitá listas con bullets, numeración y formato estructurado rígido.\n"
            "- Conectá los conceptos de forma fluida entre párrafos.\n"
            "- Usá un tono didáctico y natural.\n"
        )
        instruccion_continuidad = ""
        if historial_str:
            instruccion_continuidad = (
                "- Tenés un historial de conversación previo. Adaptá tu respuesta como "
                "continuación natural del diálogo, sin repetir información ya proporcionada.\n"
            )
        
        instruccion_imagenes = ""
        
        # Si el usuario pidió ver imágenes de la BD, agregar instrucción especial
        imagenes_para_mostrar = state.get("imagenes_para_mostrar", [])
        if state.get("mostrar_imagenes") and imagenes_para_mostrar:
            descripciones_imgs = []
            for i, img_info in enumerate(imagenes_para_mostrar, 1):
                etiqueta = img_info.get('etiqueta', '')
                caption = img_info.get('caption', '')[:300]
                nombre = img_info.get('nombre_archivo', '')
                desc = f"{i}. **{etiqueta or nombre}**: {caption}"
                descripciones_imgs.append(desc)
            imgs_listado = "\n".join(descripciones_imgs)
            instruccion_imagenes = (
                "\nIMÁGENES DE REFERENCIA ENCONTRADAS EN LA BASE DE DATOS:\n"
                "El usuario pidió ver imágenes del manual. Se encontraron las siguientes "
                "imágenes relevantes que se mostrarán junto a tu respuesta:\n"
                f"{imgs_listado}\n\n"
                "INSTRUCCIÓN: Describí brevemente cada imagen encontrada usando la información "
                "del caption del manual. Indicá qué muestra cada imagen y por qué es relevante "
                "para la consulta del usuario. Las imágenes se mostrarán automáticamente.\n"
            )

        # ── Variables del estado necesarias para el system prompt ───────
        analisis_comp_str   = _safe(state.get("analisis_comparativo"))
        estructura_str      = _safe(state.get("estructura_identificada"))
        analisis_visual_str = _safe(state.get("analisis_visual"), "No disponible")
        contexto_mem_str    = _safe(state.get("contexto_memoria"))
        terminos_str        = _safe(state.get("terminos_busqueda"))
        tema_str            = _safe(state.get("tema_encontrado"), "N/A")
        entidades_str       = json.dumps(state.get("entidades_consulta", {}), ensure_ascii=False)

        # Instrucción extra si el análisis comparativo ya identificó la estructura
        instruccion_estructura = ""
        if estructura_str:
            instruccion_estructura = (
                f"\n⚠️ ESTRUCTURA IDENTIFICADA POR ANÁLISIS COMPARATIVO: {estructura_str}\n"
                "Esta identificación proviene de una comparación directa imagen-vs-referencia "
                "y tiene MÁXIMA PRIORIDAD. Usá la descripción del manual CORRESPONDIENTE A ESTA "
                "ESTRUCTURA. Ignorá descripciones de otras imágenes que hablen de tejidos diferentes.\n"
            )

        # ── System prompt diferenciado según modo ──────────────────────
        # Construir ontología disponible para inyectar en el prompt
        temario = state.get("temario", [])
        ontologia_str = ""
        if temario:
            temas_muestra = temario[:40]
            ontologia_str = (
                "\nONTOLOGÍA DISPONIBLE (temas del manual):\n"
                + "\n".join(f"  • {t}" for t in temas_muestra)
            )
            if len(temario) > 40:
                ontologia_str += f"\n  ... y {len(temario) - 40} temas más."
            ontologia_str += (
                "\n\nIMPORTANTE: Si la consulta del usuario trata sobre un tema que NO está "
                "en esta ontología y NO aparece en las SECCIONES DEL MANUAL proporcionadas, "
                "indicá claramente que ese tema no está disponible en el manual actual. "
                "NO inventes contenido sobre temas fuera de la ontología.\n"
            )

        if es_solo_texto:
            system_prompt = (
                "Eres un asistente experto de histología. Respondés consultas de texto "
                "basándote EXCLUSIVAMENTE en el contenido del manual/base de datos.\n\n"
                "REGLAS FUNDAMENTALES:\n"
                "1. FUENTE DE VERDAD: Usá SOLO la información de las SECCIONES DEL MANUAL proporcionadas.\n"
                "2. CITAS: Citá las fuentes con [Manual: archivo].\n"
                "3. NO inventes información que no esté en las secciones proporcionadas.\n"
                "4. Si la información es parcial, indicá qué partes provienen del manual y cuáles no están disponibles.\n"
                "5. Si el tema consultado NO aparece en las secciones del manual ni en la ontología, "
                   "respondé: 'Este tema no se encuentra en el manual disponible actualmente.'\n"
                "6. No diagnósticos clínicos salvo que estén explícitos en el manual.\n\n"
                f"{ontologia_str}\n"
                f"{instruccion_prosa}"
                f"{instruccion_continuidad}"
                f"{instruccion_imagenes}"
            )
        else:
            nota_comp = (
                "\n\nIMPORTANTE: El análisis comparativo tiene PRIORIDAD en el diagnóstico diferencial."
                if tiene_comparativo else ""
            )
            regla_validacion = (
                "2. VALIDACIÓN: Revisa el 'ANÁLISIS COMPARATIVO'. "
                "Si la conclusión final del análisis dice explícitamente 'TEJIDOS DIFERENTES' o 'NO son la misma estructura', "
                "entonces respondé con algo como: "
                "'No puedo identificar esta estructura con certeza porque no se encuentra en el contenido del manual disponible. "
                "Te sugiero consultar con tu docente o subir otra imagen con mejor resolución.' "
                "Si el análisis menciona 'similitudes', 'coinciden', 'mismo tejido', o 'SÍ son la misma estructura', "
                "entonces asume que SÍ coinciden y continúa con el paso 3 detallando el tejido según el manual.\n"
            )
            system_prompt = (
                "Eres un asistente de histología. Responde SOLO con el contenido del manual o la imagen visible en el chat.\n\n"
                "REGLAS FUNDAMENTALES:\n"
                "1. PRIORIDAD ABSOLUTA: La DESCRIPCIÓN TEXTUAL DEL MANUAL es la fuente de verdad. "
                   "Si el texto del manual dice 'Tejido nervioso corteza cerebelosa', ESO es lo correcto, "
                   "sin importar tu propia interpretación visual de la imagen.\n"
                "   ATENCIÓN: Si se proporciona una 'ESTRUCTURA IDENTIFICADA' por el análisis comparativo, "
                   "esa estructura es el resultado de una comparación directa imagen-vs-referencia y tiene "
                   "MÁXIMA PRIORIDAD. Usá la descripción del manual CORRESPONDIENTE A ESA ESTRUCTURA, "
                   "no la descripción de otras imágenes recuperadas.\n"
                "2. Cita: [Manual: archivo] | [Imagen: archivo]\n"
                "3. Para cada 'IMAGEN DE REFERENCIA', indica el nombre y la descripción textual del manual.\n"
                "4. NO hagas diagnósticos propios basados en tu interpretación visual. "
                   "Usa SIEMPRE el texto del manual asociado a la imagen.\n"
                "5. Si el tema NO aparece en las secciones del manual ni en la ontología, "
                   "indicá que no está disponible en el manual actual.\n"
                "6. No diagnósticos clínicos salvo que estén explícitos.\n\n"
                f"{ontologia_str}\n"
                f"{instruccion_estructura}"
                f"{instruccion_prosa}"
                f"{instruccion_continuidad}"
                f"{instruccion_imagenes}"
                "VALIDACIÓN:\n"
                f"{regla_validacion}"
                f"{nota_comp}"
            )

        seccion_comp = (f"\n\n**ANÁLISIS COMPARATIVO:**\n{analisis_comp_str[:2000]}"
                        if analisis_comp_str else "")
        seccion_est  = (f"\n\n**ESTRUCTURA IDENTIFICADA:** {estructura_str}"
                        if estructura_str else "")

        content_parts = []
        
        # Inyectar historial conversacional si existe
        if historial_str:
            content_parts.append({"type": "text", "text": (
                f"**HISTORIAL DE CONVERSACIÓN:**\n{historial_str}\n\n---\n"
            )})
        
        # Limitar contexto_documentos para no saturar el prompt
        ctx_docs = state['contexto_documentos']
        if len(ctx_docs) > 4000:
            ctx_docs = ctx_docs[:4000] + "\n... [contexto truncado]"
        
        content_parts.append({"type": "text", "text": (
            f"**CONSULTA:** {state['consulta_texto']}\n\n"
            f"**TÉRMINOS:** {terminos_str[:300]}\n\n"
            f"**ENTIDADES (grafo):** {entidades_str}\n\n"
            f"**TEMA:** {tema_str}\n\n"
            f"**ANÁLISIS VISUAL USUARIO:**\n{analisis_visual_str[:800]}\n\n"
            f"**SECCIONES DEL MANUAL:**\n{ctx_docs}"
            f"{seccion_comp}{seccion_est}\n\n"
            "Responde EXCLUSIVAMENTE con el contenido del manual e imágenes de referencia."
        )})

        imagen_path = state.get("imagen_path")
        if state.get("tiene_imagen") and imagen_path and os.path.exists(imagen_path):
            try:
                with open(imagen_path, "rb") as f:
                    data = base64.b64encode(f.read()).decode("utf-8")
                ext  = os.path.splitext(imagen_path)[1].lower()
                mime = "image/png" if ext == ".png" else "image/jpeg"
                label = ("NUEVA IMAGEN DEL USUARIO" if state.get("imagen_es_nueva")
                         else f"IMAGEN ACTIVA (turno {self.memoria.imagen_turno_subida})")
                content_parts.append({"type": "text", "text": f"\n**{label}:**"})
                content_parts.append({"type": "image_url",
                                       "image_url": {"url": f"data:{mime};base64,{data}"}})
                print(f"   📷 {label}")
            except Exception as e:
                print(f"   ⚠️ No se pudo añadir imagen usuario: {e}")

        imagenes_usadas = 0
        # Solo enviar imágenes al LLM si el usuario subió una imagen (modo multimodal)
        # En modo solo texto, el LLM referencia imágenes por etiqueta ("Imagen 15.1")
        # desde el texto de los chunks, y _nodo_finalizar las busca en Qdrant
        if not es_solo_texto:
            for img_path in state.get("imagenes_recuperadas", [])[:3]:
                if not os.path.exists(img_path):
                    continue
                try:
                    with open(img_path, "rb") as f:
                        data = base64.b64encode(f.read()).decode("utf-8")
                    ext    = os.path.splitext(img_path)[1].lower()
                    mime   = "image/png" if ext == ".png" else "image/jpeg"
                    nombre = os.path.basename(img_path)
                    content_parts.append({"type": "text",
                                           "text": f"\n**REFERENCIA [Imagen: {nombre}]:**"})
                    content_parts.append({"type": "image_url",
                                           "image_url": {"url": f"data:{mime};base64,{data}"}})
                    imagenes_usadas += 1
                except Exception as e:
                    print(f"   ⚠️ {img_path}: {e}")

        print(f"   📊 {1 if state.get('tiene_imagen') else 0} usuario + {imagenes_usadas} manual")

        try:
            resp = await invoke_con_reintento(self.llm, [
                SystemMessage(content=system_prompt),
                HumanMessage(content=content_parts)
            ])
            if state.get("confianza_baja"):
                warning = "⚠️ *Nota: Esta respuesta se generó con confianza baja (<71% de coincidencia) en los apuntes, tómalo en cuenta.*"
                if "Hola, ¿en qué te puedo ayudar sobre histología?" in resp.content:
                    state["respuesta_final"] = resp.content
                else:
                    state["respuesta_final"] = f"{warning}\n\n{resp.content}"
            else:
                state["respuesta_final"] = resp.content
            print(f"✅ Respuesta: {len(resp.content)} chars")
        except Exception as e:
            print(f"❌ Error: {e}")
            state["respuesta_final"] = f"Error: {e}"

        state["trayectoria"].append({
            "nodo": "GenerarRespuesta", "contexto_suficiente": True,
            "imagenes_usadas": imagenes_usadas, "tiene_comparativo": tiene_comparativo,
            "tiempo": round(time.time()-t0, 2)
        })
        return state

    async def _nodo_finalizar(self, state: AgentState) -> AgentState:
        # Guardar siempre en memoria, no solo cuando hay contexto suficiente
        if state.get("respuesta_final"):
            self.memoria.add_interaction(state["consulta_texto"], state["respuesta_final"])

        total = round(time.time() - state["tiempo_inicio"], 2)
        state["trayectoria"].append({"nodo": "Finalizar", "tiempo_total": total})

        with open("trayectoria_qdrant.json", "w", encoding="utf-8") as f:
            json.dump({
                "trayectoria":             state["trayectoria"],
                "estructura_identificada": state.get("estructura_identificada"),
                "imagenes_recuperadas":    state.get("imagenes_recuperadas", []),
                "entidades_consulta":      state.get("entidades_consulta", {}),
                "mostrar_imagenes":        state.get("mostrar_imagenes", False),
                "imagenes_para_mostrar":   state.get("imagenes_para_mostrar", []),
            }, f, indent=4, ensure_ascii=False)

        print(f"✅ Flujo v4.2 completado en {total}s")
        if state.get("estructura_identificada"):
            print(f"   → Estructura: {state['estructura_identificada']}")
        return state

    # ------------------------------------------------------------------
    # Embeddings (Gemini Text)
    # ------------------------------------------------------------------

    def _embed_texto(self, texto: str) -> List[float]:
        # Usa langchain GoogleGenerativeAIEmbeddings
        return embed_query_con_reintento(self.embeddings, texto)


    # ------------------------------------------------------------------
    # Indexación en Neo4j
    # ------------------------------------------------------------------

    def _leer_pdf(self, path: str) -> str:
        try:
            import fitz
            doc = fitz.open(path)
            texto = "".join(page.get_text() for page in doc)
            doc.close()
            return texto
        except Exception as e:
            print(f"⚠️ Error leyendo {path}: {e}")
            return ""

    def _leer_pdf_por_paginas(self, path: str) -> Dict[int, str]:
        paginas = {}
        try:
            import fitz
            doc = fitz.open(path)
            for i, page in enumerate(doc):
                paginas[i + 1] = page.get_text()
            doc.close()
        except Exception as e:
            print(f"⚠️ Error leyendo por páginas {path}: {e}")
        return paginas

    def _chunks(self, texto: str, size: int = 500) -> List[str]:
        return [texto[i:i+size] for i in range(0, len(texto), size)]

    def procesar_contenido_base(self, directorio: str = DIRECTORIO_PDFS) -> str:
        pdfs = glob.glob(os.path.join(directorio, "*.pdf"))
        if not pdfs:
            print(f"⚠️ Sin PDFs en {directorio}")
            return ""
        self.contenido_base = "\n".join(self._leer_pdf(p) for p in pdfs)
        print(f"📚 {len(pdfs)} PDFs leídos ({len(self.contenido_base)} chars)")
        return self.contenido_base[:500]

    async def extraer_y_preparar_temario(self):
        if not self.contenido_base:
            print("⚠️ Contenido base vacío")
            return
        await self.extractor_temario.extraer_temario(self.contenido_base)
        # Actualizar clasificador semántico con el temario real
        if self.clasificador_semantico:
            self.clasificador_semantico.temario = self.extractor_temario.temas
            print(f"   🔄 Clasificador semántico actualizado con "
                  f"{len(self.extractor_temario.temas)} temas")

    async def indexar_en_qdrant(self, directorio_pdfs: str = DIRECTORIO_PDFS,
                                 imagen_files_extra: Optional[List[str]] = None,
                                 forzar: bool = False):
        # ── Verificar si ya hay datos ────────────────────────────────
        if not forzar:
            try:
                n_chunks = self.neo4j.client.count(collection_name=COLLECTION_CHUNKS).count
                n_imgs   = self.neo4j.client.count(collection_name=COLLECTION_IMAGENES).count
                if n_chunks > 0 and n_imgs > 0:
                    print(f"✅ Base de datos Qdrant ya poblada ({n_chunks} chunks, {n_imgs} imágenes). Saltando indexación.")
                    print("   (Usá --reindex --force para forzar re-indexación)")
                    return
            except Exception as e:
                print(f"⚠️ No se pudo verificar estado de la BD: {e}")

        print("📄 Preparando extracción de imágenes para vincular a chunks...")
        imagenes_pdf = self.extractor_imagenes.extraer_de_directorio(directorio_pdfs)
        
        img_por_pdf_pag = {}
        for img_info in imagenes_pdf:
            k = (img_info["fuente_pdf"], img_info["pagina"])
            if k not in img_por_pdf_pag:
                img_por_pdf_pag[k] = []
            img_por_pdf_pag[k].append(img_info["path"])

        print("📄 Indexando chunks de texto en Qdrant (por página)...")
        for pdf_path in glob.glob(os.path.join(directorio_pdfs, "*.pdf")):
            fuente = os.path.basename(pdf_path)
            # await self.neo4j.upsert_pdf(fuente) # No longer needed in Qdrant
            paginas_texto = self._leer_pdf_por_paginas(pdf_path)
            for num_pagina, texto_pag in paginas_texto.items():
                chunks = self._chunks(texto_pag)
                img_paths = img_por_pdf_pag.get((fuente, num_pagina), [])
                
                for i, chunk in enumerate(chunks):
                    if not chunk.strip(): continue
                    try:
                        emb       = self._embed_texto(chunk)
                        chunk_id  = f"chunk_{fuente}_p{num_pagina}_{i}"
                        entidades = self.extractor_entidades.extraer_de_texto_sync(chunk)
                        await self.neo4j.upsert_chunk(
                            chunk_id=chunk_id, texto=chunk, fuente=fuente,
                            chunk_idx=i, embedding=emb, entidades=entidades,
                            pagina=num_pagina, imagenes_pagina=img_paths
                        )
                    except Exception as e:
                        print(f"  ⚠️ Chunk {fuente} p{num_pagina}-{i}: {e}")

        print("📸 Indexando imágenes de PDFs en Neo4j...")
        imagenes_pdf = self.extractor_imagenes.extraer_de_directorio(directorio_pdfs)
        for img_info in imagenes_pdf:
            img_path = img_info["path"]
            if not os.path.exists(img_path):
                continue
            try:
                # Images from PDF extraction are already preprocessed, don't preprocess again
                emb_u  = self.uni.embed_image(img_path, preprocess=False)
                emb_p  = self.plip.embed_image(img_path, preprocess=False)
                
                img_id = f"img_{img_info['fuente_pdf']}_{img_info['pagina']}"
                
                nombre_archivo = img_info.get("nombre_archivo", os.path.basename(img_path))
                etiqueta = img_info.get("etiqueta", "")
                caption = img_info.get("caption", "")
                
                # Log de la imagen para trazabilidad
                print(f"  📷 {nombre_archivo} | Etiqueta: {etiqueta or 'N/A'} | Caption: {len(caption)} chars")
                
                emb_texto = None
                # Construir texto enriquecido priorizando etiqueta + título
                # para que el embedding capture la esencia de la imagen
                titulo_caption = ""
                if caption:
                    titulo_caption = caption.split('\n')[0].strip()
                partes_emb = []
                if etiqueta and titulo_caption:
                    partes_emb.append(f"{etiqueta}: {titulo_caption}")
                    partes_emb.append(titulo_caption)  # repetir para dar peso
                elif titulo_caption:
                    partes_emb.append(titulo_caption)
                    partes_emb.append(titulo_caption)  # repetir para dar peso
                if caption:
                    partes_emb.append(caption[:300])
                texto_relevante = "\n".join(partes_emb) if partes_emb else (img_info.get("texto_pagina", "")[:500])
                if texto_relevante:
                    try:
                        emb_texto = self._embed_texto(texto_relevante)
                    except Exception:
                        pass
                
                await self.neo4j.upsert_imagen(
                    imagen_id=img_id, path=img_path,
                    fuente=img_info["fuente_pdf"], pagina=img_info["pagina"],
                    ocr_text=img_info.get("ocr_text", ""),
                    texto_pagina=img_info.get("texto_pagina", ""),
                    emb_uni=emb_u.tolist(),
                    emb_plip=emb_p.tolist(),
                    emb_texto=emb_texto,
                    caption=caption,
                    nombre_archivo=nombre_archivo,
                    etiqueta=etiqueta
                )
            except Exception as e:
                print(f"  ⚠️ Imagen {img_path}: {e}")

        for img_path in (imagen_files_extra or []):
            if not os.path.exists(img_path):
                continue
            try:

                ocr = ""
                try:
                    ocr = pytesseract.image_to_string(Image.open(img_path)).strip()
                except Exception:
                    pass
                img_id = f"img_extra_{os.path.basename(img_path)}"
                # Extra images are not preprocessed, apply preprocessing
                emb_u = self.uni.embed_image(img_path, preprocess=True)
                emb_p = self.plip.embed_image(img_path, preprocess=True)
                emb_texto = None
                if ocr:
                    try:
                        emb_texto = self._embed_texto(ocr[:500])
                    except Exception:
                        pass
                await self.neo4j.upsert_imagen(
                    imagen_id=img_id, path=img_path, fuente=os.path.basename(img_path),
                    pagina=0, ocr_text=ocr[:300], texto_pagina="", emb_uni=emb_u.tolist(), emb_plip=emb_p.tolist(),
                    emb_texto=emb_texto
                )
            except Exception as e:
                print(f"  ❌ Imagen extra {img_path}: {e}")

        await self.neo4j.crear_relaciones_similitud(SIMILAR_IMG_THRESHOLD)
        print("✅ Indexación Neo4j completada")

    # ------------------------------------------------------------------
    # Punto de entrada público
    # ------------------------------------------------------------------

    async def consultar(self, consulta_texto: str,
                         imagen_path: Optional[str] = None,
                         user_id: str = "default_user") -> str:
        """
        La imagen es completamente opcional en cada llamada.
        Si no se pasa, el router decide si reutilizar la imagen activa en memoria.
        """
        tiene_imagen_activa = self.memoria.tiene_imagen_previa() or bool(imagen_path)

        print(f"\n{'='*70}")
        print(f"🔬 RAG Histología Neo4j v4.2 | umbral={self.SIMILARITY_THRESHOLD}")
        print(f"   Texto:         {consulta_texto}")
        print(f"   Imagen turno:  {imagen_path or 'ninguna'}")
        print(f"   Imagen memoria: {self.memoria.get_imagen_activa() or 'ninguna'}")
        print(f"{'='*70}")

        initial_state = AgentState(
            messages=[], consulta_texto=consulta_texto,
            imagen_path=imagen_path,
            imagen_embedding_uni=None, imagen_embedding_plip=None, texto_embedding=None, contexto_memoria="",
            contenido_base=self.contenido_base, terminos_busqueda="",
            entidades_consulta={"tejidos": [], "estructuras": [], "tinciones": []},
            consulta_busqueda_texto="", consulta_busqueda_visual="",
            resultados_busqueda=[], resultados_validos=[], contexto_documentos="",
            respuesta_final="", trayectoria=[], user_id=user_id, tiempo_inicio=time.time(),
            analisis_visual=None, tiene_imagen=False, imagen_es_nueva=False,
            contexto_suficiente=False, temario=self.extractor_temario.temas,
            tema_valido=True, tema_encontrado=None, imagenes_recuperadas=[],
            imagenes_texto_map={},
            analisis_comparativo=None, estructura_identificada=None,
            similitud_semantica_dominio=0.0, confianza_baja=False,
            mostrar_imagenes=False, imagenes_para_mostrar=[],
            historial_conversacional="",
        )

        config = {
            "configurable": {"thread_id": user_id},
            "run_name":     f"consulta-neo4j-v4.1-{user_id}",
            "tags":         ["rag", "histologia", "neo4j", "imagebind", "v4.1"],
            "metadata": {
                "tiene_imagen_nueva":  imagen_path is not None,
                "tiene_imagen_activa": tiene_imagen_activa,
                "consulta":            consulta_texto[:100],
                "version":             "4.1"
            }
        }
        try:
            final     = await self.compiled_graph.ainvoke(initial_state, config=config)
            respuesta = final["respuesta_final"]
        except Exception as e:
            import traceback; traceback.print_exc()
            respuesta = f"Error: {e}"
            final = {}

        print(f"\n{'='*70}\n📖 RESPUESTA:\n{'='*70}")
        print(respuesta)
        print("="*70)
        
        # Retornar dict con toda la info para el server
        self._ultimo_resultado = {
            "respuesta": respuesta,
            "mostrar_imagenes": final.get("mostrar_imagenes", False),
            "imagenes_recuperadas": final.get("imagenes_recuperadas", []),
            "estructura_identificada": final.get("estructura_identificada"),
            "imagenes_para_mostrar": final.get("imagenes_para_mostrar", []),
        }
        
        return respuesta

    async def cerrar(self):
        if self.neo4j:
            await self.neo4j.close()


# =============================================================================
# MODO INTERACTIVO MEJORADO v4.1
# =============================================================================

async def modo_interactivo(reindex: bool = False, force: bool = False):
    asistente = AsistenteHistologiaQdrant()
    await asistente.inicializar_componentes()

    print("\n🔄 Leyendo el manual...")
    asistente.procesar_contenido_base(DIRECTORIO_PDFS)

    print("\n📋 Extrayendo temario...")
    await asistente.extraer_y_preparar_temario()
    print(f"   → {len(asistente.extractor_temario.temas)} temas")

    print("\n💾 Indexando en Neo4j...")
    if reindex:
        await asistente.indexar_en_qdrant(DIRECTORIO_PDFS, forzar=force)
    else:
        print("   (Saltando indexación — usar --reindex para forzar)")

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║  RAG Histología Neo4j v4.2 — Chat Interactivo               ║
╠══════════════════════════════════════════════════════════════╣
║  • Escribí tu pregunta y presioná Enter                     ║
║  • Para subir una imagen: escribí el PATH cuando se pida    ║
║  • La imagen se recuerda entre turnos — no es obligatoria   ║
║  • ✨ Consultas de texto puro — flujo optimizado            ║
║  • Comandos especiales:                                     ║
║      temario       → ver temas disponibles                  ║
║      imagen actual → ver imagen activa en el chat           ║
║      nueva imagen  → limpiar imagen activa                  ║
║      salir         → terminar                               ║
╚══════════════════════════════════════════════════════════════╝
""")

    while True:
        try:
            print("\n" + "─"*60)

            img_activa = asistente.memoria.get_imagen_activa()
            if img_activa:
                print(f"📌 Imagen activa: {os.path.basename(img_activa)} "
                      f"(turno {asistente.memoria.imagen_turno_subida})")

            consulta = input("💬 Vos: ").strip()
            if not consulta:
                continue

            cmd = consulta.lower()

            if cmd in ("salir", "exit", "quit"):
                await asistente.cerrar()
                print("👋 ¡Hasta luego!")
                break

            if cmd == "temario":
                print("\n📚 TEMAS DISPONIBLES:")
                for i, t in enumerate(asistente.extractor_temario.temas, 1):
                    print(f"  {i:3}. {t}")
                continue

            if cmd == "imagen actual":
                if img_activa:
                    print(f"📌 Imagen activa: {img_activa}")
                    print(f"   Subida en turno: {asistente.memoria.imagen_turno_subida}")
                else:
                    print("ℹ️ No hay imagen activa en el chat.")
                continue

            if cmd == "nueva imagen":
                asistente.memoria.set_imagen(None)
                print("🗑️  Imagen activa eliminada. El próximo turno será solo texto.")
                continue

            # Imagen opcional
            imagen_path = None
            img_input   = input("🖼️  Imagen (path o Enter para omitir): ").strip()

            if img_input:
                if os.path.exists(img_input):
                    imagen_path = img_input
                    print(f"✅ Nueva imagen: {imagen_path}")
                else:
                    print(f"⚠️ No encontrada: {img_input} — se usará imagen activa (si la hay)")
            else:
                if img_activa:
                    print(f"♻️  Se usará imagen activa: {os.path.basename(img_activa)}")
                else:
                    print("ℹ️ Sin imagen — consulta solo de texto")

            await asistente.consultar(consulta, imagen_path)

        except KeyboardInterrupt:
            await asistente.cerrar()
            print("\n\n👋 Interrumpido")
            break
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--reindex", action="store_true", help="Indexar en Neo4j (salta si ya hay datos)")
    parser.add_argument("--force", action="store_true", help="Forzar re-indexación aunque haya datos")
    args = parser.parse_args()

    import logging
    logging.basicConfig(level=logging.INFO)
    print(f"✅ GPU: {torch.cuda.get_device_name()}" if torch.cuda.is_available() else "⚠️ CPU mode")
    os.makedirs("logs", exist_ok=True)
    os.makedirs(DIRECTORIO_IMAGENES, exist_ok=True)
    asyncio.run(modo_interactivo(reindex=args.reindex, force=args.force))