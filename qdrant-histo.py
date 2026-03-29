# =============================================================================
# RAG Multimodal de Histología con Qdrant — VERSIÓN 5.1
# =============================================================================
# Cambios sobre v5.0:
#   INDEXACIÓN Y RECUPERACIÓN IMAGEN↔TEXTO UNIFICADA
#   1. `busqueda_vectorial_imagen` retorna `texto_pagina` en el resultado.
#   2. `_nodo_filtrar_contexto` incluye `texto_pagina` al construir el bloque
#      de contexto para cada imagen recuperada.
#   3. `_nodo_generar_respuesta` pasa el `texto_pagina` al LLM junto con la imagen.
#   4. `upsert_imagen` ahora también indexa un vector de texto (Gemini) opcional
#      construido a partir de ocr_text + texto_pagina, para que una búsqueda de
#      texto pueda recuperar directamente nodos de imagen enriquecidos.
#   5. `busqueda_hibrida` agrega una búsqueda vectorial de texto sobre
#      COLLECTION_IMAGENES (usando el nuevo vector "texto_emb"), de modo que
#      al buscar por texto también se recuperan las imágenes cuyo texto de
#      página coincide semánticamente.
#   6. El mapa página→imagen se construye antes de indexar los chunks, y cada
#      chunk guarda las rutas absolutas de las imágenes de su página (ya existía
#      en v5.0 pero se refuerza con validación de existencia).
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

# Qdrant Cloud (vector store principal)
from qdrant_client import models
from qdrant_client.models import Filter, FieldCondition, MatchAny, MatchValue

import fitz # PyMuPDF
from pdf2image import convert_from_path
import pytesseract

from langchain_google_genai import GoogleGenerativeAIEmbeddings
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
SIMILARITY_THRESHOLD  = 0.25
# Dimensiones de embeddings
DIM_TEXTO_GEMINI = 3072
DIM_IMG_UNI      = 1024
DIM_IMG_PLIP     = 512

DIRECTORIO_IMAGENES   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "imagenes_extraidas")
DIRECTORIO_PDFS       = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pdf")

# Colecciones Qdrant Cloud
COLLECTION_CHUNKS   = "histo_chunks"       # Texto (Gemini)
COLLECTION_IMAGENES = "histo_imagenes"     # Imágenes (UNI + PLIP + texto_emb)

SIMILAR_IMG_THRESHOLD = 0.85

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
        "LANGCHAIN_PROJECT":    "q_histo"
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

    def embed_image(self, image_path: str) -> np.ndarray:
        if not self.model: return np.zeros(DIM_IMG_PLIP)
        try:
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
            self.model = timm.create_model(
                "hf_hub:MahmoodLab/UNI",
                pretrained=True,
                init_values=1e-5,
                dynamic_img_size=True
            )
            self.model.to(self.device).eval()
            from timm.data import resolve_data_config
            from timm.data.transforms_factory import create_transform
            config = resolve_data_config(self.model.pretrained_cfg, model=self.model)
            self.transform = create_transform(**config)
            print("✅ UNI cargado")
        except Exception as e:
            print(f"❌ Error cargando UNI: {e}")

    def embed_image(self, image_path: str) -> np.ndarray:
        if not self.model: return np.zeros(DIM_IMG_UNI)
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            with torch.inference_mode():
                emb = self.model(image_tensor)
            return emb.cpu().numpy().flatten()
        except Exception as e:
            print(f"⚠️ Error embedding UNI: {e}")
            return np.zeros(DIM_IMG_UNI)


# =============================================================================
# VECTOR STORE QDRANT CLOUD — v5.1
# Cambios clave:
#   • COLLECTION_IMAGENES ahora tiene un 4° vector nombrado "texto_emb" (Gemini
#     3072d) construido desde ocr_text + texto_pagina. Esto permite buscar
#     imágenes por similitud semántica de texto, igual que los chunks.
#   • busqueda_vectorial_imagen devuelve texto_pagina en cada resultado.
#   • busqueda_hibrida agrega res_img_texto: búsqueda vectorial de texto sobre
#     COLLECTION_IMAGENES, con peso 0.30.
# =============================================================================

class QdrantVectorStore:

    def __init__(self, url: str, api_key: str):
        self.url     = url
        self.api_key = api_key
        self.client  = QdrantClient(url=url, api_key=api_key, timeout=60)

    async def connect(self):
        try:
            self.client.get_collections()
            print(f"✅ Qdrant Cloud conectado: {self.url}")
        except Exception as e:
            raise ConnectionError(f"No se pudo conectar a Qdrant Cloud: {e}")

    async def close(self):
        try:
            self.client.close()
        except Exception:
            pass

    async def crear_esquema(self):
        print("🏗️ Creando esquema Qdrant Cloud (v5.1 UNI + PLIP + texto_emb)...")

        # ── Colección de chunks (vector único de texto Gemini) ──────────
        try:
            self.client.get_collection(COLLECTION_CHUNKS)
            print(f"   ✅ Colección '{COLLECTION_CHUNKS}' ya existe")
        except Exception:
            self.client.create_collection(
                collection_name=COLLECTION_CHUNKS,
                vectors_config=VectorParams(size=DIM_TEXTO_GEMINI, distance=Distance.COSINE),
            )
            print(f"   ✅ Colección '{COLLECTION_CHUNKS}' creada")

        # ── Colección de imágenes (UNI + PLIP + texto_emb) ─────────────
        # NUEVO en v5.1: añadimos "texto_emb" para búsqueda texto→imagen
        try:
            col_info = self.client.get_collection(COLLECTION_IMAGENES)
            existing_vectors = set(col_info.config.params.vectors.keys()
                                   if hasattr(col_info.config.params.vectors, 'keys')
                                   else [])
            if "texto_emb" not in existing_vectors:
                # La colección existe pero le falta el vector texto_emb.
                # En Qdrant Cloud no se puede agregar un vector a una colección existente
                # sin recrearla. Avisamos al usuario.
                print(f"   ⚠️ Colección '{COLLECTION_IMAGENES}' existe pero le falta el vector "
                      f"'texto_emb'. Para habilitar búsqueda texto→imagen deberás recrearla "
                      f"(--reindex --force).")
            else:
                print(f"   ✅ Colección '{COLLECTION_IMAGENES}' ya existe (con texto_emb)")
        except Exception:
            self.client.create_collection(
                collection_name=COLLECTION_IMAGENES,
                vectors_config={
                    "uni":       VectorParams(size=DIM_IMG_UNI,      distance=Distance.COSINE),
                    "plip":      VectorParams(size=DIM_IMG_PLIP,     distance=Distance.COSINE),
                    # NUEVO: embedding textual de ocr_text + texto_pagina
                    "texto_emb": VectorParams(size=DIM_TEXTO_GEMINI, distance=Distance.COSINE),
                },
            )
            print(f"   ✅ Colección '{COLLECTION_IMAGENES}' creada (con texto_emb)")

        # ── Índices de Payload ──────────────────────────────────────────
        print("   🔍 Creando índices de payload...")
        for field in ["tejidos", "estructuras", "tinciones", "fuente"]:
            try:
                self.client.create_payload_index(
                    collection_name=COLLECTION_CHUNKS,
                    field_name=field,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
            except Exception:
                pass

        for field in ["fuente", "pagina_str"]:
            try:
                self.client.create_payload_index(
                    collection_name=COLLECTION_IMAGENES,
                    field_name=field,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
            except Exception:
                pass

        try:
            self.client.create_payload_index(
                collection_name=COLLECTION_CHUNKS,
                field_name="pagina",
                field_schema=models.PayloadSchemaType.INTEGER,
            )
        except Exception:
            pass

        print("✅ Esquema Qdrant listo (2 colecciones + índices de payload)")

    # ------------------------------------------------------------------
    # Escritura (indexación)
    # ------------------------------------------------------------------

    async def upsert_chunk(self, chunk_id: str, texto: str, fuente: str,
                            chunk_idx: int, embedding: List[float],
                            entidades: Dict[str, List[str]],
                            pagina: int = 0,
                            imagenes_pagina: Optional[List[str]] = None):
        point = PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id)),
            vector=embedding,
            payload={
                "chunk_id":         chunk_id,
                "texto":            texto,
                "fuente":           fuente,
                "chunk_idx":        chunk_idx,
                "pagina":           pagina,
                # Solo guardar rutas que existen en disco al momento de indexar
                "imagenes_pagina":  [p for p in (imagenes_pagina or []) if os.path.exists(p)],
                "tipo":             "texto",
                "tejidos":          entidades.get("tejidos", []),
                "estructuras":      entidades.get("estructuras", []),
                "tinciones":        entidades.get("tinciones", []),
            },
        )
        self.client.upsert(collection_name=COLLECTION_CHUNKS, points=[point])

    async def upsert_imagen(self, imagen_id: str, path: str, fuente: str,
                             pagina: int, ocr_text: str,
                             emb_uni: List[float], emb_plip: List[float],
                             # NUEVO v5.1: embedding textual precalculado
                             emb_texto: Optional[List[float]] = None,
                             texto_pagina: str = ""):
        """
        Indexa una imagen con tres vectores:
          • uni / plip  → similitud visual histológica
          • texto_emb   → similitud semántica del texto de la página (NUEVO)
        Si emb_texto es None se genera un vector cero (fallback sin Gemini).
        """
        vectors: Dict[str, List[float]] = {
            "uni":  emb_uni,
            "plip": emb_plip,
        }
        # Solo agregar texto_emb si la colección lo soporta (vector no-cero)
        if emb_texto and any(v != 0.0 for v in emb_texto):
            vectors["texto_emb"] = emb_texto
        else:
            # Vector cero como placeholder: Qdrant lo ignora en búsquedas coseno
            vectors["texto_emb"] = [0.0] * DIM_TEXTO_GEMINI

        point = PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_DNS, imagen_id)),
            vector=vectors,
            payload={
                "imagen_id":     imagen_id,
                "path":          path,
                "fuente":        fuente,
                "pagina":        pagina,
                "pagina_str":    str(pagina),   # índice keyword para filtros
                "ocr_text":      ocr_text,
                "texto_pagina":  texto_pagina[:3000],  # más largo que v5.0 (era 2000)
                "tipo":          "imagen",
            },
        )
        self.client.upsert(collection_name=COLLECTION_IMAGENES, points=[point])

    # ------------------------------------------------------------------
    # Lectura (búsqueda)
    # ------------------------------------------------------------------

    async def busqueda_vectorial_texto(self, embedding: List[float],
                                        top_k: int = 10) -> List[Dict]:
        try:
            results = self.client.query_points(
                collection_name=COLLECTION_CHUNKS,
                query=embedding,
                limit=top_k,
            ).points
            return [
                {
                    "id":               str(r.id),
                    "texto":            r.payload.get("texto", ""),
                    "fuente":           r.payload.get("fuente", ""),
                    "tipo":             "texto",
                    "pagina":           r.payload.get("pagina"),
                    "imagenes_pagina":  r.payload.get("imagenes_pagina", []),
                    "imagen_path":      None,
                    "texto_pagina":     "",          # chunks no tienen texto_pagina separado
                    "similitud":        r.score,
                }
                for r in results
            ]
        except Exception as e:
            print(f"⚠️ Error búsqueda vectorial texto: {e}")
            return []

    async def busqueda_vectorial_imagen(self, embedding: List[float],
                                         using: str, top_k: int = 10) -> List[Dict]:
        """
        Busca imágenes por similitud visual (uni/plip) o textual (texto_emb).
        NUEVO v5.1: siempre retorna texto_pagina en el resultado.
        """
        try:
            results = self.client.query_points(
                collection_name=COLLECTION_IMAGENES,
                query=embedding,
                using=using,
                limit=top_k,
            ).points
            return [
                {
                    "id":           str(r.id),
                    # Para el contexto del LLM: preferir texto_pagina sobre ocr_text
                    "texto":        r.payload.get("ocr_text", ""),
                    "fuente":       r.payload.get("fuente", ""),
                    "tipo":         "imagen",
                    "imagen_path":  r.payload.get("path"),
                    "pagina":       r.payload.get("pagina"),
                    # CLAVE: texto de la página del PDF asociado a esta imagen
                    "texto_pagina": r.payload.get("texto_pagina", ""),
                    "similitud":    r.score,
                }
                for r in results
            ]
        except Exception as e:
            print(f"⚠️ Error búsqueda vectorial imagen ({using}): {e}")
            return []

    async def busqueda_por_entidades(self, entidades: Dict[str, List[str]],
                                      top_k: int = 10) -> List[Dict]:
        tejidos     = entidades.get("tejidos", [])
        estructuras = entidades.get("estructuras", [])
        tinciones   = entidades.get("tinciones", [])
        if not any([tejidos, estructuras, tinciones]):
            return []

        conditions = []
        if tejidos:
            conditions.append(FieldCondition(key="tejidos", match=MatchAny(any=tejidos)))
        if estructuras:
            conditions.append(FieldCondition(key="estructuras", match=MatchAny(any=estructuras)))
        if tinciones:
            conditions.append(FieldCondition(key="tinciones", match=MatchAny(any=tinciones)))

        try:
            results, _ = self.client.scroll(
                collection_name=COLLECTION_CHUNKS,
                scroll_filter=Filter(should=conditions),
                limit=top_k,
            )
            return [
                {
                    "id":              str(r.id),
                    "texto":           r.payload.get("texto", ""),
                    "fuente":          r.payload.get("fuente", ""),
                    "tipo":            "texto",
                    "pagina":          r.payload.get("pagina"),
                    "imagenes_pagina": r.payload.get("imagenes_pagina", []),
                    "imagen_path":     None,
                    "texto_pagina":    "",
                    "similitud":       0.5,
                }
                for r in results
            ]
        except Exception as e:
            print(f"⚠️ Error búsqueda entidades: {e}")
            return []

    async def busqueda_por_pagina(self, fuente: str, pagina: int,
                                  top_k: int = 5) -> List[Dict]:
        try:
            results, _ = self.client.scroll(
                collection_name=COLLECTION_CHUNKS,
                scroll_filter=Filter(must=[
                    FieldCondition(key="fuente", match=MatchValue(value=fuente)),
                    FieldCondition(key="pagina", match=MatchValue(value=pagina)),
                ]),
                limit=top_k,
            )
            return [
                {
                    "id":              str(r.id),
                    "texto":           r.payload.get("texto", ""),
                    "fuente":          r.payload.get("fuente", ""),
                    "tipo":            "texto",
                    "pagina":          r.payload.get("pagina"),
                    "imagenes_pagina": r.payload.get("imagenes_pagina", []),
                    "imagen_path":     None,
                    "texto_pagina":    "",
                    "similitud":       0.45,
                }
                for r in results
            ]
        except Exception as e:
            print(f"⚠️ Error búsqueda por página ({fuente} p{pagina}): {e}")
            return []

    async def expandir_vecindad(self, resultados_top: List[Dict],
                                 top_k: int = 10) -> List[Dict]:
        if not resultados_top:
            return []

        fuentes = list(set(r.get("fuente") for r in resultados_top if r.get("fuente")))
        ids_originales = set(r.get("id") for r in resultados_top)
        vecinos: List[Dict] = []

        for fuente in fuentes[:3]:
            try:
                results, _ = self.client.scroll(
                    collection_name=COLLECTION_CHUNKS,
                    scroll_filter=Filter(must=[
                        FieldCondition(key="fuente", match=MatchValue(value=fuente))
                    ]),
                    limit=5,
                )
                for r in results:
                    rid = str(r.id)
                    if rid not in ids_originales:
                        vecinos.append({
                            "id": rid, "texto": r.payload.get("texto", ""),
                            "fuente": r.payload.get("fuente", ""), "tipo": "texto",
                            "pagina": r.payload.get("pagina"),
                            "imagenes_pagina": r.payload.get("imagenes_pagina", []),
                            "imagen_path": None, "texto_pagina": "", "similitud": 0.3,
                        })
            except Exception:
                pass

            try:
                results, _ = self.client.scroll(
                    collection_name=COLLECTION_IMAGENES,
                    scroll_filter=Filter(must=[
                        FieldCondition(key="fuente", match=MatchValue(value=fuente))
                    ]),
                    limit=3,
                )
                for r in results:
                    rid = str(r.id)
                    if rid not in ids_originales:
                        vecinos.append({
                            "id": rid, "texto": r.payload.get("ocr_text", ""),
                            "fuente": r.payload.get("fuente", ""), "tipo": "imagen",
                            "imagen_path": r.payload.get("path"),
                            "pagina": r.payload.get("pagina"),
                            # NUEVO: traer texto_pagina también en vecindad
                            "texto_pagina": r.payload.get("texto_pagina", ""),
                            "similitud": 0.3,
                        })
            except Exception:
                pass

        return vecinos[:top_k]

    # ------------------------------------------------------------------
    # Búsqueda híbrida — v5.1
    # Añade res_img_texto: busca imágenes por su texto_emb (Gemini).
    # ------------------------------------------------------------------

    async def busqueda_hibrida(self,
                                texto_embedding: Optional[List[float]],
                                imagen_embedding_uni: Optional[List[float]],
                                imagen_embedding_plip: Optional[List[float]],
                                entidades: Dict[str, List[str]],
                                top_k: int = 10) -> List[Dict]:
        res_texto     = []
        res_uni       = []
        res_plip      = []
        res_ent       = []
        res_vec       = []
        res_pag       = []
        res_img_texto = []   # NUEVO v5.1: texto→imagen

        # 1. Búsqueda Texto en chunks (Gemini)
        if texto_embedding:
            res_texto = await self.busqueda_vectorial_texto(texto_embedding, top_k)

        # 2. Búsqueda Imagen UNI
        if imagen_embedding_uni:
            res_uni = await self.busqueda_vectorial_imagen(imagen_embedding_uni, "uni", top_k)

        # 3. Búsqueda Imagen PLIP
        if imagen_embedding_plip:
            res_plip = await self.busqueda_vectorial_imagen(imagen_embedding_plip, "plip", top_k)

        # 4. NUEVO v5.1: Búsqueda de texto sobre COLLECTION_IMAGENES (texto_emb)
        #    Esto recupera imágenes cuyo texto de página es semánticamente similar
        #    a la consulta, incluso sin haber subido una imagen de consulta.
        if texto_embedding:
            try:
                res_img_texto = await self.busqueda_vectorial_imagen(
                    texto_embedding, "texto_emb", top_k
                )
                # Filtrar vector-cero (imágenes sin texto_emb real)
                res_img_texto = [r for r in res_img_texto if r.get("similitud", 0) > 0.05]
            except Exception as e:
                # La colección puede no tener el vector texto_emb (v5.0 legacy)
                print(f"   ⚠️ texto_emb no disponible en COLLECTION_IMAGENES: {e}")
                res_img_texto = []

        # 5. Entidades (filtro por payload)
        res_ent = await self.busqueda_por_entidades(entidades, top_k)

        # 6. Chunks co-ubicados con imágenes (misma página)
        paginas_encontradas = set()
        for r in res_uni + res_plip + res_img_texto:
            if r.get("tipo") == "imagen" and r.get("pagina") and r.get("fuente"):
                paginas_encontradas.add((r["fuente"], r["pagina"]))
        for fuente, pagina in list(paginas_encontradas)[:5]:
            chunks_pagina = await self.busqueda_por_pagina(fuente, pagina)
            res_pag.extend(chunks_pagina)
        if res_pag:
            print(f"   📄 Co-ubicados: {len(res_pag)} chunks de {len(paginas_encontradas)} páginas")

        # 7. Vecindad sobre los mejores
        todos = res_texto + res_uni + res_plip + res_img_texto
        if todos:
            res_vec = await self.expandir_vecindad(todos[:6])

        combined: Dict[str, Dict] = {}

        def agregar(resultados: List[Dict], peso: float):
            for r in resultados:
                key = r.get("id") or f"{r.get('fuente')}_{str(r.get('texto',''))[:40]}"
                if not r.get("texto") and not r.get("imagen_path"):
                    continue
                sim_ponderada = r.get("similitud", 0) * peso
                if key not in combined:
                    combined[key] = {**r, "similitud": sim_ponderada}
                else:
                    combined[key]["similitud"] += sim_ponderada
                    # Preservar texto_pagina si el nuevo resultado lo tiene y el anterior no
                    if r.get("texto_pagina") and not combined[key].get("texto_pagina"):
                        combined[key]["texto_pagina"] = r["texto_pagina"]
                    if r.get("imagenes_pagina") and not combined[key].get("imagenes_pagina"):
                        combined[key]["imagenes_pagina"] = r["imagenes_pagina"]

        agregar(res_texto,     0.40)
        agregar(res_uni,       0.20)
        agregar(res_plip,      0.20)
        agregar(res_img_texto, 0.30)  # NUEVO: texto→imagen
        agregar(res_ent,       0.35)
        agregar(res_pag,       0.30)
        agregar(res_vec,       0.10)

        final = sorted(combined.values(), key=lambda x: x["similitud"], reverse=True)

        print(f"   📊 Híbrida: Txt={len(res_texto)} | UNI={len(res_uni)} | "
              f"PLIP={len(res_plip)} | ImgTxt={len(res_img_texto)} | "
              f"Ent={len(res_ent)} | Pag={len(res_pag)} | Vec={len(res_vec)} "
              f"-> {len(final)}")

        return final[:15]

    # ------------------------------------------------------------------
    # Utilidades
    # ------------------------------------------------------------------

    async def contar_chunks(self) -> int:
        try:
            return self.client.get_collection(COLLECTION_CHUNKS).points_count
        except Exception:
            return 0

    async def contar_imagenes(self) -> int:
        try:
            return self.client.get_collection(COLLECTION_IMAGENES).points_count
        except Exception:
            return 0


# =============================================================================
# MEMORIA SEMÁNTICA CON PERSISTENCIA DE IMAGEN
# =============================================================================

class SemanticMemory:
    def __init__(self, llm, embeddings=None, uni=None, plip=None, max_entries: int = 10):
        self.llm            = llm
        self.embeddings     = embeddings
        self.uni            = uni
        self.plip           = plip
        self.conversations  = []
        self.max_entries    = max_entries
        self.summary        = ""
        self.direct_history = ""

        self.imagen_activa_path: Optional[str] = None
        self.imagen_turno_subida: int = 0
        self.turno_actual: int = 0

        self.collection_name = "memoria_histo"
        self.qdrant = QdrantClient(path="./qdrant_memoria")

        try:
            self.qdrant.get_collection(self.collection_name)
        except Exception:
            print(f"   🗂️ Creando colección Qdrant '{self.collection_name}' para memoria semántica...")
            self.qdrant.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "texto": VectorParams(size=DIM_TEXTO_GEMINI, distance=Distance.COSINE),
                    "uni":   VectorParams(size=DIM_IMG_UNI,      distance=Distance.COSINE),
                    "plip":  VectorParams(size=DIM_IMG_PLIP,     distance=Distance.COSINE),
                }
            )

    def set_imagen(self, path: Optional[str]):
        if path and os.path.exists(path):
            self.imagen_activa_path  = path
            self.imagen_turno_subida = self.turno_actual
            print(f"   📌 Imagen activa registrada (turno {self.turno_actual}): {path}")
        elif path is None:
            self.imagen_activa_path = None
            print("   🗑️  Imagen activa limpiada")

    def get_imagen_activa(self) -> Optional[str]:
        if self.imagen_activa_path and os.path.exists(self.imagen_activa_path):
            return self.imagen_activa_path
        return None

    def tiene_imagen_previa(self) -> bool:
        return self.get_imagen_activa() is not None

    def add_interaction(self, query: str, response: str):
        self.turno_actual += 1
        self.conversations.append({
            "query": query, "response": response,
            "turno": self.turno_actual, "imagen": self.imagen_activa_path
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

        if self.turno_actual % 5 == 0 and len(self.conversations) > 0 and self.embeddings:
            self._guardar_memoria_qdrant()

    def _guardar_memoria_qdrant(self):
        print("   🧠 Generando resumen profundo para guardar en memoria (Qdrant)...")
        try:
            resp = invoke_con_reintento_sync(self.llm, [
                SystemMessage(content="Genera un resumen detallado y técnico del siguiente historial de conversación sobre histología, destacando las entidades mencionadas y las conclusiones."),
                HumanMessage(content=self.direct_history)
            ])
            resumen = resp.content
            emb_texto = embed_query_con_reintento(self.embeddings, resumen)
            emb_uni  = [0.0] * DIM_IMG_UNI
            emb_plip = [0.0] * DIM_IMG_PLIP
            if self.imagen_activa_path and os.path.exists(self.imagen_activa_path):
                if self.uni:
                    emb_uni = self.uni.embed_image(self.imagen_activa_path).tolist()
                if self.plip:
                    emb_plip = self.plip.embed_image(self.imagen_activa_path).tolist()

            point = PointStruct(
                id=str(uuid.uuid4()),
                vector={"texto": emb_texto, "uni": emb_uni, "plip": emb_plip},
                payload={
                    "resumen":      resumen,
                    "turno_fin":    self.turno_actual,
                    "tiene_imagen": self.imagen_activa_path is not None,
                    "imagen_path":  self.imagen_activa_path
                }
            )
            self.qdrant.upsert(collection_name=self.collection_name, points=[point])
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
        if query and self.embeddings:
            try:
                emb_query = embed_query_con_reintento(self.embeddings, query)
                resultados = self.qdrant.query_points(
                    collection_name=self.collection_name,
                    query=emb_query, using="texto", limit=2
                ).points
                memorias = [r.payload['resumen'] for r in resultados if r.score > 0.4]
                if memorias:
                    ctx += "\n\n[Memorias históricas recuperadas:]\n" + "\n- ".join(memorias)
            except Exception as e:
                print(f"   ⚠️ Error recuperando memoria Qdrant: {e}")
        if self.imagen_activa_path:
            ctx += (f"\n\n[Imagen activa en el chat: "
                    f"{os.path.basename(self.imagen_activa_path)}]")
        return ctx


# =============================================================================
# CLASIFICADOR SEMÁNTICO
# =============================================================================

class ClasificadorSemantico:
    UMBRAL_SIMILITUD = 0.18
    UMBRAL_LLM       = 0.5

    def __init__(self, llm, embeddings, device: str, temario: List[str]):
        self.llm        = llm
        self.embeddings = embeddings
        self.device     = device
        self.temario    = temario
        self._anclas_emb: Optional[np.ndarray] = None

    def _embed_textos(self, textos: List[str]) -> np.ndarray:
        return np.array(embed_documents_con_reintento(self.embeddings, textos))

    def _get_anclas_emb(self) -> np.ndarray:
        if self._anclas_emb is None:
            print("   🔄 Precalculando embeddings de anclas semánticas (Gemini)...")
            self._anclas_emb = self._embed_textos(ANCLAS_SEMANTICAS_HISTOLOGIA)
        return self._anclas_emb

    def similitud_con_dominio(self, consulta: str) -> float:
        try:
            q_emb = np.array(embed_query_con_reintento(self.embeddings, consulta))
            a_emb = self._get_anclas_emb()
            sims  = (q_emb @ a_emb.T).flatten()
            return float(np.max(sims))
        except Exception as e:
            print(f"   ⚠️ Error similitud semántica: {e}")
            return 0.0

    async def clasificar(self, consulta: str, analisis_visual: Optional[str] = None,
                          imagen_activa: bool = False,
                          temario_muestra: Optional[List[str]] = None) -> Dict[str, Any]:
        sim = self.similitud_con_dominio(consulta)
        print(f"   📐 Similitud semántica con dominio histológico: {sim:.4f}")
        umbral_efectivo = self.UMBRAL_SIMILITUD * (0.6 if imagen_activa else 1.0)

        if sim >= umbral_efectivo:
            return {"valido": True, "tema_encontrado": None,
                    "motivo": f"Similitud {sim:.3f} ≥ umbral {umbral_efectivo:.3f}",
                    "similitud_dominio": sim, "metodo": "semantico_imagebind"}

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
            confianza = float(data.get("confianza", 0.5))
            valido    = bool(data.get("valido", True))
            if not valido and imagen_activa and confianza < 0.7:
                valido = True
                data["motivo"] += " [aceptado por imagen activa]"
            return {"valido": valido, "tema_encontrado": data.get("tema_encontrado"),
                    "motivo": data.get("motivo", ""), "similitud_dominio": sim,
                    "metodo": "llm" if sim < umbral_efectivo * 0.5 else "combinado"}
        except Exception as e:
            print(f"   ⚠️ Error clasificador LLM: {e}")
            return {"valido": imagen_activa or sim > 0.10, "tema_encontrado": None,
                    "motivo": f"Fallback: {e}", "similitud_dominio": sim, "metodo": "fallback"}


# =============================================================================
# EXTRACTOR DE IMÁGENES DE PDF
# =============================================================================

class ExtractorImagenesPDF:
    RENDER_DPI = 200
    MIN_IMAGE_SIZE = 150
    MAX_IMAGE_SIZE = (1280, 1280)

    def __init__(self, directorio_salida: str = DIRECTORIO_IMAGENES):
        self.directorio_salida = directorio_salida
        os.makedirs(directorio_salida, exist_ok=True)

    def extraer_de_pdf(self, pdf_path: str) -> List[Dict[str, str]]:
        imagenes = []
        nombre_pdf = os.path.splitext(os.path.basename(pdf_path))[0]

        try:
            doc = fitz.open(pdf_path)
            image_count = 0
            seen_xrefs  = set()
            seen_hashes = set()

            for page_num in range(len(doc)):
                page = doc[page_num]
                valid_images_this_page = []
                img_info_list = page.get_image_info(xrefs=True)
                page_xrefs    = list(dict.fromkeys(
                    [info["xref"] for info in img_info_list if info.get("xref")]
                ))

                if page_xrefs:
                    for xref in page_xrefs:
                        if xref in seen_xrefs:
                            continue
                        seen_xrefs.add(xref)
                        try:
                            base_image  = doc.extract_image(xref)
                            if not base_image:
                                continue
                            image_bytes = base_image["image"]
                            ext         = base_image["ext"]

                            import hashlib
                            img_hash = hashlib.md5(image_bytes).hexdigest()
                            if img_hash in seen_hashes:
                                continue
                            seen_hashes.add(img_hash)

                            image_count += 1
                            img_path = os.path.join(
                                self.directorio_salida,
                                f"{nombre_pdf}_p{page_num+1}_img{image_count}.{ext}"
                            )
                            with open(img_path, "wb") as f:
                                f.write(image_bytes)

                            img_pil       = Image.open(img_path)
                            width, height = img_pil.size
                            area          = width * height

                            if width >= self.MIN_IMAGE_SIZE and height >= self.MIN_IMAGE_SIZE:
                                if img_pil.mode != "RGB":
                                    img_pil = img_pil.convert("RGB")
                                img_path_rgb = os.path.join(
                                    self.directorio_salida,
                                    f"{nombre_pdf}_p{page_num+1}_img{image_count}.jpg"
                                )
                                img_pil.save(img_path_rgb, "JPEG")
                                if img_path != img_path_rgb:
                                    try:
                                        os.remove(img_path)
                                    except OSError:
                                        pass
                                img_path = img_path_rgb

                                try:
                                    texto_pagina = page.get_text("text").strip()
                                    texto_pagina = re.sub(r'\s+', ' ', texto_pagina)
                                    ocr_img  = pytesseract.image_to_string(img_pil).strip()[:300]
                                    ocr_text = f"[OCR: {ocr_img}] [Página: {texto_pagina}]"[:1500]
                                except Exception:
                                    ocr_text = ""

                                valid_images_this_page.append({
                                    "path": img_path,
                                    "fuente_pdf": os.path.basename(pdf_path),
                                    "pagina": page_num + 1,
                                    "indice": image_count,
                                    "ocr_text": ocr_text,
                                    "area": area,
                                })
                            else:
                                try:
                                    os.remove(img_path)
                                except OSError:
                                    pass
                        except Exception as e:
                            print(f"  ⚠️ Error xref {xref} pág {page_num+1}: {e}")
                else:
                    try:
                        page_imgs = convert_from_path(
                            pdf_path, first_page=page_num+1, last_page=page_num+1,
                            dpi=self.RENDER_DPI, fmt='jpeg', size=self.MAX_IMAGE_SIZE
                        )
                        if page_imgs:
                            image_count += 1
                            img_path = os.path.join(
                                self.directorio_salida,
                                f"{nombre_pdf}_p{page_num+1}_img{image_count}.jpg"
                            )
                            page_imgs[0].save(img_path, "JPEG")
                            width, height = page_imgs[0].size
                            try:
                                texto_pagina = page.get_text("text").strip()
                                texto_pagina = re.sub(r'\s+', ' ', texto_pagina)
                                ocr_img  = pytesseract.image_to_string(page_imgs[0]).strip()[:300]
                                ocr_text = f"[OCR: {ocr_img}] [Página: {texto_pagina}]"[:1500]
                            except Exception:
                                ocr_text = ""
                            valid_images_this_page.append({
                                "path": img_path,
                                "fuente_pdf": os.path.basename(pdf_path),
                                "pagina": page_num + 1,
                                "indice": image_count,
                                "ocr_text": ocr_text,
                                "area": width * height,
                            })
                    except Exception as e:
                        print(f"  ⚠️ Error renderizando pág {page_num+1}: {e}")

                if valid_images_this_page:
                    largest = max(valid_images_this_page, key=lambda x: x["area"])
                    for img_data in valid_images_this_page:
                        if img_data["path"] != largest["path"]:
                            try:
                                os.remove(img_data["path"])
                            except OSError:
                                pass
                    largest.pop("area", None)
                    imagenes.append(largest)

            doc.close()

            if imagenes:
                print(f"  📸 {len(imagenes)} figuras extraídas de {os.path.basename(pdf_path)} (PyMuPDF)")
                return imagenes

            print(f"  ⚠️ Sin figuras vía PyMuPDF, usando fallback de página completa...")

        except Exception as e:
            print(f"  ⚠️ Error PyMuPDF ({e}), usando fallback de página completa...")

        try:
            paginas = convert_from_path(pdf_path, dpi=self.RENDER_DPI)
        except Exception as e:
            print(f"  ❌ Error renderizando {pdf_path}: {e}")
            return []

        for num_pagina, pil_img in enumerate(paginas, start=1):
            nombre_archivo = f"{nombre_pdf}_pag{num_pagina}.jpg"
            ruta_completa  = os.path.join(self.directorio_salida, nombre_archivo)
            try:
                pil_img.save(ruta_completa, format="JPEG")
                try:
                    ocr_text = pytesseract.image_to_string(pil_img).strip()[:300]
                except Exception:
                    ocr_text = ""
                imagenes.append({
                    "path": ruta_completa, "fuente_pdf": os.path.basename(pdf_path),
                    "pagina": num_pagina, "indice": 1, "ocr_text": ocr_text
                })
            except Exception as e:
                print(f"  ⚠️ Error pág {num_pagina}: {e}")

        print(f"  📸 {len(imagenes)} páginas de {os.path.basename(pdf_path)} (fallback)")
        return imagenes

    def extraer_de_directorio(self, directorio: str) -> List[Dict[str, str]]:
        todas = []
        pdfs  = glob.glob(os.path.join(directorio, "*.pdf"))
        print(f"📂 Extrayendo imágenes de {len(pdfs)} PDFs...")
        for pdf_path in pdfs:
            todas.extend(self.extraer_de_pdf(pdf_path))
        print(f"✅ Total imágenes extraídas: {len(todas)}")
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
    imagen_es_nueva:             bool
    contexto_suficiente:         bool
    temario:                     List[str]
    tema_valido:                 bool
    tema_encontrado:             Optional[str]
    imagenes_recuperadas:        List[str]
    analisis_comparativo:        Optional[str]
    estructura_identificada:     Optional[str]
    similitud_semantica_dominio: float
    confianza_baja:              bool


# =============================================================================
# ASISTENTE PRINCIPAL v5.1
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

        self.uni        = None
        self.plip       = None
        self.embeddings = None
        self.embed_dim  = DIM_TEXTO_GEMINI

        self.qdrant_store: Optional[QdrantVectorStore] = None

        self.extractor_imagenes       = ExtractorImagenesPDF(DIRECTORIO_IMAGENES)
        self.extractor_temario        = None
        self.extractor_entidades      = None
        self.clasificador_semantico: Optional[ClasificadorSemantico] = None

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda":
            try:
                cap = torch.cuda.get_device_capability(0)
                if cap[0] < 7:
                    print(f"⚠️ GPU incompatible (sm_{cap[0]}{cap[1]}). Forzando CPU.")
                    self.device = "cpu"
            except Exception:
                pass
        print(f"✅ AsistenteHistologiaQdrant v5.1 inicializado en {self.device}")

    def _setup_apis(self):
        os.environ["GOOGLE_API_KEY"] = userdata.get("GOOGLE_API_KEY") or ""
        print("✅ APIs configuradas")

    async def inicializar_componentes(self):
        self._init_modelos()
        self.memoria = SemanticMemory(
            llm=self.llm, embeddings=self.embeddings, uni=self.uni, plip=self.plip
        )
        self.extractor_temario   = ExtractorTemario(llm=self.llm)
        self.extractor_entidades = ExtractorEntidades(llm=self.llm)
        self.clasificador_semantico = ClasificadorSemantico(
            llm=self.llm, embeddings=self.embeddings, device=self.device, temario=[]
        )

        self.qdrant_store = QdrantVectorStore(
            url     = userdata.get("QDRANT_URL") or os.getenv("QDRANT_URL"),
            api_key = userdata.get("QDRANT_KEY") or os.getenv("QDRANT_KEY"),
        )
        await self.qdrant_store.connect()
        await self.qdrant_store.crear_esquema()

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

        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=userdata.get("GOOGLE_API_KEY"),
            max_retries=1
        )
        print("✅ Embeddings Gemini inicializados")

        self.plip = PlipWrapper(self.device)
        self.plip.load()

        self.uni = UniWrapper(self.device)
        self.uni.load()

    def _init_imagebind(self):
        pass

    # ------------------------------------------------------------------
    # Grafo LangGraph
    # ------------------------------------------------------------------

    def _crear_grafo(self):
        g = StateGraph(AgentState)

        g.add_node("inicializar",          self._nodo_inicializar)
        g.add_node("procesar_imagen",      self._nodo_procesar_imagen)
        g.add_node("clasificar",           self._nodo_clasificar)
        g.add_node("generar_consulta",     self._nodo_generar_consulta)
        g.add_node("buscar_qdrant",        self._nodo_buscar_qdrant)
        g.add_node("filtrar_contexto",     self._nodo_filtrar_contexto)
        g.add_node("analisis_comparativo", self._nodo_analisis_comparativo)
        g.add_node("generar_respuesta",    self._nodo_generar_respuesta)
        g.add_node("finalizar",            self._nodo_finalizar)
        g.add_node("fuera_temario",        self._nodo_fuera_temario)

        g.add_edge(START,                  "inicializar")
        g.add_edge("inicializar",          "procesar_imagen")
        g.add_edge("procesar_imagen",      "clasificar")

        g.add_conditional_edges(
            "clasificar",
            self._route_por_temario,
            {"en_temario": "generar_consulta", "fuera_temario": "fuera_temario"}
        )
        g.add_edge("fuera_temario",        "finalizar")
        g.add_edge("generar_consulta",     "buscar_qdrant")
        g.add_edge("buscar_qdrant",        "filtrar_contexto")
        g.add_edge("filtrar_contexto",     "analisis_comparativo")
        g.add_edge("analisis_comparativo", "generar_respuesta")
        g.add_edge("generar_respuesta",    "finalizar")
        g.add_edge("finalizar",            END)

        self.graph = g

    def _route_por_temario(self, state: AgentState) -> str:
        return "en_temario" if state.get("tema_valido", True) else "fuera_temario"

    # ------------------------------------------------------------------
    # Nodos
    # ------------------------------------------------------------------

    async def _nodo_inicializar(self, state: AgentState) -> AgentState:
        print("📝 Inicializando flujo v5.1 (Qdrant)")
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
        state["trayectoria"]                 = [{"nodo": "Inicializar", "tiempo": 0}]
        return state

    async def _nodo_procesar_imagen(self, state: AgentState) -> AgentState:
        t0 = time.time()
        print("🖼️ Procesando imagen...")

        imagen_path_nuevo = state.get("imagen_path")
        imagen_es_nueva   = False

        if imagen_path_nuevo and os.path.exists(imagen_path_nuevo):
            imagen_path_activo = imagen_path_nuevo
            imagen_es_nueva    = True
            self.memoria.set_imagen(imagen_path_activo)
            print(f"   🆕 Nueva imagen: {imagen_path_activo}")
        elif self.memoria.tiene_imagen_previa():
            imagen_path_activo = self.memoria.get_imagen_activa()
            state["imagen_path"] = imagen_path_activo
            print(f"   ♻️  Reutilizando imagen del turno {self.memoria.imagen_turno_subida}: "
                  f"{os.path.basename(imagen_path_activo)}")
        else:
            imagen_path_activo = None

        if imagen_path_activo and os.path.exists(imagen_path_activo):
            try:
                emb_u = self.uni.embed_image(imagen_path_activo)
                emb_p = self.plip.embed_image(imagen_path_activo)

                state["imagen_embedding_uni"]  = emb_u.tolist()
                state["imagen_embedding_plip"] = emb_p.tolist()
                state["tiene_imagen"]          = True
                state["imagen_es_nueva"]       = imagen_es_nueva

                if imagen_es_nueva or not state.get("analisis_visual"):
                    state["analisis_visual"] = await self._describir_imagen_histologica(
                        imagen_path_activo
                    )
                    print(f"   🔬 Análisis visual generado ({len(state['analisis_visual'])} chars)")
                else:
                    print("   ♻️  Reutilizando análisis visual previo del contexto")

                print(f"✅ Imagen lista | nueva={imagen_es_nueva}")
            except Exception as e:
                print(f"❌ Error imagen: {e}")
                import traceback; traceback.print_exc()
                state["imagen_embedding_uni"]  = None
                state["imagen_embedding_plip"] = None
                state["analisis_visual"]       = None
                state["tiene_imagen"]          = False
        else:
            print("ℹ️ Sin imagen — modo texto")
            state["imagen_embedding"] = None
            state["analisis_visual"]  = None
            state["tiene_imagen"]     = False
            state["imagen_es_nueva"]  = False

        state["trayectoria"].append({
            "nodo": "ProcesarImagen", "tiene_imagen": state["tiene_imagen"],
            "imagen_es_nueva": imagen_es_nueva, "tiempo": round(time.time()-t0, 2)
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
                    "Describe esta imagen histológica.\n\n"
                    "PARTE 1 — DESCRIPCIÓN GENERAL: tipo tejido, coloración, aumento, estructuras.\n\n"
                    f"PARTE 2 — FEATURES DISCRIMINATORIAS:\n{features_lista}\n\n"
                    "PARTE 3 — DIAGNÓSTICO DIFERENCIAL: 3 estructuras más probables, "
                    "diferencias morfológicas, ¿confundible con cuerpo de albicans?"
                )},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}}
            ])
            resp = await invoke_con_reintento(self.llm, [msg])
            return resp.content
        except Exception as e:
            print(f"⚠️ Error describiendo imagen: {e}")
            return ""

    async def _nodo_clasificar(self, state: AgentState) -> AgentState:
        t0 = time.time()
        print("🔍 Clasificando consulta (semántico v5.1)...")

        system = (
            "Extrae términos técnicos histológicos de la consulta.\n"
            "Devuelve:\nTEJIDO: [...]\nESTRUCTURA: [...]\nCONCEPTO: [...]\n"
            "TINCIÓN: [...]\nTÉRMINOS_CLAVE: [...]"
        )
        partes = [f"CONSULTA:\n{state['consulta_texto']}"]
        analisis_visual = _safe(state.get("analisis_visual"))
        if analisis_visual:
            partes.append(f"ANÁLISIS VISUAL:\n{analisis_visual[:600]}")
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

        texto_para_entidades = (
            state["consulta_texto"] + " " + _safe(state.get("analisis_visual"))
        )
        state["entidades_consulta"] = await self.extractor_entidades.extraer_de_texto(
            texto_para_entidades
        )
        print(f"   🏷️ Entidades: {state['entidades_consulta']}")

        try:
            state["texto_embedding"] = self._embed_texto_gemini(state["consulta_texto"])
        except Exception as e:
            print(f"⚠️ Error embedding texto: {e}")
            state["texto_embedding"] = None

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
            "nodo": "Clasificar", "tema_valido": state["tema_valido"],
            "tema_encontrado": state["tema_encontrado"],
            "entidades": state["entidades_consulta"],
            "similitud_dominio": state["similitud_semantica_dominio"],
            "metodo_clasificacion": verificacion.get("metodo"),
            "tiempo": round(time.time()-t0, 2)
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

    async def _nodo_buscar_qdrant(self, state: AgentState) -> AgentState:
        t0 = time.time()
        print("📚 Búsqueda híbrida Qdrant v5.1...")

        resultados = await self.qdrant_store.busqueda_hibrida(
            texto_embedding        = state.get("texto_embedding"),
            imagen_embedding_uni   = state.get("imagen_embedding_uni"),
            imagen_embedding_plip  = state.get("imagen_embedding_plip"),
            entidades              = state.get("entidades_consulta", {}),
            top_k                  = 10
        )

        state["resultados_busqueda"] = resultados
        print(f"✅ {len(resultados)} resultados")
        state["trayectoria"].append({
            "nodo": "BuscarQdrant", "hits": len(resultados),
            "tiempo": round(time.time()-t0, 2)
        })
        return state

    async def _nodo_filtrar_contexto(self, state: AgentState) -> AgentState:
        """
        v5.1: al construir bloques de contexto para resultados de tipo imagen,
        incluye texto_pagina (el texto completo del PDF en esa página).
        """
        t0     = time.time()
        umbral = self.SIMILARITY_THRESHOLD
        validos = [r for r in state["resultados_busqueda"] if r.get("similitud", 0) >= umbral]

        state["resultados_validos"]  = validos
        state["contexto_suficiente"] = len(validos) > 0

        vistas: set = set()
        imagenes_unicas: List[str] = []
        for r in validos:
            img_path = r.get("imagen_path")
            if img_path and os.path.exists(img_path) and img_path not in vistas:
                vistas.add(img_path)
                imagenes_unicas.append(img_path)
            for img_ref in r.get("imagenes_pagina", []):
                if img_ref and os.path.exists(img_ref) and img_ref not in vistas:
                    vistas.add(img_ref)
                    imagenes_unicas.append(img_ref)
        state["imagenes_recuperadas"] = imagenes_unicas

        if validos:
            validos_sorted = sorted(validos, key=lambda x: x.get("similitud", 0), reverse=True)
            bloques = []
            for i, r in enumerate(validos_sorted, 1):
                pag  = r.get("pagina")
                tipo = r.get("tipo", "?")
                enc  = (f"[Sección {i} | Fuente: {r.get('fuente','N/A')} | "
                        f"Tipo: {tipo} | Sim: {r.get('similitud',0):.3f}")
                if pag:
                    enc += f" | Pág: {pag}"
                if r.get("imagen_path"):
                    enc += f" | Imagen: {os.path.basename(r['imagen_path'])}"
                imgs_pag = r.get("imagenes_pagina", [])
                if imgs_pag:
                    nombres = [os.path.basename(p) for p in imgs_pag]
                    enc += f" | Imgs misma pág: {', '.join(nombres)}"
                enc += "]"

                # ── Construir el cuerpo del bloque ────────────────────
                if tipo == "imagen":
                    # NUEVO v5.1: para imágenes, incluir AMBOS: ocr_text y
                    # texto_pagina (el texto del PDF en esa página).
                    texto_ocr  = _safe(r.get("texto", ""))
                    texto_pag  = _safe(r.get("texto_pagina", ""))
                    partes_txt = []
                    if texto_ocr:
                        partes_txt.append(f"[OCR imagen]: {texto_ocr[:400]}")
                    if texto_pag:
                        partes_txt.append(f"[Texto de la página del PDF]:\n{texto_pag[:1500]}")
                    texto_ctx = "\n".join(partes_txt) if partes_txt else "(sin texto asociado)"
                else:
                    texto_ctx = _safe(r.get("texto", ""))[:1500]
                    # Para chunks de texto, si hay imágenes en la misma página,
                    # añadir referencia al texto_pagina si está disponible
                    if r.get("texto_pagina"):
                        texto_ctx += f"\n[Contexto adicional de página]: {r['texto_pagina'][:400]}"

                bloques.append(f"{enc}\n{texto_ctx}")

            state["contexto_documentos"] = "\n\n".join(bloques)
            print(f"✅ {len(validos)} válidos | {len(imagenes_unicas)} imágenes")
        else:
            state["contexto_documentos"] = ""
            print(f"⚠️ Ningún resultado supera umbral {umbral}")

        state["trayectoria"].append({
            "nodo": "FiltrarContexto", "hits_validos": len(validos),
            "imgs": len(state["imagenes_recuperadas"]), "tiempo": round(time.time()-t0, 2)
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
            content_parts.append({"type": "text",
                                   "text": f"\nAnálisis previo:\n{analisis_previo[:600]}\n"})

        # NUEVO v5.1: incluir el texto_pagina de cada imagen de referencia
        # para que el LLM tenga el contexto textual del manual al compararlas.
        for i, ref_path in enumerate(imagenes_ref, 1):
            content_parts.append({"type": "text", "text": (
                f"\n=== REFERENCIA #{i} ({os.path.basename(ref_path)}) ==="
            )})
            # Buscar texto_pagina de esta referencia en resultados válidos
            texto_pag_ref = ""
            for r in state.get("resultados_validos", []):
                if r.get("imagen_path") == ref_path and r.get("texto_pagina"):
                    texto_pag_ref = r["texto_pagina"]
                    break
            if texto_pag_ref:
                content_parts.append({"type": "text",
                                       "text": f"[Texto del manual en esta página]:\n{texto_pag_ref[:800]}\n"})
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
            "TU ROL ES SER UN JUEZ ESCÉPTICO. Tu objetivo principal es encontrar las DIFERENCIAS "
            "que demuestren que NO son el mismo tejido.\n\n"
            "1. TABLA COMPARATIVA (Markdown): | Feature | Consulta | Ref#1 | Ref#2 |\n"
            "2. VEREDICTO DE IDENTIDAD: Si la imagen de consulta parece ser un órgano distinto "
            "al de las referencias, DEBES declarar EXPLÍCITAMENTE que son TEJIDOS DIFERENTES.\n"
            "3. CONCLUSIÓN FINAL: ¿Son la misma estructura biológica? (SÍ/NO). "
            "Si es NO, indica qué crees que es realmente la imagen de la consulta."
        )})

        try:
            resp = await invoke_con_reintento(self.llm, [HumanMessage(content=content_parts)])
            state["analisis_comparativo"]    = resp.content
            state["estructura_identificada"] = await self._extraer_estructura(resp.content)
            print(f"✅ Análisis comparativo: {len(resp.content)} chars")
            print(f"   → Estructura: {state['estructura_identificada']}")
        except Exception as e:
            print(f"❌ Error análisis comparativo: {e}")
            state["analisis_comparativo"]    = None
            state["estructura_identificada"] = None

        state["trayectoria"].append({
            "nodo": "AnalisisComparativo", "refs": len(imagenes_ref),
            "estructura": state.get("estructura_identificada"),
            "tiempo": round(time.time()-t0, 2)
        })
        return state

    async def _extraer_estructura(self, analisis: str) -> Optional[str]:
        try:
            resp = await invoke_con_reintento(self.llm, [
                SystemMessage(content="Extrae el nombre de la estructura histológica más probable. Solo el nombre."),
                HumanMessage(content=analisis[-1000:])
            ])
            return resp.content.strip()
        except Exception:
            return None

    async def _nodo_generar_respuesta(self, state: AgentState) -> AgentState:
        """
        v5.1: el contexto_documentos ya incluye texto_pagina para imágenes
        (armado en _nodo_filtrar_contexto). Aquí simplemente lo pasamos al LLM.
        """
        t0 = time.time()
        print("💭 Generando respuesta v5.1...")

        aviso_db = ""
        if not state["contexto_suficiente"]:
            if state.get("tiene_imagen") and state.get("imagen_path"):
                print("   ⚠️ Sin contexto RAG pero hay imagen — permitiendo chat con imagen")
                aviso_db = (
                    "\n\nAVISO PARA EL ASISTENTE: No se encontraron resultados en la base de datos "
                    "para esta consulta. DEBES informar explícitamente al usuario que no encontraste "
                    "información referenciada en el manual, pero responde a su pregunta basándote "
                    "en la imagen subida y el historial de chat."
                )
                state["contexto_suficiente"] = True
            else:
                state["respuesta_final"] = (
                    f"❌ Sin información suficiente (umbral {self.SIMILARITY_THRESHOLD:.0%}).\n"
                    f"Consulta: {state['consulta_busqueda_texto']}"
                )
                state["trayectoria"].append({
                    "nodo": "GenerarRespuesta", "contexto_suficiente": False,
                    "tiempo": round(time.time()-t0, 2)
                })
                return state

        tiene_comparativo = bool(_safe(state.get("analisis_comparativo")))
        nota_comp = (
            "\n\nIMPORTANTE: El análisis comparativo tiene PRIORIDAD en el diagnóstico diferencial."
            if tiene_comparativo else ""
        )

        system_prompt = (
            "Eres un asistente de histología. Responde SOLO con el contenido del manual o la imagen visible en el chat.\n\n"
            "REGLAS:\n"
            "1. Solo información de SECCIONES DEL MANUAL e IMÁGENES DE REFERENCIA guardadas en la base de datos, o la propia imagen que subió el usuario.\n"
            "2. Cita: [Manual: archivo] | [Imagen: archivo]\n"
            "3. No diagnósticos clínicos salvo que estén explícitos.\n\n"
            "ESTRUCTURA:\n"
            "1. Análisis de la consulta basado en la imagen del usuario (si la hay)\n"
            "2. VALIDACIÓN: Revisa el 'ANÁLISIS COMPARATIVO'. Si concluye que la imagen del usuario NO ES LA MISMA ESTRUCTURA que las imágenes de referencia del manual, informa al usuario y DETENTE AHÍ.\n"
            "3. Características histológicas según la base de datos (SOLO si las estructuras coinciden)\n"
            "4. Conclusión y confianza"
            f"{nota_comp}{aviso_db}"
        )

        analisis_comp_str   = _safe(state.get("analisis_comparativo"))
        estructura_str      = _safe(state.get("estructura_identificada"))
        analisis_visual_str = _safe(state.get("analisis_visual"), "No disponible")
        contexto_mem_str    = _safe(state.get("contexto_memoria"))
        terminos_str        = _safe(state.get("terminos_busqueda"))
        tema_str            = _safe(state.get("tema_encontrado"), "N/A")
        entidades_str       = json.dumps(state.get("entidades_consulta", {}), ensure_ascii=False)

        seccion_comp = (f"\n\n**ANÁLISIS COMPARATIVO:**\n{analisis_comp_str[:2000]}"
                        if analisis_comp_str else "")
        seccion_est  = (f"\n\n**ESTRUCTURA IDENTIFICADA:** {estructura_str}"
                        if estructura_str else "")

        content_parts = [{"type": "text", "text": (
            f"**CONSULTA:** {state['consulta_texto']}\n\n"
            f"**HISTORIAL:** {contexto_mem_str[:300]}\n\n"
            f"**TÉRMINOS:** {terminos_str[:300]}\n\n"
            f"**ENTIDADES (grafo):** {entidades_str}\n\n"
            f"**TEMA:** {tema_str}\n\n"
            f"**ANÁLISIS VISUAL USUARIO:**\n{analisis_visual_str[:800]}\n\n"
            f"**SECCIONES DEL MANUAL (incluyen texto de páginas con imágenes):**\n"
            f"{state['contexto_documentos']}"
            f"{seccion_comp}{seccion_est}\n\n"
            "Responde EXCLUSIVAMENTE con el contenido del manual e imágenes de referencia."
        )}]

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
                state["respuesta_final"] = (
                    resp.content if "Hola, ¿en qué te puedo ayudar sobre histología?" in resp.content
                    else f"{warning}\n\n{resp.content}"
                )
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
            }, f, indent=4, ensure_ascii=False)

        print(f"✅ Flujo v5.1 completado en {total}s")
        if state.get("estructura_identificada"):
            print(f"   → Estructura: {state['estructura_identificada']}")
        return state

    # ------------------------------------------------------------------
    # Embeddings (Gemini Text)
    # ------------------------------------------------------------------

    def _embed_texto_gemini(self, texto: str) -> List[float]:
        return embed_query_con_reintento(self.embeddings, texto)

    # ------------------------------------------------------------------
    # Indexación en Qdrant — v5.1
    # NUEVO: upsert_imagen recibe emb_texto (Gemini sobre ocr+texto_pagina)
    # ------------------------------------------------------------------

    def _leer_pdf(self, path: str) -> str:
        try:
            doc   = fitz.open(path)
            texto = "".join(page.get_text() for page in doc)
            doc.close()
            return texto
        except Exception as e:
            print(f"⚠️ Error leyendo {path}: {e}")
            return ""

    def _leer_pdf_por_pagina(self, path: str) -> List[Tuple[int, str]]:
        try:
            doc     = fitz.open(path)
            paginas = [(i + 1, page.get_text()) for i, page in enumerate(doc)]
            doc.close()
            return paginas
        except Exception as e:
            print(f"⚠️ Error leyendo {path}: {e}")
            return []

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
        if self.clasificador_semantico:
            self.clasificador_semantico.temario = self.extractor_temario.temas
            print(f"   🔄 Clasificador semántico actualizado con "
                  f"{len(self.extractor_temario.temas)} temas")

    async def indexar_en_qdrant(self, directorio_pdfs: str = DIRECTORIO_PDFS,
                                 imagen_files_extra: Optional[List[str]] = None,
                                 forzar: bool = False):
        # ── Verificar si ya hay datos ──────────────────────────────────
        if not forzar:
            try:
                n_chunks = await self.qdrant_store.contar_chunks()
                n_imgs   = await self.qdrant_store.contar_imagenes()
                if n_chunks > 0 and n_imgs > 0:
                    print(f"✅ Base de datos Qdrant ya poblada "
                          f"({n_chunks} chunks, {n_imgs} imágenes). Saltando indexación.")
                    print("   (Usá --reindex --force para forzar re-indexación)")
                    return
            except Exception as e:
                print(f"⚠️ No se pudo verificar estado de la BD: {e}")

        # ── PASO 1: Extraer imágenes PRIMERO ──────────────────────────
        print("📸 Extrayendo imágenes de PDFs...")
        imagenes_pdf = self.extractor_imagenes.extraer_de_directorio(directorio_pdfs)

        from collections import defaultdict
        mapa_imagenes_pagina: Dict[tuple, List[str]] = defaultdict(list)
        for img_info in imagenes_pdf:
            key = (img_info["fuente_pdf"], img_info["pagina"])
            if os.path.exists(img_info["path"]):   # solo rutas válidas
                mapa_imagenes_pagina[key].append(img_info["path"])
        print(f"   📋 Mapa construido: {len(mapa_imagenes_pagina)} páginas con imágenes")

        mapa_texto_pagina: Dict[tuple, str] = {}

        # ── PASO 2: Indexar chunks de texto ───────────────────────────
        print("📄 Indexando chunks de texto en Qdrant (por página)...")
        for pdf_path in glob.glob(os.path.join(directorio_pdfs, "*.pdf")):
            fuente  = os.path.basename(pdf_path)
            paginas = self._leer_pdf_por_pagina(pdf_path)
            chunk_global = 0
            for pagina_num, texto_pagina in paginas:
                mapa_texto_pagina[(fuente, pagina_num)] = texto_pagina
                imgs_esta_pagina = mapa_imagenes_pagina.get((fuente, pagina_num), [])
                chunks_pagina    = self._chunks(texto_pagina)
                for i, chunk in enumerate(chunks_pagina):
                    try:
                        emb       = self._embed_texto_gemini(chunk)
                        chunk_id  = f"chunk_{fuente}_{pagina_num}_{i}"
                        entidades = self.extractor_entidades.extraer_de_texto_sync(chunk)
                        await self.qdrant_store.upsert_chunk(
                            chunk_id=chunk_id, texto=chunk, fuente=fuente,
                            chunk_idx=chunk_global, pagina=pagina_num,
                            embedding=emb, entidades=entidades,
                            imagenes_pagina=imgs_esta_pagina
                        )
                        chunk_global += 1
                    except Exception as e:
                        print(f"  ⚠️ Chunk p{pagina_num}/{i}: {e}")
            print(f"  {fuente}: {chunk_global} chunks ({len(paginas)} páginas)")

        # ── PASO 3: Indexar imágenes con embedding textual ────────────
        # NUEVO v5.1: calcular emb_texto = Gemini(ocr_text + texto_pagina)
        print("📸 Indexando imágenes en Qdrant (con texto_emb Gemini)...")
        for img_info in imagenes_pdf:
            img_path = img_info["path"]
            if not os.path.exists(img_path):
                continue
            try:
                emb_u = self.uni.embed_image(img_path)
                emb_p = self.plip.embed_image(img_path)

                img_id    = f"img_{img_info['fuente_pdf']}_{img_info['pagina']}"
                texto_pag = mapa_texto_pagina.get(
                    (img_info["fuente_pdf"], img_info["pagina"]), ""
                )

                # Construir texto combinado para el embedding textual de la imagen
                ocr_text      = img_info.get("ocr_text", "")
                texto_combined = f"{ocr_text} {texto_pag}".strip()

                # Calcular embedding Gemini del texto combinado
                emb_t: Optional[List[float]] = None
                if texto_combined:
                    try:
                        emb_t = self._embed_texto_gemini(texto_combined[:2000])
                    except Exception as e:
                        print(f"  ⚠️ emb_texto para {img_id}: {e}")

                await self.qdrant_store.upsert_imagen(
                    imagen_id=img_id, path=img_path,
                    fuente=img_info["fuente_pdf"], pagina=img_info["pagina"],
                    ocr_text=ocr_text,
                    emb_uni=emb_u.tolist(),
                    emb_plip=emb_p.tolist(),
                    emb_texto=emb_t,           # NUEVO v5.1
                    texto_pagina=texto_pag
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
                emb_u  = self.uni.embed_image(img_path)
                emb_p  = self.plip.embed_image(img_path)
                emb_t: Optional[List[float]] = None
                if ocr:
                    try:
                        emb_t = self._embed_texto_gemini(ocr[:2000])
                    except Exception:
                        pass
                await self.qdrant_store.upsert_imagen(
                    imagen_id=img_id, path=img_path, fuente=os.path.basename(img_path),
                    pagina=0, ocr_text=ocr[:300],
                    emb_uni=emb_u.tolist(), emb_plip=emb_p.tolist(),
                    emb_texto=emb_t, texto_pagina=""
                )
            except Exception as e:
                print(f"  ❌ Imagen extra {img_path}: {e}")

        print("✅ Indexación Qdrant v5.1 completada")

    # ------------------------------------------------------------------
    # Punto de entrada público
    # ------------------------------------------------------------------

    async def consultar(self, consulta_texto: str,
                         imagen_path: Optional[str] = None,
                         user_id: str = "default_user") -> str:
        imagen_activa       = imagen_path or self.memoria.get_imagen_activa()
        tiene_imagen_activa = self.memoria.tiene_imagen_previa() or bool(imagen_path)

        print(f"\n{'='*70}")
        print(f"🔬 RAG Histología Qdrant v5.1 | umbral={self.SIMILARITY_THRESHOLD}")
        print(f"   Texto:         {consulta_texto}")
        print(f"   Imagen turno:  {imagen_path or 'ninguna'}")
        print(f"   Imagen activa: {imagen_activa or 'ninguna'}")
        print(f"{'='*70}")

        initial_state = AgentState(
            messages=[], consulta_texto=consulta_texto,
            imagen_path=imagen_activa,
            imagen_embedding_uni=None, imagen_embedding_plip=None, texto_embedding=None,
            contexto_memoria="", contenido_base=self.contenido_base, terminos_busqueda="",
            entidades_consulta={"tejidos": [], "estructuras": [], "tinciones": []},
            consulta_busqueda_texto="", consulta_busqueda_visual="",
            resultados_busqueda=[], resultados_validos=[], contexto_documentos="",
            respuesta_final="", trayectoria=[], user_id=user_id, tiempo_inicio=time.time(),
            analisis_visual=None, tiene_imagen=False, imagen_es_nueva=False,
            contexto_suficiente=False, temario=self.extractor_temario.temas,
            tema_valido=True, tema_encontrado=None, imagenes_recuperadas=[],
            analisis_comparativo=None, estructura_identificada=None,
            similitud_semantica_dominio=0.0, confianza_baja=False,
        )

        config = {
            "configurable": {"thread_id": user_id},
            "run_name":     f"consulta-qdrant-v5.1-{user_id}",
            "tags":         ["rag", "histologia", "qdrant", "v5.1"],
            "metadata": {
                "tiene_imagen_nueva":  imagen_path is not None,
                "tiene_imagen_activa": tiene_imagen_activa,
                "consulta":            consulta_texto[:100],
                "version":             "5.1"
            }
        }
        try:
            final     = await self.compiled_graph.ainvoke(initial_state, config=config)
            respuesta = final["respuesta_final"]
        except Exception as e:
            import traceback; traceback.print_exc()
            respuesta = f"Error: {e}"

        print(f"\n{'='*70}\n📖 RESPUESTA:\n{'='*70}")
        print(respuesta)
        print("="*70)
        return respuesta

    async def cerrar(self):
        if self.qdrant_store:
            await self.qdrant_store.close()


# =============================================================================
# MODO INTERACTIVO
# =============================================================================

async def modo_interactivo(reindex: bool = False, force: bool = False):
    asistente = AsistenteHistologiaQdrant()
    await asistente.inicializar_componentes()

    print("\n🔄 Leyendo el manual...")
    asistente.procesar_contenido_base(DIRECTORIO_PDFS)

    print("\n📋 Extrayendo temario...")
    await asistente.extraer_y_preparar_temario()
    print(f"   → {len(asistente.extractor_temario.temas)} temas")

    print("\n💾 Indexando en Qdrant...")
    if reindex:
        await asistente.indexar_en_qdrant(DIRECTORIO_PDFS, forzar=force)
    else:
        print("   (Saltando indexación — usar --reindex para forzar)")

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║   RAG Histología Qdrant v5.1 — Chat Interactivo             ║
╠══════════════════════════════════════════════════════════════╣
║  • Escribí tu pregunta y presioná Enter                     ║
║  • Para subir una imagen: escribí el PATH cuando se pida    ║
║  • La imagen se recuerda entre turnos — no es obligatoria   ║
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
    parser.add_argument("--reindex", action="store_true",
                        help="Indexar en Qdrant (salta si ya hay datos)")
    parser.add_argument("--force", action="store_true",
                        help="Forzar re-indexación aunque haya datos")
    args = parser.parse_args()

    import logging
    logging.basicConfig(level=logging.INFO)
    print(f"✅ GPU: {torch.cuda.get_device_name()}" if torch.cuda.is_available() else "⚠️ CPU mode")
    os.makedirs("logs", exist_ok=True)
    os.makedirs(DIRECTORIO_IMAGENES, exist_ok=True)
    asyncio.run(modo_interactivo(reindex=args.reindex, force=args.force))
