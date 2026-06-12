#!/bin/bash

# Iniciar o arrancar Qdrant local en Docker
echo "🔄 Verificando contenedor local de Qdrant..."
if [ "$(docker ps -a -q -f name=qdrant-local-histo)" ]; then
    if [ ! "$(docker ps -q -f name=qdrant-local-histo)" ]; then
        echo "🚀 Iniciando contenedor Qdrant existente..."
        docker start qdrant-local-histo
    else
        echo "✅ El contenedor Qdrant ya está en ejecución."
    fi
else
    echo "📦 Creando e iniciando nuevo contenedor Qdrant..."
    docker run -d \
      --name qdrant-local-histo \
      -p 6333:6333 \
      -p 6334:6334 \
      -v "$(pwd)/qdrant_storage:/qdrant/storage" \
      qdrant/qdrant
fi

# Esperar brevemente a que el servicio Qdrant esté listo
echo "⏳ Esperando a que Qdrant responda..."
until curl -s http://localhost:6333/healthz > /dev/null; do
  sleep 1
done
echo "✅ Qdrant local está listo."

# Iniciar el servidor FastAPI
echo "🌐 Iniciando servidor FastAPI..."
uv run uvicorn server:app --reload --host 0.0.0.0 --port 10007
