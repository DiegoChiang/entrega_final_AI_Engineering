# Arquitectura técnica

## Límites del sistema

FastAPI acepta solicitudes validadas y delega su procesamiento a una cola `asyncio.Queue`. Los
endpoints permanecen libres mientras los workers ejecutan el grafo. Redis cumple dos funciones
separadas: estado consultable de cada trabajo y checkpoints internos de LangGraph.

## Recuperación

El corpus se convierte en `KnowledgeDocument` y `KnowledgeChunk`. Cada fragmento recibe un hash
estable construido a partir del documento, página, posición y contenido. La huella completa del
corpus controla la reutilización del índice local.

`HybridRetriever` consulta en paralelo lógico dos señales:

- similitud coseno de embeddings;
- relevancia BM25 sobre tokens normalizados.

Las puntuaciones BM25 se normalizan y se combinan con pesos 0.6/0.4. La salida se deduplica por
`document_id` y se valida como `RetrievedDocument`.

## Coordinación de agentes

El Supervisor selecciona un único nodo por iteración. Las guardas corrigen rutas que no cumplen los
prerrequisitos:

1. Sin fuentes: investigador.
2. Con fuentes y sin cálculo: analista.
3. Con cálculo sin validar: validador.
4. Resultado crítico pendiente: aprobación.
5. Resultado validado: sintetizador.

El investigador solo dispone de `search_incident_knowledge`. El analista solo dispone de
`calculate_incident_priority`. Esta separación reduce el acceso innecesario a herramientas y evita
que el modelo sustituya cálculos deterministas.

## Human-in-the-loop

Una prioridad P1 o riesgo mayor o igual a 75 dirige el grafo al nodo `approval`. `interrupt()`
produce un payload serializable y el worker publica `WAITING_APPROVAL`. El endpoint de decisión
vuelve a encolar el trabajo y ejecuta `Command(resume=...)` con el mismo `thread_id`.

## Contratos

Pydantic valida:

- configuración y secretos;
- cuerpos y respuestas de FastAPI;
- estados, resultados y referencias;
- entradas de herramientas;
- documentos, fragmentos y resultados RAG;
- decisiones estructuradas de los modelos.

Los modelos expuestos usan `extra="forbid"` para rechazar campos inesperados.

## Asincronía

Las llamadas al chat, embeddings, Redis, LangGraph y HTTP usan APIs asíncronas. Operaciones de disco,
BM25 y SDKs sin interfaz asíncrona se aíslan mediante `asyncio.to_thread`. Los workers se crean con
`asyncio.create_task` y se cierran de forma coordinada durante el lifespan.

## Observabilidad

Phoenix recibe OTLP/HTTP y la autoinstrumentación de LangChain. La jerarquía permite seguir API,
worker, grafo, Supervisor, RAG, herramientas, modelos, validación, interrupción y reanudación. Los
errores conservan su tipo en Redis y también aparecen en los spans.

## Modos de despliegue

El modo local persiste vectores en un volumen de Docker. El modo Pinecone conserva el corpus BM25
local y utiliza Pinecone Serverless para la señal semántica. Nombres, namespaces, región, dimensión,
modelos y pesos se reciben por variables de entorno.
