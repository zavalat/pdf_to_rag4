import os
# === FIX PARA AZURE Y OPENAI ===
# Azure App Service mete variables de proxy automáticamente
# y rompen el cliente oficial de OpenAI al iniciar.
for var in [
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
    "http_proxy", "https_proxy", "all_proxy"
]:
    if var in os.environ:
        print(f"[AZURE FIX] Eliminando variable proxy: {var}")
        os.environ.pop(var, None)

# Asegura que OpenAI no use ningún proxy
os.environ["NO_PROXY"] = "*"

print("### FIX DE PROXIES APLICADO EN AZURE ###")


from flask import Flask, request, render_template, jsonify
import os, uuid, fitz, threading, json
from qdrant_client import QdrantClient, models
from openai import OpenAI

# ========================
# ⚙️ CARGAR CONFIGURACIÓN
# ========================


with open("config.json", "r", encoding="utf-8") as f:
    cfg = json.load(f)

QDRANT_URL = cfg["QDRANT_URL"]
QDRANT_API_KEY = cfg["QDRANT_API_KEY"]
OPENAI_API_KEY = cfg["OPENAI_API_KEY"]

# ========================
# ⚙️ CONFIGURACIÓN INICIAL
# ========================

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ========================
# 🤖 CLIENTE OPENAI
# ========================

client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url="https://api.openai.com/v1"
)



# Tamaño del vector de OpenAI text-embedding-3-small
VECTOR_SIZE = 1536

# Función para obtener embeddings desde OpenAI
def embed(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding


# ===============================
# 🌐 QDRANT CLOUD (Base vectorial)
# ===============================

qdrant = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

# Crear colección si no existe
collections = qdrant.get_collections().collections
collection_names = [c.name.lower() for c in collections]

if "pdf_knowledge" not in collection_names:
    qdrant.create_collection(
        collection_name="pdf_knowledge",
        vectors_config=models.VectorParams(
            size=VECTOR_SIZE,
            distance=models.Distance.COSINE
        )
    )
    print("📁 Colección creada: pdf_knowledge")
else:
    print("📁 Colección encontrada: pdf_knowledge")


# Lista global de carreras
loaded_carreras = []


# =======================
# 🧩 FUNCIONES AUXILIARES
# =======================

def extract_text_from_pdf(path):
    text = ""
    with fitz.open(path) as pdf:
        for page in pdf:
            text += page.get_text("text")
    return text


def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


# ======================
# 🌐 RUTAS DEL BACKEND
# ======================

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/carreras')
def get_carreras():
    return jsonify(sorted(loaded_carreras))


@app.route('/upload', methods=['POST'])
def upload_pdf():
    file = request.files['pdf']
    if not file:
        return jsonify({"error": "No se recibió ningún archivo."}), 400

    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)

    text = extract_text_from_pdf(path)
    chunks = chunk_text(text)

    # Generar embeddings desde OpenAI
    vectors = [embed(c) for c in chunks]
    payloads = [{"text": c, "file": file.filename} for c in chunks]

    # Guardar en Qdrant
    qdrant.upsert(
        collection_name="pdf_knowledge",
        points=models.Batch(
            ids=[uuid.uuid4().int % (10**8) for _ in vectors],
            vectors=vectors,
            payloads=payloads
        )
    )

    return jsonify({"status": "success", "chunks_indexed": len(chunks)})


@app.route('/query', methods=['POST'])
def query_pdf():
    data = request.get_json()
    query = data.get("query", "")
    carrera = data.get("carrera", None)

    if not query:
        return jsonify({"error": "No se recibió la consulta."}), 400

    query_vector = embed(query)

    if carrera:
        filter_condition = models.Filter(
            must=[models.FieldCondition(key="carrera", match=models.MatchValue(value=carrera))]
        )
    else:
        filter_condition = None

    results = qdrant.search(
        collection_name="pdf_knowledge",
        query_vector=query_vector,
        query_filter=filter_condition,
        limit=3
    )

    if not results:
        return jsonify({"answer": "No se encontraron resultados relevantes."})

    context = "\n".join([r.payload["text"] for r in results])

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Eres un asistente que responde preguntas basadas en PDFs académicos."},
            {"role": "user", "content": f"Pregunta: {query}\n\nContexto:\n{context}"}
        ]
    )

    answer = completion.choices[0].message.content

    return jsonify({"answer": answer, "context": context})


# =======================
# 📚 PRECARGA DE PDFs
# =======================

def preload_pdfs():
    global loaded_carreras
    repo_path = "pdf_repo"

    if not os.path.exists(repo_path):
        print("⚠️ No existe la carpeta pdf_repo.")
        return

    files = [f for f in os.listdir(repo_path) if f.endswith(".pdf")]
    if not files:
        print("⚠️ No hay PDFs en pdf_repo.")
        return

    # Obtener archivos ya indexados
    try:
        points, _ = qdrant.scroll(
            collection_name="pdf_knowledge",
            limit=20000,
            with_payload=True,
        )

        indexed_files = {
            p.payload.get("file")
            for p in points
            if p.payload and "file" in p.payload
        }

    except Exception as e:
        print("⚠️ Error leyendo datos existentes:", e)
        indexed_files = set()

    for pdf_file in files:
        if pdf_file in indexed_files:
            print(f"⏭️ PDF ya indexado: {pdf_file}")
            continue

        carrera = os.path.splitext(pdf_file)[0].strip().title()
        loaded_carreras.append(carrera)

        path = os.path.join(repo_path, pdf_file)
        text = extract_text_from_pdf(path)
        chunks = chunk_text(text)

        vectors = [embed(c) for c in chunks]
        payloads = [{"text": c, "file": pdf_file, "carrera": carrera} for c in chunks]

        qdrant.upsert(
            collection_name="pdf_knowledge",
            points=models.Batch(
                ids=[uuid.uuid4().int % (10**8) for _ in vectors],
                vectors=vectors,
                payloads=payloads
            )
        )

        print(f"📚 Cargado PDF nuevo: {pdf_file} ({len(chunks)} chunks)")

    print("\n✅ Precarga completada.")


# Ejecutar la precarga en segundo plano
threading.Thread(target=preload_pdfs, daemon=True).start()

# =======================
# 🚀 SERVIDOR FLASK
# =======================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
