import os, glob, time, statistics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import face_recognition
import psycopg2
from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector
from dotenv import load_dotenv

# p0: configuracion basica
DATASET_PATH = os.getenv('DATASET_PATH')
PHOTO_PATH = os.getenv('PHOTO_PATH')
SALIDA = os.getenv('SALIDA')

DB = {
    'host': os.getenv('DB_HOST'),
    'port': int(os.getenv('DB_PORT')),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'dbname': os.getenv('DB_NAME')
}

# para no reprocesar miles de imagenes, creamos un archivo .npz que servira
# de cache (en mi computadora tarda casi una hora en procesar las imagenes)
CACHE_PATH = 'embeddings_full.npz'

os.makedirs(SALIDA, exist_ok=True) # crea la carpeta /outputs si no existe

coleccion = pd.DataFrame([
    {"person": os.path.basename(os.path.dirname(p)), "path": p}
    for p in glob.iglob(os.path.join(DATASET_PATH, "**", "*.jpg"), recursive=True)
])
print(len(coleccion), "imagenes")

def mostrarFotos(coleccion, posiciones):
    plt.figure(figsize=(16, 10))

    for i, idx in enumerate(posiciones):
        img = plt.imread(coleccion.path.iloc[idx])

        plt.subplot(4, 4, i + 1)
        plt.imshow(img)
        plt.title(coleccion.person.iloc[idx] + str(img.shape))
        plt.xticks([])
        plt.yticks([])

    plt.tight_layout()
    plt.show()

mostrarFotos(coleccion, list(range(16)))

# definimos varias funciones auxiliares que se usaran a lo largo del codigo
# obtener los k vectores mas cercanos a q usando la distancia indicada
# <-> representa distancia euclidiana y <=> representa distancia coseno
def knn(q, k=10, op='<->'):
    cur.execute(
        f"""
        SELECT id FROM face_embeddings
        ORDER BY embedding {op} %s LIMIT %s
        """, (np.asarray(q), k)
    )
    return [fila[0] for fila in cur.fetchall()]

# desactivar los indices para obtener los resultados exactos mediante
# una busqueda lineal
def knn_lineal(q, k=10, op='<->'):
    cur.execute('SET enable_indexscan = off')
    ids = knn(q, k, op)
    cur.execute('RESET enable_indexscan')
    return ids

# desactivar los recorridos secuenciales para intentar forzar el uso
# del indice aproximado
def buscar(q, k=10, op='<->'):
    cur.execute('SET enable_seqscan = off')
    res = knn(q, k, op)
    cur.execute('RESET enable_seqscan')
    return res

# ejecutar diez busquedas y devolver los resultados y la mediana
# de los tiempos medidos en milisegundos
def medir(q, k=10, op='<->', lineal=False):
    tiempos = []

    for _ in range(10):
        inicio = time.perf_counter()
        if lineal: ids = knn_lineal(q, k, op)
        else: ids = buscar(q, k, op)
        tiempos.append((time.perf_counter() - inicio) * 1000)

    return ids, statistics.median(tiempos)

# mostrar el plan de ejecucion y guardar el resultado de EXPLAIN ANALYZE
# en un archivo para incluirlo en el informe
def explicar(q, archivo, lineal=False):
    if lineal: cur.execute('SET enable_indexscan = off')
    else: cur.execute('SET enable_seqscan = off')

    try:
        cur.execute(
        """
            EXPLAIN (ANALYZE, BUFFERS)
            SELECT id FROM face_embeddings
            ORDER BY embedding <-> %s
            LIMIT 10
        """, (np.asarray(q),))

        texto = '\n'.join(fila[0] for fila in cur.fetchall())
    finally:
        if lineal:
            cur.execute('RESET enable_indexscan')
        else:
            cur.execute('RESET enable_seqscan')

    with open(os.path.join(SALIDA, archivo), 'w', encoding='utf-8') as f:
        f.write(texto)

    print(texto)


# medir el tiempo de ejecucion usando EXPLAIN ANALYZE once veces
# descartar la primera medicion y calcular la mediana de las diez restantes
def tiempo_explain(q, lineal=False):
    tiempos = []

    for i in range(11):
        if lineal:
            cur.execute('SET enable_indexscan = off')
        else:
            cur.execute('SET enable_seqscan = off')

        try:
            cur.execute(
            """
                EXPLAIN ANALYZE
                SELECT id FROM face_embeddings
                ORDER BY embedding <-> %s
                LIMIT 10
            """, (np.asarray(q),))

            plan = [fila[0] for fila in cur.fetchall()]
        finally:
            if lineal:
                cur.execute('RESET enable_indexscan')
            else:
                cur.execute('RESET enable_seqscan')

        # extraer el tiempo total de ejecucion del plan
        for linea in plan:
            if linea.startswith('Execution Time:'):
                tiempos.append(float(linea.split()[2]))
                break

    return statistics.median(tiempos[1:])

# p1: generar y almacenar embeddings

# buscar las imagenes en las que se detecten rostros y retornar
# su persona, ruta y embedding 128d dentro de las primeras n imagenes de la
# coleccion
def generate_face_embeddings(coleccion, n):
    registros = []

    for i, foto in enumerate(coleccion.head(n).itertuples(index=False), 1):
        image = face_recognition.load_image_file(foto.path)
        encodings = face_recognition.face_encodings(image)

        if encodings:
            registros.append((foto.person, foto.path, encodings[0]))

        # verificar el progreso del procesamiento
        # if i % 500 == 0: print("(p1) imgs procesadas:", i)

    return registros

# si ya existe un cache .npz, entonces usamos esos datos.
# caso contrario, procesamos las imagenes y guardamos la informacion en el
# archivo .npz
if os.path.exists(CACHE_PATH):
    datos = np.load(CACHE_PATH)
    registros = list(zip(datos['personas'], datos['rutas'], datos['vectores']))
else:
    registros = generate_face_embeddings(coleccion, None)
    np.savez(
        CACHE_PATH,
        personas=[r[0] for r in registros],
        rutas=[r[1] for r in registros],
        vectores=[r[2] for r in registros]
    )

print("(p1)", len(registros), "embeddings")

# separar cada tupla (persona, ruta, embedding) por persona
por_persona = {}
for persona, ruta, embedding in registros:
    if persona not in por_persona:
        por_persona[persona] = []
    por_persona[persona].append((persona, ruta, embedding))

# quedarnos solo con personas con al menos dos fotos en el dataset
elegibles = [
    persona for persona in por_persona
    if len(por_persona[persona]) >= 2
]

# elegimos 100 personas aleatorias
rng = np.random.default_rng(42)
personas_consulta = rng.choice(elegibles, 100, replace=False)

# elegimos una foto aleatoria por cada persona seleccionada para usarla
# como consulta, sin insertar esa foto en la tabla
consultas = []
for persona in personas_consulta:
    fotos = por_persona[persona]
    consultas.append(fotos[rng.integers(len(fotos))])

# guardar las rutas de las fotos reservadas para excluirlas del entrenamiento
rutas_consulta = {r[1] for r in consultas}
entrenamiento = [r for r in registros if r[1] not in rutas_consulta]
print('(p1) entrenamiento:', len(entrenamiento))
print('(p1) consultas:', len(consultas))

# conexion psql
conn = psycopg2.connect(**DB)
conn.autocommit = True
cur = conn.cursor()

cur.execute('CREATE EXTENSION IF NOT EXISTS vector')
register_vector(conn)

# crear la tabla donde se guardaran los embeddings de 128 dimensiones
cur.execute(
    """
    CREATE TABLE IF NOT EXISTS face_embeddings (
        id SERIAL PRIMARY KEY,
        name TEXT,
        path TEXT,
        embedding VECTOR(128)
    )
    """
)

# eliminar los indices
# esto lo usamos para borrarlos cuando necesitemos busqueda lineal pura y
# cuando queramos medir el tiempo en el testeo
def borrar_indices():
    cur.execute('DROP INDEX IF EXISTS face_ivf')
    cur.execute('DROP INDEX IF EXISTS face_hnsw')

# reemplazar los datos de la tabla por los registros recibidos
# actualizar las estadisticas de PostgreSQL para mejorar sus planes de consulta
def guardar(registros):
    borrar_indices()
    cur.execute('TRUNCATE face_embeddings RESTART IDENTITY')

    execute_values(
        cur,
        'INSERT INTO face_embeddings (name, path, embedding) VALUES %s',
        registros,
        page_size=500
    )

    cur.execute('ANALYZE face_embeddings')


# crear un indice IVFFlat o HNSW para acelerar las busquedas kNN
def crear_indice(metodo, listas=100):
    borrar_indices()

    if metodo == 'IVFFlat':
        cur.execute(
        f"""
            CREATE INDEX face_ivf ON face_embeddings
            USING ivfflat (embedding vector_l2_ops)
            WITH (lists = {listas})
        """)
    else:
        cur.execute(
        """
            CREATE INDEX face_hnsw ON face_embeddings
            USING hnsw (embedding vector_l2_ops)
            WITH (m = 16, ef_construction = 64)
        """)

    cur.execute('ANALYZE face_embeddings')
    
# ajustar el numero de clusters visitados en IVFFlat o candidatos explorados
# en HNSW durante la busqueda
def configurar(metodo, parametro):
    if metodo == 'IVFFlat':
        cur.execute('SET ivfflat.probes = %s', (parametro,))
    else:
        cur.execute('SET hnsw.ef_search = %s', (parametro,))

# p2. knn
# 2a) medicion de knn

# cargar los embeddings de entrenamiento en PostgreSQL
# tomamos el embedding de alguna de las consultas previamente definidas
guardar(entrenamiento)
q = consultas[0][2]

# comparar kNN lineal con distancia euclidiana y distancia coseno
# y medir la mediana de diez ejecuciones para cada distancia
ids_euc, t_euc = medir(q, op='<->', lineal=True)
ids_cos, t_cos = medir(q, op='<=>', lineal=True)

print('(p2) euclidiana:', ids_euc, t_euc, 'ms')
print('(p2) coseno:', ids_cos, t_cos, 'ms')
print('(p2) mismo orden?', ids_euc == ids_cos)

# 2b) dobles parecidos

# encontrar las cinco personas distintas mas parecidas a la consulta
# considerar primero los 50 vectores mas cercanos y agruparlos por nombre
def famoso_parecido(q, candidatos=50, top=5):
    cur.execute(
    """
        SELECT name, MIN(dist) AS dist, COUNT(*) AS fotos
        FROM (SELECT name, embedding <-> %s AS dist
            FROM face_embeddings ORDER BY embedding <-> %s LIMIT %s) t
        GROUP BY name ORDER BY dist LIMIT %s
    """, (q, q, candidatos, top))
    return cur.fetchall()

# mostrar nuestra foto junto a la foto mas cercana de cada uno de los
# dobles encontrados y guardar la comparacion como imagen
# (hecho por IA, solo usado para generar la imagen para el informe)
def mostrar_dobles(q, resultados, archivo):
    plt.figure(figsize=(15, 3))

    plt.subplot(1, len(resultados) + 1, 1)
    plt.imshow(face_recognition.load_image_file(PHOTO_PATH))
    plt.title('Mi foto')
    plt.axis('off')

    for i, (nombre, distancia, _) in enumerate(resultados, 2):
        cur.execute(
        """
            SELECT path
            FROM face_embeddings
            WHERE name = %s
            ORDER BY embedding <-> %s
            LIMIT 1
        """, (nombre, np.asarray(q)))

        ruta = cur.fetchone()[0]

        plt.subplot(1, len(resultados) + 1, i)
        plt.imshow(plt.imread(ruta))
        plt.title(f'{nombre}\n{distancia:.3f}', fontsize=8)
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(SALIDA, archivo))
    plt.close()

# ejecucion de codigo
mi_foto = face_recognition.face_encodings(
    face_recognition.load_image_file("mi_foto.jpg")
)[0]
resultados = famoso_parecido(mi_foto)
print("(p2) dobles lineales:", resultados)

# generar la imagen para el informe en /outputs
nombres_lineal = [r[0] for r in resultados]
mostrar_dobles(mi_foto, resultados, "p2_dobles_lineal.png")

# p3. indices IVFFlat y HNSW

# crear y probar un indice a la vez, guardar su plan de ejecucion y medir
# la latencia y comparar tambien los dobles encontrados con la busqueda lineal
for metodo, parametro, archivo in [
    ('IVFFlat', 10, 'p3_explain_ivf.txt'),
    ('HNSW', 100, 'p3_explain_hnsw.txt')
]:
    crear_indice(metodo) # crea indice especifico
    configurar(metodo, parametro) # configura el parametro del indice
    print('\n(p3):', metodo)
    explicar(q, archivo) # explain analyze
    _, tiempo = medir(q) # mediana de 10 pruebas
    print('\t(p3) latencia:', tiempo, 'ms')

    resultados = famoso_parecido(mi_foto)

    # para el informe
    mostrar_dobles(
        mi_foto,
        resultados,
        f'p3_dobles_{metodo.lower()}.png'
    )

    print(
        "\t(p3) mismos 5 famosos que en el recorrido lineal?",
        [r[0] for r in resultados] == nombres_lineal
    )


# 4a) Recall@10
def recall_at_k(exactos, aproximados):
    return len(set(exactos) & set(aproximados)) / len(exactos)

# calcular los diez vecinos exactos de cada consulta mediante busqueda lineal
# y usarlos como referencia para evaluar los indices aproximados
exactos = [knn_lineal(c[2]) for c in consultas]
filas = []

# probar distintos valores de probes para IVFFlat y ef_search para HNSW
for metodo, parametros in [
    ('IVFFlat', [1, 5, 10, 20, 100]),
    ('HNSW', [10, 40, 100, 200])
]:
    crear_indice(metodo)

    for parametro in parametros:
        configurar(metodo, parametro)
        recalls = []
        tiempos = []

        # evaluar el recall y el tiempo de busqueda para las 100 consultas
        for consulta, esperado in zip(consultas, exactos):
            encontrados, tiempo = medir(consulta[2])
            recalls.append(recall_at_k(esperado, encontrados))
            tiempos.append(tiempo)

        # guardar el recall promedio y la latencia mediana de esta configuracion
        recall = statistics.mean(recalls)
        latencia = statistics.median(tiempos)

        filas.append((metodo, parametro, recall, latencia))
        print('(p4.1)):', filas[-1])


# guardar los resultados de recall y latencia en un CSV
df_recall = pd.DataFrame(
    filas,
    columns=['indice', 'parametro', 'recall@10', 'latencia_ms']
)

print(df_recall)

# graficar una variable frente a otra, separando las curvas por metodo
# etiquetar los puntos con el parametro usado cuando se grafica la latencia
def graficar(df, x, y, grupo, archivo):
    plt.figure(figsize=(7, 4))

    for nombre, datos in df.groupby(grupo):
        plt.plot(datos[x], datos[y], marker='o', label=nombre)

        if x == 'latencia_ms':
            for _, fila in datos.iterrows():
                plt.annotate(
                    str(fila['parametro']),
                    (fila[x], fila[y]),
                    fontsize=8
                )

    plt.xlabel(x)
    plt.ylabel(y)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(SALIDA, archivo))
    plt.close()


# graficar el recall frente a la latencia para comparar ambos indices
graficar(
    df_recall,
    'latencia_ms',
    'recall@10',
    'indice',
    'p4_recall_vs_latencia.png'
)


# 4b) rendimiento segun tamaño de dataset

# comparar la latencia de la busqueda lineal y de ambos indices para
# datasets de distintos tamaños
filas = []
tamanos = [1000, 2000, 4000, 8000, 12000]

for n in tamanos:
    print('\n(p4.2): tamaño', n)

    # cargar los primeros n embeddings y calcular los vecinos exactos
    guardar(entrenamiento[:n])
    exactos_n = [knn_lineal(c[2]) for c in consultas]

    # medir el tiempo de la busqueda lineal
    tiempo = tiempo_explain(q, lineal=True)
    filas.append((n, 'lineal', '-', 1.0, tiempo))

    # probar ambos indices y elegir el parametro mas bajo que alcance
    # recall >= 0.95
    for metodo in ['IVFFlat', 'HNSW']:
        # listas se calcula dependiendo de n
        listas = round(n ** 0.5)
        crear_indice(metodo, listas)

        # probar los valores de probes o ef_search en orden creciente
        if metodo == 'IVFFlat':
            parametros = sorted(set([1, 5, 10, 20, listas]))
        else:
            parametros = [10, 40, 100, 200]

        elegido = None
        recall_elegido = 0

        for parametro in parametros:
            configurar(metodo, parametro)

            recalls = []

            # comparar las busquedas aproximadas con los vecinos exactos
            for consulta, esperado in zip(consultas, exactos_n):
                encontrados = buscar(consulta[2])
                recalls.append(recall_at_k(esperado, encontrados))

            recall = statistics.mean(recalls)

            # detener la busqueda cuando se alcance el recall requerido
            if recall >= 0.95:
                elegido = parametro
                recall_elegido = recall
                break

        # medir la latencia del indice con el parametro seleccionado
        tiempo = tiempo_explain(q)

        filas.append((n, metodo, elegido, recall_elegido, tiempo))

        print(
            metodo,
            'parámetro:', elegido,
            'recall:', round(recall_elegido, 3),
            'tiempo:', round(tiempo, 3)
        )


# guardar los resultados de rendimiento en un CSV
df_tiempos = pd.DataFrame(
    filas,
    columns=['n', 'metodo', 'parametro', 'recall@10', 'tiempo_ms']
)

print(df_tiempos)

# graficar el tiempo de consulta segun el tamaño del dataset
graficar(
    df_tiempos,
    'n',
    'tiempo_ms',
    'metodo',
    'p4_tiempo_vs_tamano.png'
)

# cerrar conexiones
cur.close()
conn.close()
