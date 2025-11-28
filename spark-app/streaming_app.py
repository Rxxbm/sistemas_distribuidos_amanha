# spark-app/streaming_app.py

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, udf
from pyspark.sql.types import *
from pyspark.ml import PipelineModel

from pyspark.sql.functions import col, from_json, udf, date_format, hour, to_timestamp # NOVO IMPORT
import math

# --- 1. DEFINIÇÃO DA UDF (Deve ser idêntica à do script de treino) ---

@udf(DoubleType())
def haversine(lat1, lon1, lat2, lon2):
    """
    Calcula a distância Haversine (em km) entre dois pontos.
    """
    if None in (lat1, lon1, lat2, lon2):
        return None
    
    R = 6371  # Raio da Terra em km
    
    dLat = math.radians(lat2 - lat1)
    dLon = math.radians(lon2 - lon1)
    lat1 = math.radians(lat1)
    lat2 = math.radians(lat2)
    
    a = math.sin(dLat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dLon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c
    return distance

# --- 2. INICIALIZAÇÃO DO SPARK ---

print("Iniciando aplicação de streaming Spark...")

spark = (SparkSession.builder
    .appName("RealTimeDeliveryPrediction")
    .getOrCreate())

# Define o log level para WARN para reduzir a verbosidade
spark.sparkContext.setLogLevel("WARN")

# --- 3. ESQUEMA E CARREGAMENTO DO MODELO ---

# O esquema deve corresponder EXATAMENTE ao seu CSV (16 colunas)
# E aos dados que o producer.py está enviando
schema = StructType([
    StructField("Order_ID", StringType(), True),
    StructField("Agent_Age", IntegerType(), True),
    StructField("Agent_Rating", DoubleType(), True),
    StructField("Store_Latitude", DoubleType(), True),
    StructField("Store_Longitude", DoubleType(), True),
    StructField("Drop_Latitude", DoubleType(), True),
    StructField("Drop_Longitude", DoubleType(), True),
    StructField("Order_Date", StringType(), True), #serão testados no calculo do tempo
    StructField("Order_Time", StringType(), True), #
    StructField("Pickup_Time", StringType(), True),
    StructField("Weather", StringType(), True),
    StructField("Traffic", StringType(), True),
    StructField("Vehicle", StringType(), True),
    StructField("Area", StringType(), True),
    StructField("Delivery_Time", IntegerType(), True), # O valor real (será ignorado pelo modelo, mas usado para referência)
    StructField("Category", StringType(), True)
])

MODEL_SAVE_PATH = '/home/rubem/Documentos/Rubem/Aplicacao_de_predicao/model/spark_delivery_pipeline'

# Carrega o Pipeline de ML treinado
try:
    pipeline_model = PipelineModel.load(MODEL_SAVE_PATH)
    print("Modelo de Pipeline carregado com sucesso.")
except Exception as e:
    print(f"ERRO: Não foi possível carregar o modelo de '{MODEL_SAVE_PATH}'")
    print(f"Verifique se o script 'train-model.py' foi executado. Erro: {e}")
    spark.stop()
    exit(1)


# --- 4. LEITURA DO KAFKA (STREAMING) ---

print("Conectando ao Kafka (localhost:9092) no tópico 'delivery_stream'...")

# Lê o stream do Kafka
kafka_df = (spark.readStream
    .format("kafka")
    .option("kafka.bootstrap.servers", "localhost:9092") # Conexão local
    .option("subscribe", "delivery_stream")
    .option("startingOffsets", "latest") # Processa apenas novos dados
    .load())

# Converte o JSON (que está em binário 'value') para String e aplica o schema
stream_df = (kafka_df
    .selectExpr("CAST(value AS STRING)")
    .select(from_json(col("value"), schema).alias("data"))
    .select("data.*"))

# --- 5. LÓGICA DE PROCESSAMENTO (foreachBatch) ---

def process_batch(batch_df, batch_id):
    """
    Função chamada para cada micro-lote de dados recebido do Kafka.
    """
    if batch_df.count() > 0:
        print(f"\n--- Processando Lote {batch_id} ---")
        
        # 1. Engenharia de Feature (idêntica ao treino)

        # 1a. NOVO: Features de Tempo
        features_df = batch_df.withColumn(
            "Order_Timestamp",
            to_timestamp(col("Order_Date") + " " + col("Order_Time"), "yyyy-MM-dd HH:mm:ss")
        ).withColumn(
            "Delivery_Day_of_Week", 
            date_format(col("Order_Timestamp"), "EEE")
        ).withColumn(
            "Delivery_Hour", 
            hour(col("Order_Timestamp"))
        )

        # Calcula a distância de entrega para os novos dados
        features_df = batch_df.withColumn(
            "Delivery_Distance", 
            haversine(
                col("Store_Latitude"), col("Store_Longitude"), 
                col("Drop_Latitude"), col("Drop_Longitude")
            )
        )

        
        # 2. Predição
        # Aplica o pipeline carregado (transformação + modelo)
        predictions_df = pipeline_model.transform(features_df)
        
        # 3. Exibição do Resultado
        # Seleciona as colunas que queremos ver
        output_df = predictions_df.select(
            col("Order_ID"),
            col("Delivery_Time").alias("Tempo_Real"),
            col("prediction").alias("Tempo_Previsto_Min"),
            col("Delivery_Day_of_Week").alias("Dia"), # NOVO
            col("Delivery_Hour").alias("Hora")       # NOVO
        )
        
        print("Predições realizadas:")
        output_df.show(truncate=False)

# --- 6. INÍCIO DO STREAMING (SINK) ---

print("Iniciando query de streaming. Aguardando dados...")

# Usa foreachBatch para aplicar nossa lógica de predição
query = (stream_df.writeStream
    .foreachBatch(process_batch)
    .outputMode("update") # O modo 'update' é necessário para foreachBatch
    .trigger(processingTime='15 seconds') # Processa dados a cada 15 segundos
    .start())

try:
    query.awaitTermination()
except KeyboardInterrupt:
    print("Parando a aplicação de streaming...")

print("Aplicação parada.")
