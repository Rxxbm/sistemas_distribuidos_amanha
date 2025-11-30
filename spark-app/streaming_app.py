# spark-app/streaming_app.py

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, from_json, udf, concat, lit, to_timestamp,
    date_format, hour
)
from pyspark.sql.types import *
from pyspark.ml import PipelineModel
import math

# ================================
# 1. UDF HAVERSINE
# ================================

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


# ================================
# 2. INICIALIZAÇÃO DO SPARK
# ================================

print("Iniciando aplicação de streaming Spark...")

spark = (
    SparkSession.builder
    .appName("RealTimeDeliveryPrediction")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")


# ================================
# 3. ESQUEMA E CARREGAMENTO DO MODELO
# ================================

schema = StructType([
    StructField("Order_ID", StringType(), True),
    StructField("Agent_Age", IntegerType(), True),
    StructField("Agent_Rating", DoubleType(), True),
    StructField("Store_Latitude", DoubleType(), True),
    StructField("Store_Longitude", DoubleType(), True),
    StructField("Drop_Latitude", DoubleType(), True),
    StructField("Drop_Longitude", DoubleType(), True),
    StructField("Order_Date", StringType(), True),
    StructField("Order_Time", StringType(), True),
    StructField("Pickup_Time", StringType(), True),
    StructField("Weather", StringType(), True),
    StructField("Traffic", StringType(), True),
    StructField("Vehicle", StringType(), True),
    StructField("Area", StringType(), True),
    StructField("Delivery_Time", IntegerType(), True),
    StructField("Category", StringType(), True)
])

MODEL_SAVE_PATH = "model/spark_delivery_pipeline"

try:
    pipeline_model = PipelineModel.load(MODEL_SAVE_PATH)
    print("Modelo de Pipeline carregado com sucesso.")
except Exception as e:
    print(f"ERRO ao carregar modelo '{MODEL_SAVE_PATH}': {e}")
    spark.stop()
    exit(1)


# ================================
# 4. LEITURA DO KAFKA
# ================================

print("Conectando ao Kafka (localhost:9092) no tópico 'delivery_stream'...")

kafka_df = (
    spark.readStream
    .format("kafka")
    .option("kafka.bootstrap.servers", "localhost:9092")
    .option("subscribe", "delivery_stream")
    .option("startingOffsets", "latest")
    .load()
)

stream_df = (
    kafka_df
    .selectExpr("CAST(value AS STRING)")
    .select(from_json(col("value"), schema).alias("data"))
    .select("data.*")
)


# ================================
# 5. PROCESSAMENTO DE CADA LOTE
# ================================

def process_batch(batch_df, batch_id):

    if batch_df.count() == 0:
        return

    print(f"\n--- Processando Lote {batch_id} ---")

    # 1. FEATURE ENGINEERING

    features_df = (
        batch_df
        .withColumn(
            "Order_Timestamp",
            to_timestamp(
                concat(
                    col("Order_Date"),
                    lit(" "),
                    col("Order_Time")
                ),
                "yyyy-MM-dd HH:mm:ss"
            )
        )
        .withColumn(
            "Delivery_Day_of_Week",
            date_format(col("Order_Timestamp"), "EEE")
        )
        .withColumn(
            "Delivery_Hour",
            hour(col("Order_Timestamp"))
        )
        .withColumn(
            "Delivery_Distance",
            haversine(
                col("Store_Latitude"),
                col("Store_Longitude"),
                col("Drop_Latitude"),
                col("Drop_Longitude")
            )
        )
    )

    # 2. APLICA O MODELO
    predictions_df = pipeline_model.transform(features_df)

    # 3. SELEÇÃO DAS COLUNAS DE SAÍDA
    output_df = predictions_df.select(
        col("Order_ID"),
        col("Delivery_Time").alias("Tempo_Real"),
        col("prediction").alias("Tempo_Previsto_Min"),
        col("Delivery_Day_of_Week").alias("Dia"),
        col("Delivery_Hour").alias("Hora")
    )

    print("Predições realizadas:")
    output_df.show(truncate=False)


# ================================
# 6. INÍCIO DO STREAMING
# ================================

print("Iniciando query de streaming...")

query = (
    stream_df.writeStream
    .foreachBatch(process_batch)
    .outputMode("update")
    .trigger(processingTime="15 seconds")
    .start()
)

try:
    query.awaitTermination()
except KeyboardInterrupt:
    print("Encerrando a aplicação de streaming...")

print("Aplicação encerrada.")
