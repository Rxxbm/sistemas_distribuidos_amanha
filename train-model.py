# train_model.py
# (Coloque este arquivo na raiz do projeto e execute-o UMA VEZ)

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import *
import math

#Importando para a previsão do dia da semana e horario de pico
from pyspark.sql.functions import (
    col, udf, date_format, hour, try_to_timestamp,
    avg, concat_ws, lit
)
#

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.regression import RandomForestRegressor

# --- 1. Função de Engenharia de Feature (UDF) ---

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

# --- 2. Inicialização do Spark ---

spark = (SparkSession.builder
    .appName("OfflineModelTraining")
    .master("local[*]") # Roda localmente para o treinamento
    .getOrCreate())

print("Sessão Spark iniciada para treinamento.")

# --- 3. Carregamento e Preparação dos Dados ---

# Esquema baseado na sua imagem
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
    StructField("Delivery_Time", IntegerType(), True), # Nosso ALVO (Label)
    StructField("Category", StringType(), True),




])

DATA_PATH = '/home/rubem/Documentos/Rubem/Aplicacao_de_predicao/data/amazon_delivery.csv' 
MODEL_SAVE_PATH = './model/spark_delivery_pipeline'

# Carrega o dataset
df = (spark.read.csv(DATA_PATH, header=True, schema=schema)
      .na.drop()) # Remove linhas com nulos para simplificar

# --- NOVO: ENGENHARIA DE FEATURES DE TEMPO ---
# 1. Combina Data e Hora em um Timestamp
# O formato dos seus dados Order_Date e Order_Time deve ser compatível
df = df.withColumn(
    "Order_Timestamp",
    try_to_timestamp(
        concat_ws(" ", col("Order_Date"), col("Order_Time")), 
        lit("yyyy-MM-dd HH:mm:ss") # <-- Alterado: lit() garante que seja uma string literal
    )
)

df = df.filter(col("Order_Timestamp").isNotNull()) 

# 2. Extrai o Dia da Semana (Ex: Mon, Tue, Wed...)
df = df.withColumn(
    "Delivery_Day_of_Week", 
    date_format(col("Order_Timestamp"), "EEE")
)

# 3. Extrai a Hora do Dia (0-23)
df = df.withColumn(
    "Delivery_Hour", 
    hour(col("Order_Timestamp"))
)

print("Dados carregados e features de Distância, Dia da Semana e Hora criadas.")


# Aplica a UDF Haversine
df = df.withColumn(
    "Delivery_Distance", 
    haversine(
        col("Store_Latitude"), col("Store_Longitude"), 
        col("Drop_Latitude"), col("Drop_Longitude")
    )
)

print("Dados carregados e feature 'Delivery_Distance' criada.")

# --- NOVO: ANÁLISE PARA PICO E DIA (APENAS INFORMATIVO) ---

# Média do Tempo de Entrega por Dia da Semana
print("\n--- Análise: Média de Entrega por Dia ---")
df.groupBy("Delivery_Day_of_Week").agg(
    {"Delivery_Time": "avg"}
).orderBy(col("avg(Delivery_Time)").desc()).show()

# Média do Tempo de Entrega por Hora do Dia
print("\n--- Análise: Média de Entrega por Hora ---")
df.groupBy("Delivery_Hour").agg(
    {"Delivery_Time": "avg"}
).orderBy(col("avg(Delivery_Time)").desc()).show()

# --- 4. Definição do Pipeline de ML ---

# Identifica colunas categóricas e numéricas
#categorical_cols = ["Weather", "Traffic", "Vehicle", "Area", "Category"]
#numeric_cols = ["Agent_Age", "Agent_Rating", "Delivery_Distance"]

# ATUALIZAÇÃO: Incluir as novas features no pipeline
categorical_cols = ["Weather", "Traffic", "Vehicle", "Area", "Category", "Delivery_Day_of_Week"]
numeric_cols = ["Agent_Age", "Agent_Rating", "Delivery_Distance", "Delivery_Hour"]

# Estágios do Pipeline
stages = []

# Estágio 1: Indexação (String -> Número)
indexers = [
    StringIndexer(inputCol=c, outputCol=f"{c}_Index", handleInvalid="keep")
    for c in categorical_cols
]
stages.extend(indexers)

# Estágio 2: One-Hot Encoding
encoder_inputs = [f"{c}_Index" for c in categorical_cols]
encoder_outputs = [f"{c}_OHE" for c in categorical_cols]
encoder = OneHotEncoder(inputCols=encoder_inputs, outputCols=encoder_outputs)
stages.append(encoder)

# Estágio 3: Assembler (Junta todas as features em um vetor)
feature_cols = numeric_cols + encoder_outputs
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
stages.append(assembler)

# Estágio 4: O Modelo (Regressão)
# Nosso alvo (label) é o 'Delivery_Time'
rf = RandomForestRegressor(featuresCol="features", labelCol="Delivery_Time")
stages.append(rf)

# Cria o Pipeline completo
pipeline = Pipeline(stages=stages)

# --- 5. Avaliação e Treinamento ---

# Importe a biblioteca de avaliação
from pyspark.ml.evaluation import RegressionEvaluator

# 1. Divida os dados em Treino (80%) e Teste (20%)
(trainingData, testData) = df.randomSplit([0.8, 0.2], seed=1234)

print("Iniciando treinamento do pipeline de ML...")
# 2. Treine o modelo APENAS nos dados de treino
pipeline_model = pipeline.fit(trainingData)

print("Treinamento concluído. Avaliando o modelo...")
# 3. Faça predições nos dados de Teste (que o modelo nunca viu)
predictions = pipeline_model.transform(testData)

# 4. Calcule o erro (RMSE)
# Compare a predição ('prediction') com o valor real ('Delivery_Time')
evaluator = RegressionEvaluator(
    labelCol="Delivery_Time", 
    predictionCol="prediction", 
    metricName="rmse"
)
rmse = evaluator.evaluate(predictions)

print(f"Acurácia do modelo (RMSE): O modelo erra, em média, por {rmse:.2f} minutos.")

# --- 6. Salvamento ---

# Agora salve o modelo (que já foi treinado)
pipeline_model.save(MODEL_SAVE_PATH)
print(f"Pipeline de ML salvo com sucesso em: '{MODEL_SAVE_PATH}'")

spark.stop()
