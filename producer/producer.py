# producer/producer.py

import pandas as pd
from kafka import KafkaProducer
import json
import time
import os

# --- Configurações ---
KAFKA_BROKER = 'localhost:9092'  # Endereço do Kafka DENTRO do Docker Compose
KAFKA_TOPIC = 'delivery_stream'
DATASET_PATH = '/home/rubem/Documentos/Rubem/Aplicacao_de_predicao/data/amazon_delivery.csv' 
# Aguarda o Kafka estar pronto (simples, mas eficaz)
print("Produtor iniciando... aguardando Kafka.")
time.sleep(30) # Tempo para o Kafka e Zookeeper iniciarem no docker-compose

# --- Conexão com o Kafka ---
try:
    producer = KafkaProducer(
        bootstrap_servers=[KAFKA_BROKER],
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )
    print("Conectado ao Kafka com sucesso!")
except Exception as e:
    print(f"Erro ao conectar ao Kafka: {e}")
    print("Encerrando o produtor.")
    os._exit(1) # Encerra o container se não conseguir conectar

# --- Carregamento dos Dados ---
try:
    df = pd.read_csv(DATASET_PATH)
    # Substitui NaN (que não é JSON válido) por None (que é)
    df = df.where(pd.notnull(df), None) 
    print(f"Dataset '{DATASET_PATH}' carregado. {len(df)} registros encontrados.")
except FileNotFoundError:
    print(f"ERRO: Dataset não encontrado em '{DATASET_PATH}'")
    os._exit(1)

# --- Loop de Simulação ---
print(f"Iniciando simulação. Enviando dados para o tópico: '{KAFKA_TOPIC}'")
while True: # Loop infinito para simular um fluxo contínuo
    try:
        # Itera sobre o dataframe e envia linha por linha
        for record in df.to_dict('records'):
            
            # Nota: Em um cenário real, você não enviaria 'Delivery_Time'
            # Estamos enviando aqui para simplificar a simulação.
            # O modelo de streaming irá ignorá-lo e focar nas features.
            
            producer.send(KAFKA_TOPIC, value=record)
            print(f"Pedido enviado: {record.get('Order_ID', 'N/A')}")
            
            # Simula a chegada de um novo pedido a cada 1-3 segundos
            time.sleep(abs(1 + (hash(str(record)) % 3))) 
        
        print("Fim do dataset, reiniciando a simulação...")
        
    except Exception as e:
        print(f"Erro durante o envio: {e}")
        time.sleep(5) # Aguarda 5s e tenta reconectar/enviar