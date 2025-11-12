"""
Utilitários para configuração e gerenciamento do Spark.
"""

import os
import logging
from typing import Dict, Any, Optional
from pyspark.sql import SparkSession
from pyspark.conf import SparkConf

from src.build.utils.memory_utils import get_memory_usage, get_cpu_info
from src.build.core.exceptions import ConfigurationError

logger = logging.getLogger(__name__)

class SparkManager:
    """Gerenciador de sessão Spark otimizado."""
    
    def __init__(self, app_name: str = "DockTKinase-Build"):
        self.app_name = app_name
        self.spark: Optional[SparkSession] = None
        self._config = {}
    
    def configure(self, 
                 memory_fraction: float = 0.8,
                 offheap_fraction: float = 0.2,
                 gc_type: str = "G1GC",
                 custom_config: Optional[Dict[str, str]] = None) -> 'SparkManager':
        """
        Configura parâmetros do Spark.
        
        Args:
            memory_fraction: Fração da memória para uso do Spark
            offheap_fraction: Fração para memória off-heap
            gc_type: Tipo de garbage collector
            custom_config: Configurações customizadas adicionais
            
        Returns:
            Self para chaining
        """
        # Obter informações do sistema
        memory_info = get_memory_usage()
        cpu_info = get_cpu_info()
        
        total_memory_gb = memory_info['total_gb']
        available_memory_gb = memory_info['available_gb']
        num_cores = cpu_info['logical_cores']
        
        # Calcular configurações otimizadas com valores mínimos garantidos
        driver_memory = max(1, int(available_memory_gb * memory_fraction * 0.6))  # 60% para driver, mínimo 1GB
        executor_memory = max(1, int(available_memory_gb * memory_fraction * 0.4))  # 40% para executor, mínimo 1GB
        offheap_memory = max(1, int(total_memory_gb * offheap_fraction))  # mínimo 1GB
        
        # Configurar executors
        executor_instances = max(1, num_cores // 4)
        executor_cores = max(1, num_cores // executor_instances)
        
        # Configurações base
        base_config = {
            "spark.master": f"local[{num_cores}]",
            "spark.driver.memory": f"{driver_memory}g",
            "spark.executor.memory": f"{executor_memory}g",
            "spark.executor.instances": str(executor_instances),
            "spark.executor.cores": str(executor_cores),
            "spark.memory.fraction": str(memory_fraction),
            "spark.memory.offHeap.enabled": "true",
            "spark.memory.offHeap.size": f"{offheap_memory}g",
            "spark.executor.extraJavaOptions": f"-XX:+Use{gc_type}",
            "spark.driver.extraJavaOptions": f"-XX:+Use{gc_type}",
            "spark.sql.debug.maxToStringFields": "200",
            "spark.sql.autoBroadcastJoinThreshold": "-1",
            "spark.driver.maxResultSize": f"{max(1, int(total_memory_gb * 0.1))}g",
            "spark.sql.shuffle.partitions": str(num_cores * 4),
            "spark.default.parallelism": str(num_cores * 2),
            "spark.executor.memoryOverhead": f"{max(1, int(total_memory_gb * 0.1))}g",
            "spark.serializer": "org.apache.spark.serializer.KryoSerializer",
            "spark.sql.adaptive.enabled": "true",
            "spark.sql.adaptive.coalescePartitions.enabled": "true"
        }
        
        # Adicionar configurações customizadas
        if custom_config:
            base_config.update(custom_config)
        
        self._config = base_config
        
        logger.info(f"Configuração Spark otimizada:")
        logger.info(f"  Driver Memory: {driver_memory}GB")
        logger.info(f"  Executor Memory: {executor_memory}GB")
        logger.info(f"  Executor Instances: {executor_instances}")
        logger.info(f"  Executor Cores: {executor_cores}")
        logger.info(f"  Total Cores: {num_cores}")
        
        return self
    
    def start(self) -> SparkSession:
        """
        Inicia sessão Spark.
        
        Returns:
            Sessão Spark configurada
        """
        if self.spark:
            return self.spark
        
        if not self._config:
            self.configure()  # Configuração padrão
        
        try:
            # Criar builder
            builder = SparkSession.builder.appName(self.app_name)
            
            # Aplicar configurações
            for key, value in self._config.items():
                builder = builder.config(key, value)
            
            # Criar sessão
            self.spark = builder.getOrCreate()
            
            # Definir nível de log
            self.spark.sparkContext.setLogLevel("WARN")
            
            logger.info(f"Sessão Spark iniciada: {self.app_name}")
            
            return self.spark
            
        except Exception as e:
            raise ConfigurationError(f"Erro ao iniciar Spark: {e}")
    
    def stop(self) -> None:
        """Para sessão Spark."""
        if self.spark:
            self.spark.stop()
            self.spark = None
            logger.info("Sessão Spark finalizada")
    
    def restart(self) -> SparkSession:
        """Reinicia sessão Spark."""
        self.stop()
        return self.start()
    
    def get_session(self) -> Optional[SparkSession]:
        """Obtém sessão ativa."""
        return self.spark
    
    def is_active(self) -> bool:
        """Verifica se sessão está ativa."""
        return self.spark is not None and not self.spark.sparkContext._jsc.sc().isStopped()
    
    def get_config(self) -> Dict[str, str]:
        """Obtém configuração atual."""
        return self._config.copy()
    
    def __enter__(self) -> SparkSession:
        """Context manager: entrada."""
        return self.start()
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager: saída."""
        self.stop()

def create_optimized_spark_session(app_name: str = "DockTKinase-Build",
                                  **config_kwargs) -> SparkSession:
    """
    Cria sessão Spark otimizada para o sistema atual.
    
    Args:
        app_name: Nome da aplicação
        **config_kwargs: Argumentos de configuração
        
    Returns:
        Sessão Spark otimizada
    """
    manager = SparkManager(app_name)
    manager.configure(**config_kwargs)
    return manager.start()

def get_spark_session_info(spark: SparkSession) -> Dict[str, Any]:
    """
    Obtém informações da sessão Spark.
    
    Args:
        spark: Sessão Spark
        
    Returns:
        Informações da sessão
    """
    sc = spark.sparkContext
    conf = sc.getConf()
    
    return {
        'app_name': sc.appName,
        'app_id': sc.applicationId,
        'master': sc.master,
        'version': sc.version,
        'default_parallelism': sc.defaultParallelism,
        'is_stopped': sc._jsc.sc().isStopped(),
        'config': dict(conf.getAll())
    }

def optimize_dataframe_partitions(df, 
                                target_partition_size_mb: int = 128) -> 'DataFrame':
    """
    Otimiza particionamento de DataFrame.
    
    Args:
        df: DataFrame para otimizar
        target_partition_size_mb: Tamanho alvo por partição em MB
        
    Returns:
        DataFrame com particionamento otimizado
    """
    # Estimar tamanho do DataFrame
    row_count = df.count()
    if row_count == 0:
        return df
    
    # Estimar tamanho por linha (aproximação)
    sample_size = min(1000, row_count)
    sample_df = df.sample(fraction=sample_size/row_count)
    
    # Calcular número ideal de partições
    estimated_size_mb = (row_count / sample_size) * target_partition_size_mb * 0.1
    optimal_partitions = max(1, int(estimated_size_mb / target_partition_size_mb))
    
    current_partitions = df.rdd.getNumPartitions()
    
    if optimal_partitions != current_partitions:
        logger.info(f"Reparticionando DataFrame: {current_partitions} -> {optimal_partitions} partições")
        return df.repartition(optimal_partitions)
    
    return df

def monitor_spark_job(func):
    """
    Decorator para monitorar jobs Spark.
    
    Args:
        func: Função que usa Spark
    """
    def wrapper(*args, **kwargs):
        logger.info(f"Iniciando job Spark: {func.__name__}")
        
        try:
            result = func(*args, **kwargs)
            logger.info(f"Job Spark concluído: {func.__name__}")
            return result
            
        except Exception as e:
            logger.error(f"Erro no job Spark {func.__name__}: {e}")
            raise
    
    return wrapper
