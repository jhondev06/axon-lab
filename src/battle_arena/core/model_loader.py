"""
Model Loader
Carrega e gerencia modelos treinados pelo AXON para uso na Battle Arena.
"""

import os
import json
import pickle
import logging
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from dataclasses import dataclass
import lightgbm as lgb
import joblib


@dataclass
class ModelInfo:
    """Informações sobre um modelo carregado."""
    model_name: str
    timestamp: str
    model_type: str
    model: Any
    metadata: Dict[str, Any]
    feature_names: List[str]
    config: Dict[str, Any]
    metrics: Dict[str, float]


class ModelLoader:
    """
    Carrega e gerencia modelos treinados pelo AXON.

    Suporte a modelos individuais e ensembles com cache inteligente.
    """

    SUPPORTED_MODEL_TYPES = ['lightgbm', 'xgboost', 'randomforest', 'catboost']

    def __init__(self, artifacts_dir: str = "outputs/artifacts",
                 cache_enabled: bool = True, cache_size: int = 10):
        """
        Inicializa o ModelLoader.

        Args:
            artifacts_dir: Diretório onde os artefatos estão salvos
            cache_enabled: Se deve usar cache de modelos carregados
            cache_size: Tamanho máximo do cache
        """
        self.artifacts_dir = Path(artifacts_dir)
        self.cache_enabled = cache_enabled
        self.cache_size = cache_size
        self.logger = logging.getLogger(self.__class__.__name__)

        # Cache de modelos carregados
        self._model_cache: Dict[str, ModelInfo] = {}
        self._cache_order: List[str] = []  # Para LRU eviction

        # Verificar se diretório existe
        if not self.artifacts_dir.exists():
            self.logger.warning(f"Diretório de artefatos não existe: {self.artifacts_dir}")
            self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"ModelLoader inicializado com cache={'habilitado' if cache_enabled else 'desabilitado'}")

    def load_model(self, model_id: str) -> Optional[ModelInfo]:
        """
        Carrega um modelo específico.

        Args:
            model_id: ID do modelo (ex: 'lightgbm_20250926_094300')

        Returns:
            ModelInfo se carregado com sucesso, None caso contrário
        """
        # Verificar cache primeiro
        if self.cache_enabled and model_id in self._model_cache:
            self.logger.debug(f"Modelo {model_id} encontrado no cache")
            self._update_cache_access(model_id)
            return self._model_cache[model_id]

        try:
            # Procurar arquivos do modelo
            metadata_file = self.artifacts_dir / f"{model_id}_metadata.json"
            if not metadata_file.exists():
                self.logger.error(f"Metadata não encontrado: {metadata_file}")
                return None

            # Carregar metadata
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)

            # Determinar tipo de modelo
            model_name = metadata.get('model_name', '')
            if model_name not in self.SUPPORTED_MODEL_TYPES:
                self.logger.error(f"Tipo de modelo não suportado: {model_name}")
                return None

            # Carregar modelo
            model = self._load_model_file(model_id, model_name, metadata)
            if model is None:
                return None

            # Criar ModelInfo
            model_info = ModelInfo(
                model_name=model_name,
                timestamp=metadata.get('timestamp', ''),
                model_type=model_name,
                model=model,
                metadata=metadata,
                feature_names=metadata.get('feature_names', []),
                config=metadata.get('config', {}),
                metrics=metadata.get('metrics', {})
            )

            # Adicionar ao cache
            if self.cache_enabled:
                self._add_to_cache(model_id, model_info)

            self.logger.info(f"Modelo {model_id} carregado com sucesso")
            return model_info

        except Exception as e:
            self.logger.error(f"Erro ao carregar modelo {model_id}: {e}")
            return None

    def load_best_model(self, model_type: str = None) -> Optional[ModelInfo]:
        """
        Carrega o melhor modelo disponível baseado nas métricas.

        Args:
            model_type: Tipo específico de modelo ou None para qualquer

        Returns:
            Melhor ModelInfo encontrado
        """
        available_models = self.list_available_models()

        if not available_models:
            self.logger.warning("Nenhum modelo disponível")
            return None

        # Filtrar por tipo se especificado
        if model_type:
            available_models = [m for m in available_models if m['model_type'] == model_type]

        if not available_models:
            self.logger.warning(f"Nenhum modelo do tipo {model_type} encontrado")
            return None

        # Ordenar por AUC (ou outra métrica principal)
        best_model = max(available_models,
                        key=lambda x: x['metrics'].get('auc', 0))

        return self.load_model(best_model['model_id'])

    def load_ensemble(self, model_ids: List[str]) -> Optional[Dict[str, ModelInfo]]:
        """
        Carrega múltiplos modelos para ensemble.

        Args:
            model_ids: Lista de IDs de modelos

        Returns:
            Dicionário com ModelInfo para cada modelo
        """
        ensemble = {}
        for model_id in model_ids:
            model_info = self.load_model(model_id)
            if model_info:
                ensemble[model_id] = model_info
            else:
                self.logger.warning(f"Falha ao carregar modelo {model_id} para ensemble")

        if not ensemble:
            self.logger.error("Nenhum modelo do ensemble pôde ser carregado")
            return None

        self.logger.info(f"Ensemble carregado com {len(ensemble)} modelos")
        return ensemble

    def list_available_models(self) -> List[Dict[str, Any]]:
        """
        Lista todos os modelos disponíveis no diretório de artefatos.

        Returns:
            Lista de dicionários com informações dos modelos
        """
        models = []

        try:
            # Procurar arquivos de metadata
            for file_path in self.artifacts_dir.glob("*_metadata.json"):
                try:
                    model_id = file_path.stem.replace('_metadata', '')

                    with open(file_path, 'r') as f:
                        metadata = json.load(f)

                    model_info = {
                        'model_id': model_id,
                        'model_type': metadata.get('model_name', ''),
                        'timestamp': metadata.get('timestamp', ''),
                        'metrics': metadata.get('metrics', {}),
                        'feature_count': len(metadata.get('feature_names', [])),
                        'config': metadata.get('config', {})
                    }

                    models.append(model_info)

                except Exception as e:
                    self.logger.warning(f"Erro ao processar metadata {file_path}: {e}")

        except Exception as e:
            self.logger.error(f"Erro ao listar modelos: {e}")

        # Ordenar por timestamp (mais recente primeiro)
        models.sort(key=lambda x: x['timestamp'], reverse=True)

        return models

    def validate_model_compatibility(self, model_info: ModelInfo) -> Dict[str, Any]:
        """
        Valida se um modelo é compatível com a Battle Arena.

        Args:
            model_info: Informações do modelo

        Returns:
            Dicionário com resultado da validação
        """
        issues = []
        warnings = []

        # Verificar features obrigatórias
        required_features = ['close', 'volume', 'returns']
        missing_features = [f for f in required_features if f not in model_info.feature_names]

        if missing_features:
            issues.append(f"Features obrigatórias faltando: {missing_features}")

        # Verificar métricas mínimas
        min_auc = 0.5
        auc = model_info.metrics.get('auc', 0)
        if auc < min_auc:
            warnings.append(f"AUC abaixo do mínimo ({auc:.3f} < {min_auc})")

        # Verificar se modelo tem predict_proba
        if not hasattr(model_info.model, 'predict_proba'):
            issues.append("Modelo não tem método predict_proba")

        # Verificar target binário
        # Assumindo que é um problema de classificação binária

        result = {
            'compatible': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'feature_count': len(model_info.feature_names),
            'model_type': model_info.model_type
        }

        return result

    def clear_cache(self) -> None:
        """Limpa o cache de modelos."""
        self._model_cache.clear()
        self._cache_order.clear()
        self.logger.info("Cache de modelos limpo")

    def get_cache_info(self) -> Dict[str, Any]:
        """
        Retorna informações sobre o cache.

        Returns:
            Informações do cache
        """
        return {
            'enabled': self.cache_enabled,
            'size': len(self._model_cache),
            'max_size': self.cache_size,
            'models': list(self._model_cache.keys())
        }

    def _load_model_file(self, model_id: str, model_type: str,
                        metadata: Dict[str, Any]) -> Optional[Any]:
        """Carrega o arquivo do modelo baseado no tipo."""
        try:
            # Tentar diferentes extensões
            extensions = ['.pkl', '.lgb', '.joblib']
            model_file = None

            for ext in extensions:
                candidate = self.artifacts_dir / f"{model_id}{ext}"
                if candidate.exists():
                    model_file = candidate
                    break

            if not model_file:
                self.logger.error(f"Arquivo do modelo não encontrado: {model_id}")
                return None

            # Carregar baseado no tipo
            if model_type == 'lightgbm':
                if model_file.suffix == '.lgb':
                    model = lgb.Booster(model_file=str(model_file))
                else:
                    with open(model_file, 'rb') as f:
                        model = pickle.load(f)

            elif model_type in ['xgboost', 'randomforest', 'catboost']:
                with open(model_file, 'rb') as f:
                    model = pickle.load(f)

            else:
                self.logger.error(f"Tipo de modelo não suportado para carregamento: {model_type}")
                return None

            return model

        except Exception as e:
            self.logger.error(f"Erro ao carregar arquivo do modelo {model_id}: {e}")
            return None

    def _add_to_cache(self, model_id: str, model_info: ModelInfo) -> None:
        """Adiciona modelo ao cache com LRU."""
        if model_id in self._model_cache:
            self._update_cache_access(model_id)
            return

        # Verificar se cache está cheio
        if len(self._model_cache) >= self.cache_size:
            # Remover menos recentemente usado
            lru_model = self._cache_order.pop(0)
            del self._model_cache[lru_model]
            self.logger.debug(f"Modelo {lru_model} removido do cache (LRU)")

        # Adicionar novo
        self._model_cache[model_id] = model_info
        self._cache_order.append(model_id)

    def _update_cache_access(self, model_id: str) -> None:
        """Atualiza ordem de acesso no cache."""
        if model_id in self._cache_order:
            self._cache_order.remove(model_id)
            self._cache_order.append(model_id)