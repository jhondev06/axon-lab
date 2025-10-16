# AXON Models API Reference

Documentação completa da API para todas as classes de modelo no AXON.

## ModelRegistry

Classe central para gerenciamento de modelos.

### Inicialização

```python
from src.models import ModelRegistry

registry = ModelRegistry(config: Dict[str, Any])
```

**Parâmetros:**
- `config`: Dicionário de configuração do AXON

### Métodos

#### get_model(model_name: str, **kwargs) -> Any

Retorna instância de modelo configurada.

```python
model = registry.get_model('xgboost', n_estimators=200)
```

**Parâmetros:**
- `model_name`: Nome do modelo ('lightgbm', 'xgboost', 'catboost', 'lstm', 'ensemble')
- `**kwargs`: Parâmetros adicionais para sobrescrever configuração

**Retorno:** Instância do modelo

#### list_available_models() -> list

Lista todos os modelos disponíveis.

```python
models = registry.list_available_models()
# ['lightgbm', 'lgb', 'xgboost', 'xgb', 'catboost', 'cb', 'lstm', 'randomforest', 'rf', 'ensemble']
```

## XGBoostModel

Wrapper para XGBoost com interface scikit-learn.

### Inicialização

```python
from src.models import XGBoostModel

model = XGBoostModel(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    **kwargs
)
```

### Métodos

#### fit(X, y, eval_set=None, early_stopping_rounds=None, verbose=False)

Treina o modelo.

```python
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=20)
```

**Parâmetros:**
- `X`: Features de treinamento
- `y`: Labels de treinamento
- `eval_set`: Conjunto de validação como [(X_val, y_val)]
- `early_stopping_rounds`: Rounds sem melhoria para parar
- `verbose`: Se deve imprimir progresso

#### predict(X, num_iteration=None)

Faz previsões de classe.

```python
predictions = model.predict(X_test)
```

#### predict_proba(X, num_iteration=None)

Faz previsões de probabilidade.

```python
probabilities = model.predict_proba(X_test)
# Retorna: array([[prob_class_0, prob_class_1], ...])
```

#### get_feature_importance(importance_type='gain')

Retorna importância das features.

```python
importance = model.get_feature_importance('gain')
# Retorna: dict com feature -> importance
```

#### save_model(filepath)

Salva modelo em arquivo.

```python
model.save_model('outputs/artifacts/my_model.xgb')
```

#### load_model(filepath) [classmethod]

Carrega modelo de arquivo.

```python
loaded_model = XGBoostModel.load_model('outputs/artifacts/my_model.xgb')
```

## CatBoostModel

Wrapper para CatBoost com interface scikit-learn.

### Inicialização

```python
from src.models import CatBoostModel

model = CatBoostModel(
    iterations=1000,
    learning_rate=0.03,
    depth=6,
    **kwargs
)
```

### Métodos

#### fit(X, y, eval_set=None, early_stopping_rounds=None, verbose=False)

Treina o modelo.

```python
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=50)
```

#### predict(X)

Faz previsões de classe.

```python
predictions = model.predict(X_test)
```

#### predict_proba(X)

Faz previsões de probabilidade.

```python
probabilities = model.predict_proba(X_test)
```

#### get_feature_importance(importance_type='FeatureImportance')

Retorna importância das features.

```python
importance = model.get_feature_importance('FeatureImportance')
# Retorna: dict com feature -> importance
```

#### save_model(filepath)

Salva modelo em arquivo.

```python
model.save_model('outputs/artifacts/my_model.cb')
```

#### load_model(filepath) [classmethod]

Carrega modelo de arquivo.

```python
loaded_model = CatBoostModel.load_model('outputs/artifacts/my_model.cb')
```

## LSTMModel

Modelo LSTM com interface scikit-learn para séries temporais.

### Inicialização

```python
from src.models import LSTMModel

model = LSTMModel(
    hidden_size=64,
    num_layers=2,
    sequence_length=20,
    **kwargs
)
```

### Métodos

#### fit(X, y, eval_set=None, early_stopping_rounds=None, verbose=False)

Treina o modelo LSTM.

```python
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=10)
```

**Nota:** Converte automaticamente dados para sequências temporais.

#### predict(X)

Faz previsões de classe.

```python
predictions = model.predict(X_test)
```

#### predict_proba(X)

Faz previsões de probabilidade.

```python
probabilities = model.predict_proba(X_test)
```

#### save_model(filepath)

Salva modelo PyTorch + scaler.

```python
model.save_model('outputs/artifacts/my_lstm.pth')
```

#### load_model(filepath) [classmethod]

Carrega modelo completo.

```python
loaded_model = LSTMModel.load_model('outputs/artifacts/my_lstm.pth')
```

## EnsembleModel

Modelo ensemble que combina múltiplos algoritmos.

### Inicialização

```python
from src.models import EnsembleModel

model = EnsembleModel(
    ensemble_type='weighted',
    base_models=['lightgbm', 'xgboost', 'catboost'],
    combination_strategy='performance_based',
    **kwargs
)
```

### Métodos

#### fit(X, y, eval_set=None, config=None, **kwargs)

Treina o ensemble.

```python
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], config=axon_config)
```

#### predict(X)

Faz previsões usando estratégia ensemble.

```python
predictions = model.predict(X_test)
```

#### predict_proba(X)

Faz previsões de probabilidade.

```python
probabilities = model.predict_proba(X_test)
```

#### save_model(filepath)

Salva ensemble completo.

```python
model.save_model('outputs/artifacts/my_ensemble.pkl')
```

#### load_model(filepath) [classmethod]

Carrega ensemble completo.

```python
loaded_model = EnsembleModel.load_model('outputs/artifacts/my_ensemble.pkl')
```

## Funções Utilitárias

### train_model(model, X_train, y_train, X_val, y_val, model_name, config)

Treina qualquer modelo com métricas padronizadas.

```python
from src.models import train_model

trained_model, metrics = train_model(
    model, X_train, y_train, X_val, y_val, 'xgboost', config
)

print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"Accuracy: {metrics['accuracy']:.4f}")
```

**Retorno:** (modelo_treinado, dict_métricas)

### save_model(model, model_name, metrics, feature_names, config)

Salva modelo com metadata completa.

```python
from src.models import save_model

model_path = save_model(
    trained_model, 'xgboost', metrics, feature_names, config
)
```

**Retorno:** Caminho do arquivo salvo

### load_model(model_path)

Carrega modelo com metadata.

```python
from src.models import load_model

model, metadata = load_model('outputs/artifacts/xgboost_20250101_120000.pkl')
```

**Retorno:** (modelo, dict_metadata)

### get_feature_importance(model, feature_names, model_name)

Extrai importância de features de qualquer modelo.

```python
from src.models import get_feature_importance

importance_df = get_feature_importance(model, feature_names, 'xgboost')
print(importance_df.head())
```

**Retorno:** DataFrame com features ordenadas por importância

### cross_validate_model(model, X, y, cv=5)

Executa validação cruzada.

```python
from src.models import cross_validate_model

cv_metrics = cross_validate_model(model, X, y, cv=5)
print(f"CV Accuracy: {cv_metrics['cv_accuracy_mean']:.4f} ± {cv_metrics['cv_accuracy_std']:.4f}")
```

**Retorno:** Dict com métricas de CV

## Tratamento de Dados

### Formatos Suportados

Todos os modelos aceitam:
- **pandas.DataFrame** com colunas nomeadas
- **numpy.ndarray** 2D
- **pandas.Series** para labels

### Pré-processamento Automático

- **Normalização**: LSTM aplica MinMaxScaler automaticamente
- **Sequências**: LSTM converte dados para janelas deslizantes
- **Tipos de dados**: Conversão automática para tipos compatíveis

### Tratamento de Valores Ausentes

- **pandas.DataFrame**: Recomendado usar `fillna()` antes
- **Valores NaN**: Podem causar erros - sempre verificar

## Exceções

### ValueError
- Modelo não encontrado
- Parâmetros inválidos
- Dados incompatíveis

### ImportError
- Biblioteca não instalada (ex: PyTorch para LSTM)
- GPU não disponível

### RuntimeError
- Erro durante treinamento
- Memória insuficiente
- Convergência falhou

## Exemplos Completos

### Treinamento Básico

```python
from src.models import ModelRegistry
from src.utils import load_config

# Carregar configuração
config = load_config()

# Criar registry
registry = ModelRegistry(config)

# Treinar LightGBM
model = registry.get_model('lightgbm')
model.fit(X_train, y_train, eval_set=[(X_val, y_val)])

# Fazer previsões
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)
```

### Ensemble Completo

```python
# Configurar ensemble
ensemble_config = {
    'ensemble_type': 'weighted',
    'base_models': ['lightgbm', 'xgboost', 'catboost'],
    'combination_strategy': 'adaptive'
}

# Criar e treinar
ensemble = registry.get_model('ensemble', **ensemble_config)
ensemble.fit(X_train, y_train, eval_set=[(X_val, y_val)], config=config)

# Salvar
ensemble.save_model('outputs/artifacts/my_ensemble.pkl')
```

### Comparação de Modelos

```python
from src.models import train_model

models_to_compare = ['lightgbm', 'xgboost', 'catboost']
results = {}

for model_name in models_to_compare:
    model = registry.get_model(model_name)
    trained_model, metrics = train_model(
        model, X_train, y_train, X_val, y_val, model_name, config
    )
    results[model_name] = metrics

# Comparar resultados
import pandas as pd
comparison = pd.DataFrame(results).T
print(comparison[['accuracy', 'f1', 'precision']])