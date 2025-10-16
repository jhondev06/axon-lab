# Tutorial: Usando Múltiplos Modelos no AXON

Este tutorial passo-a-passo mostra como configurar, treinar e comparar múltiplos modelos de machine learning no AXON.

## Pré-requisitos

- AXON instalado e configurado
- Dados de treinamento preparados
- Ambiente Python com dependências instaladas

## Passo 1: Configuração Inicial

### 1.1 Configurar axon.cfg.yml

Edite o arquivo `axon.cfg.yml` para incluir múltiplos modelos:

```yaml
# Modelos para treinamento
models:
  train: ["lightgbm", "xgboost", "catboost", "lstm"]

  # Configurações individuais
  lightgbm:
    n_estimators: 100
    learning_rate: 0.05
    num_leaves: 31
    max_depth: -1

  xgboost:
    n_estimators: 100
    max_depth: 6
    learning_rate: 0.1
    subsample: 0.8

  catboost:
    iterations: 500
    learning_rate: 0.05
    depth: 6
    verbose: False

  lstm:
    hidden_size: 64
    num_layers: 2
    sequence_length: 20
    batch_size: 32
    epochs: 50

# Otimização habilitada
optimization:
  enabled: true
  n_trials: 50
  models: ["lightgbm", "xgboost", "catboost"]
```

### 1.2 Verificar Instalação

Execute o teste básico:

```bash
python -c "
from src.models import ModelRegistry
from src.utils import load_config
config = load_config()
registry = ModelRegistry(config)
print('✅ Configuração válida!')
print('Modelos disponíveis:', registry.list_available_models())
"
```

## Passo 2: Preparação de Dados

### 2.1 Executar Pipeline de Dados

```bash
# Preparar dados
python -m src.dataset

# Verificar dados gerados
ls -la data/processed/
```

### 2.2 Carregar e Explorar Dados

```python
import pandas as pd
import numpy as np
from pathlib import Path

# Carregar dados processados
train_df = pd.read_parquet('data/processed/train_features.parquet')
val_df = pd.read_parquet('data/processed/validation_features.parquet')

print(f"Train shape: {train_df.shape}")
print(f"Validation shape: {val_df.shape}")

# Verificar distribuição de labels
target_col = 'y'
print("Distribuição de labels (treino):")
print(train_df[target_col].value_counts(normalize=True))

# Verificar features disponíveis
feature_cols = [col for col in train_df.columns if col not in ['timestamp', target_col]]
print(f"Número de features: {len(feature_cols)}")
print("Primeiras features:", feature_cols[:5])
```

## Passo 3: Treinamento Básico de Múltiplos Modelos

### 3.1 Treinar Modelos Individualmente

```python
from src.models import ModelRegistry, train_model, save_model
from src.utils import load_config

# Carregar configuração
config = load_config()
registry = ModelRegistry(config)

# Preparar dados
X_train = train_df[feature_cols]
y_train = train_df[target_col]
X_val = val_df[feature_cols]
y_val = val_df[target_col]

# Modelos para treinar
models_to_train = ['lightgbm', 'xgboost', 'catboost']

# Dicionário para armazenar resultados
trained_models = {}
model_metrics = {}

print("🚀 Iniciando treinamento de múltiplos modelos...")

for model_name in models_to_train:
    print(f"\n{'='*50}")
    print(f"🏃 Treinando {model_name.upper()}")
    print(f"{'='*50}")

    try:
        # Obter modelo
        model = registry.get_model(model_name)

        # Treinar
        trained_model, metrics = train_model(
            model, X_train, y_train, X_val, y_val, model_name, config
        )

        # Salvar modelo
        model_path = save_model(
            trained_model, model_name, metrics, feature_cols, config
        )

        # Armazenar resultados
        trained_models[model_name] = trained_model
        model_metrics[model_name] = metrics

        print(f"✅ {model_name} treinado com sucesso!")
        print(".4f"        print(".4f"        print(".4f"
    except Exception as e:
        print(f"❌ Erro ao treinar {model_name}: {e}")
        continue

print(f"\n✅ Treinamento concluído! {len(trained_models)} modelos treinados.")
```

### 3.2 Comparar Performance

```python
import pandas as pd

# Criar DataFrame de comparação
comparison_df = pd.DataFrame(model_metrics).T
print("\n📊 Comparação de Modelos:")
print(comparison_df.round(4))

# Encontrar melhor modelo
best_model = comparison_df['accuracy'].idxmax()
best_accuracy = comparison_df.loc[best_model, 'accuracy']

print(f"\n🏆 Melhor modelo: {best_model} (Accuracy: {best_accuracy:.4f})")

# Salvar comparação
comparison_df.to_csv('outputs/artifacts/model_comparison.csv')
print("💾 Comparação salva em outputs/artifacts/model_comparison.csv")
```

## Passo 4: Otimização de Hiperparâmetros

### 4.1 Executar Otimização Automática

```python
from src.optimization import OptimizationEngine

# Inicializar otimizador
opt_engine = OptimizationEngine(config)

# Modelos para otimizar
models_to_optimize = ['lightgbm', 'xgboost']

optimized_models = {}

for model_name in models_to_optimize:
    print(f"\n🎯 Otimizando {model_name.upper()}...")

    try:
        # Executar otimização
        results = opt_engine.optimize_model(
            model_name, X_train, y_train, X_val, y_val
        )

        # Treinar modelo final com melhores parâmetros
        best_params = results['best_params']

        model = registry.get_model(model_name, **best_params)
        final_model, final_metrics = train_model(
            model, X_train, y_train, X_val, y_val, f"{model_name}_optimized", config
        )

        # Salvar modelo otimizado
        model_path = save_model(
            final_model, f"{model_name}_optimized", final_metrics, feature_cols, config
        )

        optimized_models[model_name] = {
            'model': final_model,
            'metrics': final_metrics,
            'best_params': best_params,
            'optimization_results': results
        }

        print(f"✅ {model_name} otimizado!")
        print(".4f"
    except Exception as e:
        print(f"❌ Erro na otimização de {model_name}: {e}")
        continue
```

### 4.2 Comparar Modelos Otimizados vs. Padrão

```python
# Comparar otimizados com versões padrão
comparison_data = {}

for model_name in models_to_optimize:
    if model_name in model_metrics and model_name in optimized_models:
        comparison_data[f"{model_name}_base"] = model_metrics[model_name]
        comparison_data[f"{model_name}_optimized"] = optimized_models[model_name]['metrics']

comparison_opt_df = pd.DataFrame(comparison_data).T
print("\n📊 Comparação: Base vs Otimizado")
print(comparison_opt_df[['accuracy', 'f1', 'precision']].round(4))
```

## Passo 5: Treinar Modelo Ensemble

### 5.1 Configurar Ensemble

```yaml
# Adicionar ao axon.cfg.yml
ensemble:
  ensemble_type: 'weighted'
  combination_strategy: 'performance_based'
  base_models: ['lightgbm', 'xgboost', 'catboost']
  voting_type: 'soft'
  cv_folds: 5
  regime_detection: true
  regime_window: 50
```

### 5.2 Treinar Ensemble

```python
# Treinar ensemble
ensemble_config = config.get('models', {}).get('ensemble', {})
ensemble = registry.get_model('ensemble', **ensemble_config)

print("🏗️ Treinando Ensemble...")
ensemble_model, ensemble_metrics = train_model(
    ensemble, X_train, y_train, X_val, y_val, 'ensemble', config
)

# Salvar ensemble
ensemble_path = save_model(
    ensemble_model, 'ensemble', ensemble_metrics, feature_cols, config
)

print("✅ Ensemble treinado!")
print(".4f"
```

### 5.3 Comparar com Modelos Individuais

```python
# Adicionar ensemble à comparação
all_metrics = model_metrics.copy()
all_metrics['ensemble'] = ensemble_metrics

final_comparison = pd.DataFrame(all_metrics).T
print("\n🏆 Comparação Final (Incluindo Ensemble):")
print(final_comparison[['accuracy', 'f1', 'precision', 'auc']].round(4))

# Comparar ensemble vs melhor individual
individual_best = final_comparison.drop('ensemble')['accuracy'].max()
ensemble_accuracy = final_comparison.loc['ensemble', 'accuracy']

if ensemble_accuracy > individual_best:
    print("✅ Ensemble melhorou a performance!")
else:
    print("⚠️ Ensemble não melhorou significativamente")
```

## Passo 6: Análise de Features e Interpretabilidade

### 6.1 Importância de Features

```python
from src.models import get_feature_importance

# Analisar importância para cada modelo
feature_importance = {}

for model_name, model in trained_models.items():
    try:
        importance_df = get_feature_importance(model, feature_cols, model_name)
        feature_importance[model_name] = importance_df

        print(f"\n🔍 Top 5 features - {model_name}:")
        for idx, row in importance_df.head(5).iterrows():
            print(".4f"
        # Salvar importância
        importance_df.to_csv(f'outputs/artifacts/{model_name}_feature_importance.csv', index=False)

    except Exception as e:
        print(f"⚠️ Erro ao calcular importância para {model_name}: {e}")
```

### 6.2 Correlação entre Previsões

```python
# Calcular previsões de validação para todos os modelos
predictions = {}

for model_name, model in trained_models.items():
    try:
        if hasattr(model, 'predict_proba'):
            pred_proba = model.predict_proba(X_val)[:, 1]
        else:
            pred = model.predict(X_val)
            pred_proba = pred.astype(float)

        predictions[model_name] = pred_proba
    except Exception as e:
        print(f"⚠️ Erro ao fazer previsões com {model_name}: {e}")

# Calcular matriz de correlação
pred_df = pd.DataFrame(predictions)
correlation_matrix = pred_df.corr()

print("\n📈 Correlação entre previsões dos modelos:")
print(correlation_matrix.round(4))

# Correlação baixa = diversidade boa para ensemble
avg_correlation = correlation_matrix.mean().mean()
print(".4f"
if avg_correlation < 0.8:
    print("✅ Boa diversidade entre modelos!")
else:
    print("⚠️ Modelos muito correlacionados - ensemble pode não ajudar muito")
```

## Passo 7: Validação Cruzada

### 7.1 Executar CV para Robustez

```python
from src.models import cross_validate_model

# Executar CV para os melhores modelos
cv_results = {}

for model_name in ['lightgbm', 'xgboost', 'ensemble']:
    if model_name in trained_models or model_name == 'ensemble':
        model = trained_models.get(model_name, ensemble_model)

        print(f"\n🔄 Validação Cruzada - {model_name}...")
        cv_metrics = cross_validate_model(model, X_train, y_train, cv=5)

        cv_results[model_name] = cv_metrics

        print(f"  Accuracy: {cv_metrics['cv_accuracy_mean']:.4f} ± {cv_metrics['cv_accuracy_std']:.4f}")
        print(f"  F1: {cv_metrics['cv_f1_mean']:.4f} ± {cv_metrics['cv_f1_std']:.4f}")
```

## Passo 8: Avaliação Final

### 8.1 Executar Avaliação

```bash
# Executar avaliação completa
python -m src.evaluate

# Ou especificamente para um modelo
python -c "
from src.evaluate import run_evaluation
from src.models import load_model

# Carregar melhor modelo
model, _ = load_model('outputs/artifacts/ensemble_20250101_120000.pkl')

# Executar avaliação
results = run_evaluation(model, config)
print('Evaluation results:', results)
"
```

### 8.2 Analisar Resultados

```python
# Carregar resultados de avaliação
import json

with open('outputs/metrics/EV_ensemble_20250101_120000.json', 'r') as f:
    eval_results = json.load(f)

print("📊 Resultados de Avaliação:")
print(f"  Accuracy: {eval_results.get('accuracy', 0):.4f}")
print(f"  Precision: {eval_results.get('precision', 0):.4f}")
print(f"  Recall: {eval_results.get('recall', 0):.4f}")
print(f"  F1-Score: {eval_results.get('f1_score', 0):.4f}")
```

## Passo 9: Relatório Final

### 9.1 Gerar Relatórios

```bash
# Gerar relatório completo
python -m src.report
```

### 9.2 Resumo Executivo

```python
print("="*60)
print("📋 RESUMO EXECUTIVO - MULTI-MODEL AXON")
print("="*60)

print(f"📊 Modelos treinados: {len(trained_models)}")
print(f"🎯 Melhor modelo individual: {best_model}")
print(f"🏆 Accuracy individual: {best_accuracy:.4f}")

if 'ensemble' in all_metrics:
    ensemble_accuracy = all_metrics['ensemble']['accuracy']
    improvement = ((ensemble_accuracy - best_accuracy) / abs(best_accuracy)) * 100
    print(f"🎭 Ensemble Accuracy: {ensemble_accuracy:.4f}")
    print(f"📈 Melhoria do Ensemble: {improvement:.1f}%")

print(f"⏱️  Tempo total de treinamento: ~{(len(trained_models) * 2 + len(optimized_models) * 10):.0f} minutos")
print(f"💾 Modelos salvos em: outputs/artifacts/")
print(f"📈 Relatórios em: outputs/reports/")

print("\n✅ Workflow concluído com sucesso!")
print("Próximos passos:")
print("  1. Revisar relatórios em outputs/reports/")
print("  2. Ajustar parâmetros se necessário")
print("  3. Considerar deploy em produção")
print("  4. Monitorar performance ao vivo")
```

## Dicas Avançadas

### Paralelização de Treinamento

```python
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

def train_single_model(model_name):
    # Lógica de treinamento para um modelo
    pass

# Treinar em paralelo
with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
    results = executor.map(train_single_model, models_to_train)
```

### Monitoramento de Recursos

```python
import psutil
import GPUtil

def log_system_resources():
    # CPU e memória
    cpu_percent = psutil.cpu_percent()
    memory_percent = psutil.virtual_memory().percent

    # GPU
    gpus = GPUtil.getGPUs()
    gpu_memory = gpus[0].memoryUsed if gpus else 0

    print(f"CPU: {cpu_percent}%, RAM: {memory_percent}%, GPU: {gpu_memory}MB")

# Log durante treinamento
log_system_resources()
```

### Configurações por Cenário

```python
# Para desenvolvimento rápido
dev_config = {
    'lightgbm': {'n_estimators': 50, 'learning_rate': 0.1},
    'xgboost': {'n_estimators': 50, 'max_depth': 4}
}

# Para produção
prod_config = {
    'lightgbm': {'n_estimators': 200, 'learning_rate': 0.05},
    'xgboost': {'n_estimators': 200, 'max_depth': 8}
}
```

## Troubleshooting Comum

### Modelo não converge
```python
# Aumentar learning rate
config['models']['lightgbm']['learning_rate'] = 0.1

# Ou reduzir regularização
config['models']['lightgbm']['reg_alpha'] = 0.0
```

### Memória insuficiente
```python
# Reduzir batch size para LSTM
config['models']['lstm']['batch_size'] = 16

# Ou reduzir sequence length
config['models']['lstm']['sequence_length'] = 10
```

### Overfitting
```python
# Aumentar regularização
config['models']['xgboost']['reg_alpha'] = 0.1
config['models']['xgboost']['reg_lambda'] = 1.0

# Ou reduzir complexidade
config['models']['lightgbm']['num_leaves'] = 20
```

---

**🎉 Parabéns!** Você completou o tutorial de múltiplos modelos no AXON.

Para próximos passos, consulte:
- [Guia de Configuração](./configuration-guide.md)
- [API Reference](../api/models-api.md)
- [Demo Interativo](../demo/model-comparison.ipynb)