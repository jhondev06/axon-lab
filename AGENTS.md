# AGENTS.md — Guia para Agentes LLM no AXON

Este guia resume o que você precisa para contribuir de forma segura e eficaz no AXON.

- Propósito: regras, padrões e pontos de extensão do pipeline
- Onde ver o estado completo: consulte STATUS.md para panorama e histórico
- Execução do pipeline: python main.py (ou via Docker)

## Arquitetura (visão rápida)
Fluxo: Dados → Features → Modelos → Avaliação → Inteligência → Relatórios

Módulos críticos:
- <mcfile name="dataset.py" path="c:\Users\JHON-PC\Desktop\AXON-V3\src\dataset.py"></mcfile>
- <mcfile name="features.py" path="c:\Users\JHON-PC\Desktop\AXON-V3\src\features.py"></mcfile>
- <mcfile name="models.py" path="c:\Users\JHON-PC\Desktop\AXON-V3\src\models.py"></mcfile>
- <mcfile name="metrics.py" path="c:\Users\JHON-PC\Desktop\AXON-V3\src\metrics.py"></mcfile>
- <mcfile name="report.py" path="c:\Users\JHON-PC\Desktop\AXON-V3\src\report.py"></mcfile>
- <mcfile name="decision.py" path="c:\Users\JHON-PC\Desktop\AXON-V3\src\brains\decision.py"></mcfile>
- <mcfile name="memory.py" path="c:\Users\JHON-PC\Desktop\AXON-V3\src\brains\memory.py"></mcfile>
- <mcfile name="notifier.py" path="c:\Users\JHON-PC\Desktop\AXON-V3\src\brains\notifier.py"></mcfile>
- <mcfile name="tiny_llm.py" path="c:\Users\JHON-PC\Desktop\AXON-V3\src\brains\tiny_llm.py"></mcfile>

Configuração central: <mcfile name="axon.cfg.yml" path="c:\Users\JHON-PC\Desktop\AXON-V3\axon.cfg.yml"></mcfile>

## Regras fundamentais (resumo)
- Nunca hardcode parâmetros; sempre leia de config
- Determinismo: seeds fixas e timestamp para artefatos
- Estrutura de outputs obrigatória (outputs/… com timestamp)
- Logging sempre; não use print() em produção
- Error handling robusto e compatibilidade retroativa

## 🆕 Atualizações para Agentes (Set 2025)
- Export pós-decisão:
  - Orquestrador chama export do bundle após o gate
  - Exporta somente quando `pass: true` em DECISION
  - Referências: <mcfile name="main.py" path="c:\Users\JHON-PC\Desktop\AXON-V3\main.py"></mcfile>, <mcfile name="export.py" path="c:\Users\JHON-PC\Desktop\AXON-V3\src\export.py"></mcfile>
- Mensagens do Telegram (enriquecidas):
  - PASS: modelo/id, Accuracy, Precision, Recall, F1-Score, AUC, capital final (se houver), janela, artifact (se houver)
  - FAIL: inclui thresholds exigidos
  - Manutenção: <mcfile name="decision.py" path="c:\Users\JHON-PC\Desktop\AXON-V3\src\brains\decision.py"></mcfile>, <mcfile name="notifier.py" path="c:\Users\JHON-PC\Desktop\AXON-V3\src\brains\notifier.py"></mcfile>
- Estrutura do DECISION.json:
  - Adicionados (opcionais): `candidate_id`, `artifact`
  - Local: <mcfile name="DECISION.json" path="c:\Users\JHON-PC\Desktop\AXON-V3\outputs\metrics\DECISION.json"></mcfile>
- Operação vigente:
  - Fonte: Dados configuráveis (lookback configurável); manter por período definido para observação
  - Config: <mcfile name="axon.cfg.yml" path="c:\Users\JHON-PC\Desktop\AXON-V3\axon.cfg.yml"></mcfile>

## Dicas rápidas
- Execute pipeline local: `python main.py`
- Docker (build): `docker build -t axon:3.1 .`
- Docker (run): `docker run --rm -v ${PWD}\data:/app/data -v ${PWD}\outputs:/app/outputs axon:3.1`
- Telegram: defina TELEGRAM_BOT_TOKEN e TELEGRAM_CHAT_ID antes de rodar

---
Criado em: Janeiro 2025  
Versão: 1.1 (08 Setembro 2025)  
Próxima revisão: Sempre que houver mudança relevante de pipeline  
Mantenedores: Equipe AXON + Agentes LLM

### Status do tiny_llm
- O módulo tiny_llm é atualmente um stub local para futuras integrações LLM.
- Não participa de decisões, mas pode ser usado para análises automáticas simples.
- Roadmap de integração detalhado em docs/LLM_ROADMAP.md.