# Project Step-by-Step Walkthrough

이 문서는 프로젝트를 처음부터 끝까지 공부하기 위한 실행 흐름 설명이다. README는 포트폴리오 요약이고, 이 문서는 내부 파이프라인을 단계별로 이해하는 데 목적이 있다.

## 1. 프로젝트가 해결하려는 문제

이 프로젝트는 산업용 가전/설비의 시계열 센서 데이터를 사용해 `inst_heat` 전력/열 수요를 예측하고, 예측 결과를 현장 엔지니어가 이해할 수 있는 점검 가이드로 바꾸는 workflow를 보여준다.

핵심 구조는 다음과 같다.

```text
time-series sensor data
  -> TCN forecasting
  -> residual and mode-aware evidence
  -> RAG retrieval over manuals and context documents
  -> LLM answer for field-engineer inspection
  -> saved markdown answers and traces
```

중요한 점은 LLM이 예측 모델을 대체하는 것이 아니라는 것이다. TCN이 수치 예측을 담당하고, RAG와 LLM은 예측 결과를 운영 문서와 연결해 설명 가능한 답변으로 바꾼다.

## 2. 공개 저장소의 범위

이 저장소는 public portfolio release다. 그래서 다음 항목은 의도적으로 포함하지 않는다.

- 원본 raw dataset
- API key
- trained model binary
- private checkpoint
- proprietary identifier

대신 다음 항목을 포함한다.

- 공개 가능한 script 구조
- TCN 예측 모델 코드
- historical best TCN experiment summary
- 예측 trend plot
- synthetic operator manuals
- RAG corpus 생성 코드
- TF-IDF retriever
- OpenAI embedding retriever
- LangChain/LangGraph prompt workflow
- saved LLM answer examples
- retrieval evaluation and failure analysis documents

## 3. 전체 디렉터리 구조

주요 디렉터리는 다음과 같다.

```text
docs/
  manuals/          RAG에서 검색하는 synthetic operator manuals
  knowledge/        모델/운영 해석에 필요한 보조 지식
  retrieval_eval.md RAG 검색 품질 수동평가
  failure_analysis.md
  modeling_decisions.md

outputs/
  reports/          공개 가능한 forecast/RAG context summary
  rag/              saved LLM answer examples

scripts/
  data profiling, forecasting, RAG corpus, retriever, prompt, API 실행 scripts

src/power_forecast_rag/
  reusable Python modules for RAG, LangChain, LangGraph
```

## 4. 데이터 분석과 feature 정리 단계

초기 단계 scripts는 raw data를 읽고, `inst_heat` 예측에 쓸 feature를 정리하기 위한 흐름이다.

대표 scripts:

```text
scripts/00_scan_energy_data.py
scripts/01_prune_energy_features.py
scripts/02_profile_inst_heat.py
```

이 단계의 목적은 다음이다.

- 데이터 파일 구조 확인
- target인 `inst_heat` 분포 확인
- 결측/상수/불필요 feature 제거
- 예측 가능한 sensor feature 후보 정리

공개 저장소에는 raw data가 없으므로, 이 scripts는 전체 구조를 보여주는 용도다. 실제 데이터 파일 없이 end-to-end 재현은 제한된다.

## 5. 예측 모델 개발 흐름

초기에는 baseline과 residual 분석을 위해 tree-based 모델도 사용했다.

대표 scripts:

```text
scripts/03_train_inst_heat_baseline.py
scripts/04_summarize_inst_heat_residuals.py
scripts/05_feature_importance_inst_heat.py
```

하지만 최종 public framing은 TCN이다. 이유는 이 프로젝트가 시계열 예측과 FDE-LLM workflow를 보여주는 포트폴리오이기 때문이다.

최종 TCN 관련 scripts:

```text
scripts/24_train_facility_a_inst_heat_horizon_suite.py
scripts/25_train_facility_a_inst_heat_tcn.py
```

## 6. TCN 모델의 역할

TCN은 Temporal Convolutional Network다. 시계열의 최근 패턴을 convolution으로 학습하고, dilation을 통해 더 넓은 과거 구간을 볼 수 있다.

이 프로젝트에서 TCN은 다음 일을 한다.

- sensor feature history를 입력으로 받음
- short-horizon `inst_heat`를 예측함
- 실제값과 예측값의 차이인 residual을 계산할 수 있게 함
- downstream RAG/LLM workflow에 들어갈 numeric evidence를 제공함

README의 key result는 historical best TCN experiment를 기준으로 한다.

```text
input length: 10
output length: 10
features: 22
energy-area rel. error: 0.0069
```

## 7. energy-area relative error 의미

`energy-area rel. error`는 예측 `inst_heat` curve 아래 면적과 실제 `inst_heat` curve 아래 면적의 차이를 상대 오차로 본 것이다.

개념적으로는 다음과 같다.

```text
abs(area(actual) - area(predicted)) / abs(area(actual))
```

이 metric을 쓰는 이유는 `inst_heat`가 0에 가까운 구간이 있을 수 있기 때문이다. 이런 경우 point-wise MAPE는 분모가 작아져서 비정상적으로 커질 수 있다.

운영 관점에서는 매 시점의 percentage error보다 누적 소비량을 얼마나 비슷하게 따라갔는지가 더 중요할 수 있다. 그래서 README에서는 학습 과정의 내부 loss보다 energy-area relative error를 대표 metric으로 보여준다.

## 8. 예측 trend plot

README의 plot은 TCN prediction artifact에서 대표 holdout segment를 가져와 실제 `inst_heat`와 예측 `inst_heat`를 같이 그린 것이다.

파일:

```text
assets/tcn_inst_heat_prediction_trend.png
```

이 plot의 목적은 숫자 metric만 보여주는 것이 아니라, 예측이 실제 추세를 얼마나 따라가는지 시각적으로 보여주는 것이다.

## 9. residual과 mode-aware evidence

예측 모델만으로는 현장 엔지니어에게 충분하지 않다. 그래서 예측 결과를 운영 evidence로 바꾼다.

관련 개념:

- `actual`: 실제 `inst_heat`
- `prediction`: TCN 예측값
- `residual`: actual과 prediction의 차이
- `mode_label`: gas-like 또는 heating-like 운영 해석
- `top_features`: 예측/잔차 해석에 중요한 feature
- `feature_shifts`: 특정 episode에서 평소 대비 크게 변한 feature
- `cost_inputs`: 전기/가스 비용 비교에 쓰는 assumed rate

mode-aware 해석이 중요한 이유는 gas-like mode와 heating-like mode에서 같은 `inst_heat` 값이라도 의미가 다를 수 있기 때문이다. 예를 들어 gas-like mode에서 near-zero `inst_heat`는 항상 이상이라고 볼 수 없다.

## 10. RAG context summary 생성

예측 결과와 residual 분석은 RAG에서 사용할 수 있는 문서형 context로 정리된다.

대표 outputs:

```text
outputs/reports/facility_a_inst_heat_multioutput_summary.md
outputs/reports/facility_a_gas_like_rag_context.md
outputs/reports/facility_a_heating_like_rag_context.md
```

이 문서들은 모델 evidence를 자연어 문서 형태로 바꾼 것이다. 이후 RAG corpus에 들어가서 operator manual과 함께 검색된다.

## 11. RAG knowledge base

RAG corpus는 synthetic manual과 forecast context summary로 구성된다.

manual examples:

```text
docs/manuals/facility_a_operator_quickstart.md
docs/manuals/facility_a_inspection_runbook.md
docs/manuals/facility_a_mode_faq.md
docs/manuals/facility_a_mode_transition_playbook.md
docs/manuals/facility_a_compressor_diagnostics.md
docs/manuals/facility_a_valve_and_hotwater_control.md
docs/manuals/facility_a_sensor_signal_reference.md
docs/manuals/facility_a_alarm_triage_matrix.md
docs/manuals/facility_a_cost_comparison_guide.md
docs/manuals/facility_a_alarm_code_guide.md
```

이 문서들은 실제 proprietary manual이 아니라 RAG workflow를 보여주기 위한 synthetic 문서다.

## 12. RAG corpus chunk 생성

RAG 검색을 하려면 문서를 chunk 단위로 나누어야 한다.

실행:

```bash
python scripts/19_build_facility_a_mode_rag_corpus.py
```

출력:

```text
outputs/rag/facility_a_mode_chunks.jsonl
```

각 chunk는 다음 정보를 가진다.

```text
chunk_id
doc_id
title
section
source_path
text
```

이 `jsonl` 파일은 generated artifact라서 `.gitignore`에 의해 기본적으로 Git에 포함되지 않는다.

## 13. 기본 검색 방식: TF-IDF retriever

기존 public answer examples는 TF-IDF retriever를 사용했다.

index 생성:

```bash
python scripts/20_build_facility_a_mode_tfidf_index.py
```

TF-IDF의 장점:

- 구현이 단순하다.
- 로컬에서 재현 가능하다.
- API 비용이 없다.
- sensor name, alarm code, exact keyword 검색에 강하다.

TF-IDF의 한계:

- 질문과 문서가 같은 단어를 공유하지 않으면 놓칠 수 있다.
- paraphrase나 의미 기반 검색에는 약하다.

## 14. 새 검색 방식: OpenAI embedding semantic retriever

API 결제를 했기 때문에 OpenAI embedding 기반 semantic retriever를 추가했다.

index 생성:

```bash
export OPENAI_API_KEY="your-openai-api-key"
python scripts/27_build_facility_a_mode_embedding_index.py
```

기본 embedding model:

```text
text-embedding-3-small
```

출력:

```text
outputs/rag/facility_a_mode_embedding_index/
```

내부 파일:

```text
embedding_matrix.npy
embedding_metadata.jsonl
index_config.json
```

이 index는 API로 생성되는 파생 산출물이므로 `.gitignore`에 의해 Git에서 제외된다.

semantic 검색 테스트:

```bash
python scripts/28_query_facility_a_mode_embedding_rag.py \
  --query "heating mode forecast is drifting and alarm patterns repeat"
```

## 15. TF-IDF와 embedding retriever 차이

TF-IDF는 lexical search다. 질문과 문서가 같은 단어를 많이 공유할수록 점수가 높다.

Embedding retriever는 semantic search다. 질문과 문서의 의미가 가까우면 단어가 정확히 같지 않아도 검색될 수 있다.

예를 들어 다음 두 표현은 단어가 다르지만 의미가 비슷하다.

```text
"heat output is lower than expected"
"heating demand degradation"
```

TF-IDF는 이런 경우 약할 수 있지만, embedding retriever는 더 잘 찾을 가능성이 높다.

다만 embedding retriever도 단점이 있다.

- API 비용이 든다.
- corpus가 바뀌면 index를 다시 만들어야 한다.
- 외부 embedding model에 의존한다.
- sensor name처럼 정확한 문자열 매칭이 중요한 경우 TF-IDF가 더 나을 수 있다.

면접에서는 이 tradeoff를 설명하는 것이 중요하다.

## 16. LangChain retriever wrapper

검색 결과를 LangChain workflow에 연결하기 위해 retriever wrapper를 만들었다.

파일:

```text
src/power_forecast_rag/mode_langchain_rag.py
```

주요 class/function:

```text
ModeAwareLGRetriever
OpenAIEmbeddingModeRetriever
build_mode_retriever
MODE_RAG_PROMPT
format_documents_for_prompt
format_incident_evidence
build_query
```

`build_mode_retriever`는 `tfidf`와 `embedding` 중 어떤 retriever를 쓸지 선택한다.

## 17. LangGraph workflow

LangGraph는 prompt 생성 과정을 node 기반 workflow로 정리한다.

파일:

```text
src/power_forecast_rag/mode_langgraph_workflow.py
```

흐름:

```text
load_context
  -> build_query
  -> retrieve_context
  -> build_prompt
```

각 node의 역할:

- `load_context`: mode-specific RAG context JSON을 읽음
- `build_query`: user question과 incident evidence를 합쳐 retrieval query 생성
- `retrieve_context`: TF-IDF 또는 embedding retriever로 관련 chunk 검색
- `build_prompt`: incident evidence와 retrieved context를 LLM prompt로 조립

## 18. prompt suite 생성

여러 engineer question에 대해 prompt를 한 번에 생성한다.

기본 TF-IDF retriever 사용:

```bash
python scripts/24_generate_dummy_engineer_prompt_suite.py
```

embedding retriever 사용:

```bash
python scripts/24_generate_dummy_engineer_prompt_suite.py --retriever embedding
```

질문 목록:

```text
docs/dummy_engineer_questions.json
```

출력:

```text
outputs/rag/dummy_engineer_runs/
```

주요 출력:

```text
manifest.json
summary.md
*_prompt.md
*_trace.json
```

`manifest.json`은 이후 OpenAI API 답변 생성 script가 어떤 prompt를 읽어야 하는지 알려준다.

## 19. LLM API 답변 생성

prompt suite가 준비되면 OpenAI API로 답변을 생성한다.

실행:

```bash
export OPENAI_API_KEY="your-openai-api-key"
python scripts/26_run_dummy_engineer_openai_api.py --model gpt-5.4-mini
```

특정 질문만 실행:

```bash
python scripts/26_run_dummy_engineer_openai_api.py --question-id gas_cost_rise_01
```

출력:

```text
outputs/rag/dummy_engineer_api_runs/<timestamp>/
```

각 답변은 markdown으로 저장되고, API response trace는 JSON으로 저장된다.

## 20. LLM prompt의 grounding rules

LLM prompt는 다음 원칙을 강제한다.

- retrieved context와 incident evidence만 사용한다.
- 관측되지 않은 값이나 maintenance history를 만들지 않는다.
- 원인을 확정적으로 단정하지 않는다.
- gas-like와 heating-like mode 해석을 구분한다.
- context chunk ID를 인용한다.
- 증거가 부족하면 부족하다고 말한다.

이 규칙은 LLM hallucination을 줄이기 위한 장치다.

## 21. RAG 검색 품질 수동평가

문서:

```text
docs/retrieval_eval.md
```

현재 평가는 6개의 대표 engineer question을 사람이 검토한 small manual review다.

평가 기준:

- expected evidence hit rate
- grounded answer rate
- mode-aware answer rate
- unsupported root-cause claim count

이 평가는 production benchmark가 아니라, 포트폴리오에서 RAG가 어떤 근거를 검색하고 답변에 어떻게 쓰는지 설명하기 위한 것이다.

## 22. Failure analysis

문서:

```text
docs/failure_analysis.md
```

이 문서는 프로젝트 한계를 명시한다.

중요한 한계:

- raw data와 private checkpoint가 없으므로 public repo만으로 historical best TCN을 완전 재현할 수 없다.
- public TCN script는 compact demo다.
- manuals는 synthetic이다.
- TF-IDF 평가는 small manual review다.
- embedding retriever는 semantic matching을 개선하지만 API cost와 external dependency가 있다.
- LLM은 inspection priority를 제안해야 하며 definitive root cause를 단정하면 안 된다.

면접에서는 failure analysis를 말할 수 있어야 한다. 프로젝트의 강점뿐 아니라 어디까지 신뢰할 수 있고, 무엇을 더 검증해야 하는지 설명하는 것이 중요하다.

## 23. Git ignore 정책

`.gitignore`는 공개 저장소에 올리면 안 되는 파일과 generated artifact를 제외한다.

제외되는 주요 항목:

```text
.env
.env.*
data/
outputs/models/
*.joblib
*.pkl
*.npz
*.npy
outputs/rag/*_tfidf_index/
outputs/rag/*_embedding_index/
outputs/rag/*.jsonl
outputs/reports/*.csv
outputs/reports/*.json
__pycache__/
*.py[cod]
.ipynb_checkpoints/
```

이 문서처럼 `docs/*.md` 파일은 Git에 포함된다. 공부용 문서, README, evaluation note, failure analysis는 public repo에 올려도 되는 설명 문서다.

## 24. 처음부터 실행한다면

public repo 기준으로 가능한 실행 흐름은 다음과 같다.

설치:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH="$PWD/src:$PYTHONPATH"
```

RAG corpus 생성:

```bash
python scripts/19_build_facility_a_mode_rag_corpus.py
```

TF-IDF index 생성:

```bash
python scripts/20_build_facility_a_mode_tfidf_index.py
```

OpenAI embedding index 생성:

```bash
export OPENAI_API_KEY="your-openai-api-key"
python scripts/27_build_facility_a_mode_embedding_index.py
```

embedding 검색 확인:

```bash
python scripts/28_query_facility_a_mode_embedding_rag.py \
  --query "gas mode heat is near zero but cost is rising"
```

embedding 기반 prompt suite 생성:

```bash
python scripts/24_generate_dummy_engineer_prompt_suite.py --retriever embedding
```

LLM 답변 생성:

```bash
python scripts/26_run_dummy_engineer_openai_api.py --model gpt-5.4-mini
```

## 25. 면접에서 설명할 핵심 문장

다음처럼 설명하면 프로젝트의 구조가 명확하다.

```text
This project separates numeric forecasting from language generation.
A TCN forecasts short-horizon inst_heat demand, and residual/mode evidence is converted into operational context.
Then a RAG layer retrieves synthetic manuals and forecast summaries, and an LLM generates grounded field-engineer inspection guidance.
The baseline retriever is TF-IDF for transparency, and an optional OpenAI embedding retriever adds semantic search for paraphrased engineer questions.
```

한국어로는 이렇게 말할 수 있다.

```text
이 프로젝트는 TCN이 시계열 예측을 담당하고, RAG/LLM은 예측 결과를 현장 엔지니어가 이해할 수 있는 점검 가이드로 바꾸는 구조입니다.
예측 residual, operating mode, cost estimate를 operational evidence로 만든 뒤, synthetic manual과 forecast context를 검색해서 근거 기반 답변을 생성합니다.
기본 검색은 재현 가능한 TF-IDF이고, API 사용 시 OpenAI embedding 기반 semantic retrieval로 확장할 수 있습니다.
```

## 26. 다음 개선 방향

현재 구조에서 더 발전시키려면 다음을 추가하면 좋다.

1. TF-IDF와 embedding retriever의 top-k 비교 평가
2. negative-control 질문 추가
3. answer groundedness rubric 자동 평가
4. public synthetic dataset generator
5. Streamlit 또는 FastAPI demo
6. hybrid retriever, 즉 TF-IDF와 embedding score 결합
7. field engineer feedback loop

이 중 면접 대비 효과가 큰 것은 retrieval evaluation, negative-control 질문, answer-quality rubric이다.
