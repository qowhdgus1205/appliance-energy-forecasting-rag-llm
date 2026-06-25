# gas_zero_heat_02

- model: `gpt-5.4-mini`
- mode: `gas_like`
- prompt: `outputs/rag/dummy_engineer_runs/gas_zero_heat_02__facility_a_gas_like_rag_context_prompt.md`
- trace: `outputs/rag/dummy_engineer_api_runs/20260625_150159/gas_zero_heat_02_api_trace.json`

## Answer

1. **Executive summary**
- `inst_heat`가 거의 0이어도 **gas-like mode(opermode=1)** 에서는 **자동으로 이상으로 보지 않습니다**. 먼저 모드와 일치하는 동작인지 확인해야 합니다. [03][02]
- 다만 이번 에피소드는 15-step 합계 기준으로 예측 대비 실제가 크게 높아 **잔차가 큽니다**. 모드 불일치보다는 다른 신호 동반 이상을 먼저 봐야 합니다. [Incident]

2. **Key numeric evidence**
- Top episode에서 `mean_actual_sum_15=50069.0`, `mean_prediction_sum_15=12094.8766`, `mean_residual_sum_15=37974.1234`로, 임계값 `22856.8130`을 초과합니다. [Incident]
- `predicted_anomaly_ratio=0.0502`이며, `inst_heat` 변화는 `+2687`입니다. 다만 `opermode` 변화는 0이라 모드 자체는 유지된 것으로 보입니다. [Incident]

3. **Mode interpretation**
- gas-like 모드에서는 `inst_heat=0`이 정상일 수 있으므로, 이 값만으로는 비정상 판단을 하지 않습니다. [03][02]
- 해석 우선순위는 `opermode=1` 유지 여부, 그리고 15-step 합계 잔차입니다. [03]

4. **Likely hypotheses**
- `3way_DHW`, `hotwater_th` 변화가 커서 온수/3-way 경로 영향 가능성이 있습니다. [Incident]
- `inv1_input_current`, `inv1_comp_current_freq`, `inv1_comp_target_freq` 및 rmean3가 상위라서 압축기/인버터 부하 쪽 점검이 필요합니다. [Incident]

5. **Recommended inspection sequence**
- 먼저 `opermode`, `3way_DHW`, `hotwater_th`가 gas-like 기대 동작과 맞는지 확인하세요. [02][03]
- 그다음 `inv1_input_current`, `inv1_comp_target_freq`, `inv1_comp_current_freq`와 최근 3점 평균을 확인한 뒤, heating-like 전환 여부가 없었는지 보세요. [02]

6. **Limitations**
- 제공된 증거만으로 원인 확정은 불가합니다. 유지보수 이력, 밸브 상태, 센서 품질은 없습니다.
- `inst_heat` 단독 이상 여부는 모드 의존적이며, 모드별 임계값을 따로 봐야 합니다. [03][02]

7. **Cost comparison if rates are provided**
- 제공된 `electric_rate=0.18`, `gas_rate=0.07`은 있으나, 이번 질문은 비용 비교용 실제 사용량이 없어 정량 비용 차이는 계산할 수 없습니다.
- 필요 시 15-step 합계 예측/실측 차이를 에너지 단위로 환산해 비교해야 합니다. [Incident]

8. **Context references**
- [03] `facility_a_gas_like_rag_context::03`
- [02] `mode_awareness_notes::03`, `facility_a_mode_faq::02`, `facility_a_inspection_runbook::02`
- [Incident] 사용자 제공 incident evidence
