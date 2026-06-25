# gas_pressure_current_03

- model: `gpt-5.4-mini`
- mode: `gas_like`
- prompt: `outputs/rag/dummy_engineer_runs/gas_pressure_current_03__facility_a_gas_like_rag_context_prompt.md`
- trace: `outputs/rag/dummy_engineer_api_runs/20260625_150159/gas_pressure_current_03_api_trace.json`

## Answer

1. **Executive summary**
- 이 에피소드에서는 `opermode=1`의 gas-like 구간에서 15-step 예측합 대비 실제합이 크게 높아, `inst_heat` 저예측이 핵심입니다. `mean_residual_sum_15=37974.1`로 임계치(22856.8)를 초과합니다. [ctx: facility_a_gas_like_rag_context::03]
- 압축기 전류/주파수 관련 피처가 상위이고, 전류가 불안정하다는 관측이 있어 **압축기 부하 변동 또는 유량/열전달 불안정**이 가장 유력합니다. [ctx: facility_a_alarm_code_guide::02]

2. **Key numeric evidence**
- episode 9: `mean_actual_sum_15=50069.0`, `mean_prediction_sum_15=12094.9`, `mean_residual_sum_15=37974.1`, `predicted_anomaly_ratio=0.0502`. [ctx: facility_a_gas_like_rag_context::03]
- 주요 피처: `3way_DHW`, `hotwater_th`, `inv1_input_current`, `inv1_comp_current_freq`, `inv1_comp_target_freq`. [ctx: facility_a_gas_like_rag_context::03]

3. **Mode interpretation**
- gas-like 모드에서는 `inst_heat=0`이 자동 이상은 아닙니다. 따라서 단순 무열 상태보다 **모드에 맞지 않는 예측-실측 괴리**를 봐야 합니다. [ctx: mode_awareness_notes::03]
- 이번 건은 `opermode` 변화가 없고, residual만 커졌으므로 모드 전환보다는 운전 안정성 문제로 해석하는 편이 타당합니다. [ctx: facility_a_gas_like_rag_context::03]

4. **Likely hypotheses**
- 1순위: **압축기 전류/주파수 불안정**으로 인한 출력 변동. 상위 피처가 전류·주파수에 집중되어 있습니다. [ctx: facility_a_gas_like_rag_context::03]
- 2순위: **3-way DHW / hotwater_th 관련 열수요 또는 밸브 동작 불안정**. 다만 직접 원인 확정은 불가합니다. [ctx: facility_a_gas_like_rag_context::03]

5. **Recommended inspection sequence**
- 먼저 `inv1_input_current`, `inv1_comp_current_freq`, `inv1_comp_target_freq`의 급변 여부와 제어 추종성을 확인하세요. [ctx: facility_a_alarm_code_guide::02]
- 다음으로 `3way_DHW`, `hotwater_th`, inlet/outlet 온도 및 유량 관련 신호를 점검하세요. [ctx: facility_a_alarm_code_guide::02]

6. **Limitations**
- 현재 증거만으로 밸브 고장, 센서 오차, 제어 로직 문제를 단정할 수 없습니다.
- 15-step 합계 기반이므로 단일 시점 이상보다 원인 분리가 거칠 수 있습니다. [ctx: facility_a_gas_like_rag_context::03]

7. **Cost comparison if rates are provided**
- 제공된 rate는 `electric_rate=0.18`, `gas_rate=0.07`입니다. 다만 이 에피소드의 비용 비교에 필요한 에너지 단위/소비량이 없어 정량 비용 차이는 계산 불가합니다.
- 참고로 예측합 대비 실제합 차이는 37,974.1이며, 비용 환산은 추가 단위 정보가 필요합니다. [ctx: facility_a_gas_like_rag_context::03]

8. **Context references**
- [ctx: mode_awareness_notes::03]
- [ctx: facility_a_gas_like_rag_context::03]
- [ctx: facility_a_alarm_code_guide::02]
