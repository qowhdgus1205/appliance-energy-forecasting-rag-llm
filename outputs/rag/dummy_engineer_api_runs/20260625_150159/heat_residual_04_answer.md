# heat_residual_04

- model: `gpt-5.4-mini`
- mode: `heating_like`
- prompt: `outputs/rag/dummy_engineer_runs/heat_residual_04__facility_a_heating_like_rag_context_prompt.md`
- trace: `outputs/rag/dummy_engineer_api_runs/20260625_150159/heat_residual_04_api_trace.json`

## Answer

1. **Executive summary**
- Heating-like mode에서 `inst_heat`의 15-step 예측합이 실제 추세보다 높습니다. 먼저 모드 확인 후, 압축기 부하와 온도/압력 변동을 순서대로 점검하세요. [ctx: facility_a_operator_quickstart::02, mode_awareness_notes::03]
- 이번 episode는 예측합 대비 실제합 차이가 커서, 단일 시점보다 `t+1..t+15` 합 기준으로 봐야 합니다. [ctx: incident evidence]

2. **Key numeric evidence**
- Episode 11: 실제합 `164661.9091`, 예측합 `84317.8742`, 잔차합 `80344.0349`로 임계값 `33790.1552`를 초과합니다. [ctx: incident evidence]
- 예측 이상 비율은 `0.0501`이며, 주요 변화는 `vi_eev1`, `inv1_input_current`, `inv1_comp_*` 계열입니다. [ctx: incident evidence]

3. **Mode interpretation**
- `opermode=0`이므로 heating-like로 해석합니다. 이 모드에서는 zero heat 자체를 이상으로 보지 않습니다. [ctx: incident evidence, facility_a_mode_faq::02]
- 따라서 “열이 낮다/높다”보다 압축기 전류, 목표 주파수, 압력 안정성, 온도 변동을 우선 봐야 합니다. [ctx: mode_awareness_notes::03]

4. **Likely hypotheses**
- 압축기 부하 저하 또는 제어 목표 주파수 추종 문제 가능성. `inv1_input_current`, `inv1_comp_target_freq`, `inv1_comp_current_freq`가 상위 특징입니다. [ctx: incident evidence]
- EEV/온도 제어 변동성 증가 가능성. `vi_eev1`와 `minute_rstd3` 변화가 큽니다. [ctx: incident evidence]

5. **Recommended inspection sequence**
- 1) `opermode=0` 유지 여부와 해당 구간의 heating-like 상태를 먼저 확인. [ctx: facility_a_operator_quickstart::02]
- 2) `inv1_input_current` → `inv1_comp_target_freq` → `inv1_comp_current_freq` 순으로 부하/추종 불일치 확인, سپس inlet/return temperature와 pressure stability, temperature variance 점검. [ctx: facility_a_inspection_runbook::02, mode_awareness_notes::03]

6. **Limitations**
- 원인 확정은 불가합니다. 제공된 증거는 상관된 특징과 잔차만 보여줍니다.
- 유지보수 이력, 실제 센서 원시값, 알람 로그는 없어 추가 진단이 필요합니다.

7. **Cost comparison if rates are provided**
- 제공된 전기/가스 요금은 있으나, 이번 질문에는 에너지 사용량이 없어 비용 차이는 계산할 수 없습니다. [ctx: incident evidence]

8. **Context references**
- `facility_a_operator_quickstart::02`, `facility_a_inspection_runbook::02`, `mode_awareness_notes::03`, `facility_a_mode_faq::02`
