# heat_alarm_code_06

- model: `gpt-5.4-mini`
- mode: `heating_like`
- prompt: `outputs/rag/dummy_engineer_runs/heat_alarm_code_06__facility_a_heating_like_rag_context_prompt.md`
- trace: `outputs/rag/dummy_engineer_api_runs/20260625_150159/heat_alarm_code_06_api_trace.json`

## Answer

1. **Executive summary**
- Heating-like mode(`opermode=0`)에서 15-step 합계 예측이 실제보다 크게 낮아 드리프트가 반복됩니다. Top episode에서 실제 164,661.9 vs 예측 84,317.9로 잔차 80,344.0입니다.
- 먼저 heating mode용 매뉴얼의 **압축기 부하/전류, 목표 주파수, 압력 안정성, 온도 변동** 섹션을 찾으세요. [mode_awareness_notes::03], [facility_a_inspection_runbook::02]

2. **Key numeric evidence**
- 대표 구간: 2024-05-01 09:38:26~10:04:20, residual threshold 33,790.2를 크게 초과했습니다.
- 주요 변동: `vi_eev1`(+29.8), `minute_rstd3`(+2.43), `inv1_input_current`(+9.93), `inv1_comp_current_freq`/`target_freq`가 상위 중요도입니다.

3. **Mode interpretation**
- `opermode=0`이므로 heating-like로 해석합니다. 이 모드에서는 0 heat가 자동 이상은 아니지만, 현재는 예측 저하 패턴이 반복됩니다.
- 따라서 zero heat 여부보다 **압축기 전류/주파수와 온도·압력 안정성**을 우선 봐야 합니다. [mode_awareness_notes::03]

4. **Likely hypotheses**
- EEV/밸브 제어(`vi_eev1`) 변동이 커서 열출력 추종이 흔들렸을 가능성.
- 압축기 부하 변화와 목표 주파수 추종 불일치가 예측 드리프트를 만든 가능성.

5. **Recommended inspection sequence**
- 1) 매뉴얼: heating mode의 **compressor current / target frequency / pressure stability / temperature variance** 섹션 확인.
- 2) 현장: inlet/return temperature, compressor load, `inv1_input_current`, `inv1_comp_target_freq`, `inv1_comp_current_freq` 동시 확인.

6. **Limitations**
- 원인 확정은 불가합니다. 제공된 증거는 상관과 우선 점검 항목만 지지합니다.
- 압력/온도 실측값이 없어 이상 여부를 단정할 수 없습니다.

7. **Cost comparison if rates are provided**
- heating-like proxy이므로 전기요금 0.18이 비교 기준입니다. 다만 실제 비용은 15-step 합계와 단가가 함께 있어야 산출 가능합니다.
- 가스요금 0.07은 gas-like 모드 기준입니다. 현재 모드에는 직접 기준이 아닙니다. [facility_a_cost_comparison_guide::02]

8. **Context references**
- [mode_awareness_notes::03]
- [facility_a_inspection_runbook::02]
- [facility_a_mode_faq::02]
- [facility_a_cost_comparison_guide::02]
