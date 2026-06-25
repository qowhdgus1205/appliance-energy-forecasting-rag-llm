# Portfolio Demo Questions

Use these one by one when capturing screenshots for the portfolio.
The goal is to show a realistic engineer question, a grounded retrieval, and a short operational answer.

## Gas-like mode

### Q1
`15-step inst_heat 예측이 평소보다 낮은데 운전비용 추정은 올라갑니다. 먼저 무엇부터 점검해야 합니까?`

Expected answer focus:
- forecast vs cost mismatch
- first inspection order
- gas-like mode interpretation

### Q2
`gas 모드에서 inst_heat가 거의 0에 가깝게 나옵니다. 이게 정상일 수 있는지, 어디부터 확인해야 합니까?`

Expected answer focus:
- zero heat in gas-like mode
- mode-aware threshold
- manual section reference

### Q3
`가스 운전 중 예측 residual이 커지고 압축기 전류가 흔들립니다. 가능한 원인과 점검 순서를 알려주세요.`

Expected answer focus:
- residual growth
- compressor current instability
- inspection sequence

## Heating-like mode

### Q4
`heating 모드에서 다음 15-step inst_heat 예측이 실제 추세보다 높습니다. 현장에서 어떤 순서로 확인해야 합니까?`

Expected answer focus:
- heating-like forecast drift
- site inspection order
- likely root causes

### Q5
`heating 모드에서 15-step heat 합산값 기준 비용이 상승합니다. 비용 비교와 함께 점검 포인트를 짧게 정리해주세요.`

Expected answer focus:
- cost comparison
- summed 15-step forecast
- concise answer

### Q6
`heating 모드에서 예측이 drift하고 알람 패턴이 반복됩니다. 먼저 확인할 매뉴얼 섹션과 현장 점검 항목은 무엇입니까?`

Expected answer focus:
- manual section selection
- alarm code guide
- inspection runbook

## Screenshot tips

- Ask one question at a time.
- Keep the retrieved context visible in the prompt.
- Capture the prompt file path and the answer together if possible.
- Prefer Q1 or Q4 as the first screenshot because they show the main story clearly.
