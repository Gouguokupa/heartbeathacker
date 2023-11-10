# HeartBeat Hackers - TinyML 2023 Competetion

# Model Comparison

The below is the comparision for different models we have tested so far.

## Model Comparison Table

| Model Name    | Model Size (KB) | F-beta   | G-score  | Latency (ms)| Score
| ------------- | ---------------- | -------- | -------- | ----------- | -------- |
| IEGMNet      | 29.20            | 0.65525  | 0.3125   |     -       |
| IEGMNetSimple5a    | 20.60            | 0.897  | 0.5625   |     -      |
| MyNet4      | 20.61            | 0.91227  | 0.5625   |     -      |
| IEGMNet6      | 21.86            | 0.94862  | 0.75   |     28.663       | 120.4079|
| **IEGMNet2**      | 19.38            | 0.96029  | 0.8125   |     52.407       | 121.55419106549697 |



# 2022 Model on 2023 hardware Comparison

| Prize |  Model Size (KB)  |  F-beta    |  G-score  | Latency (ms) | Score |
| :------ | :------ | :------ | :------ | :------ | :------ |
|    1st    |    22.94KB    |    	0.972     |        |    2.512ms      |   98.65668426880163     |
|    2nd    |   17.05KB      |    0.946    |        |       4.235ms   |    101.81012860914743    |
|    4th    |    20.60kB      |    0.934     |        |    40.848ms      |   96.08704028617666     |

# 2022 Model on 2022 hardware Comparison
| Prize |  Model Size (KB)  |  F-beta    |  G-score  | Latency (ms) | Score |
| :------ | :------ | :------ | :------ | :------ | :------ |
|    1st    |   26.39    |    	0.972     |        |    1.747      |        |
|    2nd    |   24.48      |    0.946    |        |       1.712   |        |
|    4th    |    35.46    |    0.934     |        |    26.197      |        |