# The model tournament — every approach, one table

*Walk-forward test years only; winner (\*): **Ensemble: logit + monotone GBM**, selected by the pre-stated rule (one model family for both heads, highest combined AP lift; ties AUROC, then fewer false alarms). Raw AP is not comparable across candidacy frames — the incumbent's gates give it a frame where roughly half the candidate days are already positive — so **AP lift** (AP over its own frame's base rate) is the column that compares. 'Hit rate' = share of alerts that caught a real episode; 'Move 21d after' = median price change in the month after an alert (a good GET OUT is flat-to-negative, a good GET IN positive).*

## GET OUT — calling the top

| Model | AP | AP lift | AUROC | Episodes caught | Hit rate | FA / instr-yr | Lead (d) | Move 21d after |
|---|---|---|---|---|---|---|---|---|
| Hand rules (incumbent) | 0.545 | 1.13× | 0.549 | 22/188 (12%) | 60% | 0.09 | 12 | +0.4% |
| Logistic regression (crowd-only) | 0.124 | 1.19× | 0.572 | 53/218 (24%) | 14% | 0.68 | 17 | +0.6% |
| Monotone GBM (crowd-only) | 0.115 | 1.11× | 0.553 | 50/218 (23%) | 14% | 0.66 | 13 | +1.0% |
| MLP (crowd-only) | 0.115 | 1.11× | 0.532 | 37/218 (17%) | 11% | 0.64 | 15 | +0.6% |
| Ensemble (crowd-only) | 0.124 | 1.19× | 0.549 | 81/218 (37%) | 13% | 1.12 | 16 | +0.9% |
| Logistic regression | 0.201 | 1.93× | 0.688 | 103/218 (47%) | 32% | 0.47 | 11 | +1.8% |
| Monotone gradient boosting | 0.212 | 2.04× | 0.715 | 99/218 (45%) | 26% | 0.57 | 14 | +1.9% |
| Neural network (MLP 16-8) | 0.191 | 1.84× | 0.695 | 93/218 (43%) | 17% | 0.93 | 11 | +1.0% |
| Ensemble: logit + monotone GBM * | 0.255 | 2.45× | 0.717 | 92/218 (42%) | 36% | 0.35 | 16 | -0.2% |

## GET IN — calling the start

| Model | AP | AP lift | AUROC | Episodes caught | Hit rate | FA / instr-yr | Lead (d) | Move 21d after |
|---|---|---|---|---|---|---|---|---|
| Hand rules (incumbent) | 0.147 | 1.20× | 0.550 | 18/198 (9%) | 17% | 0.18 | 27 | +1.6% |
| Logistic regression (crowd-only) | 0.128 | 1.15× | 0.548 | 71/198 (36%) | 16% | 0.81 | 16 | +1.5% |
| Monotone GBM (crowd-only) | 0.125 | 1.13× | 0.544 | 59/198 (30%) | 21% | 0.47 | 19 | +1.8% |
| MLP (crowd-only) | 0.119 | 1.07× | 0.527 | 35/198 (18%) | 15% | 0.42 | 21 | +1.6% |
| Ensemble (crowd-only) | 0.137 | 1.23× | 0.551 | 70/198 (35%) | 23% | 0.49 | 20 | +2.0% |
| Logistic regression | 0.290 | 2.61× | 0.748 | 137/198 (69%) | 35% | 0.52 | 17 | +1.7% |
| Monotone gradient boosting | 0.269 | 2.42× | 0.740 | 122/198 (62%) | 37% | 0.43 | 17 | +2.2% |
| Neural network (MLP 16-8) | 0.225 | 2.03× | 0.709 | 111/198 (56%) | 33% | 0.47 | 18 | +1.5% |
| Ensemble: logit + monotone GBM * | 0.283 | 2.55× | 0.750 | 112/198 (57%) | 50% | 0.23 | 17 | +0.6% |

## Ground-truth sensitivity (the episode definition sweep)

| definition (boom ETF/single, crash ETF/single) | episodes | GET OUT AP | GET OUT capture | GET IN AP | GET IN capture |
|---|---|---|---|---|---|
| GT-old (25/50, 15/30) | 292 | 0.168 | 44/122 (36%) | 0.211 | 57/106 (54%) |
| GT-ADOPTED (20/40, 12/25) | 494 | 0.201 | 103/218 (47%) | 0.29 | 137/198 (69%) |
| GT-loosest (15/30, 10/20) | 779 | 0.264 | 188/346 (54%) | 0.341 | 204/320 (64%) |
