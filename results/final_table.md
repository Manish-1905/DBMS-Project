# Final evaluation (Part 8): all systems on all 856 ER questions

Qwen2.5-Coder-1.5B (Q4_K_M), temperature 0. Produced with `final_report.py`.

| Metric | S0 | S1 | S2 | S3 | S5 | S7 |
|---|---|---|---|---|---|---|
| system | no schema | full schema | retrieved schema | + value grounding | + query plan | + guard & repair |
| questions | 856 | 856 | 856 | 856 | 856 | 856 |
| syntax accuracy | 86.7% | 85.7% | 85.6% | 85.5% | 90.1% | 90.8% |
| schema accuracy | 36.1% | 64.8% | 70.2% | 69.9% | 82.6% | 82.5% |
| entity accuracy (n=515) | 67.8% | 77.5% | 80.0% | 83.7% | 84.9% | 85.4% |
| execution accuracy | 86.7% | 85.7% | 85.5% | 85.4% | 89.0% | 88.3% |
| **result accuracy** | **32.8%** | **51.5%** | **55.1%** | **56.2%** | **53.9%** | **54.7%** |
| empty-result rate | 38.3% | 17.8% | 14.8% | 13.2% | 6.4% | 4.4% |
| repair success | – | – | – | – | – | 6/214 |
| selective: answered / accuracy | – | – | – | – | – | 73.7% / 73.1% |
| avg latency (s) | 3.03 | 2.70 | 2.55 | 2.40 | 5.03 | 6.73 |
| avg tokens | – | – | 688 | 721 | 2039 | 2493 |

Paired comparison with the previous system (exact McNemar, result accuracy):

| Comparison | only previous correct | only this correct | p |
|---|---|---|---|
| S0 → S1 | 5 | 165 | < 0.001 |
| S1 → S2 | 29 | 60 | 0.001 |
| S2 → S3 | 17 | 26 | 0.222 |
| S3 → S5 | 115 | 95 | 0.190 |
| S5 → S7 | 0 | 7 | 0.016 |

S7 repair by attempt (845 questions with a query): initial 54.7%, +1 repair 55.0%,
+2 repairs 55.4%; 6 fixed, 0 broken; 7.3% still blocked at the end. Guard verdict on
the first attempt: no problem 631 (73% correct); any problem 214 (about 3% correct).

## Schema-change experiment (roadmap 8.4, G3)

Copy of ER with Genre→Category, WATCHED→VIEWED, Movie.title→name; 624 affected
questions, question text unchanged; semantic aliases off in both runs.

| System | Correct | Empty | Returned data |
|---|---|---|---|
| retrieved schema, **outdated** schema text | 0.0% | 59% | 7% |
| retrieved schema, **current** schema text | **27.9%** | 32% | 41% |

Paired: only outdated 0, only current 174, p < 0.001.
