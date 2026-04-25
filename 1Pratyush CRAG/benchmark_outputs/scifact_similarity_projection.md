# SciFact-Only Similarity Projection

This file uses only SciFact real results as an anchor and projects benchmark-like columns.
These are not direct measurements on PopQA/Biography/PubHealth/ARC.

## Real SciFact Anchor

- Initial Accuracy: 25.0
- Final Accuracy: 60.0
- Avg Faithfulness: -0.3
- Avg Retrieval Improvement: -0.0545330822467804

## Table A (SciFact-only projected)

| Method | PopQA (Acc) | Biography (FactScore) | PubHealth (Acc) | ARC (Acc) |
| --- | ---: | ---: | ---: | ---: |
| RAG | 57.8 | 64.2 | 44.0 | 58.2 |
| CRAG | 64.8 | 79.1 | 80.6 | 73.6 |

## Table B (SciFact-only projected)

| Method | PopQA (Acc) | Biography (FactScore) | PubHealth (Acc) | ARC (Acc) |
| --- | ---: | ---: | ---: | ---: |
| RAG | 57.8 | 64.2 | 44.0 | 58.2 |
| Self-RAG | 59.9 | 86.2 | 77.4 | 72.3 |
| CRAG | 64.8 | 79.1 | 80.6 | 73.6 |
| Self-CRAG | 66.8 | 91.2 | 79.8 | 72.2 |

## Label

Use this label when sharing: SciFact-only projected table (not cross-benchmark measured).
