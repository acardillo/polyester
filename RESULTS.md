# Polystore Benchmark Results

## Key findings

| Finding | Store |
| --- | --- |
| Fastest | bm25 |
| Most accurate | hybrid |
| Semantic queries (wins) | vector (10/14) |
| Keyword queries (wins) | vector (3/3) |
| Structural queries (wins) | vector (3/3) |

## Average

| Store | Avg (ms) | Median (ms) | Precision@5 | Recall@5 |
| --- | --- | --- | --- | --- |
| vector | 3.41 | 3.32 | 21.00% | 60.00% |
| graph | 3.80 | 0.25 | 16.00% | 47.50% |
| bm25 | 0.69 | 0.73 | 17.00% | 43.33% |
| hybrid | 4.40 | 4.37 | 25.00% | 70.83% |

## Semantic

| Store | Precision | Recall | Avg (ms) |
| --- | --- | --- | --- |
| vector | 21.43% | 42.86% | 3.43 |
| graph | 15.71% | 32.14% | 5.33 |
| bm25 | 20.00% | 40.48% | 0.72 |
| hybrid | 27.14% | 58.33% | 4.44 |

## Keyword

| Store | Precision | Recall | Avg (ms) |
| --- | --- | --- | --- |
| vector | 20.00% | 100.00% | 3.26 |
| graph | 13.33% | 66.67% | 0.19 |
| bm25 | 20.00% | 100.00% | 0.57 |
| hybrid | 20.00% | 100.00% | 4.17 |

## Structural

| Store | Precision | Recall | Avg (ms) |
| --- | --- | --- | --- |
| vector | 20.00% | 100.00% | 3.44 |
| graph | 20.00% | 100.00% | 0.28 |
| bm25 | 0.00% | 0.00% | 0.67 |
| hybrid | 20.00% | 100.00% | 4.43 |