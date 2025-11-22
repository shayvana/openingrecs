# Opening Complexity Metric

## TL;DR

We use **degree centrality** from the filtered network for complexity scores, NOT the raw EFC algorithm scores.

## Why Not EFC Scores?

The Economic Fitness & Complexity (EFC) algorithm from the Nature paper produces counterintuitive results for chess:

### The Problem

| Opening | Players | EFC Score | Chess Reality |
|---------|---------|-----------|---------------|
| Sicilian Defense | 99,975 | ≈ 0.000 | Very complex, deep theory |
| Queen's Pawn, Mengarini Attack | 2 | 144.000 | Obscure, simple |

**EFC measures rarity, not complexity:**
- Rare openings (few players) → HIGH EFC score
- Popular openings (many players) → LOW EFC score

This works for economics (rare exports = sophisticated), but fails for chess (popular openings are often deeply studied and complex).

## Our Solution: Network Degree Centrality

Instead, we calculate complexity from the **filtered relatedness network**:

```python
complexity = 0.3 + (degree_normalized * 0.7)
```

Where:
- `degree` = number of connections in the filtered network
- Filtered network only includes statistically significant edges (z-score > 2.0)
- Normalized to [0.3, 1.0] range

### Interpretation

| Range | Label | Meaning |
|-------|-------|---------|
| 0.3-0.4 | Beginner-friendly | Few connections, niche opening |
| 0.4-0.6 | Intermediate | Moderate connections, standard opening |
| 0.6-0.8 | Advanced | Many connections, strategic depth |
| 0.8-1.0 | Expert | Highly connected, complex theory |

### Why This Works Better

**More connections = more complexity** because:
1. Connected to diverse openings → transferable patterns
2. Played by players with varied repertoires → requires broad understanding
3. Part of larger strategic networks → deeper positional ideas
4. Survived z-score filtering → statistically significant relationships

Example results:
- **Benoni Defense**: 63 connections → complexity = 1.0 (Expert)
- **English Opening**: 41 connections → complexity = 0.74 (Advanced)
- **Horwitz Defense**: 28 connections → complexity = 0.59 (Intermediate)

## EFC Algorithm Status

The EFC algorithm implementation in `algorithms/efc.py` is **correct** per the Nature paper, but we:
1. ✅ Keep it for methodology compliance
2. ✅ Calculate it and store in metadata
3. ❌ Don't use it for UI/recommendations
4. ✅ Use degree centrality instead

## References

- Nature paper: [Prata et al. (2023)](https://www.nature.com/articles/s41598-023-31658-w)
- Original EFC: Tacchella et al. (2012) - Economic Complexity
- Our implementation: `app/app.py:55-85`
