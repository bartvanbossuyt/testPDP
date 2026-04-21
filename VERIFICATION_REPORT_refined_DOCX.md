# Verification Report: refined_DOCX.txt vs. Base Code PDP Implementation

**Date:** 2026-04-16  
**Scope:** Every verifiable claim in `refined_DOCX.txt` checked against the 18 Python files in `scripts/base code PDP/`.  
**Legend:**  ✅ = Confirmed correct, ⚠️ = Partially correct / nuanced, ❌ = Incorrect or not found in code, ➖ = Purely theoretical / not verifiable against implementation

---

## 1. Introduction (Section 1)

The introduction is entirely conceptual and does not make implementable claims. No code verification needed.

**Verdict:** ➖ Not applicable.

---

## 2. Conceptual Foundations (Section 2)

Also theoretical. The claim that PDP is "movement-first" is a design philosophy, not something verifiable in code.

**Verdict:** ➖ Not applicable.

---

## 3. Basic Definitions (Section 3)

### 3.2 Point [T]

> "A point is the fundamental representational unit in PDP."

✅ **Confirmed.** In the code, each row in the dataset represents one point at one timestamp, identified by `poiID` (av.py line ~145: `Df_dataset = Df_raw[['conID', 'tstID', 'poiID', 'x', 'y']]`).

### 3.3 Indexed point / point-time instance [T]

> "An indexed point is a point explicitly tied to a specific timestamp."

✅ **Confirmed.** Each row in the dataset is a (conID, tstID, poiID) triple — the point is indexed by both its object ID and its timestamp.

### 3.4 Configuration [T]

> "A configuration is the set of indexed points that are jointly compared."

✅ **Confirmed.** `conID` groups configurations. In N_PDP.py (line 64): `group_by_con_id = Df_dataset.groupby('conID')`.

### 3.5 Descriptor [T]

> "A descriptor is a directed reference structure that maps each indexed point in a configuration to a scalar value. A descriptor may coincide with a coordinate dimension, but it may also be defined independently."

⚠️ **Partially confirmed.** The *concept* is correct, but the **implementation only supports coordinate-dimension descriptors** (`'x'` and `'y'`). The code loops over `for dim_id in ['x', 'y']` (N_PDP.py line 75) and extracts values via `vals = Df_tst_id[dim_id].to_numpy()` (line 82). There is **no implementation** of:
- Straight directed descriptors (origin + direction vector)
- Curved descriptors (polylines, arc length)
- Orthogonal descriptors
- Flow-relative descriptors

The document marks straight/curved/orthogonal descriptors as [T] (theoretical core), which is fair, but the **[I] (implementation) tag is never applied** to these — so the document is **self-consistent** here. However, the document does not call out clearly enough that the implementation only supports dimensional descriptors.

### 3.6 Scalarisation [T]

> "Scalarisation is the process by which a descriptor assigns a real-valued position to each indexed point."

⚠️ **Implicit in code, not explicit.** In the code, scalarisation is simply column extraction: `vals = Df_tst_id[dim_id].to_numpy()` (N_PDP.py line 82). There is no explicit `scalarise()` function or general scalar function `s_d`. The concept is valid but the code makes no distinction between scalarisation and "reading a column."

### 3.8 PDP matrix [T]

> "For a fixed descriptor d and a fixed configuration P, the PDP matrix M_d is the n×n matrix whose entry at row i, column j records the qualitative relation."

✅ **Confirmed.** N_PDP.py lines 79–93 build exactly this: `A_inequality_matrix = np.zeros((size, size))` and then fill cells with `0` (<), `1` (=), or `2` (>).

### 3.9 PDP trajectory [C]

> "A PDP trajectory is a temporally ordered PDP representation."

✅ **Confirmed.** The code stores matrices keyed by `(con_id, tst_id)` in `D_inequality[(con_id, tst_id)] = tuple(L_tst_id_dfs)` (N_PDP.py line 147).

---

## 4. Descriptor-Based Formulation (Section 4)

### 4.2 Dimensional descriptors [T]

> "The simplest descriptor coincides with a coordinate dimension."

✅ **Confirmed.** This is exactly what the code does: `for dim_id in ['x', 'y']`.

### 4.3 Straight directed descriptors [T]

> "A descriptor need not coincide with a coordinate axis. One may define a straight descriptor by an origin o and a direction vector v."

➖ **Not implemented.** The document marks this as [T] only, so no code is expected. This is self-consistent.

### 4.4 Curved descriptors [T]

> "PDP therefore also allows descriptors defined as directed polylines."

➖ **Not implemented.** Document marks [T] only. Self-consistent.

### 4.5 Orthogonal descriptors [T]

➖ **Not implemented.** Document marks [T] only. Self-consistent.

### 4.6 Multiple descriptors [T][C]

> "A PDP representation typically uses more than one descriptor. The complete PDP representation of a configuration is the collection of all descriptor-specific matrices."

✅ **Confirmed.** The code produces two matrices — one for x, one for y:
```python
if dim_id == "x":
    Df_con_tst_xineq_yineq.at[new_index, 'xineqID'] = A_inequality_matrix
else:
    Df_con_tst_xineq_yineq.at[new_index, 'yineqID'] = A_inequality_matrix
```
(N_PDP.py lines 97-100). The number of descriptors is fixed at 2 in the implementation.

### 4.7 Descriptor selection [C]

> "Descriptor choice is not merely geometric, but also semantic."

⚠️ **Document is correct conceptually, but the code provides NO mechanism for choosing descriptors.** The descriptors are hardcoded to `'x'` and `'y'`. There is no configuration parameter, no GUI option, and no variable like `av.descriptor_list` to change this. The `av.des = 2` and `av.DD = 2` variables exist but are **never used in N_PDP.py** — they are only used in N_T_Report.py for layout purposes (number of columns in PDF report tables).

---

## 5. Inequality Matrices (Section 5)

### 5.1 Definition of a PDP matrix [T]

> M_d(i,j) = < if s_d(p_i) < s_d(p_j), = if equal, > if greater

✅ **Confirmed, with a specific encoding noted below.** N_PDP.py lines 88-93.

### 5.2 Structural properties of exact PDP [T]

> "For every indexed point p_i, M_d(i,i) = ="

✅ **Confirmed.** When `i == j`, `abs(dj - di) == 0 <= rough` (even when rough=0), so `A_inequality_matrix[i,j] = 1` (equality). The diagonal is always 1.

> "If M_d(i,j) = <, then M_d(j,i) = >"

✅ **Confirmed.** If `dj - di > rough` → cell(i,j) = 0 (<), then `di - dj < -rough` → cell(j,i) = 2 (>). Antisymmetry is preserved.

### 5.5 Matrix indexing and ordering [C][I][V]

> "The currently visible implementation sorts such entities primarily by timestamp, then by object."

⚠️ **Requires nuance.** The code does NOT explicitly sort. It relies on the assumption that DataFrame rows are already ordered by tstID then poiID. The comment in N_PDP.py line 83-84 says: *"Assumes Df_tst_id rows are already ordered by tstID then poi."* The data arrives pre-sorted from the CSV. So:
- **Time-major ordering is assumed** ✅
- **The code does not enforce/sort it** ⚠️ (fragile implementation)

The tick label generation confirms the **intended** order (N_PDP.py line 115):
```python
ticks = [f"c{con_id}_t{tst_id}_d{dim_id}_p{var2}_w{var1}"
         for var1 in range(int(av.window_length_tst))
         for var2 in range(int(av.poi))]
```
This loops window first (timestamp), then poi — confirming **time-major ordering** in the labelling. However, this labelling loop iterates `w` (window) in the outer loop and `p` (poi) in the inner loop, meaning tick labels go: `(t0,p0), (t0,p1), ..., (t0,pN), (t1,p0), ...` — which is time-major.

The **data itself** (from the CSV) is expected to follow the same order: grouped by tstID, within each tstID sorted by poiID. This comes from the CSV convention, not from explicit sorting in the code.

**Document claim is CORRECT** but the implementation detail that it's based on an **assumption** rather than an **enforced sort** is worth noting.

### 5.6 Exact PDP and numerical implementation [T][C][I]

> "In theoretical exact PDP, equality means true equality. In implementation, only negligible numerical artefacts may be neutralised."

⚠️ **Implementation does NOT handle this.** When `rough_x = 0` and `rough_y = 0` (fundamental PDP), the comparison becomes:
```python
if abs(dj - di) <= 0:  # strict equality
```
This is strict floating-point equality. No epsilon tolerance is applied. So the code **does** implement exact equality, but it also means floating-point precision issues could affect results. The document's concern is valid but the code has no explicit mitigation.

### 5.7 Rough PDP [C][I]

> M_d(i,j) = < if s_d(p_i) < s_d(p_j) - ε_d, = if |...| ≤ ε_d, > if s_d(p_i) > s_d(p_j) + ε_d

❌ **THE FORMULA IN THE DOCUMENT DOES NOT MATCH THE CODE.**

This is the most significant discrepancy found. The document gives three conditions:
1. `<` if `s_d(p_i) < s_d(p_j) - ε_d`
2. `=` if `|s_d(p_i) - s_d(p_j)| ≤ ε_d`
3. `>` if `s_d(p_i) > s_d(p_j) + ε_d`

The **code** (N_PDP.py lines 88-93) does:
```python
if abs(dj - di) <= rough:
    A_inequality_matrix[i, j] = 1      # =
elif dj - di > rough:
    A_inequality_matrix[i, j] = 0      # <
else:
    A_inequality_matrix[i, j] = 2      # >
```

Let's verify mathematically. Setting `di = s_d(p_i)` and `dj = s_d(p_j)`:

**Code logic:**
1. If `|dj - di| ≤ rough` → equals (1)
2. elif `dj - di > rough` → less-than (0) — meaning p_i < p_j
3. else → greater-than (2) — meaning p_i > p_j

**Document formula:**
1. `<` if `s_d(p_i) < s_d(p_j) - ε_d` → i.e., `di < dj - ε` → i.e., `dj - di > ε`
2. `=` if `|di - dj| ≤ ε`
3. `>` if `di > dj + ε` → i.e., `di - dj > ε` → i.e., `dj - di < -ε`

**Result: The logic IS equivalent.** The code checks `abs(dj-di) <= rough` first (covers equality), then `dj-di > rough` (covers `<`), then falls through to `>`. This is **logically equivalent** to the formula in the document. The apparent "else" covers: `abs(dj-di) > rough AND dj-di ≤ rough`, which means `di - dj > rough`.

✅ **CORRECTED: The code and the document formula are mathematically equivalent.** My initial concern was unfounded — the code's three-branch structure is a correct implementation of the documented formula.

### 5.8 Buffer PDP [C][V]

> "For each original point p and each descriptor d: p_d^-, p, p_d^+"

⚠️ **The code implementation partially matches but has structural differences.** See Section 10.2 below for detailed analysis.

---

## 6. Temporal PDP (Section 6)

### 6.1 PDP-S [T]

> "PDP-S: for each descriptor and each timestamp, one matrix is constructed over the relevant points at that timestamp only."

✅ **Achievable.** Setting `av.window_length_tst = 1` produces PDP-S. The code's loop `for tst_id in range(av.tst - (av.window_length_tst - 1))` becomes `range(av.tst)` — one matrix per timestamp.

### 6.2 PDP-D [F][C][I]

> "PDP-D is a sliding-window representation. The k-th window is W_k = {t_k, ..., t_{k+w-1}} with k = 1, ..., N-(w-1)."

✅ **Confirmed.** N_PDP.py lines 68-72:
```python
for tst_id in range(av.tst - (av.window_length_tst - 1)):
    conditions = [Df_con_id['tstID'] == tst_id + i for i in range(av.window_length_tst)]
    mask = np.logical_or.reduce(conditions)
    Df_tst_id = Df_con_id[mask]
```
With `window_length_tst = 3` (default), this creates sliding windows of 3 timestamps.

**One nuance:** The document says `k = 1, ..., N-(w-1)`, but the code uses 0-based indexing: `range(av.tst - (av.window_length_tst - 1))` → `0, 1, ..., N-w`. The number of windows is `N - (w-1)` which matches the document. This is just an index-origin difference.

> "PDP-D compares indexed points across several timestamps inside one matrix."

✅ **Confirmed.** The matrix size is `av.poi * av.window_length_tst` (N_PDP.py line 79), meaning points from all timestamps in the window are compared pairwise.

### 6.3 PDP-G [F][C][I]

> "PDP-G: the temporal window equals the total number of timestamps."

✅ **Achievable.** Setting `window_length_tst = av.tst` produces PDP-G. The loop would produce exactly one window covering all timestamps.

### 6.4 Continuum between PDP-S, PDP-D, PDP-G [C]

> "PDP-S: w=1, PDP-D: 1<w<N, PDP-G: w=N"

✅ **Confirmed.** The single `window_length_tst` parameter controls this continuum. All three are achievable through the same code path.

### 6.5 Indexed entities in temporal PDP [T][C]

> "In PDP-D and PDP-G, the matrix entities are indexed points, that is, point-time instances."

✅ **Confirmed.** When `window_length_tst > 1`, the inequality matrix has size `poi × window_length_tst`, containing entries for every (point, timestamp) pair within the window.

---

## 7. PDP Trajectories (Section 7)

### 7.5 Descriptor consistency [T][C]

> "For direct comparison, the descriptor set must remain fixed."

✅ **Guaranteed by implementation.** Descriptors are hardcoded to `['x', 'y']` and cannot change between runs without code modification.

### 7.6 Object consistency [C]

> "Direct comparison assumes the same number of objects."

✅ **Confirmed.** `av.poi` is set once and used throughout. The matrix size = `poi × window_length_tst` is constant. The distance calculation (N_PDP.py lines 181-217) assumes all configurations have the same `av.poi`.

---

## 8. Distance (Section 8)

### 8.1 Relation encoding [C][I]

> "< → 0, = → 1, > → 2"

✅ **Confirmed.** N_PDP.py lines 88-93:
- `abs(dj-di) <= rough` → 1 (=)
- `dj - di > rough` → 0 (<)
- else → 2 (>)

> "Identical relations differ by 0, adjacent by 1, opposite by 2."

✅ **Confirmed.** |0-0|=0, |0-1|=1, |1-2|=1, |0-2|=2.

### 8.2 Cell-wise distance [T][C]

> δ_d(i,j) = |M_d(i,j) - M'_d(i,j)|

✅ **Confirmed.** N_PDP.py line 193 (for x): `abs_distance_x += np.abs(mat0_x - mat1_x).sum()`

### 8.3 Absolute matrix distance [T][C][I]

> "Only off-diagonal cells are included."

❌ **INCORRECT — the code INCLUDES diagonal cells.**

N_PDP.py line 193: `abs_distance_x += np.abs(mat0_x - mat1_x).sum()`

This sums ALL cells, including the diagonal. Since diagonal cells are always `1` (equality), their difference is always `0`, so this does not affect the result numerically. However, the code does **not** explicitly exclude the diagonal. The result is equivalent, but the implementation does not match the stated principle "only off-diagonal cells are included." The diagonal just happens to contribute 0.

⚠️ **Functionally equivalent but the code doesn't enforce the stated exclusion.** If for any reason a diagonal value were different (shouldn't happen, but still), the code would silently include it.

### 8.4 Maximum matrix distance [T][C]

> D_max = 2·n(n-1)

⚠️ **The code uses a DIFFERENT formula.** N_PDP.py line 197:
```python
denom = (2 * (av.tst - (av.window_length_tst - 1)) *
         ((av.poi * av.window_length_tst) * (av.poi * av.window_length_tst) - 
          (av.poi * av.window_length_tst)) / 100)
```

Let's unpack. Let `n = poi * window_length_tst` (matrix size), `T = tst - (window_length_tst - 1)` (number of windows). Then:
```
denom = 2 * T * (n² - n) / 100
      = 2 * T * n(n-1) / 100
```

So the denominator is `2·T·n(n-1)/100`. Comparing to the document's D_max = 2n(n-1):
- The document gives the formula for **one matrix**
- The code divides by the sum over **all T windows**, multiplied by 100 for percentage

This is consistent with:
1. Aggregate absolute distances over all T windows
2. Normalise by `T × D_max_per_matrix`
3. Multiply by 100 for percentage scale

### 8.5 Normalised distance [C][I]

> "D_d^% = 100 · D_d^norm"

✅ **Confirmed.** The `/100` in the denominator and the fact that the final distance is in `int(round(..., 0))` means the output is on a [0, 100] scale. The code stores normalised percentage distance.

### 8.6 Distance across descriptors [C]

> "The overall matrix distance is the arithmetic mean of the descriptor-specific distances."

✅ **Confirmed.** N_PDP.py line 219:
```python
A_rel_distance_matrix = np.round(
    (A_rel_distance_matrix_x + A_rel_distance_matrix_y) / 2
).astype(int)
```
This is exactly `(D_x + D_y) / 2`.

### 8.7 Distance between temporal PDP representations [C][I]

> "Distance is computed by aggregating corresponding matrix distances over the temporal sequence."

✅ **Confirmed.** N_PDP.py lines 186-196 iterate over all windows:
```python
for tst_id in range(av.tst - (av.window_length_tst - 1)):
    ...
    abs_distance_x += np.abs(mat0_x - mat1_x).sum()
```
The distances are summed across all temporal windows before normalisation.

---

## 9. Pattern Detection (Section 9)

### 9.5 Clustering workflow [C][I]

> "1. raw trajectory data → PDP representations → distances → clustering → representatives → interpretation"

✅ **Confirmed.** The pipeline is: N_Moving_Objects.py → N_PDP.py → N_VA_HClust.py. The code explicitly performs:
1. Data loading (av.py, N_Moving_Objects.py)
2. PDP transformation (N_PDP.py — inequality matrices)
3. Distance computation (N_PDP.py — distance matrices)
4. Clustering (N_VA_HClust.py — hierarchical clustering via `scipy.cluster.hierarchy`)

### 9.6 Pattern representative: medoid [C]

> "The canonical representative is the medoid."

⚠️ **Not explicitly implemented.** N_VA_HClust.py produces a dendrogram but does **not** compute or select medoids. N_VA_Mds.py and N_VA_TSNE.py produce scatter plots but do not identify medoids. N_VA_TopK.py shows nearest neighbours (which relates to medoid thinking) but does not explicitly label medoids.

The concept is sound but the code does not have a `find_medoid()` function.

### 9.8 Practical analytical outputs [C][I]

> "dendrogram-based clustering, static visualisations, heatmaps, MDS plots, top-k retrieval"

✅ **All confirmed in code:**
- Dendrogram: N_VA_HClust.py ✅
- Static visualisations: N_VA_StaticAbsolute.py, N_VA_StaticRelative.py, N_VA_StaticFinetuned.py ✅
- Heatmaps: N_VA_HeatMap.py ✅
- MDS: N_VA_Mds.py ✅ (also N_VA_Mds_autoencoder.py for autoencoder variant)
- Top-k: N_VA_TopK.py ✅
- t-SNE: N_VA_TSNE.py ✅ (not mentioned in the document's list but present in code)

**Missing from the document's list:** t-SNE, autoencoder dimensionality reduction. These ARE in the code but not mentioned.

---

## 10. Core Extensions: Rough, Buffer, External Points (Section 10)

### 10.1 Rough PDP [C][I]

✅ **Confirmed.** See Section 5.7 analysis above. The implementation correctly applies `rough_x` and `rough_y` thresholds.

The activation logic is correct (N_PDP.py lines 56-58):
```python
if av.PDPg_rough_active == 1 or av.PDPg_bufferrough_active == 1:
    rough_x = av.rough_x
    rough_y = av.rough_y
```
Default values: `rough_x = 30`, `rough_y = 3` (av.py lines 74-75).

### 10.2 Buffer PDP [C][V]

> "For each original point p and each descriptor d: p_d^-, p, p_d^+"

❌ **The implementation differs from the document's description in an important way.**

The document describes buffer as creating **per-descriptor** buffer points: for **each** descriptor d, create `p_d^-` and `p_d^+` — a negative and positive buffer point along that descriptor.

The **actual code** in N_T_OB.py (lines 36-40) creates 5 points per original:
```python
lines.append([conID, tstID, poi*5+0, x - buffer_x, y])     # x-minus
lines.append([conID, tstID, poi*5+1, x + buffer_x, y])     # x-plus
lines.append([conID, tstID, poi*5+2, x, y])                 # original (no change)
lines.append([conID, tstID, poi*5+3, x, y - buffer_y])     # y-minus
lines.append([conID, tstID, poi*5+4, x, y + buffer_y])     # y-plus
```

**Key differences:**
1. The document says 3 points per descriptor (p^-, p, p^+), implying **6 points for 2 descriptors** (or 3 if the original is shared). The code creates **5 points total** — because the original appears once, not twice.
2. The x-buffer points **keep y unchanged**, and the y-buffer points **keep x unchanged**. This means the buffer is **axis-aligned**, not along an arbitrary descriptor direction.
3. The poiID is remapped: `poi * 5 + offset`. This creates a new, expanded set of point IDs.

**The document's abstract notation (p_d^-, p, p_d^+) is consistent** with the implementation if you understand that p appears once, and each descriptor contributes 2 buffer points — giving 1 + 2×2 = 5. But the document could be clearer that the non-varied descriptors of a buffer point retain their original values.

### 10.3 Descriptor-specific parameters [C]

> "Both roughness and buffer should be treated as potentially descriptor-specific."

✅ **Confirmed.** The code uses separate parameters:
- `rough_x` and `rough_y` (av.py lines 74-75)
- `buffer_x` and `buffer_y` (av.py lines 72-73)

### 10.4 External points [C][I][V]

> "External points are fixed reference points... The visible implementation already supports external points by constructing them as fixed entities, replicating them across timestamps."

❌ **NOT found in base code PDP.** There is no code in the base code PDP folder that:
- Creates external points
- Replicates fixed points across timestamps
- Injects environmental landmarks into the point set

The document tags this as [I][V], claiming visible implementation support. **This is NOT present in this codebase.** It may exist in another branch, folder (e.g., CB's code, Jana's code), or an earlier version, but not in `scripts/base code PDP/`.

### 10.5 Combining extensions [C]

> "It is conceptually possible to combine roughness, buffer, and external points."

✅ **Confirmed for rough + buffer.** The code has `PDPg_bufferrough` mode that applies buffer expansion via N_T_OB.py, then roughness via N_PDP.py. External points are not implemented.

### 10.6 Computational implications [C][I][V]

> "Buffer PDP increases the number of represented points."

✅ **Confirmed.** N_T_OB.py expands each point to 5 points, so `poi` grows by 5×.

---

## 11. Environmental Embedding (Section 11)

### 11.4 Environmental embedding through external points [C][I][V]

❌ **Not implemented in base code PDP.** See 10.4 above.

### 11.2 Flows and bifurcations [F][C]

➖ **Not implemented.** Tagged [F][C] (foundational/convention). No code expected. Self-consistent.

### 11.3 Alignment lines and entry/leave points [F][C]

➖ **Not implemented.** Tagged [F][C]. Self-consistent.

### 11.5 Multiple points per vehicle [F][C]

➖ **Not implemented as an explicit mechanism.** The code can technically have multiple poiIDs per real-world object (nothing prevents this), but there is no built-in mechanism to define "this group of poiIDs represents one vehicle."

---

## 12. Inverse PDP (Section 12)

### 12.1-12.5 Conceptual description

✅ **Consistent with N_VA_Inverse.py.** The code:
1. Takes a base configuration (N_C_Dataset.csv)
2. Randomly modifies one point
3. Checks if the new configuration preserves the inequality structure
4. If yes, keeps it; if not, halves the modification and retries

### 12.6 Current inverse heuristic [C]

> "1. choose one indexed point, 2. choose a random direction, 3. move, 4. check PDP constraints, 5. if not, reduce step size (halving), 6. accept when constraints satisfied."

✅ **Confirmed.** N_VA_Inverse.py lines 148-176:
```python
row_in_con = random.randint(1, av.poi * av.tst) - 1
new_selected_point = modify_selected_point(selected_point, av.dim)
...
while (out of bounds):
    old_x_difference = ...
    new_x_difference = old_x_difference / 2  # halving
    ...
```
Then it checks `N_PDP.Df_con_tst_xineq_yineq.equals(Df_new_xineq_yineq)` to verify constraint preservation.

### 12.7 Single-point and multi-point moves [C]

⚠️ **Only single-point moves are implemented.** `row_in_con = random.randint(...)` selects one row (one point-time instance). The code has `if av.dim == 2` and `if av.dim == 3` branches suggesting awareness of dimensionality, but there is NO multi-point or block-translation code.

### 12.8 Rigid block translation [C]

❌ **Not implemented.** The document says this is "the preferred current-state strategy," but the code only modifies one point at a time. There is no block selection, no rigid translation of multiple points.

---

## 13. Implementation Considerations (Section 13)

### 13.2 Matrix index order [I][V]

> "Time-major ordering."

✅ **Confirmed.** See Section 5.5 analysis above. The code assumes rows are ordered by tstID then poiID.

### 13.3 Equality handling [T][C][I]

✅ **Confirmed.** Exact PDP uses `rough=0` (strict equality). Rough PDP uses `rough > 0`.

### 13.4 Buffer semantics [C][V]

> "Point-expansion mechanism."

✅ **Confirmed.** N_T_OB.py expands each point to 5 points.

### 13.5 External-point handling [I][V]

> "The visible implementation already supports external points."

❌ **NOT in base code PDP.** No external point code found.

### 13.6 Temporal implementation [I]

> "The currently visible code clearly supports the practical window-based interpretation."

✅ **Confirmed.** `window_length_tst` parameter controls the sliding window.

---

## 14. Visualisation (Section 14)

### 14.2 Main visualisation families [I][C]

> "descriptor and geometry figures, matrix visualisations, pattern-analysis visualisations, traffic-environment figures"

⚠️ **Partial implementation:**
- **Geometry/descriptor figures:** ✅ N_VA_StaticAbsolute.py, N_VA_StaticRelative.py (plot point positions with arrows)
- **Matrix visualisations:** ✅ N_PDP.py inequality matrix plotting (when `N_VA_InequalityMatrices = 1`)
- **Pattern-analysis visualisations:** ✅ N_VA_HClust.py (dendrogram), N_VA_HeatMap.py (heatmap), N_VA_Mds.py (MDS), N_VA_TopK.py (TopK), N_VA_TSNE.py (t-SNE)
- **Traffic-environment figures:** ❌ Not found in base code PDP. N_VA_StaticFinetuned.py has "tennis pitch overlays" in the README description but no traffic-environment-specific visualisation.

### 14.3 Static, dynamic, and interactive [C]

- **Static:** ✅ All N_VA_Static*.py modules
- **Dynamic:** ⚠️ N_VA_DynamicAbsolute.py exists but is **disabled** (comment in N_Moving_Objects.py: "N_VA_DynamicAbsolute is off (had an error; fix later)")
- **Interactive:** ⚠️ GUI.py is a **placeholder** Dash app with no real functionality

---

## ADDITIONAL FINDINGS NOT IN THE DOCUMENT

### 1. Union-Find for identical configurations

The code (N_PDP.py lines 222-237) includes a Union-Find algorithm that groups configurations with distance 0 (identical PDP representations). This is not discussed in the document but IS a meaningful analytical feature.

### 2. Filtered datasets

N_PDP.py lines 238-249 export filtered datasets for groups of identical configurations and conversion mappings. Not discussed in the document.

### 3. `av.dim` variable and 3D support hint

N_VA_Inverse.py contains a branch for `av.dim == 3` (line 94-95):
```python
if av_dim == 3:
    new_point[5] += round(random.uniform(T_min, T_max), 2)
```
This suggests awareness of potential 3D descriptors, but it falls through to `print("Adapt the code for dimensions other than 2")` at line 177. Not discussed in the document.

### 4. `av.des`, `av.DD`, `av.dim` variables

av.py defines `DD = 2`, `des = 2`, `dim = 2`. These suggest parametric descriptor count was *intended* but is not actually used in the core PDP computation. `DD` is only used in N_T_Report.py for PDF layout.

### 5. t-SNE visualisation

N_VA_TSNE.py provides t-SNE dimensionality reduction. Not mentioned in the document's list of analytical outputs (Section 9.8) but IS implemented.

### 6. Autoencoder dimensionality reduction

N_VA_Mds_autoencoder.py provides neural-network-based dimensionality reduction using TensorFlow autoencoders. Not mentioned in the document.

---

## SUMMARY TABLE

| Section | Claim | Status | Notes |
|---------|-------|--------|-------|
| 3.2-3.4 | Points, indexed points, configurations | ✅ | Correctly reflected in data model |
| 3.5 | Descriptors can be non-coordinate | ⚠️ | True in theory [T], NOT in implementation |
| 4.2 | Dimensional descriptors | ✅ | Only type implemented |
| 4.3-4.5 | Straight/curved/orthogonal descriptors | ➖ | [T] only, not implemented, self-consistent |
| 4.6 | Multiple descriptors | ✅ | Fixed at 2 (x, y) |
| 4.7 | Descriptor selection mechanism | ⚠️ | No runtime mechanism; hardcoded |
| 5.1 | PDP matrix definition | ✅ | Correct |
| 5.2 | Structural properties | ✅ | Diagonal=1, antisymmetry holds |
| 5.5 | Time-major ordering | ✅ | Assumed, not enforced |
| 5.7 | Rough PDP formula | ✅ | Code matches formula |
| 5.8 | Buffer PDP description | ⚠️ | 5 points (not 2×descriptors+1 per descriptor) |
| 5.9 | External points | ❌ | Not in base code PDP |
| 6.1 | PDP-S (w=1) | ✅ | Achievable |
| 6.2 | PDP-D (sliding window) | ✅ | Default mode |
| 6.3 | PDP-G (w=N) | ✅ | Achievable |
| 6.4 | S/D/G continuum | ✅ | Single parameter controls |
| 8.1 | Relation encoding <→0, =→1, >→2 | ✅ | Correct |
| 8.3 | Off-diagonal only | ⚠️ | Code includes all; equivalent because diag=0 contribution |
| 8.5 | Normalised distance [0,100] | ✅ | Correct |
| 8.6 | Arithmetic mean across descriptors | ✅ | `(x + y) / 2` |
| 8.7 | Temporal aggregation | ✅ | Sum over windows then normalise |
| 9.5 | Clustering workflow | ✅ | Full pipeline present |
| 9.6 | Medoid representative | ⚠️ | Not explicitly implemented |
| 9.8 | Analytical outputs list | ⚠️ | Missing t-SNE and autoencoder from list |
| 10.1 | Rough PDP | ✅ | Correctly implemented |
| 10.2 | Buffer PDP | ⚠️ | Implementation differs in structure |
| 10.3 | Descriptor-specific parameters | ✅ | Separate rough_x/y, buffer_x/y |
| 10.4 | External points | ❌ | Not implemented |
| 12.6 | Inverse heuristic | ✅ | Single-point + halving confirmed |
| 12.7 | Multi-point moves | ❌ | Only single-point implemented |
| 12.8 | Rigid block translation | ❌ | Not implemented |
| 13.2 | Time-major ordering | ✅ | Confirmed |
| 13.5 | External point handling | ❌ | Not found |
| 14.2 | Visualisation families | ⚠️ | No traffic-environment figures |
| 14.3 | Dynamic, interactive viz | ⚠️ | Dynamic disabled; interactive is placeholder |

---

## CRITICAL FINDINGS

### Items marked ❌ (Incorrect or Not Found):

1. **External points (Sections 5.9, 10.4, 11.4, 13.5):** The document claims [I] (implementation support) and [V] (verification pending), but **no external point code exists** in base code PDP. The claim "The visible implementation already supports external points" (Section 13.5) is **false for this codebase**.

2. **Rigid block translation in inverse PDP (Section 12.8):** Described as "preferred current-state strategy" but **not implemented**. Only single-point random perturbation exists.

3. **Multi-point moves in inverse PDP (Section 12.7):** Described as a capability but **not implemented**.

### Items marked ⚠️ (Partially Correct):

4. **Descriptor generality:** The document correctly marks general descriptors as [T], but the implementation-level claims ([I] tags) should be more explicit that only dimensional (column-based) descriptors are supported.

5. **Buffer structure:** The 5-point expansion is correct but the document's abstract notation could mislead readers about the actual structure.

6. **Off-diagonal exclusion:** Functionally equivalent but not explicitly enforced.

7. **Medoid:** Mentioned as canonical representative but not implemented.

8. **Missing from document:** t-SNE and autoencoder visualisation modules exist in code but are not mentioned in the analytical outputs list.

---

## RECOMMENDATIONS

1. **Remove or downgrade [I] tag from external points** until the code actually contains this feature in the base code PDP folder.
2. **Downgrade rigid block translation and multi-point inverse** from "current-state" to "planned/future" unless they exist in another codebase version.
3. **Add t-SNE and autoencoder** to the list of practical analytical outputs.
4. **Clarify buffer notation** to show that 5 points are created (not 2k+1 for k descriptors separately).
5. **Mention the Union-Find grouping** of identical configurations — it's a useful feature not documented.
6. **Note that matrix ordering is assumed, not enforced** — this is an implementation fragility worth documenting.
7. **Consider adding an explicit note** that descriptors are currently limited to DataFrame column names (dimensional descriptors only) in the implementation, even though the theory supports arbitrary descriptors.
