# Investigation: Buffer & Rough Operations on Descriptors in Base Code PDP

## Executive Summary

**YES**, buffer and rough operations **ARE currently descriptor-dependent**, but **ONLY for x and y axes**.

**YES**, it **IS technically possible** to extend the code to support other descriptors beyond x and y, but this requires significant modifications to the data format, pipeline, and core calculation logic.

---

## 1. CURRENT DESCRIPTOR SUPPORT

### 1.1 Data Format (Fixed)
The current system uses a **strict 5 or 6-column CSV format**:

```
conID, tstID, poiID, x, y, [class]
```

**Only two spatial descriptors supported: `x` and `y`**

- **conID**: Configuration ID (for multiple scenarios)
- **tstID**: Timestep ID (sequential from 0, no gaps)
- **poiID**: Point/Object ID (sequential from 0, no gaps)
- **x**: X-coordinate (spatial descriptor #1)
- **y**: Y-coordinate (spatial descriptor #2)
- **class** (optional): Object classification

### 1.2 Buffer Operations (Descriptor-Dependent)

**File**: [N_T_OB.py](scripts/base%20code%20PDP/N_T_OB.py)

Buffer creates **pseudo-points** around each actual point:

```python
# Current implementation - ONLY x and y
buffer_x = av.buffer_x  # 15 (default)
buffer_y = av.buffer_y  # 1 (default)

# For each original point with (poiID) at position (x, y), it creates 5 pseudo-points:
1. (poiID*5 + 0): x - buffer_x              (left buffer)
2. (poiID*5 + 1): x + buffer_x              (right buffer)
3. (poiID*5 + 2): x (no modification)       (original x)
4. (poiID*5 + 3): x, y - buffer_y           (bottom buffer)
5. (poiID*5 + 4): x, y + buffer_y           (top buffer)
```

**Conclusion**: Buffer is **explicitly descriptor-dependent** (x and y), but **hardcoded for only those two**.

### 1.3 Rough Operations (Descriptor-Dependent)

**File**: [N_PDP.py](scripts/base%20code%20PDP/N_PDP.py) (lines ~60-90)

Rough applies a **tolerance threshold** to distance comparisons:

```python
# Current implementation - ONLY x and y
rough_x = av.rough_x  # 30 (default)
rough_y = av.rough_y  # 3 (default)

for dim_id in ['x', 'y']:  # HARDCODED TO ONLY X AND Y
    rough = rough_x if dim_id == 'x' else rough_y
    
    for i in range(size):
        di = vals[i]
        for j in range(size):
            dj = vals[j]
            
            # Three-way inequality:
            if abs(dj - di) <= rough:
                matrix[i,j] = 1        # EQUAL (within tolerance)
            elif dj - di > rough:
                matrix[i,j] = 0        # GREATER THAN
            else:
                matrix[i,j] = 2        # LESS THAN
```

**Conclusion**: Rough is **explicitly descriptor-dependent**, with separate tolerance values for each dimension.

### 1.4 Combined Buffer + Rough

When both are active:
1. **Buffer** creates pseudo-points (spatial expansion)
2. **Rough** applies tolerance in inequality calculations on the buffered dataset
3. Both only work on x and y

---

## 2. TECHNICAL ANALYSIS: CAN WE ADD OTHER DESCRIPTORS?

### 2.1 What Would Need to Change

To support descriptors like `z`, `vx`, `vy`, `speed`, `density`, etc., the following modifications are required:

#### **A) Data Format Modification**

Current (fixed):
```
conID, tstID, poiID, x, y, [class]
```

Proposed (flexible):
```
conID, tstID, poiID, x, y, descriptor_3, descriptor_4, ..., [class]
```

**Impact**: 
- av.py would need to **dynamically** detect number of spatial descriptors
- All scripts referencing `['x', 'y']` hardcoded need refactoring
- Configuration would need to specify which descriptors to use

#### **B) N_T_OB.py Modification (Buffer Creation)**

Current logic:
```python
# Hardcoded for 2 descriptors (x, y) with 5 pseudo-points each
lines.append([conID, tstID, poi*5 + 0, x - buffer_x, y])  # left
lines.append([conID, tstID, poi*5 + 1, x + buffer_x, y])  # right
lines.append([conID, tstID, poi*5 + 2, x, y])              # center
lines.append([conID, tstID, poi*5 + 3, x, y - buffer_y])  # bottom
lines.append([conID, tstID, poi*5 + 4, x, y + buffer_y])  # top
```

If we had **n descriptors**, we'd need **2n+1 pseudo-points** per original point:
```python
# Proposed for n descriptors
for i in range(n_descriptors):
    desc_minus = descriptor[i] - buffer[i]  # Apply buffer -
    desc_plus = descriptor[i] + buffer[i]   # Apply buffer +
    # Create two pseudo-points per descriptor
```

**Impact**: Exponential growth in point cloud size (currently 5x, would become 2n+1x).

#### **C) N_PDP.py Modification (Rough & Distance Calculation)**

Current:
```python
for dim_id in ['x', 'y']:  # HARDCODED
    rough = rough_x if dim_id == 'x' else rough_y
    # Create inequality matrices...
```

Proposed:
```python
descriptors = av.descriptor_list  # ['x', 'y', 'vx', 'vy', ...]
for i, dim_id in enumerate(descriptors):
    rough = av.rough_values[i]  # rough_x, rough_y, rough_vx, rough_vy, ...
    # Create inequality matrices...
```

**Impact**: 
- Need to track N inequality matrices instead of 2
- Storage increases with N descriptors
- Distance matrix computation becomes more complex (combine N dimensions)

#### **D) Configuration (av.py Modification)**

Current:
```python
buffer_x = 15
buffer_y = 1
rough_x = 30
rough_y = 3
```

Proposed:
```python
# Dynamic descriptor support
descriptors = ['x', 'y']  # User-configurable
buffer_values = {'x': 15, 'y': 1, 'vx': 0.5, 'vy': 0.5}
rough_values = {'x': 30, 'y': 3, 'vx': 10, 'vy': 10}
```

---

## 3. FEASIBILITY ASSESSMENT

### 3.1 Is It Possible? 
**YES**, but with significant engineering effort.

### 3.2 Effort Required

| Task | Difficulty | Time Est. |
|------|-----------|-----------|
| Redesign data format | Medium | 2-3 hours |
| Modify av.py for dynamic descriptors | Medium | 2-3 hours |
| Refactor N_T_OB.py (buffer logic) | Medium | 3-4 hours |
| Refactor N_PDP.py (core calculation) | **High** | 5-8 hours |
| Update all visualization scripts | Medium | 4-6 hours |
| Testing & validation | **High** | 8-12 hours |
| **Total** | | **24-36 hours** |

### 3.3 Risks & Considerations

1. **Exponential Complexity**: Buffer creates 2n+1x more points; memory/performance impacts are severe
2. **Distance Metric**: How to combine distances from N descriptors? (current: average of x and y)
3. **Backward Compatibility**: Breaking change - all existing datasets, scripts, reports affected
4. **Visualization**: How to visualize N-dimensional PDP results? (currently shows 2D x-y plots)
5. **Parameter Tuning**: Each new descriptor requires careful buffer_x/buffer_y and rough_x/rough_y tuning

---

## 4. CURRENT LIMITATIONS IN CODE

### 4.1 Hardcoded Limitations

**N_PDP.py line 60-80**: Only loops over x and y
```python
for dim_id in ['x', 'y']:  # <-- HARDCODED
```

**N_T_OB.py lines 25-40**: Buffer only in x-y plane
```python
lines.append([...poiID*5+0, x-buffer_x, y])      # x dimension only
lines.append([...poiID*5+1, x+buffer_x, y])
lines.append([...poiID*5+3, x, y-buffer_y])      # y dimension only
lines.append([...poiID*5+4, x, y+buffer_y])
```

**av.py lines 58-64**: Only accepts 5 or 6 columns (x, y only)
```python
if ncols == 5:
    colnames = ['conID', 'tstID', 'poiID', 'x', 'y']
elif ncols == 6:
    colnames = ['conID', 'tstID', 'poiID', 'x', 'y', 'class']
else:
    raise ValueError(...)  # No other formats allowed
```

### 4.2 Where Dynamic Descriptor Support Would Help

- **IMEC STREAMS Data**: Currently has `x`, `y`, `vx`, `vy`, `YOLO_class`, but only x, y are used
- **LiDAR Data**: Could use 3D coordinates (x, y, z) or intensity values
- **Velocity Analysis**: Could apply buffer/rough to velocity components separately
- **Classification Features**: Could include other tracked features (speed, acceleration, density, etc.)

---

## 5. RECOMMENDATIONS

### 5.1 **Short Term (No Code Changes)**
- Use current x, y descriptor support as-is
- Apply buffer and rough as intended for spatial positioning
- Document current limitations in the README

### 5.2 **Medium Term (Minimal Changes)**
- If you need a 3rd descriptor (e.g., `z` for elevation):
  - Add as 7th column to CSV (conID, tstID, poiID, x, y, z, class)
  - Modify av.py to optionally read 7+ columns
  - Extend N_PDP.py to loop over `['x', 'y', 'z']` with corresponding rough parameters
  - This is **much simpler** than generalizing to arbitrary descriptors

### 5.3 **Long Term (Full Refactor)**
- Design a new **DescriptorPDP** class-based architecture
- Support arbitrary descriptor lists
- Use configurable descriptor combinations
- Implement adaptive distance metric combining (not just mean)

---

## 6. SUMMARY TABLE

| Aspect | Current State | Can Be Extended? | Effort |
|--------|--------------|------------------|--------|
| **Number of descriptors** | 2 (x, y) only | Yes, but requires significant refactoring | High |
| **Buffer on descriptors** | Hardcoded for x, y | Yes, possible to generalize | Medium-High |
| **Rough on descriptors** | Separate thresholds per descriptor | Yes, already designed for flexibility | Medium |
| **Data format** | Fixed 5-6 columns | Yes, but breaking change | Medium |
| **Configuration** | av.py with hardcoded values | Yes, can make dynamic | Low |
| **Backward compatibility** | N/A | Would require migration scripts | Medium |

---

## Conclusion

**Buffer and Rough ARE descriptor-dependent** — they are explicitly designed to work on spatial descriptors with configurable tolerance/expansion parameters. However, they are **limited to the x, y descriptors only** due to hardcoded iteration logic.

**Extending to other descriptors is technically feasible** but requires substantial code refactoring across 5+ core modules. The effort is justified only if:
1. Your data has meaningful additional descriptors (z, velocity, etc.)
2. You need to apply buffer/rough logic to those descriptors
3. You're willing to invest 24-36 hours in refactoring and testing
