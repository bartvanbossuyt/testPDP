# PDP Variants Integration in inverse.py

## Overview
All four PDP (Pairwise Distance Patterns) variants from the tennis_infer_rf codebase have been successfully integrated into inverse.py. The app now supports the complete PDP methodology for spatiotemporal pattern matching.

## PDP Variants

### 1. **Fundamental** (Basic PDP)
- **Description**: Core PDP algorithm with no tolerance
- **Use Case**: Exact pattern matching
- **Parameters**: None (uses roughness = 0.0)
- **Logic**: Compares inequality matrices directly

### 2. **Buffer** (Spatial Tolerance)
- **Description**: Applies spatial buffer zones around each point
- **Use Case**: When spatial measurement uncertainty exists
- **Parameters**: 
  - `buffer_x`: Buffer distance in x-direction (default: 25.0)
  - `buffer_y`: Buffer distance in y-direction (default: 10.0)
- **Logic**: Each point is expanded into 5 variants:
  - Variant 0: `(x - buffer_x, y)`
  - Variant 1: `(x + buffer_x, y)`
  - Variant 2: `(x, y)` (original)
  - Variant 3: `(x, y - buffer_y)`
  - Variant 4: `(x, y + buffer_y)`

### 3. **Rough** (Equality Tolerance)
- **Description**: Allows tolerance in equality comparisons
- **Use Case**: When small differences should be considered equal
- **Parameters**:
  - `rough_x`: Tolerance for equality in x-direction (default: 0.0)
  - `rough_y`: Tolerance for equality in y-direction (default: 0.0)
- **Logic**: In inequality matrix:
  - If `|point_j - point_i| <= roughness`: considered equal (value = 1)
  - If `point_j - point_i > roughness`: greater than (value = 0)
  - If `point_j - point_i < -roughness`: less than (value = 2)

### 4. **BufferRough** (Combined)
- **Description**: Combines both buffer transformation and roughness tolerance
- **Use Case**: Comprehensive tolerance handling for real-world data
- **Parameters**: All of the above (buffer_x, buffer_y, rough_x, rough_y)
- **Logic**: 
  1. First applies buffer transformation (5 variants per point)
  2. Then computes inequality matrices with roughness tolerance

## Implementation Details

### Core Functions

#### `compute_inequality_matrix(points, dimension, roughness)`
```python
# Computes PDP inequality matrix for one dimension
# Returns NxN matrix with values:
#   0 = greater than (beyond roughness)
#   1 = equal (within roughness tolerance)
#   2 = less than (beyond roughness)
```

#### `apply_buffer_transformation(points, buffer_x, buffer_y)`
```python
# Expands N points into 5*N buffered variants
# Each original point becomes 5 spatial variations
# Returns (5*N, 2) array of buffered coordinates
```

#### `check_pdp_match(..., pdp_variant, buffer_x, buffer_y, rough_x, rough_y)`
```python
# Main matching function supporting all variants
# 1. Combines k and l points
# 2. Applies buffer transformation if needed
# 3. Computes inequality matrices with appropriate roughness
# 4. Compares matrices for equality
# Returns (d1_match, d2_match) for both dimensions
```

### UI Integration

The settings card now includes a new PDP Variant Configuration section with:

1. **PDP Variant Selector**: Dropdown to choose variant
2. **Buffer Parameters** (shown only for buffer/bufferrough):
   - Buffer X: 0.0 - 100.0 (default: 25.0)
   - Buffer Y: 0.0 - 100.0 (default: 10.0)
3. **Roughness Parameters** (shown only for rough/bufferrough):
   - Roughness X: 0.0 - 100.0 (default: 0.0)
   - Roughness Y: 0.0 - 100.0 (default: 0.0)

Parameters are conditionally displayed based on selected variant for cleaner UI.

### Session State Integration

All variant parameters are stored in `st.session_state`:
- `cfg_pdp_variant`: Selected variant name
- `cfg_buffer_x`, `cfg_buffer_y`: Buffer parameters
- `cfg_rough_x`, `cfg_rough_y`: Roughness parameters

These are retrieved in three locations:
1. `update_order_match_flags()`: Real-time matching during animation
2. `generate_exp()`: Batch generation (exponential strategy)
3. Main animation loop: Live matching display

## Consistency with Original Implementation

The integration maintains exact consistency with the original PDP implementation from tennis_infer_rf:

### From N_PDP.py:
- ✅ Inequality matrix computation (values 0, 1, 2)
- ✅ Roughness tolerance logic
- ✅ Dimension-wise comparison (separate x and y)

### From N_T_OB.py:
- ✅ Buffer transformation (5 variants per point)
- ✅ Point index multiplication (original_idx * 5 + variant)
- ✅ Buffer distance application

### From N_Moving_Objects.py:
- ✅ Variant selection workflow
- ✅ Conditional parameter application
- ✅ Sequential processing (buffer → PDP or fundamental → PDP with roughness)

## Usage Example

### Fundamental (Exact Matching)
```
Configuration: c=11
PDP Variant: fundamental
→ Points must match exactly (no tolerance)
```

### Buffer (Spatial Tolerance)
```
Configuration: c=11
PDP Variant: buffer
Buffer X: 25.0
Buffer Y: 10.0
→ Allows spatial measurement uncertainty
→ Each point has ±25 tolerance in x, ±10 in y
```

### Rough (Equality Tolerance)
```
Configuration: c=11
PDP Variant: rough
Roughness X: 5.0
Roughness Y: 2.0
→ Points within 5 units in x or 2 units in y are considered equal
```

### BufferRough (Combined)
```
Configuration: c=11
PDP Variant: bufferrough
Buffer X: 25.0, Buffer Y: 10.0
Roughness X: 5.0, Roughness Y: 2.0
→ Full tolerance: spatial zones + equality tolerance
→ Most permissive matching
```

## Testing

To test the variants:

1. Open the app: `streamlit run inverse.py`
2. Select a configuration (e.g., c=11)
3. Choose a PDP variant from the dropdown
4. Adjust parameters if applicable (buffer/rough variants)
5. Click "Start Animation" or "Generate Configurations"
6. Observe the match indicators (✓/✗) for d₁ and d₂

## Benefits

1. **Flexibility**: Choose appropriate tolerance level for your data
2. **Consistency**: Exact same logic as original research implementation
3. **Transparency**: All parameters visible and adjustable in UI
4. **Validation**: Multiple variants help understand matching sensitivity

## Technical Notes

### Buffer Transformation Details
- Original point at index `i` becomes indices `5*i` through `5*i+4`
- Variants 0 and 1 modify x-coordinate
- Variants 3 and 4 modify y-coordinate
- Variant 2 keeps original coordinates
- This creates spatial tolerance zones

### Roughness Mechanism
- Applied during inequality matrix computation
- Affects equality detection (value = 1 in matrix)
- Separate roughness for each dimension
- Zero roughness = exact comparison (fundamental behavior)

### Performance Considerations
- Buffer transformation increases matrix size by 5x
- Inequality matrix is O(N²) for N points
- Buffered variants: O((5N)²) = O(25N²)
- Rough variant adds minimal overhead (just tolerance checks)

## Future Enhancements

Potential improvements:
- [ ] Asymmetric buffer (different ± values)
- [ ] Dynamic roughness based on point density
- [ ] Visualization of buffer zones
- [ ] Batch comparison of all variants
- [ ] Variant recommendation based on data characteristics

---

**Integration Date**: 2025-11-25  
**Based On**: tennis_infer_rf/SAM_2-main/SAM/SAM/N_PDP.py (v230209)  
**Status**: ✅ Complete and tested
