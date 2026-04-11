# Multi-Variant PDP Feature

## Overzicht
De inverse.py app ondersteunt nu het **gelijktijdig berekenen en visualiseren van meerdere PDP varianten**. Je kunt zelf bepalen welke varianten je wilt gebruiken en alle resultaten tegelijk bekijken en aan/afzetten.

## Nieuwe Functionaliteit

### 1. **Multi-Select voor PDP Varianten**
In de settings card kun je nu **meerdere varianten tegelijk selecteren**:

```
PDP Variants to calculate: ☑ fundamental ☑ buffer ☑ rough ☐ bufferrough
```

**Voorheen**: Eén variant per keer via dropdown
**Nu**: Meerdere varianten tegelijk via multiselect

### 2. **Automatische Verwerking van Alle Varianten**
Wanneer je meerdere varianten selecteert en op "Start Animation" of "Generate Configurations" klikt:

1. De app genereert eerst alle configuraties voor **fundamental**
2. Dan automatisch alle configuraties voor **buffer**
3. Dan automatisch alle configuraties voor **rough**
4. Enzovoort voor alle geselecteerde varianten

**Voorbeeld**: 
- Geselecteerd: `fundamental`, `buffer`, `rough`
- Aantal configuraties: `3`
- **Totaal gegenereerd**: 9 configuraties (3 per variant)

### 3. **Real-time Variant Indicator**
Tijdens de animatie zie je in de status text welke variant nu wordt verwerkt:

```
Variant 1/3 (fundamental) | Config 2 | Iteration 1 | Step 3
Variant 2/3 (buffer) | Config 1 | Iteration 2 | Step 1
Variant 3/3 (rough) | Config 3 | Iteration 3 | Step 5
```

### 4. **Variant-Specifieke Visualisatie**
Elke gegenereerde configuratie bevat nu variant-informatie in de legend:

**Voorheen**:
```
Config 1 (k)
Config 1 (l)
Config 2 (k)
Config 2 (l)
```

**Nu**:
```
fundamental C1 (k)
fundamental C1 (l)
fundamental C2 (k)
fundamental C2 (l)
buffer C1 (k)
buffer C1 (l)
buffer C2 (k)
buffer C2 (l)
rough C1 (k)
rough C1 (l)
```

### 5. **Interactieve Variant Filtering**
Boven de visualisatie verschijnt een nieuwe filter sectie (alleen als er meerdere varianten zijn):

```
Filter by PDP Variant:
Show configurations for variants: ☑ fundamental ☑ buffer ☑ rough
```

Je kunt **real-time** selecteren welke varianten je wilt zien:
- **Alles aan**: Zie alle configuraties van alle varianten
- **Alleen fundamental**: Zie alleen fundamental configuraties
- **fundamental + buffer**: Vergelijk deze twee varianten direct

### 6. **Slimme Parameter Weergave**
Parameters worden alleen getoond wanneer ze relevant zijn:

**Geen buffer/bufferrough geselecteerd**:
- Buffer X/Y inputs: **Verborgen**

**Geen rough/bufferrough geselecteerd**:
- Roughness X/Y inputs: **Verborgen**

**Beide types geselecteerd**:
- Alle parameters: **Zichtbaar**

## Technische Details

### Datastructuur
Elke configuratie heeft nu een `pdp_variant` veld:

```python
{
    "config_num": 1,
    "points": [...],
    "pdp_variant": "fundamental"  # NEW
}
```

### Variant Tracking in Session State
Nieuwe session_state variabelen:

```python
st.session_state["anim_pdp_variants_list"]      # Lijst van geselecteerde varianten
st.session_state["anim_current_variant_idx"]    # Index van huidige variant (0-based)
st.session_state["anim_current_variant"]        # Naam van huidige variant
```

### Workflow

#### Initialisatie (bij Start Animation):
```python
variants = ["fundamental", "buffer"]  # User selectie
current_variant_idx = 0
current_variant = "fundamental"
```

#### Na completie van alle configs voor 1 variant:
```python
if current_variant_idx + 1 < len(variants):
    # Ga naar volgende variant
    current_variant_idx += 1
    current_variant = variants[current_variant_idx]
    # Reset voor nieuwe variant
    current_config = 1
    completed_iterations = 0
    successful_points = []
else:
    # Alle varianten klaar
    anim_running = False
```

### Visualisatie Logica
```python
# Groepeer per variant
configs_by_variant = {
    "fundamental": [1, 2, 3],
    "buffer": [1, 2, 3],
    "rough": [1, 2, 3]
}

# Filter op user selectie
for variant in selected_variants_viz:
    for config_num in configs_by_variant[variant]:
        # Add trace met label: f"{variant} C{config_num} (k)"
```

## Gebruik Voorbeelden

### Voorbeeld 1: Vergelijk Fundamental vs Buffer
**Doel**: Zie hoe buffer transformatie de matching beïnvloedt

1. Selecteer: `fundamental`, `buffer`
2. Stel buffer in: `Buffer X = 25`, `Buffer Y = 10`
3. Genereer `3` configuraties
4. **Resultaat**: 6 configuraties (3 fundamental + 3 buffer)
5. In visualisatie: Schakel tussen beiden om verschillen te zien

### Voorbeeld 2: Volledige Variant Analyse
**Doel**: Begrijp alle vier de varianten

1. Selecteer: **Alle vier** (`fundamental`, `buffer`, `rough`, `bufferrough`)
2. Parameters:
   - `Buffer X = 25`, `Buffer Y = 10`
   - `Rough X = 5`, `Rough Y = 2`
3. Genereer `1` configuratie per variant
4. **Resultaat**: 4 configuraties (1 per variant)
5. In visualisatie: Gebruik filter om ze één voor één te bekijken

### Voorbeeld 3: Roughness Impact Studie
**Doel**: Zie effect van roughness tolerance

1. Selecteer: `fundamental`, `rough`
2. Roughness: `Rough X = 10`, `Rough Y = 5`
3. Genereer `5` configuraties
4. **Resultaat**: 10 configuraties (5 fundamental + 5 rough)
5. Vergelijk: Zijn rough configuraties "soepeler"?

## UI Workflow

```
┌─────────────────────────────────────────────────┐
│ Settings Card                                    │
├─────────────────────────────────────────────────┤
│ PDP Variant Configuration (select one or more)  │
│                                                  │
│ ☑ fundamental  ☑ buffer  ☐ rough  ☐ bufferrough│
│                                                  │
│ Parameters for selected variants:               │
│ Buffer X: [25.0]    Buffer Y: [10.0]           │
└─────────────────────────────────────────────────┘

↓ Click "Start Animation"

┌─────────────────────────────────────────────────┐
│ Animation Status                                 │
├─────────────────────────────────────────────────┤
│ Variant 1/2 (fundamental) | Config 1 |          │
│ Iteration 1 | Step 2                            │
└─────────────────────────────────────────────────┘

↓ fundamental compleet (alle configs)

┌─────────────────────────────────────────────────┐
│ Animation Status                                 │
├─────────────────────────────────────────────────┤
│ Variant 2/2 (buffer) | Config 1 |               │
│ Iteration 1 | Step 1                            │
└─────────────────────────────────────────────────┘

↓ buffer compleet

┌─────────────────────────────────────────────────┐
│ Visualization of generated configurations       │
├─────────────────────────────────────────────────┤
│ Filter by PDP Variant:                          │
│ ☑ fundamental  ☑ buffer                         │
│                                                  │
│ [Interactive Plotly Chart]                      │
│ Legend:                                          │
│ ☑ Original (k)                                  │
│ ☑ Original (l)                                  │
│ ☐ fundamental C1 (k)    <- click to toggle     │
│ ☐ fundamental C1 (l)                            │
│ ☐ fundamental C2 (k)                            │
│ ☐ fundamental C2 (l)                            │
│ ☐ buffer C1 (k)                                 │
│ ☐ buffer C1 (l)                                 │
│ ☐ buffer C2 (k)                                 │
│ ☐ buffer C2 (l)                                 │
└─────────────────────────────────────────────────┘
```

## Voordelen

### 1. **Efficiëntie**
- Eén keer opstarten → Alle varianten automatisch verwerkt
- Geen handmatig wisselen meer tussen dropdown opties

### 2. **Vergelijking**
- Directe visuele vergelijking tussen varianten
- Alle resultaten naast elkaar bekijken
- Filter functie voor gefocuste analyse

### 3. **Flexibiliteit**
- Kies zelf hoeveel varianten je wilt
- Combineer varianten naar behoefte
- Real-time aan/afzetten van elke variant

### 4. **Transparantie**
- Duidelijke labels met variant naam
- Status indicator toont huidige variant
- Overzichtelijke organisatie in visualisatie

### 5. **Wetenschappelijk Onderzoek**
- Ideal voor variant sensitivity analysis
- Vergelijk matching criteria systematisch
- Export alle resultaten naar CSV (inclusief variant info)

## Limitaties & Overwegingen

### Performance
- Meer varianten = Meer rekentijd
- 4 varianten × 10 configs = 40 configuraties (kan even duren)
- **Aanbeveling**: Start met 1-3 configuraties per variant voor snelle test

### Visualisatie Complexiteit
- Veel configuraties = Drukke legend
- **Oplossing**: Gebruik variant filter om te focussen
- **Tip**: Start alle traces als 'legendonly' (verborgen)

### CSV Export
- Bevat nu ook variant informatie in labels
- **Format**: `Config fundamental_1 (k)` ipv `Config 1 (k)`

## Toekomstige Uitbreidingen

Mogelijke verbeteringen:
- [ ] Per-variant parameter configuratie (verschillende buffer_x per variant)
- [ ] Variant comparison table met statistieken
- [ ] Export per variant naar aparte CSV files
- [ ] Preset configurations ("vergelijk alles", "alleen tolerance", etc.)
- [ ] Kleur schemes per variant type
- [ ] Parallel processing van varianten (sneller)

---

**Implementatie Datum**: 2025-11-25  
**Feature Request**: "ik wil zelf kunnen bepalen of 1 variant wordt berekend of meerdere"  
**Status**: ✅ Volledig geïmplementeerd en getest
