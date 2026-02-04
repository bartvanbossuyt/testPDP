# Voorstel: Realistische Maneuver-Varianten (Config 2-9)

## Achtergrond
Dit document bevat een doordacht voorstel voor het genereren van realistische varianten van verkeersconfiguraties 2-9. Het doel is om natuurlijke variatie in trajecten te creëren die rekening houdt met:
- Onnauwkeurigheden in positiebepaling (GPS, sensoren)
- Natuurlijke variatie in menselijk rijgedrag
- Verschillende rijsnelheden en rijstijlen
- Laterale positie binnen een rijstrook

## Strategie-overzicht

### Beschikbare Strategies:
1. **exponential**: Halveert de afstand tot match (snelst convergeren, goed voor kleine aanpassingen)
2. **linear**: Vermindert met 10% van maxdist per stap (geleidelijk, voorspelbaar)
3. **binary**: 7-stap binair zoeken (systematisch, efficiënt)

### Beschikbare PDP Variants:
1. **fundamental**: Basis PDP zonder toleranties
2. **buffer**: ±buffer in x en y richting (5x5 grid per punt)
3. **rough**: Gelijkheids-tolerantie binnen afstand
4. **bufferrough**: Combinatie van buffer + rough
5. **realistic**: Buffer alleen op d1 (x/rijrichting) + rough alleen op d2 (y/lateraal)
6. **frenet**: Road-relative coördinaten (s=langs weg, n=perpendiculair)

---

## Voorstel voor Config 2-9

### **Config 2: Eenvoudig inhalen (2 voertuigen)**
**Scenario**: Klassieke inhaalmanoeuvre op rechte weg

**Aanbevolen instellingen:**
- **Strategy**: `exponential`
- **PDP Variant**: `realistic`
- **Buffer X (d1)**: `15.0` meter
  - *Rationale*: Longitudinale variatie door verschillende acceleratie/snelheidskeuzes
- **Rough Y (d2)**: `0.8` meter
  - *Rationale*: Laterale positie binnen rijstrook (standaard strook = 3.5m breed)
- **Iterations**: `20-30`

**Waarom deze keuze?**
- Realistic variant is ideaal voor standaard verkeerssituaties
- Laterale positie (y) varieert minimaal (within lane)
- Longitudinale positie (x) heeft meer variatie door rijstijl

---

### **Config 3: Inhalen met grotere stroken**
**Scenario**: Inhalen op bredere weg (lane_width = 5.0m)

**Aanbevolen instellingen:**
- **Strategy**: `linear`
- **PDP Variant**: `realistic`
- **Buffer X (d1)**: `20.0` meter
  - *Rationale*: Meer ruimte betekent meer variatie in inhaalgedrag
- **Rough Y (d2)**: `1.2` meter
  - *Rationale*: Bredere stroken = meer laterale vrijheid
- **Iterations**: `25-35`

**Waarom deze keuze?**
- Linear strategy voor geleidelijke aanpassingen
- Grotere toleranties vanwege ruimere wegbreedte

---

### **Config 4: Meerdere voertuigen inhalen**
**Scenario**: Complexere inhaalmanoeuvre met mogelijk 3 voertuigen

**Aanbevolen instellingen:**
- **Strategy**: `binary`
- **PDP Variant**: `bufferrough`
- **Buffer X (d1)**: `12.0` meter
- **Buffer Y (d2)**: `0.6` meter
- **Rough X (d1)**: `5.0` meter
- **Rough Y (d2)**: `0.8` meter
- **Iterations**: `30-40`

**Waarom deze keuze?**
- Binary search voor efficiëntie bij complexere scenario
- Bufferrough voor robuuste matching bij meerdere voertuigen
- Kleinere buffer omdat voertuigen dichter op elkaar kunnen zijn

---

### **Config 5: Korte inhaalmanoeuvre**
**Scenario**: Snelle inhaalactie op korte afstand

**Aanbevolen instellingen:**
- **Strategy**: `exponential`
- **PDP Variant**: `realistic`
- **Buffer X (d1)**: `10.0` meter
  - *Rationale*: Korte manoeuvre = minder longitudinale variatie
- **Rough Y (d2)**: `0.6` meter
  - *Rationale*: Strakke laterale controle bij snelle manoeuvre
- **Iterations**: `15-25`

**Waarom deze keuze?**
- Kleinere toleranties voor precisie bij korte manoeuvre
- Exponential convergeert snel naar correcte variant

---

### **Config 6: Langzame inhaalmanoeuvre**
**Scenario**: Geleidelijke inhaalactie over grotere afstand

**Aanbevolen instellingen:**
- **Strategy**: `linear`
- **PDP Variant**: `realistic`
- **Buffer X (d1)**: `25.0` meter
  - *Rationale*: Lange manoeuvre = meer variatie in exacte timing
- **Rough Y (d2)**: `1.0` meter
  - *Rationale*: Bij langzame manoeuvre meer laterale variatie mogelijk
- **Iterations**: `30-40`

**Waarom deze keuze?**
- Linear strategy past bij geleidelijk karakter
- Grotere buffer_x voor timing-variatie

---

### **Config 7: Inhaalmanoeuvre met vertraging**
**Scenario**: Inhalen waarbij voertuig moet remmen/accelereren

**Aanbevolen instellingen:**
- **Strategy**: `exponential`
- **PDP Variant**: `bufferrough`
- **Buffer X (d1)**: `15.0` meter
- **Buffer Y (d2)**: `0.8` meter
- **Rough X (d1)**: `8.0` meter
- **Rough Y (d2)**: `1.0` meter
- **Iterations**: `25-35`

**Waarom deze keuze?**
- Bufferrough om dynamische snelheidsveranderingen te accommoderen
- Grotere rough_x voor variatie in rem-/acceleratiepunten

---

### **Config 8: Agressieve inhaalmanoeuvre**
**Scenario**: Snelle, assertieve rijstijl

**Aanbevolen instellingen:**
- **Strategy**: `exponential`
- **PDP Variant**: `realistic`
- **Buffer X (d1)**: `18.0` meter
  - *Rationale*: Agressief = meer variatie in timing
- **Rough Y (d2)**: `0.5` meter
  - *Rationale*: Strakke laterale controle (dicht op strookrand)
- **Iterations**: `20-30`

**Waarom deze keuze?**
- Realistic voor typisch verkeersgedrag
- Kleinere rough_y (assertieve rijders blijven strak in strook)

---

### **Config 9: Defensieve inhaalmanoeuvre**
**Scenario**: Voorzichtige, ruimte-nemende rijstijl

**Aanbevolen instellingen:**
- **Strategy**: `linear`
- **PDP Variant**: `realistic`
- **Buffer X (d1)**: `20.0` meter
  - *Rationale*: Defensief = langere volgafstanden/inhaalafstanden
- **Rough Y (d2)**: `1.2` meter
  - *Rationale*: Meer laterale ruimte (midden van strook)
- **Iterations**: `25-35`

**Waarom deze keuze?**
- Linear voor geleidelijke, voorspelbare aanpassingen
- Grotere toleranties voor veilige marges

---

## Aanbevolen Test-Protocol

### Fase 1: Eerste Tryout (Configs 2-5)
Start met **Config 2-5** om basispatronen te valideren:

```python
# Test Config 2 (basis inhalen)
Strategy: exponential
Variant: realistic
Buffer X: 15.0
Rough Y: 0.8
Iterations: 25

# Test Config 3 (brede stroken)
Strategy: linear
Variant: realistic
Buffer X: 20.0
Rough Y: 1.2
Iterations: 30

# Test Config 4 (complex)
Strategy: binary
Variant: bufferrough
Buffer X: 12.0, Buffer Y: 0.6
Rough X: 5.0, Rough Y: 0.8
Iterations: 35

# Test Config 5 (kort)
Strategy: exponential
Variant: realistic
Buffer X: 10.0
Rough Y: 0.6
Iterations: 20
```

### Fase 2: Verfijning (Configs 6-9)
Na evaluatie van Fase 1, test **Config 6-9**:

```python
# Test Config 6 (langzaam)
Strategy: linear
Variant: realistic
Buffer X: 25.0
Rough Y: 1.0
Iterations: 35

# Test Config 7 (dynamisch)
Strategy: exponential
Variant: bufferrough
Buffer X: 15.0, Buffer Y: 0.8
Rough X: 8.0, Rough Y: 1.0
Iterations: 30

# Test Config 8 (agressief)
Strategy: exponential
Variant: realistic
Buffer X: 18.0
Rough Y: 0.5
Iterations: 25

# Test Config 9 (defensief)
Strategy: linear
Variant: realistic
Buffer X: 20.0
Rough Y: 1.2
Iterations: 30
```

---

## Evaluatiecriteria

Na het genereren van varianten, evalueer op:

1. **Realisme**: Zien de trajecten er natuurlijk uit?
2. **Variatie**: Is er voldoende diversiteit tussen varianten?
3. **PDP-consistentie**: Behouden varianten dezelfde ordinale relaties?
4. **Iteraties**: Convergeert het algoritme binnen gestelde iteraties?
5. **Edge cases**: Blijven voertuigen binnen strookgrenzen?

---

## Algemene Aanbevelingen

### Buffer X (d1) - Longitudinale richting:
- **Klein (5-10m)**: Voor korte, precieze manoeuvres
- **Gemiddeld (10-20m)**: Voor standaard inhaalmanoeuvres
- **Groot (20-30m)**: Voor lange, geleidelijke manoeuvres

### Rough Y (d2) - Laterale richting:
- **Klein (0.3-0.6m)**: Voor strakke strookpositie (agressief)
- **Gemiddeld (0.6-1.0m)**: Voor normale binnen-strook variatie
- **Groot (1.0-1.5m)**: Voor ruime marges (defensief)

### Strategy Selectie:
- **Exponential**: Snelle convergentie, geschikt voor simpele scenarios
- **Linear**: Voorspelbare stappen, goed voor complexe manoeuvres
- **Binary**: Efficiënt voor systematisch zoeken in groot oplossingsruimte

### PDP Variant Selectie:
- **Realistic**: DEFAULT keuze voor 90% van verkeerssituaties
- **Bufferrough**: Voor zeer dynamische scenarios met veel variatie
- **Frenet**: Voor bochtige wegen (configs 15, 17)

---

## Volgende Stappen

1. **Implementeer Config 2** als proof-of-concept
2. **Valideer output** visueel en numeriek
3. **Itereer parameters** gebaseerd op observaties
4. **Scale up** naar overige configuraties
5. **Documenteer resultaten** voor rapportage

---

**Datum**: 3 februari 2026
**Auteur**: GitHub Copilot AI Assistant
**Versie**: 1.0 (Eerste tryout voorstel)
