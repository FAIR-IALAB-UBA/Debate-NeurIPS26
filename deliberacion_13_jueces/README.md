# Experimento de Deliberación Multi-Juez — `deliberacion_13_jueces`

Scripts para ejecutar paneles de jueces LLM que deliberan sobre transcripciones de debates estructurados.
Se proporcionan dos variantes, que difieren únicamente en si se instruye a los jueces a llegar a un consenso al final de la Ronda 3.

---

## Scripts

| Script | Instrucción de consenso |
|---|---|
| `run_deliberation_sin_consenso.py` | Los jueces deliberan libremente; cada uno emite un veredicto independiente en la Ronda 3 |
| `run_deliberation_con_consenso.py` | Se indica explícitamente a los jueces que **deben** acordar la misma afirmación en la Ronda 3 |

---

## Diseño experimental

### Jueces

- **13 jueces por debate** (`N_JUDGES = 13`), configurable a 3.
- Cada juez recibe un `general_belief` (mainstream o escéptico) como persona inyectada en el system prompt.
- La composición del panel por debate se controla mediante `majority_type`:
  - `"mainstream"` → 7 mainstream, 6 escépticos
  - `"skeptical"` → 6 mainstream, 7 escépticos
- Las posiciones dentro del panel se asignan en orden aleatorio con semilla (seed = número canónico del debate), de modo que las versiones original e invertida de un mismo debate produzcan la misma alineación de jueces.

### Protocolo de cuatro fases

Cada debate ejecuta las mismas cuatro fases de forma secuencial:

| Fase | Nombre | Qué producen los jueces |
|---|---|---|
| 1 | Creencia previa (pre-debate) | Afirmación + Confianza (0–100) |
| 2 | Evaluación post-debate + Ronda 1 de deliberación | Afirmación + Confianza + Puntos clave |
| 3 | Ronda 2 de deliberación | Respuesta a otros + Temas pendientes |
| 4 | Ronda 3 de deliberación final | Ver tabla de variantes abajo |

**Producción en Ronda 3 según variante:**

| Variante | Los jueces producen |
|---|---|
| `sin_consenso` | Razonamiento final + Afirmación (`Statement 1` / `Statement 2`) + Confianza |
| `con_consenso` | Solo Afirmación (`Statement 1` / `Statement 2` / `No agreement reached`) |

### Compartición de transcripciones

- **Ronda 1 → Ronda 2**: solo se comparten los `Key points` de cada juez (no la afirmación ni la confianza), evitando anclaje en señales numéricas.
- **Ronda 2 → Ronda 3**: se concatenan los puntos clave de la Ronda 1 y las respuestas completas de la Ronda 2; los turnos de la Ronda 2 se presentan en **orden invertido** respecto a la Ronda 1 para mitigar efectos de primacía/recencia.

---

## Entrada

### Dataset — `dataset_deliberation.csv`

Una fila por tema de debate. Columnas requeridas:

| Columna | Descripción |
|---|---|
| `general_topic` | Dominio amplio (e.g., "Supersonic Aircrafts") |
| `specific_topic` | Afirmación debatida (usado como identificador de tema) |
| `statement_1` | Afirmación correcta (siempre verdad) |
| `statement_2` | Afirmación contrafactual |
| `general_belief_mainstream` | Texto de creencia previa para jueces mainstream |
| `general_belief_skeptical` | Texto de creencia previa para jueces escépticos |

### JSONs de debates

Archivos individuales en `debates/` e `inverted_debates/`, con nombre `debate_NNN_*.json`.

Claves JSON requeridas: `debate_id`, `topic`, `statement_1`, `statement_2`, `transcript` (lista de dicts `{debater, argument}`).

`inverted_debates/` contiene los mismos debates con el orden de presentación de argumentos dentro de cada ronda intercambiado — se usa para medir y controlar el sesgo de posición.

---

## Salida

Se ejecutan cuatro runs por script (2 tipos de mayoría × 2 órdenes de debate):

```
deliberaciones_{con,sin}_consenso/
  13_jueces_mayoria_mainstream/
    orden_original/
      delib_NNN_*.json              # un archivo por debate
      deliberaciones_13j_mainstream_original_TIMESTAMP.csv
      error_log.csv
      llm_parsing_log.csv
      llm_parsing_results/
      errors_summary.json
    orden_invertido/
      ...
  13_jueces_mayoria_skeptical/
    ...
```

### Estructura del JSON (`delib_NNN_*.json`)

```json
{
  "deliberation_id": "delib_001_20260429_120000",
  "debate_id": "debate_001_...",
  "topic": "...",
  "majority_type": "mainstream",
  "judges": [{"judge_id", "judge_name", "general_belief", "belief_type"}, ...],
  "phases": {
    "prior_beliefs":  [...],
    "round_1":        [...],
    "round_2":        [...],
    "round_3":        [...]
  },
  "collective_decision": {
    "final_statement": "Statement 1",
    "vote_count": {"Statement 1": 9, "Statement 2": 4},
    "unanimous": false
  },
  "has_errors": false,
  "errors": [],
  "metadata": {...}
}
```

### Columnas del CSV

**Comunes a ambas variantes** — los campos por juez usan el prefijo `jN_` (e.g., `j1_`, `j2_`, …, `j13_`):

| Columna | Descripción |
|---|---|
| `deliberation_id`, `debate_id`, `topic`, `general_topic` | Identificadores |
| `num_judges`, `majority_type` | Configuración del panel |
| `has_errors`, `num_errors` | Indicadores de error de parseo |
| `collective_final_statement` | Ganador por mayoría en Ronda 3 |
| `collective_unanimous` | Si todos los jueces coincidieron en Ronda 3 |
| `collective_vote_statement_1/2` | Conteo de votos |
| `collective_ground_truth_alignment` | Si la decisión colectiva = Statement 1 |
| `jN_judge_name`, `jN_belief_type`, `jN_general_belief` | Identidad del juez |
| `jN_prior_statement/confidence/ground_truth_alignment/response_raw` | Fase 1 |
| `jN_r1_statement/confidence/ground_truth_alignment/key_points/response_raw` | Fase 2 |
| `jN_r2_response_to_others/outstanding_issues/response_raw` | Fase 3 |
| `jN_r3_statement/ground_truth_alignment/response_raw` | Fase 4 |
| `jN_belief_changed_prior_to_r1/r3`, `jN_belief_changed_r1_to_r3` | Indicadores de cambio de creencia |
| `jN_confidence_change_prior_to_r1` | Delta de confianza Fase 1 → Fase 2 |

**Solo `sin_consenso`** (campos adicionales de Ronda 3):

| Columna | Descripción |
|---|---|
| `jN_r3_confidence` | Confianza final |
| `jN_r3_final_reasoning` | Texto del razonamiento final |
| `jN_confidence_change_r1_to_r3` | Delta de confianza Fase 2 → Fase 4 |

**Solo `con_consenso`** (campos adicionales):

| Columna | Descripción |
|---|---|
| `jN_r3_no_agreement` | Booleano: el juez devolvió "No agreement reached" |
| `collective_no_agreement_count` | Conteo de votos "No agreement reached" |

> **Verdad de referencia**: Statement 1 es siempre la afirmación correcta. `ground_truth_alignment = True` si y solo si `statement == "Statement 1"`. En `con_consenso`, los indicadores de cambio de creencia se establecen como `None` cuando `r3_statement == "No agreement reached"`.

---

## Modelos

| Rol | Modelo |
|---|---|
| Juez | `claude-sonnet-4-20250514` |
| Fallback de parseo | `claude-haiku-4-5-20251001` |

---

## Parseo de respuestas

Primero regex, con fallback de salida estructurada vía Haiku:

| Fase | Campos extraídos | Parser de fallback |
|---|---|---|
| Fase 1 | `Statement`, `Confidence` | `llm_parse_prior_belief` |
| Fase 2 | `Statement`, `Confidence`, `Key points` | `llm_parse_round_1` |
| Fase 3 | `Response to other judges' arguments`, `Outstanding issues` | `llm_parse_round_2` |
| Fase 4 | `Statement` (+ `Confidence`, `Final reasoning` en sin_consenso) | `llm_parse_round_3` |

Los fallos de parseo se registran en `error_log.csv` y `llm_parsing_log.csv`. Un fallo de parseo **no** aborta el debate (el valor se almacena como `None`). Un fallo de API tras 3 reintentos **sí** aborta el debate y queda registrado en `errors_summary.json`.

---

## Guardado progresivo

El JSON y el CSV se escriben/actualizan tras **cada fase** (una vez que todos los jueces de esa fase han respondido). Una interrupción a mitad de fase solo pierde la fase en curso; todas las fases completadas quedan persistidas. Cada run añade un timestamp al nombre del CSV para evitar sobreescrituras entre ejecuciones.

---

## Ejecución

```bash
# Requiere ANTHROPIC_API_KEY en .env
python run_deliberation_sin_consenso.py
python run_deliberation_con_consenso.py
```

Para limitar el número de debates procesados, establecer `N_DELIBERATIONS` al final de cada script:

```python
N_DELIBERATIONS = 5   # o None para procesar todos
```

---

## Constante clave

```python
N_JUDGES = 13  # Cambiar a 3 o 13 — controla el tamaño del panel y la composición de la mayoría
```
