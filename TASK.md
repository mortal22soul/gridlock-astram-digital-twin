# TASK.md — Predictive Traffic Incident Intelligence (ASTraM Event Forecasting)

> This document consolidates an entire prior research/scoping conversation. It is meant to be handed to an LLM coding assistant (or read by a human) as the single source of truth for building this hackathon prototype. Nothing discussed in scoping has been omitted — including dead ends, rejected approaches, and environment limitations discovered along the way — so the implementer doesn't repeat wasted effort. Deadlines/timelines are intentionally NOT included here; sequencing below is dependency-driven, not date-driven.

---

## 1. Origin: The Hackathon Brief

Three problem statements were provided, all themed around Bengaluru traffic congestion. Full text preserved below for context, even though only Problem 2 was selected.

### Problem Statement 1 — Poor Visibility on Parking-Induced Congestion

**Operational challenge:** On-street illegal parking and spillover parking near commercial areas, metro stations, and events choke carriageways and intersections.
**Why it's hard today:** Enforcement is patrol-based and reactive; no heatmap of parking violations vs. congestion impact; difficult to prioritize enforcement zones.
**Problem direction:** How can AI-driven parking intelligence detect illegal parking hotspots and quantify their impact on traffic flow to enable targeted enforcement?
**Dataset link:** `https://uc.hackerearth.com/he-public-ap-south-1/jan%20to%20may%20police%20violation_anonymized791b166.csv`
**Status:** NOT selected. Only a 6-row sample was ever reviewed (full file was never downloaded/audited). See Section 4 for why it was deprioritized.

### Problem Statement 2 — Event-Driven Congestion (Planned & Unplanned) — **SELECTED**

**Operational challenge:** Political rallies, festivals, sports events, construction activities, and sudden gatherings create localized traffic breakdowns.
**Why it's hard today:** Event impact is not quantified in advance; resource deployment is experience-driven; no post-event learning system.
**Problem direction:** How can historical and real-time data be used to forecast event-related traffic impact and recommend optimal manpower, barricading, and diversion plans?
**Dataset link:** `https://uc.hackerearth.com/he-public-ap-south-1/Astram%20event%20data_anonymized%20-%20Astram%20event%20data_anonymizedb40ac87.csv`
**Status:** SELECTED and fully audited (full CSV, not just sample). See Section 5.

### Problem Statement 3 — Automated Photo ID/Classification for Traffic Violations (CV)

**Overview:** Develop a CV system that processes traffic images to detect vehicles/road users, identify and classify violations, and generate annotated evidence.
**Tasks specified in brief:**

1. Image preprocessing (low light, rain, shadows, motion blur handling)
2. Vehicle/road-user detection and classification
3. Violation detection: helmet non-compliance, seatbelt non-compliance, triple riding, wrong-side driving, stop-line violation, red-light violation, illegal parking
4. Violation classification with confidence scores
5. License plate recognition (detection + OCR)
6. Evidence generation (annotated images, metadata, timestamps)
7. Analytics and reporting (violation stats/trends, searchable records)
8. Performance evaluation (Accuracy, Precision, Recall, F1, mAP; computational efficiency)
   **Expected outcome:** A scalable AI-based traffic image analysis system for automated violation identification/documentation.
   **No dataset was provided** — submission explicitly allowed as a concept note / prototype proposal / solution framework rather than a fully validated pipeline.
   **Status:** NOT selected. See Section 4 for why.

---

## 2. Decision Process — How We Got to Problem 2

Three independently-sourced planning documents were reviewed during scoping, each proposing a different angle. All three are summarized and critiqued here because each contributed something, and an implementer should understand which parts were kept vs. discarded.

### Document A — "Maximize hackathon impact" framing

- Ranked: Event-Driven Congestion > Parking Intelligence > Violation Detection.
- Argued Violation Detection has a **low innovation ceiling** — every team builds YOLO + OCR + helmet/triple-riding detection + dashboard; judges have seen it many times.
- Proposed an extremely elaborate 5-layer architecture for Problem 2: event detection → GNN/Temporal-GNN traffic forecasting → OR-Tools resource optimization → SUMO/RL digital-twin simulator → post-event learning loop.
- **Critique:** The 5-layer plan (esp. the SUMO-calibrated digital twin + RL layer) is research-grade infrastructure, not buildable in any realistic hackathon timeframe — standing up a calibrated SUMO network alone is a multi-week professional effort. Much of the document's framing ("judges love simulations", "most teams will build X, you should build Y") is generic AI-pitch rhetoric not grounded in the actual dataset.
- **What was kept:** The instinct to go beyond "predict congestion" into "recommend a response" and "learn from post-event outcomes" — just executed at a much smaller, tabular-ML scale (see Section 6), not via GNNs/SUMO/RL.

### Document B — "Dataset-grounded" framing (written after seeing 6-row samples)

- Correctly identified that Problem 1's sample data has no speed/volume/queue-length fields, so "quantify impact on traffic flow" can only ever be estimated, not measured directly — this weakens Problem 1.
- Correctly identified that Problem 2's `end_datetime - start_datetime` (or closure timestamps) gives a real, computable supervised target for incident duration.
- Proposed 4 models: (1) Impact/duration prediction, (2) Road-closure probability, (3) **Resource recommendation** via clustering/kNN/XGBoost using "incident type + location + priority → recommended response plan", (4) Corridor risk heatmap.
- Gave illustrative example output: "45 min disruption, 2 traffic units needed, 1 tow truck needed, diversion route A recommended."
- Included a numeric scoring table (Dataset Strength/Innovation/Winning potential, e.g. 9.5/10) presented with false precision — no real methodology behind the numbers.
- **Critique — IMPORTANT, confirmed by full audit (Section 5):** Model 3 (Resource Recommendation) is **not buildable as a supervised model**. There is no field anywhere in the real dataset recording historical officer/tow-truck/barricade counts. `meta_data` (the most likely candidate to hide this) came back **100% null** across all 8,173 rows in the full audit. `assigned_to_police_id` is **98.4% null** (only 128 populated rows) — far too sparse for even a weak-label proxy. The "2 traffic units, 1 tow truck" example output in this document is illustrative fiction, not something derivable from this schema. Do not attempt to train Model 3 as written.

### Document C — "Practical/grounded MVP" framing (the most useful of the three)

- Correctly proposed a **heuristic fallback** for resource recommendation instead of a supervised model, given the missing labels: map severity buckets (Low/Medium/High) to default staffing/tow/barricade recommendations, with supervised resourcing explicitly deferred as a stretch goal contingent on finding real deployment labels later (e.g., via dispatch logs that don't currently exist in this dataset).
- Proposed LightGBM/XGBoost for duration (regression, log-transformed target) and closure probability (binary classification) — appropriately scoped, fast to build, no exotic infrastructure needed.
- Proposed time-based/walk-forward cross-validation, SHAP explainability, calibration (Platt/isotonic) for probability outputs.
- Flagged real engineering concerns: timezone conversion (timestamps are UTC `+00`, need IST conversion), missing `end_datetime` should be treated as censored data, PII removal needed before modeling.
- Listed external datasets to consider augmenting with (traffic speed/volume APIs, OSM road network, event calendars, weather, simulation tools) — this list was investigated in Section 7; most were found to be either inaccessible in the build sandbox, synthetic, or not worth the integration cost for an MVP.
- Proposed a long rollout timeline (data audit → models → dashboard → **2–4 weeks of live shadow validation**) — this is a production rollout timeline, not hackathon scope. Drop the shadow-validation phase and any external-API integration entirely; they don't fit a short build window.
- **What was kept:** This document's core MVP shape (duration regression + closure classification + heuristic resourcing + heatmap, via LightGBM/XGBoost) is the foundation of the final recommended build in Section 6.

### Why Problem 2 over Problem 1

Problem 1's only available data (sampled, not fully audited) is point-in-time violation records (lat/long, violation type, timestamp, police station) with **no speed, volume, or travel-time fields** — any "impact on traffic flow" claim would be a constructed proxy (e.g., weighting by violation subtype like "PARKING NEAR ROAD CROSSING" vs generic "NO PARKING", or enriching with OSM road class), not a measured quantity. Problem 2's dataset, by contrast, has real historical timestamp fields that produce genuine supervised-learning targets (see Section 5).

### Why Problem 2 over Problem 3

Beyond the "low innovation ceiling" argument (every team builds YOLO+OCR+helmet detection), independent web research (Section 8) confirmed that Bengaluru Traffic Police **already has a production CV violation-detection system deployed**: the **Videonetics Traffic Management System**, live at 50 of the city's busiest intersections, doing ANPR + speed detection, red-light violation detection, two-wheeler helmet/triple-riding detection via deep learning, and in-cabin seatbelt/phone-usage detection for four-wheelers. Building Problem 3 would mean re-implementing an already-deployed commercial system. Problem 2, by contrast, targets a capability that BTP's own materials describe as still aspirational/in-procurement (see Section 8 — the Mobility Digital Twin tender).

---

## 3. Real-World System Context — ASTraM

This dataset originates from **ASTraM (Actionable Intelligence for Sustainable Traffic Management)**, a real, currently-deployed Bengaluru Traffic Police (BTP) system, not a hypothetical scenario. This context should be used in the pitch narrative.

**IMPORTANT CAVEAT before using any of this in a pitch:** Much of the detail below came from two long AI-synthesized research documents that read as polished and comprehensive but cite sources only generically ("IEEE Spectrum report," "Arcadis case study," "Times of India coverage") without single clickable/verifiable links for many specific numbers. Treat all numeric claims below as **plausible background color to verify before quoting confidently to judges**, not as confirmed facts. Where something was independently confirmed via direct web search, it's marked as such.

**Independently confirmed via direct web search:**

- ASTraM = Actionable Intelligence for Sustainable Traffic Management, led by BTP (Joint Commissioner of Police Traffic figures: M.N. Anucheth and Karthik Reddy referenced across sources).
- B-ATCS (Bengaluru Adaptive Traffic Control System) is built on **CoSiCoSt**, an adaptive signal control algorithm developed indigenously by C-DAC, specifically because Western adaptive systems assume lane discipline that doesn't hold in Indian traffic.
- Real, recent news activity (award coverage, late 2025/2026) confirms the system is actively operated and expanding, not a defunct pilot.

**From the AI-synthesized background reports (verify before quoting precisely):**

- Three-tier ASTraM architecture: (1) pervasive data integration — CCTV, ANPR, GPS, app/social reports, map APIs; (2) AI & predictive analytics — distinguishing recurring vs. non-recurring congestion, forecasting chokeholds, simulating event impact; (3) automated communication — batched officer alerts at ~15-minute intervals.
- **B-ATCS rollout:** Phase 1 (2024–2025) covers ~155–169 junctions; reported corridor-level speed improvements (e.g., Bannerghatta Road 17.9→20.8 km/h) and an estimated ~25% average travel-time reduction, ~10% emissions reduction.
- **Videonetics TMS:** deployed at 50 intersections; ANPR+speed, red-light detection, helmet/triple-riding detection, in-cabin seatbelt/phone-usage detection (this is the system that makes Problem 3 redundant — see Section 2).
- **E-Path:** GPS-based dynamic signal preemption creating a rolling "green corridor" for registered ambulances; escalation alerts if an ambulance is stationary for 60–120 seconds.
- **Operation ROPE** (Removal of Obstructive Parking & Encroachments): physical enforcement initiative reclaiming road geometry; officials cite up to a 30% traffic-flow improvement from this alone — physical enforcement, not AI, but relevant context for why Problem 1's framing has real-world precedent.
- **Mobility Digital Twin (MDT) — highly relevant to this project's pitch:** BTP floated a ₹1 crore tender for an AI-driven Digital Twin under the Bengaluru City Road Safety and Traffic Management Programme, explicitly because long-term predictive infrastructural modeling is the system's acknowledged current gap. ~3,200 km of the city's arterial network has been mapped so far (target: full 14,000 km network).
- **Anecdote worth using in the pitch:** A conventional deterministic traffic simulation once predicted that shifting a U-turn by 200 meters would ease congestion; in reality, non-compliant driver behavior caused a 20% _increase_ in gridlock. This is cited (per JCP Karthik Reddy) as the reason BTP wants behavior-grounded, real-historical-data models rather than idealized-compliance simulations — which is exactly the kind of model this project builds (trained on actual historical incident behavior, not assumed compliance).
- Academic caveat: Prof. Ashish Verma (IISc Sustainable Transportation Lab) has publicly noted that sophisticated data/simulation is sometimes ignored in final political/policy decisions — worth a brief mention in a "limitations/future work" pitch slide for credibility, not a blocker.

**Pitch framing recommendation:** Position this project as _"the predictive intelligence layer ASTraM's own Digital Twin tender says doesn't fully exist yet, built from ASTraM's own historical incident data"_ — not as a generic, hypothetical congestion-prediction exercise.

---

## 4. Problem 1 Dataset — Parking Violations (NOT selected, reference only)

Only a 6-row CSV sample was ever reviewed (full file was never downloaded). Schema observed:

```
id, latitude, longitude, location, vehicle_number, vehicle_type, description,
violation_type, offence_code, created_datetime, closed_datetime, modified_datetime,
device_id, created_by_id, center_code, police_station, data_sent_to_scita,
junction_name, action_taken_timestamp, data_sent_to_scita_timestamp,
updated_vehicle_number, updated_vehicle_type, validation_status, validation_timestamp
```

Observations from the sample: `violation_type` and `offence_code` are JSON-array strings (a record can have multiple simultaneous violations, e.g., `["WRONG PARKING","PARKING NEAR ROAD CROSSING"]` with matching offence codes `[112,104]`). `closed_datetime`, `action_taken_timestamp`, and `data_sent_to_scita_timestamp` were NULL in every sampled row, and `validation_status` was inconsistently populated — if this holds across the full file, enforcement-turnaround analysis would be very thin. No speed/volume/queue-length field exists anywhere in this schema, which is the core reason this problem was deprioritized (see Section 2).

If ever revisited: hotspot detection (DBSCAN/HDBSCAN on lat/long), severity weighting from violation subtype + junction proximity + OSM road class, and a temporal (hour/day-of-week) layer were the feasible directions identified.

---

## 5. Problem 2 Dataset — Full Audit Findings (ASTraM Event Data)

**File:** `Astram_event_data_anonymized_-_Astram_event_data_anonymizedb40ac87.csv`
**Location used during scoping:** `/mnt/user-data/uploads/Astram_event_data_anonymized_-_Astram_event_data_anonymizedb40ac87.csv`
**Size:** 8,173 rows, 43 columns. No duplicate `id`s.

### 5.1 Full column list

```
id, event_type, latitude, longitude, endlatitude, endlongitude, address, end_address,
event_cause, requires_road_closure, start_datetime, end_datetime, status, authenticated,
modified_datetime, map_file, direction, description, veh_type, veh_no, corridor, priority,
cargo_material, reason_breakdown, age_of_truck, created_date, route_path, client_id,
created_by_id, last_modified_by_id, assigned_to_police_id, citizen_accident_id, comment,
police_station, meta_data, kgid, resolved_at_address, resolved_at_latitude,
resolved_at_longitude, closed_by_id, closed_datetime, resolved_by_id, resolved_datetime,
gba_identifier, zone, junction
```

### 5.2 Missingness of key fields (full-file audit, n=8,173)

| Field                   | % Null                          | Notes                                                                                                 |
| ----------------------- | ------------------------------- | ----------------------------------------------------------------------------------------------------- |
| `meta_data`             | **100%**                        | Completely dead field. No hidden resource-deployment data here.                                       |
| `comment`               | 100%                            | Dead field.                                                                                           |
| `map_file`              | 100%                            | Dead field.                                                                                           |
| `resolved_datetime`     | ~99.1%                          | Essentially unusable as a target.                                                                     |
| `assigned_to_police_id` | **98.4%** (only 128 non-null)   | Far too sparse to use even as a weak label for resourcing.                                            |
| `end_datetime`          | ~94% (490 non-null)             | Populated almost exclusively for **planned** events (467 planned events exist; counts roughly match). |
| `closed_datetime`       | ~61.6% (3,141 non-null)         | The main usable closure-time field for unplanned/closed incidents.                                    |
| `zone`                  | ~58% (4,729 missing of 8,173)   | Significant gap — prefer `corridor` as the primary geo-aggregation unit instead.                      |
| `junction`              | ~31% populated (2,510 non-null) | Moderate coverage.                                                                                    |

### 5.3 Categorical distributions

- **`status`:** closed = 7,095, active = 1,007, resolved = 71. **`active` rows are right-censored (still open) — exclude from any duration-target training**, since their true duration is unknown, not small/zero.
- **`event_cause`** (full counts, descending): vehicle_breakdown ≈ 4,896 (~60% of all rows), others ≈ 638, pot_holes ≈ 537, construction ≈ 480, water_logging ≈ 458, accident ≈ 365, tree_fall ≈ 284, road_conditions ≈ 170, congestion ≈ 136, public_event ≈ 84, procession ≈ 72, vip_movement ≈ 20, protest ≈ 15, debris-type entries ≈ 13 (inconsistent casing — needs cleaning), test_demo ≈ 3, fog/low-visibility ≈ 2.
  - **Critical framing note:** the problem statement's narrative ("political rallies, festivals, sports events") corresponds to `public_event` + `procession` + `vip_movement` + `protest` combined ≈ **191 rows, only ~2.3% of the dataset**. The bulk of the data is everyday vehicle breakdowns and infrastructure issues. The pitch should be framed as a general incident-impact intelligence system that also covers marquee events, not oversold as highly accurate specifically for rallies/festivals — sample sizes there are too small for strong statistical confidence.
- **`corridor`:** "Non-corridor" is most common (~3,124 rows, ~38%), with substantial counts on named arterials (Mysore Road, Bellary Road, Tumkur Road, Hosur Road, Outer Ring Road segments, etc.) — good granularity for a corridor-level risk heatmap.
- **`priority`:** High/Low values present (exact counts not fully tabulated in scoping — verify in implementation) — usable directly as a severity feature/target.
- **`requires_road_closure`:** boolean, well-populated across the dataset — a clean binary classification target.

### 5.4 Geometry observations

- `endlatitude`/`endlongitude` are exactly **0** (placeholder, not real) in ~91.4% of non-null rows — most events are point incidents, not route/segment incidents. Only a minority (e.g., tree-fall-type entries) have genuine start/end route geometry.

### 5.5 Constructing a usable "duration" target

Neither `end_datetime` alone (94% null) nor `closed_datetime` alone (61.6% null) gives good coverage individually, but they are **complementary** (planned events populate `end_datetime`; closed unplanned incidents populate `closed_datetime`). Combining them:

```python
df['resolution_dt'] = df['end_datetime'].fillna(df['closed_datetime'])
usable = df['resolution_dt'].notna() & df['start_datetime'].notna()
# usable.mean() ≈ 42.6% of all rows
```

This is the best achievable duration-label coverage from this schema — be transparent in any write-up that the duration model trains on under half the data, and that this subset may be mildly biased toward better-documented incident types.

**Data quality issues found in the raw duration computation (`resolution_dt - start_dt`):**

- **51 rows have negative duration** (resolution timestamp before start timestamp) — concentrated in entries where `end_datetime` appears to represent an _estimated/target_ time rather than actual closure (seen disproportionately in construction-type entries). **Drop these rows** for training, or treat as a separate data-quality flag.
- **440 rows show durations exceeding 7 days** — concentrated in infrastructure-type causes (pot_holes, water_logging, construction, road_conditions, "others") where multi-day persistence before official closure is plausible, not necessarily an error. These have a fundamentally different duration distribution than quick-clearing causes (vehicle_breakdown, accident) and should either be (a) winsorized/capped (e.g., at 48 hours) for a single "quick-clear" model, or (b) modeled as a separate regime/segment from short-duration incidents.
- **Cleaned median duration by `event_cause`** (after filtering to a believable 0–48h window) showed a sensible, demoable gradient: vip_movement and protest cleared fastest (single-digit minutes — small sample, treat cautiously), vehicle_breakdown/accident ≈ 40 minutes, procession ≈ 60 min, congestion ≈ 71 min, others ≈ 85 min, tree_fall ≈ 116 min, pot_holes ≈ 135 min, water_logging ≈ 155 min, public_event ≈ 163 min, road_conditions ≈ 277 min, construction ≈ 457 min (~7.6 hours). This ordering itself is a good sanity-check chart for the pitch deck even before training a model.
- Date range of `start_datetime` across the file: **2023-11-09 to 2024-04-08** (~5 months), giving a reasonable density (~54 incidents/day average) for walk-forward validation, though it's not multi-year history.

### 5.6 Other engineering notes

- Timestamps carry a `+00` (UTC) suffix — **convert to IST (UTC+5:30)** before any hour-of-day/day-of-week feature engineering, or peak-hour features will be wrong.
- Remove PII risk: `description` field mixes English and Kannada free text, with some entries already containing redaction placeholders (`[LOCATION]`, `[PERSON]`) — treat as partially anonymized; don't rely on raw text without multilingual handling, and don't re-expose anything in `veh_no` casually.
- `veh_type`, `veh_no`, `cargo_material`, `age_of_truck` are populated only when relevant to `event_cause` (e.g., vehicle_breakdown) — expect heavy structural sparsity, not missing-at-random.

---

## 6. Final Recommended Build Scope

No external dataset is required for this core scope (see Section 7 for why external augmentation was deprioritized). Everything below is buildable from the audited ASTraM event file alone.

### 6.1 Data cleaning & target construction

- Build `resolution_dt = end_datetime.fillna(closed_datetime)`; compute `duration_minutes = resolution_dt - start_datetime`.
- Exclude `status == 'active'` rows from duration-model training (censored, not resolved).
- Drop the 51 negative-duration rows (data entry errors).
- Either cap/winsorize duration at ~48 hours for a single "quick-clear" model, or split into two regimes: fast-clearing causes (breakdown/accident/procession/etc.) vs. long-running infrastructure causes (pothole/construction/water_logging/road_conditions) — recommend trying the simpler single-model-with-capping approach first, and only splitting if validation error is clearly bimodal.
- Convert all timestamp columns from UTC to IST.
- Drop dead/near-dead columns entirely from modeling: `meta_data`, `comment`, `map_file`, `resolved_datetime`, `assigned_to_police_id` (document why — see Section 5.2 — so nobody re-attempts the resource-recommendation model from Document B).
- Decide a `zone` strategy: since 58% is missing, prefer `corridor` as the primary geographic aggregation key; optionally attempt to backfill `zone` via a `police_station → zone` or `corridor → zone` lookup table derived from the non-null rows, if a clean 1:1 mapping exists.
- Clean inconsistent casing/duplicates in `event_cause` (e.g., "Debris" vs "debris").

### 6.2 Feature engineering

- **Temporal:** hour of day (IST), day of week, weekend flag.
- **Categorical:** `event_cause`, `event_type` (planned/unplanned), `corridor`, `zone` (where available), `priority`, `requires_road_closure`.
- **Historical/rolling:** count of incidents on the same corridor in the trailing 1/7/30 days; historical mean duration per corridor and per cause (computed only from training-period data to avoid leakage in walk-forward validation).

### 6.3 Models

1. **Duration regression** — LightGBM or XGBoost, target = `log(duration_minutes + 1)` on the cleaned/capped subset (~42.6% raw label coverage before cleaning — note the exact usable row count once cleaning rules are applied in code). Be explicit in any report that this model is trained on a documented-incident subset, which may not generalize perfectly to undocumented ones.
2. **Road-closure probability** — binary classifier (LightGBM/XGBoost) on `requires_road_closure`, using the full dataset (this field is well-populated, no censoring issue).
3. **Severity bucket** — either use the existing `priority` field directly as a feature/target, or construct an ordinal severity score combining `priority` + `event_cause` + `corridor` if a more granular scale is wanted.

- **Validation:** time-based / walk-forward split (train on earlier months, validate on later months within the Nov 2023–Apr 2024 range) — not random k-fold, to avoid leaking future information.
- **Metrics:** MAE and median absolute error for duration; AUC and Brier score (with probability calibration — Platt or isotonic) for closure probability.
- **Explainability:** SHAP values for feature importance and per-incident explanations, to justify recommendations to a non-technical judge/officer audience.

### 6.4 Resource recommendation — heuristic only, NOT a trained model

There are no historical resource-deployment labels anywhere in this schema (confirmed in Section 5.2). Do not attempt a supervised "resource recommendation" model (this was Document B's mistake). Instead:

- Map severity bucket → a fixed heuristic lookup table, e.g.:
  - Low severity: 1 traffic officer, no tow, signage only.
  - Medium severity: 2 officers, 1 tow truck, one temporary barricade/diversion point.
  - High severity: 4 officers, 2 tow trucks, two barricades, diversion signage + public alert.
- Frame this explicitly in any pitch/report as a **rule-based starting point**, with a clearly labeled "future work" note that it could become a learned policy once real dispatch/deployment logs are obtained (none exist today).

### 6.5 Corridor/zone risk heatmap

- Aggregate incident counts, average predicted severity, and average predicted duration by `corridor` (primary) and `zone` (where available) over rolling time windows.
- Visualize geographically using the `latitude`/`longitude` fields already present.

### 6.6 Dashboard

- Lightweight build (Streamlit recommended for speed of iteration) showing: the corridor risk heatmap, a per-incident prediction panel (duration estimate + closure probability + heuristic resource recommendation + SHAP explanation), and historical trend charts by cause/corridor.
- **Tech stack:** Python — `pandas`, `lightgbm` and/or `xgboost`, `scikit-learn`, `shap`, `streamlit`, `plotly` or `folium` for maps. `geopandas` only if/when OSM enrichment is pursued (see Section 7).

### 6.7 Optional stretch goals (only if core scope above is fully done with time to spare)

- **CV-based live congestion proxy** using the UVH-26 pretrained models (Section 7.1) to estimate vehicle density from sample camera frames near high-predicted-risk corridors, layered on top of the tabular forecasting model as a multi-modal differentiator.
- **OSM road-class enrichment** (Section 7.2) — requires manual acquisition (see environment constraints in Section 9); not auto-fetchable from within a typical sandboxed dev environment.
- **Post-event learning loop** (predicted vs. actual duration/severity, retraining over time) — conceptually valuable (and explicitly mirrors a gap named in the original problem statement: "no post-event learning system"), but only worth implementing if the core models are solid first; can otherwise be described as a roadmap item in the pitch.

---

## 7. External Datasets — Research Findings (what's usable, what isn't)

**Top-line decision: no external dataset is required for the core MVP.** The `corridor` field already gives a reasonably strong arterial-vs-local categorical signal, covering most of what road-network enrichment would have added. The items below were investigated during scoping; keep this record so they aren't re-investigated from scratch.

### 7.1 UVH-26 (IISc AIM, Bengaluru) — best stretch-goal candidate

- Real, recent (released Nov 2025), and directly Bengaluru-specific: 26,646 high-resolution (1080p) images sampled from ~2,800 of Bengaluru's actual Safe-City CCTV cameras, ~1.8 million bounding boxes across 14 India-specific vehicle classes (Hatchback, Sedan, SUV, MUV, Bus, Truck, Three-wheeler, Two-wheeler, LCV, Mini-bus, Tempo-traveller, Bicycle, Van, Other), built in collaboration with Bengaluru Traffic Police and IISc's CiSTUP/ARTPARK.
- **Pretrained models already published** (YOLOv11, DAMO-YOLO, RT-DETRv2 variants fine-tuned on UVH-26, reporting up to 31.5% mAP@50:95 improvement over COCO-trained baselines for Indian traffic scenes) — no training from scratch needed if this is pursued.
- Hosted on Hugging Face:
  - Dataset: `https://huggingface.co/datasets/iisc-aim/UVH-26`
  - Models: `https://huggingface.co/iisc-aim/UVH-26`
  - Technical report: `https://arxiv.org/abs/2511.02563`
- License: dataset CC-BY-4.0, models Apache 2.0.
- **Relevance:** plays directly to a YOLO/CV background (prior project: SafeScape, IIT Bombay Techfest 2024 winner, YOLO-based person detection) and would make a strong differentiator if time allows — a live vehicle-density signal corroborating the tabular incident-forecasting model.

### 7.2 OpenStreetMap road network (Geofabrik / Overpass)

- Would provide road class, lane count, junction geometry to enrich `corridor`/lat-long features.
- **Confirmed NOT directly fetchable from within the sandboxed build environment used during scoping** — `geofabrik.de` and `overpass-api.de` are outside that environment's network allowlist, and its web-fetch tool could only retrieve URLs that had already appeared in a prior search result, not arbitrary constructed query URLs. A search for pre-made Bengaluru road GeoJSON files on GitHub (which _was_ reachable) only turned up tiny, single-neighborhood files — not usable city-wide.
- **If pursued:** must be manually downloaded by a human (via the Geofabrik Karnataka/India extract, or by manually exporting a bounding-box query from the Overpass Turbo web UI at `overpass-turbo.eu`) and then supplied as a local file for processing — do not assume an LLM coding assistant in a similarly sandboxed environment can fetch this automatically; check network access first before planning around it.
- Given that `corridor` already provides a usable arterial/local signal, this was judged not worth the manual-acquisition friction for the core MVP.

### 7.3 Kaggle "Dataset of Bangalore's Traffic" / "Bangalore's Traffic Pulse" (by preethamgouda, also mirrored via Opendatabay)

- ~8,900–16,700 rows covering traffic volume, congestion level, average speed, weather, signal compliance, incident reports, parking usage, pedestrian/cyclist counts.
- **Caveat:** the unusually clean structure (no missing values, round numbers) strongly suggests this is **synthetically generated** for ML practice rather than sensor-collected. If used at all, treat strictly as an illustrative/decorative congestion-correlation layer for visualization, and say so explicitly — do not present it as verified ground-truth telemetry in a pitch.

### 7.4 Other options considered and deprioritized

- **data.gov.in** road-accidents dataset group — exists, but likely state/national aggregate stats rather than granular incident-level data; not deeply investigated.
- **Kaggle Bengaluru accidents (1990–2006)** — real but dated; only useful for a long-term hotspot-stability framing point, not current modeling.
- **India Urban Data Exchange (IUDX)** — the right long-term institutional source for real Bengaluru sensor data, but operates on a request/access-grant model between cities and partner organizations, not instant public download. Not viable on a short build timeline; worth a single line in a "future scale-up" pitch slide only.

---

## 8. Source Material Quality Notes

Two long background documents on the ASTraM ecosystem were reviewed (summarized into Section 3). Both read as comprehensive, well-structured AI-synthesized research reports. They are useful for context and pitch framing but were **not independently fact-checked line-by-line** — several very specific statistics (e.g., "commuters lose 117 hours annually," "16.6 lakh Public Eye reports over 10 years," "Rs 25 crore in fines," "2.3/5 App Store rating," "87% automated enforcement rate," "3 million e-challans in 7 months") came from generic, non-clickable citations ("IEEE Spectrum report," "Times of India coverage," "Arcadis case study") rather than verifiable single sources. **Anyone preparing pitch slides from this material should re-verify any specific number via direct search before presenting it to judges**, especially since this is a Bengaluru-based hackathon where a judge could plausibly know the real figures.

---

## 9. Environment / Tooling Constraints Discovered During Scoping

- A `web_fetch`-style tool in the scoping environment could only retrieve URLs that had already appeared in a prior search result — it could not fetch arbitrary constructed query URLs (e.g., a hand-built Overpass API query string failed with a permissions error for this reason).
- The scoping environment's outbound network (for a bash/code-execution tool) was restricted to an allowlist that did **not** include `geofabrik.de`, `overpass-api.de`, or `huggingface.co`, but **did** include `github.com` / `raw.githubusercontent.com` / `codeload.github.com`, `pypi.org`/`pythonhosted.org`, and `npmjs.com`/`registry.npmjs.org`, among others.
- **Action for the implementing LLM/environment:** before planning any step that depends on fetching OSM data, Hugging Face model weights, or other external sources, verify what's actually reachable in the current execution environment — don't assume the same restrictions apply, but don't assume they don't, either. If blocked, fall back to asking the human to manually download and upload the needed file.

---

## 10. Open Questions / Things to Verify Before Finalizing

- Whether `zone` can be reliably backfilled via a clean `corridor → zone` or `police_station → zone` mapping derived from the non-null rows (would reduce reliance on the 58%-missing field).
- Whether Problem 1's full dataset should ever be downloaded and fully audited (only a 6-row sample was reviewed) — not needed unless there's a reason to reconsider the problem choice.
- Re-verify any specific ASTraM/B-ATCS/Videonetics statistic from Section 3 before it appears in pitch material, per the caveat in Section 8.
- Hackathon submission format (live demo vs. slide deck vs. notebook vs. written concept note) was never specified in this conversation and materially affects how much dashboard polish vs. narrative-writing effort matters — confirm this before allocating final-stage effort.
- Exact `priority` field value counts and its relationship to `event_cause`/`corridor` were not fully tabulated during scoping — worth a quick check during implementation.

---

## 11. Working Title / Narrative Hooks (optional, for pitch use)

- Position the system as filling the gap explicitly named by BTP's own Mobility Digital Twin tender (Section 3) — a real, current, acknowledged institutional need, not a hypothetical exercise.
- Use the U-turn/behavioral-non-compliance anecdote (Section 3) to justify training on real historical incident behavior rather than building an idealized simulation.
- Be upfront in any "limitations" slide about the ~2.3% sample size for marquee public events/processions/rallies, and about the heuristic (not learned) nature of the resource-recommendation layer — both are honest, defensible scope decisions rather than weaknesses to hide.
