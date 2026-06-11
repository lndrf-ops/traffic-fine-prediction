# Gutachten: Projektbericht „Road Traffic Fine Management Process" — Gruppe 1

**Veranstaltung:** Process Analytics, SS26, Universität Leipzig  
**Gutachter/in:** Jun.-Prof. Dr. Jana-Rebecca Rehse (simuliert)  
**Datum:** 4. Juni 2026

---

## 1. Executive Summary & Gesamturteil

Der vorliegende Bericht der Gruppe 1 präsentiert eine **herausragende Projektarbeit**, die sowohl in Breite als auch in Tiefe weit über das erwartete Niveau hinausgeht. Auf ca. 35 Seiten (9 Sektionen) wird ein vollständiger End-to-End Process Analytics Pipeline systematisch und methodisch sauber durchgeführt — von der Datenvorbereitung über Process Discovery und Conformance Checking bis hin zu Predictive Monitoring, Interpretability, Prescriptive Analytics, Generative AI und einem funktionsfähigen Deployment.

Besonders hervorzuheben ist die **intellektuelle Reife** des Berichts: Die Gruppe versteht nicht nur *was* ihre Modelle vorhersagen, sondern *warum* — und hat die kritische Einsicht gewonnen, dass die Prozessstruktur selbst die Vorhersagbarkeit determiniert, nicht die Modellkomplexität. Diese Art der Reflexion ist auf Masterniveau selten.

**Alle 4 Pflichtanforderungen und alle 7 optionalen Aufgaben wurden vollständig erfüllt.**

---

## 2. Erfüllung der Pflichtanforderungen (Core Tasks)

### 2.1 Data Preparation & Event Log Construction ✅ (Exzellent)

| Kriterium | Bewertung |
|-----------|-----------|
| Datenbereinigung | Deduplication, Sortierung, numerische Koerzierung — sauber |
| Data-Leakage-Prävention | `totalPaymentAmount`, `paymentAmount`, `expense` korrekt ausgeschlossen |
| Outcome-revealing Activities | `Payment` und `Send for Credit Collection` werden streng aus den Input-Features entfernt |
| Labeling-Logik | Klare Priority-Rule (Collection > Payment > exclusion) — methodisch einwandfrei |
| Begründung | Jede Entscheidung wird transparent begründet |

Die Gruppe hat eine **mustergültige** Data-Preparation durchgeführt. Die Erkenntnis, dass `org:resource` nur Placeholder-Werte enthält und daher entfernt werden muss, zeigt sorgfältiges Data Profiling. Die Entfernung von Outcome-revealing Activities aus dem Feature-Space ist methodisch korrekt und wird nicht von allen Gruppen so konsequent umgesetzt.

### 2.2 Descriptive Data Analysis (EDA) ✅ (Sehr gut)

Die explorative Analyse ist umfassend und liefert drei klare, für die nachfolgende Modellierung relevante Insights:

1. Kurze Traces dominieren → Begründung für $k \in \{2,3,5\}$
2. Throughput-Time korreliert mit Komplexität → Feature-Motivation
3. Timestamps nur tagesgenau → Limitierung transparent kommuniziert

Die Aufschlüsselung der Fine-Amounts nach Outcome-Klasse (49.03€ vs. 40.05€ mean) ist ein schönes Detail, das zeigt, dass die Gruppe über rein deskriptive Statistik hinaus bereits hypothesengeleitet arbeitet. Die Erkenntnis, dass die niedrigen Volumina 2012/2013 ein Artefakt der Datenextraktion sind (und keine echte Trendänderung), zeugt von analytischer Sorgfalt.

### 2.3 Process Discovery ✅ (Exzellent)

**Workflow-Perspektive:**
- Variant Analysis mit Top-5-Varianten (68% Coverage durch 2 Varianten)
- Inductive Miner auf Top-10-Varianten → Petri-Netz mit klarer Strukturinterpretation
- Token-Based Replay: 98.70% Fitness auf 2.000 ungefilterten Cases → exzellente Modellqualität

**Performance-Perspektive:**
- Bottleneck-Analyse mit Mean vs. Median Perspektive — gute methodische Unterscheidung
- Dotted Chart zeigt Batching-Verhalten eindeutig (vertikale Alignment-Muster)
- Performance Spectrum als komplementäre Visualisierung
- Korrekte Interpretation: Administrative Steps = schnelle Batches; spätere Steps = externe Wartezeiten (Bürgerverhalten)

**Organisatorische Perspektive:**
- Fehlen des `org:resource`-Attributs korrekt identifiziert und als strukturelle Limitierung eingeordnet (< 1% Fill Rate, nur Placeholder). Dies ist die einzig korrekte Reaktion bei diesem Datensatz.

### 2.4 Predictive Process Analytics ✅ (Exzellent)

**Flache ML-Modelle:** Logistic Regression, Random Forest, XGBoost — alle korrekt implementiert mit class weighting und Early Stopping.

**Deep Learning:** LSTM mit Embedding → LSTM → Dropout → Linear — architektonisch angemessen für Sequenzdaten.

**Validierung:** Temporaler Split (64/16/20) — **methodisch korrekt**. Keine Random-Splits, keine Leakage durch zeitliche Überlappung. Dies ist der Gold-Standard für Predictive Process Monitoring nach Teinemaa et al. (2019).

---

## 3. Erfüllung der Kür-Anforderungen (Optional Tasks)

| # | Aufgabe | Status | Qualität |
|---|---------|--------|----------|
| 1 | Systematischer Modellvergleich | ✅ | CF vs. DA × k∈{2,3,5} — vollständiges Experimental Grid |
| 2 | Interpretability (XAI) | ✅ | SHAP TreeExplainer auf XGBoost, Summary Plots für alle Konfigurationen |
| 3 | Prescriptive Analytics | ✅ | Threshold-basiertes Tier-System mit Kosten-Nutzen-Analyse |
| 4 | Conformance Checking | ✅ | 5 juristische Regeln basierend auf dem Codice della Strada — exzellent |
| 5 | Generative AI | ✅ | Markov-Kette mit JSD-Qualitätskontrolle |
| 6 | Python Implementation | ✅ | Vollständiger Pipeline-Code mit modularer Struktur |
| 7 | Deployment | ✅ | Streamlit-Dashboard mit 6 interaktiven Tabs |

**7 von 7 optionalen Aufgaben wurden erfüllt.** Dies ist die maximale Abdeckung.

### Besondere Würdigung:

**Conformance Checking (Aufgabe 4)** verdient besondere Hervorhebung. Die Gruppe hat nicht einfach generische Compliance-Regeln formuliert, sondern fünf spezifische Regeln aus dem italienischen Straßenverkehrsrecht (Art. 201, 202, 203, 204 CdS) abgeleitet. Die Ergebnisse sind inhaltlich relevant:
- 52.14% Compliance bei der 90-Tage-Frist → schwerwiegendes reales Compliance-Problem
- 100% bei der Penalty-Grace-Period → systematische Einhaltung
- 89.21% Payment Accuracy → systemischer Verarbeitungsfehler identifiziert

**Prescriptive Analytics (Aufgabe 3)** geht über eine reine Empfehlung hinaus und liefert eine formal begründete Expected-Gain-Formel:
$$E[\text{Gain}] = p \cdot \text{Benefit}_{\text{prevented}} - \text{Cost}_{\text{action}}$$

Die Gruppe räumt ehrlich ein, dass die Kostenparameter illustrativ sind — eine wichtige wissenschaftliche Einschränkung.

---

## 4. Kritische Würdigung von Methodik & Interpretation

### 4.1 Herausragende Interpretationstiefe

Die **zentrale Erkenntnis** des Berichts — und das, was ihn von einer durchschnittlichen Arbeit unterscheidet — ist die folgende Analyse (Section 5, Discussion):

> *„Cases whose second activity is Payment have a credit collection rate of only 0.1%, while cases whose second activity is Send Fine escalate in 74% of cases. Since these two activities account for over 99% of all second events, the second activity alone is nearly sufficient to determine the outcome."*

Diese Einsicht erklärt, **warum** alle Modelle nahezu identisch performen und warum ein LSTM keinen Mehrwert bietet. Die Gruppe hat verstanden, dass die Vorhersagbarkeit eine **Eigenschaft des Prozesses** ist, nicht des Modells. Das ist eine tiefgreifende methodische Erkenntnis, die in der Literatur häufig als „early predictability" diskutiert wird und die viele Studierende nicht erreichen.

### 4.2 Methodische Stärken

1. **Temporal Split:** Keine Information aus der Zukunft fließt in das Training — korrekte Simulation eines Deployment-Szenarios.
2. **Leakage-Prävention auf Feature-Ebene:** Outcome-revealing Activities werden nicht nur als Label verwendet, sondern explizit aus dem Input-Vokabular entfernt.
3. **Baseline-Vergleich:** Majority Baseline (Outcome) und Mean Baseline (Remaining Time) ermöglichen eine faire Einordnung der Modellqualität.
4. **Overfitting-Awareness:** Die Konvergenz aller Modelle wird nicht als Erfolg, sondern als Indikator für Problem-Einfachheit interpretiert.

### 4.3 Kleinere methodische Anmerkungen

- **Remaining Time MAE (≈100 Tage):** Die Gruppe interpretiert dies korrekt als inherente Prozessunsicherheit, die durch externe Faktoren (Bürgerverhalten) getrieben wird. Eine Diskussion, ob dieser MAE für ein Deployment *ausreichend* wäre, fehlt jedoch.
- **LSTM-Hyperparameter:** Es werden keine Details zu Learning Rate, Hidden Size oder Trainings-Epochen genannt. Für Reproduzierbarkeit wäre eine kurze Tabelle wünschenswert.
- **Markov-Modell (DFC Coverage 41.8%):** Die Gruppe erkennt die Limitierung richtig, aber ein kurzer quantitativer Vergleich mit einem 2nd-order Markov-Modell wäre ein lohnender Absatz gewesen.
- **Sample Size für Prescriptive (200 Cases):** Die Evaluation auf nur 200 Test-Cases ist etwas knapp für statistisch robuste Aussagen über die Tier-Verteilung.

---

## 5. Reflexion der Limitierungen

Die Gruppe hat eine **bemerkenswert ehrliche und wissenschaftlich fundierte** Limitierungsdiskussion geliefert (Section 9.2):

| Limitierung | Bewertung der Reflexion |
|-------------|------------------------|
| Fehlende Ressourcen-Daten | ✅ Korrekt identifiziert; kein Versuch einer unsinnigen Workaround-Analyse |
| Tagesgenaue Timestamps | ✅ Direkte Konsequenz: keine Intra-Day-Analyse möglich |
| Historische Daten (2000–2013) | ✅ Generalisierbarkeit explizit eingeschränkt |
| Illustrative Kostenparameter | ✅ Transparenz über Modellierungsannahmen |
| Model Saturation | ✅ Die wichtigste Meta-Erkenntnis: Das Problem ist zu einfach für komplexe Modelle |

Besonders lobenswert: Die Gruppe hat auch beim Generative-AI-Ansatz die Limitierung (keine Long-Range Dependencies, keine realistischen Timestamps) explizit benannt und Higher-Order-Modelle als Verbesserung identifiziert.

---

## 6. Fazit & Geschätzte Benotung

### Gesamtbewertung

| Dimension | Note |
|-----------|------|
| Pflichtanforderungen (Data Prep, EDA, Discovery, Predictive) | 1.0 |
| Optionale Aufgaben (7/7 erfüllt) | 1.0 |
| Methodik und Validierung | 1.0 |
| Interpretation und kritische Reflexion | 1.0 |
| Schreibstil und Strukturierung | 1.3 |
| Limitierungsdiskussion | 1.0 |

**Anmerkung zum Schreibstil:** Der Bericht ist klar strukturiert und gut lesbar. Kleinere Abzüge ergeben sich aus dem gelegentlichen Fehlen technischer Details (LSTM-Hyperparameter) und der etwas knappen Diskussion der Remaining-Time-Ergebnisse im Kontext eines tatsächlichen Deployments.

### Gesamtnote: **1.0 (Sehr gut / Exzellent)**

Dieser Bericht gehört zu den besten, die in dieser Veranstaltung eingereicht werden können. Die Gruppe demonstriert:
- Vollständige methodische Kompetenz über alle geforderten und optionalen Tasks
- Tiefgehendes Verständnis der Ergebnisse (nicht nur der Metriken)
- Wissenschaftliche Ehrlichkeit bei Limitierungen
- Professionelle Software-Engineering-Praxis (modularer Code, Dashboard, reproduzierbare Pipeline)

Die zentrale Erkenntnis — dass die Prozessstruktur selbst die Vorhersagbarkeit determiniert und komplexe Modelle keinen Mehrwert bieten — ist eine reife analytische Schlussfolgerung, die über das typische Studierendenniveau hinausgeht.

---

*Ende des Gutachtens*
