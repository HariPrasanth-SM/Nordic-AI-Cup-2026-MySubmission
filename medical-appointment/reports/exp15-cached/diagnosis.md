# Run diagnosis

Accuracy **0.9897**; tIoU **0.6343**; score **0.7765**.
All 390 gold questions are included; unsent/failed requests count wrong.

## Error categories

| Category | Count |
|---|---:|
| too_narrow | 30 |
| too_wide | 31 |
| false_negative | 3 |
| strong | 56 |
| boundary_shift_or_partial | 56 |
| correct_no | 194 |
| nonoverlap_review_occurrence | 19 |
| false_positive | 1 |

## Oracles and paired diagnostics

Oracles use gold and are ceilings, not deployable predictions. Means below cover only rows with the required trace.
Candidate oracle covers LLM-proposed hypotheses only; it is not an exhaustive retrieval oracle.

| Metric | Mean | Positive rows covered |
|---|---:|---:|
| word_oracle | 0.89459217029172 | 195 |
| candidate_oracle | 0.6595261033912743 | 192 |
| aligned_word_oracle | None | 0 |
| baseline_tiou | None | 0 |
| pre_alignment_tiou | 0.5992359814891036 | 192 |

## What to change next

- Many nonoverlapping intervals: read gold_text versus pred_text; inspect occurrence choice. Timing alone cannot repair this.
- Candidate oracle high, selected tIoU low: selection/prompt convention is the bottleneck.
- Word oracle high, candidate oracle low: broaden source hypotheses and boundary choices.
- Aligned word oracle higher but actual tIoU lower: better acoustic timestamps disagree with annotation convention; reject the alignment change.
- Many false negatives: inspect ASR numbers/negations and classification threshold before localization.
- Missing trace/stage fallbacks: fix execution before judging model quality.

## Twenty worst positive spans

### sample_4_yes_q06: tIoU 0.000
Was the patient listened to with a stethoscope?
Gold 28.520000000000003–42.72: Let me listen to your chest and your heart, and then we will talk. Go ahead. Breathe normally for me. And again. Is it alright? Your chest and heart both sound normal.
Prediction None–None: 

### sample_6_yes_q05: tIoU 0.000
Does the patient also have a fungal infection in the mouth?
Gold 72.96–77.6: And fluconazole, 50 milligrams, for seven days, for the mouth.
Prediction 12.84–14.96: And I have got thrush in my mouth as well.

### sample_17_yes_q01: tIoU 0.000
Will the treatment last two weeks?
Gold 48.18–50.7: After a meal every day for 2 weeks.
Prediction 42.13–45.52: 100 mg daily for 2 weeks.

### sample_20_yes_q05: tIoU 0.000
Is Pamol one of the medicines requested?
Gold 182.44–184.98: am creating prescriptions for both PAMEL
Prediction 42.6–48.32: and IBUMETIN. Those are the two I use, and I have been taking them for a good while now.

### sample_23_yes_q02: tIoU 0.000
Does the patient want the skin changes removed?
Gold 85.28–88.5: I would still rather have them off than keep catching them
Prediction 19.76–20.9: I want them gone.

### sample_23_yes_q04: tIoU 0.000
Is an appointment for the removal going to be arranged?
Gold 90.2–92.2: the appointment for removal is the plan
Prediction 67.66–69.92: I will arrange an appointment for the removal

### sample_33_yes_q03: tIoU 0.000
Has the medication helped the patient?
Gold 55.34–58.28: the medicine is genuinely working? Yes.
Prediction 25.71–26.6: It has helped.

### sample_37_yes_q04: tIoU 0.000
Are redness and swelling absent?
Gold 17.58–22.18: redness at all. Any swelling? No swelling either.
Prediction 51.46–55.38: redness, no swelling, and no rash at the site.

### sample_37_yes_q01: tIoU 0.000
Did the patient get a tick bite?
Gold 13.74–14.52: tick bite.
Prediction 7.13–9.3: I found a tick attached to my leg.

### sample_57_yes_q03: tIoU 0.000
Was it concluded that no changes to treatment are needed?
Gold 117.6–122.54: continue your lifestyle exactly as you are and you continue your current medication.
Prediction 100.59–101.64: changes are needed.

### sample_63_yes_q02: tIoU 0.000
The lipid profile came back normal, didn't it?
Gold 0.0–0.26: Good
Prediction 50.71–52.2: lipid profile is normal

### sample_64_yes_q01: tIoU 0.000
Was the condition assessed as xerosis?
Gold 72.26–73.98: condition is called cirrhosis.
Prediction None–None: 

### sample_64_yes_q02: tIoU 0.000
Did the examination rule out any wounds?
Gold 0.0–0.16: Good
Prediction 60.06–64.44: is dry, with some light flaking. There are no sores.

### sample_66_yes_q02: tIoU 0.000
Is the patient free of any reaction after the injection?
Gold 93.14–96.36: Perfectly fine, thank you. Nothing out of the ordinary.
Prediction 97.06–99.52: And I can see no reaction at all.

### sample_70_yes_q04: tIoU 0.000
The vaccination went as planned, didn't it?
Gold 80.46000000000001–83.34: Then I can say that there were no side effects today.
Prediction 54.83–58.62: is the vaccination given, exactly as planned for this season.

### sample_70_yes_q01: tIoU 0.000
Were no further measures decided on?
Gold 116.98–118.92: extra measures from my side.
Prediction 113.27–116.16: for what comes next, nothing further is needed.

### sample_77_yes_q03: tIoU 0.000
Was the infection judged to follow a penetrating trauma?
Gold 49.56–52.96: thorn broke the skin, and the infection came after that.
Prediction 46.48–48.94: is an infection following the penetrating injury.

### sample_77_yes_q01: tIoU 0.000
Does the patient have an infection in the big toe?
Gold 49.56–52.96: thorn broke the skin, and the infection came after that.
Prediction 13.84–19.2: big toe. I pricked it on a thorn about two weeks ago and now I think it has become infected.

### sample_79_yes_q05: tIoU 0.000
Do the pains come and go rather than being constant?
Gold 24.72–28.16: keep getting pain behind my right kneecap, and it keeps coming back.
Prediction 39.51–45.1: exactly. It comes and goes. Some days I barely notice it, and then it turns up again

### sample_79_yes_q04: tIoU 0.000
Was the suspicion reached without imaging?
Gold 80.88–87.4: From what you describe, I suspect a condition called chondromalacia patelli,
Prediction 102.23–107.08: want to be clear that this is a clinical suspicion at this stage, not something confirmed.
