# What do 920 heart examinations tell us?

I started this project with a broad question: which measurements and symptoms are related to coronary artery disease, and how far could I go in explaining those relationships? I used the four historical UCI Heart Disease files because they let me ask the same question in different medical centers. The records describe people at examination. They do not tell me when their disease began, how treatment changed their measurements, or who was never referred. That shaped the rest of the analysis.

[Read the illustrated study (PDF)](study.pdf) · [Download the interactive report (HTML)](index.html). Download the HTML file and open it in a browser to explore the figures offline.

## The question that guided the analysis

**When angina was recorded during exercise, how often was the artery examination positive?** Among the 865 examinations with a known exercise response, 282 of 337 with angina were positive (83.7%), compared with 192 of 528 without it (36.4%). Reversing the question changes the denominator: among the 474 positives with a recorded response, 192 had no angina during exercise. A symptom can accompany disease frequently and still be absent in many positive examinations.

## How I chose the methods

| Question | Method and reason | What I learned |
| --- | --- | --- |
| Did the hospitals examine similar groups? | I counted positive examinations separately in each center before pooling them. I used Wilson intervals to show the uncertainty in each proportion. | Positive shares ranged from 36.1% in Hungary to 93.5% in Switzerland. The pooled 55.3% reflects the mix of people examined. |
| Could that mix explain the symptom gap? | I calculated the two positive shares within each center, then averaged each set using the same center weights from the 865 complete symptom records. This keeps the comparison in percentage points. | The gap fell from 47.3 to 42.3 points (approximate 95% interval 36.6–48.0). |
| Could the 55 missing symptom responses reverse the result? | The diagnoses were known, so I assigned the 35 positives and 20 negatives to opposite symptom groups in both extreme combinations. This does not require guessing which individual had angina. | Across all 920 records, the **unweighted** gap ranged from 38.7 to 50.2 points. This is a separate calculation from the center adjustment. |
| Did people with opposite results have very different measurements? | I compared medians, histograms and violin plots. The histograms use percentages within each group so unequal sample sizes do not determine bar height. | Median ages were 51 and 57 years in the negative and positive groups, with overlapping ranges. Median blood pressure was 130 mmHg in both. |

### Why logistic regression?

The outcome I wanted to explain had two values: disease present or absent. I used logistic regression to compare the odds of a positive examination while accounting for a few measured differences between people. I fitted **five separate models**, one for each question. Each pooled model allowed a different starting level for each center and assumed a common odds ratio across its included centers.

| Comparison | Other variables in the model | Records | Odds ratio (95% interval) |
| --- | --- | ---: | --- |
| Age, per 10 additional years | Recorded sex and center | 920 | 1.71 (1.42–2.05) |
| Recorded male vs. female | Age and center | 920 | 3.97 (2.67–5.92) |
| Cholesterol, per 50 mg/dL | Age, recorded sex and center | 718 | 1.34 (1.16–1.56) |
| Admission blood pressure, per 10 mmHg | Age, recorded sex and center | 860 | 1.14 (1.04–1.24) |
| Angina during exercise, present vs. absent | Age, recorded sex and center | 865 | 8.23 (5.65–12.00) |

Odds compare positive examinations with negative ones. For example, if a group had 20 positives and 80 negatives, its odds would be 0.25. Multiplying by 1.71 gives odds of about 0.43, equivalent to a probability near 30%. The 20% starting probability is an illustration; the model does not say that every individual gains ten percentage points over ten years.

## Two decisions that changed what I could conclude

- **Check what a positive label means.** I used diagnosis codes 1–4 for presence and 0 for absence. Raising the cutoff to 2 would drop the count from 509 to 244. The processed Hungarian file groups all 106 positives under code 1. The original documentation lists 69 of them in higher categories. That discrepancy makes comparisons of severity across centers unreliable.
- **Use the records needed for each question.** I treated zero cholesterol and blood-pressure readings as unavailable. Every Swiss cholesterol reading was zero. Requiring all 14 fields would leave only 299 records, including 297 from Cleveland. The cholesterol model therefore describes a different group than the age model.

## Where I stopped

The study supports an association between exercise angina and disease recorded at the same examination. It also shows how much that comparison changes with center composition and missing answers. To study whether a symptom predicts disease developing later, I would need dated observations and follow-up of people initially free of the diagnosis.

## Reproduce the calculations

Download the original ZIP from the [UCI Heart Disease dataset](https://archive.ics.uci.edu/dataset/45/heart+disease). From the repository directory, run the following with Python 3.12, replacing the ZIP path if needed.

~~~sh
python -m pip install -r requirements.txt
python fit_models.py "heart+disease.zip" results
python check_angina.py "heart+disease.zip"
~~~

The first script writes pooled and center-specific model estimates, sample-size checks, curvature comparisons and the alternative diagnosis counts. Compare its generated estimates with [model_estimates.csv](model_estimates.csv). The second script prints the symptom counts, common-weight comparison and extreme missing-response scenarios.

Data credit: Andras Janosi, William Steinbrunn, Matthias Pfisterer and Robert Detrano. *Heart Disease*, UCI Machine Learning Repository. [DOI](https://doi.org/10.24432/C52P4X) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
