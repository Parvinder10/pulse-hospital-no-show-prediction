# PULSE: Exploratory Data Analysis & Empirical Findings

**Dataset Split:** Chronological Training Set (77,364 records)

## 1. Class Imbalance Profile
- **Attended (0):** 59,412 (76.8%)
- **No-Show (1):** 17,952 (23.2%)
- **Imbalance Ratio:** ~3.3:1 negative to positive ratio.
- **Methodological takeaway:** Standard accuracy is misleading; evaluation must focus on PR-AUC, Recall, Precision, and calibrated probabilities.

## 2. Lead Time Dynamics
| Lead Time Horizon | Volume | No-Show Rate |
| :--- | :--- | :--- |
| Same Day (0d) | 20,563 | 5.06% |
| 1-2d | 7,784 | 22.85% |
| 3-7d (1w) | 12,991 | 26.08% |
| 8-14d (2w) | 9,091 | 31.82% |
| 15-30d (1m) | 16,557 | 32.78% |
| 31-60d (2m) | 8,283 | 34.15% |
| 60d+ | 2,095 | 28.45% |

**Key Takeaway:** Same-day bookings have a tiny no-show rate (~4.6%), whereas bookings scheduled more than 2 weeks out experience elevated no-show rates approaching 32-34%.

## 3. Prior Patient History Signal
| Prior No-Shows | Volume | Current No-Show Rate |
| :--- | :--- | :--- |
| 0 prior missed | 67,192 | 21.89% |
| 1 prior missed | 7,880 | 30.53% |
| 2 prior missed | 1,607 | 33.67% |
| 3 prior missed | 471 | 37.15% |
| 4 prior missed | 117 | 51.28% |

**Key Takeaway:** Past missed appointments are strongly associated with higher future no-show risk, confirming the predictive validity of our point-in-time cumulative features.

## 4. Visualizations
- Target Distribution: `reports/figures/target_distribution.png`
- Lead Time Curve: `reports/figures/lead_time_vs_noshow.png`
- Age Distribution: `reports/figures/age_distribution.png`
- Weekday Pattern: `reports/figures/weekday_pattern.png`
