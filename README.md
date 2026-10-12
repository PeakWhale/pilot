<img src="pilot.svg" alt="PeakWhale Pilot logo" width="112" height="112">

# PeakWhale Pilot

### Academic performance forecasting

A linear-regression demo that estimates final mathematics grades from earlier grades, study time, failures, and absences.

Originally created in 2019 as `adrkv/GradePredictor`, this educational project is now part of [PeakWhale](https://github.com/PeakWhale). Its original dataset and commit history are preserved. It is kept as an example of early work.

## What it does

- Reads the included `student-mat.csv` dataset.
- Uses `G1`, `G2`, `studytime`, `failures`, and `absences` to predict `G3`.
- Fits a scikit-learn linear regression model with a random 90% training / 10% test split.
- Prints the test score, coefficients, intercept, and predicted versus actual grades.

The script reports **R-squared**, not classification accuracy. The original README described approximately 80% to 90% accuracy, but that wording should not be treated as validated performance. There is no fixed random seed, so results vary between runs.

## Files

| File | Purpose |
| --- | --- |
| `Tensor_GradePrediction.py` | Original training and prediction script |
| `student-mat.csv` | Dataset used by the script |

## Running the demo

The script uses Python with pandas, NumPy, and scikit-learn.

```bash
pip install -r requirements.txt
python Tensor_GradePrediction.py
```

Run from the repository root so the CSV can be found. The script was updated in 2026 to work with current pandas and scikit-learn, and the train/test split now uses a fixed random seed, so results repeat between runs. With the current seed the test-set R-squared is about 0.75.

This is an educational regression example, not a validated system for student-risk assessment or academic eligibility decisions. The original repository does not document the dataset's provenance or reuse terms.

## License

No explicit license was specified in the original repository. Moving this project into PeakWhale does not add a reuse license for its code or dataset.

## Author

[Addy Kaveti](https://adrkv.github.io/) · [PeakWhale](https://github.com/PeakWhale)
