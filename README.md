# ICAIC Elimination Submissions

This repository contains my Jupyter Notebook submissions for the four APOAI 2025 Mock Competition questions as part of the ICAIC eliminations, together with two additional experiments.

## Main submissions

| Question | Notebook                                       | Task                                            | Method                          |
| -------- | ---------------------------------------------- | ----------------------------------------------- | ------------------------------- |
| 1        | [Submission](Question1/submission_model.ipynb) | Predict basketball shot outcomes                | Multilayer perceptron           |
| 2        | [Submission](Question2/submission_model.ipynb) | Classify real and generated images              | Convolutional neural network    |
| 3        | [Submission](Question3/submission_model.ipynb) | Classify news articles by topic                 | Word embeddings and LSTM        |
| 4        | [Submission](Question4/submission_model.ipynb) | Estimate pendulum parameters and predict motion | Physics-based parameter fitting |

## Recorded local results

| Question | Evaluation                | Result                                    |
| -------- | ------------------------- | ----------------------------------------- |
| 1        | 5,000-shot local holdout  | 60.36% accuracy; majority baseline 55.26% |
| 2        | 2,000-image local holdout | 76.50% accuracy; majority baseline 50.00% |
| 3        | 194-article local holdout | 0.820178 macro F1; 81.96% accuracy        |
| 4        | Fit to recorded angles    | Approximately 0.000379605 radians RMSE    |

### Note

Question 3 generates `submission.csv` when `test_news_nolabel.csv` is supplied using the saved training vocabulary and original category names.

Question 4 generates `submission_train.csv` from the supplied training data and generates `submissionA.csv` and `submissionB.csv` when the corresponding hidden datasets are available.

## Additional experiments

- [Question 1 extension](Question1-Extension/submission_model.ipynb): compares input features, learning rates, and training lengths.

- [Question 3 extension](Question3-Extension/experiment.ipynb): compares changes to an initial LLM-suggested training recipe using validation macro F1.

Note that the exploratory notebooks are separate from the main submissions.
