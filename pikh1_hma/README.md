# Active Learning to train PLMs

## Objective

- Test different strategies to efficiently fine-tune ESM, using an active learning approach.

## Approaches

1. Ensemble approach
   - Fine-tune a set of 5-10 ESM2 models with different random initializations of the regressor.
   - Use each of these models to predict on the rest of the dataset (this would normally be de novo generated mutants)
   - Calculate the mean and variance of these predictions for each sequence.
   - Update the model with a batch of sequences with the highest variants.
2. Monte Carlo Dropout
   - Fine-tune a single ESM2 model on a small, initial subset of data.
   - Use this model to predict on the rest of the dataset, keeping the dropout layer in the regression head active.
   - Repeat this multiple times, then calculate the mean and variance for each sequence.
   - Update the model with a batch of sequences with the highest variants.
3. Mean variance estimation
   - Rather than outputting a single value from the regression head, fine-tune ESM2 to output two values, one representing the predicted enrichment score, the other representing the predicted uncertainty.
   - Train this model using Gaussian Negative Log-Likelihood (NLL) loss. This will penalize incorrect predictions with high confidence.
   - Do an initial training set, use the variance to select new mutants, update the model.

## Notes

- Be sure to reserve a universal test set.
- When selecting new mutants for retraining, it would be best to keep these diverse. We have a limited, predetermined dataset to work with in these experiments, but play around with ESM2 embedding clustering and using this to maximize diversity and uncertainty.
- The environment manager used for this project is Pixi, we will see how well it works.
