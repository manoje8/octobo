**Naive Bayes** is a _generative_ model. It models how the data was generated for each class — specifically, it estimates P(features | class) and P(class), then uses Bayes' theorem to compute P(class | features). It makes the "naive" assumption that all features are conditionally independent given the class label. This assumption is almost never true in practice, but the model still works surprisingly well, especially for text classification (spam filtering, sentiment analysis).

**Logistic Regression** is a _discriminative_ model. It doesn't bother modeling how the data was generated — it directly learns the decision boundary by modeling P(class | features) as a sigmoid function of a weighted sum of the features. It's trained by optimizing (usually via gradient descent) to minimize a loss function like log-loss.

Key differences:

|                             | Naive Bayes                                           | Logistic Regression                                          |
| --------------------------- | ----------------------------------------------------- | ------------------------------------------------------------ |
| Type                        | Generative                                            | Discriminative                                               |
| Core assumption             | Feature independence given class                      | Log-odds is linear in features                               |
| Training                    | Just count/estimate probabilities (fast, closed-form) | Iterative optimization (gradient descent)                    |
| Data efficiency             | Works well with small data                            | Usually needs more data                                      |
| Handles correlated features | Poorly (violates independence assumption)             | Well                                                         |
| Output                      | Probabilities from Bayes' rule                        | Probabilities from sigmoid                                   |
| Typical use case            | Text classification, spam detection                   | General-purpose classification, when features are correlated |
