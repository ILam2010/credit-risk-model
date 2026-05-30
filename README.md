# credit-risk-model

## Credit Scoring Business Understanding
1. Basel II and Interpretability

Basel II emphasizes accurate risk measurement, transparency, and regulatory compliance. Financial institutions must be able to explain how credit decisions are made and document the assumptions behind their models. Therefore, credit scoring models should be interpretable, auditable, and reproducible. Well-documented models help regulators verify that lending decisions are fair, consistent, and based on measurable risk factors.

2. Why a Proxy Variable is Necessary

The Xente dataset does not contain a direct default label indicating whether customers failed to repay loans. Since supervised machine learning requires target labels, a proxy variable must be created. In this project, customer behavior measured through Recency, Frequency, and Monetary (RFM) analysis will be used to identify customers who appear disengaged and therefore potentially higher risk.

Using a proxy target introduces business risks because the proxy may not perfectly represent true loan default behavior. Misclassification may occur, leading to good customers being denied credit or risky customers receiving loans.

3. Trade-offs Between Interpretable and High-Performance Models
Logistic Regression with WoE

Advantages:

Easy to explain
Regulatory friendly
Stable and transparent
Supports scorecard development

Disadvantages:

May miss complex relationships
Lower predictive performance
Gradient Boosting Models

Advantages:

Higher predictive accuracy
Captures nonlinear patterns
Handles complex interactions

Disadvantages:

Harder to explain
More difficult to validate
Greater regulatory scrutiny

In regulated banking environments, interpretability is often prioritized, although high-performance models may be used when supported by strong documentation and monitoring procedures.