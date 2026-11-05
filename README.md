# LQDF Portfolio Construction 2026/27

Research and development repository for the Portfolio Construction team at the Loughborough Quantitative & Discretionary Fund (LQDF).

## Project aim

Develop and compare portfolio allocation methods to understand how they affect diversification, risk and performance. The project covers model research, Python implementation, backtesting and evaluation using a consistent framework.

## Models

Our primary models are:

- **Mean–Variance Optimisation (MVO):** constructs portfolios using estimated returns and covariances.
- **Hierarchical Risk Parity (HRP):** uses clustering and asset relationships to allocate portfolio risk.

Optional extensions include Equal Risk Contribution (ERC) and covariance shrinkage.

## Research approach

1. Review the models and document their assumptions.
2. Prepare the data and agree common input/output formats.
3. Implement MVO and HRP.
4. Compare both models against a simple equal-weight benchmark.
5. Evaluate performance, risk, turnover and allocation stability.
6. Document findings, limitations and potential improvements.

Backtests should use consistent datasets and assumptions, avoid look-ahead bias, and account for transaction costs where possible.

## Repository organisation

The planned structure is:

- `src/` — reusable Python code for models and evaluation.
- `notebooks/` — exploratory analysis and research experiments.
- `tests/` — checks for model behaviour and data processing.
- `docs/` — methodology notes and research findings.

The repository is currently being set up. Installation and execution instructions will be added as the code develops.

## Team workflow

- Develop changes on a separate branch with a descriptive name.
- Use clear commit messages explaining what changed.
- Open a pull request for review before merging into `main`.
- Document assumptions and explain how results were produced.
- Use the shared Excel tracker to record progress and blockers.
- Review progress during our Friday meetings.

## Project documents and data

The research overview, project timeline and data feeds tracker are maintained in our [shared Google Drive folder](https://drive.google.com/drive/folders/1U81Ei2gT2FsnlDGAvDCA3XVIdR_clmQ0).

Do not commit API keys, passwords or restricted datasets. Record data sources and access requirements so other team members can reproduce the analysis.

## Project lead

Caner Karakoc — Director of Portfolio Construction
