# The ask (step 1 input — paste this with `data.csv`, `covariance.csv`, and `covariance_recession.csv`)

> We're allocating a portfolio across 30 ETFs (`data.csv`): expected return, volatility,
> and dividend yield per fund, plus each fund's asset-class group. Portfolio risk is
> covariance-driven — `covariance.csv` carries the pairwise matrix from our analyst.
>
> The decision is each fund's weight — weights total 100%. Maximize return and yield
> (weighted averages); minimize portfolio volatility (through the covariance matrix).
>
> Hard rules:
> - No single fund above 30%.
> - At most 3 active holdings per asset-class group.
> - Portfolio volatility at or below 20%.
>
> Three macro futures to stress-test:
> - **Rate cuts** — yields read 15% lower across the board.
> - **Inflation** — returns −10% and yields +15% across the board.
> - **Recession** — co-movement changes: swap in `covariance_recession.csv`;
>   everything else unchanged.

This brief plus the three CSVs is the complete upstream input: framing it should land on
the canonical model (proportional; Return/Yield avg maximize + Volatility quadratic
minimize; the 30% cap; five ≤3 asset-class group limits; the volatility ≤20 bound; all
three scenarios).
