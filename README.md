# Salmon Environmental Signal Strategy

Trades salmon futures from free public data: ocean temperature and weather at
every major Atlantic salmon farming region, plus Norway's weekly export price
and volume.

```
pip install -r requirements.txt
python main.py              # first run downloads ~2 min of data, then it's cached
python main.py --refresh    # weekly: pull the new SSB week + latest ocean/weather
```

Options: `--horizon 8`, `--model gbm`, `--weather power`, `--test-start 2015-01-01`,
`--cost-bps 50`, `--leverage 1`, `--no-vol-cap`, `--months 4-9`, `--no-plots`.
Everything else lives in `config.py`.

## Data (no API keys)

| Source | What | Used for |
|---|---|---|
| [SSB table 03024](https://www.ssb.no/en/statbank/table/03024) | Weekly export price (NOK/kg) and tonnes of fresh farmed salmon, 2000-present | Target, momentum, supply |
| [NOAA OISST v2.1](https://psl.noaa.gov/data/gridded/data.noaa.oisst.v2.highres.html) | Weekly 0.25° sea surface temperature | Growth and heat-stress signals |
| [ERA5 via Open-Meteo](https://open-meteo.com/en/docs/historical-weather-api) (default) | Daily max wind, solar radiation | Storm and algal-bloom signals |
| [NASA POWER](https://power.larc.nasa.gov/) (`--weather power`) | Same variables | Alternative, lagged 8 weeks (see below) |

The farm locations are the 13 Norwegian production areas plus Chile (Los Lagos,
Aysén, Magallanes), Scotland, Shetland, the Faroes, Iceland and Canada. Each is
weighted by its approximate share of global harvest (`config.FARM_REGIONS`)
and grouped into Norway, Chile and "other".

## How it works

1. **Target.** Futures already price the normal seasonal cycle (spring highs,
   autumn lows), so predicting that cycle earns nothing. The model predicts the
   *seasonally adjusted* spot return over the next 4 weeks. That is the actual
   log change minus the average change for those weeks over the last 8 years.
   It starts one week after the data week, because SSB publishes week *t* on
   Wednesday of week *t+1*.
2. **Features** (all known at publication time):
   - Market: 1/4/13-week price momentum, the seasonally adjusted 4-week change, and price vs. its 52-week average.
   - Supply: export volume year on year.
   - Environment, per region group:
     - Sea temperature anomaly over 4, 26 and 52 weeks.
     - Heat-stress weeks above 16 °C.
     - Max-wind anomaly.
     - Solar-radiation anomaly.

   Anomalies are measured against a rolling 10-year baseline for the same ISO week.
3. **Model.** Ridge regression, fitted walk-forward. It refits every 13 weeks on
   labels that were already known, and is out of sample from 2010.
4. **Trading.** Each week opens a tranche sized by the forecast's z-score,
   capped at ±1. Tranches are held for 4 weeks, so the book is their average.
   Costs are 25 bp per unit of position change.
5. **Sizing.**
   - **Vol cap:** when the book's recent P&L volatility (13-week half-life) runs
     above its own historical median, the book shrinks in proportion. It never
     scales up.
   - **Leverage:** the book is levered 1.3x, which returns max drawdown to where
     it was without the cap.
   - **No-trade windows:** none by default. Windows can be added in
     `config.no_trade_windows`.

## Results (run of 2026-09-29, 2010-2026 out of sample)

| Features | IC | IC t (overlap-adj.) | Sharpe (calendar) | Sharpe (active) | Ann. return | Max DD |
|---|---|---|---|---|---|---|
| all | 0.204 | 3.06 | 0.77 | 0.78 | 12.7% | -28% |
| market + supply | 0.183 | 2.73 | 0.63 | 0.64 | 10.6% | -32% |
| environment only | 0.059 | 0.87 | 0.17 | 0.17 | 3.2% | -51% |

Definitions:

- **Calendar Sharpe and annual return** count every week, with flat weeks as
  zero. This is what the capital earns per year.
- **Active Sharpe** uses only weeks in the market (97% of weeks, excluding
  the start-up period), annualised by √52. It is the right measure
  when the capital is used elsewhere while flat.
- **IC** covers every week.

Effect of the sizing on the `all` book (older run, COVID window flat, so the
absolute numbers are slightly lower than the default above):

| Sizing | Sharpe (calendar) | Ann. return | Ann. vol | Max DD |
|---|---|---|---|---|
| 1x, no vol cap | 0.62 | 9.2% | 14.8% | -28.9% |
| 1x, vol cap | 0.69 | 8.7% | 12.6% | -21.6% |
| 1.3x, vol cap (default) | 0.69 | 11.4% | 16.4% | -28.1% |

With no COVID window the default is Sharpe 0.77, 12.7% return, -28.1% max DD.

**COVID.** Earlier versions forced the book flat from 2020-03-02 to 2020-06-28.
That window was picked with hindsight, so it was removed. The strategy was short
into the 2020 crash and gained about 27 log-points from March to mid-June, so
the headline numbers include a gain that looks more like luck than signal.

The cap works because the strategy's two big drawdowns (2011-13 and spring
2022) came with spikes in its own P&L volatility. It cut size to about half in
2012 and about two-thirds in 2022. It also improved both halves of the sample
separately (Sharpe 0.11 -> 0.18 for 2010-17, 1.02 -> 1.10 for 2018-26).
These alternatives were tested and dropped:

| Alternative | Result |
|---|---|
| Market-vol targeting | Salmon's own volatility is too stable to flag trouble. Sharpe 0.68. |
| Drawdown brake | Mean reversion recovers right after its drawdowns, so cutting size there hurts. Sharpe 0.53-0.61. |
| Averaging 4/8/13-week horizons | The longer horizons are much weaker, so averaging diluted the 4-week book. Sharpe 0.58. |

Most of the edge is in the price itself: short-term mean reversion and
reversion to the 52-week average. The environmental data adds a modest,
consistent lift. It comes mainly from **storm weeks**: after an unusually
windy week in the farming regions, prices underperform their seasonal path.
The wind coefficient is negative for all three region groups. The likely cause is harvest held back by the storm reaching the market
afterwards. The effect has the same sign in 2003-14 and 2015-26. On its own the
environmental data has no reliable signal.

## Seasonality: when the model works

The model is much sharper from April to September. In those months:

- sea temperatures rise and growth accelerates;
- the spring price peak gives way to the summer collapse as new-generation fish arrive;
- heat stress and algal blooms are live risks.

In winter, growth is slow and harvests are planned, which leaves less to predict.

| Quarter (return week) | Sharpe, all years | 2010-17 | 2018-26 | Forecast IC |
|---|---|---|---|---|
| Q1 Jan-Mar | 0.41 | -0.33 | 0.92 | 0.09 |
| Q2 Apr-Jun | 1.09 | 0.67 | 1.36 | 0.29 |
| Q3 Jul-Sep | 1.49 | 1.32 | 1.62 | 0.31 |
| Q4 Oct-Dec | -0.06 | -0.52 | 0.51 | 0.06 |

`--months 4-9` trades only April to September (2010-26, 1.3x; run with the
COVID window flat, not re-run since):

| | In market | Sharpe (calendar) | Sharpe (active) | Ann. return | Max DD |
|---|---|---|---|---|---|
| All year (default) | 95% | 0.69 | 0.71 | 11.4% | -28.1% |
| Apr-Sep | 49% | 0.84 | 1.21 | 9.6% | -27.4% |

The default stays all year for three reasons:

- **Apr-Sep was chosen after seeing the table above.** In a walk-forward test
  where each year's quarters are picked using only prior years, active Sharpe
  improves less (0.87 -> 1.04 over 2013-26), and calendar Sharpe doesn't
  improve (0.86 -> 0.81).
- **Winter is weaker but still profitable since 2018.** Dropping it cuts the
  annual return.
- **The max drawdown barely moves.** The worst drawdown (spring 2022) falls
  inside the season, so the Apr-Sep book can't be levered up to beat the
  all-year return.

Picking individual months walk-forward is worse still (active Sharpe 0.68).
With about 70 weekly observations per month, month-level Sharpe estimates
carry roughly ±0.9 of noise.

Apr-Sep is the better choice if the capital does something else from October
to March. In that case the active Sharpe is the relevant number.

## Caveats

- **This is a spot proxy, not a futures backtest.** P&L is the seasonally
  adjusted SSB return. Real futures also price in what traders know about
  biomass reports, harvest guidance and so on, so live returns will be lower.
  The next step is to load Euronext salmon futures settlement prices and score
  against them.
- **Some settings were chosen after seeing out-of-sample results.** These are
  the 10-year rolling climatology, dropping cold-snap features, ridge
  alpha = 200 and the vol cap. Each has a physical justification (`config.py`),
  but treat the numbers above as optimistic.
- **The 1.3x leverage is sized to the backtest's own worst drawdown.** A future
  drawdown can be deeper than any in the sample, and salmon futures are thin
  enough that size costs more than the flat 25 bp assumed here.
- **NASA POWER is not usable for same-week signals.** It splices its
  near-real-time GEOS-IT stream onto MERRA-2 history for roughly the last 1-2
  months. The spliced wind is biased: Chilean max wind drops 2-7 m/s overnight
  on 2026-09-01. With `--weather power` the features are lagged 8 weeks, so only
  MERRA-2 data is used. That lag removes the storm signal: all-feature IC falls
  to 0.17, below market + supply alone.
- Open-Meteo's free API is for non-commercial use. A full ERA5 download of all
  22 regions uses most of its hourly quota, so `--refresh` only re-pulls the
  last 90 days.

## Ideas for more signal

- BarentsWatch fish health API (barentswatch.no, free OAuth key): weekly sea
  lice counts and sea temperature reported by every Norwegian farm.
- Fiskeridirektoratet monthly biomass, harvest and mortality by production area.
- NOK/EUR exchange rate. Salmon trades in EUR, and the SSB price is in NOK.
