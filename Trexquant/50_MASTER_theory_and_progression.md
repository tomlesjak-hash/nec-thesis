# Alpha research: theory, motivation, and progression

*Master reference for the Trexquant interview. Synthesises notes 14–44.
Every figure was measured on Trexsim unless stated otherwise.*

---

## Contents

1. [The problem, the Sharpe constraint, and how IC and IR relate](#1)
2. [Phase 1 — infrastructure, and six bugs that would have invalidated the work](#2)
3. [Genetic programming — theory](#3)
4. [Gate theory](#4)
5. [Regime and conditioner theory](#5)
6. [The mechanism, and the evidence that overturned it](#6)
7. [The submitted alphas — full specification, theory, and development](#7)
8. [Method: what generalised](#8)
9. [Errors, and how each was caught](#9)
10. [Interview preparation](#10)
11. [What I would do next](#11)

---

<a name="1"></a>
## 1. The problem, and the one constraint that shaped everything

### 1.1 The competition

Build alphas on Trexsim: expressions over a ~1,000-name US equity universe,
2006–2022 visible, evaluated out-of-sample. Each alpha is a stocks × days matrix
of positions built from 32 built-in operators. Score is
**rank(alpha count) + rank(combined out-of-sample performance)** — so breadth
matters as much as quality.

Qualification per alpha:

- **IR > 0.07** (daily information ratio)
- **≤ 50% correlation** to any alpha already in the pool
- override if correlated: IR/√TVR must be ≥ 20% higher than the alpha it collides with
- `numstk > 160`

### 1.2 The insight that should have come first

**IR 0.07 daily is an annualised Sharpe of 1.11.** Restating the bar this way
changes what is worth attempting:

| strategy | gross Sharpe (approx.) | clears 1.11? |
|---|---|---|
| Size (SMB) | ~0.22 | no |
| Value (HML) | ~0.32 | no |
| Momentum (UMD) | ~0.48 | no |
| Quality (QMJ) | ~0.6 | no |
| Betting-Against-Beta | ~0.8 | no |
| 52-week high | ~0.55 | no |
| **Short-term reversal** | ~1.3 | **yes** |
| **Residual reversal** | ~2× conventional reversal | **yes** |
| **Intraday microstructure** | 1.5–3 | **yes** |

*SMB/HML/UMD figures are annualised long-short Sharpes over 1973–2021. **These
are strongly period- and construction-dependent** — HML has been reported
anywhere from 0.11 to 0.64 depending on sample vintage and whether decile or
tercile portfolios are used, and daily-rebalanced decile versions since 1950
give SMB 0.14, UMD 0.67, HML 0.64. Cite them as orders of magnitude, not
precise values. The conclusion is robust to the range: no construction of these
factors reaches 1.11.*

**The entire slow-anomaly literature is mathematically incapable of clearing
this threshold.** Value, momentum, quality and low-beta are real, replicated
effects that cannot qualify — not because of implementation, but because their
standalone Sharpes are below the bar.

I learned this the expensive way. Two full sweeps were spent on fundamentals and
on beta/momentum/52-week-high before doing the arithmetic. Both returned
nothing, and the calculation that would have predicted it takes one line.

**Everything that qualified is fast, high-turnover, and microstructural.** That
was not a stylistic preference; it is the only region of the space where the
threshold is reachable.

### 1.3 IC and IR — the two numbers everything else is built on

These two appear throughout and measure different things. Getting them straight
makes the rest of the document readable.

#### IC — Information Coefficient: skill on **one bet**

$$\text{IC}_t = \text{corr}\big(\text{signal}_{i,t},\; \text{return}_{i,t+1}\big)$$

The correlation, **across stocks on a single day**, between what your signal said
and what actually happened next. Usually a rank correlation, so it is unaffected
by outliers.

- Range is [−1, +1], but **realistic values are 0.02–0.05.** An IC of 0.03 means
  your ranking is right about 51.5% of the time — a tiny edge on any one bet.
- IC is measured **across the cross-section, at a point in time.**
- It says nothing about how much money you make; it is pure ranking skill.

#### IR — Information Ratio: skill of the **whole strategy**

$$\text{IR} = \frac{\text{mean}(\text{daily PnL})}{\text{std}(\text{daily PnL})}$$

Return per unit of risk, measured **over time**. This platform reports the
**daily** figure; multiply by √252 for the annualised Sharpe:

| daily IR | annualised Sharpe |
|---|---|
| 0.07 (the qualification gate) | **1.11** |
| 0.10 | 1.59 |
| 0.124 (this pool's best) | 1.97 |

**IR is scale-free** — double every position and both numerator and denominator
double. Return is *not*, which is why §7.0 warns that a heavily-gated alpha with
half a book reports an inflated return while its IR is unaffected.

#### The bridge between them — Grinold's Fundamental Law

$$\boxed{\;\text{IR} \;\approx\; \text{IC} \times \sqrt{\text{BR}}\;}$$

where **BR is breadth** — the number of *independent* bets per year.

**The intuition: a tiny edge repeated independently many times becomes a good
strategy.** Skill per bet (IC) and number of bets (BR) are substitutes. A manager
with high skill on few bets and one with low skill on many can reach the same IR.

**A worked figure.** An alpha with daily IR 0.10 is annualised IR 1.59. At a
typical IC of 0.03 that implies

$$\text{BR} = (\text{IR}/\text{IC})^2 = (1.59/0.03)^2 \approx 2{,}800 \text{ independent bets per year}$$

against 700 names × 252 days = **176,400 raw name-days**. So only about **1.6% of
name-days are effectively independent** — bets on the same day share common
factors, and consecutive days share a slow-moving signal. **Breadth is far smaller
than it looks**, which is why cutting it is expensive.

#### Where each one shows up in this document

| | used as |
|---|---|
| **IR** | the qualification metric (> 0.07), the tuning objective, the GP fitness function |
| **IC** | the *reason* breadth matters (§4.3 splits, §5.2) and the weight inside alpha 9 |

**Two consequences that recur:**

1. **A conditioner trades breadth for IC, and pays only if the IC gain exceeds the
   √breadth loss.** Halving the universe costs √2 ≈ 1.41 in breadth, so a split
   needs the retained half to have an IC at least 41% higher (§7.6).
2. **Ranking a noisy IC is more defensible than using it raw** — ranking keeps the
   ordering and discards a magnitude you cannot estimate reliably. That is exactly
   what alpha 9 does with `cs_rank(ts_corr_binary(...))`.

---

<a name="2"></a>
## 2. Phase 1 — infrastructure

Built a point-in-time panel to search on locally: yfinance daily bars, S&P 500
membership reconstructed from constituent-change history, 2006–2022,
4,279 days × 584 tickers, monthly-rebalanced top-500-by-ADV universe.

Validation caught six bugs, each of which would have silently corrupted every
downstream result:

| bug | consequence if unfixed |
|---|---|
| `auto_adjust=False` mixes conventions | adjusted close sat **below** unadjusted low on 77% of bars |
| dollar volume computed from adjusted price | pre-split turnover understated by the split factor, distorting universe selection |
| universe mask accumulated forward | names removed from the index stayed members forever — **full survivorship bias** |
| mask re-admitted names after halts | mid-month additions, indistinguishable from look-ahead |
| cache keyed on ticker, not date range | a 2010-start run silently accepted 2015–2024 data |
| five reused ticker symbols | CPWR, EP, MI, COL, SLE — different companies wearing dead tickers |

**Documented and unfixable:** 212 of 796 index members (26.5%) are absent
because the data source purges delisted tickers. The gap concentrates in
companies that went to zero — Lehman, Countrywide, Fannie, Freddie, Ambac — the
worst possible shape of bias for a sample containing 2008.

The point worth making: **none of these were found by reading code.** All six
came from writing explicit invariant checks — adjusted close must lie within the
day's range, universe membership must be monotone within a segment, cached files
must cover the requested window. Bugs of this class do not raise exceptions.

---

<a name="3"></a>
## 3. Genetic programming — theory

### 3.1 The formal object

GP searches a **term algebra**. Given a function set $F$ (operators, each with an
arity) and a terminal set $T$ (variables and constants), the set of well-formed
expressions $\mathcal{T}$ is defined inductively: every terminal is a tree, and
if $g \in F$ has arity $k$ and $t_1 \ldots t_k \in \mathcal{T}$, then
$g(t_1,\ldots,t_k) \in \mathcal{T}$. The problem is

$$\arg\max_{t \in \mathcal{T}} f(t), \qquad f : \mathcal{T} \to \mathbb{R}$$

Three properties make this unlike continuous optimisation:

- **Discrete** — no derivatives exist, so no gradient method applies.
- **Variable-dimension** — trees do not live in a fixed $\mathbb{R}^n$, so there
  is no parameter vector to write down.
- **No intrinsic metric** — `add(x,y)` and `div(x,y)` differ by one symbol and
  are semantically unrelated; `x+x` and `2*x` are syntactically distant and
  semantically identical.

**The consequence that matters: the variation operators define the topology.**
In gradient descent the space supplies a notion of "nearby" and the algorithm
follows it. In GP there is no such notion until you choose crossover and
mutation — subtree crossover declares two programs adjacent if one subtree swap
separates them. Different operators give a different neighbourhood graph over the
same set of programs, hence a different optimisation problem.

### 3.2 The algorithm

1. Initialise a population of $N$ random trees.
2. Evaluate $f$ on each.
3. Select parents with probability increasing in fitness (tournament: sample $k$,
   keep the best).
4. Produce $N$ children by **subtree crossover** (swap a random subtree between
   two parents) and **subtree mutation** (replace a random subtree with a fresh
   random one).
5. Repeat for $G$ generations; retain the best ever seen.

Crossover is the only thing distinguishing GP from repeated random restart: it
**reuses substructures that already scored well** rather than rebuilding from
scratch.

### 3.3 The classical theory, and its limits

Holland's **schema theorem**. A schema $H$ is a template with wildcards;
$m(H,t)$ counts instances at generation $t$:

$$\mathbb{E}[m(H,t+1)] \;\ge\; m(H,t)\,\frac{f(H)}{\bar f}\Big[1 - p_c\frac{\delta(H)}{\ell-1} - p_m\, o(H)\Big]$$

with $f(H)$ the mean fitness of instances, $\delta$ the defining length, $o$ the
order. **Short, low-order, above-average schemata proliferate roughly
exponentially** — the *building-block hypothesis*: the algorithm assembles small
good fragments into larger good solutions.

Its limits are worth knowing. The bound is loose; it is a one-generation
statement about expectations; and it describes how *existing* schemata propagate
while saying nothing about where new ones come from. Poli and Langdon derived
exact schema theorems for GP, but variable tree shape makes the accounting
unwieldy and non-predictive.

### 3.4 The fitness landscape — the framing that predicts behaviour

Treat $(\mathcal{T}, N, f)$ as a graph with neighbourhood $N$ induced by the
operators, and ask three questions:

- **Ruggedness** — the autocorrelation of $f$ along a random walk. Low
  autocorrelation means local information is worthless.
- **Neutrality** — the fraction of the graph that is fitness-plateau.
- **Deception** — whether local structure points away from the global optimum.

GP landscapes are **extremely neutral**, because the map genotype → phenotype →
fitness is many-to-one twice: vast numbers of syntactically distinct trees
compute the same function, and many distinct functions produce identical fitness.
Most of the search is drift across plateaus rather than ascent.

### 3.5 Why "search" is the wrong word — the size of the space

With $|T|$ terminals, $u$ unary and $b$ binary primitives, $w$
window-parameterised operators over $W$ window values, the count of trees of
height $\le h$ satisfies

$$C(h) = |T| + u\,C(h{-}1) + b\,C(h{-}1)^2 + w\,W\,C(h{-}1)$$

For the configuration used here (11 terminals per seed, 5 unary, 4 binary, 11
window operators, 4 windows):

| height | distinct trees |
|---|---|
| ≤ 1 | 1.0 × 10³ |
| ≤ 2 | 4.3 × 10⁶ |
| ≤ 3 | 7.5 × 10¹³ |
| ≤ 4 | 2.2 × 10²⁸ |
| **≤ 5** | **2.0 × 10⁵⁷** |

Against 1.57 × 10⁶ evaluations, the covered fraction is **8 × 10⁻⁵²**.

**A GP is therefore a Markov chain on populations**, with transition kernel
(selection ∘ crossover ∘ mutation), concentrating mass on high-fitness regions
*reachable from the initialisation*. It is not enumeration and not global
optimisation. Two accurate alternative framings: **beam search with a stochastic
fitness-weighted beam**, or an **implicit estimation-of-distribution algorithm**
in which the population is a sample from a distribution reweighted each
generation.

### 3.6 Bloat, and the theory that explains it

Programs grow without fitness improving. Three explanations:

- **Introns / hitchhiking** — neutral code buffers offspring against destructive
  crossover, so it accumulates.
- **Removal bias** — deleting a subtree is more damaging than adding one, so
  selection is asymmetric in size.
- **Crossover bias** (Poli, Dignum & Langdon) — the one with real force. Subtree
  crossover *preserves* mean program size but changes the size *distribution*,
  driving it toward a Lagrange distribution of the second kind: heavy-tailed,
  with large mass on very small programs. Small programs are usually unfit, so
  selection removes that tail and the mean drifts upward.

**Bloat is a sampling artifact of the operator plus selection, not an adaptive
response.** That is why a depth cap and a size penalty work so bluntly — you are
truncating a distribution, not countering an incentive.

### 3.7 The bound that governs everything

**No Free Lunch** (Wolpert & Macready): averaged over all objective functions,
every search algorithm performs identically. GP's value is therefore entirely its
**inductive bias** — the prior that good solutions are compositional, built from
simple operators on the inputs. For formulaic alpha discovery that prior is
largely correct, which is what makes GP a reasonable choice. *The bias is the
value; the search is machinery.*

And since GP is a fitness-maximising machine that is agnostic about meaning:

$$\text{quality of result} \;\le\; \text{quality of } f$$

If $f$ is misspecified, GP finds the maximum of the wrong function efficiently.
Measured here: the rank correlation between local fitness and platform IR was
**−0.70**, so selection was systematically pointed away from the objective. A
better-engineered search on the same $f$ changed nothing, which is the expected
outcome rather than a surprise.

**Specification gaming follows directly.** With fitness $= \text{IR}/\sqrt{\text{TVR}}$,
the objective is unbounded as $\text{TVR} \to 0$. The search located that
singularity independently in 16 of 26 runs, producing `div(ret20, ret20)` = 1
everywhere at TVR ≈ 0.0018. Five separate guards were required — dispersion
floor, turnover floor, position count, **effective N** $=1/\sum w_i^2$ (inverse
Herfindahl, which catches books holding 99% in one name while reporting 400
positions), and a denominator floor.

### 3.8 The frontier

**Geometric semantic GP** (Moraglio, Krawiec & Johnson) operates on *semantics* —
the vector of outputs on the training points — rather than syntax. For regression
under a metric loss, geometric semantic operators make the fitness landscape a
**cone: unimodal, with no local optima whatsoever.** The optimisation becomes
provably easy; the cost is that offspring size grows exponentially, since
semantic crossover is literally $\alpha t_1 + (1-\alpha)t_2$ as an expression
(implementations keep pointers rather than materialising trees).

It reframes GP's difficulty as an artifact of searching in syntax space when the
objective lives in semantic space.

---

<a name="4"></a>
## 4. Gate theory

### 4.1 Definition and taxonomy

A gate is a non-negative multiplier applied before a signal becomes a book:

$$\alpha_{i,t} = \text{core}_{i,t} \times w_{i,t}, \qquad w \ge 0$$

| dimension | options | consequence |
|---|---|---|
| **shape** | binary $\{0,1\}$ / graded $[0,\infty)$ | book on-or-off / always on, sized by conviction |
| **domain** | time-series ($w_{i,t} = w_t$) / cross-sectional | removes *days*, full breadth on active days / removes *names*, every day |
| **source** | exogenous (calendar) / endogenous (price) | invisible to a price-only search / discoverable, usually already in the core |

### 4.2 Why gating raises IR

Let the core have daily PnL $r_t$. Gate to a fraction $p$ of days: $w_t = 1$ on
the active set $S$, $0$ elsewhere. Write $\mu_S,\mu_N$ for mean PnL on active and
inactive days, $\sigma$ for daily volatility.

**Ungated**, the mean blends both regimes:

$$\text{IR}_{\text{full}} = \frac{p\,\mu_S + (1-p)\,\mu_N}{\sigma}$$

**Gated**, inactive days contribute a zero to both moments. The mean scales by
$p$; the variance also scales by $p$, since zeros add no variance:

$$\mathbb{E}[r] = p\,\mu_S, \qquad \text{Var}(r) \approx p\,\sigma^2, \qquad \text{IR}_{\text{gate}} = \frac{\sqrt{p}\,\mu_S}{\sigma}$$

**The mean falls linearly in $p$; the standard deviation falls only as $\sqrt p$.**
That asymmetry is the entire mechanism:

$$\text{GAIN} = \frac{\sqrt p\,\mu_S}{p\,\mu_S + (1-p)\,\mu_N}$$

**Special case** — the signal earns nothing outside the window ($\mu_N = 0$):

$$\boxed{\;\text{GAIN} = 1/\sqrt p\;}$$

| $p$ | 0.20 | 0.30 | 0.40 | 0.50 | 0.70 | 0.90 |
|---|---|---|---|---|---|---|
| gain | 2.24× | 1.83× | **1.58×** | 1.41× | 1.20× | 1.05× |

**Condition for gating to help at all.** Setting $\text{GAIN} > 1$:

$$\frac{\mu_N}{\mu_S} \;<\; \frac{\sqrt p}{1+\sqrt p}$$

At $p = 0.4$, the excluded days must carry **less than 38.7%** of the edge of the
included days. A gate that removes days where the signal works nearly as well
*destroys* IR — the breadth loss is not compensated.

**Empirical check.** A1's core: 0.055 ungated → 0.091 gated, a gain of
**1.655×**. Inverting $1/\sqrt p$ gives an implied $p = 0.365$ against a
documented ~40% coverage — the theory predicts the observed gain to within 9%
from one parameter. Solving the general form at $p = 0.40$ gives
$\mu_N/\mu_S = -0.03$.

**The interpretation this forces:** the gate is not finding days when the signal
is *stronger*. It is deleting days when the signal is *worthless* and only
contributes variance. If events amplified the signal, an intensity weighting
would add further; if they merely separate signal from noise, a binary switch
captures the entire effect.

### 4.3 Breadth, `numstk`, and the time-series privilege

Grinold's **Fundamental Law**: $\text{IR} \approx \text{IC} \times \sqrt{\text{BR}}$,
with breadth BR the number of independent bets. **A gate trades breadth for IC,
and pays iff the IC gain exceeds the $\sqrt{\text{breadth}}$ loss.**

`numstk` is positions-per-day averaged over 120 days, floor 160. The formula is
identical for both gate types:

$$\text{numstk} = p \times N \quad\text{(time-series)}, \qquad \text{numstk} = q \times N \quad\text{(cross-sectional)}$$

| conditioner | type | names/day | active days | `numstk` | IR |
|---|---|---|---|---|---|
| earnings ≤ 12 | cross-sectional | ~156 | 100% | **156** ✗ | 0.081 |
| earnings ≤ 16 | cross-sectional | ~204 | 100% | 204 ✓ | 0.071 |
| **8-event calendar** | **time-series** | ~1000 | ~40% | **~400** ✓ | **0.094** |

Same arithmetic, opposite outcome — and the reason is what each conditioner is
*made of*. **A market-wide event is shared by every stock**, so $p$ can be small
while breadth *on each active day* remains $N$: diversification within the day is
untouched, only the number of days falls, and §4.2 shows that is free when the
excluded days carry no edge. **An earnings window is per-stock**, so it removes
names from every day and the $\sqrt{\text{BR}}$ penalty applies unconditionally.

> **Rule: when a conditioner costs coverage, ask whether the same economic idea
> can be expressed in the time-series dimension instead.**

**A third case.** A cross-sectional variable used as an *additive term inside a
graded weight* costs no coverage, because the time-series terms already keep the
book invested: `stress = fomc + opex + ... + (days_to_earnings ≤ 12)` never
zeroes anything, it only raises the weight on names near reporting. The
distinction is between a conditioner that **multiplies to zero** and one that
**adds to a floor**.

### 4.4 Turnover — count blocks, not days

Turnover is the change in positions, normalised by book:

$$\text{tvr}_t = \frac{\sum_i |p_{i,t} - p_{i,t-1}|}{\text{booksize}}$$

With $w_t \in \{0,1\}$ shared across stocks, there are four cases:

| $w_{t-1} \to w_t$ | positions | turnover |
|---|---|---|
| on → on | shift with the signal | normal signal churn |
| **on → off** | all go to **zero** | **100% of book** |
| **off → on** | built from **zero** | **100% of book** |
| off → off | nothing held | 0 |

**A transition costs a full book turnover** — the largest possible single-day
trade. So each contiguous active block costs exactly **two** full turnovers,
regardless of the signal.

$$\text{tvr} \;\approx\; \underbrace{s}_{\text{signal churn}} \;+\; \underbrace{\frac{2 \times (\text{blocks per year})}{252\,p}}_{\text{gate churn}}$$

**Gate churn depends on the number of blocks, not on $p$.** If gate days were
i.i.d. Bernoulli($p$) the transition rate would be $2p(1-p)$, but calendar events
are strongly persistent and arrive in blocks, so that expression is an upper
bound and overstates the effect several-fold.

**Empirical check.** Simulating the actual A3-era gate over 2007–2021 and
widening `mend`:

| `mend` | blocks/yr | gate churn (predicted) | measured TVR | residual = signal churn |
|---|---|---|---|---|
| 2 | 17.1 | 0.173 | 0.348 | **0.175** |
| 5 | 14.5 | 0.134 | 0.292 | **0.158** |
| 10 | 8.2 | 0.070 | 0.228 | **0.158** |
| 15 | 8.2 | 0.070 | 0.226 | **0.156** |

The residual is **flat at ~0.16** across all four settings, which is the test:
widening the gate should not change how fast the *signal* churns. The model also
predicts the flattening between 10 and 15 — by 10 days the month-end window has
merged with the neighbouring block and widening adds nothing.

**Widening a gate reduces turnover by merging separate event windows into fewer,
longer blocks**, not by trading less.

### 4.5 Design principles

1. **Exogenous to price.** A price-derived conditioner is discoverable by the
   core itself and adds no information the search did not already have.
2. **A union, not an intersection.** An intersection is active on almost no days;
   §4.2 shows the IR gain saturates while §4.4 shows churn is worst at moderate
   coverage. Doubling the event count from four to eight raised IR *and* active
   days together.
3. **Time-series, for the breadth reason in §4.3.**
4. **One window at a time, whole range walked.** The largest single gain in the
   project (+0.025) came from moving one bound by one day.

---

<a name="5"></a>
## 5. Regime and conditioner theory

Nine conditioners were tried. Four appear in submitted alphas; five were tested
and rejected. This section covers what each one *is*, why the four worked, and
why the five did not — with only as much theory as each outcome needs.

### 5.1 Every conditioner tried

| conditioner | construction | type | outcome |
|---|---|---|---|
| **event calendar** | 8–10 calendar terms summed | time-series | **in 6 of 9** (not 1, 2) |
| **low volatility** | `cs_rank(ts_std(ret1, N)) ≤ 1.5` | cross-sectional | **in alphas 6, 7, 8** |
| **cross-sectional earnings window** | `days_to_earnings` in/out of a band | cross-sectional | **in alphas 1, 2** |
| **mandated-flow share** | `ts_sum(volume·stress, 250) / ts_sum(volume, 250)` | cross-sectional weight | tested, **not submitted** — best measured result |
| high volatility | complement of low-vol | cross-sectional | lost to low-vol, 3 cores |
| small size | complement of large | cross-sectional | lost on every core tried |
| low / high beta | `cs_rank(ts_corr_binary(ret1, ret1_spx, 60))` | cross-sectional | both lost |
| low / high price | `cs_rank(close)` | cross-sectional | both lost |
| macro volatility regime | VIX level, VIX shock, realised mkt vol, drawdown, VIX momentum | time-series (state) | **added nothing to any core** |

Two structural families, and they behave completely differently.

### 5.2 The cross-sectional split — the arithmetic

A split multiplies the alpha by a 0/1 mask that differs **per stock**:
`alpha = core × w × lovol`. Three consequences, all measurable.

**Coverage halves.** `numstk` is positions-per-day averaged over 120 days, floor
160. A median split takes ~735 names to ~368 — comfortably clear, but a
*quarter* (two splits stacked) lands near ~190, which is legal with little
margin.

**Gross exposure does not halve.** Measured on A5: 368 names, **booksize 0.96**.
The platform renormalises the book, so a split costs *coverage* but not
*exposure*. That makes splitting a much better trade than the naive reading
suggests — you are not running half a book, you are running a full book over half
the names.

**Breadth falls, so IC must rise to compensate.** By Grinold,
IR ≈ IC × √BR. Halving the name count costs a factor √2 ≈ 1.41 in breadth, so a
split only pays if the retained half has an IC at least 41% higher than the
full universe. That is a demanding bar, and it is why most of the splits failed.

**The measured outcome on A1's core:**

| split | IR |
|---|---|
| whole universe | 0.091 |
| high beta | 0.076 |
| **low volatility** | **0.089 → 0.094 tuned** |

Low-vol cleared the bar; high-beta did not.

### 5.3 Why low volatility won — the chain that explains it

This is the result that matters most: it recurs on three unrelated cores (alpha 6
reversal, alpha 7 earnings-cycle, alpha 8 intraday skew), and it is
**backwards** under the obvious reading. If volatile *days* pay more, volatile
*stocks* should too.

**The resolution is that a volatility split is not really a volatility split. It
is a sort on who owns the stock.**

Four steps, each either definitional or an uncontroversial empirical fact:

1. **The alpha needs forced traders.** It is paid for absorbing trades from
   someone who must transact regardless of price. No mandated flow, no fee.
2. **Mandated flow comes from index funds and ETFs.** They trade on rebalancing,
   expiration and month-end dates because they are compelled to, not because they
   have a view — the premise of the event gate in §4.
3. **Index weights are market-cap based.** So passive ownership concentrates in
   the largest companies.
4. **Large companies are low-volatility companies.** Apple moves less day to day
   than a small biotech.

**Therefore "low volatility" and "heavily index-owned" describe roughly the same
set of stocks**, and sorting on volatility sorts, incidentally, on flow
composition.

**Concretely.** Apple is held by every S&P 500 fund, every total-market fund and
every broad ETF; on a rebalancing day, billions of dollars of Apple trades for
reasons that have nothing to do with Apple. That is the counterparty this
strategy exists to trade against — and Apple sits in the low-volatility half. A
small biotech is barely indexed; its volume is people betting on trial outcomes,
which is an opinion rather than an obligation — and it sits in the high-volatility
half.

**So the finding is not about volatility at all.** It is that the alpha works
best where forced traders are, and the volatility split located them by accident.

#### The confirmation — measuring ownership directly

If the volatility split works *because* it accidentally finds index-heavy names,
then measuring index ownership directly should work better. That is `mflow`:

$$\text{mflow}_i = \frac{\sum_t v_{i,t}\,g_t}{\sum_t v_{i,t}}$$

with $g_t$ the event-day indicator. In words: **what fraction of this stock's
annual volume arrives on rebalancing and expiration days?**

- Index-fund-owned stock → **high**, because index funds only trade on those days.
- Stock-picker-owned stock → **low**, because they trade when news breaks, spread
  evenly through the year.

Substituting `mflow` for the volatility proxy on the same core took IR from
**0.094 to 0.106** — the theory predicted the improvement before the run and
delivered it. *That variant was measured but not submitted, so it stands as
evidence for the mechanism rather than as an alpha in the pool.*

#### What is documented and what is mine

**Documented, and more strongly than I expected.** Passive investors preschedule
their rebalancing trades, so volume concentrates on reconstitution days — spikes
of **3× for CRSP Mid Cap to over 27× for the S&P 500**. Roughly **30% of US
equity volume now occurs in the final 30 minutes**, because index funds and ETFs
must execute at the closing auction to match the NAV strike, making market-on-
close the dominant end-of-day mechanism. And passive flows are *"systematic,
regular, and largely insensitive to price or valuation."* That is the
mandated-flow mechanism stated almost verbatim in the practitioner literature.

**Mine, not cited:** the four-step chain above as an explanation for *this*
result, and the `mflow` construction as a way to measure it. The chain uses only
uncontroversial links, but assembling it into an account of why a volatility
split works is my inference, not a citation.

**A weaker version I removed:** an earlier draft justified step 4 with a model
relating passive share to *idiosyncratic* volatility. That is contested — research
on passive ETF ownership finds higher passive ownership **raises** idiosyncratic
volatility, the opposite sign. The cap-based chain above reaches the same
conclusion without needing it.

### 5.4 Why size worked once and volatility elsewhere

A7 is the one alpha that splits on **size** rather than volatility, and it is
also the one whose core is a *liquidity* measure — dollars absorbed per unit of
price move. Its signal is only meaningful where there is enough depth to measure,
so the large half is where the variable has support at all, not where the premium
is concentrated. **Different reason, same direction** — both splits select
heavily-indexed mega-caps.

The small half lost on every core tried, which is worth noting because
short-horizon reversal is documented as *stronger* in smaller names. On a
top-1,000 universe that literature does not bind: the "small" half here is still
large-cap, and the flow-composition effect dominates whatever size effect remains.

### 5.5 Why beta and price splits failed

Both were tested on the same cores and both lost.

**Beta.** $\beta_i$ is a systematic-exposure measure, not a flow measure. Two
stocks with identical beta can have completely different passive shares — a
high-beta small tech name and a high-beta mega-cap bank sit in the same bucket.
The split does not sort on ownership, so it does not sort on anything the
mechanism cares about.

**Price.** Nominal price is close to independent of size, beta and volatility —
a \$40 and a \$400 stock can be the same market cap. The economic case is that
tick size is fixed at a cent, so the *relative* tick is ~10× larger on the \$40
name, which should change how much of short-horizon reversal is bid-ask bounce
rather than premium. On a top-1,000 universe that effect is evidently too small
to survive halving the breadth — it fails the √2 bar in §5.2.

**Both failures are informative:** they say the working splits are not "any
partition helps," they are specifically partitions that correlate with flow
composition.

### 5.6 Why the macro volatility regime added nothing

This is the one time-series *state* conditioner tried, and it failed on every
core. Two reasons, and neither requires much development.

**Classification error.** You never observe the regime $s_t$; you act on an
estimate $\hat s_t$ with accuracy $q$. For a binary regime the realised
separation is

$$\mathbb{E}[\mu \mid \hat s = H] - \mathbb{E}[\mu \mid \hat s = L] = (2q-1)(\mu_H - \mu_L)$$

so the gain scales with $(2q-1)$ — your advantage over a coin flip. A calendar
conditioner has $q = 1$ exactly: FOMC dates are published years ahead,
expirations fixed by exchange rule, month-end arithmetic. A volatility regime
inferred from a trailing window plausibly reaches $q \approx 0.7$, which
**discards 60% of the available edge before any other consideration.** Realised
volatility is persistent (half-life ~2–5 weeks) and episodes are front-loaded, so
a 20-day estimator is typically 15–50% into the regime before it fires.

**Episode count.** Statistical power comes from independent *episodes*, not days.
The event calendar supplies ~900 over fifteen years — FOMC 120, monthly
expirations 180, month ends and starts 180 each, and so on. Macro volatility
supplies **six**: 2008, 2010, 2011, 2015–16, 2018, 2020. Roughly 150:1 in count,
12:1 in standard error.

**The practical rule:** for any state-space conditioner, count the episodes before
reading the IR. Single digits means the alpha is one observation of a market
regime wearing fifteen years of clothing — and every standard metric (IR, TVR,
drawdown, `numstk`) is blind to that distinction.

*This cuts both ways: the rejection of the volatility channel also rests on those
same six episodes.*

### 5.7 The unifying result

| conditioner | what it actually identifies | worked? |
|---|---|---|
| event calendar | days when flow is mandated, $q = 1$, ~900 episodes | **yes** |
| `mflow` weight | passive ownership, measured directly from volume timing | **yes** |
| low-volatility split | stocks with high passive share, via the cap→low-vol link | **yes** |
| large-size split | stocks with high passive share, via market cap | **yes** |
| beta split | systematic exposure — uncorrelated with passive share | no |
| price split | tick-size economics — too weak to pay the √2 breadth cost | no |
| macro volatility regime | market risk appetite, $q \approx 0.7$, 6 episodes | no |

> **Every conditioner that worked selects on flow composition — who is trading
> and why. Every one that failed selects on something else.**

The three that worked cross-sectionally (`mflow`, low volatility, large size)
are three different measurements of **passive ownership** — directly, via
volatility, and via market cap. That is why they point the same way, and why the
direct measurement beat both proxies.

---

<a name="6"></a>
## 6. The mechanism, and the evidence that overturned it

**This is the most important section in the document, and the strongest thing I
have to talk about.**

### 6.1 The original thesis

Short-horizon reversal returns are compensation for **liquidity provision** —
the fee for taking the other side of someone's urgent trade and carrying the
inventory. That fee is a *price*, set by supply and demand for immediacy.
Nagel (2012), "Evaporating Liquidity," shows it is **predictable by expected
volatility**: when volatility rises, providers withdraw, supply falls and the
fee widens.

This anchored notes 14, 16 and 18. It made a clear, testable prediction: the
premium should be **largest where volatility is highest**.

### 6.2 Two independent falsifications

**Test 1 — calendar time.** Built a macro volatility-regime conditioner from
VIX level, VIX shock, realised market volatility, market drawdown and VIX
momentum. Applied it to every core in the pool. **It never added anything to
anything.**

**Test 2 — cross-section.** Split the universe at the median of trailing
volatility and ran the same core on each half. The prediction, written down
before the run: *"reversal pays most where volatility is highest, so this half
should carry most of the IR… if the low-vol half is just as good, the
liquidity-provision reading is wrong."*

| core | high-vol / high-beta half | **low-vol half** |
|---|---|---|
| A1 reversal core | 0.076 | **0.089 → 0.094 tuned** |
| mandated-flow weighted core | worse | **0.106** |
| earnings cycle position | worse | **qualified** |

**Three separate cores. The low-volatility half won every time.**

### 6.3 The revised mechanism

Both hypotheses sit under "liquidity provision", but they are different claims:

| | prediction | status |
|---|---|---|
| **Volatility channel** (Nagel) — providers withdraw when risk rises | high-vol names pay more | **rejected twice** |
| **Mandated-flow channel** — the fee is paid by traders who *must* transact regardless of price | premium largest where mandated flow is a large *share* of volume | **supported** |

Low-volatility stocks are large, stable, heavily index-weighted names. Their
idiosyncratic trading is light relative to the mechanical index, ETF and pension
flow they receive, so **mandated flow is a much larger fraction of their
volume**. That is exactly where a flow-fading strategy should be paid best.

**This also explains, retrospectively, a pattern I could not account for at the
time:** the calendar event gate worked on every core, while every
volatility-based conditioner failed. The gate is a direct measurement of when
mandated flow arrives. Volatility was a proxy for a mechanism that is not
operating.

### 6.4 Acting on the revision

If the mechanism is mandated flow, then measuring *which stocks receive it*
should beat using volatility as a proxy. Built a passive-ownership proxy — the
share of a stock's annual volume arriving on high-event days — and used it as a
continuous weight instead of the volatility split:

| | IR |
|---|---|
| volatility proxy (lovol split) | 0.094 |
| **direct mandated-flow measurement** | **0.106** |

The revision predicted the improvement and delivered it. That is the strongest
evidence in the project that the revised account is right.

### 6.5 What this does and does not refute — say this before being asked

Nagel's result is well established and correctly stated above: reversal returns
proxy the returns to liquidity provision, and they are *highly predictable with
the VIX*, with conditional Sharpe ratios rising sharply in turmoil. I have not
overturned that paper, and claiming so would be indefensible.

**Three limits on my evidence:**

1. **Universe.** Nagel's effect is documented on a broad cross-section. Mine is
   the platform's ~1,000 largest US names. Liquidity in mega-caps arguably never
   truly evaporates — market makers and index arbitrageurs remain willing at
   almost any VIX level — so the volatility channel may simply be inactive in
   this universe rather than absent generally.
2. **Sample.** 2007–2021 for the daily alphas, 2007-03 onward for the intraday
   ones. Nagel's strongest evidence comes from 2007–09; my window includes it
   but the alphas built on intraday data do not span its start.
3. **Non-independence.** The three volatility splits share a gate and a
   universe, so a common factor could produce the pattern.

**The defensible claim is therefore narrow and should be stated narrowly:** *on
a top-1,000 US universe over 2007–2021, conditioning on volatility — in calendar
time via VIX and cross-sectionally via realised volatility — did not improve any
of my reversal alphas, while conditioning on mandated-flow dates improved all of
them, and a direct measure of mandated-flow exposure beat the volatility proxy.*

That is a statement about where the premium is concentrated in large caps, not a
refutation of Nagel.

---

<a name="7"></a>
## 7. The submitted alphas — specification, theory, and development

Each entry opens with a **summary box** (what it does, which theory applies, what
concept is introduced here) and closes with a **recap box**. Concepts are
explained at their first appearance, so the section can be read straight through
without §4 and §5 in front of you.

### 7.0 The pool at a glance

| # | alpha | conditioner | split | IR | Sharpe | TVR | break-even | **capacity** |
|---|---|---|---|---|---|---|---|---|
| **1** | closing-auction VWAP displacement | earnings **far** (>10d) | none | 0.085 | 1.35 | 1.308 | 2.0 bp | $118m |
| **2** | reversal core, earnings-gated | earnings **window** (0–12d) | none | 0.116 | **1.84** | 0.830 | 8.8 bp | **$8m** ⚠ |
| **3** | rank-25 tuned core | 8-event calendar | none | 0.091 | 1.44 | 0.739 | 6.7 bp | **$448m** |
| **4** | GP vol-regime coupling | 8-event calendar | none | 0.073 | 1.16 | 0.251 | 11.2 bp | $356m |
| **5** | closing-hour skew | 8-event calendar | none | 0.102 | 1.62 | 1.380 | 1.4 bp | $156m |
| **6** | reversal core, zscore-80 | 8-event calendar | **lovol** | 0.094 | 1.49 | 0.666 | 4.2 bp | *[verify]* |
| **7** | earnings cycle position | 10-term (incl. earnings) | **lovol** | *[verify]* | | | | |
| **8** | ts_rank closing-hour skew | 10-term (incl. earnings) | **lovol** | 0.081 | 1.29 | 1.355 | 1.0 bp | *[verify]* |
| **9** | adaptive IC blend | 8-event calendar | none | 0.075 | 1.19 | 0.870 | 2.6 bp | $221m |

**Conditioner usage:** 2 of 9 use a cross-sectional earnings gate and no calendar
(1, 2); 4 use the calendar gate alone (3, 4, 5, 9); 3 use calendar plus a
low-volatility split (6, 7, 8).

**Two axes that do not track IR.** Break-even cost runs 1.0bp to 11.2bp, and
capacity runs **$8m to $448m — a 56× range**. Alpha 2 has the best Sharpe and the
worst capacity. An alpha that cannot be run at size is a research result, not a
strategy.

**A caution on comparing returns.** Reported return is per unit of *average* book,
and booksizes differ. Alpha 3 at booksize 0.48 reports 12.5% but earns 6.0% on the
same deployed capital as alpha 5's 4.9%. **IR is scale-free; return is not.**

---

### 7.A How the conditioner evolved

The pool reads best chronologically, because each generation fixed a defect in the
previous one.

| gen | conditioner | alphas | fixed what | cost |
|---|---|---|---|---|
| **1** | cross-sectional earnings window | 1, 2 | nothing yet — the original idea | destroys coverage and capacity |
| **2** | 8-event calendar union | 3, 4, 5, 9 | coverage: full universe on active days | gives up ~40% of the Sharpe |
| **3** | calendar + low-volatility split | 6, 7, 8 | *where* the premium sits, not just when | halves coverage again |

---

### 7.1 Alpha 1 — closing-auction VWAP displacement

> **In one line.** Short stocks whose closing price printed far above the final
> hour's volume-weighted average, betting the displacement was uninformed and
> reverts.
>
> **Theory it rests on.** Mandated flow concentrates in the closing auction (§4.5)
> · cross-sectional conditioners cost coverage (§4.3).
>
> **First appearance here.** The **conditioner** idea itself — multiplying a
> signal by a 0/1 mask.

```python
push = (close - vwap_last_hour) / at_zero2nan(vwap_last_hour)
z    = push / at_zero2nan(ts_std(push, 20))
far  = (trading_days_until_next_earnings_announcement > 10)

alpha = at_zero2nan(-np.tanh(z) * far)
```

**IR 0.085 · Sharpe 1.35 · return 6.7% · vol 4.97% · TVR 1.308 · DD 8.8 ·
Calmar 0.76 · IR/√TVR 0.074 · break-even 2.0bp · booksize 1.00 · 632 names ·
capacity $118m**

> **New concept — what a conditioner is.**
> `alpha = signal × w`, with `w ≥ 0`. Where `w = 0` you hold nothing; where
> `w = 1` you hold the signal. Here `w = far`, a per-stock flag. The signal
> decides *which* stocks to hold; the conditioner decides *when* — or, as here,
> *which*. Everything in this pool is some version of this one line.

**What it measures.** How far the closing price printed from the final hour's
volume-weighted average, normalised by that stock's own recent dispersion of the
same quantity.

**The theory.** The closing auction is where mechanical flow concentrates — index
funds must print at the official close to track NAV, ETF create/redeem settles
there, VWAP and MOC algorithms finish there. When that flow is one-sided it
**pushes the closing print away from where the stock actually traded through the
hour.** Because the flow is uninformed, the displacement carries no information
about value, and price reverts. **Shorting the displacement collects that
reversion** — hence the leading minus.

**Why divide by `ts_std(push, 20)`.** Raw displacement is not comparable across
stocks; a volatile name deviates from its VWAP more in normal conditions.
Dividing by each stock's own 20-day dispersion converts it into *"how unusual is
today's push for this stock."*

**Why `tanh`.** Bounded. A 10-sigma push should not get ten times the position of
a 1-sigma push — reversion is not linear in displacement, and unbounded weights
concentrate the book.

**Why gate AWAY from earnings.** This runs *opposite* to alpha 2, which wants to
be *near* earnings — and that is the interesting part. Alpha 2 is paid for
providing liquidity into event risk. **This one is betting a price move is
uninformed.** Near an announcement, late-day buying may be informed — someone
positioning ahead of a print. Fading informed flow loses money, so the gate
excludes the window where "this flow is mechanical" is least safe.

**The coverage cost is visible and matches the arithmetic.** Earnings are roughly
quarterly (~63 trading days apart), so `days > 10` excludes 11/63 = **17.5%** of
stock-days. Predicted `numstk` = 735 × 0.825 ≈ **606**; observed **631**. That is
the cross-sectional penalty of §4.3 measured directly.

**Provenance.** The expression is original. **The phenomenon is documented** —
Baltussen, Da & Soebhag on end-of-day reversal; the MOC-imbalance literature,
where the largest buy-imbalance decile beats the largest sell-imbalance decile by
~32bp into the close with **~83% reversing over 3–5 days**; Wu on passive funds
creating transient mispricing. **The least original alpha in the pool.** It does
use `vwap_last_hour`, absent from the Alpha101 dataset, so it cannot collide with
a competitor's Alpha101 submission.

> **Remember.** Fade uninformed closing-auction pressure, stand aside near
> earnings where the flow may be informed. **Capacity $118m, break-even 2.0bp** —
> a paper alpha standalone. Its opposite-signed earnings gate versus alpha 2 is
> the strongest single piece of evidence that the conditioners track something
> real rather than filtering noise.

---

### 7.2 Alpha 2 — reversal core, earnings-gated

> **In one line.** Short-horizon reversal — fade stocks that closed high in their
> range — restricted to names within 12 days of an earnings report.
>
> **Theory it rests on.** Reversal as liquidity provision (§4.2) · the earnings
> announcement premium · why cross-sectional gates destroy coverage (§4.3).
>
> **First appearance here.** The **liquidity-provision mechanism** that underpins
> four of nine alphas · the **coverage/capacity trade-off**.

```python
stoch  = (close - ts_min(low, 20)) / at_zero2nan(ts_max(high, 20) - ts_min(low, 20))
relvol = volume / at_zero2nan(ts_mean(volume, 20))
volret = cs_winsor(ts_std(ret1, 20) / at_zero2nan(ret20), 0.02, remove_extreme=False)

gate = (trading_days_until_next_earnings_announcement >= 0) \
     * (trading_days_until_next_earnings_announcement <= 12)

core = (-(ts_zscore(stoch, 10))
        - (((ts_median(cs_zscore(relvol), 20) - ts_mean(ret20, 20))
            + ts_skew(relvol - ts_delay(relvol, 3), 33))
           - ts_mean(-stoch - ts_median(volret, 3), 15)))

alpha = at_zero2nan(cs_winsor(core, 0.01, remove_extreme=False) * gate)
```

**IR 0.116 · Sharpe 1.84 · return 18.5% · vol 10.05% · TVR 0.830 · DD 13.7 ·
Calmar 1.35 · IR/√TVR 0.128 · break-even 8.8bp · booksize 0.58 · 156 names ·
Liq 14 / capacity $8m**

*(0.124 in the development log below is the pre-submission measurement.)*

> **New concept — reversal is a fee, not a forecast.**
> Short-horizon reversal returns are **compensation for liquidity provision**:
> the fee for taking the other side of someone's urgent trade and carrying the
> inventory until it can be unwound (Nagel 2012). The key consequence is that
> this fee is a **price**, set by supply and demand for immediacy — *not a
> constant*. It widens when demand for immediacy rises or when providers step
> back. **If the fee were constant there would be nothing for a conditioner to
> exploit**, and none of this pool would exist.

**The signal.** `stoch` is position inside the 20-day range; `ts_zscore(stoch,10)`
asks how unusual that is for this stock, and the minus inverts it — short names
that closed high. Two further terms condition on **participation**: whether the
name has been unusually busy, and whether *changes* in participation have been
one-sided. **Fade the extension, weighted by whether volume says the move was flow
rather than information.**

**Why the earnings window.** Two effects. **(a) Days 1–12:** approaching a
scheduled announcement, other mean-reversion traders reduce inventory — nobody
carries a position into a binary event — while institutions reposition. Supply of
liquidity falls, demand rises, the fee widens. **(b) Day 0:** the **earnings
announcement premium** (Savor & Wilson — firms scheduled to report earn ~9.9%
annualised abnormal return; over 60% of the equity risk premium is earned on
announcement days, 13% of the sample).

**Development.**

| change | IR | Δ |
|---|---|---|
| GP core, gate `≤ 90` | 0.063 | — |
| gate `≤ 60` | 0.069 | +0.006 |
| gate `≤ 15` | 0.078 | +0.009 |
| gate `≥ 3, ≤ 18` | 0.084 | +0.006 |
| gate `≥ 2, ≤ 18` | 0.086 | +0.002 |
| gate `≥ 1, ≤ 18` | 0.088 | +0.002 |
| **gate `≥ 0, ≤ 18`** | **0.113** | **+0.025** |
| internal window tuning | 0.117 | +0.004 |
| gate `≥ 0, ≤ 12` | 0.122 | +0.005 |
| remaining window tuning | **0.124** | +0.002 |

**Gate +0.055; every internal window combined +0.006.** The single largest step is
**one day** — including the announcement day itself added +0.025.

Each window walked individually: `ts_zscore(stoch,·)` at 5/10/15 (10 best),
`ts_median` at 10/20/30 (20), `ts_skew` at 10/20/30/33 (33), `ts_mean` at
5/10/15/20 (15), `ts_delay(relvol,·)` at 3/5 (3). What did nothing:
`cs_remove_middle` at any level — the signal is diffuse across the cross-section,
not concentrated in tails. What hurt: `ts_sum(core,5)` accumulation, `at_signsqrt`
on it, and normalising components to a common scale.

**What I got wrong.** The first theory argued the last ~3 days before an
announcement should be *excluded* — implied vol peaks, hedging dominates. The `≥3`
bound appeared to confirm it. **Wrong:** walking down 3 → 2 → 1 → 0 improved IR at
every step. The apparent confirmation was **confounded** — when `≥3, ≤18` beat
`≤15`, the upper bound moved at the same time, so I credited to the exclusion a
gain that came from the extension.

> **New concept — coverage and capacity are separate costs.**
> **`numstk`** is positions-per-day averaged over 120 days; the platform floor is
> **160**. This alpha holds **156** — *below it*. Note 16 predicted exactly this,
> because a cross-sectional gate removes names from every day.
>
> **`LiqN`** is the booksize at which the alpha's average daily stock volume
> reaches 1% of market volume — a **dollar capacity limit**, not a name count.
>
> | alpha | numstk | capacity |
> |---|---|---|
> | 3 | 365 | **$448m** |
> | 4 | 712 | $356m |
> | 9 | 738 | $221m |
> | 5 | 707 | $156m |
> | 1 | 632 | $118m |
> | **2** | **156** | **$8m** |
>
> **A 56× gap.** The best-Sharpe alpha in the book runs about eight million
> dollars before it moves 1% of the market in its names.

**The trade-off, measurable.** A widened variant (`≤ 16`, ~204 names) scored 0.071
against this one's 0.116 — tightening bought +0.045 of IR and cost most of the
capacity. **That curve is why generation 2 replaced this conditioner rather than
tuning it further.**

> **Remember.** Best Sharpe (1.84), best Calmar (1.35), **worst capacity by
> 56×**, and `numstk` below the floor. **A demonstration that the cross-sectional
> gate works, not a strategy that could be run.** Confirm whether it actually
> qualified.

---

### 7.3 Alpha 3 — rank-25 tuned core, 8-event calendar gate

> **In one line.** A 14-window tuned reversal expression, weighted by how many
> mandated-flow calendar events are live that day.
>
> **Theory it rests on.** The √p gate arithmetic (§4.2) · time-series versus
> cross-sectional conditioners (§4.3) · turnover from gate transitions (§4.4).
>
> **First appearance here.** **`stress` and the event calendar gate** — the single
> most important construct in the project, used by six of nine alphas.

```python
W_STOCH, W_SIZE, W_VOLSTD, W_VRMEAN = 10, 175, 12, 10
W_IN1, W_IN2, W_ZS, W_OUT           = 20, 17, 10, 10
W_MED, W_SKEW, W_MIN                = 20, 60, 20
W_S1, W_S2, W_S3, WINSOR            = 10, 20, 20, 0.005

fomc_pre  = (days_until_next_fomc_meeting <= 1)
fomc_post = (days_since_last_fomc_meeting <= 0)
opex      = (days_until_next_monthly_options_expiration <= 1)
qopex     = (days_until_next_quarterly_options_expiration <= 1)
qend      = (days_until_last_trading_day_of_quarter <= 7)
mend      = (days_until_last_trading_day_of_month <= 2)
mstart    = (days_since_first_trading_day_of_month <= 0)
holiday   = (days_until_next_trading_holiday <= 7)

stress = fomc_pre + fomc_post + opex + qopex + qend + mend + mstart + holiday
w      = (stress > 0) * (1.0 + stress)

stoch = (close - ts_min(low, W_STOCH)) / at_zero2nan(ts_max(high, W_STOCH) - ts_min(low, W_STOCH))
size  = cs_rank(ts_mean(close * volume, W_SIZE)) - 1
vrank = cs_rank(ts_std(ret1, W_VOLSTD)) - 1
co    = (close - open) / at_zero2nan(open)

denom1 = ts_mean(vrank, W_VRMEAN)
denom2 = ts_mean(ts_mean(stoch + size, W_IN1), W_IN2)
C      = (-stoch / at_zero2nan(denom1)) / at_zero2nan(denom2)
A      = ts_zscore(C, W_ZS) - size
half1  = ts_mean(A, W_OUT)

smooth = ts_mean(ts_mean(ts_mean(size, W_S1), W_S2), W_S3)
E      = ts_min(smooth, W_MIN)
skew   = ts_skew(ts_median(co, W_MED), W_SKEW)
D      = stoch + ((skew + (size + ret5)) + E)
half2  = cs_rank(D) - 1

core   = half1 - half2

alpha  = at_zero2nan(cs_winsor(core, WINSOR, remove_extreme=False) * w)
```

**IR 0.091 (slow) / 0.094 (fast) · Sharpe 1.44 · return 12.5% · TVR 0.739 ·
DD 10.5 · Calmar 1.19 · break-even 6.7bp · booksize 0.48 · 365 names ·
capacity $448m — the largest in the pool**

> **New concept — `stress` and the event calendar gate.**
> Eight boolean flags, each marking a date when **large, price-insensitive,
> mandated flow** hits the tape:
>
> - **FOMC** — macro repricing; risk desks flatten before, reposition after
> - **Monthly / quarterly expiration** — dealers delta-hedging expiring options
>   must trade the underlying regardless of view; triple witching is the largest
>   such day of the year
> - **Quarter-end / month-end** — index reconstitution, funds rebalancing to
>   target weights
> - **Month start** — pension and payroll inflows deployed mechanically
> - **Pre-holiday** — participation thins, so a given order moves price further
>
> `stress` sums them; `w = (stress > 0) × (1 + stress)` turns that into a weight.
> **None of this flow expresses a view** — it is compelled by a mandate, a hedge
> or a calendar, which is exactly the flow a reversal alpha is designed to fade.
>
> **Why it beats the earnings gate.** The earnings variable is *per-stock*, so
> conditioning on it removes names from every day. These events are *shared by
> every stock*, so they remove **days** while keeping the full universe on each
> active day. That is the whole reason capacity went from $8m to $448m.
>
> **Why gating raises IR at all** (§4.2): gating to a fraction *p* of days scales
> the mean by *p* but the standard deviation only by *√p*. If the excluded days
> earn nothing, IR improves by **1/√p** — at ~40% coverage, 1.58×. Measured on
> this core: 0.055 ungated → 0.091 gated, a gain of 1.655×, implying p = 0.365
> against ~40% documented.

**Why the calendar replaced earnings.**

| conditioner | names/day | active days | `numstk` | IR |
|---|---|---|---|---|
| earnings `≤ 12` | ~156 | 100% | **156** ✗ | 0.081 |
| earnings `≤ 16` (widened) | ~204 | 100% | 204 ✓ | 0.071 |
| 4-event union | ~1000 | ~25% | ✓ | 0.087 |
| **8-event union** | ~1000 | ~40% | **~400** ✓ | **0.094** |

**Doubling the event count from four to eight added +0.007 *and* roughly doubled
active days** — improving IR and coverage together, the signature of a conditioner
adding information rather than merely filtering.

**Point versus period events — an unplanned finding.**

| event | optimal window | type |
|---|---|---|
| `fomc_post`, `mstart` | **0 days** | point |
| `fomc_pre`, `opex`, `qopex` | 1 day | point |
| `mend` | 2 days | short period |
| `qend`, `holiday` | **7 days** | period |

Point-in-time events want 0–1 days; the two genuinely extended processes want 7.
An FOMC decision happens *at a moment*; quarter-end rebalancing is worked over a
week and holiday illiquidity builds over sessions. **The parameters recovered the
distinction without being told about it.**

Fourteen internal windows tuned individually — `W_SIZE` walked 60/80/100/150/175
(175 best, +0.018 over 60), `W_ZS` 5/10/20, `WINSOR` 0.005/0.05/0.2. Also tested
and rejected: linear decay at 3–50, exponential decay at 0.005–8, `ts_mean_exp`
smoothing of both the weight and the output.

> **Remember.** The alpha that established the calendar gate. Same thesis as
> alpha 2, expressed in the **time-series** dimension instead of the
> cross-section: 40% of the Sharpe given up, **56× the capacity**. Best Calmar in
> the pool and the largest capacity at $448m.

---

### 7.4 Alpha 4 — GP vol-regime coupling, 8-event gate

> **In one line.** A genetic-programming expression that separates a stock's
> volatility *level* from its volatility *shocks*, minus a moving-average
> reversal term, weighted by the same event calendar.
>
> **Theory it rests on.** The event gate (§7.3) · gate turnover from transitions
> (§4.4) · the human/machine division of labour (§3.7).
>
> **First appearance here.** A **GP core used unmodified** · the **turnover
> finding** that gate width changes churn.

```python
fomc_pre  = (days_until_next_fomc_meeting <= 0)
fomc_post = (days_since_last_fomc_meeting <= 0)
opex      = (days_until_next_monthly_options_expiration <= 10)
qopex     = (days_until_next_quarterly_options_expiration <= 1)
qend      = (days_until_last_trading_day_of_quarter <= 10)
mend      = (days_until_last_trading_day_of_month <= 15)
mstart    = (days_since_first_trading_day_of_month <= 0)
holiday   = (days_until_next_trading_holiday <= 1)

stress = fomc_pre + fomc_post + opex + qopex + qend + mend + mstart + holiday
w      = (stress > 0) * (1.0 + stress)

core = (ts_max(ts_mean(ts_max(ts_corr_binary(
            (ts_mean(ts_std(ret1, 20), 20) - ts_zscore(ts_std(ret1, 20), 10)),
            ret20, 60), 10), 20), 5)
        - (close / at_zero2nan(ts_mean(close, 20))))

alpha = at_zero2nan(cs_winsor(core, 0.01, remove_extreme=False) * w)
```

**IR 0.073 · Sharpe 1.16 · return 7.1% · TVR 0.251 · DD 11.8 · Calmar 0.60 ·
IR/√TVR 0.146 · break-even 11.2bp · 712 names · capacity $356m**

**Term B** — `close / ts_mean(close, 20)`, subtracted: short stocks above their
20-day average. Textbook reversal, dimensionless.

**Term A — the volatility decomposition, and the genuinely interesting part.**

```python
v = ts_std(ret1, 20)                    # 20-day realised volatility
X = ts_mean(v, 20) - ts_zscore(v, 10)   # vol LEVEL minus vol SURPRISE
```

`ts_mean(v,20)` is the slow **level** of a stock's volatility; `ts_zscore(v,10)`
is a volatility **innovation**. Subtracting gives a variable that is high when a
stock is persistently volatile but *not currently spiking*.

**Realised volatility conflates two economically distinct things** — how risky a
stock is, and whether something just happened to it. Ang, Hodrick, Xing & Zhang
showed these price differently; Term A separates them. Then
`ts_corr_binary(X, ret20, 60)` is a **per-stock coefficient** asking whether this
name's returns co-move with its own volatility regime — a vol-feedback signature
(Black 1976; Christie 1982).

**The honest part.** `ts_max(ts_mean(ts_max(c, 10), 20), 5)` — the inner
`ts_max(c,10)` has a reading (has the coupling been strong *at any point*
recently, so the flag persists rather than flickering). **The outer mean-then-max
is much harder to justify; a human would not write max-of-mean-of-max.** Most
plausibly a search artifact. *The ablation replacing all three with a single
`ts_mean(c, 20)` has not been run.*

**Why it matters beyond its IR.** **The first GP core to qualify with nothing but
a gate attached** — not one internal window touched. It converts the human/machine
claim into a repeatable procedure: *search for structure on daily bars, then
condition on event-time variables the local panel never contained.* The local
panel has no FOMC dates, no expiration calendar and no earnings dates.

> **New finding — gate width changes turnover, and not for the reason you'd
> guess.** Widening `mend` from 2 → 5 → 10 → 15 moved IR 0.064 → 0.070 **while
> TVR fell 0.348 → 0.226.** Two metrics improving together.
>
> The turnover half is mechanical. **Every time the gate switches off, the whole
> book liquidates; every time it switches on, it rebuilds.** That is a 100%
> turnover day — the largest possible. So each contiguous active block costs
> **two full book turnovers**, regardless of the signal.
>
> Widening the gate **merges separate event windows into fewer, longer blocks** —
> ~17 blocks a year at `mend=2`, ~8 at `mend=10`. Simulating the actual gate
> explains **85%** of the observed TVR fall, and the residual (the signal's own
> churn) stays flat at ~0.16 across all four settings, which is the test.

> **Remember.** Best IR/√TVR (0.146) and 11.2bp break-even at only 63×
> turnover per year. **The only alpha that can invoke the correlation override.**
> Against it: Calmar 0.60 — it loses more in a drawdown than it makes in a year.

---

### 7.5 Alpha 5 — closing-hour return skew, 8-event gate

> **In one line.** Long stocks whose final trading hour contained a few sharp
> up-ticks amid otherwise quiet trading — the fingerprint of a large unfinished
> buy order.
>
> **Theory it rests on.** The event gate (§7.3) · mandated flow concentrating at
> the close (§7.1).
>
> **First appearance here.** **Intraday (sub-daily) data** · a signal whose
> information has a **one-day half-life**.

```python
fomc_pre  = (days_until_next_fomc_meeting <= 0)
fomc_post = (days_since_last_fomc_meeting <= 0)
opex      = (days_until_next_monthly_options_expiration <= 10)
qopex     = (days_until_next_quarterly_options_expiration <= 1)
qend      = (days_until_last_trading_day_of_quarter <= 10)
mend      = (days_until_last_trading_day_of_month <= 15)
mstart    = (days_since_first_trading_day_of_month <= 0)
holiday   = (days_until_next_trading_holiday <= 1)

stress = fomc_pre + fomc_post + opex + qopex + qend + mend + mstart + holiday
w      = (stress > 0) * (1.0 + stress)

sk    = ts_mean(skew_return_last_hour, 1)
score = cs_rank(sk) - 1.0

alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False) * w)
```

**IR 0.102 · Sharpe 1.62 · return 4.9% · TVR 1.380 · DD 4.7 · Calmar 1.04 ·
break-even 1.4bp · 707 names · capacity $156m**

> **New concept — `GROUP_intra1`, the only sub-daily data in the set.**
> Thirty variables covering the first and last trading hour: OHLC, VWAP, volume,
> trade counts, return, volatility, skew, information ratio. **It is the part of
> the dataset a daily-bar competitor cannot reach at all**, and the data
> dictionary flags it as the highest alpha density per unit of crowding.
>
> Its cost: coverage runs ~3,247 days against ~4,279 for the daily groups, so
> intraday alphas start around 2007 and their drawdowns are **not comparable** to
> daily-bar alphas that span the crisis.

**What the variable is.** The **third moment of returns within the final trading
hour**. Positive skew means mostly small moves punctuated by a few sharp up-ticks.
It describes the **shape** of the hour, not its direction — a stock can close up
with negative skew.

**The theory.** A few sharp up-ticks inside an otherwise quiet hour is the
signature of a **large buy order working through the book against thin resting
liquidity** — lifting the offer, the book refilling, lifting again. And that order
is not finished: institutional orders are worked across sessions. **Positive
closing-hour skew is a fingerprint of incomplete institutional demand.**

**Why it is not momentum.** Skew is scale-free and direction-agnostic. A stock up
3% smoothly has near-zero skew; a stock flat with two spikes has high skew. That
orthogonality is what lets it sit in the same pool as reversal alphas.

**Development.**

| step | IR | TVR |
|---|---|---|
| `−` sign, `ts_mean(sk, 5)`, ungated | **negative** | — |
| `+` sign, `ts_mean(sk, 5)`, ungated | 0.061 | 0.553 |
| composite with two other intra1 signals | 0.054 | 0.861 |
| `+` sign, `ts_mean(sk, 10)`, ungated | worse | — |
| **`+` sign, `ts_mean(sk, 1)`, ungated** | **0.088** | 1.315 |
| **+ event gate** | **0.102** | 1.380 |

**Smoothing destroys the signal** — window 1 beats window 5 by +0.027. The
information has a **one-session half-life**; averaging five days averages five
unrelated execution events. **This is the opposite of alphas 2 and 3**, where
every component wants a multi-day window — and worth noticing, because it means
"smooth to reduce turnover" is not a universally safe move.

**Combining also hurt** (0.054 vs 0.061 for the best component alone): equal
weighting is only sound when components are comparably strong.

**What I got wrong.** I specified the sign backwards, reasoning from **Boyer,
Mitton & Vorkink (2010)** — investors overpay for lottery-like positive-skew
stocks, so high skew should underperform (their low-skewness quintile beats the
high by 1.00%/month). But that concerns **expected skewness of the return
distribution over months**, priced as a preference. This is **realised skewness
within one hour** — a microstructure observation about order execution. Applying a
monthly pricing result to an hourly execution statistic was a category error.

**Slow-mode check:** fast 0.097 → slow **0.102**. It *gained*, which rules out
look-ahead.

> **Remember.** Best Sharpe (1.62) and lowest drawdown (4.7); the only alpha
> built purely on sub-daily structure. **Break-even 1.4bp makes it a paper alpha
> standalone.** Its one-day half-life is the counterexample to "always smooth."

---

### 7.6 Alpha 6 — reversal core (zscore-80), calendar gate + low-volatility half

> **In one line.** Alpha 3's core, slowed from a two-week to a quarter-long
> signal, restricted to the calmer half of the universe.
>
> **Theory it rests on.** The event gate (§7.3) · cross-sectional splits and the
> √2 breadth cost (§5.2) · why low volatility wins (§5.3).
>
> **First appearance here.** The **cross-sectional split** — universe restriction
> as a conditioner · the **falsification** of the volatility mechanism.

```python
vl    = cs_rank(ts_std(ret1, 20))
lovol = (vl <= 1.5)

fomc_pre  = (days_until_next_fomc_meeting <= 2)
fomc_post = (days_since_last_fomc_meeting <= 12)
opex      = (days_until_next_monthly_options_expiration <= 10)
qopex     = (days_until_next_quarterly_options_expiration <= 1)
qend      = (days_until_last_trading_day_of_quarter <= 10)
mend      = (days_until_last_trading_day_of_month <= 13)
mstart    = (days_since_first_trading_day_of_month <= 0)
holiday   = (days_until_next_trading_holiday <= 5)

stress = fomc_pre + fomc_post + opex + qopex + qend + mend + mstart + holiday
w      = (stress > 0) * (1.0 + stress)

stoch  = (close - ts_min(low, 10)) / at_zero2nan(ts_max(high, 10) - ts_min(low, 10))
relvol = volume / at_zero2nan(ts_mean(volume, 20))
volret = cs_winsor(ts_std(ret1, 20) / at_zero2nan(ret20), 0.02, remove_extreme=False)

core1 = (-(ts_zscore(stoch, 80))
         - (((ts_median(cs_zscore(relvol), 10) - ts_mean(ret20, 20))
             + ts_skew(relvol - ts_delay(relvol, 3), 33))
            - ts_mean(-stoch - ts_median(volret, 3), 15)))

alpha = at_zero2nan(cs_winsor(core1, 0.01, remove_extreme=False) * w * lovol)
```

**IR 0.094 · Sharpe 1.49 · return 7.1% · TVR 0.666 · DD 12.0 · break-even 4.2bp ·
booksize 0.96 · 368 names**

> **New concept — the cross-sectional split.**
> `cs_rank(x)` ranks every stock by `x` each day and returns a value in **[1,2]**,
> so `1.5` is the median. `lovol = (vl <= 1.5)` keeps the calmer half.
>
> **What it costs.** `numstk` halves, ~735 → ~368 (floor is 160, so one split is
> affordable; two stacked lands near 190 with no margin).
>
> **What it does *not* cost.** Booksize still reads **0.96** at 368 names — the
> platform **renormalises the book**, so a split costs coverage but *not gross
> exposure*. You are running a full book over half the names, which makes
> splitting a better trade than it first appears.
>
> **The bar it must clear.** By Grinold, IR ≈ IC × √breadth. Halving the name
> count costs √2 ≈ 1.41 in breadth, so a split only pays if the retained half has
> an IC at least **41% higher**. That is demanding, and it is why most splits
> tested (beta, price, size, high-vol) failed.

**Two independent departures from alpha 3.** The universe halves, **and** the
`ts_zscore` window moves **10 → 80** — an eightfold increase converting a two-week
reversal into a quarter-long one. **That window change was the largest single
lever**, taking the ungated split to 0.115.

> **The falsification — the most important result in the project.**
> The prediction was written down *before* the run. Nagel says liquidity provision
> pays most where volatility is **highest**, so the high-vol half should win, and
> the design note said explicitly: *"if the low-vol half is just as good, the
> liquidity-provision reading is wrong."*
>
> | split | IR | TVR |
> |---|---|---|
> | high beta | 0.076 | 0.796 |
> | **low volatility** | **0.089 → 0.094 tuned** | 0.819 |
>
> **The low-vol half won**, and after tuning beat the whole-universe parent on
> IR/√TVR (0.115 vs 0.106). Combined with a macro-volatility regime gate that
> never added anything to any core, that is **two independent rejections of the
> volatility channel.**
>
> **The resolution (§5.3):** a volatility split is not really a volatility split.
> The alpha needs forced traders → forced flow comes from index funds → index
> weights are cap-based → large companies are low-volatility. **So "low
> volatility" and "heavily index-owned" describe roughly the same stocks**, and
> sorting on volatility sorts incidentally on *who owns the stock*.
>
> **The confirmation:** measuring index ownership directly — what fraction of a
> stock's annual volume arrives on rebalancing days — took the same core from
> **0.094 to 0.106**. That variant was measured but not submitted, so it stands as
> evidence for the mechanism rather than as an alpha in the pool.

> **Remember.** Same core as alpha 3, halved universe, eight-times-slower
> z-score window. **It falsified the mechanism I started from and pointed at a
> better one.** Against it: not an independent idea, and Calmar 0.59.

---

### 7.7 Alpha 7 — earnings cycle position, 10-term gate + low-volatility half

> **In one line.** Long stocks approaching an earnings report, short those far
> from one — the announcement premium harvested continuously.
>
> **Theory it rests on.** The earnings announcement premium (§7.2) · the
> low-volatility split (§7.6) · additive versus multiplicative conditioners
> (§4.3).
>
> **First appearance here.** A cross-sectional variable used as an **additive
> term inside `stress`** — the third way to use one, and the only one that costs
> no coverage · the earnings variable used as a **signal** rather than a gate.

```python
fomc_pre  = (days_until_next_fomc_meeting <= 2)
fomc_post = (days_since_last_fomc_meeting <= 15)
opex      = (days_until_next_monthly_options_expiration <= 10)
qopex     = (days_until_next_quarterly_options_expiration <= 1)
qend      = (days_until_last_trading_day_of_quarter <= 10)
mend      = (days_until_last_trading_day_of_month <= 13)
mstart    = (days_since_first_trading_day_of_month <= 0)
holiday   = (days_until_next_trading_holiday <= 5)
earn_pre  = (trading_days_until_next_earnings_announcement <= 15)
earn_post = (trading_days_until_next_earnings_announcement >= 12)

stress = (fomc_pre * 1.0) + (fomc_post * 1.0) + (opex * 1.0) + (qopex * 1.0) \
       + (qend * 1.0) + (mend * 1.0) + (mstart * 1.0) + (holiday * 1.0) \
       + (earn_pre * 1.0) + (earn_post * 1.0)
w      = (stress > 0) * (1.0 + stress)

vl    = cs_rank(ts_std(ret1, 60))
lovol = (vl <= 1.5)

du    = trading_days_until_next_earnings_announcement
score = -(cs_rank(du) - 1.0)

alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False) * w * lovol)
```

***[metrics to be recorded]***

> **New concept — three ways to use a cross-sectional variable.**
>
> | usage | example | coverage cost |
> |---|---|---|
> | **multiplicative gate** | alpha 2: `× (du ≤ 12)` | **severe** — 156 names, $8m capacity |
> | **universe split** | alpha 6: `× lovol` | **halves** coverage, keeps exposure |
> | **additive term in `stress`** | here: `+ (du ≤ 15)` | **none** |
>
> The calendar terms already make `stress > 0` on nearly every day, so an added
> earnings term **never zeroes anything** — it only *raises the weight* on names
> near reporting. Same variable that cost 56× capacity as a gate in alpha 2, now
> free. **The distinction is between a conditioner that multiplies to zero and one
> that adds to a floor.**

**The theory.** The **earnings announcement premium** (Savor & Wilson): firms
scheduled to report earn an annualised abnormal return of ~9.9%, and over
1958–2009 the average excess daily return on a broad US index was **11.4bp on
announcement days against 1.1bp on all others** — more than 60% of the equity risk
premium earned on 13% of the days.

**Why the risk is systematic rather than idiosyncratic.** Investors use
announcements to revise expectations for *non-announcing* firms and can only do so
imperfectly, so the covariance between firm-specific and market cash-flow news
spikes around announcements. **Announcers are especially risky in a systematic
sense** — which makes the premium a compensation rather than an anomaly.

Since some subset of names is always approaching a report, the premium can be
harvested **continuously** across the cross-section.

**Why it belongs.** `trading_days_until_next_earnings_announcement` produced the
largest single gain in the project as a *gate* (alpha 2). **This is the first time
it is used as a signal**, and working in both roles is evidence the effect is real
rather than a gating artifact.

**Third confirmation of the low-volatility finding** — alphas 6, 7 and 8 use three
completely different cores and low-vol wins every time.

**A construction error to disclose.** `earn_pre = (du ≤ 15)` and
`earn_post = (du ≥ 12)` **overlap and together span every value of `du`.** The
pair contributes a constant +1 on every stock every day, plus +2 in the narrow
12–15 band. It is not a pre/post decomposition. The intended `earn_post` was
`(du ≥ 55)` — the counter resets to ~62 the day after a report, so a high value
marks names that have *just reported*, the PEAD window (Bernard & Thomas 1989).
**As submitted, PEAD is not covered**, and the only live earnings term sits
nowhere near day 0 — the day alpha 2 identified as carrying a quarter of its gain.

> **Remember.** The announcement premium as a bare cross-sectional signal.
> **The least original alpha in the pool** — one line, well-known effect; what is
> mine is the gate and the split. Its real contribution is demonstrating the
> additive-conditioner trick that costs no coverage.

---

### 7.8 Alpha 8 — time-series-ranked closing-hour skew, 10-term gate + low-vol half

> **In one line.** Alpha 5's variable, but ranked against each stock's *own
> history* first — trading the **surprise** in closing-hour demand rather than
> its level.
>
> **Theory it rests on.** Intraday microstructure (§7.5) · the low-volatility
> split (§7.6) · decorrelation by changing the transform.
>
> **First appearance here.** **`ts_rank`** — time-series ranking, and the idea
> that a different *transform* of the same variable is a different alpha.

```python
fomc_pre  = (days_until_next_fomc_meeting <= 2)
fomc_post = (days_since_last_fomc_meeting <= 15)
opex      = (days_until_next_monthly_options_expiration <= 10)
qopex     = (days_until_next_quarterly_options_expiration <= 1)
qend      = (days_until_last_trading_day_of_quarter <= 10)
mend      = (days_until_last_trading_day_of_month <= 13)
mstart    = (days_since_first_trading_day_of_month <= 0)
holiday   = (days_until_next_trading_holiday <= 5)
earn_pre  = (trading_days_until_next_earnings_announcement <= 15)
earn_post = (trading_days_until_next_earnings_announcement >= 12)

stress = (fomc_pre * 1.0) + (fomc_post * 1.0) + (opex * 1.0) + (qopex * 1.0) \
       + (qend * 1.0) + (mend * 1.0) + (mstart * 1.0) + (holiday * 1.0) \
       + (earn_pre * 1.0) + (earn_post * 1.0)
w      = (stress > 0) * (1.0 + stress)

vl    = cs_rank(ts_std(ret1, 60))
lovol = (vl <= 1.5)

score = cs_rank(ts_rank(skew_return_last_hour, 60)) - 1.0

alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False) * w * lovol)
```

**IR 0.081 · Sharpe 1.29 · return 3.3% · TVR 1.355 · IR/√TVR 0.070 ·
break-even 1.0bp**

> **New concept — `ts_rank` versus `cs_rank`.**
>
> - `cs_rank(x)` — *which stocks have the highest x **today***, ranked against
>   each other.
> - `ts_rank(x, 60)` — *where does today's x sit in **this stock's own** past 60
>   days*.
>
> **`ts_rank` removes the cross-sectional level entirely**, keeping only the
> within-stock innovation. A stock with structurally high closing-hour skew —
> because it is permanently index-heavy — tops alpha 5's book **every single day**,
> but sits mid-pack here.
>
> The two books therefore hold substantially different names **from identical
> data**, which is what let this clear the 50% correlation gate against alpha 5.
> **A different transform of the same variable is a different alpha.**

**The economic reading.** Alpha 5 trades the *level* of institutional closing
demand; this trades the *surprise* in it. If the mechanism is unfinished order
flow, a name suddenly showing unusual closing-hour skew is a **new order
arriving**, whereas one that always shows it may simply be permanently index-heavy
with no incremental information. This isolates the arrival.

**Development.** Window 60 gave IR 0.081; window 20 was worse. That direction is
informative — a *longer* baseline works better, so "unusual for this stock" needs
a reasonably long reference period. **Note this is the opposite of alpha 5's
one-day half-life**, and consistent: the *level* decays in a day, but the
*reference distribution* you measure it against should be stable.

> **Remember.** Same data as alpha 5, orthogonal book, because `ts_rank` strips
> the persistent cross-sectional level. **Break-even 1.0bp is the worst in the
> pool** and IR/√TVR 0.070 the lowest, so no access to the correlation override.
> It earns its place on **decorrelation and count**, not economics.

---

### 7.9 Alpha 9 — adaptive IC-weighted blend

> **In one line.** Two signals blended, each weighted by how well it has recently
> predicted **that specific stock**.
>
> **Theory it rests on.** Grinold's fundamental law (IR ≈ IC × √breadth) · the
> event gate (§7.3) · slow-mode verification (§9.4).
>
> **First appearance here.** A **self-referential** conditioner — a weight that is
> a function of the alpha's own past performance rather than of market data.

```python
fomc_pre  = (days_until_next_fomc_meeting <= 2)
fomc_post = (days_since_last_fomc_meeting <= 12)
opex      = (days_until_next_monthly_options_expiration <= 10)
qopex     = (days_until_next_quarterly_options_expiration <= 1)
qend      = (days_until_last_trading_day_of_quarter <= 10)
mend      = (days_until_last_trading_day_of_month <= 15)
mstart    = (days_since_first_trading_day_of_month <= 0)
holiday   = (days_until_next_trading_holiday <= 1)

stress = fomc_pre + fomc_post + opex + qopex + qend + mend + mstart + holiday
w      = (stress > 0) * (1.0 + stress)

sk = ts_mean(skew_return_last_hour, 1)
a  = cs_rank(sk) - 1.0                                        # alpha 5's core

beta = ts_corr_binary(ret1, ret1_spx, 60)
res  = ret5 - (beta * ts_sum(ret1_spx, 5))
b    = cs_rank(-res / at_zero2nan(ts_std(ret1, 20))) - 1.0    # residual reversal

ic_a = cs_rank(ts_corr_binary(ts_delay(a, 1), ret1, 60)) - 1.0
ic_b = cs_rank(ts_corr_binary(ts_delay(b, 1), ret1, 60)) - 1.0

score = (a * ic_a) + (b * ic_b)

alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False) * w)
```

**IR 0.075 · Sharpe 1.19 · return 5.7% · TVR 0.870 · DD 8.0 · Calmar 0.71 ·
break-even 2.6bp · 738 names · capacity $221m**

> **New concept — a per-stock rolling information coefficient.**
> `ts_corr_binary(ts_delay(signal, 1), ret1, 60)` correlates **yesterday's signal
> value** against **today's realised return**, over a trailing 60 days, separately
> for each stock. High means the signal has been predicting *this specific name*.
>
> Everything else in the pool is a fixed function of market data. **This is a
> function of the alpha's own recent forecasting performance** — a self-referential
> object, which is why it decorrelates from its own components: it only carries a
> component while that component is working.

**The theory.** Signal validity varies by stock, and the variation **persists long
enough to estimate**. Closing-hour skew should be far more reliable in names with
heavy index and ETF membership — and **index membership is stable over months**,
long enough for a 60-day window to measure it and act on it. Residual reversal's
reliability tracks shareholder base and analyst coverage, also stable. **The
weighting estimates a structural property of each stock's information
environment**, not noise.

**Grinold's Fundamental Law** (IR ≈ IC × √breadth) is why an IC estimate is the
right weighting quantity in principle — and why *ranking* a noisy IC is defensible:
ranking discards the magnitude and keeps only the ordering, which is far more
robust to a noisy estimate.

**Provenance.** The technique is the second stage of **AlphaForge** (AAAI 2025),
whose own ablation shows the dynamic variant beating the static one — the
*combination* stage, not the generation stage, is where its edge lives. **What is
mine:** compressing it into a single expression via a per-stock rolling IC rather
than a trained model, and doing it **cross-sectionally per stock** rather than as
one time-varying scalar per factor. **Rolling-IC weighting is decades old in
practice — I would not claim the concept.**

> **Why slow mode matters here specifically.**
> This puts **realised returns inside the signal** — the most likely place in the
> project for look-ahead to hide, and **fast mode cannot detect it**: a peeking
> expression looks superb in fast and collapses in slow.
>
> The window is trailing and `ret1` is Delay-0, so it is clean on inspection. But
> *"legitimate on inspection"* is worth very little against a class of bug that is
> invisible by construction. **The slow-mode run is the evidence, not the
> reasoning.**

> **Remember.** The only structurally self-referential alpha in the pool.
> **The static-blend ablation has not been run**, so there is no evidence the
> adaptive weighting contributes anything over a fixed 50/50 mix. That is the
> first question an interviewer will ask, and the honest answer is currently *"I
> don't know."*

---

### 7.10 What the pool is, taken as a whole

**Three distinct mechanisms:**

1. **Reversal / liquidity provision** (2, 3, 4, 6) — paid for absorbing mandated
   flow, conditioned on when and where that flow arrives.
2. **Intraday microstructure** (1, 5, 8) — the only sub-daily structure in the
   dataset.
3. **Event-time premia** (7) — compensation for bearing scheduled event risk.

Plus one structural device (9) that is a function of the others' performance.

**Best property — dispersion on two axes that do not track IR.** Break-even runs
1.0bp to 11.2bp; capacity runs $8m to $448m. Alpha 5 has the best Sharpe and is
unusable standalone; alpha 4 has the lowest Sharpe and survives 11bp with $356m of
capacity. In a combined book both are worth holding, since marginal turnover
across a multi-alpha portfolio is far below the sum of standalone turnovers.

**Worst property — a shared spine.** Alphas 2, 3 and 6 share the same reversal
core; 1 and 2 share the earnings conditioner; 5 and 8 are built on
`skew_return_last_hour`; six of nine wear the calendar gate. The correlation
checks passed, but clearing 50% is not independence, and **the decomposition that
would settle it — measuring each core ungated — has been done for two of nine.**

**Two conditioners pointing opposite ways.** Alpha 1 gates *away* from earnings
while alpha 2 gates *toward* them. That is not inconsistent: alpha 2 is paid for
bearing risk into an announcement, alpha 1 is betting a move is uninformed, and
near an announcement that assumption is least safe. **Each in the direction its own
mechanism predicts is stronger evidence than either alone.**

---

<a name="8"></a>
## 8. Method: what generalised

Eight things that transferred across alphas.

**1. Characteristics beat forecasts.** Everything that qualified ranks stocks by
a *structural property* of how they trade — return shape, efficiency, depth,
cycle position. Every *directional forecast* failed: momentum, value, quality,
52-week high, intraday periodicity, signal-agreement products.

**2. Expressions must be adjustment-invariant, not merely dimensionless.**
Splits are applied retrospectively; `close`/`high`/`low` are DIV-adjusted while
`volume` is MUL-adjusted. A ratio mixing conventions changes meaning between
fast and slow mode. **And the error has a direction: companies split after the
price rises, so the split factor is a function of future returns.** `close *
volume` is invariant; `volume / |return|` is not. (§9.4)

**3. Time-series conditioning preserves coverage; cross-sectional conditioning
destroys it — unless used as an additive weight.** (§4.2)

**4. Window tuning has repeatedly beaten gate tuning — after the first alpha.**
In note 14 the gate contributed **+0.055** and internal windows **+0.006**
(total +0.061, 0.063 → 0.124), and I generalised from that single case. It did
not hold: family 2 gained +0.027 from a window sweep against +0.014 from its
gate, and family 5's largest single lever was a `ts_zscore` window moving
10 → 80. **The note-14 generalisation was drawn from one alpha and is wrong as
a general rule** — which knob dominates depends on how well-tuned the core
already is, and a GP-derived core arrives with its windows near-optimal while a
hand-built one does not.

**5. Sign is a parameter, not a conclusion.** Three intraday signals were
specified with the wrong sign from mechanism reasoning, and all three worked
inverted. Measure direction; do not derive it.

**6. Normalise levels into surprises.** Raw trade size ranks stocks by how big
their typical trades are — a static size proxy. Divided by its own 20-day mean
it becomes a surprise. The same failure killed the fundamental *levels* sweep
and produced family 7's repair.

**7. Slow mode is the test; fast mode is the screen.** No result is believed
until slow mode confirms it. One family looked excellent in fast mode and
collapsed in slow — correctly (§9.4).

**8. Alphas must be judged on IR/√TVR and break-even cost, not IR.** IR only has
to clear 0.07; every point above that is worth less than the turnover it costs.
Family 2 has the highest Sharpe in the book (1.62) and breaks even at 1.4bp,
making it a paper alpha. Family 7 has a lower Sharpe and survives 73bp.

---

<a name="9"></a>
## 9. Errors, and how each was caught

*The most useful section for an interview. Each is stated as what went wrong,
how it was found, and what changed.*

### 9.1 The earnings window — reasoning that survived too long

The first version of the theory argued the last ~3 days before an announcement
should be **excluded**: implied vol peaks, options hedging dominates the tape,
and a reversal signal has nothing to say about mechanical delta-hedging. The
`≥ 3` bound appeared to confirm it.

**Wrong.** Walking the lower bound down — 3, 2, 1, 0 — improved IR monotonically
at every step, and the last step was the largest jump in the entire exercise.
The days I argued were noise are the most valuable in the sample.

The apparent confirmation was **confounded**: when `≥3, ≤18` beat `≤15`, the
upper bound moved at the same time. I attributed the gain to the exclusion when
it came from the extension.

*Lesson:* the mechanism story was doing the reasoning, and the mechanism was
wrong. What saved it was testing the parameter across its whole range rather
than stopping once the theory looked confirmed.

### 9.2 Three inverted signs

Specified three intraday microstructure signals with a leading minus, reasoning
from mechanism. All three were run inverted and all three produced positive IR.

For the closing-hour skew signal specifically I argued from **Boyer, Mitton &
Vorkink (2010)** — investors overpay for lottery-like positive-skew stocks, so
high-skew names should underperform. That is a real and well-replicated result,
but it concerns **expected skewness of the return distribution over months**,
priced as a preference. The variable here is **realised skewness within one
hour** — a microstructure observation about order execution, not a
distributional property anyone forms preferences over. Applying a
monthly-horizon pricing result to an hourly execution statistic was a category
error.

### 9.3 A gate that was never doing what the notes claimed

`stress = fomc_pre + fomc_post + ...` on raw booleans. **In numpy, `+` on
boolean arrays is logical OR, not arithmetic addition.** So `stress` was a
boolean — `True` if *any* event was live — and `w = (stress > 0) * (1.0 +
stress)` evaluated to `{0.0, 2.0}`.

**The gate was a binary on/off switch in every alpha, and the "(1 + stress)
intensity weighting" described in note 16 never executed.** Found only when a
later expression passed `stress` into a `ts_` operator and the type error
surfaced.

Fixed by multiplying each condition by 1.0 before summing. The graded weight is
a genuinely untested lever.

### 9.4 A look-ahead with no forward reference

One alpha family produced strong fast-mode results and failed badly in slow
mode. Three documented causes of divergence; the first two ruled out
structurally.

The expression divided **share volume** (MUL-adjusted) by **absolute return**
(invariant). Nothing cancelled, so its fast-mode value was scaled by every split
the stock would ever undergo.

**And the error is not neutral. Companies split after the price has risen, so
the split factor is a function of future returns.** Apple split 7:1 in 2014 and
4:1 in 2020 — in fast mode its 2007–2014 share volume is multiplied by **28**,
and Apple was one of the best performers in the sample. Repeat across every
large-cap splitter and the ranking acquires a systematic tilt toward future
winners.

**This is a genuine look-ahead that arrives through the data rather than through
the expression**, which is why it survives reading the code. Fixed with dollar
volume, where DIV × MUL cancels exactly — and the repaired quantity is also
better defined economically, since Kyle's lambda uses dollar volume precisely
because share counts are not comparable across price levels.

### 9.5 The mechanism revision

Covered in §6. The prediction was written down in advance with the falsifying
outcome specified, and the falsifying outcome occurred — three times.

### 9.6 Reward hacking in the GP

Fitness = IR/√TVR diverges as TVR → 0. The search found this before I did:
sixteen of 26 "winners" contained `div(ret20, ret20)` = 1 everywhere, at
TVR ≈ 0.0018. Fixed with dispersion and turnover floors, plus effective-N to
catch books that report 400 positions while holding 99% in one name.

---

<a name="10"></a>
## 10. Interview preparation

### 10.1 The narrative arc

> "I started by building infrastructure and running a genetic program — about a
> million and a half expressions across two runs. It produced almost nothing:
> two cores that ever qualified, and only after I attached a conditioner by
> hand. The diagnosis was that my local backtest had a rank correlation of −0.70
> with platform IR, so the search was optimising against an objective that was
> anti-correlated with the target. That reframed the problem: I had a cheap
> broken fitness function and an expensive perfect one, and the question was how
> to spend a hundred queries against the expensive one — which means strong
> priors, not brute force.
>
> The prior that worked was conditioning in **event time**. Reversal is
> compensation for absorbing flow, and flow is heaviest on dates when someone is
> compelled to trade — expirations, rebalancing, month-end. Gating on those
> dates was worth more than every internal parameter in the expression combined.
>
> Then the interesting part. My thesis was Nagel's: the premium should be
> largest where volatility is highest. I tested that two ways — a macro
> volatility regime conditioner, and a cross-sectional volatility split — and
> both rejected it. The **low**-volatility half won on three separate cores. So
> the mechanism isn't volatility-driven liquidity withdrawal, it's mandated
> flow: index funds, ETF creation, MOC algorithms. Low-volatility mega-caps win
> because mechanical flow is the largest *share* of their volume. I then built a
> direct measurement of that and it beat the volatility proxy, 0.106 against
> 0.094."

### 10.2 Question bank

**"Your GP failed. Was it a waste?"** It produced two of the nine submitted
cores, so not entirely — but the honest answer is that the *diagnostic* was
worth more than the output. Learning that local fitness was anti-predictive is
what redirected the work toward economic reasoning, and everything that followed
came from that.

**"Isn't 1.5 million expressions just mining?"** Yes, and the defence is not
that it isn't. It is that selection happened on a *different* dataset from
evaluation — the search ran on my own 370-name panel, and candidates were then
tested on the platform's 1,000-name universe over a different sample. The
survivor rate is consistent with genuine selection rather than transfer of a
mined artifact.

**"How do you know your gate isn't just reducing turnover?"** I don't, fully.
The falsification is a **random gate keeping the same fraction of days** — if it
helps equally, the gain is coverage or turnover rather than events. **I have not
run it**, and it is the first thing I would run with more time.

**"Half your alphas share one gate. How much of your combined performance is one
idea applied nine times?"** The right question, and the correlation check only
establishes that finished books differ by under 50% — it does not decompose
where the correlation comes from. Ungated, the cores measure 0.055 and 0.088, so
they stand on their own and the gate contributes 0.036 and 0.014 on top. But I
have not measured that for every core, and for the two that qualify most
marginally it is the number that matters most.

**"Which of these would you actually trade?"** Family 7, on cost — 73bp
break-even against family 2's 1.4bp. Family 2 has the best Sharpe in the book
and is a paper alpha standalone. In a combined portfolio I would want both,
because marginal turnover across a multi-alpha book is far below the sum of
standalone turnovers — which is a large part of why a firm runs hundreds of
alphas rather than the best five.

**"You were wrong about the mechanism, the sign three times, and your gate
wasn't doing what you documented. Why should I trust any of this?"** Because
each was caught by a test I designed, and in the mechanism case the falsifying
outcome was written down *before* the run. None of them were caught by reading
the code — they were caught by testing parameters across their full range,
checking fast against slow, and writing down predictions in advance. That
process is the transferable part; the alphas are not.

### 10.3 What not to claim

- **Novelty of underlying phenomena.** Closing-auction reversal, the
  announcement premium, PEAD, body-to-range efficiency and rolling-IC weighting
  are all documented. What is mine is the construction, the conditioning and the
  combination — not the effects.
- **That split halves are independent ideas.** They score separately; they are
  one idea.
- **That any of this is robust out-of-sample.** Nine alphas, twenty-plus tuned
  parameters, one visible sample.

---

<a name="11"></a>
## 11. What I would do next

Ordered by what would change my confidence most, not by expected IR.

1. **The random-gate falsification.** A conditioner keeping the same fraction of
   days, chosen at random. If it helps as much as the event gate, the entire
   mandated-flow story is a story about turnover. This is the single most
   informative unrun test in the project.

2. **Decompose gate contribution per core.** Run every submitted core ungated.
   Two of nine have been measured. If the gate carries most of the performance
   everywhere, I have one alpha and eight decorations.

3. **The static-blend control for family 4.** Fixed 50/50 weights, no rolling
   IC. Without it there is no evidence the adaptive weighting contributes
   anything.

4. **A direct passive-ownership measure.** The mandated-flow proxy is built from
   volume timing because the dataset has no ownership fields. Index weight or
   passive ownership share would test the revised mechanism directly rather than
   through two proxies.

5. **Decay analysis 2007–2021.** Split the sample and check whether the effects
   are flat or concentrated early. If they are concentrated pre-2015 they are
   being arbitraged, and that matters more for a production book than any IR
   figure here.

6. **Transaction-cost modelling that is not a single break-even number.** All
   cost figures in this document assume a flat round-trip spread. Real impact
   scales with participation rate and is worst in exactly the names and dates
   these alphas concentrate in.

---

## Appendix — literature

- **Nagel (2012), "Evaporating Liquidity," RFS** — reversal as compensation for
  liquidity provision, predictable by expected volatility. The original anchor,
  and the claim the evidence in §6 rejects.
- **Savor & Wilson** — the earnings announcement premium.
- **Bernard & Thomas (1989)** — post-earnings announcement drift.
- **Stoll & Whaley** — expiration-day effects from index-derivative unwinding.
- **Ariel (1987); Lakonishok & Smidt (1988)** — turn-of-the-month effects.
- **Boyer, Mitton & Vorkink (2010), RFS** — expected idiosyncratic skewness; the
  result misapplied in §9.2.
- **Blitz, Huij, Lansdorp & Verbeek (2013), JFM** — short-term residual
  reversal; residuals from a rolling **36-month Fama-French three-factor**
  model earn risk-adjusted returns roughly **twice** those of conventional
  reversal. *My implementation approximates this with industry neutralisation
  and a 60-day market beta — far shorter and a different factor set, so it is
  inspired by the paper rather than a replication of it.*
- **Kyle (1985), Econometrica; Amihud (2002), JFM** — price impact and
  illiquidity; the correct form of family 7's measure.
- **Lou, Polk & Skouras (2019), JFE** — overnight versus intraday returns.
- **Heston, Korajczyk & Sadka (2010), JoF** — intraday return periodicity.
- **Frazzini & Pedersen (2014), JFE** — betting against beta; the anomaly
  family 5 must be distinguished from.
- **Shi et al. (2025), AAAI** — AlphaForge; source of family 4's technique.
- **Grinold (1989)** — the fundamental law; why an IC estimate is the right
  weighting quantity.
