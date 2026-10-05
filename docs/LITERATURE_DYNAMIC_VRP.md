# Literature map — dynamic CVRP & learning (for FAROS burst line)

Last updated from web search Oct 2026. Use this to cite related work and position **`docs/dynamic_burst_benchmark.md`**.

---

## How to describe *your* problem in OR language

| Your setup | Closest standard name |
|------------|------------------------|
| All customers exist; **reveal time** is uncertain; locations/demands fixed once revealed | **Dynamic / static–stochastic VRP with random reveal times** (not “unknown coordinates”) |
| Replan when stopped; **wait** with zero travel; frozen **insertion** among known feasible customers | **Recourse / online policy** on top of a fixed routing heuristic (not full reoptimization each event) |
| Label = mean completion cost under **resampled** future reveals | **Look-ahead / rollout** value approximation (ADP), same spirit as Ulmer et al. |
| **Burst** generator (clustered space + overlapping windows) | **Spatially correlated demand**; anticipatory routing when partial cluster is visible |
| **Uniform** independent reveals | Harder; aligns with **no spatial forecast** — consistent with failed MLP/DeepSet |

You are **not** doing: full online reoptimization (Psaraftis-style), accept/reject pricing DVRPSR, or static CVRP **world models** where transition is exact and only ranking remains.

---

## A. Surveys & taxonomy (read first)

1. **Zhang & Van Woensel (2023)** — *Dynamic vehicle routing with random requests: A literature review.* **Int. J. Production Economics**, 256, 108751.  
   [DOI 10.1016/j.ijpe.2022.108751](https://doi.org/10.1016/j.ijpe.2022.108751)  
   Taxonomy by request type and planning horizon; emphasizes **decision strategies** under dynamic requests.

2. **Psaraftis, Wen & Kontovas (2016)** — *Dynamic vehicle routing problems: Three decades and counting.* **Networks**, 67(1), 3–31.  
   [DOI 10.1002/net.21628](https://doi.org/10.1002/net.21628)  
   Classic 11-criteria taxonomy; defines **dynamic vs static** information; catalogues reoptimization and stochasticity.

3. **Pillac et al. (2013)** — *A review of dynamic vehicle routing problems.* **EURO J. Transport & Logistics** (CIRRELT 2011-62 preprint widely cited).  
   Plan–execute vs dynamic; online customer arrival as main dynamism source.

4. **Ríos, López-Sánchez & Juan (2021)** — *Recent dynamic vehicle routing problems: A survey.* **Computers & Industrial Engineering**.  
   Application-heavy post-2015 literature.

5. **Jaillet & Wagner (2008)** — *Online vehicle routing problems* (book chapter).  
   [MIT PDF](https://www.mit.edu/~jaillet/general/online-routing-18.pdf) — **Rolling reoptimization** of offline VRP on known customers every Δt.

---

## B. Random **reveal times** & recourse (closest OR relatives)

6. **Saint-Guillain, Solnon & Deville (2017)** — SS-VRPTW with **random customers and random reveal times** (SS-VRPTW-CR). EvoSTOC / LIRIS.  
   [PDF](https://perso.liris.cnrs.fr/christine.solnon/publications/evostoc17.pdf)  
   **Waiting locations** in first-stage routes; **recourse** when requests appear; expected rejected requests — structurally similar to your **wait + go** fork.

7. **Saint-Guillain et al. (2018+)** — Extended algorithms & recourse with capacity.  
   [arXiv:1708.03151](https://arxiv.org/abs/1708.03151)

8. **Saint-Guillain et al. (2019)** — *Progressive Focus Search* for SS-VRPTW-CR.  
   [arXiv:1902.03930](https://arxiv.org/abs/1902.03930)

9. **Bent & Van Hentenryck (2004)** — *Scenario-based planning for partially dynamic VRP with stochastic customers.* **Operations Research**.  
   [DOI 10.1287/opre.1040.0124](https://doi.org/10.1287/opre.1040.0124)  
   Multiple-scenario plans + **consensus** — offline scenarios vs your online **M futures** resampling.

---

## C. **Waiting** explicitly (direct precedent for your wait action)

10. **Mitrović-Minić & Laporte (2004)** — *Waiting strategies for the dynamic pickup and delivery problem with time windows.* **Transportation Research Part B**, 38(7), 635–655.  
    Drive-first vs wait-first vs dynamic waiting — same design knob you learned.

11. **Branke et al. (2005)** — *Waiting strategies for dynamic vehicle routing.* **Transportation Science**, 39(3), 298–312.  
    [DOI 10.1287/trsc.1040.0095](https://doi.org/10.1287/trsc.1040.0095)

12. **Eilerman et al. (2007)** — *Waiting strategies for anticipating service requests from known customer locations.* **Transportation Science**.  
    MDP on **known locations**, probabilistic future requests — “wait where future demand is likely.”

---

## D. Stochastic **requests** & value approximation (ADP, not deep WM)

13. **Ulmer, Goodson, Mattfeld & Hennig (2019)** — Offline–online ADP for DVRP with stochastic requests. **Transportation Science**, 53(1), 185–202.

14. **Klapp, Erera & Goodson (2018)**; **van Heeswijk et al. (2019)** — Lookahead / rollout for DVRPSR.

15. **Goodson et al. (2023)** — *Solving large-scale dynamic vehicle routing problems with stochastic requests.* **European J. Operational Research**, 306(2), 596–614.  
    [arXiv:2202.12983](https://arxiv.org/abs/2202.12983) — **Knapsack-based linear value approximation** for reward-to-go (scalable, interpretable baseline vs neural).

16. **van Heeswijk et al. (2024)** — *A comparison of reinforcement learning policies for dynamic vehicle routing problems with stochastic customer requests.* **Computers & Industrial Engineering**.  
    Compares **LVFA, NNVFA, LPFA, NNPFA**; post-decision state value — closest “learned critic for DVRP” survey to your scorer.

---

## E. Anticipatory / forecast-driven routing

17. **van Lochem et al. (2020)** — *Anticipatory vehicle routing for same-day pickup and delivery using historical data clustering.* **IEEE ITSC**.  
    Predict request **types** from history; insert anticipated requests — analogue of **burst** when cluster partially visible.

18. **Look-ahead partial routing** — dynamic customers with forecast horizon parameter.  
    [USC PDF](https://bpb-us-w1.wpmucdn.com/sites.usc.edu/dist/0/249/files/2021/02/DVRP_Paper_Final-1.pdf)

19. **Forecast-driven B2B food delivery (TU Delft)** — neural demand forecast + two-stage route optimization.  
    [Repository](https://resolver.tudelft.nl/636628d5-2fb1-445b-b6a0-f24065a1ea0a)

20. **SOUP (Li et al., 2020)** — spatial–temporal forecasting + idle vehicle repositioning.  
    [arXiv:2009.12157](https://arxiv.org/abs/2009.12157)

---

## F. Deep RL & ML + combinatorial layer (dynamic or online)

21. **Nazari et al. (2018)** — RL for VRP; reoptimization framing.

22. **Kool, van Hoof & Welling (2019)** — *Attention, learn to solve routing problems!* **ICLR**.  
    [arXiv:1803.08475](https://arxiv.org/abs/1803.08475) — **Static** full information; MnLP/LEHD line builds on this — explains why **static WM** underperforms when dynamics are arrivals.

23. **Soeffker et al. — EURO-NeurIPS 2022 dynamic VRP competition** — **NN assigns prizes** → prize-collecting VRPTW solved by HGS each epoch.  
    [Summary PDF](https://roadef2023.sciencesconf.org/435757/ROADEF2023_updated.pdf)  
    Same pattern as your **frozen backbone + learned scoring** of alternatives.

24. **Soeffker, Ulmer & Mattfeld (2022)** — ML pipeline with **embedded CO layer** for dynamic VRP.  
    [arXiv:2304.00789](https://arxiv.org/abs/2304.00789)

25. **MDPI Logistics (2024)** — *Optimizing a dynamic VRP with deep RL: analyzing state-space components.*  
    [MDPI](https://www.mdpi.com/2305-6290/8/4/96) — accept/reject + routing; stresses **state design** (like your 11-D burst features).

26. **Bertsimas, Li & Peng (2020)** — *RL with combinatorial actions for vehicle routing.* **NeurIPS**.  
    [Paper](https://proceedings.neurips.cc/paper/2020/hash/06a9d51e04213572ef0720dd27a84792-Abstract.html) — Value + **MIP action selection** (route as combinatorial action).

27. **Mazyavkina et al. (2021)** — GNN + DQN + **MCTS** for online CVRP.  
    [arXiv:2104.01646](https://arxiv.org/abs/2104.01646)

28. **MC-VRP (2022)** — Online MCTS for DVRP with stochastic trip requests (paratransit).  
    [arXiv:2203.15127](https://arxiv.org/abs/2203.15127)

29. **Structured RL for combinatorial decision-making (NeurIPS 2025)** — Combinatorial MDP + CO layers.  
    [Paper](https://proceedings.neurips.cc/paper_files/paper/2025/file/fe100e5e7bca984803f0e4c6e94a6a88-Paper-Conference.pdf)

30. **Integrated dispatching + DRL routing oracle (last-mile pickup, 2025)** —  
    [arXiv HTML](https://arxiv.org/html/2607.22356) — **Learned routing oracle + heuristic dispatch** (parallel to your split).

---

## G. What we did **not** find (as of this search)

- **“World model” / JEPA / latent dynamics** for CVRP with **exact** transitions — essentially absent; OR uses **value function / scenarios / recourse**, ML uses **policy or Q on hand-crafted state**.
- **Dynamic CVRP** papers that match **all** of: fixed N, reveal-time-only uncertainty, **no** accept/reject, **capacity 80 / 200 nodes**, and **learned wait among separated candidates** — no exact match; **SS-VRPTW-CR + waiting strategies** is the citation bundle.
- Papers beating **uniform reveal** with **only** the visible known set without spatial correlation — consistent with your **negative result**.

---

## Suggested related-work paragraph (copy-edit for paper/mail)

> We study a dynamic CVRP in which customer locations and demands are fixed but **reveal times** are uncertain until service begins, related to the static–stochastic VRPTW with random reveal times (Saint-Guillain et al.) and waiting strategies in dynamic PDPTW (Mitrović-Minić & Laporte; Branke et al.). Unlike full online reoptimization (Psaraftis; Jaillet & Wagner) or accept/reject DVRP with stochastic requests (Ulmer et al.), we keep a **frozen nearest-feasible backbone** and learn only a **local wait/route fork** at spatially separated decisions. This follows the learning-augmented pattern seen in recent dynamic VRP competitions (EURO-NeurIPS 2022), but our labels are **rollout completion costs** under resampled cluster reveals, and we show **uniform** reveals are not learnable from the visible set while **burst** arrivals are.

---

## Search notes (diminishing returns)

Further queries (`world model vehicle routing`, `JEPA routing`, `LaRP world model`) largely return **static** neural routing (Kool et al.) or **generic ADP** surveys already listed. Stop here unless you add **pickup–delivery**, **time windows**, or **accept/reject** to the benchmark.
