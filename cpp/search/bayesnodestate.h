#ifndef SEARCH_BAYESNODESTATE_H_
#define SEARCH_BAYESNODESTATE_H_

//Per-node side state for the Bayesian posterior search (useBayesSearch).
//M2: a pure passenger audited by gates — selection still reads normal stats.
//All values are WHITE-PERSPECTIVE WINRATE units in [0,1] (the M1 heads were
//fitted in that space); variances are winrate^2.
//
//Contract (ported from bmcts algorithms.py seq-reveal path, round-3 stack:
//Stein-corrected d-mean-aware shrinkage, P-var projection):
// - A node's sibling-set state lives on the PARENT (vsKids, betaKids).
// - A child's (mu, vPriv, b, vsOwn) and its frozen anchor are written by the
//   PARENT's bayes recompute at the first recompute where the child has its
//   own NN eval (KataGo's backup unwinds leaf->root, so the parent recompute
//   runs right after the child's eval lands — the bmcts reveal ordering).
// - Once anchorFrozen, the parent only refreshes vsOwn (the bmcts
//   "expanded children keep their children-derived state" rule); the node's
//   own recompute overwrites (mu, b, vPriv) from its children once its own
//   set holds at least one eval.
//
//Threading: M2 is single-threaded by contract (useBayesSearch requires
//numSearchThreads == 1, enforced at beginSearch). No locks here.
struct BayesNodeState {
  bool anchorFrozen = false;
  //Posterior of this node's value: error = b * X_ownset + private
  double mu = 0.0;
  double vPriv = 0.0;
  double b = 0.0;
  //Shared variance V_X of the sibling set this node belongs to
  //(refreshed by the parent on every parent recompute)
  double vsOwn = 0.0;
  //Anchor for this node's own children-set, frozen at reveal
  double anchMu = 0.0;
  double anchVar = 0.0;
  double bExp = 0.0;
  //Children-set state (valid once this node has computed its set at least once)
  double vsKids = 0.0;
  double betaKids = 0.0;
  //Resolvable-variance backup (sum_i w_i R_i); leaf/terminal children 0
  double resolvable = 0.0;
  //Diagnostics: own first eval (winrate) and its corrected sd
  double mu0 = 0.0;
  double sigma0 = 0.0;
};

#endif
