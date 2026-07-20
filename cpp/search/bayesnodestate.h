#ifndef SEARCH_BAYESNODESTATE_H_
#define SEARCH_BAYESNODESTATE_H_

#include <algorithm>
#include <cmath>
#include <vector>

#include "../game/board.h"
#include "../search/bayesposterior.h"

//Per-node side state for the Bayesian posterior search (useBayesSearch).
//M2: a pure passenger audited by gates — selection still reads normal stats.
//M3 adds the voi-KG routing currency (dKids/dBackup) consumed by the
//selection fork (useBayesSelection).
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

//M8 2d-i resolution kernel (docs/bayes-m8-integration.md 2d-i
//registration + amendment 1): eval-error correlation is a product of
//per-edge retentions driven by the stErr-head ratio,
//  corr(e_u, e_w) = A * prod phi_e,  phi_e = THETA0 * min(1, s_c/s_p)^GAMMA
//Constants pinned on the search-tree pair table (label data only;
//m8_kernel_fit.py --pin-tree). Consumed by the per-node accumulators
//(accN/accS/accQ below) and the contrast noise in the chooser and the
//contrast-voi selection.
namespace BayesKernel {
  constexpr double A = 0.923;
  constexpr double THETA0 = 0.85;
  constexpr double GAMMA = 1.5;
  //Retention on the edge parent -> child, from corrected head sigmas.
  inline double phi(double sChild, double sParent) {
    double r = sChild / std::max(sParent, 1e-12);
    if(r > 1.0)
      r = 1.0;
    return THETA0 * std::pow(r, GAMMA);
  }
}

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
  //One-reveal drop of this node's children-set at the current |E| (the KG
  //routing currency D for its unevaluated children; bmcts _reveal_drop)
  double dKids = 0.0;
  //KG backup: max_i w_i * D_i over this node's children-set
  //(bmcts _recompute_seq node.D — MAX, not sum)
  double dBackup = 0.0;
  //Diagnostics: own first eval (winrate) and its corrected sd
  double mu0 = 0.0;
  double sigma0 = 0.0;
  //M8 2d-i resolution-kernel accumulators over this node's subtree
  //(docs/bayes-m8-integration.md 2d-i engine forms). Sigma-weighted:
  //  accN = NN evals in subtree (terminal-exact values count in N only)
  //  accS = fade-weighted sigma mass, s_v + sum_c phi_c * accS_c
  //  accQ = pair-covariance mass; Var(subtree-mean error) = accQ / accN^2
  //Initialized at freeze to the leaf values (1, sigma0, sigma0^2) and
  //rebuilt from children on every recompute of this node.
  int64_t accN = 1;
  double accS = 0.0;
  double accQ = 0.0;
};

struct SearchNode;

//Snapshot of a node's children-set posterior computation (the shared middle
//of bayesRecomputeNodeStats: policy-set building through
//nodePosteriorFromChildren), computed read-only by
//Search::bayesComputeSetState so that both the backup and the M3 selection
//fork consume the identical set state. One entry per legal move of the node,
//in descending-prior order (stable under prior ties).
struct BayesSetState {
  //Per-entry vectors, all length k
  std::vector<int> pos;                    //policy index
  std::vector<Loc> moveLoc;                //board location of the move
  std::vector<double> prior;               //policy prior (floored at 1e-12)
  std::vector<const SearchNode*> child;    //allocated child node, if any
  std::vector<bool> evaled;                //has an eval (own NN output, or terminal value)
  std::vector<bool> terminalEvaled;        //evaled via exact terminal value (no NN output)
  std::vector<double> evalWinrate;         //valid iff evaled
  std::vector<double> evalStErr;           //0.5*shorttermWinlossError, 0 if unavailable
  std::vector<double> mu;                  //assembled per-entry posterior mean (clipped)
  std::vector<double> vPriv;               //assembled per-entry private variance
  std::vector<double> b;                   //assembled per-entry shared-loading
  std::vector<double> R;                   //per-entry resolvable variance
  std::vector<double> D;                   //per-entry one-reveal KG currency
  //Scalars
  double vX = 0.0;                         //set shared variance (Stein V_X)
  double kappaAlpha = 0.0;                 //Stein anchor gain
  int n = 0;                               //number of evaled entries
  int k = 0;                               //number of entries (legal moves)
  double mKids = 0.0;                      //E[max/min] over the set
  double vKidsPriv = 0.0;                  //private variance of the max
  double bOut = 0.0;                       //max's loading on the set shared variable
  std::vector<double> w;                   //P(entry attains the max/min), length k
  double dKids = 0.0;                      //one-reveal drop at current |E| (bmcts _reveal_drop)
  double dBackup = 0.0;                    //max_j w_j * D_j (bmcts _recompute_seq node.D)
  //Raw Stein posterior over the set (the backup's freeze step consumes it)
  BayesPosterior::SteinShrink stein;
};

#endif
