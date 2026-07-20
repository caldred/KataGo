//Bayesian posterior side-state maintenance (useBayesSearch), M2, plus the
//voi-KG selection fork (useBayesSelection), M3.
//
//This is the engine port of the bmcts round-3 sequential-reveal stack
//(docs/bayes-m2-design-notes.md): Stein-corrected d-mean-aware shrinkage
//(P-var projection) for fresh/partially-revealed sibling sets, and the
//Clark/quadrature max backup (node_posterior_from_children) for the node
//posterior. At M2 this state was a pure PASSENGER: selection ran PUCT on the
//ordinary stats; gates audited the posterior. M3 (docs/bayes-m3-gate.md)
//adds the selection fork: score_j = contested_j x D_j with the one-reveal
//KG currency D (bmcts policy="voi", seq_kg), behind useBayesSelection
//(default false; requires useBayesSearch).
//
//Units: white-perspective WINRATE in [0,1] (the space the M1 heads were
//fitted in). sigma head: log sigma = bayesSigmaA + bayesSigmaB * log(st),
//st = 0.5*shorttermWinlossError. sigma_d (residual, d-mean-aware):
//log sigma_r^2 = bayesSigmaDA + bayesSigmaDB * log(st_node^2), lognormal
//mean correction folded into bayesSigmaDA by the config. d-mean head:
//white-persp per-child mean = moverSign * bayesDMean * centered log prior
//(priors floored at 5e-3 for the feature — the fitted support ended at
//prior 0.01; consumption guard, see gate doc Amendment B outcome).
//
//Reveal ordering (matches bmcts _reveal): backup unwinds leaf -> root, so
//the parent's recompute runs right after a child's first eval lands; the
//parent freezes the child's anchor from the Stein posterior AT that
//recompute (the anchor thus includes the child's own eval and all sibling
//information, and the child's own eval is never re-fused afterwards).
//At the same freeze, M3 initializes the child's own KG state (dKids,
//dBackup) from the child's fresh n=0 set — the engine's analogue of bmcts
//_reveal -> _expand_seq(child) -> _refresh_set/_recompute_seq, which give a
//just-revealed internal child D = drop * max_i w_i immediately (without
//this the child would carry D = 0 until its own next recompute and be
//spuriously non-viable for selection).
//
//Known M2 caveats, documented: terminal children enter the eval set with
//the shared homogeneous v_u (their values are exact; conservative);
//tree reuse keeps anchors frozen in a previous search's state; noResult
//mass is ignored (tromp-taylor).
//
//M2 gate outcome (docs/bayes-m2-gate.md): Gate A PASS; Gate B
//FAIL-attributed — the location drift vs a deep reference is dominated by
//the net's own shallow-vs-deep opening bias (stock PUCT drifts MORE at
//matched evals) plus the tracked adaptive-selection residual
//(+0.14..+0.22 z, round-3 testbed range). Spread claims calibrated.

#include "../search/search.h"
#include "../search/searchnode.h"
#include "../search/bayesnodestate.h"
#include "../search/bayesposterior.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <numeric>
#include <sstream>
#include <vector>

static double winrateOfNN(const NNOutput* nnOutput) {
  return 0.5 + 0.5 * ((double)nnOutput->whiteWinProb - (double)nnOutput->whiteLossProb);
}
static double clipWinrate(double x) {
  return std::min(0.99, std::max(0.01, x));
}

double Search::bayesSigmaFromStErr(double stErrWinrate) const {
  double st = stErrWinrate > 1e-8 ? stErrWinrate : searchParams.bayesDefaultSigma;
  st = std::max(st, 1e-4);
  return std::exp(searchParams.bayesSigmaA + searchParams.bayesSigmaB * std::log(st));
}

//Per-entry KG currency D (bmcts _refresh_set cn.D / _decision_scores):
//terminal-evaled entries are resolved exactly (D = 0); evaluated non-terminal
//children with frozen state carry their backed-up dBackup; everything else
//(unevaluated, or an evaled-but-not-yet-frozen child from a previous search's
//final playout, treated as unrevealed) gets the set's one-reveal drop dKids.
static double bayesEntryD(const SearchNode* child, bool evaled, bool terminalEvaled, double dKids) {
  if(terminalEvaled)
    return 0.0;
  if(evaled && child != NULL && child->bayesState != NULL && child->bayesState->anchorFrozen)
    return child->bayesState->dBackup;
  return dKids;
}

//One entry per legal move of the node, in descending-prior order.
//(Local scratch used only while building; results land in BayesSetState.)
struct BayesSetEntry {
  int pos;                    //policy index
  double prior;
  const SearchNode* child;    //allocated child node, if any
  bool evaled;                //has an eval (own NN output, or terminal value)
  bool terminalEvaled;        //eval is an exact terminal value (no NN output)
  double evalWinrate;         //valid iff evaled
  double evalStErr;           //0.5*shorttermWinlossError, 0 if unavailable
};

//Compute the children-set posterior state of `node` read-only: policy-set
//building through nodePosteriorFromChildren, exactly the middle of the M2
//bayesRecomputeNodeStats. Shared by the backup (which then applies the
//freeze/refresh mutations) and the M3 selection fork (which consumes it
//as-is). Requires node's own anchor to be frozen. The assembled per-entry
//(mu, vPriv, b, R) read frozen children from their bayesState and everything
//else from the Stein posterior; for a child that the backup is ABOUT to
//freeze the two coincide bit-for-bit (the freeze writes exactly the Stein
//values, clipped mean included), so computing them pre-freeze is exact.
bool Search::bayesComputeSetState(const SearchNode& node, BayesSetState& out) const {
  if(!searchParams.useBayesSearch)
    return false;
  const NNOutput* nnOutput = node.getNNOutput();
  if(nnOutput == NULL)
    return false;
  if(node.bayesState == NULL || !node.bayesState->anchorFrozen)
    return false;
  const BayesNodeState& bs = *node.bayesState;

  const double rho = searchParams.bayesRho;
  const bool isMaxNode = (node.nextPla == P_WHITE);

  //---- Build the sibling set: every legal move, descending prior ----
  const float* policyProbs = nnOutput->getPolicyProbsMaybeNoised();
  std::vector<BayesSetEntry> set_;
  set_.reserve(64);
  for(int pos = 0; pos < policySize; pos++) {
    if(policyProbs[pos] < 0)
      continue;
    BayesSetEntry e;
    e.pos = pos;
    e.prior = std::max((double)policyProbs[pos], 1e-12);
    e.child = NULL;
    e.evaled = false;
    e.terminalEvaled = false;
    e.evalWinrate = 0.0;
    e.evalStErr = 0.0;
    set_.push_back(e);
  }
  const int k = (int)set_.size();
  if(k <= 0)
    return false;
  std::stable_sort(set_.begin(), set_.end(),
                   [](const BayesSetEntry& a, const BayesSetEntry& b) { return a.prior > b.prior; });
  std::vector<int> posToSetIdx(policySize, -1);
  for(int j = 0; j < k; j++)
    posToSetIdx[set_[j].pos] = j;

  //---- Attach allocated children and their evals ----
  ConstSearchNodeChildrenReference children = node.getChildren();
  int childrenCapacity = children.getCapacity();
  for(int i = 0; i < childrenCapacity; i++) {
    const SearchChildPointer& childPointer = children[i];
    const SearchNode* child = childPointer.getIfAllocated();
    if(child == NULL)
      break;
    int pos = getPos(childPointer.getMoveLocRelaxed());
    if(pos < 0 || pos >= policySize || posToSetIdx[pos] < 0)
      continue;
    BayesSetEntry& e = set_[posToSetIdx[pos]];
    e.child = child;
    const NNOutput* childNN = child->getNNOutput();
    if(childNN != NULL) {
      e.evaled = true;
      e.evalWinrate = winrateOfNN(childNN);
      e.evalStErr = 0.5 * (double)childNN->shorttermWinlossError;
    }
    else {
      //No NN output but real visits: terminal node with exact value.
      NodeStats childStats(child->stats);
      if(childStats.visits > 0 && childStats.weightSum > 0.0) {
        e.evaled = true;
        e.terminalEvaled = true;
        e.evalWinrate = 0.5 + 0.5 * childStats.winLossValueAvg;
        e.evalStErr = 0.0;  //exact; enters with shared v_u (documented conservative)
      }
    }
  }

  //---- Heads: d-means, sigma_r, per-eval sigma ----
  double moverSign = isMaxNode ? 1.0 : -1.0;
  std::vector<double> dMeans(k);
  {
    double meanLogP = 0.0;
    std::vector<double> logP(k);
    for(int j = 0; j < k; j++) {
      logP[j] = std::log(std::max(set_[j].prior, 5e-3));
      meanLogP += logP[j];
    }
    meanLogP /= (double)k;
    for(int j = 0; j < k; j++)
      dMeans[j] = moverSign * searchParams.bayesDMean * (logP[j] - meanLogP);
  }
  double stNode = 0.5 * (double)nnOutput->shorttermWinlossError;
  if(stNode <= 1e-8)
    stNode = searchParams.bayesDefaultSigma;
  double sigmaR2 = std::exp(searchParams.bayesSigmaDA
                            + searchParams.bayesSigmaDB * std::log(std::max(stNode * stNode, 1e-8)));
  double sigmaR = std::sqrt(sigmaR2);

  std::vector<double> evals;
  std::vector<int> evalIdx;
  double sigma2Sum = 0.0;
  for(int j = 0; j < k; j++) {
    if(!set_[j].evaled)
      continue;
    evals.push_back(set_[j].evalWinrate);
    evalIdx.push_back(j);
    double s = bayesSigmaFromStErr(set_[j].evalStErr);
    sigma2Sum += s * s;
  }
  const int n = (int)evals.size();
  double vBar = n > 0 ? sigma2Sum / (double)n
                      : searchParams.bayesDefaultSigma * searchParams.bayesDefaultSigma;
  double vU = (1.0 - rho) * vBar;
  double varS = rho * vBar;

  //---- Stein-corrected partial-observation shrinkage over the set ----
  double eD, vD, g;
  BayesPosterior::extremeMomentsGaussian(dMeans, sigmaR, isMaxNode, eD, vD, g);
  out.stein = BayesPosterior::shrinkSiblingsStein(
    evals, evalIdx, k, vU, varS, bs.anchMu, bs.anchVar, sigmaR, eD, vD, g, &dMeans);

  //---- One-reveal drop of this set at the current |E| (bmcts _reveal_drop):
  //two dummy zero-eval Stein calls at n and n+1, indices 0..n-1 / 0..n
  //regardless of which children are actually evaluated — the variance side
  //of the posterior is data-independent. ----
  out.dKids = 0.0;
  if(n < k) {
    std::vector<double> zerosN(n, 0.0);
    std::vector<int> idxN(n);
    std::iota(idxN.begin(), idxN.end(), 0);
    BayesPosterior::SteinShrink stN = BayesPosterior::shrinkSiblingsStein(
      zerosN, idxN, k, vU, varS, bs.anchMu, bs.anchVar, sigmaR, eD, vD, g, &dMeans);
    std::vector<double> zerosN1(n + 1, 0.0);
    std::vector<int> idxN1(n + 1);
    std::iota(idxN1.begin(), idxN1.end(), 0);
    BayesPosterior::SteinShrink stN1 = BayesPosterior::shrinkSiblingsStein(
      zerosN1, idxN1, k, vU, varS, bs.anchMu, bs.anchVar, sigmaR, eD, vD, g, &dMeans);
    out.dKids = std::max(stN.vU - stN1.vE, 0.0);
  }

  //---- Assemble per-entry posteriors (frozen children from their state,
  //everything else from the Stein posterior) ----
  out.pos.resize(k);
  out.moveLoc.resize(k);
  out.prior.resize(k);
  out.child.resize(k);
  out.evaled.resize(k);
  out.terminalEvaled.resize(k);
  out.evalWinrate.resize(k);
  out.evalStErr.resize(k);
  out.mu.resize(k);
  out.vPriv.resize(k);
  out.b.resize(k);
  out.R.resize(k);
  out.D.resize(k);
  for(int j = 0; j < k; j++) {
    const BayesSetEntry& e = set_[j];
    out.pos[j] = e.pos;
    out.moveLoc[j] = NNPos::posToLoc(e.pos, rootBoard.x_size, rootBoard.y_size, nnXLen, nnYLen);
    out.prior[j] = e.prior;
    out.child[j] = e.child;
    out.evaled[j] = e.evaled;
    out.terminalEvaled[j] = e.terminalEvaled;
    out.evalWinrate[j] = e.evalWinrate;
    out.evalStErr[j] = e.evalStErr;
    bool frozen = e.child != NULL && e.child->bayesState != NULL
                  && e.child->bayesState->anchorFrozen;
    if(frozen) {
      const BayesNodeState& cbs = *e.child->bayesState;
      out.mu[j] = cbs.mu;
      out.vPriv[j] = cbs.vPriv;
      out.b[j] = cbs.b;
      out.R[j] = cbs.resolvable;
    }
    else {
      out.mu[j] = clipWinrate(out.stein.mus[j]);
      out.vPriv[j] = out.stein.vPriv[j];
      out.b[j] = out.stein.b[j];
      out.R[j] = e.terminalEvaled ? 0.0
                 : (out.stein.b[j] * out.stein.b[j] * out.stein.vX + out.stein.vPriv[j]);
    }
    out.D[j] = bayesEntryD(e.child, e.evaled, e.terminalEvaled, out.dKids);
  }

  //---- Back up through the max ----
  BayesPosterior::nodePosteriorFromChildren(out.mu, out.vPriv, out.b, out.stein.vX, isMaxNode,
                                            out.mKids, out.vKidsPriv, out.bOut, out.w);

  out.vX = out.stein.vX;
  out.kappaAlpha = out.stein.kappaAlpha;
  out.n = n;
  out.k = k;
  out.dBackup = 0.0;
  for(int j = 0; j < k; j++)
    out.dBackup = std::max(out.dBackup, out.w[j] * out.D[j]);
  return true;
}

void Search::bayesRecomputeNodeStats(SearchNode& node, bool isRoot) {
  if(!searchParams.useBayesSearch)
    return;
  const NNOutput* nnOutput = node.getNNOutput();
  if(nnOutput == NULL)
    return;

  const double rho = searchParams.bayesRho;

  //Root init: the root has no parent set; its own eval is the anchor.
  //(bmcts root init: b = 1, private/shared split of the own-eval variance.)
  if(node.bayesState == NULL) {
    if(!isRoot)
      return;  //parent has not frozen this node's anchor yet; next playout will
    double st = 0.5 * (double)nnOutput->shorttermWinlossError;
    double sigma0 = bayesSigmaFromStErr(st);
    BayesNodeState* bs = new BayesNodeState();
    bs->mu0 = winrateOfNN(nnOutput);
    bs->sigma0 = sigma0;
    bs->mu = clipWinrate(bs->mu0);
    bs->b = 1.0;
    bs->vsOwn = rho * sigma0 * sigma0;
    bs->vPriv = (1.0 - rho) * sigma0 * sigma0;
    bs->anchMu = bs->mu;
    bs->anchVar = sigma0 * sigma0;
    bs->bExp = 1.0;
    bs->anchorFrozen = true;
    bs->resolvable = sigma0 * sigma0;
    bs->accN = 1;
    bs->accS = sigma0;
    bs->accQ = sigma0 * sigma0;
    node.bayesState = bs;
  }
  BayesNodeState& bs = *node.bayesState;
  if(!bs.anchorFrozen)
    return;

  //---- Set-state computation shared with the selection fork ----
  BayesSetState ss;
  if(!bayesComputeSetState(node, ss))
    return;

  //---- Freeze newly-evaled children; refresh vsOwn on frozen ones ----
  for(int j = 0; j < ss.k; j++) {
    SearchNode* childMutable = const_cast<SearchNode*>(ss.child[j]);
    if(childMutable == NULL)
      continue;
    if(ss.terminalEvaled[j])
      continue;  //terminals carry no set of their own; handled via Stein state below
    if(ss.evaled[j] && childMutable->bayesState == NULL)
      childMutable->bayesState = new BayesNodeState();
    if(childMutable->bayesState == NULL)
      continue;
    BayesNodeState& cbs = *childMutable->bayesState;
    if(ss.evaled[j] && !cbs.anchorFrozen) {
      cbs.mu = clipWinrate(ss.stein.mus[j]);
      cbs.b = ss.stein.b[j];
      cbs.vPriv = ss.stein.vPriv[j];
      cbs.vsOwn = ss.stein.vX;
      cbs.anchMu = cbs.mu;
      cbs.anchVar = cbs.b * cbs.b * ss.stein.vX + cbs.vPriv;
      cbs.bExp = cbs.b;
      cbs.anchorFrozen = true;
      cbs.mu0 = ss.evalWinrate[j];
      cbs.sigma0 = bayesSigmaFromStErr(ss.evalStErr[j]);
      cbs.resolvable = cbs.anchVar;  //unexpanded non-leaf: fully resolvable
      //2d-i kernel accumulators: leaf init (one eval, its own sigma).
      cbs.accN = 1;
      cbs.accS = cbs.sigma0;
      cbs.accQ = cbs.sigma0 * cbs.sigma0;
      //2d-k instrument: a new leaf never recomputes on its creation
      //playout (bayesState was NULL when the backup reached it), so
      //AUDIT_ALL otherwise never sees the tip edge of the playout.
      //Emit a minimal leafReveal record so per-playout tip-edge head
      //ratios are measurable (docs/bayes-m8-integration.md 2d-k).
      {
        static const char* auditPathLR = std::getenv("KATAGO_BAYES_AUDIT");
        static const char* auditAllLR = std::getenv("KATAGO_BAYES_AUDIT_ALL");
        if(auditPathLR != NULL && auditAllLR != NULL) {
          std::ostringstream o;
          o.precision(17);
          o << "{\"nid\":" << (uintptr_t)childMutable
            << ",\"isRoot\":0,\"leafReveal\":1"
            << ",\"nodeAccN\":1,\"nodeSigma0\":" << cbs.sigma0
            << ",\"parentNid\":" << (uintptr_t)&node
            << ",\"parentSigma0\":" << bs.sigma0 << "}";
          std::ofstream f(auditPathLR, std::ios::app);
          f << o.str() << "\n";
        }
      }
      //M3: initialize the child's own KG state from its fresh n=0 set — the
      //engine analogue of bmcts _reveal -> _expand_seq(child) (algorithms.py
      //_expand_seq/_refresh_set/_recompute_seq), which give a just-revealed
      //internal child D = dKids * max_i w_i immediately.
      BayesSetState css;
      if(bayesComputeSetState(*childMutable, css)) {
        cbs.dKids = css.dKids;
        cbs.dBackup = css.dBackup;
      }
    }
    else if(cbs.anchorFrozen) {
      cbs.vsOwn = ss.stein.vX;  //bmcts _refresh_set: expanded children only get V_X
    }
  }

  //---- Refresh per-entry D now that newly frozen children carry their
  //initial dBackup, and redo the KG backup max ----
  for(int j = 0; j < ss.k; j++)
    ss.D[j] = bayesEntryD(ss.child[j], ss.evaled[j], ss.terminalEvaled[j], ss.dKids);
  ss.dBackup = 0.0;
  for(int j = 0; j < ss.k; j++)
    ss.dBackup = std::max(ss.dBackup, ss.w[j] * ss.D[j]);

  //---- 2d-i resolution-kernel accumulators (docs/bayes-m8-integration.md
  //2d-i engine forms): rebuilt from the children on every recompute.
  //Children on the backup path recompute before this node, so their
  //accumulators are current; off-path subtrees are unchanged. The Q
  //recursion is the ordered-pair decomposition: within-child pairs keep
  //their A (inside accQ_c); every pair first meeting at this node — own
  //eval vs a descendant, or across two children — gets A once. Terminal
  //exact values count in N only (they carry no NN error). ----
  {
    int64_t nAcc = 1;
    double sumPhiS = 0.0;
    double sumPhiS2 = 0.0;
    double sumQ = 0.0;
    for(int j = 0; j < ss.k; j++) {
      if(ss.child[j] == NULL || !ss.evaled[j])
        continue;
      if(ss.terminalEvaled[j]) {
        nAcc += 1;
        continue;
      }
      const BayesNodeState* cbsAcc = ss.child[j]->bayesState;
      if(cbsAcc == NULL || !cbsAcc->anchorFrozen)
        continue;
      double phiC = BayesKernel::phi(cbsAcc->sigma0, bs.sigma0);
      double ps = phiC * cbsAcc->accS;
      nAcc += cbsAcc->accN;
      sumPhiS += ps;
      sumPhiS2 += ps * ps;
      sumQ += cbsAcc->accQ;
    }
    bs.accN = nAcc;
    bs.accS = bs.sigma0 + sumPhiS;
    bs.accQ = bs.sigma0 * bs.sigma0 + sumQ
              + 2.0 * BayesKernel::A * bs.sigma0 * sumPhiS
              + BayesKernel::A * std::max(sumPhiS * sumPhiS - sumPhiS2, 0.0);
  }

  bs.vsKids = ss.vX;
  bs.betaKids = ss.kappaAlpha * bs.bExp;
  if(ss.n > 0) {
    //bmcts _recompute_seq: with no revealed eval the children carry no
    //information (the info-free anchor round trip would inflate variance).
    bs.mu = clipWinrate(ss.mKids);
    bs.b = ss.bOut * bs.betaKids;
    bs.vPriv = ss.vKidsPriv
               + ss.bOut * ss.bOut * std::max(ss.vX - bs.betaKids * bs.betaKids * bs.vsOwn, 0.0);
  }
  double rSum = 0.0;
  for(int j = 0; j < ss.k; j++)
    rSum += ss.w[j] * ss.R[j];
  bs.resolvable = rSum;
  bs.dKids = ss.dKids;
  bs.dBackup = ss.dBackup;

  //M7 Phase C audit dump (docs/bayes-m7-sim2real.md Amendment B);
  //M8 phase 1 extends it to every node via KATAGO_BAYES_AUDIT_ALL
  //(docs/bayes-m8-integration.md). Diagnostic-only: no behavior change
  //unless KATAGO_BAYES_AUDIT is set.
  {
    static const char* auditPath = std::getenv("KATAGO_BAYES_AUDIT");
    static const char* auditAll = std::getenv("KATAGO_BAYES_AUDIT_ALL");
    if(auditPath != NULL && (isRoot || auditAll != NULL))
      bayesAuditDumpRoot(auditPath, node, ss, bs, isRoot);
  }
}

//Append one JSON line with the full root set state after a recompute.
//Single-threaded by the useBayesSearch contract, so plain append is safe.
void Search::bayesAuditDumpRoot(
  const char* path, const SearchNode& node, const BayesSetState& ss,
  const BayesNodeState& bs, bool isRoot) const
{
  const NNOutput* nnOutput = node.getNNOutput();
  double stNode = nnOutput != NULL ? 0.5 * (double)nnOutput->shorttermWinlossError : -1.0;
  int64_t rootVisits = node.stats.visits.load(std::memory_order_acquire);
  std::ostringstream o;
  o.precision(17);
  o << "{\"nid\":" << (uintptr_t)&node
    << ",\"isRoot\":" << (isRoot ? 1 : 0)
    << ",\"rootVisits\":" << rootVisits
    << ",\"nextPla\":\"" << (node.nextPla == P_WHITE ? "W" : "B") << "\""
    << ",\"k\":" << ss.k << ",\"n\":" << ss.n
    << ",\"stNode\":" << stNode
    << ",\"anchMu\":" << bs.anchMu << ",\"anchVar\":" << bs.anchVar
    << ",\"vX\":" << ss.vX << ",\"kappaAlpha\":" << ss.kappaAlpha
    << ",\"dKids\":" << ss.dKids << ",\"mKids\":" << ss.mKids
    << ",\"vKidsPriv\":" << ss.vKidsPriv << ",\"bOut\":" << ss.bOut
    << ",\"dBackup\":" << ss.dBackup
    << ",\"nodeMu\":" << bs.mu << ",\"nodeB\":" << bs.b
    << ",\"nodeVPriv\":" << bs.vPriv << ",\"nodeResolvable\":" << bs.resolvable
    << ",\"nodeSigma0\":" << bs.sigma0
    << ",\"nodeAccN\":" << bs.accN << ",\"nodeAccS\":" << bs.accS
    << ",\"nodeAccQ\":" << bs.accQ
    << ",\"params\":{\"rho\":" << searchParams.bayesRho
    << ",\"sigmaA\":" << searchParams.bayesSigmaA
    << ",\"sigmaB\":" << searchParams.bayesSigmaB
    << ",\"sigmaDA\":" << searchParams.bayesSigmaDA
    << ",\"sigmaDB\":" << searchParams.bayesSigmaDB
    << ",\"dMean\":" << searchParams.bayesDMean
    << ",\"defaultSigma\":" << searchParams.bayesDefaultSigma << "}"
    << ",\"arms\":[";
  for(int j = 0; j < ss.k; j++) {
    int64_t childVisits = 0;
    double childAvg = -1.0;
    if(ss.child[j] != NULL) {
      childVisits = ss.child[j]->stats.visits.load(std::memory_order_acquire);
      NodeStats cst(ss.child[j]->stats);
      if(cst.visits > 0 && cst.weightSum > 0.0)
        childAvg = 0.5 + 0.5 * cst.winLossValueAvg;
    }
    bool frozen = ss.child[j] != NULL && ss.child[j]->bayesState != NULL
                  && ss.child[j]->bayesState->anchorFrozen;
    if(j > 0)
      o << ",";
    o << "{\"loc\":\"" << Location::toString(ss.moveLoc[j], rootBoard) << "\""
      << ",\"prior\":" << ss.prior[j]
      << ",\"evaled\":" << (ss.evaled[j] ? 1 : 0)
      << ",\"terminal\":" << (ss.terminalEvaled[j] ? 1 : 0)
      << ",\"frozen\":" << (frozen ? 1 : 0)
      << ",\"evalWinrate\":" << ss.evalWinrate[j]
      << ",\"evalStErr\":" << ss.evalStErr[j]
      << ",\"steinMu\":" << ss.stein.mus[j]
      << ",\"steinVPriv\":" << ss.stein.vPriv[j]
      << ",\"steinB\":" << ss.stein.b[j]
      << ",\"mu\":" << ss.mu[j]
      << ",\"vPriv\":" << ss.vPriv[j]
      << ",\"b\":" << ss.b[j]
      << ",\"R\":" << ss.R[j]
      << ",\"D\":" << ss.D[j]
      << ",\"w\":" << ss.w[j]
      << ",\"childVisits\":" << childVisits
      << ",\"childAvg\":" << childAvg
      << ",\"accN\":" << (frozen ? ss.child[j]->bayesState->accN : (int64_t)0)
      << ",\"accS\":" << (frozen ? ss.child[j]->bayesState->accS : 0.0)
      << ",\"accQ\":" << (frozen ? ss.child[j]->bayesState->accQ : 0.0)
      << ",\"childSigma0\":" << (frozen ? ss.child[j]->bayesState->sigma0 : 0.0)
      << ",\"childNid\":" << (uintptr_t)ss.child[j] << "}";
  }
  o << "]}";
  std::ofstream f(path, std::ios::app);
  f << o.str() << "\n";
}

//M3 selection fork (bmcts _select_voi with seq_kg, algorithms.py): score
//each entry by contested_j x D_j; contested = the root-decision overlap
//density dE[max]/dv_j at the root (min-root: negate means) and the p_argmax
//weights w_j at interior nodes. Deterministic argmax over viable entries,
//exact score ties toward the better MOVER mean; if nothing is viable, fall
//back to the best mover mean over non-terminal entries.
void Search::bayesSelectBestChildToDescend(
  SearchThread& thread, const SearchNode& node, SearchNodeState nodeState,
  int& numChildrenFound, int& bestChildIdx, Loc& bestChildMoveLoc, bool& countEdgeVisit,
  bool isRoot) const
{
  bestChildIdx = -1;
  bestChildMoveLoc = Board::NULL_LOC;
  countEdgeVisit = true;

  //Count existing allocated children (the PUCT path's new-child-slot contract:
  //bestChildIdx == numChildrenFound signals a brand-new child).
  ConstSearchNodeChildrenReference children = node.getChildren(nodeState);
  int childrenCapacity = children.getCapacity();
  numChildrenFound = 0;
  for(int i = 0; i < childrenCapacity; i++) {
    if(children[i].getIfAllocated() == NULL)
      break;
    numChildrenFound++;
  }

  BayesSetState ss;
  if(!bayesComputeSetState(node, ss)) {
    //Guarded at the call site (anchor frozen, NN output present); if the set
    //is somehow empty, signal the same way the PUCT path signals a node with
    //no selectable move (bestChildIdx = -1: caller adds a leaf value).
    return;
  }

  //Root-move legality: same filters the PUCT new-child loop applies at the
  //root (isAllowedRootMove + avoidMoveUntilByLoc); interior nodes need none.
  const std::vector<int>& avoidMoveUntilByLoc = thread.pla == P_BLACK ? avoidMoveUntilByLocBlack : avoidMoveUntilByLocWhite;
  std::vector<bool> allowed(ss.k, true);
  if(isRoot) {
    for(int j = 0; j < ss.k; j++) {
      Loc moveLoc = ss.moveLoc[j];
      if(moveLoc == Board::NULL_LOC) {
        allowed[j] = false;
        continue;
      }
      if(!isAllowedRootMove(moveLoc)) {
        allowed[j] = false;
        continue;
      }
      if(avoidMoveUntilByLoc.size() > 0) {
        assert(avoidMoveUntilByLoc.size() >= Board::MAX_ARR_SIZE);
        int untilDepth = avoidMoveUntilByLoc[moveLoc];
        if((int)(thread.history.moveHistory.size() - rootHistory.moveHistory.size()) < untilDepth)
          allowed[j] = false;
      }
    }
  }

  //Score vector: M3 voi-KG (contested x D on the set state) or M8 2d
  //contrast-voi (contested x D in contrast units; docs/bayes-m8-integration
  //.md 2d registration).
  const double moverSign = (node.nextPla == P_WHITE) ? 1.0 : -1.0;
  std::vector<double> scoreV(ss.k, 0.0);
  std::vector<double> tieV(ss.k, 0.0);
  std::vector<bool> resolvedV(ss.k, false);
  if(searchParams.useBayesContrastSelection) {
    //---- Contrast state (mover persp; mirrors the 2a chooser) ----
    const double rho = searchParams.bayesRho;
    const NNOutput* nnOut = node.getNNOutput();
    double stNode = nnOut != NULL ? 0.5 * (double)nnOut->shorttermWinlossError : 0.0;
    if(stNode <= 1e-8)
      stNode = searchParams.bayesDefaultSigma;
    double sr2 = std::exp(searchParams.bayesSigmaDA
                          + searchParams.bayesSigmaDB * std::log(std::max(stNode * stNode, 1e-8)));
    std::vector<int64_t> nVis(ss.k, 0);
    std::vector<double> cAvg(ss.k, -1.0);
    for(int j = 0; j < ss.k; j++) {
      if(ss.child[j] != NULL) {
        NodeStats cst(ss.child[j]->stats);
        nVis[j] = cst.visits;
        if(cst.visits > 0 && cst.weightSum > 0.0)
          cAvg[j] = 0.5 + 0.5 * cst.winLossValueAvg;
      }
    }
    //M8 2d-i resolution kernel (same forms as the chooser;
    //docs/bayes-m8-integration.md 2d-i engine forms).
    const double sNode = node.bayesState->sigma0;
    auto armAcc = [&](int j, int64_t& nA, double& sA, double& qA, double& phiA) {
      const SearchNode* ch = ss.child[j];
      if(ch != NULL && ch->bayesState != NULL && ch->bayesState->anchorFrozen
         && !ss.terminalEvaled[j]) {
        const BayesNodeState& cbs = *ch->bayesState;
        nA = cbs.accN;
        sA = cbs.accS;
        qA = cbs.accQ;
        phiA = BayesKernel::phi(cbs.sigma0, sNode);
      }
      else {
        double s = bayesSigmaFromStErr(ss.evalStErr[j]);
        nA = 1;
        sA = s;
        qA = s * s;
        phiA = BayesKernel::phi(s, sNode);
      }
    };
    int rIdx = 0;
    for(int j = 1; j < ss.k; j++) {
      if(nVis[j] > nVis[rIdx] || (nVis[j] == nVis[rIdx] && ss.prior[j] > ss.prior[rIdx]))
        rIdx = j;
    }
    const bool refIsAnchor = !(nVis[rIdx] >= 1 && cAvg[rIdx] >= 0.0);
    int64_t nR = 1;
    double sR = 0.0, qR = 0.0, phiR = 0.0;
    if(!refIsAnchor)
      armAcc(rIdx, nR, sR, qR, phiR);
    auto contrastNoise = [&](int64_t nA, double sA, double qA, double phiA) -> double {
      double varA = qA / ((double)nA * (double)nA);
      if(refIsAnchor)
        return varA + sNode * sNode
               - 2.0 * BayesKernel::A * phiA * sA * sNode / (double)nA;
      double varR = qR / ((double)nR * (double)nR);
      double cov = BayesKernel::A * phiA * phiR * sA * sR
                   / ((double)nA * (double)nR);
      return varA + varR - 2.0 * cov;
    };
    double Lr = !refIsAnchor
                ? 0.5 + moverSign * (cAvg[rIdx] - 0.5)
                : 0.5 + moverSign * (node.bayesState->anchMu - 0.5);
    double logPr = std::log(std::max(ss.prior[rIdx], 1e-12));
    const double pv = 2.0 * sr2;
    std::vector<double> gm(ss.k, 0.0), gv(ss.k, 1e-12), gvNext(ss.k, 1e-12);
    for(int j = 0; j < ss.k; j++) {
      if(j == rIdx)
        continue;
      double pm = searchParams.bayesDMean
                  * (std::log(std::max(ss.prior[j], 1e-12)) - logPr);
      double y = 0.0, nv = 0.0, nvNext = 0.0;
      bool haveEv = false;
      if(nVis[j] >= 1 && cAvg[j] >= 0.0) {
        y = 0.5 + moverSign * (cAvg[j] - 0.5) - Lr;
        int64_t nA;
        double sA, qA, phiA;
        armAcc(j, nA, sA, qA, phiA);
        nv = contrastNoise(nA, sA, qA, phiA);
        nvNext = nv;  //2d-c: D dropped; kept only for dead-code symmetry
        haveEv = true;
      }
      else if(ss.evaled[j]) {
        y = 0.5 + moverSign * (ss.evalWinrate[j] - 0.5) - Lr;
        double s = bayesSigmaFromStErr(ss.evalStErr[j]);
        nv = contrastNoise(1, s, s * s, BayesKernel::phi(s, sNode));
        nvNext = nv;
        haveEv = true;
      }
      if(haveEv) {
        nv = std::max(nv, 1e-9);
        nvNext = std::max(nvNext, 1e-9);
        double w = pv / (pv + nv);
        gm[j] = pm + w * (y - pm);
        gv[j] = pv * nv / (pv + nv);
        gvNext[j] = pv * nvNext / (pv + nvNext);
      }
      else {
        gm[j] = pm;
        gv[j] = pv;
      }
    }
    (void)rho;  //superseded for contrasts by w_cross (2d-g registration)
    //Contested weights + overlap from the contrast field (ref at 0).
    std::vector<double> fieldMu(ss.k), fieldVar(ss.k);
    for(int j = 0; j < ss.k; j++) {
      fieldMu[j] = (j == rIdx) ? 0.0 : gm[j];
      fieldVar[j] = (j == rIdx) ? 1e-12 : gv[j];
    }
    std::vector<double> contested = isRoot
      ? BayesPosterior::overlapDensity(fieldMu, fieldVar)
      : [&]() {
          double m, v;
          std::vector<double> w;
          BayesPosterior::clarkMaxAndWeights(fieldMu, fieldVar, m, v, w);
          return w;
        }();
    //2d-c (docs/bayes-m8-integration.md): resolution allocation. The
    //variance-drop currency D is refuted by the 2d-b noise-curve study
    //(mean error is resolution-limited, not sample-limited); visits buy
    //resolution AT THE DECISION BOUNDARY: score = contested alone at
    //the root, local w at interior nodes (PV extension).
    (void)gvNext;
    for(int j = 0; j < ss.k; j++) {
      //2d-k-b (docs/bayes-m8-integration.md, registered shootout: H
      //pins at 0.868 top-quartile capture): the ROOT score prices
      //decision relevance x remaining contrast uncertainty x
      //diminishing returns. The reference arm's gv ~ 0 kills its score
      //once it holds the visit lead — the 2d-i-b hoarding lock-in ends
      //by construction. Interior nodes keep the 2d-c w-routing.
      scoreV[j] = isRoot
        ? contested[j] * gv[j] / ((double)nVis[j] + 1.0)
        : contested[j];
      tieV[j] = (j == rIdx) ? 0.0 : gm[j];
      resolvedV[j] = false;
    }
  }
  else {
    //Contested: overlap density at the root, p_argmax weights w_j at interior
    //nodes (bmcts _decision_scores).
    std::vector<double> contested;
    if(isRoot) {
      const bool minRoot = (node.nextPla == P_BLACK);
      std::vector<double> mus(ss.k), tot(ss.k);
      for(int j = 0; j < ss.k; j++) {
        //Total variance per entry, the same total the backup sees: frozen
        //children b^2*vsOwn + vPriv, virtual entries b^2*vX + vPriv.
        bool frozen = ss.child[j] != NULL && ss.child[j]->bayesState != NULL
                      && ss.child[j]->bayesState->anchorFrozen;
        double vShared = frozen ? ss.child[j]->bayesState->vsOwn : ss.vX;
        tot[j] = ss.b[j] * ss.b[j] * vShared + ss.vPriv[j];
        mus[j] = minRoot ? -ss.mu[j] : ss.mu[j];
      }
      contested = BayesPosterior::overlapDensity(mus, tot);
    }
    else {
      contested = ss.w;
    }
    for(int j = 0; j < ss.k; j++) {
      scoreV[j] = contested[j] * ss.D[j];
      tieV[j] = moverSign * ss.mu[j];
      resolvedV[j] = (ss.D[j] <= 1e-18);
    }
  }

  //Deterministic argmax of the score over viable entries; exact score ties
  //break toward the better mover mean.
  int bestJ = -1;
  double bestScore = 0.0;
  double bestMoverMu = 0.0;
  for(int j = 0; j < ss.k; j++) {
    if(!allowed[j])
      continue;
    bool viable;
    if(!ss.evaled[j])
      viable = true;  //unallocated/unevaluated entries are always viable
    else if(ss.terminalEvaled[j])
      viable = false; //terminal children with revealed values are never selected
    else
      viable = !resolvedV[j];  //exhausted/resolved entries are done
    if(!viable)
      continue;
    double score = scoreV[j];
    double moverMu = tieV[j];
    if(bestJ < 0 || score > bestScore || (score == bestScore && moverMu > bestMoverMu)) {
      bestJ = j;
      bestScore = score;
      bestMoverMu = moverMu;
    }
  }
  //Fallback: nothing viable -> best mover mean over non-terminal entries.
  if(bestJ < 0) {
    for(int j = 0; j < ss.k; j++) {
      if(!allowed[j])
        continue;
      if(ss.terminalEvaled[j])
        continue;
      double moverMu = moverSign * ss.mu[j];
      if(bestJ < 0 || moverMu > bestMoverMu) {
        bestJ = j;
        bestMoverMu = moverMu;
      }
    }
  }
  if(bestJ < 0)
    return;  //literally nothing selectable: terminal semantics (bestChildIdx = -1)

  Loc chosenLoc = ss.moveLoc[bestJ];
  bestChildMoveLoc = chosenLoc;
  bestChildIdx = numChildrenFound;  //default: the new-child slot
  for(int i = 0; i < numChildrenFound; i++) {
    if(children[i].getMoveLocRelaxed() == chosenLoc) {
      bestChildIdx = i;
      break;
    }
  }
}
