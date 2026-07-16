//Bayesian posterior side-state maintenance (useBayesSearch), M2.
//
//This is the engine port of the bmcts round-3 sequential-reveal stack
//(docs/bayes-m2-design-notes.md): Stein-corrected d-mean-aware shrinkage
//(P-var projection) for fresh/partially-revealed sibling sets, and the
//Clark/quadrature max backup (node_posterior_from_children) for the node
//posterior. At M2 this state is a pure PASSENGER: selection still runs
//PUCT on the ordinary stats; gates audit the posterior.
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
//
//Known M2 caveats, documented: terminal children enter the eval set with
//the shared homogeneous v_u (their values are exact; conservative);
//tree reuse keeps anchors frozen in a previous search's state; noResult
//mass is ignored (tromp-taylor).

#include "../search/search.h"
#include "../search/searchnode.h"
#include "../search/bayesnodestate.h"
#include "../search/bayesposterior.h"

#include <algorithm>
#include <cmath>
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

//One entry per legal move of the node, in descending-prior order.
struct BayesSetEntry {
  int pos;                    //policy index
  double prior;
  const SearchNode* child;    //allocated child node, if any
  SearchNode* childMutable;
  bool evaled;                //has an eval (own NN output, or terminal value)
  double evalWinrate;         //valid iff evaled
  double evalStErr;           //0.5*shorttermWinlossError, 0 if unavailable
};

void Search::bayesRecomputeNodeStats(SearchNode& node, bool isRoot) {
  if(!searchParams.useBayesSearch)
    return;
  const NNOutput* nnOutput = node.getNNOutput();
  if(nnOutput == NULL)
    return;

  const double rho = searchParams.bayesRho;
  const bool isMaxNode = (node.nextPla == P_WHITE);

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
    node.bayesState = bs;
  }
  BayesNodeState& bs = *node.bayesState;
  if(!bs.anchorFrozen)
    return;

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
    e.childMutable = NULL;
    e.evaled = false;
    e.evalWinrate = 0.0;
    e.evalStErr = 0.0;
    set_.push_back(e);
  }
  const int k = (int)set_.size();
  if(k <= 0)
    return;
  std::stable_sort(set_.begin(), set_.end(),
                   [](const BayesSetEntry& a, const BayesSetEntry& b) { return a.prior > b.prior; });
  std::vector<int> posToSetIdx(policySize, -1);
  for(int j = 0; j < k; j++)
    posToSetIdx[set_[j].pos] = j;

  //---- Attach allocated children and their evals ----
  SearchNodeChildrenReference children = node.getChildren();
  int childrenCapacity = children.getCapacity();
  for(int i = 0; i < childrenCapacity; i++) {
    SearchChildPointer& childPointer = children[i];
    SearchNode* child = childPointer.getIfAllocated();
    if(child == NULL)
      break;
    int pos = getPos(childPointer.getMoveLocRelaxed());
    if(pos < 0 || pos >= policySize || posToSetIdx[pos] < 0)
      continue;
    BayesSetEntry& e = set_[posToSetIdx[pos]];
    e.child = child;
    e.childMutable = child;
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
  BayesPosterior::SteinShrink st = BayesPosterior::shrinkSiblingsStein(
    evals, evalIdx, k, vU, varS, bs.anchMu, bs.anchVar, sigmaR, eD, vD, g, &dMeans);

  //---- Freeze newly-evaled children; refresh vsOwn on frozen ones ----
  for(int j = 0; j < k; j++) {
    BayesSetEntry& e = set_[j];
    if(e.childMutable == NULL)
      continue;
    bool isTerminalChild = e.evaled && e.child->getNNOutput() == NULL;
    if(isTerminalChild)
      continue;  //terminals carry no set of their own; handled via Stein state below
    if(e.evaled && e.childMutable->bayesState == NULL)
      e.childMutable->bayesState = new BayesNodeState();
    if(e.childMutable->bayesState == NULL)
      continue;
    BayesNodeState& cbs = *e.childMutable->bayesState;
    if(e.evaled && !cbs.anchorFrozen) {
      cbs.mu = clipWinrate(st.mus[j]);
      cbs.b = st.b[j];
      cbs.vPriv = st.vPriv[j];
      cbs.vsOwn = st.vX;
      cbs.anchMu = cbs.mu;
      cbs.anchVar = cbs.b * cbs.b * st.vX + cbs.vPriv;
      cbs.bExp = cbs.b;
      cbs.anchorFrozen = true;
      cbs.mu0 = e.evalWinrate;
      cbs.sigma0 = bayesSigmaFromStErr(e.evalStErr);
      cbs.resolvable = cbs.anchVar;  //unexpanded non-leaf: fully resolvable
    }
    else if(cbs.anchorFrozen) {
      cbs.vsOwn = st.vX;  //bmcts _refresh_set: expanded children only get V_X
    }
  }

  //---- Assemble per-child posteriors and back up through the max ----
  std::vector<double> mus(k), vPrivs(k), bsVec(k), rVec(k);
  for(int j = 0; j < k; j++) {
    const BayesSetEntry& e = set_[j];
    bool frozen = e.child != NULL && e.child->bayesState != NULL
                  && e.child->bayesState->anchorFrozen;
    if(frozen) {
      const BayesNodeState& cbs = *e.child->bayesState;
      mus[j] = cbs.mu;
      vPrivs[j] = cbs.vPriv;
      bsVec[j] = cbs.b;
      rVec[j] = cbs.resolvable;
    }
    else {
      mus[j] = clipWinrate(st.mus[j]);
      vPrivs[j] = st.vPriv[j];
      bsVec[j] = st.b[j];
      bool isTerminalChild = e.evaled && e.child != NULL && e.child->getNNOutput() == NULL;
      rVec[j] = isTerminalChild ? 0.0 : (st.b[j] * st.b[j] * st.vX + st.vPriv[j]);
    }
  }
  double mKids, vKidsPriv, bOut;
  std::vector<double> w;
  BayesPosterior::nodePosteriorFromChildren(mus, vPrivs, bsVec, st.vX, isMaxNode,
                                            mKids, vKidsPriv, bOut, w);

  bs.vsKids = st.vX;
  bs.betaKids = st.kappaAlpha * bs.bExp;
  if(n > 0) {
    //bmcts _recompute_seq: with no revealed eval the children carry no
    //information (the info-free anchor round trip would inflate variance).
    bs.mu = clipWinrate(mKids);
    bs.b = bOut * bs.betaKids;
    bs.vPriv = vKidsPriv
               + bOut * bOut * std::max(st.vX - bs.betaKids * bs.betaKids * bs.vsOwn, 0.0);
  }
  double rSum = 0.0;
  for(int j = 0; j < k; j++)
    rSum += w[j] * rVec[j];
  bs.resolvable = rSum;
}
