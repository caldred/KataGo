#ifndef SEARCH_BAYESPOSTERIOR_H_
#define SEARCH_BAYESPOSTERIOR_H_

#include <utility>
#include <vector>

//Exact C++ port of the bmcts posterior math (bmcts repo, src/bmcts/posterior.py).
//Gaussian machinery: Clark's moment-matched max of Gaussians, P(argmax) weights,
//hierarchical shrinkage of fresh sibling sets, and the two-component-error
//(private + sibling-shared) children-derived node posterior.
//
//All ops treat min nodes by negation: min(X) = -max(-X).
//
//Correlation handling (two-component error filter): sibling a's estimate error
//is decomposed as err_a = b_a * s + eta_a where s ~ N(0, var_s) is the
//parent-level draw shared by ALL siblings, b_a is a's coefficient on it, and
//eta_a ~ N(0, v_priv_a) is independent across siblings. Conditional on s the
//siblings are independent, so the max is handled by 1-D Gauss-Hermite
//quadrature over s with a Clark moment-match at each quadrature point.
//
//This library is deliberately dependency-free (vectors and doubles only) and
//must stay numerically identical to the Python reference: same clamps, same
//1e-12/1e-18 floors, same descending-mean processing order (stable argsort).
//Golden-value tests against the Python implementation live in
//tests/testbayes.cpp / tests/data/bayesgolden.txt.
namespace BayesPosterior {

  //(mean, variance) of the max of k iid standard normals.
  //Numerical integration against the exact density k*phi(x)*Phi(x)^(k-1),
  //x in [-12,12] with 100001 points, Phi floored at 1e-300; cached per k.
  //k == 1 returns (0,1). NOT thread-safe (static cache); single-threaded use only.
  std::pair<double,double> stdMaxMoments(int k);

  //Moment-matched Gaussian approx of max(X1, X2), X_i ~ N(m_i, v_i) independent.
  //If v1+v2 < 1e-18, returns (max(m1,m2), 0); otherwise v is floored at 1e-12.
  void clarkMaxPair(double m1, double v1, double m2, double v2, double& m, double& v);

  //(E[max], Var[max], P(argmax)) of independent Gaussians in one pass over
  //shared prefix/suffix Clark accumulators, in descending-mean order.
  //Weights are normalized to sum 1 (uniform fallback if the sum is <= 0).
  //k == 1 returns (mus[0], vars[0], {1}).
  void clarkMaxAndWeights(
    const std::vector<double>& mus, const std::vector<double>& vars,
    double& mOut, double& vOut, std::vector<double>& wOut
  );

  //dE[max]/dv_i for independent Gaussians: the decision-boundary density
  //0.5*phi(Delta_i/s_i)/s_i with the leave-one-out rest moment-matched via
  //the same prefix/suffix Clark accumulators. k == 1 returns {0}.
  std::vector<double> overlapDensity(
    const std::vector<double>& mus, const std::vector<double>& vars
  );

  //Random-effects posterior for a fresh sibling set, anchored on the parent.
  //Exact port of bmcts shrink_siblings; see the Python docstring for the full
  //derivation. (cK, vK) are the within-set max moments of d: pass
  //stdMaxMoments(k) for the Gaussian-d fallback, or head-provided moments.
  //Min nodes flip the sign of the anchor offset. If sigmaD^2 + vU <= 1e-18
  //(exact evals of identical values) returns musOut = evals, 0, 0, 0.
  //vPrivOut is scalar (common to the fresh set); vXOut is floored at 0.
  void shrinkSiblings(
    const std::vector<double>& evals,
    double vU, double varS, double sigmaD,
    double anchorMu, double anchorVar,
    bool isMaxNode, double cK, double vK,
    std::vector<double>& musOut, double& vPrivOut, double& vXOut, double& kappaAlphaOut
  );

  //(E_D, v_D, g) of the within-set extreme D = max/min_a (m_a + delta_a),
  //delta_a iid N(0, sigmaR^2): Clark moments over {N(m_a, sigmaR^2)} (exact at
  //k = 2, standard Clark approximation above), and the homogenized Stein
  //covariance g = Cov(D, d_a) averaged over the set — exactly sigmaR^2/k.
  //eD is SIGNED (min nodes negative-mirrored). Exact port of bmcts
  //extreme_moments_gaussian.
  void extremeMomentsGaussian(
    const std::vector<double>& dMeans, double sigmaR, bool isMaxNode,
    double& eD, double& vD, double& g
  );

  //Result of shrinkSiblingsStein: full-vector per-child state (mus, vPriv, b
  //all length k) plus the exact covariance blocks (diagnostics; the (b, vX)
  //claim approximates cUU/cUE per the P-var projection). Mirrors the Python
  //SteinShrink named tuple field-for-field (vU here is the tuple's v_u output,
  //the unevaluated-child total variance, not the input measurement noise).
  struct SteinShrink {
    std::vector<double> mus, vPriv, b;
    double vX, kappaAlpha, cEE, cUU, cUE, vE, vU;
  };

  //Stein-corrected, d-mean-aware partial-observation posterior. Exact port of
  //bmcts shrink_siblings_stein (M2 round 3); see the Python docstring for the
  //full derivation. evals are the n observed values, evalIdx their child
  //indices in [0, k). (eD, vD, g) are the D_moments triple, eD signed (caller
  //mirrors min nodes). dMeans may be NULL (treated as all zeros). If
  //sigmaR^2 + vU <= 1e-18 (exact identical residuals) evaluated children come
  //back exact and everything else rides the mean residual shift, all
  //variances/blocks zero. c0 = anchorVar + vD - 2g may be NEGATIVE and is
  //deliberately not clamped; only the documented output clamps apply
  //(vPriv floors at 0, kappaAlpha clipped to [0,1], den floored at 1e-12).
  SteinShrink shrinkSiblingsStein(
    const std::vector<double>& evals, const std::vector<int>& evalIdx,
    int k, double vU, double varS, double anchorMu, double anchorVar,
    double sigmaR, double eD, double vD, double g,
    const std::vector<double>* dMeans
  );

  //Children-derived measurement of a node's value with two-component errors.
  //Child a's estimate error is b_a*s + eta_a, s ~ N(0, varS) shared by all
  //children, eta_a ~ N(0, childVPrivs[a]) independent. Integrates the Clark
  //moment-match over s with 9-point probabilists' Gauss-Hermite quadrature;
  //homogeneous-b fast path (varS <= 0 or ptp(bs) < 1e-12) passes the shared
  //part through additively without quadrature (bOut = bs[0], or 0 if varS <= 0).
  //Outputs: mOut = E[max], vPrivOut = private variance (floored 1e-12 on the
  //quadrature path), bOut = cov(max, s)/varS, wOut = P(child attains the
  //max/min) averaged over s. Min nodes handled by negating mus and bs.
  //forceQuadrature is for testing only: it skips the homogeneous fast path
  //when varS > 0 so the two paths can be compared.
  void nodePosteriorFromChildren(
    const std::vector<double>& childMus,
    const std::vector<double>& childVPrivs,
    const std::vector<double>& childBs,
    double varS, bool isMaxNode,
    double& mOut, double& vPrivOut, double& bOut, std::vector<double>& wOut,
    bool forceQuadrature = false
  );

}

#endif  // SEARCH_BAYESPOSTERIOR_H_
