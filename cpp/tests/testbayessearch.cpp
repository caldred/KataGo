#include "../tests/tests.h"

#include <cmath>

#include "../neuralnet/nninputs.h"
#include "../search/search.h"
#include "../search/searchnode.h"
#include "../search/bayesnodestate.h"
#include "../search/reportedsearchvalues.h"
#include "../tests/testsearchcommon.h"

using namespace std;
using namespace TestSearchCommon;

//M2 Gate A (docs/bayes-m2-gate.md): model-free correctness tests of the
//Bayesian posterior passenger state (search/bayessearch.cpp), run with
//debugSkipNeuralNet = true.
//
//A1: posterior invariants over every node with bayesState, plus the
//    freeze-by-next-parent-recompute contract.
//A2: terminal-dominance minimax agreement on decided 5x5 endgames.
//A3 (reported, not gated): bayes root mu vs ordinary search winrate.

static SearchParams makeBayesTestParams(int64_t maxVisits) {
  SearchParams params;
  params.maxVisits = maxVisits;
  params.useBayesSearch = true;
  params.numThreads = 1;
  params.useGraphSearch = false;
  params.useUncertainty = false;
  params.bayesDefaultSigma = 0.02;
  params.bayesRho = 0.26;
  params.bayesDMean = 0.0638;
  //Fixed eval sigma of 0.02 (the gate doc's A2 tolerance basis). NOTE: the
  //debugSkipNeuralNet stub does NOT produce shorttermWinlossError = 0: the
  //zero logit goes through the standard NN postprocessing softplus, yielding
  //a constant 2*softPlus(0)*sqrt(0.25) = ln(2)/2 ~= 0.3466, i.e. st ~= 0.1733.
  //So the sigma head must be pinned flat (B = 0) rather than identity
  //(A = 0, B = 1) to get the intended fixed sigma; with the identity head the
  //homogeneous v_u becomes ~0.022 and exact terminal values are shrunk ~70%
  //toward stale anchors, which is not what A2 pre-registered.
  params.bayesSigmaA = std::log(0.02);
  params.bayesSigmaB = 0.0;
  //Fixed sigma_r ~ 0.05: log sigma_r^2 = -6.0, no slope.
  params.bayesSigmaDA = -6.0;
  params.bayesSigmaDB = 0.0;
  return params;
}

struct BayesWalkStats {
  int numNodesWithState = 0;
  int numFrozen = 0;
  int numUnfrozenEvaledChildren = 0;
};

//Recursively walk the completed (single-threaded) search tree checking the
//A1 invariants on every node holding a bayesState, and the freeze contract:
//every evaluated non-terminal child of a bayes-computed (anchorFrozen)
//parent must itself be frozen; at most ONE unfrozen evaled child per parent
//is tolerated (the very last playout's leaf), and it is reported.
static void walkAndCheckBayes(const Search* search, const SearchNode* node, const Board& board, BayesWalkStats& ws) {
  const BayesNodeState* bs = node->bayesState;
  if(bs != NULL) {
    ws.numNodesWithState++;
    testAssert(bs->vPriv >= 0.0);
    testAssert(bs->vsOwn >= 0.0);
    testAssert(bs->vsKids >= 0.0);
    testAssert(std::isfinite(bs->mu));
    testAssert(std::isfinite(bs->b));
    testAssert(std::isfinite(bs->betaKids));
    testAssert(std::isfinite(bs->resolvable));
    testAssert(bs->mu >= 0.01 && bs->mu <= 0.99);
    testAssert(bs->resolvable >= 0.0);
    if(bs->anchorFrozen) {
      ws.numFrozen++;
      testAssert(node->getNNOutput() != NULL || node == search->rootNode);
    }
  }

  ConstSearchNodeChildrenReference children = node->getChildren();
  int childrenCapacity = children.getCapacity();
  int numUnfrozenEvaledHere = 0;
  for(int i = 0; i < childrenCapacity; i++) {
    const SearchNode* child = children[i].getIfAllocated();
    if(child == NULL)
      break;
    //Freeze-by-next-parent-recompute contract: only checked for
    //bayes-computed parents, only for children with their own NN output
    //(terminal children have none and carry no state of their own).
    if(bs != NULL && bs->anchorFrozen && child->getNNOutput() != NULL) {
      if(child->bayesState == NULL || !child->bayesState->anchorFrozen) {
        numUnfrozenEvaledHere++;
        cout << "A1 report: unfrozen evaled child of frozen parent at move "
             << Location::toString(children[i].getMoveLocRelaxed(), board) << endl;
      }
    }
    walkAndCheckBayes(search, child, board, ws);
  }
  //The backup unwinds leaf->root so the parent recompute runs right after a
  //child's first eval lands; after a completed search at most the very last
  //playout's leaf may be unfrozen.
  testAssert(numUnfrozenEvaledHere <= 1);
  ws.numUnfrozenEvaledChildren += numUnfrozenEvaledHere;
}

static double whiteWinrateOf(const Search* search) {
  ReportedSearchValues values = search->getRootValuesRequireSuccess();
  return 0.5 * (1.0 + values.winLossValue);
}

//Run one A1 search on the given position, walk the tree, print the A3 line.
static void runA1Case(
  NNEvaluator* nnEval, Logger& logger, const string& label, const string& searchSeed,
  int64_t maxVisits, const Board& board, const BoardHistory& hist, Player nextPla
) {
  SearchParams params = makeBayesTestParams(maxVisits);
  Search* search = new Search(params, nnEval, &logger, searchSeed);
  search->setPosition(nextPla, board, hist);
  search->runWholeSearch(nextPla);
  testAssert(search->rootNode != NULL);
  testAssert(search->rootNode->bayesState != NULL);
  testAssert(search->rootNode->bayesState->anchorFrozen);

  BayesWalkStats ws;
  walkAndCheckBayes(search, search->rootNode, board, ws);
  testAssert(ws.numNodesWithState > 0);

  double bayesMu = search->rootNode->bayesState->mu;
  double searchWinrate = whiteWinrateOf(search);
  cout << "A3 root: " << label
       << " bayesMu=" << Global::strprintf("%.4f", bayesMu)
       << " searchWinrate=" << Global::strprintf("%.4f", searchWinrate)
       << " diff=" << Global::strprintf("%+.4f", bayesMu - searchWinrate)
       << " (bayesNodes=" << ws.numNodesWithState
       << " frozen=" << ws.numFrozen
       << " unfrozenEvaledChildren=" << ws.numUnfrozenEvaledChildren << ")"
       << endl;
  delete search;
}

//Play ~8 reasonable stub moves from the empty board to get a mid-game position.
static void makeMidGamePosition(
  NNEvaluator* nnEval, Logger& logger, const string& seed, const Rules& rules,
  Board& board, BoardHistory& hist, Player& pla
) {
  board = Board(board.x_size, board.y_size);
  hist = BoardHistory(board, P_BLACK, rules, 0);
  pla = P_BLACK;
  SearchParams params = makeBayesTestParams(20);
  Search* search = new Search(params, nnEval, &logger, seed + "midgen");
  for(int i = 0; i < 8; i++) {
    search->setPosition(pla, board, hist);
    Loc moveLoc = search->runWholeSearchAndGetMove(pla);
    if(moveLoc == Board::NULL_LOC || moveLoc == Board::PASS_LOC)
      break;
    bool suc = hist.makeBoardMoveTolerant(board, moveLoc, pla);
    if(!suc)
      break;
    pla = getOpp(pla);
    if(hist.isGameFinished)
      break;
  }
  delete search;
}

struct A2Tally {
  int numQualifying = 0;
  int numQualifyingBlackWon = 0;
  int numQualifyingWhiteWon = 0;
  int numQualifyingBlackToMove = 0;
  int numQualifyingWhiteToMove = 0;
};

//A2: decided 5x5 endgame; ordinary PUCT search (bayes still a passenger) at
//high visits must have converged to <0.02 or >0.98, then the bayes root mu
//must agree with the (rounded) exhausted-search minimax value within 0.06.
static void runA2Case(
  NNEvaluator* nnEval, Logger& logger, const char* name,
  const string& boardStr, Player nextPla, float komi, A2Tally& tally
) {
  Rules rules = Rules::getTrompTaylorish();
  rules.komi = komi;
  Board board = Board::parseBoard(5, 5, boardStr);
  BoardHistory hist(board, nextPla, rules, 0);

  SearchParams params = makeBayesTestParams(3000);
  Search* search = new Search(params, nnEval, &logger, string("bayesA2") + name);
  search->setPosition(nextPla, board, hist);
  search->runWholeSearch(nextPla);
  testAssert(search->rootNode != NULL);
  testAssert(search->rootNode->bayesState != NULL);

  double ordinaryWinrate = whiteWinrateOf(search);
  double bayesMu = search->rootNode->bayesState->mu;
  bool qualifies = ordinaryWinrate < 0.02 || ordinaryWinrate > 0.98;
  double minimax = ordinaryWinrate > 0.5 ? 1.0 : 0.0;
  double gap = std::fabs(bayesMu - minimax);
  cout << "A2 gap: " << name
       << " toMove=" << PlayerIO::playerToString(nextPla)
       << " ordinaryWinrate=" << Global::strprintf("%.4f", ordinaryWinrate)
       << " minimax=" << minimax
       << " bayesMu=" << Global::strprintf("%.4f", bayesMu)
       << " gap=" << Global::strprintf("%.4f", gap)
       << (qualifies ? "" : " (NOT QUALIFYING, ordinary search not converged)")
       << endl;

  if(qualifies) {
    tally.numQualifying++;
    if(minimax == 0.0) tally.numQualifyingBlackWon++;
    else tally.numQualifyingWhiteWon++;
    if(nextPla == P_BLACK) tally.numQualifyingBlackToMove++;
    else tally.numQualifyingWhiteToMove++;
    testAssert(gap <= 0.06);
  }

  //Also check the A1 invariants on this search for free.
  BayesWalkStats ws;
  walkAndCheckBayes(search, search->rootNode, board, ws);
  delete search;
}

void Tests::runBayesSearchTests() {
  cout << "Running bayes search tests" << endl;
  NeuralNet::globalInitialize();
  //runtests frees the score tables before reaching us; search needs them.
  ScoreValue::initTables();

  const bool logToStdout = false;
  const bool logToStderr = false;
  const bool logTime = false;
  Logger logger(nullptr, logToStdout, logToStderr, logTime);

  //Placeholder, doesn't actually do anything since we have debugSkipNeuralNet = true
  string modelFile = "/dev/null";

  //-------------------------------------------------------------------------
  //A1 + A3: invariants at maxVisits {20,80,300} on 5x5 and 7x7, empty board
  //and mid-game, 2 rng seeds.
  //-------------------------------------------------------------------------
  const vector<string> nnSeeds = {"bayesGateSeedA", "bayesGateSeedB"};
  const vector<int64_t> visitSchedule = {20, 80, 300};
  for(const string& nnSeed: nnSeeds) {
    NNEvaluator* nnEval = startNNEval(
      modelFile, logger, nnSeed, NNPos::MAX_BOARD_LEN, NNPos::MAX_BOARD_LEN,
      0, true, false, false, true, false);
    for(int size: {5, 7}) {
      Rules rules = Rules::getTrompTaylorish();
      //Empty board
      for(int64_t maxVisits: visitSchedule) {
        Board board(size, size);
        BoardHistory hist(board, P_BLACK, rules, 0);
        string label = Global::strprintf("%s %dx%d empty v=%d", nnSeed.c_str(), size, size, (int)maxVisits);
        runA1Case(nnEval, logger, label, "bayesA1" + nnSeed + Global::intToString(size) + "e" + Global::int64ToString(maxVisits),
                  maxVisits, board, hist, P_BLACK);
      }
      //Mid-game position: ~8 stub moves from empty
      {
        Board board(size, size);
        BoardHistory hist(board, P_BLACK, rules, 0);
        Player pla = P_BLACK;
        makeMidGamePosition(nnEval, logger, "bayesMid" + nnSeed + Global::intToString(size), rules, board, hist, pla);
        for(int64_t maxVisits: visitSchedule) {
          string label = Global::strprintf("%s %dx%d midgame v=%d", nnSeed.c_str(), size, size, (int)maxVisits);
          runA1Case(nnEval, logger, label, "bayesA1" + nnSeed + Global::intToString(size) + "m" + Global::int64ToString(maxVisits),
                    maxVisits, board, hist, pla);
        }
      }
    }
    delete nnEval;
  }

  //-------------------------------------------------------------------------
  //A2: terminal-dominance minimax agreement on decided 5x5 endgames.
  //-------------------------------------------------------------------------
  {
    NNEvaluator* nnEval = startNNEval(
      modelFile, logger, "bayesGateSeedA2", NNPos::MAX_BOARD_LEN, NNPos::MAX_BOARD_LEN,
      0, true, false, false, true, false);

    //Black owns everything (eyes at corners and center); komi 0.5 -> B+24.5.
    const string blackAll = R"%%(
.xxx.
xxxxx
xx.xx
xxxxx
.xxx.
)%%";
    //Color-swapped: white owns everything; komi 0.5 -> W+25.5.
    const string whiteAll = R"%%(
.ooo.
ooooo
oo.oo
ooooo
.ooo.
)%%";
    //Black owns everything, only 3 eyes left; B+24.5 at komi 0.5. (A denser
    //variant: contested-capture constructions do NOT qualify here because the
    //stub search cannot exhaust the reopened tree at 3000 visits.)
    const string blackDense = R"%%(
.xxxx
xxxxx
xx.xx
xxxxx
xxxx.
)%%";
    //Color swap of blackDense; W+25.5 at komi 0.5.
    const string whiteDense = R"%%(
.oooo
ooooo
oo.oo
ooooo
oooo.
)%%";

    A2Tally tally;
    runA2Case(nnEval, logger, "blackAll-whiteToMove", blackAll, P_WHITE, 0.5f, tally);
    runA2Case(nnEval, logger, "blackAll-blackToMove", blackAll, P_BLACK, 0.5f, tally);
    runA2Case(nnEval, logger, "whiteAll-blackToMove", whiteAll, P_BLACK, 0.5f, tally);
    runA2Case(nnEval, logger, "whiteAll-whiteToMove", whiteAll, P_WHITE, 0.5f, tally);
    runA2Case(nnEval, logger, "blackDense-whiteToMove", blackDense, P_WHITE, 0.5f, tally);
    runA2Case(nnEval, logger, "whiteDense-blackToMove", whiteDense, P_BLACK, 0.5f, tally);

    testAssert(tally.numQualifying >= 5);
    testAssert(tally.numQualifyingBlackWon >= 1);
    testAssert(tally.numQualifyingWhiteWon >= 1);
    testAssert(tally.numQualifyingBlackToMove >= 1);
    testAssert(tally.numQualifyingWhiteToMove >= 1);

    delete nnEval;
  }

  ScoreValue::freeTables();
  cout << "Done bayes search tests" << endl;
}
