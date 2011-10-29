#include "lm/model.hh"
#include "util/tokenize_piece.hh"

#define BOOST_TEST_MODULE ModelTest
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <algorithm>
#include <vector>

#include <math.h>

namespace lm {
namespace ngram {
namespace {

void GatherLefts(const RestProbingModel &model, std::vector<float> &out) {
  out.clear();
  RestProbingModel::State ignored;
  for (WordIndex i = 0; i < 34; ++i) {
    out.push_back(model.FullScore(model.NullContextState(), i, ignored).left_rest);
  }
}

void Check(const RestProbingModel &model, const std::vector<float> &lefts, StringPiece str) {
  std::vector<WordIndex> indices;
  for (util::TokenIter<util::AnyCharacter> i(str, util::AnyCharacter(" ")); i; ++i) {
    indices.push_back(model.GetVocabulary().Index(*i));
  }
  std::reverse(indices.begin(), indices.end());
  RestProbingModel::State cont;
  FullScoreReturn orig(model.FullScoreForgotState(&*indices.begin() + 1, &*indices.end(), indices.front(), cont));
  double sum = 0.0;
  RestProbingModel::State ignored;
  for (WordIndex i = 0; i < 34; ++i) {
    float prob = model.FullScore(cont, i, ignored).prob;
    sum += pow(10.0, prob) * (prob - lefts[i]);
  }
  BOOST_CHECK_CLOSE(sum, orig.right_rest, 0.0001);
  std::cerr << "Checked " << str << " at " << sum << std::endl;
}

struct Checker {
  Checker() : model("5.arpa") {
    GatherLefts(model, lefts);
  }

  Checker &operator,(StringPiece str) {
    Check(model, lefts, str);
    return *this;
  }

  RestProbingModel model;
  std::vector<float> lefts;
};

BOOST_AUTO_TEST_CASE(Rest) {
  Checker checker;
  checker,
    ",",
    ".",
    "<s> little",
    "also call",
    "immediate concerns",
    "on a little more loin",
    "on a little more also",
    "foo </s>",
    "</s>";
}

} // namespace
} // namespace ngram
} // namespace lm
