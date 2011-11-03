#ifndef LM_AWFUL__
#define LM_AWFUL__

// TODO: remove hack
#include "lm/model.hh"
#include "lm/weights.hh"

#include <vector>

namespace lm {
namespace ngram {

class AwfulGlobal {
  public:
    AwfulGlobal();

    ~AwfulGlobal();

    float GetRest(const WordIndex *vocab_ids, unsigned int n) {
      if (n == 1) {
        return unigram_[vocab_ids[0]].prob;
      } else {
        State ignored;
        return models_[n - 2]->FullScoreForgotState(vocab_ids + 1, vocab_ids + n, vocab_ids[0], ignored).prob;
      }
    }

    void ApplyUnigram(Rest *weights);
    void ApplyUnigram(ProbBackoff * /*weights */) {}

  private:
    std::vector<ProbBackoff> unigram_;
    const ProbingModel *models_[3];
};

extern AwfulGlobal awful;

} // namespace ngram
} // namespace lm
#endif // LM_AWFUL__
