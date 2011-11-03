#ifndef LM_AWFUL__
#define LM_AWFUL__

// TODO: remove hack
#include "lm/model.hh"
#include "lm/weights.hh"

#include <vector>
#include <math.h>

namespace lm {
namespace ngram {

template <class M> class AwfulGlobal {
  public:
    AwfulGlobal();

    ~AwfulGlobal();

    void Load();

    float GetRest(const WordIndex *vocab_ids, unsigned int n) {
      if (loading_) return NAN;
      float ret;
      if (n == 1) {
        ret = unigram_[vocab_ids[0]].prob;
      } else {
        State ignored;
        ret = models_[n - 2]->FullScoreForgotState(vocab_ids + 1, vocab_ids + n, vocab_ids[0], ignored).prob;
      }
      assert(!isnan(ret));
      return ret;
    }

    void ApplyUnigram(Rest *weights);
    void ApplyUnigram(ProbBackoff * /*weights */) {}

  private:
    std::vector<ProbBackoff> unigram_;
    const M *models_[3];

    bool loading_;
};

// Prevent hashed from using trie's models
extern AwfulGlobal<TrieModel> trie_awful;
extern AwfulGlobal<ProbingModel> probing_awful;

} // namespace ngram
} // namespace lm
#endif // LM_AWFUL__
