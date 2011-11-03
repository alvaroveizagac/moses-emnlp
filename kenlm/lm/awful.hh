#ifndef LM_AWFUL__
#define LM_AWFUL__

// TODO: remove hack
#include "lm/model.hh"
#include "lm/weights.hh"

#include <vector>

namespace lm {
namespace ngram {

template <class M> class AwfulGlobal {
  public:
    AwfulGlobal();

    ~AwfulGlobal();

    void Load();

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
    const M *models_[3];
};

// Prevent hashed from using trie's models
extern AwfulGlobal<TrieModel> trie_awful;
extern AwfulGlobal<ProbingModel> probing_awful;

} // namespace ngram
} // namespace lm
#endif // LM_AWFUL__
