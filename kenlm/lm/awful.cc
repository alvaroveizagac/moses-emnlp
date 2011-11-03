#include "lm/awful.hh"
#include "lm/read_arpa.hh"

#include "util/file_piece.hh"

namespace lm {
namespace ngram {

template <class M> AwfulGlobal<M>::AwfulGlobal() {
  models_[0] = NULL;
  models_[1] = NULL;
  models_[2] = NULL;
}

template <class M> void AwfulGlobal<M>::Load() {
  if (loading_) return;
  util::FilePiece uni("1");
  std::vector<uint64_t> number;
  ReadARPACounts(uni, number);
  assert(number.size() == 1);
  unigram_.resize(number[0]);
  typedef typename M::Vocabulary Vocab;
  Vocab vocab;
  std::vector<char> vocab_backing(Vocab::Size(number[0] + 1, Config()));
  vocab.SetupMemory(&vocab_backing.front(), Vocab::Size(number[0] + 1, Config()), number[0] + 1, Config()); 
  PositiveProbWarn warn;
  Read1Grams(uni, (size_t)number[0], vocab, &*unigram_.begin(), warn);

  loading_ = true;

  models_[0] = new M("2");
  models_[1] = new M("3");
  models_[2] = new M("4");
  loading_ = false;
}

template <class M> AwfulGlobal<M>::~AwfulGlobal() {
  delete models_[0];
  delete models_[1];
  delete models_[2];
}

template <class M> void AwfulGlobal<M>::ApplyUnigram(Rest *weights) {
  for (size_t i = 0; i < unigram_.size(); ++i) {
    weights[i].rest = unigram_[i].prob;
    std::cout << "1 " << -fabsf(weights[i].prob) << ' ' << weights[i].rest << '\n';
  }
  unigram_.clear();
}

AwfulGlobal<TrieModel> trie_awful;
AwfulGlobal<ProbingModel> probing_awful;

template class AwfulGlobal<TrieModel>;
template class AwfulGlobal<ProbingModel>;

} // namespace ngram
} // namespace lm
