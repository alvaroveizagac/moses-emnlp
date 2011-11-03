#include "lm/awful.hh"
#include "lm/read_arpa.hh"

#include "util/file_piece.hh"

namespace lm {
namespace ngram {

AwfulGlobal::AwfulGlobal() {
  util::FilePiece uni("1");
  std::vector<uint64_t> number;
  ReadARPACounts(uni, number);
  assert(number.size() == 1);
  unigram_.resize(number[0]);
  SortedVocabulary vocab;
  std::vector<char> vocab_backing(SortedVocabulary::Size(number[0] + 1, Config()));
  vocab.SetupMemory(&vocab_backing.front(), SortedVocabulary::Size(number[0] + 1, Config()), number[0] + 1, Config()); 
  PositiveProbWarn warn;
  Read1Grams(uni, (size_t)number[0], vocab, &*unigram_.begin(), warn);

  models_[0] = new TrieModel("2");
  models_[1] = new TrieModel("3");
  models_[2] = new TrieModel("4");
}

AwfulGlobal::~AwfulGlobal() {
  delete models_[0];
  delete models_[1];
  delete models_[2];
}

void AwfulGlobal::ApplyUnigram(Rest *weights) {
  for (size_t i = 0; i < unigram_.size(); ++i) {
    weights[i].rest = unigram_[i].prob;
    std::cout << "1 " << -fabsf(weights[i].prob) << ' ' << weights[i].rest << '\n';
  }
  unigram_.clear();
}

AwfulGlobal awful;

} // namespace ngram
} // namespace lm
