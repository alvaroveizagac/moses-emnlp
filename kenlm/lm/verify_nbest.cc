#include "lm/model.hh"
#include "util/file_piece.hh"

#include <iostream>

template <class Model> void Verify(const Model &m) {
  util::FilePiece f(0, "stdin");
  try { while (true) {
    while (f.ReadDelimited() != "|||") {}
    typename Model::State state = m.BeginSentenceState(), state2;
    float score = 0.0;
    StringPiece word;
    while ("|||lm:" != (word = f.ReadDelimited())) {
      score += m.FullScore(state, m.GetVocabulary().Index(word), state2).prob;
      state = state2; 
    }
    score += m.FullScore(state, m.GetVocabulary().EndSentence(), state2).prob;
    float file_says = f.ReadFloat();
    UTIL_THROW_IF(fabs(score - file_says) > 0.001, util::Exception, "Score disagreement: " << score << " != " << file_says);
    f.ReadLine();
  } } catch (const util::EndOfFileException &e) {}
}

int main(int argc, char *argv[]) {
  if (argc != 2) {
    std::cerr << "Specify model file." << std::endl;
    return 1;
  }

  lm::ngram::Model m(argv[1]);
  Verify(m);
}
