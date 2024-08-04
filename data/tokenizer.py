import sentencepiece as spm
import os 

class TextTransform:
  ''' Map characters to integers and vice versa '''
  def __init__(self):
    self.char_map = {}  
    for i, char in enumerate(range(65, 91)):
      self.char_map[chr(char)] = i
    self.char_map["'"] = 26
    self.char_map[' '] = 27
    self.index_map = {} 
    for char, i in self.char_map.items():
      self.index_map[i] = char

  def text_to_int(self, text):
      ''' Map text string to an integer sequence '''
      int_sequence = []
      for c in text:
        ch = self.char_map[c]
        int_sequence.append(ch)
      return int_sequence

  def int_to_text(self, labels):
      ''' Map integer sequence to text string '''
      string = []
      for i in labels:
          if i == 28: # blank char
            continue
          else:
            string.append(self.index_map[i])
      return ''.join(string)

class BPETextTransform:
    ''' Use BPE to build a vocabulary and map text and numbers to each other '''
    def __init__(self):
        file_path = 'spm/librispeech/1000_bpe.models'
        vocab_size = 1000
        if not os.path.exists(file_path):
            spm.SentencePieceTrainer.train(
                input = 'data/train-clean-100.text',
                model_prefix = f'spm/librispeech/{vocab_size}_bpe',
                vocab_size = vocab_size,
                character_coverage = 1.0,
                model_type = 'bpe'
            )
        self.sp = spm.SentencePieceProcessor(model_file=file_path)
    
    def text_to_int(self, text):
        return self.sp.encode(text, out_type=int)
    
    def int_to_text(self, labels):
        return self.sp.decode(labels)

if __name__ == '__main__':
    bpe = TextTransform()
    Text = 'PEARL SAW AND GAZED INTENTLY BUT NEVER SOUGHT TO MAKE ACQUAINTANCE'
    print(Text)
    int_sequence = bpe.text_to_int(Text)
    Text = bpe.int_to_text(int_sequence)
    print(int_sequence)
    print(Text)
    
