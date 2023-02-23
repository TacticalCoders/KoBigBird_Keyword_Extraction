import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class KeywordExtractor:
    def __init__(self, model_path, full_attention=False):
        if full_attention:
            self.model = AutoModel.from_pretrained(model_path, attention_type="original_full")
        else:
            self.model = AutoModel.from_pretrained(model_path)

        self.tokenizer = AutoTokenizer.from_pretrained("monologg/kobigbird-bert-base")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()

    def extract_main_keywords(self, nouns, title, text, use_freq_w=False, counts=None):

        with torch.no_grad():
            tokenized_nouns = {}
            for keyword in nouns:
                tokenized_nouns[keyword] = self.tokenizer(keyword, max_length=32, padding='max_length', return_tensors='pt')

            keyword_vectors = {}
            for keyword, tokenized in tokenized_nouns.items():
                output = self.model(**tokenized.to(self.device))[0][:, 0, :]
                keyword_vectors[keyword] = output

            tokenized_title = self.tokenizer(title, max_length=128, padding="max_length", return_tensors='pt')
            tokenized_text = self.tokenizer(text, max_length=4096, padding="max_length", return_tensors='pt')

            title_vec = self.model(**tokenized_title.to(self.device))[0][:, 0, :]
            body_text_vec = self.model(**tokenized_text.to(self.device))[0][:, 0, :]

            if use_freq_w:
                total_words_counts = 0
                for word, count in counts.items():
                    total_words_counts += count
                print(total_words_counts)

                freq_w = {}

                for keyword in nouns:
                    if counts[keyword] <= 1:
                        freq_w[keyword] = 0
                        continue
                    freq_w[keyword] = counts[keyword] / total_words_counts

                print(freq_w)
                for keyword, vec in keyword_vectors.items():
                    body_text_vec = ((freq_w[keyword] * vec) + body_text_vec)

                body_text_vec = body_text_vec / total_words_counts

            cos = nn.CosineSimilarity(dim=1, eps=1e-6)
            similarity = {}

            for keyword, vector in keyword_vectors.items():
                sim = cos(vector, title_vec * (0.2) + body_text_vec * (0.8))
                similarity[keyword] = sim

            sorted_dict = sorted(similarity.items(), key=lambda item: item[1], reverse=True)

            # print(keyword_vectors)
            print(f"제목과 본문의 유사도 {cos(title_vec, body_text_vec)}\n")
            # print(title + '\n')
            # print(text + '\n')
            # print(sorted_dict, '\n')
            # print(nouns)

            return sorted_dict
