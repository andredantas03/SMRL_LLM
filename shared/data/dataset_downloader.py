from datasets import load_dataset

imdb = load_dataset("stanfordnlp/imdb")
ag = load_dataset("fancyzhx/ag_news")

# Salvar em disco (Arrow + opcional CSV/JSON)
imdb.save_to_disk("raw/imdb")
ag.save_to_disk("raw/ag_news")

