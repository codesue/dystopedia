import gradio as gr
import lemminflect
import spacy
import wikipedia
from transformers import pipeline

nlp = spacy.load("en_core_web_lg")
sentiment_analyzer = pipeline(
  "sentiment-analysis",
  model="distilbert-base-uncased-finetuned-sst-2-english",
  revision="af0f99b"
)

def is_positive(text):
  return sentiment_analyzer(text)[0]["label"] == "POSITIVE"

def make_past_tense(token):
  if token.tag_ in ("VBP", "VBZ"):
    return f'{token._.inflect("VBD")} '
  return token.text_with_ws

def make_dystopian(term, text):
  doc = nlp(text)
  if is_positive(term):
    return "".join([make_past_tense(token) for token in doc])
  return doc.text

def get_dystopian_summary(term):
  if term == "":
    return term

  try:
    results = wikipedia.search(term, results=1)
  except wikipedia.exceptions.DisambiguationError as e:
    raise gr.Error(e.error)

  if len(results) == 0:
    raise gr.Error(
      f'Could not find an article on the term "{term}". '
      'Try searching for a different topic.'
    )

  summary = wikipedia.summary(
    results[0],
    sentences=1,
    auto_suggest=False,
    redirect=True
  )
  return make_dystopian(term, summary)

def launch_demo(**kwargs):
  title = "Dystopedia"
  description = (
    "Make any Wikipedia topic dystopian. Inspired by "
    "[this Tweet](https://twitter.com/lbcyber/status/1115015586243862528). "
    "Dystopedia uses [DistilBERT base uncased finetuned SST-2](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english) "
    "for sentiment analysis and is subject to its limitations and biases."
  )
  examples = ["joy", "hope", "peace", "Earth", "water", "food"]
  gr.Interface(
    fn=get_dystopian_summary,
    inputs=gr.Textbox(label="term", placeholder="Enter a term...", max_lines=1),
    outputs=gr.Textbox(label="description"),
    title=title,
    description=description,
    examples=examples,
    cache_examples=True,
    allow_flagging="never",
  ).launch(**kwargs)

launch_demo(show_error=True)
