# receomendation_with_python
[movies.csv](https://github.com/user-attachments/files/19715518/movies.csv)
## The above file is from keagle
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import ipywidgets as widgets
from IPython.display import display
movies =  pd.read_csv('movies.csv')
movies

def clean_name(name):
  return re.sub("[^a-zA-Z0-9 ]", "", name)## its going to clean any character thats not a space 
  movies["clean_name"] = movies["name"].apply(clean_name)
  movies
  

vectorizer = TfidfVectorizer(ngram_range=(1,2))
tfidf = vectorizer.fit_transform(movies["clean_name"])


def search(name):
  name = clean_name(name)
  querry_vec = vectorizer.transform([name])
  similarities = cosine_similarity(querry_vec,tfidf).flatten()
  indices = np.argpartition(similarities,-5)[-5:]
  results = movies.iloc[indices].iloc[::-1]
  return results 

  import ipywidgets as widgets
from IPython.display import display
movie_input = widgets.Text(
    value="Dressed to kill",
    description="Movie name:",
    disabled=False
)
movie_list = widgets.Output()

def on_type(data):
  with movie_list:
    movie_list.clear_output()
    name = data["new"]
    if len(name) > 2:
      display(search(name))

movie_input.observe( on_type, names = 'value')

display(movie_input, movie_list)




