<div class="alert" style="background-color:#fff; color:white; padding:0px 10px; border-radius:5px;"><h1 style='margin:15px 15px; color:#5d3a8e; font-size:40px'> Topic Modeling For NBC News Digital Social Media Content</h1>
</div>

This is hacky way to try to understand the distribution of top-performing content on Facebook over the past 12 months. We'll be using Latent Dirichlet Allocation(LDA) for topic modeling along with the Python Gensim package. To make this worthwhile, we'll start cleaning up the text, breaking the posts into bigrams, and then determining the optimal number of topics.

<div class="alert alert-info" style="background-color:#5d3a8e; color:white; padding:0px 10px; border-radius:5px;"><h2 style='margin:10px 5px'> Download nltk stopwords and spacy model</h2>
</div>

Stopwords from NLTK and spacy’s en model for text pre-processing. Spacy model for lemmatization.


```python
# Run in python console
import nltk; nltk.download('stopwords')

# Run in terminal or command prompt
#python3 -m spacy download en
```

    [nltk_data] Downloading package stopwords to
    [nltk_data]     /Users/a206679878/nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!





    True



<div class="alert alert-info" style="background-color:#5d3a8e; color:white; padding:0px 10px; border-radius:5px;"><h2 style='margin:10px 5px'> Import Packages</h2>
</div>

The core packages are `re`, `gensim`, `spacy` and `pyLDAvis`. `matplotlib`,`numpy` and `pandas` for data handling and visualization.


```python
import sys
!{sys.executable} -m pip install spacy
```

    Requirement already satisfied: spacy in /Users/a206679878/opt/anaconda3/lib/python3.8/site-packages (2.3.5)
    Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /Users/a206679878/opt/anaconda3/lib/python3.8/site-packages (from spacy) (2.0.5)
    Requirement already satisfied: plac<1.2.0,>=0.9.6 in /Users/a206679878/opt/anaconda3/lib/python3.8/site-packages (from spacy) (1.1.3)
    Requirement already satisfied: blis<0.8.0,>=0.4.0 in /Users/a206679878/opt/anaconda3/lib/python3.8/site-packages (from spacy) (0.7.4)
    Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /Users/a206679878/opt/anaconda3/lib/python3.8/site-packages (from spacy) (1.0.5)
    Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /Users/a206679878/opt/anaconda3/lib/python3.8/site-packages (from spacy) (3.0.5)
    Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /Users/a206679878/opt/anaconda3/lib/python3.8/site-packages (from spacy) (1.0.5)
    Requirement already satisfied: requests<3.0.0,>=2.13.0 in /Users/a206679878/opt/anaconda3/lib/python3.8/site-packages (from spacy) (2.24.0)
    Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /Users/a206679878/opt/anaconda3/lib/python3.8/site-packages (from spacy) (1.0.0)
    Requirement already satisfied: setuptools in /Users/a206679878/opt/anaconda3/lib/python3.8/site-packages (from spacy) (50.3.1.post20201107)
    Requirement already satisfied: numpy>=1.15.0 in /Users/a206679878/opt/anaconda3/lib/python3.8/site-packages (from spacy) (1.21.0)
    Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /Users/a206679878/opt/anaconda3/lib/python3.8/site-packages (from spacy) (4.50.2)
    Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /Users/a206679878/opt/anaconda3/lib/python3.8/site-packages (from spacy) (0.8.2)
    Requirement already satisfied: thinc<7.5.0,>=7.4.1 in /Users/a206679878/opt/anaconda3/lib/python3.8/site-packages (from spacy) (7.4.5)
    Requirement already satisfied: chardet<4,>=3.0.2 in /Users/a206679878/opt/anaconda3/lib/python3.8/site-packages (from requests<3.0.0,>=2.13.0->spacy) (3.0.4)
    Requirement already satisfied: idna<3,>=2.5 in /Users/a206679878/opt/anaconda3/lib/python3.8/site-packages (from requests<3.0.0,>=2.13.0->spacy) (2.10)
    Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /Users/a206679878/opt/anaconda3/lib/python3.8/site-packages (from requests<3.0.0,>=2.13.0->spacy) (1.25.11)
    Requirement already satisfied: certifi>=2017.4.17 in /Users/a206679878/opt/anaconda3/lib/python3.8/site-packages (from requests<3.0.0,>=2.13.0->spacy) (2020.6.20)



```python
import re
import numpy as np
import pandas as pd
from pprint import pprint

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# spacy for lemmatization
import spacy

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
import matplotlib.pyplot as plt
%matplotlib inline

# Enable logging for gensim 
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
```

    /Users/a206679878/opt/anaconda3/lib/python3.8/site-packages/sklearn/decomposition/_lda.py:28: DeprecationWarning: `np.float` is a deprecated alias for the builtin `float`. To silence this warning, use `float` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.float64` here.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      EPS = np.finfo(np.float).eps


<div class="alert alert-info" style="background-color:#5d3a8e; color:white; padding:0px 10px; border-radius:5px;"><h2 style='margin:10px 5px'> Why LDA?</h2>
</div>

LDA’s approach to topic modeling is it considers each document as a collection of topics in a certain proportion. And each topic as a collection of keywords, again, in a certain proportion.

A topic is a collection of dominant keywords that are typical representatives. 

<div class="alert alert-info" style="background-color:#5d3a8e; color:white; padding:0px 10px; border-radius:5px;"><h2 style='margin:10px 5px'> Prepare Stopwords</h2>
</div>


Importing list in `stop_words`.


```python
# NLTK Stop words
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['www', 'http://'])
```

<div class="alert alert-info" style="background-color:#5d3a8e; color:white; padding:0px 10px; border-radius:5px;"><h2 style='margin:10px 5px'> Import Facebook Data</h2>
</div>

We'll be using public posts from top publishers for a 12-month period. Using CrowdTangle's historic reporting, we'll download the full post data and upload a local copy for our analysis.

Import using `pandas.read_csv`.


```python
# Import Dataset
df = pd.read_csv('/Users/a206679878/Documents/Untitled Folder/postings.csv')
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Page Name</th>
      <th>User Name</th>
      <th>Facebook Id</th>
      <th>Page Category</th>
      <th>Page Admin Top Country</th>
      <th>Page Description</th>
      <th>Page Created</th>
      <th>Likes at Posting</th>
      <th>Followers at Posting</th>
      <th>Post Created</th>
      <th>...</th>
      <th>Message</th>
      <th>Link</th>
      <th>Final Link</th>
      <th>Image Text</th>
      <th>Link Text</th>
      <th>Description</th>
      <th>Sponsor Id</th>
      <th>Sponsor Name</th>
      <th>Sponsor Category</th>
      <th>Overperforming Score (weighted  —  Likes 1x Shares 3x Comments 2x )</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>The Hill</td>
      <td>TheHill</td>
      <td>7533944086</td>
      <td>NEWS_SITE</td>
      <td>US</td>
      <td>The Hill is the premier source for policy and ...</td>
      <td>2008-01-10 18:20:58</td>
      <td>1464754</td>
      <td>1562141</td>
      <td>2021-07-06 16:45:20 EDT</td>
      <td>...</td>
      <td>Former President Trump considered issuing a pa...</td>
      <td>http://hill.cm/AY9Rs6r</td>
      <td>https://thehill.com/homenews/media/561735-trum...</td>
      <td>NaN</td>
      <td>Trump discussed pardoning Ghislaine Maxwell: book</td>
      <td>Wolff has written two other books about Trump,...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.57</td>
    </tr>
    <tr>
      <th>1</th>
      <td>MSNBC</td>
      <td>msnbc</td>
      <td>273864989376427</td>
      <td>MEDIA_NEWS_COMPANY</td>
      <td>US</td>
      <td>The destination for in-depth analysis of daily...</td>
      <td>2012-05-14 16:26:44</td>
      <td>2479926</td>
      <td>2586695</td>
      <td>2021-07-06 16:43:11 EDT</td>
      <td>...</td>
      <td>The Vice President’s life was threatened. Cops...</td>
      <td>https://on.msnbc.com/3jM4ivl</td>
      <td>https://www.msnbc.com/ali-velshi/watch/velshi-...</td>
      <td>NaN</td>
      <td>Velshi: Covering up the insurrection is the Bi...</td>
      <td>We are witnessing an active cover-up on Capito...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4.07</td>
    </tr>
    <tr>
      <th>2</th>
      <td>MSNBC</td>
      <td>msnbc</td>
      <td>273864989376427</td>
      <td>MEDIA_NEWS_COMPANY</td>
      <td>US</td>
      <td>The destination for in-depth analysis of daily...</td>
      <td>2012-05-14 16:26:44</td>
      <td>2479926</td>
      <td>2586695</td>
      <td>2021-07-06 16:38:31 EDT</td>
      <td>...</td>
      <td>LATEST: Tropical Storm Elsa as of 3:30pm ET. H...</td>
      <td>https://www.facebook.com/msnbc/videos/30595231...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NBC News Meteorologist Bill Karins on Tropical...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-6.83</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NBC News</td>
      <td>NBCNews</td>
      <td>155869377766434</td>
      <td>BROADCASTING_MEDIA_PRODUCTION</td>
      <td>US</td>
      <td>A leading source of global news and informatio...</td>
      <td>2010-09-30 19:35:28</td>
      <td>10166610</td>
      <td>10717941</td>
      <td>2021-07-06 16:33:05 EDT</td>
      <td>...</td>
      <td>LATEST: Elsa could be a hurricane when it hits...</td>
      <td>https://nbcnews.to/2V38lcn</td>
      <td>https://www.nbcnews.com/news/weather/elsa-fore...</td>
      <td>NaN</td>
      <td>Elsa intensifies as it turns toward Florida co...</td>
      <td>Hurricane-force winds, tornadoes and up to 8 i...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.92</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Washington Post</td>
      <td>washingtonpost</td>
      <td>6250307292</td>
      <td>BROADCASTING_MEDIA_PRODUCTION</td>
      <td>US</td>
      <td>Our award-winning journalists have covered Was...</td>
      <td>2007-11-07 18:26:05</td>
      <td>6622347</td>
      <td>7005382</td>
      <td>2021-07-06 16:30:28 EDT</td>
      <td>...</td>
      <td>“I’m the king of the tax code,” said Trump, wh...</td>
      <td>https://www.washingtonpost.com/politics/2021/0...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Analysis | How Trump’s claims to being ‘the ki...</td>
      <td>"I’m the king of the tax code," said Trump, wh...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3.24</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 40 columns</p>
</div>




```python
print(df.shape)
```

    (185647, 40)


For our text analysis, we'll need the "Description" column formatted as a string. Once that text is a string, we can start to break it up into its components.


```python
df['Link Text']= df['Link Text'].map(str)
```


```python
df.rename(columns = {"Link Text": "text"}, inplace=True)
print(df.columns)
```

    Index(['Page Name', 'User Name', 'Facebook Id', 'Page Category',
           'Page Admin Top Country', 'Page Description', 'Page Created',
           'Likes at Posting', 'Followers at Posting', 'Post Created',
           'Post Created Date', 'Post Created Time', 'Type', 'Total Interactions',
           'Likes', 'Comments', 'Shares', 'Love', 'Wow', 'Haha', 'Sad', 'Angry',
           'Care', 'Video Share Status', 'Is Video Owner?', 'Post Views',
           'Total Views', 'Total Views For All Crossposts', 'Video Length', 'URL',
           'Message', 'Link', 'Final Link', 'Image Text', 'text', 'Description',
           'Sponsor Id', 'Sponsor Name', 'Sponsor Category',
           'Overperforming Score (weighted  —  Likes 1x Shares 3x Comments 2x )'],
          dtype='object')



```python
datatypes = df.dtypes
  
# Print the data types
# of each column
datatypes
```




    Page Name                                                               object
    User Name                                                               object
    Facebook Id                                                              int64
    Page Category                                                           object
    Page Admin Top Country                                                  object
    Page Description                                                        object
    Page Created                                                            object
    Likes at Posting                                                         int64
    Followers at Posting                                                     int64
    Post Created                                                            object
    Post Created Date                                                       object
    Post Created Time                                                       object
    Type                                                                    object
    Total Interactions                                                      object
    Likes                                                                    int64
    Comments                                                                 int64
    Shares                                                                   int64
    Love                                                                     int64
    Wow                                                                      int64
    Haha                                                                     int64
    Sad                                                                      int64
    Angry                                                                    int64
    Care                                                                     int64
    Video Share Status                                                      object
    Is Video Owner?                                                         object
    Post Views                                                               int64
    Total Views                                                              int64
    Total Views For All Crossposts                                           int64
    Video Length                                                            object
    URL                                                                     object
    Message                                                                 object
    Link                                                                    object
    Final Link                                                              object
    Image Text                                                              object
    text                                                                    object
    Description                                                             object
    Sponsor Id                                                             float64
    Sponsor Name                                                            object
    Sponsor Category                                                        object
    Overperforming Score (weighted  —  Likes 1x Shares 3x Comments 2x )    float64
    dtype: object



<div class="alert alert-info" style="background-color:#5d3a8e; color:white; padding:0px 10px; border-radius:5px;"><h2 style='margin:10px 5px'> The Best of the Best</h2>
</div>

Now, we only want the best posts, so we will filter our results for posts only above a threshold. We'll go with all posts above 3000 likes, a typical engagement rate.


```python
df_2 = df[df["Likes"] > 3000.0]

df_2
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Page Name</th>
      <th>User Name</th>
      <th>Facebook Id</th>
      <th>Page Category</th>
      <th>Page Admin Top Country</th>
      <th>Page Description</th>
      <th>Page Created</th>
      <th>Likes at Posting</th>
      <th>Followers at Posting</th>
      <th>Post Created</th>
      <th>...</th>
      <th>Message</th>
      <th>Link</th>
      <th>Final Link</th>
      <th>Image Text</th>
      <th>text</th>
      <th>Description</th>
      <th>Sponsor Id</th>
      <th>Sponsor Name</th>
      <th>Sponsor Category</th>
      <th>Overperforming Score (weighted  —  Likes 1x Shares 3x Comments 2x )</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>64</th>
      <td>The Hill</td>
      <td>TheHill</td>
      <td>7533944086</td>
      <td>NEWS_SITE</td>
      <td>US</td>
      <td>The Hill is the premier source for policy and ...</td>
      <td>2008-01-10 18:20:58</td>
      <td>1464754</td>
      <td>1562141</td>
      <td>2021-07-06 14:00:10 EDT</td>
      <td>...</td>
      <td>A New Jersey man was seen in a now-viral video...</td>
      <td>http://hill.cm/hJGUF3Y</td>
      <td>https://thehill.com/blogs/blog-briefing-room/n...</td>
      <td>NaN</td>
      <td>WATCH: 100 protesters show up at home of man w...</td>
      <td>Mathews, during a phone interview with the Inq...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>16.19</td>
    </tr>
    <tr>
      <th>77</th>
      <td>The Hill</td>
      <td>TheHill</td>
      <td>7533944086</td>
      <td>NEWS_SITE</td>
      <td>US</td>
      <td>The Hill is the premier source for policy and ...</td>
      <td>2008-01-10 18:20:58</td>
      <td>1464754</td>
      <td>1562141</td>
      <td>2021-07-06 13:30:05 EDT</td>
      <td>...</td>
      <td>Asked in a new interview if he suspected that ...</td>
      <td>http://hill.cm/xT55zWD</td>
      <td>https://thehill.com/homenews/house/561672-kinz...</td>
      <td>NaN</td>
      <td>Kinzinger says he suspects some lawmakers knew...</td>
      <td>"...the whole reason I brought my gun and kept...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>12.75</td>
    </tr>
    <tr>
      <th>224</th>
      <td>CNN</td>
      <td>cnn</td>
      <td>5550296508</td>
      <td>MEDIA_NEWS_COMPANY</td>
      <td>US</td>
      <td>Instant breaking news alerts and the most talk...</td>
      <td>2007-11-07 22:14:27</td>
      <td>34564937</td>
      <td>38372380</td>
      <td>2021-07-06 07:11:45 EDT</td>
      <td>...</td>
      <td>A Republican official from a key Arizona count...</td>
      <td>https://cnn.it/3AvEcTt</td>
      <td>https://www.cnn.com/2021/07/05/politics/clint-...</td>
      <td>NaN</td>
      <td>Maricopa County official on rejecting Trump al...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4.49</td>
    </tr>
    <tr>
      <th>232</th>
      <td>Washington Post</td>
      <td>washingtonpost</td>
      <td>6250307292</td>
      <td>BROADCASTING_MEDIA_PRODUCTION</td>
      <td>US</td>
      <td>Our award-winning journalists have covered Was...</td>
      <td>2007-11-07 18:26:05</td>
      <td>6622347</td>
      <td>7005382</td>
      <td>2021-07-06 06:40:46 EDT</td>
      <td>...</td>
      <td>A man police say yelled racist slurs in front ...</td>
      <td>https://www.washingtonpost.com/nation/2021/07/...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>A man who went on a racist rant gave out his a...</td>
      <td>The New Jersey man was later arrested as prote...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>13.15</td>
    </tr>
    <tr>
      <th>292</th>
      <td>CNN</td>
      <td>cnn</td>
      <td>5550296508</td>
      <td>MEDIA_NEWS_COMPANY</td>
      <td>US</td>
      <td>Instant breaking news alerts and the most talk...</td>
      <td>2007-11-07 22:14:27</td>
      <td>34564937</td>
      <td>38372380</td>
      <td>2021-07-06 01:31:07 EDT</td>
      <td>...</td>
      <td>San Francisco's lavish Millennium Tower opened...</td>
      <td>https://cnn.it/2V65Vtv</td>
      <td>https://www.cnn.com/2021/07/05/us/san-francisc...</td>
      <td>NaN</td>
      <td>Surfside catastrophe raises concerns about San...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>5.61</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>185625</th>
      <td>Washington Post</td>
      <td>washingtonpost</td>
      <td>6250307292</td>
      <td>BROADCASTING_MEDIA_PRODUCTION</td>
      <td>US</td>
      <td>Our award-winning journalists have covered Was...</td>
      <td>2007-11-07 18:26:05</td>
      <td>6465204</td>
      <td>6908872</td>
      <td>2020-07-06 17:40:54 EDT</td>
      <td>...</td>
      <td>The book will be published on July 14 because ...</td>
      <td>https://www.washingtonpost.com/politics/tell-a...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Tell-all book by President Trump’s niece to be...</td>
      <td>Mary Trump’s book said to show “how Donald acq...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>10.70</td>
    </tr>
    <tr>
      <th>185629</th>
      <td>MSNBC</td>
      <td>msnbc</td>
      <td>273864989376427</td>
      <td>MEDIA_NEWS_COMPANY</td>
      <td>US</td>
      <td>The destination for in-depth analysis of daily...</td>
      <td>2012-05-14 16:26:44</td>
      <td>2374006</td>
      <td>2504567</td>
      <td>2020-07-06 17:31:14 EDT</td>
      <td>...</td>
      <td>Sen. Grassley will not attend the Republican N...</td>
      <td>https://on.msnbc.com/3e2B84Q</td>
      <td>https://www.nbcnews.com/politics/politics-news...</td>
      <td>NaN</td>
      <td>Chuck Grassley to skip GOP convention over cor...</td>
      <td>This would mark the first time in 40 years Gra...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.36</td>
    </tr>
    <tr>
      <th>185636</th>
      <td>Washington Post</td>
      <td>washingtonpost</td>
      <td>6250307292</td>
      <td>BROADCASTING_MEDIA_PRODUCTION</td>
      <td>US</td>
      <td>Our award-winning journalists have covered Was...</td>
      <td>2007-11-07 18:26:05</td>
      <td>6465204</td>
      <td>6908872</td>
      <td>2020-07-06 17:20:19 EDT</td>
      <td>...</td>
      <td>Disney and Colin Kaepernick announced a new pr...</td>
      <td>https://www.washingtonpost.com/sports/2020/07/...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Colin Kaepernick and Disney announce new partn...</td>
      <td>The larger deal between Kaepernick’s productio...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>5.21</td>
    </tr>
    <tr>
      <th>185642</th>
      <td>The Guardian</td>
      <td>theguardian</td>
      <td>10513336322</td>
      <td>MEDIA_NEWS_COMPANY</td>
      <td>GB</td>
      <td>The world's leading liberal voice, since 1821</td>
      <td>2007-11-26 17:15:26</td>
      <td>8388654</td>
      <td>8716143</td>
      <td>2020-07-06 17:06:26 EDT</td>
      <td>...</td>
      <td>NaN</td>
      <td>https://www.theguardian.com/us-news/2020/jul/0...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>White woman who called police over black birdw...</td>
      <td>Amy Cooper charged with filing a false report ...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.55</td>
    </tr>
    <tr>
      <th>185645</th>
      <td>CNN</td>
      <td>cnn</td>
      <td>5550296508</td>
      <td>MEDIA_NEWS_COMPANY</td>
      <td>US</td>
      <td>Instant breaking news alerts and the most talk...</td>
      <td>2007-11-07 22:14:27</td>
      <td>33363423</td>
      <td>37359039</td>
      <td>2020-07-06 17:01:17 EDT</td>
      <td>...</td>
      <td>It isn't entirely surprising: Many galaxies se...</td>
      <td>https://cnn.it/38xrZ2P</td>
      <td>https://www.cnn.com/2020/07/06/world/hubble-sc...</td>
      <td>NaN</td>
      <td>Hubble spots galaxy moving away from us at 3 m...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.93</td>
    </tr>
  </tbody>
</table>
<p>22164 rows × 40 columns</p>
</div>




```python
print(df_2.shape)
```

    (22164, 40)


Let's replace our data with the trimmed down version.


```python
df = df_2
```


```python
print(df.shape)
```

    (22164, 40)


<div class="alert alert-info" style="background-color:#5d3a8e; color:white; padding:0px 10px; border-radius:5px;"><h2 style='margin:10px 5px'> Remove newline characters</h2>
</div>

Using [regular expressions].


```python
# Convert to list
data = df.text.values.tolist()

# Remove new line characters
data = [re.sub('\s+', ' ', sent) for sent in data]

# Remove single quotes
data = [re.sub("\'"," ", sent) for sent in data]

pprint(data[:2])
```

    ['WATCH: 100 protesters show up at home of man whose racist rant at neighbor '
     'went viral',
     'Kinzinger says he suspects some lawmakers knew what was going to happen on '
     'Jan. 6']


<div class="alert alert-info" style="background-color:#5d3a8e; color:white; padding:0px 10px; border-radius:5px;"><h2 style='margin:10px 5px'> Tokenize words and Clean-up text </h2>
</div>

Tokenize each sentence into a list of words, removing punctuations and unneccessary characters.

Using Gensim's `simple_preprocess()` and `deacc=True` to remove the punctuations.


```python
def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence).encode('utf-8'), deacc=True))  # deacc=True removes punctuations

data_words = list(sent_to_words(data))
print(data_words[:1])

```

    [['watch', 'protesters', 'show', 'up', 'at', 'home', 'of', 'man', 'whose', 'racist', 'rant', 'at', 'neighbor', 'went', 'viral']]


<div class="alert alert-info" style="background-color:#5d3a8e; color:white; padding:0px 10px; border-radius:5px;"><h2 style='margin:10px 5px'> Create Bigram and Trigram Models</h2>
</div>



```python
# Build the bigram and trigram models
bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  

# Parser for sentences
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

# See trigram example
print(trigram_mod[bigram_mod[data_words[0]]])
```

    ['watch', 'protesters', 'show', 'up', 'at', 'home', 'of', 'man', 'whose', 'racist', 'rant', 'at', 'neighbor', 'went_viral']


<div class="alert alert-info" style="background-color:#5d3a8e; color:white; padding:0px 10px; border-radius:5px;"><h2 style='margin:10px 5px'> Remove Stopwords, Make Bigrams and Lemmatize</h2>
</div>


Remove the stopwords, make bigrams and lemmatization and call them sequentially. Spacy!


```python
# Define functions for stopwords, bigrams, trigrams and lemmatization
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out
```


```python
import sys
!{sys.executable} -m pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.0/en_core_web_sm-2.2.0.tar.gz
```

    Collecting https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.0/en_core_web_sm-2.2.0.tar.gz
      Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.0/en_core_web_sm-2.2.0.tar.gz (12.0 MB)
    [K     |████████████████████████████████| 12.0 MB 3.3 MB/s eta 0:00:01
    [?25hRequirement already satisfied: spacy>=2.2.0 in /Users/a206679878/opt/anaconda3/lib/python3.8/site-packages (from en-core-web-sm==2.2.0) (2.3.5)
    Requirement already satisfied: blis<0.8.0,>=0.4.0 in /Users/a206679878/opt/anaconda3/lib/python3.8/site-packages (from spacy>=2.2.0->en-core-web-sm==2.2.0) (0.7.4)
    Requirement already satisfied: requests<3.0.0,>=2.13.0 in /Users/a206679878/opt/anaconda3/lib/python3.8/site-packages (from spacy>=2.2.0->en-core-web-sm==2.2.0) (2.24.0)
    Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /Users/a206679878/opt/anaconda3/lib/python3.8/site-packages (from spacy>=2.2.0->en-core-web-sm==2.2.0) (0.8.2)
    Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /Users/a206679878/opt/anaconda3/lib/python3.8/site-packages (from spacy>=2.2.0->en-core-web-sm==2.2.0) (1.0.5)
    Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /Users/a206679878/opt/anaconda3/lib/python3.8/site-packages (from spacy>=2.2.0->en-core-web-sm==2.2.0) (4.50.2)
    Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /Users/a206679878/opt/anaconda3/lib/python3.8/site-packages (from spacy>=2.2.0->en-core-web-sm==2.2.0) (3.0.5)
    Requirement already satisfied: numpy>=1.15.0 in /Users/a206679878/opt/anaconda3/lib/python3.8/site-packages (from spacy>=2.2.0->en-core-web-sm==2.2.0) (1.21.0)
    Requirement already satisfied: plac<1.2.0,>=0.9.6 in /Users/a206679878/opt/anaconda3/lib/python3.8/site-packages (from spacy>=2.2.0->en-core-web-sm==2.2.0) (1.1.3)
    Requirement already satisfied: thinc<7.5.0,>=7.4.1 in /Users/a206679878/opt/anaconda3/lib/python3.8/site-packages (from spacy>=2.2.0->en-core-web-sm==2.2.0) (7.4.5)
    Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /Users/a206679878/opt/anaconda3/lib/python3.8/site-packages (from spacy>=2.2.0->en-core-web-sm==2.2.0) (1.0.0)
    Requirement already satisfied: setuptools in /Users/a206679878/opt/anaconda3/lib/python3.8/site-packages (from spacy>=2.2.0->en-core-web-sm==2.2.0) (50.3.1.post20201107)
    Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /Users/a206679878/opt/anaconda3/lib/python3.8/site-packages (from spacy>=2.2.0->en-core-web-sm==2.2.0) (1.0.5)
    Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /Users/a206679878/opt/anaconda3/lib/python3.8/site-packages (from spacy>=2.2.0->en-core-web-sm==2.2.0) (2.0.5)
    Requirement already satisfied: chardet<4,>=3.0.2 in /Users/a206679878/opt/anaconda3/lib/python3.8/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.0->en-core-web-sm==2.2.0) (3.0.4)
    Requirement already satisfied: idna<3,>=2.5 in /Users/a206679878/opt/anaconda3/lib/python3.8/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.0->en-core-web-sm==2.2.0) (2.10)
    Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /Users/a206679878/opt/anaconda3/lib/python3.8/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.0->en-core-web-sm==2.2.0) (1.25.11)
    Requirement already satisfied: certifi>=2017.4.17 in /Users/a206679878/opt/anaconda3/lib/python3.8/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.0->en-core-web-sm==2.2.0) (2020.6.20)
    Building wheels for collected packages: en-core-web-sm
      Building wheel for en-core-web-sm (setup.py) ... [?25ldone
    [?25h  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.0-py3-none-any.whl size=12019122 sha256=0501e14727038c595e222702e791b1af40eb2df9d819712069d35336cd11c0aa
      Stored in directory: /Users/a206679878/Library/Caches/pip/wheels/fc/31/e9/092e6f05b2817c9cb45804a3d1bf2b9bf6575742c01819337c
    Successfully built en-core-web-sm
    Installing collected packages: en-core-web-sm
      Attempting uninstall: en-core-web-sm
        Found existing installation: en-core-web-sm 2.3.1
        Uninstalling en-core-web-sm-2.3.1:
          Successfully uninstalled en-core-web-sm-2.3.1
    Successfully installed en-core-web-sm-2.2.0



```python
import sys
!{sys.executable} -m pip install spacy==2.3.5

!{sys.executable} -m pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.1/en_core_web_sm-2.3.1.tar.gz

!{sys.executable} -m pip install pyresparser
```

    Requirement already satisfied: spacy==2.3.5 in /Users/a206679878/opt/anaconda3/lib/python3.8/site-packages (2.3.5)
    Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /Users/a206679878/opt/anaconda3/lib/python3.8/site-packages (from spacy==2.3.5) (3.0.5)
    Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /Users/a206679878/opt/anaconda3/lib/python3.8/site-packages (from spacy==2.3.5) (1.0.0)
    Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /Users/a206679878/opt/anaconda3/lib/python3.8/site-packages (from spacy==2.3.5) (4.50.2)
    Requirement already satisfied: thinc<7.5.0,>=7.4.1 in /Users/a206679878/opt/anaconda3/lib/python3.8/site-packages (from spacy==2.3.5) (7.4.5)
    Requirement already satisfied: numpy>=1.15.0 in /Users/a206679878/opt/anaconda3/lib/python3.8/site-packages (from spacy==2.3.5) (1.21.0)
    Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /Users/a206679878/opt/anaconda3/lib/python3.8/site-packages (from spacy==2.3.5) (1.0.5)
    Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /Users/a206679878/opt/anaconda3/lib/python3.8/site-packages (from spacy==2.3.5) (1.0.5)
    Requirement already satisfied: setuptools in /Users/a206679878/opt/anaconda3/lib/python3.8/site-packages (from spacy==2.3.5) (50.3.1.post20201107)
    Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /Users/a206679878/opt/anaconda3/lib/python3.8/site-packages (from spacy==2.3.5) (0.8.2)
    Requirement already satisfied: plac<1.2.0,>=0.9.6 in /Users/a206679878/opt/anaconda3/lib/python3.8/site-packages (from spacy==2.3.5) (1.1.3)
    Requirement already satisfied: blis<0.8.0,>=0.4.0 in /Users/a206679878/opt/anaconda3/lib/python3.8/site-packages (from spacy==2.3.5) (0.7.4)
    Requirement already satisfied: requests<3.0.0,>=2.13.0 in /Users/a206679878/opt/anaconda3/lib/python3.8/site-packages (from spacy==2.3.5) (2.24.0)
    Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /Users/a206679878/opt/anaconda3/lib/python3.8/site-packages (from spacy==2.3.5) (2.0.5)
    Requirement already satisfied: idna<3,>=2.5 in /Users/a206679878/opt/anaconda3/lib/python3.8/site-packages (from requests<3.0.0,>=2.13.0->spacy==2.3.5) (2.10)
    Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /Users/a206679878/opt/anaconda3/lib/python3.8/site-packages (from requests<3.0.0,>=2.13.0->spacy==2.3.5) (1.25.11)
    Requirement already satisfied: chardet<4,>=3.0.2 in /Users/a206679878/opt/anaconda3/lib/python3.8/site-packages (from requests<3.0.0,>=2.13.0->spacy==2.3.5) (3.0.4)
    Requirement already satisfied: certifi>=2017.4.17 in /Users/a206679878/opt/anaconda3/lib/python3.8/site-packages (from requests<3.0.0,>=2.13.0->spacy==2.3.5) (2020.6.20)
    Collecting https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.1/en_core_web_sm-2.3.1.tar.gz
      Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.1/en_core_web_sm-2.3.1.tar.gz (12.0 MB)
    [K     |████████████████████████████████| 12.0 MB 2.5 MB/s eta 0:00:01    |██████████████                  | 5.2 MB 8.1 MB/s eta 0:00:01
    [?25hRequirement already satisfied: spacy<2.4.0,>=2.3.0 in /Users/a206679878/opt/anaconda3/lib/python3.8/site-packages (from en-core-web-sm==2.3.1) (2.3.5)
    Requirement already satisfied: plac<1.2.0,>=0.9.6 in /Users/a206679878/opt/anaconda3/lib/python3.8/site-packages (from spacy<2.4.0,>=2.3.0->en-core-web-sm==2.3.1) (1.1.3)
    Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /Users/a206679878/opt/anaconda3/lib/python3.8/site-packages (from spacy<2.4.0,>=2.3.0->en-core-web-sm==2.3.1) (1.0.5)
    Requirement already satisfied: numpy>=1.15.0 in /Users/a206679878/opt/anaconda3/lib/python3.8/site-packages (from spacy<2.4.0,>=2.3.0->en-core-web-sm==2.3.1) (1.21.0)
    Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /Users/a206679878/opt/anaconda3/lib/python3.8/site-packages (from spacy<2.4.0,>=2.3.0->en-core-web-sm==2.3.1) (0.8.2)
    Requirement already satisfied: requests<3.0.0,>=2.13.0 in /Users/a206679878/opt/anaconda3/lib/python3.8/site-packages (from spacy<2.4.0,>=2.3.0->en-core-web-sm==2.3.1) (2.24.0)
    Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /Users/a206679878/opt/anaconda3/lib/python3.8/site-packages (from spacy<2.4.0,>=2.3.0->en-core-web-sm==2.3.1) (2.0.5)
    Requirement already satisfied: blis<0.8.0,>=0.4.0 in /Users/a206679878/opt/anaconda3/lib/python3.8/site-packages (from spacy<2.4.0,>=2.3.0->en-core-web-sm==2.3.1) (0.7.4)
    Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /Users/a206679878/opt/anaconda3/lib/python3.8/site-packages (from spacy<2.4.0,>=2.3.0->en-core-web-sm==2.3.1) (4.50.2)
    Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /Users/a206679878/opt/anaconda3/lib/python3.8/site-packages (from spacy<2.4.0,>=2.3.0->en-core-web-sm==2.3.1) (1.0.0)
    Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /Users/a206679878/opt/anaconda3/lib/python3.8/site-packages (from spacy<2.4.0,>=2.3.0->en-core-web-sm==2.3.1) (3.0.5)
    Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /Users/a206679878/opt/anaconda3/lib/python3.8/site-packages (from spacy<2.4.0,>=2.3.0->en-core-web-sm==2.3.1) (1.0.5)
    Requirement already satisfied: thinc<7.5.0,>=7.4.1 in /Users/a206679878/opt/anaconda3/lib/python3.8/site-packages (from spacy<2.4.0,>=2.3.0->en-core-web-sm==2.3.1) (7.4.5)
    Requirement already satisfied: setuptools in /Users/a206679878/opt/anaconda3/lib/python3.8/site-packages (from spacy<2.4.0,>=2.3.0->en-core-web-sm==2.3.1) (50.3.1.post20201107)
    Requirement already satisfied: idna<3,>=2.5 in /Users/a206679878/opt/anaconda3/lib/python3.8/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en-core-web-sm==2.3.1) (2.10)
    Requirement already satisfied: chardet<4,>=3.0.2 in /Users/a206679878/opt/anaconda3/lib/python3.8/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en-core-web-sm==2.3.1) (3.0.4)
    Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /Users/a206679878/opt/anaconda3/lib/python3.8/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en-core-web-sm==2.3.1) (1.25.11)
    Requirement already satisfied: certifi>=2017.4.17 in /Users/a206679878/opt/anaconda3/lib/python3.8/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en-core-web-sm==2.3.1) (2020.6.20)
    Building wheels for collected packages: en-core-web-sm
      Building wheel for en-core-web-sm (setup.py) ... [?25ldone
    [?25h  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.3.1-py3-none-any.whl size=12047106 sha256=f498ed194b92424e5d910d27ccb5ef2def6ed3b0b32451fbb41ea358c879b3bb
      Stored in directory: /Users/a206679878/Library/Caches/pip/wheels/ee/4d/f7/563214122be1540b5f9197b52cb3ddb9c4a8070808b22d5a84
    Successfully built en-core-web-sm
    Installing collected packages: en-core-web-sm
      Attempting uninstall: en-core-web-sm
        Found existing installation: en-core-web-sm 2.2.0
        Uninstalling en-core-web-sm-2.2.0:
          Successfully uninstalled en-core-web-sm-2.2.0
    Successfully installed en-core-web-sm-2.3.1
    Requirement already satisfied: pyresparser in /Users/a206679878/opt/anaconda3/lib/python3.8/site-packages (1.0.6)
    Requirement already satisfied: pdfminer.six>=20181108 in /Users/a206679878/opt/anaconda3/lib/python3.8/site-packages (from pyresparser) (20201018)
    Requirement already satisfied: pycryptodome>=3.8.2 in /Users/a206679878/opt/anaconda3/lib/python3.8/site-packages (from pyresparser) (3.10.1)
    Requirement already satisfied: cymem>=2.0.2 in /Users/a206679878/opt/anaconda3/lib/python3.8/site-packages (from pyresparser) (2.0.5)
    Requirement already satisfied: pyrsistent>=0.15.2 in /Users/a206679878/opt/anaconda3/lib/python3.8/site-packages (from pyresparser) (0.17.3)
    Requirement already satisfied: preshed>=2.0.1 in /Users/a206679878/opt/anaconda3/lib/python3.8/site-packages (from pyresparser) (3.0.5)
    Requirement already satisfied: numpy>=1.16.4 in /Users/a206679878/opt/anaconda3/lib/python3.8/site-packages (from pyresparser) (1.21.0)
    Requirement already satisfied: thinc>=7.0.4 in /Users/a206679878/opt/anaconda3/lib/python3.8/site-packages (from pyresparser) (7.4.5)
    Requirement already satisfied: attrs>=19.1.0 in /Users/a206679878/opt/anaconda3/lib/python3.8/site-packages (from pyresparser) (20.3.0)
    Requirement already satisfied: six>=1.12.0 in /Users/a206679878/opt/anaconda3/lib/python3.8/site-packages (from pyresparser) (1.15.0)
    Requirement already satisfied: blis>=0.2.4 in /Users/a206679878/opt/anaconda3/lib/python3.8/site-packages (from pyresparser) (0.7.4)
    Requirement already satisfied: python-dateutil>=2.8.0 in /Users/a206679878/opt/anaconda3/lib/python3.8/site-packages (from pyresparser) (2.8.1)
    Requirement already satisfied: wasabi>=0.2.2 in /Users/a206679878/opt/anaconda3/lib/python3.8/site-packages (from pyresparser) (0.8.2)
    Requirement already satisfied: pytz>=2019.1 in /Users/a206679878/opt/anaconda3/lib/python3.8/site-packages (from pyresparser) (2020.1)
    Requirement already satisfied: tqdm>=4.32.2 in /Users/a206679878/opt/anaconda3/lib/python3.8/site-packages (from pyresparser) (4.50.2)
    Requirement already satisfied: sortedcontainers>=2.1.0 in /Users/a206679878/opt/anaconda3/lib/python3.8/site-packages (from pyresparser) (2.2.2)
    Requirement already satisfied: requests>=2.22.0 in /Users/a206679878/opt/anaconda3/lib/python3.8/site-packages (from pyresparser) (2.24.0)
    Requirement already satisfied: jsonschema>=3.0.1 in /Users/a206679878/opt/anaconda3/lib/python3.8/site-packages (from pyresparser) (3.2.0)
    Requirement already satisfied: nltk>=3.4.3 in /Users/a206679878/opt/anaconda3/lib/python3.8/site-packages (from pyresparser) (3.5)
    Requirement already satisfied: idna>=2.8 in /Users/a206679878/opt/anaconda3/lib/python3.8/site-packages (from pyresparser) (2.10)
    Requirement already satisfied: srsly>=0.0.7 in /Users/a206679878/opt/anaconda3/lib/python3.8/site-packages (from pyresparser) (1.0.5)
    Requirement already satisfied: spacy>=2.1.4 in /Users/a206679878/opt/anaconda3/lib/python3.8/site-packages (from pyresparser) (2.3.5)
    Requirement already satisfied: urllib3>=1.25.3 in /Users/a206679878/opt/anaconda3/lib/python3.8/site-packages (from pyresparser) (1.25.11)
    Requirement already satisfied: chardet>=3.0.4 in /Users/a206679878/opt/anaconda3/lib/python3.8/site-packages (from pyresparser) (3.0.4)
    Requirement already satisfied: certifi>=2019.6.16 in /Users/a206679878/opt/anaconda3/lib/python3.8/site-packages (from pyresparser) (2020.6.20)
    Requirement already satisfied: pandas>=0.24.2 in /Users/a206679878/opt/anaconda3/lib/python3.8/site-packages (from pyresparser) (1.2.5)
    Requirement already satisfied: docx2txt>=0.7 in /Users/a206679878/opt/anaconda3/lib/python3.8/site-packages (from pyresparser) (0.8)
    Requirement already satisfied: cryptography in /Users/a206679878/opt/anaconda3/lib/python3.8/site-packages (from pdfminer.six>=20181108->pyresparser) (3.1.1)
    Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /Users/a206679878/opt/anaconda3/lib/python3.8/site-packages (from preshed>=2.0.1->pyresparser) (1.0.5)
    Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /Users/a206679878/opt/anaconda3/lib/python3.8/site-packages (from thinc>=7.0.4->pyresparser) (1.0.0)
    Requirement already satisfied: plac<1.2.0,>=0.9.6 in /Users/a206679878/opt/anaconda3/lib/python3.8/site-packages (from thinc>=7.0.4->pyresparser) (1.1.3)
    Requirement already satisfied: setuptools in /Users/a206679878/opt/anaconda3/lib/python3.8/site-packages (from jsonschema>=3.0.1->pyresparser) (50.3.1.post20201107)
    Requirement already satisfied: regex in /Users/a206679878/opt/anaconda3/lib/python3.8/site-packages (from nltk>=3.4.3->pyresparser) (2020.10.15)
    Requirement already satisfied: click in /Users/a206679878/opt/anaconda3/lib/python3.8/site-packages (from nltk>=3.4.3->pyresparser) (7.1.2)
    Requirement already satisfied: joblib in /Users/a206679878/opt/anaconda3/lib/python3.8/site-packages (from nltk>=3.4.3->pyresparser) (0.17.0)
    Requirement already satisfied: cffi!=1.11.3,>=1.8 in /Users/a206679878/opt/anaconda3/lib/python3.8/site-packages (from cryptography->pdfminer.six>=20181108->pyresparser) (1.14.3)
    Requirement already satisfied: pycparser in /Users/a206679878/opt/anaconda3/lib/python3.8/site-packages (from cffi!=1.11.3,>=1.8->cryptography->pdfminer.six>=20181108->pyresparser) (2.20)



```python
# Remove Stop Words
data_words_nostops = remove_stopwords(data_words)

# Form Bigrams
data_words_bigrams = make_bigrams(data_words_nostops)

# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
# python3 -m spacy download en
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

# Do lemmatization keeping only noun, adj, vb, adv
data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

print(data_lemmatized[:1])
```

    [['protester', 'show', 'home', 'man', 'racist', 'rant', 'neighbor']]


<div class="alert alert-info" style="background-color:#5d3a8e; color:white; padding:0px 10px; border-radius:5px;"><h2 style='margin:10px 5px'> Use id2word to create the Dictionary and Corpus</h2>
</div>


```python
# Create Dictionary
id2word = corpora.Dictionary(data_lemmatized)

# Create Corpus
texts = data_lemmatized

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]

# View
print(corpus[:1])
```

    [[(0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1)]]


(0, 1) above implies, word id 0 occurs once in the first document. 

Pass the id as a key to the dictionary.


```python
id2word[0]
```




    'home'




```python
corpus[:1][0][:10]
```




    [(0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1)]




```python
# Human readable format of corpus (term-frequency)
[[(id2word[id], freq) for id, freq in cp] for cp in corpus[:1]]
```




    [[('home', 1),
      ('man', 1),
      ('neighbor', 1),
      ('protester', 1),
      ('racist', 1),
      ('rant', 1),
      ('show', 1)]]



<div class="alert alert-info" style="background-color:#5d3a8e; color:white; padding:0px 10px; border-radius:5px;"><h2 style='margin:10px 5px'> Building the Topic Model</h2>
</div>

From the documentation `alpha` and `eta` are hyperparameters that affect sparsity of the topics. Both default to 1.0/num_topics prior.

`chunksize` is the number of documents to be used in each training chunk.  `update_every` determines how often the model parameters should be updated and `passes` is the total number of training passes.


```python
# Build LDA model
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=20, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)
```

<div class="alert alert-info" style="background-color:#5d3a8e; color:white; padding:0px 10px; border-radius:5px;"><h2 style='margin:10px 5px'> View the topics in LDA model</h2>
</div>

The above LDA model is built with 20 different topics where each topic is a combination of keywords and each keyword contributes a certain weightage to the topic.

The keywords for each topic and the weightage(importance) of each keyword.


```python
# Print the Keyword in the 10 topics
pprint(lda_model.print_topics())
doc_lda = lda_model[corpus]
```

    [(0,
      '0.382*"new" + 0.093*"join" + 0.091*"response" + 0.073*"worker" + '
      '0.068*"set" + 0.031*"boy" + 0.016*"politic" + 0.006*"rescue" + '
      '0.000*"state" + 0.000*"lead"'),
     (1,
      '0.137*"people" + 0.112*"make" + 0.104*"face" + 0.087*"claim" + '
      '0.083*"support" + 0.080*"could" + 0.057*"officer" + 0.035*"keep" + '
      '0.031*"full" + 0.029*"suspend"'),
     (2,
      '0.197*"raise" + 0.119*"arrest" + 0.106*"late" + 0.100*"concern" + '
      '0.092*"turn" + 0.002*"alleged" + 0.000*"convention" + 0.000*"campaign" + '
      '0.000*"federal" + 0.000*"democratic"'),
     (3,
      '0.370*"back" + 0.194*"city" + 0.099*"watch" + 0.000*"reopen" + '
      '0.000*"black_live" + 0.000*"paint" + 0.000*"federal" + 0.000*"hit" + '
      '0.000*"tower" + 0.000*"leg"'),
     (4,
      '0.242*"covid" + 0.142*"vote" + 0.111*"report" + 0.077*"honor" + '
      '0.050*"vaccine" + 0.049*"way" + 0.048*"last" + 0.039*"month" + '
      '0.037*"expert" + 0.028*"political"'),
     (5,
      '0.146*"find" + 0.082*"couple" + 0.079*"send" + 0.076*"protect" + '
      '0.074*"drop" + 0.068*"bad" + 0.045*"fine" + 0.044*"girl" + 0.040*"least" + '
      '0.039*"act"'),
     (6,
      '0.142*"black" + 0.139*"analysis" + 0.122*"charge" + 0.096*"become" + '
      '0.062*"accuse" + 0.047*"history" + 0.047*"threat" + 0.042*"still" + '
      '0.034*"year_old" + 0.030*"step"'),
     (7,
      '0.134*"poll" + 0.129*"give" + 0.115*"see" + 0.111*"virus" + 0.073*"voting" '
      '+ 0.069*"prosecutor" + 0.032*"seek" + 0.031*"violence" + 0.027*"evidence" + '
      '0.020*"celebrate"'),
     (8,
      '0.197*"opinion" + 0.177*"protester" + 0.119*"show" + 0.091*"use" + '
      '0.084*"attack" + 0.083*"home" + 0.050*"racist" + 0.014*"launch" + '
      '0.000*"federal" + 0.000*"portland"'),
     (9,
      '0.231*"rule" + 0.209*"case" + 0.102*"require" + 0.064*"issue" + '
      '0.061*"delay" + 0.037*"patient" + 0.020*"agency" + 0.020*"dozen" + '
      '0.019*"criminal" + 0.019*"plead"'),
     (10,
      '0.113*"child" + 0.107*"man" + 0.105*"come" + 0.082*"white" + 0.073*"try" + '
      '0.059*"power" + 0.050*"remark" + 0.034*"female" + 0.025*"chinese" + '
      '0.018*"story"'),
     (11,
      '0.140*"former" + 0.120*"want" + 0.107*"tell" + 0.063*"speak" + '
      '0.054*"handle" + 0.046*"test" + 0.045*"away" + 0.041*"right" + 0.039*"gun" '
      '+ 0.031*"stay"'),
     (12,
      '0.099*"big" + 0.086*"lie" + 0.085*"republican" + 0.058*"player" + '
      '0.054*"throw" + 0.049*"strip" + 0.046*"team" + 0.036*"learn" + 0.035*"tie" '
      '+ 0.032*"doctor"'),
     (13,
      '0.520*"trump" + 0.109*"biden" + 0.083*"pandemic" + 0.043*"plan" + '
      '0.037*"record" + 0.027*"job" + 0.024*"bill" + 0.016*"approve" + '
      '0.014*"visit" + 0.013*"slam"'),
     (14,
      '0.744*"say" + 0.037*"fire" + 0.019*"company" + 0.016*"oil" + '
      '0.013*"responsible" + 0.009*"gas" + 0.003*"eye" + 0.001*"leak" + '
      '0.000*"mask" + 0.000*"school"'),
     (15,
      '0.469*"call" + 0.167*"official" + 0.035*"reject" + 0.032*"study" + '
      '0.022*"provide" + 0.015*"protection" + 0.011*"strong" + 0.011*"ally" + '
      '0.000*"mask" + 0.000*"woman"'),
     (16,
      '0.214*"go" + 0.107*"know" + 0.077*"fail" + 0.057*"low" + 0.049*"body" + '
      '0.048*"lawmaker" + 0.040*"place" + 0.039*"mile" + 0.014*"suspect" + '
      '0.011*"happen"'),
     (17,
      '0.231*"order" + 0.203*"student" + 0.063*"end" + 0.058*"offer" + '
      '0.048*"attempt" + 0.032*"probe" + 0.016*"supreme_court" + 0.000*"college" + '
      '0.000*"school" + 0.000*"close"'),
     (18,
      '0.263*"first" + 0.131*"year" + 0.124*"time" + 0.075*"public" + 0.063*"pay" '
      '+ 0.056*"return" + 0.036*"military" + 0.034*"open" + 0.032*"change" + '
      '0.025*"free"'),
     (19,
      '0.340*"day" + 0.138*"need" + 0.119*"break" + 0.000*"hour" + '
      '0.000*"election" + 0.000*"death" + 0.000*"single" + 0.000*"leave" + '
      '0.000*"coronavirus" + 0.000*"opening"')]


Topic 0 is a represented as '0.382*"new" + 0.093*"join" + 0.091*"response" + 0.073*"worker" + '
  '0.068*"set" + 0.031*"boy" + 0.016*"politic" + 0.006*"rescue" + '
  '0.000*"state" + 0.000*"lead"'.

<div class="alert alert-info" style="background-color:#5d3a8e; color:white; padding:0px 10px; border-radius:5px;"><h2 style='margin:10px 5px'> Compute Model Perplexity and Coherence Score</h2>
</div>



```python
# Compute Perplexity
print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is.

# Compute Coherence Score
coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)
```

    
    Perplexity:  -21.5281969692066
    
    Coherence Score:  0.4581361164158878


<div class="alert alert-info" style="background-color:#5d3a8e; color:white; padding:0px 10px; border-radius:5px;"><h2 style='margin:10px 5px'> Visualize the topics-keywords</h2>
</div>
Pulling in pyLDAvis interactive chart.


```python
# Visualize the topics
pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, id2word)
vis
```





<link rel="stylesheet" type="text/css" href="https://cdn.jsdelivr.net/gh/bmabey/pyLDAvis@3.3.1/pyLDAvis/js/ldavis.v1.0.0.css">


<div id="ldavis_el176541404059900113927020288209"></div>
<script type="text/javascript">

var ldavis_el176541404059900113927020288209_data = {"mdsDat": {"x": [0.4274113155040989, 0.03948952743116043, 0.06442710056788815, 0.09405396743444867, 0.0390659600905163, 0.045449073301091984, -0.02627050278445839, 0.0015927102117455224, -0.027770343171267396, -0.04054939189382362, -0.056526048855742, -0.04979762154550341, -0.04172220503421808, -0.005698726439990262, -0.05736603927861148, -0.07546714708513218, -0.07959516485497989, -0.08830920333549905, -0.08574365944208856, -0.07667360081963506], "y": [0.1689912063206498, -0.04300139362832186, -0.11130272891987739, -0.3508524670232685, -0.04258258054711049, -0.05502460246391991, 0.02374316909126374, 0.004757454218761424, 0.024354939001971547, 0.029878403039582332, 0.035735984405418815, 0.03376327890892879, 0.030371294404342365, 0.010902881977319557, 0.03590903109191582, 0.04015192646274465, 0.0407453281335862, 0.04174017527539651, 0.041529521088006026, 0.040189179162610555], "topics": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], "cluster": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], "Freq": [19.52443314437361, 8.724583883695832, 7.714319885536801, 7.276659964861312, 6.582084043247393, 6.424237833301542, 4.494140759222637, 4.45430760304694, 4.443703583996737, 4.0549368173739655, 3.450280151138949, 3.445390383864589, 3.416338000059994, 3.3954508453567214, 2.8154957995594647, 2.750320944433056, 2.280516973315162, 1.8110701970548717, 1.733372590959965, 1.2083565956004556]}, "tinfo": {"Term": ["trump", "say", "call", "covid", "first", "new", "biden", "vote", "opinion", "people", "rule", "pandemic", "year", "black", "day", "protester", "analysis", "case", "time", "report", "make", "official", "charge", "go", "face", "order", "back", "former", "poll", "give", "trump", "biden", "pandemic", "plan", "record", "job", "bill", "approve", "visit", "slam", "office", "remain", "miss", "border", "drug", "abortion", "damage", "add", "mask", "school", "coronavirus", "lead", "take", "state", "election", "woman", "campaign", "get", "reopen", "win", "ban", "covid", "vote", "report", "honor", "vaccine", "way", "last", "month", "expert", "political", "friend", "divide", "marriage", "disease", "mail", "mask", "death", "remove", "long", "allow", "ban", "coronavirus", "election", "state", "week", "life", "school", "reopen", "nearly", "voter", "take", "woman", "get", "die", "people", "make", "face", "claim", "support", "could", "officer", "keep", "full", "suspend", "resign", "criticism", "panel", "level", "weekend", "play", "request", "mask", "take", "federal", "wear", "false", "coronavirus", "paint", "president", "black_live", "refuse", "state", "speech", "environmental", "school", "woman", "voter", "get", "first", "year", "time", "public", "pay", "return", "military", "open", "change", "free", "major", "car", "worth", "school", "close", "policy", "woman", "state", "coronavirus", "mask", "share", "win", "work", "taxis", "national", "take", "flag", "far", "candace", "health", "confederate", "name", "country", "election", "death", "get", "wear", "say", "fire", "company", "oil", "responsible", "gas", "eye", "leak", "mask", "school", "state", "ban", "reopen", "get", "wear", "election", "spread", "leader", "photo", "customer", "mandate", "consider", "surge", "demand", "look", "president", "coronavirus", "threaten", "control", "may", "refuse", "country", "take", "federal", "woman", "help", "campaign", "lead", "black", "analysis", "charge", "become", "accuse", "history", "threat", "still", "year_old", "step", "member", "young", "huge", "teen", "agree", "trip", "black_live", "woman", "painting", "state", "felony", "warn", "policy", "shoot", "school", "take", "police", "country", "world", "get", "win", "mask", "coronavirus", "reopen", "former", "want", "tell", "speak", "handle", "test", "away", "right", "gun", "stay", "thing", "racism", "buy", "rip", "parent", "fed", "run", "crony", "professional", "corrupt", "brandish", "aim", "jeep", "voter", "seem", "choice", "hubble", "endorse", "spot", "bring", "stop", "week", "speech", "group", "die", "move", "campaign", "take", "police", "get", "help", "school", "coronavirus", "opinion", "protester", "show", "use", "attack", "home", "racist", "launch", "federal", "portland", "replace", "authority", "matter", "camouflage", "masked", "statue", "video", "attorney", "future", "investigation", "brandish", "gather", "agent", "felony", "pointed_gun", "nurse", "disaster", "postmaster_general", "sue", "unfold", "let", "force", "police", "campaign", "well", "fight", "result", "black_live", "coronavirus", "get", "death", "election", "school", "take", "state", "mask", "poll", "give", "see", "virus", "voting", "prosecutor", "seek", "violence", "evidence", "celebrate", "trial", "access", "address", "mail", "withdrawal", "notice", "formal", "lead", "lawsuit", "personal", "grant", "video", "emerge", "transmission", "airborne", "confirm", "percent", "blame", "option", "follow", "coronavirus", "take", "get", "voter", "surge", "school", "national", "woman", "state", "death", "call", "official", "reject", "study", "provide", "protection", "strong", "ally", "confederate", "emergency", "top", "symbol", "undermine", "police", "propose", "bizarre", "look", "racially", "biased", "unity", "illegal", "declare", "global", "wear", "court", "risk", "preserve", "urgent", "hard", "divisive", "taboo", "woman", "mask", "would", "health", "protest", "death", "school", "coronavirus", "state", "federal", "take", "child", "man", "come", "white", "try", "power", "remark", "female", "chinese", "story", "together", "reach", "study_find", "drive", "real", "care", "statewide", "much", "police", "pilot", "deal", "black_live", "nationalist", "wave", "dress", "toxic", "deliver", "paint", "catch", "twitter", "old", "large", "reopen", "election", "may", "school", "woman", "announce", "mask", "lead", "big", "lie", "republican", "player", "throw", "strip", "team", "learn", "tie", "doctor", "ahead", "enough", "almost", "oppose", "failure", "space", "safety", "performance", "credit", "title", "visa", "entirely", "coursework", "pitch", "online", "opening", "season", "national", "fumble", "convention", "name", "state", "fight", "new", "join", "response", "worker", "set", "boy", "politic", "rescue", "beret", "single", "nationwide", "host", "ad", "green", "special", "polling", "dog", "forge", "thanksgiving", "name", "jump", "force", "situation", "ask", "extend", "reveal", "far", "officially", "little", "industry", "really", "lead", "good", "state", "national", "woman", "campaign", "take", "protest", "mask", "win", "federal", "election", "rule", "case", "require", "issue", "delay", "patient", "agency", "dozen", "criminal", "plead", "airline", "building", "several", "medium", "unveil", "rioter", "guilty", "coronavirus", "customer", "surge", "sue", "visa", "rise", "federal", "election", "mask", "law", "coursework", "entirely", "block", "reopen", "state", "school", "find", "couple", "send", "protect", "drop", "bad", "fine", "girl", "least", "act", "restore", "ruling", "lobster", "brandish", "felony", "instead", "pointed_gun", "voting_right", "urgent", "economy", "felon", "adopt", "employee", "dog", "cook", "administration", "teach", "punishable", "pollution", "ignoring", "even", "mom", "percent", "federal", "name", "mask", "death", "school", "work", "get", "police", "online", "go", "know", "fail", "low", "body", "lawmaker", "place", "mile", "suspect", "happen", "rate", "vaccination", "camera", "incompetence", "decide", "kid", "horribly", "mad", "nee", "fall", "experiment", "tomorrow", "wrong", "hit", "hubble", "tammy_duckworth", "deep", "leg", "spot", "comeback", "move", "school", "die", "online", "lose", "hour", "mask", "coronavirus", "order", "student", "end", "offer", "attempt", "probe", "supreme_court", "college", "lift", "bar", "indoor", "restaurant", "bizarre", "garbage", "collect", "head", "close", "teacher", "operation", "visa", "would", "chaos", "consulate", "crowd", "learning", "suspension", "census", "photo", "escalate", "restrain", "bike", "force", "release", "school", "book", "reopen", "family", "election", "mask", "death", "day", "need", "break", "single", "transition", "university", "hour", "learning", "international_student", "opening", "mostly", "brand", "rhetoric", "inventor", "pitch", "leave", "retire", "sad", "mammoth", "straight", "racial", "sale", "depressing", "bike", "fundraise", "seat", "bidder", "flight", "candidate", "bill_gate", "funeral", "online", "death", "may", "high", "national", "election", "fight", "coronavirus", "mask", "ban", "take", "get", "raise", "arrest", "late", "concern", "turn", "alleged", "million", "attorney", "scheme", "bribery", "convention", "pipeline", "challenger", "shatter", "environmental", "connection", "portland", "investigation", "speaker", "draw", "nearly", "camouflage", "masked", "powerful", "race", "democratic", "authority", "ring", "due", "gloved", "campaign", "large", "federal", "state", "video", "rename", "health", "back", "city", "watch", "paint", "leg", "tower", "hit", "reopen", "roll", "black_live", "incompetence", "kid", "manner", "reckless", "decide", "matter_mural", "bring", "dnc", "lose", "aim", "serve", "restriction", "international_student", "jeep", "mayor", "tammy_duckworth", "walk", "suggest", "federal", "bike", "push", "school", "speech"], "Freq": [11131.0, 5363.0, 2084.0, 2314.0, 2101.0, 1432.0, 2330.0, 1356.0, 962.0, 1161.0, 860.0, 1768.0, 1050.0, 1000.0, 676.0, 866.0, 977.0, 778.0, 991.0, 1061.0, 947.0, 744.0, 857.0, 647.0, 877.0, 579.0, 492.0, 689.0, 653.0, 631.0, 11129.465971366453, 2328.4681952127135, 1766.7466774141371, 927.2179251495704, 787.4631017355254, 575.6054071525709, 517.3566723538345, 334.6070347995992, 296.5443054748198, 271.9878575212803, 246.5411501060526, 238.40944454588933, 151.3932785787276, 141.52557403851623, 137.91723810837962, 81.34771471715698, 62.15054288940587, 52.33790327574455, 0.15903136484860192, 0.15903343608457246, 0.15897124066078067, 0.1588686415493943, 0.1588959079609503, 0.15887292015891094, 0.1588383898352892, 0.1588233952537559, 0.1587816690868562, 0.15877823647513034, 0.15878060915858958, 0.15874510642371406, 0.15874567042224125, 2312.1409354805674, 1354.0370580712674, 1059.2960010508318, 739.0748196811368, 474.4790814019685, 464.5423433582865, 456.2194397859869, 372.4884234794626, 350.4909095663185, 270.291840321124, 161.63860221006018, 23.46667122283468, 22.444918524760293, 7.7523960048382925, 0.17651081624542836, 0.17662269812804632, 0.17650442001538277, 0.1764300638411029, 0.17642347641939835, 0.17641104896156523, 0.17644730585253013, 0.176535601636855, 0.17648867277510752, 0.17652628538874512, 0.17642057378239395, 0.176408858948017, 0.1765665051613687, 0.1764679371706391, 0.17640015103700385, 0.17643232337889073, 0.1764781398528042, 0.17645842973087025, 0.17644249129893602, 0.1764215297406888, 1159.7249315345823, 945.0907303510719, 875.4755106597718, 738.8007394348559, 702.9318716894938, 674.6015294223869, 485.0199138870856, 293.74116070392944, 257.7278721188726, 241.27977528952295, 180.4600790224057, 176.26312475441415, 128.88296240739217, 119.9118977108571, 104.23948258338554, 89.88706952429122, 85.3584131479128, 0.14140775367553884, 0.14127827474996102, 0.14126341348930418, 0.14123410591528912, 0.1411983128686813, 0.14128442211734749, 0.14116811392639522, 0.14118583371288676, 0.14119467055350482, 0.14115419013926483, 0.1412439417031075, 0.14115978424358652, 0.1411303229853868, 0.14127698380280984, 0.14120267749952573, 0.1411695739261495, 0.14117350824127686, 2099.7334424018013, 1047.981478393891, 989.4831956916876, 598.6708179777212, 499.7036459804823, 448.53040740497266, 283.80270052429853, 273.11824278210304, 256.1059007582379, 201.01891295980795, 148.41231091814052, 68.25438453961483, 30.626207310601323, 0.12169723023690407, 0.12153361531824854, 0.12149185085799545, 0.12154302355660476, 0.12156872587803091, 0.12156745018469448, 0.12158886878588293, 0.1214663659842971, 0.12149280037974018, 0.12147799943773448, 0.12145536312927034, 0.12146564840679536, 0.12151905356874343, 0.12145701573200163, 0.12144633180030898, 0.12143960541726231, 0.12145938591223467, 0.12146017597231235, 0.12147004085089694, 0.12147515087250028, 0.12149358319156027, 0.12147627435242726, 0.12148197148289566, 0.12147178768097695, 5361.013762371584, 268.51370028094465, 138.53121035419372, 115.36329419827692, 93.42912289487795, 64.2901284500843, 20.12385328605121, 10.280404531907699, 0.13499718888344245, 0.13498551850434526, 0.13494169557519606, 0.1348625730274742, 0.13488848389162705, 0.134872014233036, 0.1348523712803533, 0.13487353531615426, 0.13480942690783274, 0.1348097153891138, 0.13479304903874018, 0.13479308837709666, 0.1347857583300008, 0.13478800061632173, 0.1348027656127975, 0.1347915935195494, 0.13479900224335828, 0.1348157079320884, 0.13488468118383135, 0.1347739961614051, 0.13477747104956325, 0.13481480314988875, 0.13479206557982748, 0.13481039725395993, 0.13483127280847984, 0.13482277572347423, 0.13481949752709863, 0.13479784831823408, 0.1347990415817148, 0.13479510774606407, 998.3586609212588, 975.1357413147612, 855.3119240439048, 678.0925823326788, 435.62063711787476, 328.896014319415, 327.8732360508313, 296.3077053689461, 241.33874012872351, 211.07417784775106, 207.96419381033553, 182.60847445232483, 74.31592452135835, 70.00889202717211, 61.638182791718954, 26.709666682461425, 0.1264450862549765, 0.12645660474774478, 0.12638424301650938, 0.12645966354749102, 0.1263423412994834, 0.1263645591922009, 0.1263372347676895, 0.12633265296723276, 0.12644772270998791, 0.1263868538748702, 0.12634804935256638, 0.12634318598895308, 0.12631183009197278, 0.12636252425847852, 0.1263344063377986, 0.1263934834073746, 0.12636285701493627, 0.1263396792478214, 687.3574327943012, 592.6635024491346, 526.2258362034595, 307.8861112212433, 263.8792041529803, 226.86315603008464, 222.46424026259362, 202.68134952103944, 193.2423761596348, 151.70046193362958, 102.94321761202357, 97.11696090778786, 89.99307319302471, 84.38575012167483, 71.91548567249701, 48.753801709413864, 0.1247999116801911, 0.1247829364161777, 0.12478303490135921, 0.12478379592321635, 0.12478393917438946, 0.12478978561289196, 0.12477853144260459, 0.12482506121427003, 0.12477719741605502, 0.12478247980306341, 0.12477845086381972, 0.1247930445770802, 0.12478028626947518, 0.12478582829923483, 0.1248107629565541, 0.12480581183788353, 0.12480368992988185, 0.12479367130096254, 0.12480869476774234, 0.12479654527762304, 0.12480771886912553, 0.12481409354632889, 0.12479909693914404, 0.12480614310622135, 0.12479646469883816, 0.12481688694420451, 0.12480294681442135, 960.6863803535912, 864.2786878003055, 582.7735511453595, 442.8415767506891, 410.0910228217778, 403.3443169719359, 241.6240679760786, 67.68573270603427, 0.10737026975817063, 0.10728902972585594, 0.10726970249588527, 0.10727026154799187, 0.10727858521268997, 0.10725858357065421, 0.10725858357065421, 0.10726437819010823, 0.10728427334602844, 0.10725912487507488, 0.10725781154631654, 0.10726346418428317, 0.10725593916545162, 0.10725480331355251, 0.10725743884491215, 0.10726310035672174, 0.10725347223710825, 0.1072575897002425, 0.10725175071157368, 0.10725226539446546, 0.1072801292613653, 0.1072488223433963, 0.10726495498990074, 0.1072671202075834, 0.1072804132243401, 0.10728126511326441, 0.10726710245989748, 0.10727156600290724, 0.10726050032073395, 0.10727170798439463, 0.10728054633198451, 0.107271388526048, 0.10726840691481286, 0.10726866425625875, 0.1072755414845541, 0.10726855777014321, 0.1072683980409699, 0.10726735092750041, 651.7781675158419, 629.126886151319, 562.014726162254, 540.1906971310135, 357.28192101996615, 334.0494889082915, 155.02842527813516, 151.7880937708682, 132.10977540833215, 96.84592373234948, 80.93252994794393, 64.15375113826904, 56.532724372117706, 0.125197152177972, 0.12513965377648986, 0.12513094270226993, 0.12513094270226993, 0.1251658755263267, 0.12511709705178217, 0.1251062790307489, 0.12510331337031833, 0.12512987152342786, 0.1250979663288256, 0.12509809911959116, 0.12509929423648108, 0.1250992765310457, 0.12511421991852864, 0.12509969260877774, 0.12509001658832816, 0.12510194119907433, 0.12518562593952248, 0.12514325683259506, 0.12513209355557137, 0.12512070896060506, 0.12510996176131342, 0.12514106135860467, 0.12510635870520823, 0.1251171767262415, 0.1251188410371697, 0.1251113427852751, 2082.079361288432, 742.1047157442206, 156.88821460601255, 142.13900071201022, 97.77023899586086, 67.36141737657873, 47.516418442802454, 46.7458189030827, 0.12547477584245628, 0.12545703607342068, 0.1254681193899584, 0.12544128354627157, 0.12544023337779497, 0.12547766784487646, 0.1254428103296722, 0.1254359923127933, 0.12545659984959193, 0.12542862497701895, 0.12542862497701895, 0.12543188857751547, 0.12542921468700965, 0.12543790685070833, 0.12543181587354402, 0.12547487278108488, 0.12545070274968484, 0.12543541068102163, 0.12542865728989513, 0.12542333374354067, 0.12543180779532498, 0.12542122532836839, 0.12542243706122602, 0.12548990634673846, 0.12553159803525973, 0.1254390135667183, 0.12543831883987994, 0.1254414774235288, 0.1254496929723035, 0.12547801520829566, 0.12546137407705096, 0.12545987960652655, 0.12543969213711856, 0.12543975676287097, 428.3512549848231, 404.73297479368125, 397.7704549551945, 309.85803483472165, 274.1151025803828, 223.8366030964919, 189.67542568168253, 127.34641867274149, 93.70288753792799, 69.21815655450553, 61.322783837837356, 58.633010176648945, 56.88612290789372, 52.76481205793832, 19.364935895313128, 0.11996575076408372, 0.11994539108408259, 0.11994427755668212, 0.11998142263120141, 0.1199401946228804, 0.11994748066636494, 0.1199840208618025, 0.11993435204084092, 0.11994000216135439, 0.11993461323862621, 0.11993358219473689, 0.11993708774396059, 0.11995563278671652, 0.11993095646963209, 0.11993649661213071, 0.11993448951335949, 0.11994249041394063, 0.1199765973457994, 0.1199807490158604, 0.11995550906144979, 0.11998682530118146, 0.11995747491846544, 0.11994554230385301, 0.11996062303914083, 0.11994660084224605, 374.77197358043304, 324.89108200821704, 322.09034650354783, 220.1891926745295, 202.41949739295524, 183.19871585303426, 172.03937937916066, 137.10218673149276, 131.70312585742846, 121.50148072989492, 117.31885570907545, 101.15041310504823, 97.32077229853525, 82.6584430015755, 79.76900634032745, 68.87543596253632, 51.58797063360061, 18.26332389512261, 15.619138551766467, 3.4810848410619526, 0.11251359887860057, 0.11248122879898782, 0.11248110524906564, 0.11247406290350104, 0.1124843930497727, 0.11245513230986831, 0.11245185137304581, 0.11247080255833224, 0.1124404230052436, 0.11246437796237856, 0.11247037699748916, 0.11250808717929417, 0.11246535949787148, 1430.038516073136, 346.7567571301034, 341.96031055559104, 273.25363522508763, 253.2150512916256, 114.5819666781157, 58.857365828612615, 24.21831440227135, 0.10651708631252887, 0.10651820249761024, 0.10651708631252887, 0.1065211699164851, 0.10652621997337763, 0.10651189332949787, 0.10650725163300097, 0.10650919134488017, 0.10651113105675938, 0.10650190211181831, 0.10650308635696562, 0.1065318689588504, 0.1065010445549875, 0.10651909408447402, 0.1064988870508973, 0.10651227446586713, 0.10649699498106424, 0.10650411406396126, 0.10650378056963818, 0.10650400516785577, 0.10649925457525336, 0.10649536834548835, 0.10649926138125995, 0.10653983198656505, 0.1065094840031637, 0.10656133896740116, 0.10650875576045818, 0.10652433470955118, 0.10651584761932882, 0.10652377661701048, 0.10650858561029333, 0.10651491519642547, 0.1065068909146515, 0.10650887146257026, 0.10650782333755483, 858.3406105204988, 776.7035947286817, 379.97021930850434, 237.09779455095193, 227.66063660707528, 138.115831831958, 75.93664750186389, 75.45294626320481, 71.74615632380119, 70.02127317665443, 30.726087196580757, 22.958956075874983, 20.671207700847678, 14.219488040854747, 6.043969268654416, 2.662399118865483, 2.0076257809458533, 0.08418485489703129, 0.08409919735886072, 0.08409916353688403, 0.08410205869808889, 0.08409020747745595, 0.08408365954276835, 0.08411041949072719, 0.0841149110492319, 0.08415155377878018, 0.08406977900353388, 0.0840580765995984, 0.0840580765995984, 0.0840580765995984, 0.08409299440833538, 0.08409590309833091, 0.08410159195481053, 448.82751991942234, 253.03180463887182, 242.26613037080523, 233.97159501307695, 229.00572236476077, 208.74010882293095, 137.7774125856702, 134.7832473378151, 123.40252230953409, 120.7065892501794, 85.47472139770153, 31.133329121445897, 0.09894440740844379, 0.09894049231763098, 0.09894779525206979, 0.09893621825001012, 0.09893674549719694, 0.09894004359662091, 0.0989350179213082, 0.09894082885838854, 0.09893409804323756, 0.09892898262372282, 0.0989333015634447, 0.09893515253761122, 0.09892467490202621, 0.0989326509179801, 0.09893247142957608, 0.0989257406144251, 0.09892525823933929, 0.0989263015156877, 0.09893353714197498, 0.09893149546137919, 0.09893910128249978, 0.098966619098442, 0.09894422792003976, 0.09897346209384547, 0.098945798443575, 0.09896117835619496, 0.09893672306114644, 0.09894224232957023, 0.0989370820379545, 0.09893517497366172, 645.5924017438363, 321.50886588700604, 231.97167782437583, 172.4594950423745, 147.31210486518447, 145.10689763466243, 121.632632236681, 117.3342628918688, 41.78793023797301, 34.6103913476943, 32.65315056356307, 11.162829240902939, 1.4821045477819903, 0.11713039027697233, 0.1171305108187497, 0.11713057656881008, 0.11711641838914101, 0.11711641838914101, 0.1171202976427036, 0.11714360603910924, 0.11711658276429197, 0.1171138212617559, 0.11711784297378264, 0.11713715157484834, 0.11710805717312903, 0.11710816675656299, 0.11710520800384577, 0.11710823250662337, 0.11710828729834036, 0.11710030962434727, 0.11712589735617951, 0.11720007438263455, 0.11712448372988128, 0.11711774434869207, 0.11711505955455978, 0.11711141042620854, 0.11713214361191587, 0.11712481248018321, 577.5734025210993, 506.3129637357795, 156.69866673408401, 144.50518186765328, 121.13555166657373, 79.23782366218052, 41.02990308725767, 0.10328630987328653, 0.10324854653514867, 0.10324488469076235, 0.10324121375991353, 0.10324225870310069, 0.1032388331067393, 0.10323587091996525, 0.10323587091996525, 0.10323590726581523, 0.10325846895219548, 0.10324576607762456, 0.10322476726279381, 0.10324894633949855, 0.10324246769173813, 0.10322373140606915, 0.10321732545000871, 0.10322062383589514, 0.10322131440704492, 0.10321511743962192, 0.10321575349199671, 0.10322803838929276, 0.10321529008240936, 0.10321302755324759, 0.10321557176274677, 0.10323010101627961, 0.10322800204344278, 0.10327996752246357, 0.10322344063926923, 0.10324480291259988, 0.10322656638236824, 0.10323545294269038, 0.10324195884983829, 0.10322829281024268, 674.1825620820306, 274.37589480739024, 235.54646075718253, 0.09469641753029218, 0.09469226111152758, 0.09469212400743639, 0.0947023490862375, 0.0946921312234412, 0.09469213843944599, 0.0946937259605019, 0.09468683467591825, 0.09468683467591825, 0.09468683467591825, 0.09468683467591825, 0.09469148178300922, 0.09469604951404739, 0.09468684189192304, 0.0946817473925345, 0.0946776414858035, 0.09467746830168831, 0.09468688518795185, 0.0946776559178131, 0.09467104605741662, 0.09467321085885652, 0.094671644985815, 0.0946716666338294, 0.09466901836006793, 0.09467324693888052, 0.094671644985815, 0.09466901836006793, 0.09467745386967871, 0.09469218895147959, 0.09469948433233204, 0.09469238378360918, 0.09467712193345793, 0.09468243291299047, 0.09470119452546955, 0.09468741195630222, 0.09469464239311147, 0.09469162610310522, 0.09468075879987695, 0.09468071550384814, 0.09467940940697941, 373.9737442389704, 225.11168832584022, 200.6990545272514, 190.28240978870076, 175.0989165523283, 4.646232438651686, 0.08633608936324817, 0.08633591670256174, 0.08633245658240567, 0.08633048134415289, 0.08634759547139195, 0.0863300117070858, 0.08633001861351325, 0.08633218723173483, 0.08633001861351325, 0.08632345060100141, 0.08633410031214049, 0.08633000480065835, 0.08632076400072054, 0.08632532914926978, 0.08633471498418419, 0.08631825696755356, 0.08631825696755356, 0.08632150298845846, 0.08633082666552576, 0.08633737395875522, 0.08631968659803721, 0.08631309095981554, 0.0863169240270543, 0.08631281470271725, 0.0863421601129831, 0.08632214528621199, 0.0863399155240595, 0.08633424534711709, 0.08632293952536958, 0.08632001120012771, 0.08632014932867685, 489.8958339618963, 256.6528853731866, 130.50397364022106, 0.05289447057975643, 0.05288407594368453, 0.05288633397254869, 0.05289227514017848, 0.05290830762657007, 0.052878235881824776, 0.05289525053855386, 0.05287521233784462, 0.052877003354342415, 0.052871370318583216, 0.052871370318583216, 0.05287521233784462, 0.05287674336807661, 0.052875173821360796, 0.05287378722794315, 0.05288336338873379, 0.05287330577189535, 0.05286762940509185, 0.052868226410591114, 0.052868226410591114, 0.05286528952869956, 0.05286766792157567, 0.05286528952869956, 0.05287123551088983, 0.05286762940509185, 0.052892434020674256, 0.05286051348470543, 0.05287099959742641, 0.05288301674037938, 0.05286809160289773], "Total": [11131.0, 5363.0, 2084.0, 2314.0, 2101.0, 1432.0, 2330.0, 1356.0, 962.0, 1161.0, 860.0, 1768.0, 1050.0, 1000.0, 676.0, 866.0, 977.0, 778.0, 991.0, 1061.0, 947.0, 744.0, 857.0, 647.0, 877.0, 579.0, 492.0, 689.0, 653.0, 631.0, 11131.623884695991, 2330.626108542251, 1768.9045907436755, 929.3758384791089, 789.6210150650638, 577.7633204821094, 519.514585683373, 336.76494812913757, 298.70221880435815, 274.1457708508187, 248.69906343559097, 240.5673578842804, 153.55119190826596, 143.6834873838137, 140.07515143791798, 83.50562804669532, 64.30845621894423, 54.49581660528291, 2.319010891856696, 2.319056805895848, 2.318518935475307, 2.317570174536537, 2.3180363435057068, 2.3184396836024037, 2.317937614441368, 2.317936903380713, 2.3174504748470737, 2.3177395558165874, 2.317792847296626, 2.3173175686064003, 2.317347581201616, 2314.281077960169, 1356.1772005508699, 1061.4361435304343, 741.2149621607393, 476.6192238815708, 466.6824858378888, 458.3595822655892, 374.62856595906493, 352.63105204592085, 272.43198280072636, 163.77874468966252, 25.606813702437023, 24.585061004362636, 9.89253851284206, 2.3173808027095006, 2.319010891856696, 2.317528521357556, 2.316979736034039, 2.317004465683838, 2.3168488046904296, 2.317347581201616, 2.318518935475307, 2.317937614441368, 2.3184396836024037, 2.317081564611424, 2.3169850927149547, 2.319056805895848, 2.317792847296626, 2.3169345594821755, 2.3173851543139086, 2.3180363435057068, 2.317936903380713, 2.3177395558165874, 2.317308141642883, 1161.9002915162307, 947.2660903327204, 877.6508706414203, 740.9760994165044, 705.1072316711424, 676.7768894040354, 487.1952738687339, 295.9165206855778, 259.90323210052094, 243.4551352711713, 182.63543900405404, 178.4384847360625, 131.0583223890405, 122.08725769250545, 106.4148425738866, 92.06242950593956, 87.53377312956114, 2.319010891856696, 2.3180363435057068, 2.3178080372468686, 2.317432349421573, 2.316919514160042, 2.318518935475307, 2.317005464402502, 2.3173075155439946, 2.31747462407999, 2.316964293476539, 2.3184396836024037, 2.3170735650115626, 2.3166263177595403, 2.319056805895848, 2.317936903380713, 2.3173851543139086, 2.3177395558165874, 2101.928472880683, 1050.1765088727732, 991.6782261705699, 600.8658484566035, 501.8986764593644, 450.7254378838548, 285.99773100318066, 275.31327326098517, 258.30093123712004, 203.2139434386901, 150.60734139702268, 70.44941501849698, 32.82123778948349, 2.319056805895848, 2.3171066186392317, 2.316853709945168, 2.317936903380713, 2.3184396836024037, 2.318518935475307, 2.319010891856696, 2.3167319073053476, 2.3173175686064003, 2.317065523289807, 2.316770090324786, 2.3170163566950306, 2.3180363435057068, 2.3168738903811064, 2.3166783160313837, 2.3165524199237666, 2.3169453773979694, 2.31696811705449, 2.317183275354637, 2.3173342046121355, 2.317937614441368, 2.317528521357556, 2.3177395558165874, 2.317432349421573, 5363.195456347145, 270.6953942565073, 140.7129043297564, 117.54498817383963, 95.61081794517169, 66.471822425647, 22.305547276110435, 12.46209850747041, 2.319010891856696, 2.319056805895848, 2.3184396836024037, 2.317347581201616, 2.317792847296626, 2.3177395558165874, 2.317432349421573, 2.317937614441368, 2.316982960702426, 2.317056438780376, 2.3167767824338483, 2.316800647131277, 2.316675359233778, 2.3167574313495805, 2.317046520726426, 2.316869697216542, 2.317020192888374, 2.3173075155439946, 2.318518935475307, 2.3166264355073154, 2.316688209755262, 2.3173422769904883, 2.316964293476539, 2.3173342046121355, 2.3180363435057068, 2.3178080372468686, 2.317936903380713, 2.3172564366395676, 2.3174504748470737, 2.317570174536537, 1000.5488337661669, 977.3259141596693, 857.502096888813, 680.282755177587, 437.8108099627828, 331.086187164323, 330.0634088957393, 298.4978782138541, 243.52891297363152, 213.26435069265906, 210.15436665524354, 184.79864729723283, 76.50609736626636, 72.19906487208011, 63.828355636626966, 28.899839527369434, 2.31747462407999, 2.317936903380713, 2.316636560824466, 2.3184396836024037, 2.3167436777865147, 2.3172299876354363, 2.316853709945168, 2.3167959635085844, 2.319056805895848, 2.3180363435057068, 2.3173256652532728, 2.3173342046121355, 2.3167885455343145, 2.3177395558165874, 2.3173175686064003, 2.319010891856696, 2.318518935475307, 2.317792847296626, 689.5491105556915, 594.855180210525, 528.4175139648498, 310.0777889826334, 266.07088191437043, 229.05483379147472, 224.6559180239837, 204.87302728242952, 195.43405393224293, 153.89213969501967, 105.1348953734137, 99.30863866917798, 92.18475095441484, 86.57742788306496, 74.10716343388714, 50.945479607658335, 2.3166383209079258, 2.316500169970604, 2.3165078284798244, 2.316554463343817, 2.316566084208592, 2.3167122088464613, 2.31651681825018, 2.3173851543139086, 2.316506343578347, 2.3166045032279308, 2.3165305657657798, 2.3168090306605227, 2.316592082151274, 2.316701591935687, 2.3171736080839307, 2.317081564611424, 2.3170735650115626, 2.3169042635216166, 2.317308141642883, 2.317017284158625, 2.3174504748470737, 2.3180363435057068, 2.3173256652532728, 2.3177395558165874, 2.3172564366395676, 2.319056805895848, 2.318518935475307, 962.895576807638, 866.4878842543524, 584.9827475994064, 445.05077320473566, 412.30021927582436, 405.55351342598243, 243.83326443012513, 69.89492916008082, 2.3178080372468686, 2.316788486348993, 2.3166072585401563, 2.316655400718263, 2.3169162878778153, 2.3164987659512475, 2.3164987659512475, 2.316626267585851, 2.317067938109603, 2.3165767592580164, 2.3165867221060665, 2.3167270025196207, 2.316566084208592, 2.316541670519158, 2.3166031579575392, 2.3167436777865147, 2.3165357278608076, 2.3166373237033735, 2.316522729983523, 2.3165721103353856, 2.3171869940668612, 2.3165118619696723, 2.3168642592492317, 2.3169271810910477, 2.3173256652532728, 2.3174504748470737, 2.3169854921616597, 2.3173013444014674, 2.316780394345948, 2.31747462407999, 2.318518935475307, 2.3177395558165874, 2.317528521357556, 2.317937614441368, 2.319056805895848, 2.3180363435057068, 2.3184396836024037, 2.319010891856696, 653.9695341911772, 631.3182528266543, 564.2060928375893, 542.3820638063488, 359.4732876953012, 336.24085558362657, 157.2197919534702, 153.97946044620323, 134.3011420836672, 99.03729040768455, 83.12389662327901, 66.34511781360412, 58.72409104745278, 2.3173808027095006, 2.3166487958286672, 2.3165767264545627, 2.3165767938377, 2.317570174536537, 2.316750356121742, 2.3165802707097787, 2.3165540655528316, 2.317067938109603, 2.316510524812065, 2.316527780477167, 2.31655662325194, 2.316622967160147, 2.3169088357183, 2.3166962146200487, 2.316527431868939, 2.3167687690707313, 2.318518935475307, 2.3180363435057068, 2.3177395558165874, 2.3173851543139086, 2.317046520726426, 2.319056805895848, 2.3170163566950306, 2.317936903380713, 2.3184396836024037, 2.317528521357556, 2084.2703941430036, 744.2957485987924, 159.07924746058424, 144.3300335665819, 99.96127185043257, 69.55245023115043, 49.70745129737417, 48.936851757654416, 2.31696811705449, 2.3166552730391987, 2.3169104550016146, 2.3165960453737604, 2.3165845465192203, 2.3173256652532728, 2.3166852109381257, 2.316587570472397, 2.317020192888374, 2.3165036551903935, 2.3165036551903935, 2.316567936006128, 2.31655238627628, 2.3167206946805528, 2.316634000883856, 2.317432349421573, 2.3169942245629707, 2.3167133551434693, 2.3165913887838836, 2.316538385810156, 2.316702680661, 2.316509075539197, 2.316532675538957, 2.317936903380713, 2.319010891856696, 2.316944750114863, 2.3169453773979694, 2.3171325531754494, 2.317528521357556, 2.319056805895848, 2.318518935475307, 2.3184396836024037, 2.3178080372468686, 2.3180363435057068, 430.5477806301812, 406.9295004390394, 399.96698060055263, 312.0545604800798, 276.31162822574095, 226.03312874185005, 191.87195132704068, 129.54294431809964, 95.89941318328614, 71.41468219986368, 63.519309483195514, 60.8295358220071, 59.08264855325188, 54.96133770329648, 21.561461540671278, 2.316876389072685, 2.316593154129241, 2.3165728591018184, 2.3173256652532728, 2.3165945209055, 2.316755214498501, 2.31747462407999, 2.316525049497618, 2.3166375787898508, 2.3165438356261494, 2.316528482169576, 2.316636703324562, 2.317005464402502, 2.3165397877810365, 2.316650337946032, 2.3166218627495185, 2.3168089634239726, 2.317792847296626, 2.317937614441368, 2.3173422769904883, 2.319056805895848, 2.317936903380713, 2.3171800584180824, 2.319010891856696, 2.317570174536537, 376.97598858165105, 327.09509700943505, 324.29436150476585, 222.39320767574753, 204.62351239417328, 185.40273089784597, 174.2433943803787, 139.3062017327108, 133.9071408586465, 123.70549573111296, 119.52287071029349, 103.35442810626627, 99.52478729975329, 84.86245800279355, 81.97302134154549, 71.07945098720407, 53.79198563481866, 20.467338896340653, 17.823153552984508, 5.685099842279997, 2.3170510110703066, 2.3166225276585397, 2.31662217058246, 2.316671869746437, 2.3171705887669325, 2.316666128487107, 2.316622727812923, 2.3170163566950306, 2.316474208776888, 2.316982737672934, 2.317183275354637, 2.3184396836024037, 2.3173013444014674, 1432.2484661110068, 348.9667071679741, 344.17026059346176, 275.46358526295836, 255.4250013294963, 116.79191671598642, 61.06731586648333, 26.42826444014206, 2.316564031618143, 2.316610524473578, 2.316649655511518, 2.3167405914075245, 2.316851037704983, 2.316565096128608, 2.3165236872703, 2.316569148686393, 2.316668266507417, 2.316517474547579, 2.316543336544822, 2.317183275354637, 2.316531368465964, 2.3169271810910477, 2.316520934196245, 2.3168532691439143, 2.316527310178843, 2.316683245518855, 2.3166783160313837, 2.316694339800774, 2.3166044861142465, 2.3165206921073787, 2.3166058087562544, 2.317570174536537, 2.316869502164965, 2.3184396836024037, 2.3170163566950306, 2.317936903380713, 2.3174504748470737, 2.3180363435057068, 2.3171325531754494, 2.319010891856696, 2.3173175686064003, 2.3178080372468686, 2.317937614441368, 860.573019194349, 778.9360034025318, 382.20262798235444, 239.33020322480203, 229.89304528092538, 140.3482405058081, 78.16905618693205, 77.68535493705495, 73.97856499765133, 72.25368185050458, 32.958495870430895, 25.191366643351092, 22.903616374697812, 16.451896714704883, 8.27637794250455, 4.894807792715618, 4.2400344616019945, 2.318518935475307, 2.316800647131277, 2.317046520726426, 2.3171869940668612, 2.3170510110703066, 2.3169067657345477, 2.3178080372468686, 2.317937614441368, 2.319010891856696, 2.3167993871199806, 2.31662217058246, 2.3166225276585397, 2.3167417589957924, 2.317792847296626, 2.3184396836024037, 2.319056805895848, 451.045039792679, 255.24932451212845, 244.48365024406186, 236.18911488633358, 231.2232422380174, 210.95762869618758, 139.99493245892683, 137.00076721107175, 125.62004218279075, 122.92410912343607, 87.69224127095819, 33.35084903750001, 2.316482432114103, 2.316566084208592, 2.3167436777865147, 2.3164890988108175, 2.3165357278608076, 2.316614799706981, 2.316538385810156, 2.316723882914342, 2.3165908251146257, 2.3165120062955995, 2.3166186035296352, 2.316668266507417, 2.3164575652795274, 2.3166463863114606, 2.316651922398366, 2.3164959894583697, 2.3165046298565333, 2.316535880617579, 2.316734110041382, 2.3166859958474477, 2.3169088357183, 2.3178080372468686, 2.317183275354637, 2.319010891856696, 2.317528521357556, 2.319056805895848, 2.317065523289807, 2.3177395558165874, 2.3173256652532728, 2.3171705887669325, 647.7917584123425, 323.7082225555121, 234.17103449288194, 174.6588517108806, 149.51146231767282, 147.30625430316854, 123.83198890518712, 119.53361959840545, 43.98728690647911, 36.80974801620041, 34.85250723206917, 13.362185943353785, 3.681461216288097, 2.316610548943389, 2.3166758186179415, 2.31669627303186, 2.3164904860382802, 2.316492806369212, 2.316595527752736, 2.317057711064455, 2.3165508299182265, 2.316549093698527, 2.316670729863309, 2.317056833300933, 2.3165305657657798, 2.316545054946247, 2.31650975785803, 2.3165865925410527, 2.316592082151274, 2.3164870756699143, 2.317017284158625, 2.319056805895848, 2.317308141642883, 2.3171705887669325, 2.3171376322355415, 2.3168286616375453, 2.319010891856696, 2.318518935475307, 579.7866326344616, 508.5261938491418, 158.91189684744631, 146.71841198101558, 123.34878177993603, 81.45105377554282, 43.24313320061999, 2.3168050442534125, 2.3166021527205065, 2.3166327333483787, 2.316573228343517, 2.316634348919335, 2.316587570472397, 2.316531483874838, 2.316532179182712, 2.3165976999992663, 2.3171066186392317, 2.3169339885173295, 2.3165082789921745, 2.3170510110703066, 2.316944750114863, 2.316539403253796, 2.3164814277065897, 2.3165765351783136, 2.3166054662674544, 2.3164699452167334, 2.316491507361018, 2.3167767824338483, 2.31651043809265, 2.3164634312108503, 2.316530940410124, 2.3169271810910477, 2.3169754341704576, 2.319056805895848, 2.316807959156955, 2.317792847296626, 2.3169528512125677, 2.317937614441368, 2.319010891856696, 2.317528521357556, 676.4043371420536, 276.59766986741323, 237.76823581720552, 2.316610524473578, 2.3165234284467466, 2.3165397477536933, 2.3168286616375453, 2.3166054662674544, 2.3166229807765855, 2.316666128487107, 2.3165304052305022, 2.3165328792614797, 2.3165448046034265, 2.3165524129061055, 2.316671869746437, 2.316804746417602, 2.316603195741903, 2.3165035583721973, 2.3164925838463706, 2.3164991283273033, 2.316753790051336, 2.3166010160130104, 2.3164660645992865, 2.316530940410124, 2.31651453182549, 2.316545032516599, 2.316483983132777, 2.3165893016683166, 2.316552037094259, 2.3164922410280364, 2.3167160832969107, 2.3171705887669325, 2.317528521357556, 2.3173422769904883, 2.316764336991943, 2.3170163566950306, 2.317937614441368, 2.3173013444014674, 2.318518935475307, 2.319010891856696, 2.317347581201616, 2.3180363435057068, 2.3177395558165874, 376.2038716576609, 227.34181574453066, 202.92918194594185, 192.5125372073912, 177.32904397101873, 6.876359857342147, 2.316571149334256, 2.3165767592580164, 2.3165068478316075, 2.3165131320322576, 2.316982737672934, 2.3165185855141406, 2.316528801313441, 2.31662565453436, 2.3166263177595403, 2.316482838158703, 2.316788486348993, 2.3167270025196207, 2.316500204107804, 2.3166364199353717, 2.3169345594821755, 2.3164987659512475, 2.3164987659512475, 2.316596798007458, 2.316864761887598, 2.317043852830591, 2.316655400718263, 2.3164950182002144, 2.316602144366401, 2.316492459581783, 2.3174504748470737, 2.3168089634239726, 2.3178080372468686, 2.3184396836024037, 2.317067938109603, 2.3167624187505655, 2.3169453773979694, 492.15940991852585, 258.9164613298162, 132.76754959685064, 2.317005464402502, 2.3165865925410527, 2.316734007533867, 2.317056833300933, 2.317792847296626, 2.3165277432523443, 2.31747462407999, 2.316610548943389, 2.31669627303186, 2.3164660930534597, 2.3164993275342534, 2.3166758186179415, 2.316760156492418, 2.316701591935687, 2.3166862189514927, 2.3171376322355415, 2.3167122088464613, 2.3165690847778238, 2.3166100642122633, 2.3166229807765855, 2.31651681825018, 2.3166288120007854, 2.316545054946247, 2.316818013672757, 2.3166947983016057, 2.3178080372468686, 2.316530940410124, 2.3169939605932917, 2.319056805895848, 2.3170735650115626], "Category": ["Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic11", "Topic11", "Topic11", "Topic11", "Topic11", "Topic11", "Topic11", "Topic11", "Topic11", "Topic11", "Topic11", "Topic11", "Topic11", "Topic11", "Topic11", "Topic11", "Topic11", "Topic11", "Topic11", "Topic11", "Topic11", "Topic11", "Topic11", "Topic11", "Topic11", "Topic11", "Topic11", "Topic11", "Topic11", "Topic11", "Topic11", "Topic11", "Topic11", "Topic11", "Topic11", "Topic11", "Topic11", "Topic11", "Topic11", "Topic11", "Topic12", "Topic12", "Topic12", "Topic12", "Topic12", "Topic12", "Topic12", "Topic12", "Topic12", "Topic12", "Topic12", "Topic12", "Topic12", "Topic12", "Topic12", "Topic12", "Topic12", "Topic12", "Topic12", "Topic12", "Topic12", "Topic12", "Topic12", "Topic12", "Topic12", "Topic12", "Topic12", "Topic12", "Topic12", "Topic12", "Topic12", "Topic12", "Topic12", "Topic13", "Topic13", "Topic13", "Topic13", "Topic13", "Topic13", "Topic13", "Topic13", "Topic13", "Topic13", "Topic13", "Topic13", "Topic13", "Topic13", "Topic13", "Topic13", "Topic13", "Topic13", "Topic13", "Topic13", "Topic13", "Topic13", "Topic13", "Topic13", "Topic13", "Topic13", "Topic13", "Topic13", "Topic13", "Topic13", "Topic13", "Topic13", "Topic13", "Topic13", "Topic13", "Topic13", "Topic13", "Topic13", "Topic13", "Topic13", "Topic13", "Topic13", "Topic13", "Topic14", "Topic14", "Topic14", "Topic14", "Topic14", "Topic14", "Topic14", "Topic14", "Topic14", "Topic14", "Topic14", "Topic14", "Topic14", "Topic14", "Topic14", "Topic14", "Topic14", "Topic14", "Topic14", "Topic14", "Topic14", "Topic14", "Topic14", "Topic14", "Topic14", "Topic14", "Topic14", "Topic14", "Topic14", "Topic14", "Topic14", "Topic14", "Topic14", "Topic15", "Topic15", "Topic15", "Topic15", "Topic15", "Topic15", "Topic15", "Topic15", "Topic15", "Topic15", "Topic15", "Topic15", "Topic15", "Topic15", "Topic15", "Topic15", "Topic15", "Topic15", "Topic15", "Topic15", "Topic15", "Topic15", "Topic15", "Topic15", "Topic15", "Topic15", "Topic15", "Topic15", "Topic15", "Topic15", "Topic15", "Topic15", "Topic15", "Topic15", "Topic15", "Topic15", "Topic15", "Topic15", "Topic15", "Topic15", "Topic15", "Topic15", "Topic16", "Topic16", "Topic16", "Topic16", "Topic16", "Topic16", "Topic16", "Topic16", "Topic16", "Topic16", "Topic16", "Topic16", "Topic16", "Topic16", "Topic16", "Topic16", "Topic16", "Topic16", "Topic16", "Topic16", "Topic16", "Topic16", "Topic16", "Topic16", "Topic16", "Topic16", "Topic16", "Topic16", "Topic16", "Topic16", "Topic16", "Topic16", "Topic16", "Topic16", "Topic16", "Topic16", "Topic16", "Topic16", "Topic17", "Topic17", "Topic17", "Topic17", "Topic17", "Topic17", "Topic17", "Topic17", "Topic17", "Topic17", "Topic17", "Topic17", "Topic17", "Topic17", "Topic17", "Topic17", "Topic17", "Topic17", "Topic17", "Topic17", "Topic17", "Topic17", "Topic17", "Topic17", "Topic17", "Topic17", "Topic17", "Topic17", "Topic17", "Topic17", "Topic17", "Topic17", "Topic17", "Topic17", "Topic17", "Topic17", "Topic17", "Topic17", "Topic17", "Topic17", "Topic18", "Topic18", "Topic18", "Topic18", "Topic18", "Topic18", "Topic18", "Topic18", "Topic18", "Topic18", "Topic18", "Topic18", "Topic18", "Topic18", "Topic18", "Topic18", "Topic18", "Topic18", "Topic18", "Topic18", "Topic18", "Topic18", "Topic18", "Topic18", "Topic18", "Topic18", "Topic18", "Topic18", "Topic18", "Topic18", "Topic18", "Topic18", "Topic18", "Topic18", "Topic18", "Topic18", "Topic18", "Topic18", "Topic18", "Topic18", "Topic18", "Topic18", "Topic18", "Topic19", "Topic19", "Topic19", "Topic19", "Topic19", "Topic19", "Topic19", "Topic19", "Topic19", "Topic19", "Topic19", "Topic19", "Topic19", "Topic19", "Topic19", "Topic19", "Topic19", "Topic19", "Topic19", "Topic19", "Topic19", "Topic19", "Topic19", "Topic19", "Topic19", "Topic19", "Topic19", "Topic19", "Topic19", "Topic19", "Topic19", "Topic19", "Topic19", "Topic19", "Topic19", "Topic19", "Topic19", "Topic20", "Topic20", "Topic20", "Topic20", "Topic20", "Topic20", "Topic20", "Topic20", "Topic20", "Topic20", "Topic20", "Topic20", "Topic20", "Topic20", "Topic20", "Topic20", "Topic20", "Topic20", "Topic20", "Topic20", "Topic20", "Topic20", "Topic20", "Topic20", "Topic20", "Topic20", "Topic20", "Topic20", "Topic20", "Topic20", "Topic20", "Topic20", "Topic20"], "logprob": [30.0, 29.0, 28.0, 27.0, 26.0, 25.0, 24.0, 23.0, 22.0, 21.0, 20.0, 19.0, 18.0, 17.0, 16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, -0.653, -2.2174, -2.4935, -3.1382, -3.3016, -3.615, -3.7216, -4.1574, -4.2782, -4.3646, -4.4628, -4.4964, -4.9505, -5.0179, -5.0437, -5.5716, -5.8408, -6.0127, -11.809, -11.809, -11.8094, -11.8101, -11.8099, -11.81, -11.8102, -11.8103, -11.8106, -11.8106, -11.8106, -11.8108, -11.8108, -1.4189, -1.954, -2.1995, -2.5595, -3.0026, -3.0238, -3.0419, -3.2446, -3.3055, -3.5654, -4.0795, -6.0093, -6.0538, -7.1169, -10.8992, -10.8986, -10.8993, -10.8997, -10.8997, -10.8998, -10.8996, -10.8991, -10.8994, -10.8991, -10.8997, -10.8998, -10.8989, -10.8995, -10.8999, -10.8997, -10.8994, -10.8995, -10.8996, -10.8997, -1.9859, -2.1905, -2.267, -2.4368, -2.4865, -2.5277, -2.8576, -3.3591, -3.4899, -3.5558, -3.8463, -3.8698, -4.1829, -4.255, -4.3951, -4.5432, -4.5949, -10.9979, -10.9988, -10.9989, -10.9991, -10.9994, -10.9988, -10.9996, -10.9995, -10.9994, -10.9997, -10.9991, -10.9997, -10.9999, -10.9988, -10.9993, -10.9996, -10.9996, -1.3338, -2.0288, -2.0862, -2.5887, -2.7694, -2.8774, -3.3351, -3.3735, -3.4378, -3.68, -3.9834, -4.7601, -5.5615, -11.0896, -11.0909, -11.0913, -11.0909, -11.0907, -11.0907, -11.0905, -11.0915, -11.0913, -11.0914, -11.0916, -11.0915, -11.0911, -11.0916, -11.0917, -11.0917, -11.0916, -11.0916, -11.0915, -11.0914, -11.0913, -11.0914, -11.0914, -11.0915, -0.2962, -3.2902, -3.952, -4.135, -4.3459, -4.7197, -5.8812, -6.5528, -10.8856, -10.8856, -10.886, -10.8866, -10.8864, -10.8865, -10.8866, -10.8865, -10.887, -10.887, -10.8871, -10.8871, -10.8871, -10.8871, -10.887, -10.8871, -10.887, -10.8869, -10.8864, -10.8872, -10.8872, -10.8869, -10.8871, -10.8869, -10.8868, -10.8869, -10.8869, -10.887, -10.887, -10.8871, -1.9527, -1.9762, -2.1073, -2.3395, -2.782, -3.063, -3.0662, -3.1674, -3.3726, -3.5066, -3.5214, -3.6514, -4.5505, -4.6102, -4.7375, -5.5738, -10.9267, -10.9266, -10.9272, -10.9266, -10.9275, -10.9274, -10.9276, -10.9276, -10.9267, -10.9272, -10.9275, -10.9275, -10.9278, -10.9274, -10.9276, -10.9271, -10.9274, -10.9276, -1.9686, -2.1169, -2.2358, -2.7718, -2.926, -3.0771, -3.0967, -3.1898, -3.2375, -3.4796, -3.8673, -3.9256, -4.0018, -4.0661, -4.226, -4.6147, -10.5825, -10.5827, -10.5827, -10.5827, -10.5827, -10.5826, -10.5827, -10.5823, -10.5827, -10.5827, -10.5827, -10.5826, -10.5827, -10.5826, -10.5824, -10.5825, -10.5825, -10.5826, -10.5825, -10.5826, -10.5825, -10.5824, -10.5825, -10.5825, -10.5826, -10.5824, -10.5825, -1.6249, -1.7307, -2.1248, -2.3994, -2.4762, -2.4928, -3.0052, -4.2777, -10.7241, -10.7248, -10.725, -10.725, -10.7249, -10.7251, -10.7251, -10.725, -10.7249, -10.7251, -10.7251, -10.725, -10.7251, -10.7251, -10.7251, -10.7251, -10.7251, -10.7251, -10.7252, -10.7252, -10.7249, -10.7252, -10.725, -10.725, -10.7249, -10.7249, -10.725, -10.725, -10.7251, -10.725, -10.7249, -10.725, -10.725, -10.725, -10.7249, -10.725, -10.725, -10.725, -2.0105, -2.0459, -2.1587, -2.1983, -2.6117, -2.6789, -3.4466, -3.4677, -3.6066, -3.9171, -4.0966, -4.3289, -4.4554, -10.5681, -10.5685, -10.5686, -10.5686, -10.5683, -10.5687, -10.5688, -10.5688, -10.5686, -10.5689, -10.5689, -10.5688, -10.5688, -10.5687, -10.5688, -10.5689, -10.5688, -10.5682, -10.5685, -10.5686, -10.5687, -10.5688, -10.5685, -10.5688, -10.5687, -10.5687, -10.5687, -0.7575, -1.7892, -3.3431, -3.4418, -3.816, -4.1886, -4.5376, -4.5539, -10.4743, -10.4744, -10.4743, -10.4746, -10.4746, -10.4743, -10.4746, -10.4746, -10.4744, -10.4747, -10.4747, -10.4746, -10.4747, -10.4746, -10.4746, -10.4743, -10.4745, -10.4746, -10.4747, -10.4747, -10.4746, -10.4747, -10.4747, -10.4742, -10.4738, -10.4746, -10.4746, -10.4746, -10.4745, -10.4743, -10.4744, -10.4744, -10.4746, -10.4746, -2.1772, -2.2339, -2.2513, -2.5011, -2.6236, -2.8262, -2.9919, -3.3903, -3.697, -3.9999, -4.121, -4.1659, -4.1961, -4.2713, -5.2737, -10.3577, -10.3579, -10.3579, -10.3576, -10.3579, -10.3579, -10.3576, -10.358, -10.3579, -10.358, -10.358, -10.358, -10.3578, -10.358, -10.358, -10.358, -10.3579, -10.3576, -10.3576, -10.3578, -10.3575, -10.3578, -10.3579, -10.3578, -10.3579, -2.3094, -2.4523, -2.4609, -2.8413, -2.9254, -3.0252, -3.088, -3.315, -3.3552, -3.4358, -3.4709, -3.6191, -3.6577, -3.821, -3.8566, -4.0034, -4.2925, -5.3309, -5.4873, -6.9884, -10.4204, -10.4207, -10.4207, -10.4208, -10.4207, -10.4209, -10.421, -10.4208, -10.4211, -10.4209, -10.4208, -10.4205, -10.4209, -0.9618, -2.3787, -2.3926, -2.6169, -2.693, -3.486, -4.1522, -5.0402, -10.4667, -10.4667, -10.4667, -10.4667, -10.4666, -10.4668, -10.4668, -10.4668, -10.4668, -10.4669, -10.4669, -10.4666, -10.4669, -10.4667, -10.4669, -10.4668, -10.4669, -10.4669, -10.4669, -10.4669, -10.4669, -10.4669, -10.4669, -10.4665, -10.4668, -10.4663, -10.4668, -10.4667, -10.4667, -10.4667, -10.4668, -10.4668, -10.4668, -10.4668, -10.4668, -1.4661, -1.5661, -2.2811, -2.7527, -2.7933, -3.2931, -3.8912, -3.8976, -3.948, -3.9723, -4.796, -5.0874, -5.1924, -5.5665, -6.4221, -7.2419, -7.5242, -10.6959, -10.6969, -10.6969, -10.6969, -10.697, -10.6971, -10.6968, -10.6967, -10.6963, -10.6973, -10.6974, -10.6974, -10.6974, -10.697, -10.6969, -10.6969, -1.9272, -2.5003, -2.5438, -2.5786, -2.6001, -2.6928, -3.1082, -3.1302, -3.2184, -3.2405, -3.5856, -4.5956, -10.347, -10.3471, -10.347, -10.3471, -10.3471, -10.3471, -10.3471, -10.3471, -10.3472, -10.3472, -10.3472, -10.3471, -10.3472, -10.3472, -10.3472, -10.3472, -10.3472, -10.3472, -10.3472, -10.3472, -10.3471, -10.3468, -10.347, -10.3468, -10.347, -10.3469, -10.3471, -10.3471, -10.3471, -10.3471, -1.5403, -2.2374, -2.5638, -2.8603, -3.0179, -3.033, -3.2094, -3.2454, -4.2778, -4.4663, -4.5245, -5.5978, -7.617, -10.1549, -10.1549, -10.1549, -10.155, -10.155, -10.155, -10.1548, -10.155, -10.155, -10.155, -10.1548, -10.1551, -10.1551, -10.1551, -10.1551, -10.1551, -10.1552, -10.1549, -10.1543, -10.1549, -10.155, -10.155, -10.1551, -10.1549, -10.1549, -1.4643, -1.596, -2.7688, -2.8498, -3.0262, -3.4507, -4.1088, -10.0934, -10.0937, -10.0938, -10.0938, -10.0938, -10.0938, -10.0939, -10.0939, -10.0939, -10.0936, -10.0938, -10.094, -10.0937, -10.0938, -10.094, -10.094, -10.094, -10.094, -10.0941, -10.094, -10.0939, -10.0941, -10.0941, -10.094, -10.0939, -10.0939, -10.0934, -10.094, -10.0938, -10.0939, -10.0939, -10.0938, -10.0939, -1.0791, -1.9781, -2.1307, -9.9497, -9.9498, -9.9498, -9.9496, -9.9498, -9.9498, -9.9497, -9.9498, -9.9498, -9.9498, -9.9498, -9.9498, -9.9497, -9.9498, -9.9499, -9.9499, -9.9499, -9.9498, -9.9499, -9.95, -9.95, -9.95, -9.95, -9.95, -9.95, -9.95, -9.95, -9.9499, -9.9498, -9.9497, -9.9498, -9.9499, -9.9499, -9.9497, -9.9498, -9.9497, -9.9498, -9.9499, -9.9499, -9.9499, -1.6246, -2.1322, -2.247, -2.3003, -2.3834, -6.0127, -9.9983, -9.9983, -9.9983, -9.9984, -9.9982, -9.9984, -9.9984, -9.9983, -9.9984, -9.9984, -9.9983, -9.9984, -9.9985, -9.9984, -9.9983, -9.9985, -9.9985, -9.9985, -9.9983, -9.9983, -9.9985, -9.9986, -9.9985, -9.9986, -9.9982, -9.9984, -9.9982, -9.9983, -9.9984, -9.9985, -9.9985, -0.9938, -1.6402, -2.3166, -10.1274, -10.1276, -10.1276, -10.1275, -10.1272, -10.1277, -10.1274, -10.1278, -10.1278, -10.1279, -10.1279, -10.1278, -10.1278, -10.1278, -10.1278, -10.1276, -10.1278, -10.1279, -10.1279, -10.1279, -10.128, -10.1279, -10.128, -10.1279, -10.1279, -10.1275, -10.1281, -10.1279, -10.1276, -10.1279], "loglift": [30.0, 29.0, 28.0, 27.0, 26.0, 25.0, 24.0, 23.0, 22.0, 21.0, 20.0, 19.0, 18.0, 17.0, 16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 1.6333, 1.6326, 1.6323, 1.6312, 1.6308, 1.6298, 1.6293, 1.6271, 1.6263, 1.6256, 1.6248, 1.6245, 1.6194, 1.6184, 1.618, 1.6073, 1.5994, 1.5931, -1.0463, -1.0463, -1.0465, -1.0467, -1.0467, -1.047, -1.047, -1.0471, -1.0472, -1.0473, -1.0473, -1.0474, -1.0474, 2.4381, 2.4374, 2.437, 2.4361, 2.4345, 2.4344, 2.4343, 2.4333, 2.4329, 2.4311, 2.4259, 2.3517, 2.348, 2.1952, -0.1358, -0.1359, -0.1359, -0.1361, -0.1361, -0.1361, -0.1361, -0.1361, -0.1362, -0.1362, -0.1362, -0.1362, -0.1362, -0.1362, -0.1362, -0.1362, -0.1363, -0.1363, -0.1363, -0.1363, 2.5602, 2.5598, 2.5596, 2.5592, 2.559, 2.5589, 2.5576, 2.5547, 2.5537, 2.5531, 2.5501, 2.5498, 2.5454, 2.5441, 2.5414, 2.5382, 2.5369, -0.2352, -0.2357, -0.2357, -0.2357, -0.2357, -0.2358, -0.236, -0.236, -0.236, -0.2361, -0.2361, -0.2361, -0.2361, -0.2361, -0.2361, -0.2361, -0.2363, 2.6195, 2.6184, 2.6183, 2.6168, 2.6161, 2.6156, 2.6128, 2.6125, 2.612, 2.6096, 2.6058, 2.5888, 2.5513, -0.3269, -0.3274, -0.3276, -0.3277, -0.3277, -0.3277, -0.3278, -0.3278, -0.3278, -0.3278, -0.3279, -0.3279, -0.3279, -0.3279, -0.3279, -0.3279, -0.3279, -0.3279, -0.3279, -0.328, -0.3281, -0.328, -0.3281, -0.328, 2.7204, 2.7127, 2.7052, 2.7021, 2.6977, 2.6874, 2.6179, 2.5284, -0.1228, -0.1229, -0.123, -0.1231, -0.1231, -0.1232, -0.1232, -0.1233, -0.1233, -0.1234, -0.1234, -0.1234, -0.1234, -0.1234, -0.1234, -0.1234, -0.1234, -0.1234, -0.1234, -0.1234, -0.1235, -0.1235, -0.1235, -0.1235, -0.1236, -0.1236, -0.1237, -0.1235, -0.1236, -0.1237, 2.7429, 2.7428, 2.7425, 2.7419, 2.7401, 2.7385, 2.7384, 2.7377, 2.7361, 2.7348, 2.7346, 2.7332, 2.716, 2.7143, 2.7102, 2.6663, -0.1633, -0.1634, -0.1635, -0.1636, -0.1638, -0.1639, -0.1639, -0.1639, -0.164, -0.164, -0.164, -0.1641, -0.1641, -0.1641, -0.1641, -0.1644, -0.1644, -0.1643, 3.0992, 3.0987, 3.0982, 3.0953, 3.0941, 3.0928, 3.0926, 3.0916, 3.0911, 3.0881, 3.0813, 3.0801, 3.0783, 3.0768, 3.0724, 3.0584, 0.1812, 0.1812, 0.1812, 0.1811, 0.1811, 0.1811, 0.1811, 0.1811, 0.1811, 0.1811, 0.1811, 0.1811, 0.1811, 0.1811, 0.1811, 0.1811, 0.1811, 0.1811, 0.181, 0.181, 0.1809, 0.1807, 0.1809, 0.1808, 0.1809, 0.1803, 0.1804, 3.109, 3.1087, 3.1075, 3.1063, 3.1059, 3.1058, 3.1022, 3.0792, 0.0392, 0.0389, 0.0388, 0.0388, 0.0387, 0.0387, 0.0387, 0.0387, 0.0387, 0.0387, 0.0387, 0.0387, 0.0387, 0.0387, 0.0387, 0.0387, 0.0387, 0.0387, 0.0387, 0.0386, 0.0386, 0.0386, 0.0386, 0.0386, 0.0386, 0.0385, 0.0386, 0.0385, 0.0386, 0.0384, 0.0381, 0.0383, 0.0384, 0.0382, 0.0378, 0.0382, 0.038, 0.0377, 3.1103, 3.1102, 3.1098, 3.1096, 3.1076, 3.1071, 3.0996, 3.0993, 3.0972, 3.0913, 3.087, 3.0801, 3.0757, 0.1954, 0.1952, 0.1952, 0.1952, 0.195, 0.195, 0.195, 0.195, 0.195, 0.195, 0.195, 0.195, 0.1949, 0.1949, 0.1949, 0.1949, 0.1949, 0.1948, 0.1947, 0.1947, 0.1948, 0.1948, 0.1942, 0.1948, 0.1945, 0.1943, 0.1946, 3.2042, 3.2023, 3.1914, 3.1899, 3.1831, 3.1732, 3.1602, 3.1594, 0.2893, 0.2893, 0.2893, 0.2892, 0.2892, 0.2892, 0.2892, 0.2892, 0.2892, 0.2892, 0.2892, 0.2892, 0.2891, 0.2891, 0.2891, 0.2891, 0.2891, 0.2891, 0.2891, 0.2891, 0.2891, 0.2891, 0.2891, 0.289, 0.2889, 0.2891, 0.289, 0.289, 0.2889, 0.2884, 0.2885, 0.2886, 0.2887, 0.2886, 3.3616, 3.3613, 3.3612, 3.3597, 3.3587, 3.3569, 3.3552, 3.3496, 3.3435, 3.3355, 3.3315, 3.3299, 3.3288, 3.3259, 3.2593, 0.4059, 0.4059, 0.4059, 0.4059, 0.4059, 0.4058, 0.4058, 0.4058, 0.4058, 0.4058, 0.4058, 0.4058, 0.4058, 0.4058, 0.4058, 0.4058, 0.4058, 0.4056, 0.4056, 0.4057, 0.4052, 0.4054, 0.4056, 0.405, 0.4055, 3.3623, 3.3614, 3.3613, 3.3582, 3.3573, 3.3562, 3.3554, 3.3522, 3.3515, 3.3502, 3.3495, 3.3466, 3.3457, 3.3418, 3.3409, 3.3366, 3.3263, 3.2542, 3.2361, 2.8776, 0.3432, 0.3431, 0.3431, 0.343, 0.3428, 0.3428, 0.3428, 0.3428, 0.3428, 0.3427, 0.3427, 0.3425, 0.3426, 3.3751, 3.3702, 3.3702, 3.3685, 3.3679, 3.3575, 3.3397, 3.2893, 0.2971, 0.2971, 0.297, 0.297, 0.297, 0.297, 0.297, 0.297, 0.297, 0.2969, 0.2969, 0.2969, 0.2969, 0.2969, 0.2969, 0.2969, 0.2969, 0.2969, 0.2969, 0.2969, 0.2969, 0.2969, 0.2969, 0.2968, 0.2969, 0.2967, 0.2968, 0.2965, 0.2967, 0.2965, 0.2967, 0.296, 0.2966, 0.2965, 0.2964, 3.3801, 3.3799, 3.3769, 3.3734, 3.373, 3.3667, 3.3538, 3.3536, 3.3521, 3.3513, 3.3126, 3.2899, 3.2802, 3.2369, 3.0684, 2.7738, 2.6351, 0.0671, 0.0668, 0.0667, 0.0667, 0.0666, 0.0666, 0.0665, 0.0665, 0.0665, 0.0664, 0.0664, 0.0664, 0.0663, 0.0663, 0.066, 0.0658, 3.5651, 3.5613, 3.5609, 3.5606, 3.5604, 3.5595, 3.5541, 3.5537, 3.5522, 3.5518, 3.5444, 3.5012, 0.4168, 0.4167, 0.4167, 0.4167, 0.4167, 0.4167, 0.4167, 0.4166, 0.4166, 0.4166, 0.4166, 0.4166, 0.4166, 0.4166, 0.4166, 0.4166, 0.4166, 0.4166, 0.4166, 0.4166, 0.4165, 0.4164, 0.4165, 0.416, 0.4163, 0.4158, 0.4165, 0.4162, 0.4163, 0.4164, 3.5901, 3.5866, 3.584, 3.5808, 3.5786, 3.5784, 3.5755, 3.5749, 3.5422, 3.5318, 3.5283, 3.4136, 2.6836, 0.6089, 0.6089, 0.6088, 0.6088, 0.6088, 0.6088, 0.6088, 0.6088, 0.6088, 0.6087, 0.6087, 0.6087, 0.6087, 0.6087, 0.6087, 0.6087, 0.6087, 0.6087, 0.6084, 0.6085, 0.6085, 0.6085, 0.6086, 0.6079, 0.608, 3.7769, 3.7764, 3.7667, 3.7656, 3.7627, 3.7532, 3.7282, 0.6703, 0.6701, 0.67, 0.67, 0.67, 0.67, 0.67, 0.67, 0.6699, 0.6699, 0.6699, 0.6699, 0.6699, 0.6698, 0.6698, 0.6698, 0.6698, 0.6698, 0.6698, 0.6698, 0.6698, 0.6698, 0.6698, 0.6698, 0.6697, 0.6697, 0.6693, 0.6697, 0.6695, 0.6697, 0.6693, 0.6689, 0.6695, 4.008, 4.0032, 4.0019, 0.8141, 0.8141, 0.8141, 0.814, 0.814, 0.814, 0.814, 0.814, 0.814, 0.814, 0.814, 0.814, 0.814, 0.814, 0.814, 0.8139, 0.8139, 0.8139, 0.8139, 0.8139, 0.8139, 0.8138, 0.8138, 0.8138, 0.8138, 0.8138, 0.8138, 0.8138, 0.8138, 0.8137, 0.8137, 0.8138, 0.8137, 0.8135, 0.8137, 0.8132, 0.813, 0.8136, 0.8133, 0.8134, 4.0492, 4.0452, 4.0441, 4.0434, 4.0424, 3.6631, 0.7655, 0.7655, 0.7655, 0.7655, 0.7655, 0.7655, 0.7655, 0.7654, 0.7654, 0.7654, 0.7654, 0.7654, 0.7654, 0.7654, 0.7653, 0.7653, 0.7653, 0.7653, 0.7653, 0.7653, 0.7653, 0.7653, 0.7653, 0.7653, 0.7652, 0.7652, 0.765, 0.7647, 0.7651, 0.7652, 0.7652, 4.4113, 4.4071, 4.3987, 0.6362, 0.6362, 0.6361, 0.6361, 0.6361, 0.6361, 0.636, 0.636, 0.636, 0.636, 0.636, 0.636, 0.6359, 0.6359, 0.6359, 0.6359, 0.6359, 0.6359, 0.6359, 0.6358, 0.6358, 0.6358, 0.6358, 0.6358, 0.6358, 0.6358, 0.6357, 0.6357, 0.6351, 0.6356]}, "token.table": {"Topic": [1, 9, 6, 15, 1, 9, 14, 6, 12, 14, 19, 10, 12, 6, 1, 19, 8, 17, 7, 20, 15, 6, 1, 12, 1, 6, 16, 1, 13, 18, 14, 7, 10, 16, 4, 14, 9, 4, 6, 11, 11, 20, 3, 11, 5, 19, 3, 15, 2, 12, 14, 3, 1, 18, 14, 2, 2, 12, 14, 11, 15, 1, 17, 12, 9, 2, 5, 3, 16, 12, 7, 11, 15, 15, 5, 4, 7, 4, 2, 3, 5, 15, 9, 16, 14, 7, 7, 16, 6, 8, 2, 6, 14, 1, 13, 3, 16, 2, 19, 8, 16, 5, 12, 15, 3, 12, 16, 4, 3, 11, 2, 14, 6, 16, 4, 1, 2, 18, 13, 17, 1, 3, 10, 5, 4, 8, 12, 17, 1, 3, 7, 14, 4, 3, 12, 16, 1, 3, 12, 14, 13, 2, 9, 11, 17, 9, 15, 10, 8, 10, 4, 7, 8, 19, 16, 11, 11, 1, 10, 1, 11, 2, 12, 3, 14, 13, 3, 13, 5, 15, 4, 7, 14, 7, 14, 15, 12, 5, 9, 9, 15, 13, 14, 8, 1, 12, 7, 7, 6, 6, 11, 12, 10, 17, 10, 11, 3, 17, 16, 3, 12, 6, 7, 7, 7, 6, 12, 12, 4, 12, 11, 9, 6, 1, 11, 19, 14, 8, 16, 2, 9, 9, 1, 2, 9, 7, 20, 2, 3, 11, 13, 4, 4, 6, 6], "Freq": [0.9699945009060442, 0.964652744755195, 0.9958639441476177, 0.9843471786197454, 0.95420168444561, 0.9706408219063006, 0.9722517285900829, 0.9713551192351603, 0.9788921509724396, 0.9405769038086479, 0.727128903043274, 0.9604214066069041, 0.9746315730155843, 0.9976201243352181, 0.9947591097620386, 0.9896991420744073, 0.9944210088467463, 0.9809582085364537, 0.9881778408183302, 0.9956123770570935, 0.9907202754018113, 0.9966444024044222, 0.9988732175733268, 0.9947583171302619, 0.9951597399713711, 0.9974525643525336, 0.9832022088558225, 0.988283362170096, 0.9846571854767656, 0.992563195789681, 0.913011204418738, 0.9763002998674347, 0.9989107007663768, 0.2716312739016898, 0.9652315776099224, 0.9975145539632588, 0.9794290574863458, 0.9910920521033361, 0.9970821098888375, 0.9940824671620602, 0.9801936933685306, 0.9925981479896139, 0.9973331131489119, 0.9950821425368684, 0.9878269563270311, 0.9869487086719735, 0.9973744827403901, 0.9911877356916509, 0.9990143470549482, 0.8977086996661587, 0.9732548881190903, 0.9863343115714674, 0.9641033799492111, 0.9964454143623437, 0.9917655391505554, 0.808690306296483, 0.8981984352786176, 0.9862132581819967, 0.9654329321243267, 0.9643142291425916, 0.9903848669515288, 0.9851854421243463, 0.9879688249566254, 0.9772198622796741, 0.982865804058214, 0.9925387964824544, 0.8966379417832213, 0.9969795841033201, 0.9907288512535997, 0.9759308451822853, 0.9618125175650347, 0.9803698740098472, 0.9954659964920155, 0.9857499666317413, 0.993736892860095, 0.9990825221192994, 0.9963032211677609, 0.9891053566442006, 0.9891393434903107, 0.9926771510875831, 0.9628139813917108, 0.9853959415571064, 0.9963279173122674, 0.9972340518552847, 0.4716942794008208, 0.9875453950666816, 0.9922168036597221, 0.950835088156433, 0.9936989604362818, 0.9937036338203282, 0.9970117141803473, 0.9672431681586288, 0.9902636474903533, 0.9969480227982662, 0.9943641982814497, 0.9935234414045637, 0.9947229559322696, 0.9948521153328437, 0.9904933241860909, 0.9728888893249916, 0.9843438127316572, 0.8024330728894091, 0.9834450892779651, 0.9791431197023616, 0.982903558225851, 0.9935948382333147, 0.9847768854264431, 0.9826878200435837, 0.9976077573599996, 0.9952583913504486, 0.894852365674264, 0.8509657119040048, 0.9897486467232073, 0.9788041255094793, 0.9930148711453992, 0.9833853982078492, 0.992983541037946, 0.9906084896931402, 0.9984301144918576, 0.9882876868839207, 0.9931681952793884, 0.9954940575443156, 0.9969155425069747, 0.9783488159437662, 0.9915976689623959, 0.9980313786320189, 0.9780532163853628, 0.9969184652872326, 0.998923293684893, 0.9842946075341139, 0.9715659952931947, 0.983268472071006, 0.9962170124201988, 0.9983644969107023, 0.8794499417419728, 0.985205850916359, 0.9974436192757099, 0.9775974899097519, 0.9892388454631363, 0.9688087611207478, 0.9661469341307996, 0.9910730642719535, 0.9969883395353988, 0.9910051736523452, 0.9699076480668101, 0.993335564234938, 0.9907315166179986, 0.9633017927813093, 0.9971287720237505, 0.980379682909926, 0.9968947337223505, 0.9767528917915325, 0.9924814834661311, 0.9941418155853898, 0.946847231972895, 0.9699235610253464, 0.8812018593526415, 0.9966806670351246, 0.9869294864429163, 0.9893279042225032, 0.9902437468629796, 0.9977048609609888, 0.9929250650732263, 0.9710537654327909, 0.994237015077625, 0.908118656613192, 0.9855699473310022, 0.9936942239294019, 0.972693278843521, 0.9692989797964052, 0.996171864867544, 0.9908576189492849, 0.6128943417276889, 0.9702297937685774, 0.9970101093841429, 0.929511568510394, 0.9666867542874503, 0.9995906439798782, 0.9960899166712397, 0.9858809636758255, 0.9898412419743304, 0.9905060142238462, 0.9168857728161749, 0.9966105879061511, 0.9921728836299053, 0.9707446954313361, 0.9932991363572004, 0.9877047671260568, 0.9893824228695293, 0.9916318392988223, 0.9661878744610827, 0.9870404773100682, 0.9656499930531669, 0.9950323230549433, 0.983856211288782, 0.9647502506361955, 0.9970114734660314, 0.9481274127336401, 0.954821334839217, 0.9899154508758394, 0.9871249387194484, 0.9695416432889215, 0.9954249927359322, 0.9910290747527057, 0.9796937509108553, 0.9937484469949497, 0.9871788321709604, 0.9857577359473332, 0.9972992992082602, 0.5276952178902906, 0.9603378956148443, 0.9744490247743727, 0.9342612430228133, 0.9997642855415193, 0.9916339813833228, 0.9868659757089799, 0.7249548101454046, 0.9953920466423002, 0.823218599608793, 0.9945045777628524, 0.9871446461725016, 0.9956081442117907, 0.9943012850350701, 0.9983946046652418, 0.9931196898908449, 0.9968812909893995, 0.9866868854458953, 0.9963947954145568, 0.977307276734353, 0.9934160216184025, 0.9910565846276682, 0.9445103868060989, 0.9979274828046674, 0.9896155534767843, 0.9902669888360175], "Term": ["abortion", "access", "accuse", "act", "add", "address", "agency", "agree", "ahead", "airline", "alleged", "ally", "almost", "analysis", "approve", "arrest", "attack", "attempt", "away", "back", "bad", "become", "biden", "big", "bill", "black", "body", "border", "boy", "break", "building", "buy", "call", "camera", "car", "case", "celebrate", "change", "charge", "child", "chinese", "city", "claim", "come", "company", "concern", "could", "couple", "covid", "credit", "criminal", "criticism", "damage", "day", "delay", "disease", "divide", "doctor", "dozen", "drive", "drop", "drug", "end", "enough", "evidence", "expert", "eye", "face", "fail", "failure", "fed", "female", "find", "fine", "fire", "first", "former", "free", "friend", "full", "gas", "girl", "give", "go", "guilty", "gun", "handle", "happen", "history", "home", "honor", "huge", "issue", "job", "join", "keep", "know", "last", "late", "launch", "lawmaker", "leak", "learn", "least", "level", "lie", "low", "major", "make", "man", "marriage", "medium", "member", "mile", "military", "miss", "month", "need", "new", "offer", "office", "officer", "official", "oil", "open", "opinion", "oppose", "order", "pandemic", "panel", "parent", "patient", "pay", "people", "performance", "place", "plan", "play", "player", "plead", "politic", "political", "poll", "power", "probe", "prosecutor", "protect", "protection", "protester", "provide", "public", "racism", "racist", "raise", "rate", "reach", "real", "record", "reject", "remain", "remark", "report", "republican", "request", "require", "rescue", "resign", "response", "responsible", "restore", "return", "right", "rioter", "rip", "rule", "ruling", "safety", "say", "see", "seek", "send", "set", "several", "show", "slam", "space", "speak", "stay", "step", "still", "story", "strip", "strong", "student", "study", "study_find", "support", "supreme_court", "suspect", "suspend", "team", "teen", "tell", "test", "thing", "threat", "throw", "tie", "time", "title", "together", "trial", "trip", "trump", "try", "turn", "unveil", "use", "vaccination", "vaccine", "violence", "virus", "visit", "vote", "voting", "want", "watch", "way", "weekend", "white", "worker", "worth", "year", "year_old", "young"]}, "R": 30, "lambda.step": 0.01, "plot.opts": {"xlab": "PC1", "ylab": "PC2"}, "topic.order": [14, 5, 2, 19, 15, 7, 12, 9, 8, 16, 11, 13, 1, 10, 6, 17, 18, 20, 3, 4]};

function LDAvis_load_lib(url, callback){
  var s = document.createElement('script');
  s.src = url;
  s.async = true;
  s.onreadystatechange = s.onload = callback;
  s.onerror = function(){console.warn("failed to load library " + url);};
  document.getElementsByTagName("head")[0].appendChild(s);
}

if(typeof(LDAvis) !== "undefined"){
   // already loaded: just create the visualization
   !function(LDAvis){
       new LDAvis("#" + "ldavis_el176541404059900113927020288209", ldavis_el176541404059900113927020288209_data);
   }(LDAvis);
}else if(typeof define === "function" && define.amd){
   // require.js is available: use it to load d3/LDAvis
   require.config({paths: {d3: "https://d3js.org/d3.v5"}});
   require(["d3"], function(d3){
      window.d3 = d3;
      LDAvis_load_lib("https://cdn.jsdelivr.net/gh/bmabey/pyLDAvis@3.3.1/pyLDAvis/js/ldavis.v3.0.0.js", function(){
        new LDAvis("#" + "ldavis_el176541404059900113927020288209", ldavis_el176541404059900113927020288209_data);
      });
    });
}else{
    // require.js not available: dynamically load d3 & LDAvis
    LDAvis_load_lib("https://d3js.org/d3.v5.js", function(){
         LDAvis_load_lib("https://cdn.jsdelivr.net/gh/bmabey/pyLDAvis@3.3.1/pyLDAvis/js/ldavis.v3.0.0.js", function(){
                 new LDAvis("#" + "ldavis_el176541404059900113927020288209", ldavis_el176541404059900113927020288209_data);
            })
         });
}
</script>



Each bubble on the left-hand side plot represents a topic. The larger the bubble, the more prevalent is that topic.

A good topic model will have fairly big, non-overlapping bubbles scattered throughout the chart instead of being clustered in one quadrant.

A model with too many topics, will typically have many overlaps, small sized bubbles clustered in one region of the chart.

<div class="alert alert-info" style="background-color:#5d3a8e; color:white; padding:0px 10px; border-radius:5px;"><h2 style='margin:10px 5px'> Building LDA Mallet Model</h2>
</div>

Using Mallet for a better quality of topics.

Gensim provides a wrapper to implement Mallet's LDA from within gensim itself. We'll [download](http://mallet.cs.umass.edu/dist/mallet-2.0.8.zip) the zipfile, unzip it and provide the path to mallet in the unzipped directory to `gensim.models.wrappers.LdaMallet`.


```python
import os
os.environ.update({'MALLET_HOME':r'/Users/a206679878/Documents/Untitled Folder/mallet-2.0.8'})
```


```python
import os
from gensim.models.wrappers import LdaMallet
```


```python
import os
os.environ.update({'MALLET_HOME': r'/Users/a206679878/Documents/mallet-2.0.8'})
mallet_path = '/Users/a206679878/Documents/mallet-2.0.8/bin/mallet'  # update this path
ldamallet = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=20, id2word=id2word)
```


```python
# Show Topics
pprint(ldamallet.show_topics(formatted=False))

# Compute Coherence Score
coherence_model_ldamallet = CoherenceModel(model=ldamallet, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
coherence_ldamallet = coherence_model_ldamallet.get_coherence()
print('\nCoherence Score: ', coherence_ldamallet)
```

    [(10,
      [('trump', 0.17915504184934236),
       ('analysis', 0.03965723395775209),
       ('back', 0.03308090872857712),
       ('fight', 0.03268234356317258),
       ('impeachment', 0.0296931048226385),
       ('lie', 0.02510960542048625),
       ('court', 0.024910322837783976),
       ('american', 0.021522518931845355),
       ('turn', 0.018931845356715823),
       ('threat', 0.01654045436428856)]),
     (11,
      [('covid', 0.20630202140309156),
       ('vaccine', 0.1403091557669441),
       ('world', 0.030122869599682918),
       ('offer', 0.022988505747126436),
       ('dose', 0.02100673801030519),
       ('receive', 0.020214030915576695),
       ('vaccination', 0.019817677368212445),
       ('protect', 0.017637732857709077),
       ('administration', 0.01605231866825208),
       ('vaccinate', 0.014863258026159334)]),
     (8,
      [('mask', 0.06119925819080981),
       ('face', 0.05378116628889347),
       ('wear', 0.02637543787348032),
       ('protester', 0.024933031114774365),
       ('refuse', 0.024933031114774365),
       ('public', 0.023490624356068412),
       ('require', 0.020399752730269935),
       ('investigate', 0.019575520296723676),
       ('power', 0.018751287863177417),
       ('violence', 0.018339171646404286)]),
     (15,
      [('year', 0.09398797595190381),
       ('time', 0.06973947895791584),
       ('man', 0.05811623246492986),
       ('die', 0.04088176352705411),
       ('long', 0.0312625250501002),
       ('kill', 0.031062124248496994),
       ('year_old', 0.01963927855711423),
       ('sentence', 0.01903807615230461),
       ('murder', 0.018637274549098196),
       ('free', 0.015831663326653308)]),
     (2,
      [('state', 0.06840665123131973),
       ('police', 0.06293411913281415),
       ('video', 0.03451904862134288),
       ('sue', 0.031361818564512735),
       ('show', 0.029046516522837296),
       ('chief', 0.027994106503893917),
       ('company', 0.02146916438644496),
       ('deal', 0.021258682382656283),
       ('gun', 0.018311934329614817),
       ('policy', 0.017890970322037465)]),
     (4,
      [('election', 0.181010101010101),
       ('win', 0.06707070707070707),
       ('presidential', 0.04464646464646465),
       ('result', 0.036565656565656565),
       ('live', 0.03333333333333333),
       ('race', 0.0298989898989899),
       ('effort', 0.028282828282828285),
       ('reject', 0.028282828282828285),
       ('suspend', 0.026060606060606062),
       ('victory', 0.025050505050505052)]),
     (19,
      [('opinion', 0.09183886453767481),
       ('attack', 0.036944270507201005),
       ('lose', 0.03506574827802129),
       ('group', 0.029221456898351074),
       ('nation', 0.022333542058025464),
       ('city', 0.019828845752452515),
       ('late', 0.01878522229179712),
       ('address', 0.017741598831141726),
       ('blast', 0.015236902525568774),
       ('book', 0.014610728449175537)]),
     (7,
      [('official', 0.06688755435908056),
       ('death', 0.043487264443984264),
       ('report', 0.04307310002070822),
       ('fire', 0.037896044729757716),
       ('member', 0.03023400289915096),
       ('warn', 0.03002692068751294),
       ('week', 0.02981983847587492),
       ('top', 0.029405674052598883),
       ('call', 0.02712776972458066),
       ('honor', 0.026713605301304618)]),
     (1,
      [('plan', 0.05947271612507664),
       ('pay', 0.03903535663192315),
       ('move', 0.035561005518087066),
       ('worker', 0.032904148783977114),
       ('raise', 0.02861230329041488),
       ('hit', 0.02534232577151032),
       ('tax', 0.02411608420192111),
       ('job', 0.023094216227263438),
       ('back', 0.022685469037400367),
       ('approve', 0.022481095442468832)]),
     (5,
      [('trump', 0.12246569731722302),
       ('day', 0.08437436002457506),
       ('debate', 0.045463854187999184),
       ('leave', 0.04423510137210731),
       ('office', 0.030104443989350808),
       ('change', 0.024165472045873437),
       ('final', 0.02027442146221585),
       ('join', 0.018226500102396067),
       ('bring', 0.01576899447061233),
       ('play', 0.015154618062666393)])]
    
    Coherence Score:  0.3992169346463529


<div class="alert alert-info" style="background-color:#5d3a8e; color:white; padding:0px 10px; border-radius:5px;"><h2 style='margin:10px 5px'> Finding the optimal number of topics for LDA</h2>
</div>

The `compute_coherence_values()` (see below) trains multiple LDA models and provides the models and their corresponding coherence scores.


```python
def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=num_topics, id2word=id2word)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values
```


```python
model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=corpus, texts=data_lemmatized, start=2, limit=40, step=6)
```


```python
# Show graph
limit=40; start=2; step=6;
x = range(start, limit, step)
plt.plot(x, coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show()
```


    
![png](output_64_0.png)
    



```python
# Print the coherence scores
for m, cv in zip(x, coherence_values):
    print("Num Topics =", m, " has Coherence Value of", round(cv, 4))
```

    Num Topics = 2  has Coherence Value of 0.3061
    Num Topics = 8  has Coherence Value of 0.3647
    Num Topics = 14  has Coherence Value of 0.3937
    Num Topics = 20  has Coherence Value of 0.4017
    Num Topics = 26  has Coherence Value of 0.4245
    Num Topics = 32  has Coherence Value of 0.4434
    Num Topics = 38  has Coherence Value of 0.4559


If the coherence score seems to keep increasing, it may make better sense to pick the model that gave the highest CV before flattening out. This is exactly the case here.

So for further steps I will choose the model with 20 topics itself.


```python
# Select the model and print the topics
optimal_model = model_list[3]
model_topics = optimal_model.show_topics(formatted=False)
pprint(optimal_model.print_topics(num_words=10))
```

    [(0,
      '0.121*"vote" + 0.057*"bill" + 0.047*"state" + 0.032*"pass" + 0.032*"law" + '
      '0.030*"voting" + 0.027*"ballot" + 0.026*"sign" + 0.026*"relief" + '
      '0.025*"early"'),
     (1,
      '0.048*"support" + 0.041*"leader" + 0.032*"republican" + 0.031*"member" + '
      '0.029*"group" + 0.029*"effort" + 0.026*"honor" + 0.026*"protester" + '
      '0.024*"lawmaker" + 0.021*"begin"'),
     (2,
      '0.094*"year" + 0.055*"make" + 0.038*"record" + 0.035*"history" + '
      '0.028*"set" + 0.028*"return" + 0.027*"life" + 0.026*"break" + 0.023*"tax" + '
      '0.019*"sentence"'),
     (3,
      '0.249*"trump" + 0.045*"claim" + 0.038*"campaign" + 0.030*"sue" + '
      '0.029*"impeachment" + 0.028*"reject" + 0.024*"lawsuit" + 0.023*"make" + '
      '0.019*"lawyer" + 0.018*"false"'),
     (4,
      '0.092*"people" + 0.073*"trump" + 0.038*"coronavirus" + 0.036*"lose" + '
      '0.031*"urge" + 0.031*"case" + 0.027*"hold" + 0.027*"suspend" + 0.022*"send" '
      '+ 0.018*"twitter"'),
     (5,
      '0.045*"order" + 0.029*"fight" + 0.026*"court" + 0.025*"seek" + 0.023*"end" '
      '+ 0.022*"nation" + 0.022*"release" + 0.021*"issue" + 0.020*"expect" + '
      '0.018*"policy"'),
     (6,
      '0.052*"analysis" + 0.041*"move" + 0.041*"remove" + 0.039*"school" + '
      '0.037*"work" + 0.031*"office" + 0.026*"month" + 0.022*"american" + '
      '0.022*"run" + 0.021*"end"'),
     (7,
      '0.183*"election" + 0.045*"presidential" + 0.039*"debate" + 0.037*"result" + '
      '0.036*"state" + 0.030*"live" + 0.025*"victory" + 0.025*"change" + '
      '0.022*"democratic" + 0.021*"senator"'),
     (8,
      '0.050*"find" + 0.044*"voter" + 0.041*"back" + 0.038*"poll" + 0.037*"child" '
      '+ 0.030*"student" + 0.023*"job" + 0.022*"fall" + 0.020*"percent" + '
      '0.019*"late"'),
     (9,
      '0.061*"police" + 0.052*"die" + 0.050*"officer" + 0.050*"family" + '
      '0.038*"fire" + 0.029*"home" + 0.027*"chief" + 0.025*"resign" + '
      '0.021*"year_old" + 0.020*"pick"'),
     (10,
      '0.095*"woman" + 0.061*"man" + 0.049*"arrest" + 0.035*"pay" + 0.031*"turn" + '
      '0.018*"free" + 0.016*"thousand" + 0.016*"play" + 0.015*"female" + '
      '0.014*"dead"'),
     (11,
      '0.152*"call" + 0.065*"official" + 0.058*"trump" + 0.053*"report" + '
      '0.028*"top" + 0.028*"open" + 0.025*"case" + 0.023*"prosecutor" + '
      '0.021*"investigation" + 0.017*"celebrate"'),
     (12,
      '0.070*"win" + 0.052*"rule" + 0.049*"federal" + 0.031*"race" + '
      '0.024*"protest" + 0.022*"company" + 0.022*"trial" + 0.019*"stand" + '
      '0.019*"head" + 0.019*"judge"'),
     (13,
      '0.074*"black" + 0.065*"trump" + 0.062*"show" + 0.039*"stop" + 0.034*"video" '
      '+ 0.030*"give" + 0.022*"campaign" + 0.022*"post" + 0.021*"ad" + '
      '0.016*"talk"'),
     (14,
      '0.201*"covid" + 0.137*"vaccine" + 0.030*"worker" + 0.025*"world" + '
      '0.023*"offer" + 0.020*"dose" + 0.020*"receive" + 0.019*"vaccination" + '
      '0.018*"vaccinate" + 0.016*"protect"'),
     (15,
      '0.297*"biden" + 0.058*"plan" + 0.042*"announce" + 0.042*"president" + '
      '0.025*"ahead" + 0.024*"inauguration" + 0.022*"team" + 0.022*"endorse" + '
      '0.018*"transition" + 0.018*"gun"'),
     (16,
      '0.187*"trump" + 0.100*"opinion" + 0.054*"lead" + 0.039*"attack" + '
      '0.031*"lie" + 0.024*"big" + 0.021*"country" + 0.021*"bad" + '
      '0.019*"national" + 0.017*"capitol"'),
     (17,
      '0.061*"mask" + 0.044*"leave" + 0.034*"ban" + 0.032*"long" + 0.026*"virus" + '
      '0.026*"wear" + 0.025*"test" + 0.025*"refuse" + 0.024*"public" + '
      '0.020*"require"'),
     (18,
      '0.086*"charge" + 0.053*"face" + 0.043*"death" + 0.033*"kill" + '
      '0.031*"speech" + 0.027*"shoot" + 0.024*"accuse" + 0.019*"murder" + '
      '0.019*"violence" + 0.018*"watch"'),
     (19,
      '0.084*"day" + 0.066*"time" + 0.065*"pandemic" + 0.029*"week" + 0.028*"warn" '
      '+ 0.026*"close" + 0.026*"hit" + 0.022*"start" + 0.019*"high" + '
      '0.016*"back"')]


Those were the topics for the chosen LDA model.

<div class="alert alert-info" style="background-color:#5d3a8e; color:white; padding:0px 10px; border-radius:5px;"><h2 style='margin:10px 5px'> Finding the dominant topic in each sentence</h2>
</div>

Determine what topic a given document is about.

The topic number that has the highest percentage contribution in that document is what we call out here.


```python
def format_topics_sentences(ldamodel=lda_model, corpus=corpus, texts=data):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)


df_topic_sents_keywords = format_topics_sentences(ldamodel=optimal_model, corpus=corpus, texts=data)

# Format
df_dominant_topic = df_topic_sents_keywords.reset_index()
df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']

# Show
df_dominant_topic.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Document_No</th>
      <th>Dominant_Topic</th>
      <th>Topic_Perc_Contrib</th>
      <th>Keywords</th>
      <th>Text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>9.0</td>
      <td>0.0789</td>
      <td>police, die, officer, family, fire, home, chie...</td>
      <td>WATCH: 100 protesters show up at home of man w...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1.0</td>
      <td>0.0660</td>
      <td>support, leader, republican, member, group, ef...</td>
      <td>Kinzinger says he suspects some lawmakers knew...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>11.0</td>
      <td>0.0843</td>
      <td>call, official, trump, report, top, open, case...</td>
      <td>Maricopa County official on rejecting Trump al...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>10.0</td>
      <td>0.0804</td>
      <td>woman, man, arrest, pay, turn, free, thousand,...</td>
      <td>A man who went on a racist rant gave out his a...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>5.0</td>
      <td>0.0848</td>
      <td>order, fight, court, seek, end, nation, releas...</td>
      <td>Surfside catastrophe raises concerns about San...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>0.0</td>
      <td>0.0500</td>
      <td>vote, bill, state, pass, law, voting, ballot, ...</td>
      <td>Blake Shelton and Gwen Stefani are married</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6</td>
      <td>0.0</td>
      <td>0.0500</td>
      <td>vote, bill, state, pass, law, voting, ballot, ...</td>
      <td>Courteney Cox, Jennifer Aniston and Lisa Kudro...</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7</td>
      <td>4.0</td>
      <td>0.0639</td>
      <td>people, trump, coronavirus, lose, urge, case, ...</td>
      <td>At least 150 people fatally shot in more than ...</td>
    </tr>
    <tr>
      <th>8</th>
      <td>8</td>
      <td>13.0</td>
      <td>0.0744</td>
      <td>black, trump, show, stop, video, give, campaig...</td>
      <td>Vanessa Williams and PBS slammed for  Black na...</td>
    </tr>
    <tr>
      <th>9</th>
      <td>9</td>
      <td>11.0</td>
      <td>0.0716</td>
      <td>call, official, trump, report, top, open, case...</td>
      <td>Trump could be called to testify before House ...</td>
    </tr>
  </tbody>
</table>
</div>



The tabular output above had 20 rows, one each for a topic. It has the topic number, the keywords and the most representative document. The `Perc_Contribution` column is nothing but the percentage contribution of the topic in the given document.

<div class="alert alert-info" style="background-color:#5d3a8e; color:white; padding:0px 10px; border-radius:5px;"><h2 style='margin:10px 5px'> Topic distribution across documents</h2>
</div>


```python
# Number of Documents for Each Topic
topic_counts = df_topic_sents_keywords['Dominant_Topic'].value_counts()

# Percentage of Documents for Each Topic
topic_contribution = round(topic_counts/topic_counts.sum(), 4)

# Topic Number and Keywords
topic_num_keywords = df_topic_sents_keywords[['Dominant_Topic', 'Topic_Keywords']]

# Concatenate Column wise
df_dominant_topics = pd.concat([topic_num_keywords, topic_counts, topic_contribution], axis=1)

# Change Column names
df_dominant_topics.columns = ['Dominant_Topic', 'Topic_Keywords', 'Num_Documents', 'Perc_Documents']

# Show
df_dominant_topics.head(20)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Dominant_Topic</th>
      <th>Topic_Keywords</th>
      <th>Num_Documents</th>
      <th>Perc_Documents</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0.0</th>
      <td>9.0</td>
      <td>police, die, officer, family, fire, home, chie...</td>
      <td>2588.0</td>
      <td>0.1168</td>
    </tr>
    <tr>
      <th>1.0</th>
      <td>1.0</td>
      <td>support, leader, republican, member, group, ef...</td>
      <td>1393.0</td>
      <td>0.0628</td>
    </tr>
    <tr>
      <th>2.0</th>
      <td>11.0</td>
      <td>call, official, trump, report, top, open, case...</td>
      <td>1312.0</td>
      <td>0.0592</td>
    </tr>
    <tr>
      <th>3.0</th>
      <td>10.0</td>
      <td>woman, man, arrest, pay, turn, free, thousand,...</td>
      <td>1072.0</td>
      <td>0.0484</td>
    </tr>
    <tr>
      <th>4.0</th>
      <td>5.0</td>
      <td>order, fight, court, seek, end, nation, releas...</td>
      <td>1157.0</td>
      <td>0.0522</td>
    </tr>
    <tr>
      <th>5.0</th>
      <td>0.0</td>
      <td>vote, bill, state, pass, law, voting, ballot, ...</td>
      <td>1069.0</td>
      <td>0.0482</td>
    </tr>
    <tr>
      <th>6.0</th>
      <td>0.0</td>
      <td>vote, bill, state, pass, law, voting, ballot, ...</td>
      <td>1098.0</td>
      <td>0.0495</td>
    </tr>
    <tr>
      <th>7.0</th>
      <td>4.0</td>
      <td>people, trump, coronavirus, lose, urge, case, ...</td>
      <td>1105.0</td>
      <td>0.0499</td>
    </tr>
    <tr>
      <th>8.0</th>
      <td>13.0</td>
      <td>black, trump, show, stop, video, give, campaig...</td>
      <td>1060.0</td>
      <td>0.0478</td>
    </tr>
    <tr>
      <th>9.0</th>
      <td>11.0</td>
      <td>call, official, trump, report, top, open, case...</td>
      <td>1105.0</td>
      <td>0.0499</td>
    </tr>
    <tr>
      <th>10.0</th>
      <td>4.0</td>
      <td>people, trump, coronavirus, lose, urge, case, ...</td>
      <td>1003.0</td>
      <td>0.0453</td>
    </tr>
    <tr>
      <th>11.0</th>
      <td>15.0</td>
      <td>biden, plan, announce, president, ahead, inaug...</td>
      <td>1036.0</td>
      <td>0.0467</td>
    </tr>
    <tr>
      <th>12.0</th>
      <td>16.0</td>
      <td>trump, opinion, lead, attack, lie, big, countr...</td>
      <td>936.0</td>
      <td>0.0422</td>
    </tr>
    <tr>
      <th>13.0</th>
      <td>5.0</td>
      <td>order, fight, court, seek, end, nation, releas...</td>
      <td>844.0</td>
      <td>0.0381</td>
    </tr>
    <tr>
      <th>14.0</th>
      <td>6.0</td>
      <td>analysis, move, remove, school, work, office, ...</td>
      <td>1136.0</td>
      <td>0.0513</td>
    </tr>
    <tr>
      <th>15.0</th>
      <td>2.0</td>
      <td>year, make, record, history, set, return, life...</td>
      <td>868.0</td>
      <td>0.0392</td>
    </tr>
    <tr>
      <th>16.0</th>
      <td>13.0</td>
      <td>black, trump, show, stop, video, give, campaig...</td>
      <td>921.0</td>
      <td>0.0416</td>
    </tr>
    <tr>
      <th>17.0</th>
      <td>11.0</td>
      <td>call, official, trump, report, top, open, case...</td>
      <td>851.0</td>
      <td>0.0384</td>
    </tr>
    <tr>
      <th>18.0</th>
      <td>14.0</td>
      <td>covid, vaccine, worker, world, offer, dose, re...</td>
      <td>842.0</td>
      <td>0.0380</td>
    </tr>
    <tr>
      <th>19.0</th>
      <td>19.0</td>
      <td>day, time, pandemic, week, warn, close, hit, s...</td>
      <td>768.0</td>
      <td>0.0347</td>
    </tr>
  </tbody>
</table>
</div>


