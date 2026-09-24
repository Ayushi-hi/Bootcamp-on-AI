import nltk
import matplotlib.pyplot as plt

from wordcloud import WordCloud, STOPWORDS
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# ============================================================
# 1. DOWNLOAD NLTK DATA
# ============================================================

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)


# ============================================================
# 2. TEXT
# ============================================================

text = """
Artificial Intelligence is changing the world in many different ways.
Artificial intelligence allows computers to learn from data and make
intelligent decisions.

Machine learning is an important part of artificial intelligence.
Machine learning algorithms learn patterns from large amounts of data.
Deep learning uses neural networks to solve complex problems.

Python is one of the most popular programming languages for
artificial intelligence, machine learning, data science and automation.

Data science helps organizations understand data and discover useful
information. Big data allows companies to analyze millions of records
and make better decisions.

Artificial intelligence is used in healthcare, education, finance,
business, cybersecurity, robotics, transportation and entertainment.

Students can learn Python, machine learning, deep learning, data
science and artificial intelligence to build useful applications.

The future of technology will include artificial intelligence,
automation, robotics, machine learning and intelligent software.

Artificial intelligence helps businesses solve problems, analyze data,
understand patterns and build better applications.

Machine learning and deep learning are helping researchers create
powerful systems that can recognize images, understand language,
predict outcomes and automate complex tasks.
"""


# ============================================================
# 3. TOKENIZE
# ============================================================

tokens = word_tokenize(text.lower())


# ============================================================
# 4. REMOVE STOPWORDS
# ============================================================

stop_words = set(stopwords.words("english"))

# Add some extra words that are not useful in the cloud
stop_words.update([
    "the",
    "and",
    "is",
    "are",
    "to",
    "of",
    "in",
    "for",
    "a",
    "an",
    "with",
    "from",
    "can",
    "be",
    "will",
    "one",
    "many",
    "different",
    "used",
    "uses"
])


words = []

for word in tokens:

    if word.isalpha() and word not in stop_words:
        words.append(word)


clean_text = " ".join(words)


# ============================================================
# 5. CREATE WORD CLOUD
# ============================================================

wordcloud = WordCloud(
    width=1600,
    height=900,

    # Background
    background_color="white",

    # Number of words
    max_words=60,

    # Better word sizing
    min_font_size=15,
    max_font_size=180,

    # Makes word sizes more balanced
    relative_scaling=0.5,

    # Avoids repeating common word combinations
    collocations=False,

    # More horizontal words
    prefer_horizontal=0.85,

    # Color theme
    colormap="viridis",

    # Makes the result reproducible
    random_state=42,

    # Small border around words
    margin=8,

    # Slight contour
    contour_width=1,
    contour_color="lightgray"
).generate(clean_text)


# ============================================================
# 6. DISPLAY
# ============================================================

plt.figure(figsize=(16, 9))

plt.imshow(
    wordcloud,
    interpolation="bilinear"
)

plt.axis("off")

plt.title(
    "Artificial Intelligence",
    fontsize=32,
    fontweight="bold",
    pad=25
)

plt.tight_layout(pad=2)


# ============================================================
# 7. SAVE IMAGE
# ============================================================

wordcloud.to_file("wordcloud.png")


# ============================================================
# 8. SUCCESS MESSAGE
# ============================================================

print()
print("==========================================")
print("       WORD CLOUD CREATED SUCCESSFULLY")
print("==========================================")
print("File saved as: wordcloud.png")
print("==========================================")
print()


# ============================================================
# 9. SHOW
# ============================================================

plt.show()