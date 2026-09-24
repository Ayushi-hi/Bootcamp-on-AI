import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter

# Ensure necessary NLTK data is downloaded
nltk.download('punkt')
nltk.download('stopwords')

text = """
Adventure, sunshine, mountain, river, notebook, coffee, garden, laptop, music, journey, ocean, butterfly, camera, window, morning, friendship, dream, bicycle, rainbow, forest, laughter, creativity, chocolate,
keyboard, sunset, planet, flower, curiosity, story, adventure, technology, cloud, pencil, school, festival, football, movie, book, travel, happiness, discovery, nature, painting, computer, moonlight, stars, 
village, city, bridge, beach, island, rocket, science, future, learning, ideas, energy, freedom, success, challenge, courage, kindness, smile, family, memories, imagination, design, coding, website, application,
project, innovation, teamwork, knowledge, experience, opportunity, passion, goal, ambition, achievement, practice, growth, confidence, patience, focus, balance, health, exercise, games, drawing, photography, writing,
reading, dancing, singing, cooking, gardening, swimming, running, cycling, hiking, exploring, experimenting, creating, building, sharing, helping, volunteering, celebrating, dreaming, planning, improving, discovering,
observing, thinking, questioning, solving, connecting, communicating, inspiring, supporting, enjoying, relaxing, laughing, learning, developing, designing, testing, creating, researching, presenting, discussing, collaborating, 
organizing, leading, participating, competing, winning, practicing, preparing, traveling, meeting, exploring, experiencing, remembering, appreciating, smiling, enjoying, wondering, imagining, believing, trying, failing, improving, 
succeeding, growing, adapting, changing, progressing, achieving, motivating, inspiring, encouraging, helping, caring, trusting, respecting, understanding, listening, communicating, creating, learning, discovering, building, exploring.

"""
print(text)
# Tokenization of words
words = word_tokenize(text.lower())  # Convert to lowercase for uniformity
print(words)
# Remove stopwords
stop_words = set(stopwords.words('english'))
filtered_words = [word for word in words if word.isalpha() and word not in stop_words]
# The general syntax for a list comprehension is: [expression for item in iterable if condition]
print(filtered_words)
wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)
# Plotting the word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')  # Hide axes
plt.show()