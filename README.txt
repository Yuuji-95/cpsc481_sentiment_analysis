# Sentiment Analysis on Movie Reviews
CPSC 481 – Final Project
Team Members:
- Yuuji Kobayashi
- Jonathan Quiroz

------------------------------------------------------
1. System Requirements
------------------------------------------------------
OS: Windows 10/11 or macOS 12+
Python Version: 3.11+
Required Python libraries:
- numpy
- pandas
- scikit-learn
- matplotlib
- nltk
- joblib
- tkinter (for GUI)

------------------------------------------------------
2. Installation Instructions
------------------------------------------------------
1. Extract the project folder.
2. Open a terminal inside the project directory.
3. Create a virtual environment:
   Windows:
      py -m venv venv
      venv\Scripts\activate
   macOS:
      python3 -m venv venv
      source venv/bin/activate

4. Install dependencies:
   pip install -r requirements.txt

5. (macOS only) Install Tkinter:
   brew install python-tk

------------------------------------------------------
3. Running the Application
------------------------------------------------------

make sure to locate and go to src with "cd src"
make sure to locate and go to src with "cd src"
make sure to locate and go to src with "cd src"
make sure to locate and go to src with "cd src"
make sure to locate and go to src with "cd src"

A. Explore the Dataset
------------------------
Windows: py explore_data.py
macOS: python3 explore_data.py

Shows:
- First 5 dataset rows
- Class distribution
- Sentiment distribution chart

B. Train the Model
------------------------
Windows: py train_model.py
macOS: python3 train_model.py

This:
- Cleans the text
- Vectorizes reviews using TF-IDF
- Trains Logistic Regression
- Saves:
    /data/sentiment_model.pkl
    /data/tfidf_vectorizer.pkl

C. Evaluate the Model
------------------------
Windows: py evaluate.py
macOS: python3 evaluate.py

Outputs:
- Accuracy
- Precision/Recall/F1
- Confusion Matrix (saved to /figures)

D. Predict Sentiment (Command Line)
------------------------------------
Windows: py predict.py
macOS: python3 predict.py

Allows typing your own review.

E. GUI Application
------------------------
Windows: py gui_app.py
macOS: python3 gui_app.py

Review Comment example:
Positive:
"This is an interesting, hard to find movie from the early 70's starring Jan Michael Vincent as a young man who doesn't make the cut as a marine. Dressed in 'baby blue' outfits to humiliate them as they are sent home, the failed recruits are sent packing. Vincent stops at a bar and runs into a very young Richard Gere who has just returned from a tour in the Pacific as a hard-core Marine 'Raider'. Gere's character is already jaded and contemplating desertion, and he takes advantage of Vincent's innocence, stealing his 'baby blue' uniform after getting him drunk and beating him in an alleyway. Vincent's character, whose name is Marion, takes Gere's outfit and is suddenly transformed into a Marine 'Raider'. Marion hitch-hikes his way into Wyoming and stops at a little Norman Rockwell-like little town. In the local café he meets Rose Hudkins, who immediately catches his eye. Staying with Hudkins parents, Marion attracts all sorts of attention from the towns folks. Mr Hudkins suspects Marion and wonders how a Marine 'Raider' could still be so innocent. The story also brings up the Japanese Internment Camps, as the towns folks go 'hunting' 3 escapees. Marion is shot accidentally during this hunt. But there's still a happy ending, which befuddled me a bit. I would have preferred a little more drama! Anyway, this captures JMV at the peak of his 70's performances. BUSTER AND BILLIE, BABY BLUE MARINE and WHITE LINE FEVER in the mid-70's were amazingly good JMV performances. He was both an action star and a heart-throb all at the same time!!! He made a lot of quality movies during his career, and continued to do so up into the mid 80's with the great TV show Airwolf. He does a very good job in this as 'Hedge', quietly observing the way people treat him (in his uniform) as he travels across the country. He must have performed some of the stunt work as well- there is a harrowing river scene at the end of the movie-and it looks like he's the guy getting tossed down the river to me! But really, at the height of his popularity, this movie could have done so much more with JMV's talent and his looks. Innocence can only be so interesting. Evil, as explored in ""Buster and Billie"", is much more dramatic! Anyway, Glynnis O'Connor is delightful as Rose. The whole look of the movie is like a Norman Rockwell painting. The outdoor scenes are gorgeous - must have been filmed in Canada. Overall, this is very very great."
Negative:
"This is a really horrible film in the vein of ""Buckaroo Banzai."" The cast runs around like ""Mad Max"" wannabes, and they seem to be sharing a joke that they do not want to share with the audience. Wheeler-Nicholson is one of the those guilty pleasure actresses you are delighted to stumble across in films, but she isn't worth the price of rental. Space Maggot starts an electrical fire, and burns a vote of 4. Overall, this is very bad. Really bad."

Opens a window where users can type movie reviews
and get “positive” or “negative” predictions.

------------------------------------------------------
4. Project Structure
------------------------------------------------------
project/
│── data/
│     ├── imdb_reviews.csv
│     ├── sentiment_model.pkl
│     └── tfidf_vectorizer.pkl
│── figures/
│     └── confusion_matrix.png
│── src/
│     ├── explore_data.py
│     ├── preprocess.py
│     ├── train_model.py
│     ├── evaluate.py
│     ├── predict.py
│     └── gui_app.py
│── README.txt
│── requirements.txt
│── venv/ (ignored for submission)

------------------------------------------------------
5. Credentials or Test Accounts
------------------------------------------------------
Not applicable — project uses offline data only.

------------------------------------------------------
6. Notes for Instructor
------------------------------------------------------
If any script fails to find files, ensure you run commands 
from the ROOT of the project directory, not inside /src.
