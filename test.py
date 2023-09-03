import uvicorn
# from typing import Optional
from fastapi import FastAPI, Body, Request, File, UploadFile, Form
from fastapi import responses
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier,OneVsOneClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score
from pandas.plotting import scatter_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score

import seaborn as sns
import pdfplumber
import re
import aiofiles
 


def cleanResume(resumeText):
    resumeText = resumeText.lower()
    resumeText = re.sub('http\S+\s*', ' ', resumeText)  # remove URLs
    resumeText = re.sub('RT|cc', ' ', resumeText)  # remove RT and cc
    resumeText = re.sub('#\S+', '', resumeText)  # remove hashtags
    resumeText = re.sub('@\S+', '  ', resumeText)  # remove mentions
    resumeText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', resumeText)  # remove punctuations
    resumeText = re.sub(r'[^\x00-\x7f]',r' ', resumeText) 
    resumeText = re.sub('\s+', ' ', resumeText)  # remove extra whitespace
    return resumeText




df = pd.read_csv('UpdatedResumeDataSet.csv',encoding = 'utf-8')
#print(df.head().encode('utf-8'))

#print(df['Category'].unique())

#print(df['Category'].value_counts())

plt.figure(figsize=(20,20))
sns.countplot(y='Category',data=df)
df['modified_resume'] =''
df['modified_resume'] = df['Resume'].apply(lambda x: cleanResume(x))


import nltk
from nltk.corpus import stopwords
#from wordcloud import WordCloud
nltk.download('punkt')
nltk.download('stopwords')
import string

stopwordsList = set(stopwords.words('english')+['``',"''"])
sentences = df['Resume'].values   #returning only the values of the column without any axis label
cleanedSentences = ""
allwords =[]
for i in range(len(sentences)):
    cleanedText = cleanResume(sentences[i])        
    cleanedSentences += cleanedText
    words = nltk.word_tokenize(cleanedText)
    for word in words:
        if word not in stopwordsList and word not in string.punctuation:
            allwords.append(word)


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
catChange = ['Category']
for i in catChange:   
    df[i] = le.fit_transform(df[i])
#catChange


from sklearn.feature_extraction.text import TfidfVectorizer
requiredText = df['modified_resume'].values
cv = TfidfVectorizer(stop_words='english',max_features = 500)

WordFeatures = cv.fit_transform(requiredText).toarray()



from sklearn.model_selection import train_test_split

requiredTarget = df['Category'].values
X_train,X_test,y_train,y_test = train_test_split(WordFeatures,requiredTarget,random_state=0, test_size=0.3)
print("yesss")
print(X_train.shape)
print(X_test.shape)


clf = OneVsOneClassifier(KNeighborsClassifier(n_neighbors=13))
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)
print(clf.score(X_test, y_test))
print(prediction)
accuracy = accuracy_score(y_test, prediction)
f11 = f1_score(y_test,prediction,average='weighted')

print("F-score: ",f11)

print("Accuracy: ",accuracy)
#mlModel()

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="htmlviews")


@app.get("/")
def home(request:Request):
    return templates.TemplateResponse("index.html",{"request":request,"accuracy":accuracy})


@app.post("/submitfile",response_class=HTMLResponse)
async def func(request: Request,uploadedFile: UploadFile = File(...)):    
    print(uploadedFile.content_type)
    content_file = await uploadedFile.read()
    filepath = "docs/" + uploadedFile.filename.replace(" ","-")
    with open(filepath, 'wb') as f:        
        f.write(content_file)
        f.close()
    ans = solve(filepath)
    return templates.TemplateResponse("predict.html",{"request":request,"ans":ans})
    # return{
    #     "ans": ans
    # }
    

    #print(content_file)
def solve(filepath):
    resumeText = extractText(filepath)
    cleanedText = cleanResume(resumeText)
    text = re.split("curricular",cleanedText)
    print(text)
    textp = cv.transform([text[0]]).toarray()
    prediction2 = clf.predict(textp)
    print(prediction2[0]);
    predictedResult = le.inverse_transform(prediction2)[0]
    #return prediction2[0].item()   numpy.int32 to int
    return predictedResult

def extractText(filepath):
    with pdfplumber.open(filepath) as pdf:
        first_page = pdf.pages[0]
        resumeText = first_page.extract_text()
        #print(resumeText)
        pattern = r'[0-9]'
        # # Match all digits in the string and replace them with an empty string
        resumeText= re.sub(pattern, '', resumeText)
        return resumeText



# if __name__ == '__main__':
#   uvicorn.run(app, host='localhost', port=3050)
