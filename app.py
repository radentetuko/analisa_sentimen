import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import re
from bs4 import BeautifulSoup
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.chrome.service import Service
from selenium import webdriver
from selenium.webdriver.common.by import By
import time

def scrape_tokopedia(url):
    if url:
        options = webdriver.ChromeOptions()
        options.add_argument("--start-maximized")
        driver = webdriver.Chrome(options=options)
        driver.get(url)

        time.sleep(2)

        # Scroll ke bawah dan mengklik ulasan
        driver.execute_script("window.scrollTo(0, window.scrollY + 300);")
        WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.CSS_SELECTOR, "button[data-testid='review']"))).click()
        
        reviews = []
        ratings = []
        while True:  
            soup = BeautifulSoup(driver.page_source, "html.parser")
            containers = soup.findAll('article', attrs={'class': 'css-72zbc4'})

            for container in containers:
                try:
                    review = container.find('span', attrs={'data-testid': 'lblItemUlasan'}).text
                    rating = container.find('div', attrs={'data-testid': 'icnStarRating'}).get('aria-label')
                    
                    reviews.append(review)
                    ratings.append(rating)
                except AttributeError:
                    continue

            time.sleep(2)
            try:
                WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.CSS_SELECTOR, "button[aria-label^='Laman berikutnya']"))).click()
            except:
                break
            time.sleep(3)

        driver.quit()

        return reviews, ratings

def preprocess(text):
    # Case Folding
    text = text.lower()
    # Cleansing
    text = re.sub(r'\W', ' ', text)
    # Penghapusan stopword
    stop_words = set(stopwords.words('indonesian'))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    text = ' '.join(words)
    # Stemming
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    text = stemmer.stem(text)
    return text

# Contoh URL produk teratas di Tokopedia
url = 'https://www.tokopedia.com/kaospolosmania/kaos-polo-shirt-pique-cvc-kpm-apparel-big-size-part-1-merah-cabe-2xl?extParam=ivf%3Dtrue%26src%3Dsearch'
reviews, ratings = scrape_tokopedia(url)

# Praproses ulasan
preprocessed_reviews = [preprocess(review) for review in reviews]

# Simpan data ke dalam DataFrame
data = pd.DataFrame({'Review': preprocessed_reviews, 'Rating': ratings})

# Simpan data ke dalam file CSV
data.to_csv('tokopedia_reviews.csv', index=False)

# Load data
data = pd.read_csv('tokopedia_reviews.csv')

# Hapus baris yang memiliki nilai NaN pada kolom Review
data = data.dropna(subset=['Review'])

# Saat menyimpan rating ke dalam DataFrame, ubah tipe datanya menjadi numerik
data['Rating'] = data['Rating'].str.extract('(\d+)').astype(int)

# Pisahkan ulasan produk dan non-produk
produk_reviews = data[data['Rating'] >= 4]['Review']
non_produk_reviews = data[data['Rating'] < 4]['Review']

# Ekstraksi fitur dengan TF-IDF
vectorizer = TfidfVectorizer()
produk_features = vectorizer.fit_transform(produk_reviews)
non_produk_features = vectorizer.transform(non_produk_reviews)

# Penanganan jika jumlah sampel kelas minoritas kurang dari 3
if len(data[data['Rating'] >= 4]) < 3:
    print("Jumlah sampel dalam kelas minoritas kurang dari 3.")
    exit()

# Penyeimbangan dataset dengan SMOTE
smote = SMOTE()
produk_features_resampled, produk_labels_resampled = smote.fit_resample(produk_features, data[data['Rating'] >= 4]['Rating'])

# Split data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(produk_features_resampled, produk_labels_resampled, test_size=0.2, random_state=42)

# Klasifikasi untuk analisis sentimen
svm_sentimen = SVC(kernel='linear')
svm_sentimen.fit(X_train, y_train)

# Evaluasi model
y_pred = svm_sentimen.predict(X_test)

# Menghitung metrik evaluasi
precision = precision_score(y_test, y_pred, pos_label=4)  # Atur pos_label sesuai label yang ada dalam data
recall = recall_score(y_test, y_pred, pos_label=4)
f1 = f1_score(y_test, y_pred, pos_label=4)

# Menampilkan metrik evaluasi
print("Precision: {:.2f}".format(precision))
print("Recall: {:.2f}".format(recall))
print("F1-score: {:.2f}".format(f1))

# Laporan klasifikasi
print(classification_report(y_test, y_pred))