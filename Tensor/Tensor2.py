# -*- coding: utf-8 -*-
"""Tensor2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/10CkB3iysqa_2ArmUZFhJjR77WvNTGpz-
"""

import numpy as np
import matplotlib.pyplot as plt

#özellik kele
X= np.array([-7.0, -4.0, -1.0, 2.0, 5.0, 8.0, 11.0, 14.0])

## Etiket oluştur
y= np.array([3.0,6.0,9.0,12.0,15.0, 18.0,21.0, 24.0])

# Göster

plt.scatter(X,y)

"""### **Regresyon girdi ve çıktı şekilleri**

Giriş şekli, modelin içine giden verinin şeklidir.
Modelden çekilen datanın şekli ise çıkış şeklidir.

Problem şekline göre girdi ve çıktı şekilleri değişebilir.

Sinir ağları sayıları ve çıktı sayılarını kabul eder. Bu sayılar tipik olarak tensörler (veya diziler) olarak temsil edilir.


"""

## Bir regresyon modelinin girdi ve çıktı şekillerinin örneği
import tensorflow as tf


hause_info = tf.constant(["bed", "bath", "garage"])
hause_price = tf.constant([939700])

hause_info, hause_price

hause_info.shape

import numpy as np
import matplotlib.pyplot as plt

# Tensor kullanrak özellik oluşturma
X = tf.constant([-7.0, -4.0, -1.0, 2.0, 5.0, 8.0, 11.0, 14.0])

# Tensor kullanarak etiket oluşturma
y = tf.constant([3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0])

# Görselleştirme
plt.scatter(X, y);

"""Yukarıdaki grafikte ki amaç, ileride X eksenini kullanarak Y eksenini tahmin etmektir.

#**Tensorflowda modelleme**
"""

# Random seed ayarlama (rastgele sayı)
tf.random.set_seed(42)

# Sequential API ile model oluşturma
model = tf.keras.Sequential([
  tf.keras.layers.Dense(1)
])

# Modeli derleme
model.compile(loss=tf.keras.losses.mae, # mae ortalama mutlak hata teriminin kısaltmasıdır.
              optimizer=tf.keras.optimizers.SGD(), # SGD stokastik gradyan azalışın kısaltmasıdır. Adam vs. kullnılabilir.
              metrics=["mae"])

# Fit the model

model.fit(tf.expand_dims(X, axis=-1), y, epochs=5)

##tahminleme

X,y
model.predict([17.0])

"""# **Model geliştirme**"""

# Random seed atama
tf.random.set_seed(42)

# Model oluşturma
model = tf.keras.Sequential([
  tf.keras.layers.Dense(1)
])

# Model derleme
model.compile(loss=tf.keras.losses.mae,
              optimizer=tf.keras.optimizers.SGD(),
              metrics=["mae"])

# Fit model (this time we'll train for longer)
model.fit(tf.expand_dims(X, axis=-1), y, epochs=100) # train for 100 epochs not 10

X,y

model.predict([17.0])



"""# **MODEL Değerlendirmesi**"""

# Daha büyük bir dataset yapımı
X = np.arange(-100, 100, 4)
X

# Dataset için çıktı yapımı
y = np.arange(-90, 110, 4)
y

# Aynı sonucu verecektir.
y = X + 10
y

"""# **Datasetleri train ve test setineayırma**"""

# Ne kadar örneğin olduğunu tespit ediyoruz.
len(X)

# Splitleme işlemi
X_train = X[:40] # İlk 40 örnek
y_train = y[:40]

X_test = X[40:] # Son 10 örnek
y_test = y[40:]

len(X_train), len(X_test)

"""#**Datayı görselleştirme**"""

plt.figure(figsize=(10, 7))
# Train verisini mavi ile gösterme
plt.scatter(X_train, y_train, c='b', label='Training data')
# Test verisini yeşil ile gösterme
plt.scatter(X_test, y_test, c='g', label='Testing data')
# Lejantı gösterme
plt.legend();

# Random seed belirleme
tf.random.set_seed(42)

# Model yapımı
model = tf.keras.Sequential([
  tf.keras.layers.Dense(1)
])

# Modelin derlenmesi
model.compile(loss=tf.keras.losses.mae,
              optimizer=tf.keras.optimizers.SGD(),
              metrics=["mae"])

"""# **Modelin Görselleştirilmesi**"""

# Çalışmaz
model.summary()

"""input shape verilmedi, eğer verilmesse keras kendi yapmayı dener.

"""

# Random seed atama
tf.random.set_seed(42)

# Model yapımı
model = tf.keras.Sequential([
  tf.keras.layers.Dense(1, input_shape=[1]) # input shape tanımlama
])

# Model derlemesi
model.compile(loss=tf.keras.losses.mae,
              optimizer=tf.keras.optimizers.SGD(),
              metrics=["mae"])

# Input shape özelleştirildikten sonra çalışacaktır.
model.summary()

""""Modelımız üzerinde summary() çağırarak, içerdiği katmanları, çıkış şeklini ve parametrelerin sayısını görebiliriz.

Toplam parametreler - modeldeki toplam parametre sayısı.
Eğitilebilir parametreler - bu parametreler (kalıplar), modelin eğitim sırasında güncelleyebileceği parametrelerdir.
Eğitilemeyen parametreler - bu parametreler, eğitim sırasında güncellenmeyen parametrelerdir (bu, transfer öğrenme sırasında zaten öğrenilmiş kalıpları diğer modellere getirdiğinizde tipiktir)."
"""

# Model entegrasyonu
model.fit(X_train, y_train, epochs=100, verbose=0) # Verbose, ne kadar çıktının görüntülendiğini kontrol eder.

model.summary()

from tensorflow.keras.utils import plot_model

plot_model(model, show_shapes=True)

"""# **Tahminleri Görüntüleme**"""

y_preds = model.predict(X_test)

y_preds

def plot_predictions(train_data=X_train,
                     train_labels=y_train,
                     test_data=X_test,
                     test_labels=y_test,
                     predictions=y_preds):
  """
  Train ve test datasını çizer ve tahminleri karşılaştırır.
  """
  plt.figure(figsize=(10, 7))
  # Train datasını mavi ile çizer.
  plt.scatter(train_data, train_labels, c="b", label="Training data")
  # Test datasını yeşil ile çizer
  plt.scatter(test_data, test_labels, c="g", label="Testing data")
  # Tahminleri kırmızı ile çizer (Tahminler test datasının üzerinde yapılır unutma)
  plt.scatter(test_data, predictions, c="r", label="Predictions")
  # Lejant çizimi
  plt.legend();

plot_predictions(train_data=X_train,
                 train_labels=y_train,
                 test_data=X_test,
                 test_labels=y_test,
                 predictions=y_preds)

# Modeli test setinde değerlendirme
model.evaluate(X_test, y_test)

# Ortalama kare değer hesabı
mae = tf.metrics.mean_absolute_error(y_true=y_test,
                                     y_pred=y_preds)
mae
#y test ve y train farklı shapelerde olduğu için tek çıktı vermedi.

y_test

y_preds

y_test.shape, y_preds.shape

#squeeze öncesi kontrol
y_preds.shape

#  squeeze() sonrası
y_preds.squeeze().shape

y_test, y_preds.squeeze()

#Tekrardan mae kontrolü
mae = tf.metrics.mean_absolute_error(y_true=y_test,
                                     y_pred=y_preds.squeeze()) # use squeeze() to make same shape
mae

# Mse kontrolü
mse = tf.metrics.mean_squared_error(y_true=y_test,
                                    y_pred=y_preds.squeeze())
mse

# Returns the same as tf.metrics.mean_absolute_error()
tf.reduce_mean(tf.abs(y_test-y_preds.squeeze()))

"""# **Model geliştirme**

"Değerlendirme metriklerini ve modelinizin yaptığı tahminleri gördükten sonra, muhtemelen modelinizi iyileştirmek isteyeceksiniz.

Tekrar belirtmek gerekirse, bunu yapmanın birçok farklı yolu vardır, ancak bunların başlıcası 3 tanesidir:

Daha fazla veri elde etmek - modelinizin öğrenme fırsatları için daha fazla örnek almak (daha fazla kalıp öğrenme şansı).
Modelinizi büyütmek (daha karmaşık bir model kullanmak) - bu, daha fazla katman veya her katmandaki daha fazla gizli birimler şeklinde gelebilir.
Daha uzun süre eğitmek - modelinize verilerdeki kalıpları bulma şansı vermek.

Veri kümesini kendimiz oluşturduğumuz için, daha fazla veri oluşturmak kolay olabilir, ancak gerçek dünya veri kümesiyle çalışırken her zaman böyle olmayabilir.

Şimdi, modelimizi 2 ve 3 ile nasıl iyileştirebileceğimize bir göz atalım.

Bunu yapmak için, 3 model oluşturacağız ve sonuçlarını karşılaştıracağız:

model_1 - orijinal modelle aynı, 1 katman, 100 epok eğitildi.
model_2 - 2 katman, 100 epok eğitildi.
model_3 - 2 katman, 500 epok eğitildi."
"""

# Random seed belirleme
tf.random.set_seed(42)

# Orijinal modelin kopyası
model_1 = tf.keras.Sequential([
  tf.keras.layers.Dense(1)
])

# Model derleme
model_1.compile(loss=tf.keras.losses.mae,
                optimizer=tf.keras.optimizers.SGD(),
                metrics=['mae'])

# Model entegresi
model_1.fit(tf.expand_dims(X_train, axis=-1), y_train, epochs=100)

y_preds_1 = model_1.predict(X_test)
plot_predictions(predictions=y_preds_1)

# Model1 metriklerini yazalım
mae_1 = tf.metrics.mean_absolute_error(y_test, y_preds_1.squeeze()).numpy()
mse_1 = tf.metrics.mean_squared_error(y_test, y_preds_1.squeeze()).numpy()
mae_1, mse_1

tf.random.set_seed(42)

# Model1 kopyası ve fazladan katman ekleme
model_2 = tf.keras.Sequential([
  tf.keras.layers.Dense(1),
  tf.keras.layers.Dense(1) # add a second layer
])


model_2.compile(loss=tf.keras.losses.mae,
                optimizer=tf.keras.optimizers.SGD(),
                metrics=['mae'])


model_2.fit(tf.expand_dims(X_train, axis=-1), y_train, epochs=100, verbose=0) # set verbose to 0 for less output

y_preds_2 = model_2.predict(X_test)
plot_predictions(predictions=y_preds_2)

mae_2 = tf.metrics.mean_absolute_error(y_test, y_preds_2.squeeze()).numpy()
mse_2= tf.metrics.mean_squared_error(y_test, y_preds_2.squeeze()).numpy()
mae_2, mse_2

tf.random.set_seed(42)

# moled2 kopyası
model_3 = tf.keras.Sequential([
  tf.keras.layers.Dense(1),
  tf.keras.layers.Dense(1)
])

model_3.compile(loss=tf.keras.losses.mae,
                optimizer=tf.keras.optimizers.SGD(),
                metrics=['mae'])

# Fit the model (500 epoch ile)
model_3.fit(tf.expand_dims(X_train, axis=-1), y_train, epochs=500, verbose=0) # set verbose to 0 for less output

# Make and plot predictions for model_3
y_preds_3 = model_3.predict(X_test)
plot_predictions(predictions=y_preds_3)

mae_3 = tf.metrics.mean_absolute_error(y_test, y_preds_3.squeeze()).numpy()
mse_3 = tf.metrics.mean_squared_error(y_test, y_preds_3.squeeze()).numpy()
mae_3, mse_3

"""# Sonuçları karşılaştırma"""

model_results = [["model_1", mae_1, mse_1],
                 ["model_2", mae_2, mse_2],
                 ["model_3", mae_3, mae_3]]

import pandas as pd
all_results = pd.DataFrame(model_results, columns=["model", "mae", "mse"])
all_results

"""# **Modelin kaydedilmesi**"""

# Modelin kaydı
model_2.save('best_model_SavedModel_format')

#"Bu kontrol eder - bir protobuf ikili dosyası (.pb) ve diğer dosyaları çıkarır."
!ls best_model_SavedModel_format

#HADF5 formatında kaydı
model_2.save("best_model_HDF5_format.h5") # note the addition of '.h5' on the end

"""# **Modelin yüklenmesi**"""

#Modelin yüklenmesiç
loaded_saved_model = tf.keras.models.load_model("best_model_SavedModel_format")
loaded_saved_model.summary()

# model_2yi SavedModel versiyonu ile karşılaştırıyoruz ( True döndürmeli)
mae_saved_model = tf.metrics.mean_absolute_error(y_test, saved_model_preds.squeeze()).numpy()
mae_model_2 = tf.metrics.mean_absolute_error(y_test, model_2_preds.squeeze()).numpy()
mae_saved_model == mae_model_2

# Load a model from the HDF5 format
loaded_h5_model = tf.keras.models.load_model("best_model_HDF5_format.h5")
loaded_h5_model.summary()

"""# **Colabten model çekme**"""

from google.colab import files
files.download("best_model_HDF5_format.h5")

"""# **Daha geniş bir örnek**"""

import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

insurance = pd.read_csv("https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv")

insurance.head()

# one hot encoding
insurance_one_hot = pd.get_dummies(insurance)
insurance_one_hot.head() # değiştirilen kolonları gösterir

# X &y değerlerinin oluşturulması
X = insurance_one_hot.drop("charges", axis=1)
y = insurance_one_hot["charges"]

X.head()

# Train ve test setlerinin oluşturulması
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42) # rastgele durum sayısı ayarı

tf.random.set_seed(42)

insurance_model = tf.keras.Sequential([
  tf.keras.layers.Dense(1),
  tf.keras.layers.Dense(1)
])


insurance_model.compile(loss=tf.keras.losses.mae,
                        optimizer=tf.keras.optimizers.SGD(),
                        metrics=['mae'])

insurance_model.fit(X_train, y_train, epochs=100)

insurance_model.evaluate(X_test, y_test)

"""Modelimiz güzel olmadı, daha büyük bir model ile devam edeceğiz

Üç şey deneyeceğiz:

1. Katman sayısını artırmak (2'den 3'e).
2. Her katmandaki birim sayısını artırmak (çıkış katmanı hariç).
3. Optimizasyon algoritmasını değiştirmek (SGD'den Adam'a).

Diğer her şey aynı kalacak.
"""

tf.random.set_seed(42)

# Fazladan katman ekledik ve katmanlardaki ünite sayısını artırdık.
insurance_model_2 = tf.keras.Sequential([
  tf.keras.layers.Dense(100), # 100 unite
  tf.keras.layers.Dense(10), # 10 unite
  tf.keras.layers.Dense(1) # 1 unite (important for output layer)
])


insurance_model_2.compile(loss=tf.keras.losses.mae,
                          optimizer=tf.keras.optimizers.Adam(), # Adam works but SGD doesn't
                          metrics=['mae'])

history = insurance_model_2.fit(X_train, y_train, epochs=100, verbose=0)

insurance_model_2.evaluate(X_test, y_test)

pd.DataFrame(history.history).plot()
plt.ylabel("loss")
plt.xlabel("epochs");

# Daha fazla epoch ile deniyoruz.
history_2 = insurance_model_2.fit(X_train, y_train, epochs=100, verbose=0)

# Model değerlendirmesi.
insurance_model_2_loss, insurance_model_2_mae = insurance_model_2.evaluate(X_test, y_test)
insurance_model_2_loss, insurance_model_2_mae

pd.DataFrame(history_2.history).plot()
plt.ylabel("loss")
plt.xlabel("epochs"); #Not: Eğitim geçmişini geçersiz kıldığımız için yalnızca 100 epoch görünecek.

## Veri ön işleme

"""Nöral ağlarla çalışırken yaygın bir uygulama, onlara ilettiğiniz tüm verilerin 0 ile 1 aralığında olduğundan emin olmaktır.

Bu uygulamaya normalizasyon denir (tüm değerlerin orijinal aralıklarından örneğin 0 ile 1 arasına ölçeklendirilmesi).

Başka bir işlem de standardizasyon olarak adlandırılır ve tüm verilerin birim varyans ve 0 ortalama değere sahip olmasını sağlar.

Bu iki uygulama genellikle bir ön işleme boru hattının (verilerin nöral ağlarla kullanım için hazırlanmasını sağlamak için kullanılan bir dizi işlev) bir parçasıdır.

Bu bilgiler ışığında, verilerinizi nöral ağlar için ön işleme alırken atacağınız başlıca adımlar şunlar olacaktır:

1. Tüm verilerinizi sayılara dönüştürmek (nöral ağlar metinleri işleyemez).
2. Verilerinizin doğru şekilde olduğundan emin olmak (giriş ve çıkış şekillerini doğrulamak).
3. Özellik ölçekleme:
   - Verileri normalleme (tüm değerlerin 0 ile 1 arasında olduğundan emin olmak). Bu, minimum değeri çıkarmak ve ardından maksimum değeri minimum değeri çıkartıp bölerek yapılır. Bu ayrıca min-max ölçeklendirme olarak da adlandırılır.
   - Standartlaştırma (tüm değerlerin ortalama değeri 0 ve varyansı 1 olduğundan emin olmak). Bu, hedef özelliğin ortalamasını çıkartmak ve ardından standart sapmayla bölmek suretiyle yapılır.

Hangi yöntemi kullanmalısınız?
Nöral ağlarla çalışırken genellikle normalizasyonu tercih edersiniz, çünkü genellikle 0 ile 1 arasındaki değerleri tercih ederler (bu özellikle görüntü işlemeyle özellikle görülecektir). Bununla birlikte, nöral ağların genellikle minimum özellik ölçeklemesiyle iyi performans gösterebileceğini göreceksiniz.
"""

import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
insurance = pd.read_csv("https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv")

insurance.head()

from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

# Create column transformer (this will help us normalize/preprocess our data)
ct = make_column_transformer(
    (MinMaxScaler(), ["age", "bmi", "children"]), # get all values between 0 and 1
    (OneHotEncoder(handle_unknown="ignore"), ["sex", "smoker", "region"])
)

X = insurance.drop("charges", axis=1)
y = insurance["charges"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

## Sütun transformatörünü yalnızca eğitim verilerine uyguladık (bunu test verilerinde yapmak veri sızıntısına neden olur)
ct.fit(X_train)

# Normalleştirme (MinMaxScalar) ve bir etkin kodlama (OneHotEncoder) ile eğitim ve test verilerini dönüştürdük
X_train_normal = ct.transform(X_train)
X_test_normal = ct.transform(X_test)

X_train.loc[0]

X_train_normal[0]

X_train_normal.shape, X_train.shape
# Ekstra sütunlar nedeniyle normalleştirilmiş/one-hot encoded şeklin daha büyük olduğuna dikkat et.

tf.random.set_seed(42)


insurance_model_3 = tf.keras.Sequential([
  tf.keras.layers.Dense(100),
  tf.keras.layers.Dense(10),
  tf.keras.layers.Dense(1)
])


insurance_model_3.compile(loss=tf.keras.losses.mae,
                          optimizer=tf.keras.optimizers.Adam(),
                          metrics=['mae'])

insurance_model_3.fit(X_train_normal, y_train, epochs=200, verbose=0)

insurance_model_3_loss, insurance_model_3_mae = insurance_model_3.evaluate(X_test_normal, y_test)

insurance_model_2_mae, insurance_model_3_mae