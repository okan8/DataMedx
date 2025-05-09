import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
from sklearn.preprocessing import MultiLabelBinarizer
df = pd.read_excel("diyabet_dataset.xlsx")
dfsutunlar = ["Tüm tanılar","Tüm hizmetler","Tüm laboratuvar testler","Tüm radyoloji testler","Tüm ilaçlar"]
for sutun in dfsutunlar:
    df[sutun] = df[sutun].fillna("")
def tanı_ayır(x):
    x = str(x)
    if 'E10' in x:
        return 'E10'
    elif 'E11' in x:
        return 'E11'
    else:
        return 'Diğer'
df['hedef_tanı'] = df['Tüm tanılar'].apply(tanı_ayır)
def one_hot_ayir(df, kolon_adi):
    mlb = MultiLabelBinarizer()
    veriler = df[kolon_adi].apply(lambda x: [item.strip(" []") for item in x.split("], [") if item])
    onehot = pd.DataFrame(mlb.fit_transform(veriler), columns=[f"{kolon_adi}_{etiket}" for etiket in mlb.classes_])
    return onehot
ozellik_df_list = [one_hot_ayir(df, sutun) for sutun in dfsutunlar]
X = pd.concat(ozellik_df_list, axis=1)
y = df['hedef_tanı']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("\nTeşhis tahmin modeli (yüzeysel ve basit)")
print("Doğruluk:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
joblib.dump(model, "model_tani_tahmini.pkl")
print("\n✅ Model başarıyla eğitildi ve kaydedildi.")
