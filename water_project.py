import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import plotly.express as px

df=pd.read_csv('water.csv')
print(df.head())

cleaned_df=df.dropna()
print(cleaned_df)
print(cleaned_df.isnull().sum())

print('stampiamo la distribuzione di acqua potabile e non potabile')
plt.figure(figsize=(15,10))
sns.countplot(cleaned_df.Potability)
plt.title('Distribution of Unsafe and Safe Water')
plt.show()

print("stampiamo i fattori che influenzano la qualitá dell'acqua")
df=df
figure=px.histogram(df, x= "ph", color="Potability", title="Factors Affecting Water Quality:PH")
figure.show()

print("stampiamo i fattori che influenzano la durezza dell'acqua")
figure=px.histogram(df, x= 'Hardness', color= 'Potability', title="Factors Affecting Water Quality: Hardness")
figure.show()

print("stampiamo la distribuzione dei solidi totali disciolti nell'acqua")
figure=px.histogram(df, x= 'Solids', color= 'Potability', title="Factors Affecting Water Quality: Solids")
figure.show()

print("stampiamo la distribuzione della cloramina nell'acqua ")
figure=px.histogram(df, x= 'Chloramines', color= 'Potability', title="Factors Affecting Water Quality: Chloramine")
figure.show()

print("stampiamo la distribuzione del solfato nell'acqua")
figure=px.histogram(df, x= 'Sulfate', color= 'Potability', title="Factors Affecting Water Quality: Sulfate")
figure.show()

print("stampiamo la distribuzione della conduttività dell'acqua ")
figure=px.histogram(df, x= 'Conductivity', color= 'Potability', title="Factors Affecting Water Quality: Conductivity")
figure.show()

print("stampiamo la distribuzione del carbonio organico nell'acqua")
figure=px.histogram(df, x= 'Organic_carbon', color= 'Potability', title="Factors Affecting Water Quality: Organic_carbon")
figure.show()

print("stampiamo la distribuzione dei trialometani o THM nell'acqua")
figure=px.histogram(df, x= 'Trihalomethanes', color= 'Potability', title="Factors Affecting Water Quality: Trihalomethanes")
figure.show()

print("stampiamo la distribuzione della torbidità nell'acqua")
figure=px.histogram(df, x= 'Turbidity', color= 'Potability', title="Factors Affecting Water Quality: Turbidity")
figure.show()

print('calcolo massimo torbitá é')
print(df['Turbidity'].max())

print('calcolo minimo torbitá é')
print(df['Turbidity'].min())

print('stampiamo quanti PH sono maggiori di 8')
ph_maggiore_di_otto= df[df['ph']>8]
print(ph_maggiore_di_otto)
print('dal risultato in alto vediamo in uscita 731 righe che hanno il pH maggiore di 8 su 10 colonne presente nel dataset')


print('stampiamo correlazione potabilitá')
correlation=df.corr()
correlation['ph'].sort_values(ascending=False)
print(correlation)


## Il codice in basso non funziona su pycharm ma solo su Jupyter Nootebook.
from pycaret.classification import*
clf=setup(df, target='Potability', silent= True, session_id=786)
compare_models()

model=create_model('rf')
predict=-predict_model(model, df=df)
print(predict.head())