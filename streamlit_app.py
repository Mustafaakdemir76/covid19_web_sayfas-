import streamlit as st
import torch
import torch.nn as nn
import numpy as np

# 1. MODEL MİMARİSİ (Hata mesajındaki OrderedDict yapısına %100 uyumlu)
class CovidModel(nn.Module):
    def __init__(self):
        super(CovidModel, self).__init__()
        self.katmanlar = nn.Sequential(
            nn.Linear(20, 64), # 20 Giriş
            nn.ReLU(),
            nn.Identity(),     # Katman indisi uyumu için (katmanlar.2)
            nn.Linear(64, 32), 
            nn.ReLU(),
            nn.Linear(32, 1),  
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.katmanlar(x)

# 2. MODELİ YÜKLE
@st.cache_resource
def load_model():
    model = CovidModel()
    try:
        # 'covid_model.pth' dosyasının adının doğru olduğundan emin olun
        state_dict = torch.load("covid_model.pth", map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Model dosyası yüklenemedi: {e}")
        return None

# 3. ARAYÜZ
st.set_page_config(page_title="Covid Risk Analizi", layout="wide")
st.title("🦠 Kesin Sonuçlu COVID-19 Analiz Sistemi")

model = load_model()

if model:
    with st.form("kesin_form"):
        col1, col2, col3 = st.columns(3)
        
        # CSV'deki sütun sırasına göre (DATE_DIED hariç tam 20 adet)
        with col1:
            usmer = st.number_input("USMER", 1, 2, 2)
            medunit = st.number_input("Medical Unit", 1, 13, 1)
            sex = st.selectbox("Cinsiyet", [1, 2], format_func=lambda x: "Kadın" if x==1 else "Erkek")
            p_type = st.selectbox("Hasta Tipi", [1, 2], format_func=lambda x: "Ayaktan" if x==1 else "Yatan")
            intubed = st.selectbox("Entübe", [1, 2, 97, 99])
            pneumonia = st.selectbox("Zatürre", [1, 2, 97, 99])
            age = st.number_input("Yaş", 0, 120, 30)
            
        with col2:
            pregnant = st.selectbox("Hamile", [1, 2, 97, 98])
            diabetes = st.selectbox("Diyabet", [1, 2, 98])
            copd = st.selectbox("KOAH", [1, 2, 98])
            asthma = st.selectbox("Astım", [1, 2, 98])
            inmsupr = st.selectbox("Bağışıklık Bask.", [1, 2, 98])
            hiper = st.selectbox("Hipertansiyon", [1, 2, 98])
            other = st.selectbox("Diğer Hastalık", [1, 2, 98])

        with col3:
            cardio = st.selectbox("Kalp Damar", [1, 2, 98])
            obesity = st.selectbox("Obezite", [1, 2, 98])
            renal = st.selectbox("Böbrek Yetmezliği", [1, 2, 98])
            tobacco = st.selectbox("Tütün", [1, 2, 98])
            classif = st.number_input("Sınıflandırma", 1, 7, 3)
            icu = st.selectbox("Yoğun Bakım", [1, 2, 97, 99])

        submit = st.form_submit_button("HESAPLA", use_container_width=True)

        if submit:
            # Liste tam olarak 20 adet olmalı
            veriler = [
                usmer, medunit, sex, p_type, intubed, pneumonia, age, 
                pregnant, diabetes, copd, asthma, inmsupr, hiper, 
                other, cardio, obesity, renal, tobacco, classif, icu
            ]
            
            # HESAPLAMA ADIMI
            try:
                # 1. Veriyi Tensor'a çevir
                input_tensor = torch.tensor([veriler], dtype=torch.float32)
                
                # 2. Tahmin yap
                with torch.no_grad():
                    cikti = model(input_tensor)
                    olasılık = cikti.item()
                
                # 3. Sonucu Yazdır
                st.divider()
                st.balloons()
                st.subheader(f"Tahmin Edilen Risk Skoru: %{olasılık*100:.2f}")
                
                if olasılık > 0.5:
                    st.error("YÜKSEK RİSK: Pozitif ihtimali kuvvetli.")
                else:
                    st.success("DÜŞÜK RİSK: Negatif ihtimali kuvvetli.")
            
            except Exception as e:
                st.error(f"Hesaplama sırasında teknik bir hata oluştu: {e}")
