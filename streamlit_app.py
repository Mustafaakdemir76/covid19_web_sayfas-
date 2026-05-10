import streamlit as st
import torch
import torch.nn as nn

# 1. MODEL MİMARİSİ (Senin model yapınla birebir aynı)
class CovidModel(nn.Module):
    def __init__(self):
        super(CovidModel, self).__init__()
        self.katmanlar = nn.Sequential(
            nn.Linear(20, 64), 
            nn.ReLU(),
            nn.Identity(),     
            nn.Linear(64, 32), 
            nn.ReLU(),
            nn.Linear(32, 1),  
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.katmanlar(x)

# 2. SAYFA AYARLARI
st.set_page_config(page_title="Covid Tahmin", page_icon="🦠")
st.title("🦠 COVID-19 Tahmin Sistemi")

# 3. MODELLİ YÜKLE
@st.cache_resource
def load_model():
    try:
        model = CovidModel()
        # Modeli CPU'da çalışacak şekilde yükle
        model.load_state_dict(torch.load("covid_model.pth", map_location=torch.device('cpu')))
        model.eval()
        return model
    except Exception as e:
        st.error(f"Model yüklenirken hata oluştu: {e}")
        return None

model = load_model()

if model:
    with st.form("hesaplama_formu"):
        st.subheader("Lütfen Bilgileri Giriniz")
        
        # VERİ SETİNDEKİ (CSV) TAM SIRALAMA (DATE_DIED HARİÇ)
        # Modelin 'Size Mismatch' vermemesi için bu 20 değer şarttır.
        c1, c2 = st.columns(2)
        with c1:
            usmer = st.number_input("USMER (1-2)", 1, 2, 2)
            med_unit = st.number_input("Medical Unit (1-13)", 1, 13, 1)
            sex = st.selectbox("Cinsiyet (1:K, 2:E)", [1, 2])
            p_type = st.selectbox("Hasta Tipi (1:Ayaktan, 2:Yatan)", [1, 2])
            intubed = st.selectbox("Entübe (1:E, 2:H, 97:NA)", [1, 2, 97])
            pneumonia = st.selectbox("Zatürre (1:E, 2:H)", [1, 2])
            age = st.number_input("Yaş", 0, 120, 30)
            pregnant = st.selectbox("Hamile (1:E, 2:H, 98:NA)", [1, 2, 98])
            diabetes = st.selectbox("Diyabet (1:E, 2:H)", [1, 2])
            copd = st.selectbox("KOAH (1:E, 2:H)", [1, 2])
        
        with c2:
            asthma = st.selectbox("Astım (1:E, 2:H)", [1, 2])
            inmsupr = st.selectbox("Bağışıklık Bask. (1:E, 2:H)", [1, 2])
            hiper = st.selectbox("Hipertansiyon (1:E, 2:H)", [1, 2])
            other = st.selectbox("Diğer Hastalıklar (1:E, 2:H)", [1, 2])
            cardio = st.selectbox("Kalp Damar (1:E, 2:H)", [1, 2])
            obesity = st.selectbox("Obezite (1:E, 2:H)", [1, 2])
            renal = st.selectbox("Böbrek Yetmezliği (1:E, 2:H)", [1, 2])
            tobacco = st.selectbox("Sigara/Tütün (1:E, 2:H)", [1, 2])
            classif = st.number_input("Sınıflandırma (1-7)", 1, 7, 3)
            icu = st.selectbox("Yoğun Bakım (1:E, 2:H, 97:NA)", [1, 2, 97])

        submit = st.form_submit_button("HESAPLA")

        if submit:
            # Liste sırası CSV'deki orijinal sıralama ile aynı olmalı (DATE_DIED atlandı)
            girdiler = [
                usmer, med_unit, sex, p_type, intubed, pneumonia,
                age, pregnant, diabetes, copd, asthma, inmsupr,
                hiper, other, cardio, obesity, renal, tobacco, classif, icu
            ]
            
            # 20 sütunluk veriyi modele gönder
            tensor_girdi = torch.tensor([girdiler], dtype=torch.float32)
            
            try:
                with torch.no_grad():
                    cikti = model(tensor_girdi).item()
                
                st.divider()
                st.write(f"### Analiz Sonucu: %{cikti*100:.2f}")
                if cikti > 0.5:
                    st.error("Yüksek Risk Tespit Edildi!")
                else:
                    st.success("Düşük Risk.")
            except Exception as calc_error:
                st.error(f"Hesaplama hatası: {calc_error}")
