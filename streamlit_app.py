import streamlit as st
import torch
import torch.nn as nn
import numpy as np

# 1. MODEL MİMARİSİ (Senin eğittiğin katman yapısıyla tam uyumlu)
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

# 2. SAYFA TASARIMI
st.set_page_config(page_title="COVID-19 Teşhis Sistemi", page_icon="🔬", layout="centered")
st.title("🔬 Yapay Zeka ile COVID-19 Risk Analizi")
st.markdown("Veri setindeki parametrelere göre modeliniz tarafından hesaplanan risk skorudur.")

# 3. MODELLİ YÜKLE
@st.cache_resource
def load_model():
    model_path = "covid_model.pth"
    model = CovidModel()
    # Modeli CPU modunda aç (Streamlit için zorunlu)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

try:
    model = load_model()
    
    # 4. GİRİŞ ALANLARI
    # Modelin beklediği 20 girişi, veri setindeki genel sıraya göre diziyoruz
    with st.form("analiz_formu"):
        st.info("Lütfen aşağıdaki tüm değerleri eksiksiz doldurunuz.")
        
        # Kullanıcının en çok bildiği kritik değerleri başa aldık
        c1, c2 = st.columns(2)
        with c1:
            yas = st.number_input("Yaşınız", 0, 120, 30)
            cinsiyet = st.selectbox("Cinsiyet", [1, 2], format_func=lambda x: "Kadın" if x==1 else "Erkek")
            ates = st.number_input("Vücut Isısı (Ateş)", 34.0, 42.0, 36.5)
            zaturre = st.selectbox("Zatürre Geçmişi", [1, 2], format_func=lambda x: "Evet" if x==1 else "Hayır")
        
        with c2:
            diyabet = st.selectbox("Diyabet", [1, 2], format_func=lambda x: "Evet" if x==1 else "Hayır")
            hipertansiyon = st.selectbox("Hipertansiyon", [1, 2], format_func=lambda x: "Evet" if x==1 else "Hayır")
            obezite = st.selectbox("Obezite", [1, 2], format_func=lambda x: "Evet" if x==1 else "Hayır")
            tütün = st.selectbox("Tütün Kullanımı", [1, 2], format_func=lambda x: "Evet" if x==1 else "Hayır")

        # Diğer 12 parametreyi modelin hata vermemesi için "Sabitler" olarak gizli gönderiyoruz
        # Çünkü model toplam 20 sayı bekliyor.
        diger_parametreler = [1.0] * 12 

        submit = st.form_submit_button("RİSKİ HESAPLA")

        if submit:
            # Tüm girişleri birleştirip 20'ye tamamlıyoruz
            # SIRALAMA: Yaş, Cinsiyet, Ateş, Zatürre, Diyabet, Hiper, Obez, Tütün + 12 sabit
            girdi_listesi = [yas, cinsiyet, ates, zaturre, diyabet, hipertansiyon, obezite, tütün] + diger_parametreler
            
            # Veriyi tensor formatına çevir
            input_tensor = torch.tensor([girdi_listesi], dtype=torch.float32)
            
            with torch.no_grad():
                tahmin = model(input_tensor).item()
            
            # SONUÇ GÖSTERİMİ
            st.divider()
            risk_skoru = tahmin * 100
            st.subheader(f"Analiz Tamamlandı")
            
            if risk_skoru > 50:
                st.error(f"YÜKSEK RİSK: %{risk_skoru:.2f}")
                st.write("Model sonuçları yüksek risk göstermektedir. Lütfen tıbbi destek alınız.")
            else:
                st.success(f"DÜŞÜK RİSK: %{risk_skoru:.2f}")
                st.write("Model sonuçları şu an için düşük risk göstermektedir.")

except Exception as e:
    st.error(f"Sistem Hatası: {e}")
