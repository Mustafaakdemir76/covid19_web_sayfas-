import streamlit as st
import torch
import torch.nn as nn

# 1. BİREBİR AYNI MODEL MİMARİSİ
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

@st.cache_resource
def load_model():
    model = CovidModel()
    try:
        model.load_state_dict(torch.load("covid_model.pth", map_location=torch.device('cpu')))
        model.eval()
        return model
    except Exception as e:
        st.error(f"Model yüklenemedi: {e}")
        return None

st.set_page_config(page_title="Covid Tahmin", layout="wide")
st.title("🦠 COVID-19 Akıllı Analiz Sistemi")

model = load_model()

if model:
    with st.form("analiz_formu"):
        c1, c2, c3 = st.columns(3)
        
        with c1:
            usmer = st.selectbox("USMER", [1, 2], index=1)
            medunit = st.slider("Medical Unit", 1, 13, 1)
            sex = st.selectbox("Cinsiyet", [1, 2], format_func=lambda x: "Kadın" if x==1 else "Erkek")
            p_type = st.selectbox("Hasta Tipi", [1, 2], format_func=lambda x: "Ayaktan" if x==1 else "Yatan")
            intubed = st.selectbox("Entübe mi?", [1, 2, 97, 99])
            pneumonia = st.selectbox("Zatürre?", [1, 2, 97, 99])
            
        with c2:
            age = st.number_input("Yaş", 0, 120, 30)
            pregnant = st.selectbox("Hamilelik", [1, 2, 97, 98])
            diabetes = st.selectbox("Diyabet", [1, 2, 98])
            copd = st.selectbox("KOAH", [1, 2, 98])
            asthma = st.selectbox("Astım", [1, 2, 98])
            inmsupr = st.selectbox("Bağışıklık Bask.", [1, 2, 98])
            hiper = st.selectbox("Hipertansiyon", [1, 2, 98])

        with c3:
            other = st.selectbox("Diğer Hastalık", [1, 2, 98])
            cardio = st.selectbox("Kalp Damar", [1, 2, 98])
            obesity = st.selectbox("Obezite", [1, 2, 98])
            renal = st.selectbox("Böbrek Yetmezliği", [1, 2, 98])
            tobacco = st.selectbox("Tütün Kullanımı", [1, 2, 98])
            classif = st.slider("Sınıflandırma Skoru", 1, 7, 3)
            icu = st.selectbox("Yoğun Bakım", [1, 2, 97, 99])

        submit = st.form_submit_button("HESAPLA VE ANALİZ ET", use_container_width=True)

        if submit:
            # Eğitimde uyguladığımız "Bilinmeyen (97-99) değerleri temizleme" işlemi
            def temizle(val):
                return 2.0 if val > 2 else float(val)

            veriler = [
                float(usmer), float(medunit), float(sex), float(p_type), 
                temizle(intubed), temizle(pneumonia), float(age)/100.0, # Eğitimdeki gibi yaş/100
                temizle(pregnant), temizle(diabetes), temizle(copd), 
                temizle(asthma), temizle(inmsupr), temizle(hiper), 
                temizle(other), temizle(cardio), temizle(obesity), 
                temizle(renal), temizle(tobacco), float(classif), temizle(icu)
            ]
            
            input_tensor = torch.tensor([veriler], dtype=torch.float32)
            
            try:
                with torch.no_grad():
                    cikti = model(input_tensor).item()
                
                st.divider()
                # Eşik değeri %50 (Dengeli eğitim yaptığımız için model artık cesur davranacak)
                st.subheader(f"Mortalite / Ağır Seyir Risk Skoru: %{cikti*100:.2f}")
                
                if cikti > 0.50: 
                    st.error("⚠️ YÜKSEK RİSK: Veriler klinik olarak yüksek risk grubuna işaret etmektedir.")
                else:
                    st.success("✅ DÜŞÜK RİSK: Veriler düşük risk grubuna işaret etmektedir.")
            except Exception as e:
                st.error(f"Hesaplama Hatası: {e}")
