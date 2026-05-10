import streamlit as st
import torch
import torch.nn as nn

# 1. MODEL MİMARİSİ
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
st.set_page_config(page_title="COVID-19 Tahmin Sistemi", page_icon="🦠", layout="wide")
st.title("🦠 Veri Odaklı COVID-19 Risk Analizi")
st.write("Veri setinizdeki parametrelere göre %100 uyumlu analiz yapar.")

# 3. MODELİ YÜKLE
@st.cache_resource
def load_model():
    model_path = "covid_model.pth"
    model = CovidModel()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Seçenekler (Veri setindeki standartlara göre: 1=Evet, 2=Hayır, 97-98-99=Bilinmiyor)
evet_hayir = {1: "Evet", 2: "Hayır", 97: "Bilinmiyor", 98: "Bilinmiyor", 99: "Bilinmiyor"}

try:
    model = load_model()
    
    with st.form("covid_form"):
        st.subheader("📋 Hasta Bilgileri (CSV Sütun Sırasına Göre)")
        c1, c2, c3, c4 = st.columns(4)
        
        inputs = []
        with c1:
            inputs.append(st.number_input("USMER (1 veya 2)", 1, 2, 2))
            inputs.append(st.number_input("Medical Unit (1-13)", 1, 13, 1))
            inputs.append(st.selectbox("Cinsiyet", [1, 2], format_func=lambda x: "Kadın" if x==1 else "Erkek"))
            inputs.append(st.selectbox("Hasta Tipi (1:Ayaktan, 2:Yatan)", [1, 2]))
            inputs.append(st.selectbox("Entübe mi?", [1, 2, 97, 99], format_func=lambda x: evet_hayir.get(x)))

        with c2:
            inputs.append(st.selectbox("Zatürre (Pnömoni)", [1, 2, 97, 99], format_func=lambda x: evet_hayir.get(x)))
            inputs.append(st.number_input("Yaş", 0, 120, 30))
            inputs.append(st.selectbox("Hamile mi?", [1, 2, 97, 98], format_func=lambda x: evet_hayir.get(x)))
            inputs.append(st.selectbox("Diyabet", [1, 2, 98], format_func=lambda x: evet_hayir.get(x)))
            inputs.append(st.selectbox("KOAH", [1, 2, 98], format_func=lambda x: evet_hayir.get(x)))

        with c3:
            inputs.append(st.selectbox("Astım", [1, 2, 98], format_func=lambda x: evet_hayir.get(x)))
            inputs.append(st.selectbox("Bağışıklık Baskılama", [1, 2, 98], format_func=lambda x: evet_hayir.get(x)))
            inputs.append(st.selectbox("Hipertansiyon", [1, 2, 98], format_func=lambda x: evet_hayir.get(x)))
            inputs.append(st.selectbox("Diğer Hastalıklar", [1, 2, 98], format_func=lambda x: evet_hayir.get(x)))
            inputs.append(st.selectbox("Kalp Damar Hast.", [1, 2, 98], format_func=lambda x: evet_hayir.get(x)))

        with c4:
            inputs.append(st.selectbox("Obezite", [1, 2, 98], format_func=lambda x: evet_hayir.get(x)))
            inputs.append(st.selectbox("Böbrek Yetmezliği", [1, 2, 98], format_func=lambda x: evet_hayir.get(x)))
            inputs.append(st.selectbox("Tütün Kullanımı", [1, 2, 98], format_func=lambda x: evet_hayir.get(x)))
            inputs.append(st.selectbox("Temas Durumu", [1, 2, 99], format_func=lambda x: evet_hayir.get(x)))
            inputs.append(st.selectbox("Yoğun Bakım", [1, 2, 97, 99], format_func=lambda x: evet_hayir.get(x)))

        submit = st.form_submit_button("RİSKİ HESAPLA", use_container_width=True)

        if submit:
            # Modelin beklediği 20 parametre tam burada Tensor'a dönüşüyor
            input_tensor = torch.tensor([inputs], dtype=torch.float32)
            
            with torch.no_grad():
                olasılık = model(input_tensor).item()
            
            st.divider()
            skor = olasılık * 100
            st.subheader(f"Analiz Sonucu: %{skor:.2f}")
            
            if skor > 50:
                st.error("⚠️ YÜKSEK RİSK: Veriler kuvvetle COVID-19 pozitifliği veya ağır seyir ihtimali gösteriyor.")
            else:
                st.success("✅ DÜŞÜK RİSK: Veriler şu an için düşük risk kategorisinde.")

except Exception as e:
    st.error(f"Hata: {e}")
