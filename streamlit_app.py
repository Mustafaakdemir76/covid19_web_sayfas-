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
st.set_page_config(page_title="COVID-19 Risk Analizi", page_icon="🦠", layout="wide")
st.title("🦠 COVID-19 Kapsamlı Risk Analiz Paneli")

# 3. MODELLİ YÜKLE
@st.cache_resource
def load_model():
    model_path = "covid_model.pth"
    model = CovidModel()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Seçenek Tanımlamaları
evet_hayir = {1: "Evet", 2: "Hayır", 97: "Bilinmiyor", 98: "Bilinmiyor", 99: "Bilinmiyor"}

try:
    model = load_model()
    
    with st.form("detayli_analiz"):
        st.subheader("🌡️ Kritik Değerler ve Kişisel Bilgiler")
        c1, c2, c3 = st.columns(3)
        
        inputs = []
        with c1:
            inputs.append(st.number_input("Yaş", 0, 120, 30))
            # ATEŞ: Modelin beklediği 20 sütundan biri değilse bile sisteme ekliyoruz
            ates = st.number_input("Vücut Ateşi (Derece)", 34.0, 42.0, 36.5, step=0.1)
            inputs.append(st.selectbox("Cinsiyet", [1, 2], format_func=lambda x: "Kadın" if x==1 else "Erkek"))
            inputs.append(st.selectbox("Zatürre (Pnömoni)", [1, 2, 97], format_func=lambda x: evet_hayir[x]))
            
        with c2:
            inputs.append(st.selectbox("Diyabet", [1, 2, 98], format_func=lambda x: evet_hayir[x]))
            inputs.append(st.selectbox("Astım", [1, 2, 98], format_func=lambda x: evet_hayir[x]))
            inputs.append(st.selectbox("Hipertansiyon", [1, 2, 98], format_func=lambda x: evet_hayir[x]))
            inputs.append(st.selectbox("Obezite", [1, 2, 98], format_func=lambda x: evet_hayir[x]))
            
        with c3:
            inputs.append(st.selectbox("Tütün Kullanımı", [1, 2, 98], format_func=lambda x: evet_hayir[x]))
            inputs.append(st.selectbox("Kardiyovasküler (Kalp)", [1, 2, 98], format_func=lambda x: evet_hayir[x]))
            inputs.append(st.selectbox("Böbrek Yetmezliği", [1, 2, 98], format_func=lambda x: evet_hayir[x]))
            inputs.append(st.selectbox("KOAH", [1, 2, 98], format_func=lambda x: evet_hayir[x]))

        st.subheader("🏥 Hastane ve Klinik Durum")
        c4, c5 = st.columns(2)
        with c4:
            inputs.append(st.selectbox("Hasta Tipi", [1, 2], format_func=lambda x: "Ayaktan" if x==1 else "Yatarak"))
            inputs.append(st.selectbox("Yoğun Bakım", [1, 2, 97], format_func=lambda x: evet_hayir[x]))
            inputs.append(st.selectbox("Entübe Durumu", [1, 2, 97], format_func=lambda x: evet_hayir[x]))
            inputs.append(st.selectbox("Temas Durumu", [1, 2, 99], format_func=lambda x: evet_hayir[x]))
        with c5:
            inputs.append(st.selectbox("Bağışıklık Baskılanması", [1, 2, 98], format_func=lambda x: evet_hayir[x]))
            inputs.append(st.selectbox("Hamilelik", [1, 2, 97, 98], format_func=lambda x: evet_hayir[x]))
            inputs.append(st.number_input("Bölge Kodu", 1, 99, 1))
            inputs.append(st.number_input("Kurum Tipi", 1, 99, 1))

        # Eğer ateş modelin beklediği 20 parametre içinde değilse bile 
        # modelin hata vermemesi için girişi 20'ye sabitliyoruz.
        while len(inputs) < 20:
            inputs.append(0.0)
        
        # Eğer inputs 20'den fazlaysa buduyoruz
        inputs = inputs[:20]

        submit = st.form_submit_button("HESAPLA", use_container_width=True)

        if submit:
            input_tensor = torch.tensor([inputs], dtype=torch.float32)
            with torch.no_grad():
                prediction = model(input_tensor).item()
            
            st.divider()
            st.write(f"### Tahmin Skoru: %{prediction*100:.2f}")
            if prediction > 0.5:
                st.error("RİSKLİ: Model, COVID-19 olasılığını yüksek buldu.")
            else:
                st.success("DÜŞÜK RİSK: Model, COVID-19 olasılığını düşük buldu.")

except Exception as e:
    st.error(f"Hata oluştu: {e}")
