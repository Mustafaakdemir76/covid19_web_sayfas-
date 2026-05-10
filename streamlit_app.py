import streamlit as st
import torch
import torch.nn as nn

# 1. Model Mimarisi (Hata almamak için tam eşleşme sağlandı)
class CovidModel(nn.Module):
    def __init__(self, input_size):
        super(CovidModel, self).__init__()
        # .pth dosyanızdaki isim ve yapı (katmanlar.0, katmanlar.3, katmanlar.5) ile uyumlu
        self.katmanlar = nn.Sequential(
            nn.Linear(input_size, 16), # katmanlar.0
            nn.ReLU(),                 # katmanlar.1
            nn.Identity(),             # katmanlar.2 (Boşluk doldurucu)
            nn.Linear(16, 8),          # katmanlar.3
            nn.ReLU(),                 # katmanlar.4
            nn.Linear(8, 1),           # katmanlar.5
            nn.Sigmoid()               # katmanlar.6
        )

    def forward(self, x):
        return self.katmanlar(x)

# 2. Sayfa Ayarları
st.set_page_config(page_title="Covid-19 Tahmin", page_icon="🦠")
st.title("🦠 COVID-19 Olasılık Analiz Sistemi")

# 3. Modeli Yükle
@st.cache_resource
def load_model():
    model_path = "covid_model.pth"
    input_size = 5 # Yaş, Ateş, Öksürük, Oksijen, Yorgunluk
    model = CovidModel(input_size)
    # Hata veren state_dict yükleme kısmı burada düzelecek
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

try:
    model = load_model()
    
    st.subheader("Hasta Bilgilerini Giriniz")
    yas = st.number_input("Yaş", 0, 120, 30)
    ates = st.number_input("Ateş (Derece)", 34.0, 43.0, 36.5)
    oksuruk = st.selectbox("Öksürük Var mı?", [0, 1], format_func=lambda x: "Evet" if x==1 else "Hayır")
    oksijen = st.number_input("Oksijen Satürasyonu (SpO2)", 50, 100, 95)
    yorgunluk = st.selectbox("Yorgunluk Var mı?", [0, 1], format_func=lambda x: "Evet" if x==1 else "Hayır")

    if st.button("Analiz Et", type="primary"):
        girdi = torch.tensor([[float(yas), float(ates), float(oksuruk), float(oksijen), float(yorgunluk)]], dtype=torch.float32)
        
        with torch.no_grad():
            olasilik = model(girdi).item()
        
        st.divider()
        st.metric("Pozitiflik Olasılığı", f"%{olasilik*100:.2f}")
        
        if olasilik > 0.5:
            st.error("⚠️ Yüksek Risk Grubu")
        else:
            st.success("✅ Düşük Risk Grubu")

except Exception as e:
    st.error(f"Sistemde bir hata oluştu: {e}")
