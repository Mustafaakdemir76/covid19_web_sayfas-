import streamlit as st
import torch
import torch.nn as nn

# 1. Model Mimarisi (Hata mesajındaki boyutlarla BİREBİR AYNI)
class CovidModel(nn.Module):
    def __init__(self):
        super(CovidModel, self).__init__()
        # Checkpoint: [64, 20], [32, 64], [1, 32]
        self.katmanlar = nn.Sequential(
            nn.Linear(20, 64), # Giriş: 20, Çıkış: 64
            nn.ReLU(),
            nn.Identity(),     # Katman indisi uyumu için
            nn.Linear(64, 32), # Giriş: 64, Çıkış: 32
            nn.ReLU(),
            nn.Linear(32, 1),  # Giriş: 32, Çıkış: 1
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.katmanlar(x)

# 2. Sayfa Ayarları
st.set_page_config(page_title="Covid-19 Analiz", page_icon="🦠")
st.title("🦠 COVID-19 Olasılık Analiz Sistemi")

# 3. Modeli Yükle
@st.cache_resource
def load_model():
    model_path = "covid_model.pth"
    model = CovidModel()
    # image_07f15d.png'deki anahtar uyuşmazlığı burada çözüldü
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

try:
    model = load_model()
    
    st.info("Not: Modeliniz 20 farklı veri sütunu beklemektedir. Lütfen değerleri giriniz.")
    
    # 20 Giriş Alanı (Modelin çalışması için bu sayı şarttır)
    inputs = []
    col1, col2 = st.columns(2)
    for i in range(20):
        with col1 if i < 10 else col2:
            val = st.number_input(f"Parametre {i+1}", value=0.0, step=0.1, key=f"in_{i}")
            inputs.append(val)

    if st.button("Analiz Et", type="primary"):
        girdi = torch.tensor([inputs], dtype=torch.float32)
        
        with torch.no_grad():
            olasilik = model(girdi).item()
        
        st.divider()
        st.metric("Pozitiflik Olasılığı", f"%{olasilik*100:.2f}")
        
        if olasilik > 0.5:
            st.error("⚠️ Yüksek Risk")
        else:
            st.success("✅ Düşük Risk")

except Exception as e:
    st.error(f"Hata: {e}")
