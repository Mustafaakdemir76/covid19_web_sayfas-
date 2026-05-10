import streamlit as st
import torch
import torch.nn as nn
import numpy as np

# 1. Model Mimarisi (Eğitimdeki yapıyla birebir aynı olmalı)
class CovidModel(nn.Module):
    def __init__(self, input_size):
        super(CovidModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

# 2. Sayfa Ayarları
st.set_page_config(page_title="COVID-19 Tahmin Sistemi", page_icon="🦠")

st.title("🦠 COVID-19 Olasılık Analiz Sistemi")
st.markdown("Lütfen aşağıdaki bilgileri girerek analiz butonuna basınız.")

# 3. Modeli Yükleme
@st.cache_resource
def load_my_model():
    # GitHub'a yüklediğin model dosyasının adı 'covid_model.pth' olmalı
    model_path = "covid_model.pth" 
    input_size = 5 # Yaş, Ateş, Öksürük, Oksijen, Yorgunluk
    model = CovidModel(input_size)
    
    # Modeli CPU modunda yükle (Streamlit sunucuları için şart)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

try:
    model = load_my_model()

    # 4. Kullanıcı Giriş Alanları
    col1, col2 = st.columns(2)
    
    with col1:
        v1 = st.number_input("Yaş", min_value=0, max_value=120, value=30)
        v2 = st.number_input("Ateş (Derece)", min_value=30.0, max_value=45.0, value=36.5, step=0.1)
        v3 = st.selectbox("Öksürük Var mı?", options=[0, 1], format_func=lambda x: "Evet" if x==1 else "Hayır")
        
    with col2:
        v4 = st.number_input("Oksijen Değeri (SpO2)", min_value=50, max_value=100, value=95)
        v5 = st.selectbox("Yorgunluk Var mı?", options=[0, 1], format_func=lambda x: "Evet" if x==1 else "Hayır")

    # 5. Tahmin Butonu
    if st.button("Analiz Et", type="primary"):
        input_data = torch.tensor([[float(v1), float(v2), float(v3), float(v4), float(v5)]], dtype=torch.float32)
        
        with torch.no_grad():
            prediction = model(input_data).item()
        
        prob = prediction * 100
        
        st.divider()
        st.subheader(f"Sonuç: %{prob:.2f} Pozitiflik Olasılığı")
        
        if prediction > 0.5:
            st.error("⚠️ Yüksek Risk Grubu: Lütfen bir sağlık kuruluşuna danışınız.")
        else:
            st.success("✅ Düşük Risk Grubu: Belirtileri takip etmeye devam ediniz.")

except Exception as e:
    st.error(f"Model yüklenirken bir hata oluştu: {e}")
    st.info("Lütfen 'covid_model.pth' dosyasının GitHub deponuzda olduğundan emin olun.")