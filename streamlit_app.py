import streamlit as st
import torch
import torch.nn as nn
import numpy as np

# 1. MODEL MİMARİSİ (Hata almamak için tam uyumlu yapı)
class CovidModel(nn.Module):
    def __init__(self):
        super(CovidModel, self).__init__()
        self.katmanlar = nn.Sequential(
            nn.Linear(20, 64), # 20 Parametre girişi
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
st.set_page_config(page_title="COVID-19 Risk Analizi", page_icon="🦠", layout="wide")

st.title("🦠 COVID-19 Olasılık Analiz Sistemi")
st.markdown("---")
st.sidebar.header("Sistem Hakkında")
st.sidebar.info("Bu sistem, yapay zeka modeli kullanarak girdiğiniz verilere göre COVID-19 risk analizi yapar.")

# 3. MODELİ YÜKLE
@st.cache_resource
def load_model():
    model_path = "covid_model.pth"
    model = CovidModel()
    # CPU üzerinde yükleme yaparak sunucu hatalarını önler
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Beklenen 20 parametrenin isim listesi
ozellik_isimleri = [
    "Cinsiyet (1:K, 2:E)", "Hasta Tipi", "Zatürre (Pnömoni)", "Yaş", "Hamilelik",
    "Diyabet", "KOAH", "Astım", "İmmün Süpresyon", "Hipertansiyon",
    "Diğer Hastalıklar", "Kardiyovasküler", "Obezite", "Kronik Böbrek Yetm.",
    "Tütün Kullanımı", "Temas Durumu", "Yoğun Bakım", "Entübe Durumu",
    "Bölge Kodu", "Kurum Kodu"
]

try:
    model = load_model()
    
    st.subheader("📋 Hasta Bilgilerini Eksiksiz Giriniz")
    
    # 20 Giriş Alanını Düzenli Bir Şekilde Göster
    inputs = []
    cols = st.columns(4) # 4 sütunlu yapı
    
    for i in range(20):
        with cols[i % 4]:
            val = st.number_input(f"{ozellik_isimleri[i]}", value=0.0, step=1.0, key=f"in_{i}")
            inputs.append(val)

    st.markdown("---")
    
    if st.button("🔍 ANALİZ ET VE HESAPLA", type="primary", use_container_width=True):
        # Veriyi modele uygun formata getir
        input_tensor = torch.tensor([inputs], dtype=torch.float32)
        
        with torch.no_grad():
            olasilik = model(input_tensor).item()
        
        # Sonuç Ekranı
        st.subheader("📊 Analiz Sonucu")
        yuzde = olasilik * 100
        
        col_res1, col_res2 = st.columns(2)
        col_res1.metric("Pozitiflik Olasılığı", f"%{yuzde:.2f}")
        
        if olasilik > 0.5:
            col_res2.error("🚨 YÜKSEK RİSK: Lütfen en yakın sağlık kuruluşuna başvurunuz.")
        else:
            col_res2.success("✅ DÜŞÜK RİSK: Şu anki verilere göre risk düşük görünmektedir.")

except Exception as e:
    st.error(f"Sistem Hatası: {e}")
    st.warning("Lütfen 'covid_model.pth' dosyasının GitHub'da olduğundan emin olun.")
