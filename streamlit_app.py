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
st.set_page_config(page_title="COVID-19 Detaylı Analiz", page_icon="🦠", layout="wide")
st.title("🦠 COVID-19 Risk Analiz Paneli")
st.write("Lütfen hastanın durumuna uygun seçenekleri işaretleyiniz.")

# 3. MODELİ YÜKLE
@st.cache_resource
def load_model():
    model_path = "covid_model.pth"
    model = CovidModel()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# 4. YARDIMCI SEÇENEKLER (Sayısal değerlerin anlamları)
evet_hayir = {1: "Evet", 2: "Hayır", 97: "Bilinmiyor", 98: "Bilinmiyor"}
cinsiyet_opt = {1: "Kadın", 2: "Erkek"}
hasta_tipi_opt = {1: "Evde Tedavi", 2: "Hastaneye Yatış"}

try:
    model = load_model()
    
    with st.form("analiz_formu"):
        col1, col2, col3, col4 = st.columns(4)
        
        # Değerleri toplamak için liste
        inputs = []

        with col1:
            inputs.append(st.selectbox("Cinsiyet", options=[1, 2], format_func=lambda x: cinsiyet_opt[x]))
            inputs.append(st.selectbox("Hasta Tipi", options=[1, 2], format_func=lambda x: hasta_tipi_opt[x]))
            inputs.append(st.selectbox("Zatürre (Pnömoni)", options=[1, 2, 97], format_func=lambda x: evet_hayir.get(x, "Bilinmiyor")))
            inputs.append(st.number_input("Yaş", 0, 120, 30))
            inputs.append(st.selectbox("Hamilelik", options=[1, 2, 97, 98], format_func=lambda x: evet_hayir.get(x, "Bilinmiyor")))

        with col2:
            inputs.append(st.selectbox("Diyabet", options=[1, 2, 98], format_func=lambda x: evet_hayir.get(x, "Bilinmiyor")))
            inputs.append(st.selectbox("KOAH", options=[1, 2, 98], format_func=lambda x: evet_hayir.get(x, "Bilinmiyor")))
            inputs.append(st.selectbox("Astım", options=[1, 2, 98], format_func=lambda x: evet_hayir.get(x, "Bilinmiyor")))
            inputs.append(st.selectbox("İmmün Süpresyon", options=[1, 2, 98], format_func=lambda x: evet_hayir.get(x, "Bilinmiyor")))
            inputs.append(st.selectbox("Hipertansiyon", options=[1, 2, 98], format_func=lambda x: evet_hayir.get(x, "Bilinmiyor")))

        with col3:
            inputs.append(st.selectbox("Diğer Hastalıklar", options=[1, 2, 98], format_func=lambda x: evet_hayir.get(x, "Bilinmiyor")))
            inputs.append(st.selectbox("Kardiyovasküler", options=[1, 2, 98], format_func=lambda x: evet_hayir.get(x, "Bilinmiyor")))
            inputs.append(st.selectbox("Obezite", options=[1, 2, 98], format_func=lambda x: evet_hayir.get(x, "Bilinmiyor")))
            inputs.append(st.selectbox("Kronik Böbrek Yetm.", options=[1, 2, 98], format_func=lambda x: evet_hayir.get(x, "Bilinmiyor")))
            inputs.append(st.selectbox("Tütün Kullanımı", options=[1, 2, 98], format_func=lambda x: evet_hayir.get(x, "Bilinmiyor")))

        with col4:
            inputs.append(st.selectbox("Temas Durumu", options=[1, 2, 99], format_func=lambda x: evet_hayir.get(x, "Bilinmiyor")))
            inputs.append(st.selectbox("Yoğun Bakım", options=[1, 2, 97, 99], format_func=lambda x: evet_hayir.get(x, "Bilinmiyor")))
            inputs.append(st.selectbox("Entübe Durumu", options=[1, 2, 97, 99], format_func=lambda x: evet_hayir.get(x, "Bilinmiyor")))
            inputs.append(st.number_input("Bölge Kodu", 1, 99, 1))
            inputs.append(st.number_input("Kurum Kodu", 1, 99, 1))

        submitted = st.form_submit_button("ANALİZ ET", use_container_width=True)

        if submitted:
            input_tensor = torch.tensor([inputs], dtype=torch.float32)
            with torch.no_grad():
                olasilik = model(input_tensor).item()
            
            st.divider()
            st.subheader(f"📊 Analiz Sonucu: %{olasilik*100:.2f}")
            if olasilik > 0.5:
                st.error("🚨 Yüksek Risk: Hastanın durumu kritik olabilir.")
            else:
                st.success("✅ Düşük Risk: Mevcut verilere göre risk seviyesi normal.")

except Exception as e:
    st.error(f"Hata: {e}")
