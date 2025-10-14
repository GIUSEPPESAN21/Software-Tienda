# -*- coding: utf-8 -*-
"""
HI-DRIVE: Sistema Avanzado de Gestión de Inventario con IA
Versión 2.0 - Arquitectura Robusta y Analítica Avanzada
"""
import streamlit as st
from PIL import Image
import pandas as pd
import plotly.express as px
import json
import base64
from datetime import datetime
import numpy as np
from pyzbar.pyzbar import decode

# --- Importaciones de utilidades y modelos ---
from firebase_utils import FirebaseManager
from gemini_utils import GeminiUtils
from ultralytics import YOLO
from statsmodels.tsa.holtwinters import ExponentialSmoothing

try:
    from twilio.rest import Client
    IS_TWILIO_AVAILABLE = True
except ImportError:
    IS_TWILIO_AVAILABLE = False
    Client = None

# --- CONFIGURACIÓN DE PÁGINA Y ESTILOS ---
st.set_page_config(
    page_title="HI-DRIVE | Gestión Avanzada de Inventario",
    page_icon="🧠",
    layout="wide"
)

# --- INYECCIÓN DE CSS PARA UNA INTERFAZ PROFESIONAL ---
@st.cache_data
def load_css():
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Intenta cargar el CSS localmente, si no, usa un estilo por defecto.
try:
    load_css()
except FileNotFoundError:
    st.warning("Archivo style.css no encontrado. Se usarán estilos por defecto.")
    st.markdown("""
    <style>
        .main-header { font-size: 2.5rem; font-weight: bold; text-align: center; }
        .stMetric { background-color: #f0f2f6; border-radius: 0.5rem; padding: 1rem; }
    </style>
    """, unsafe_allow_html=True)


# --- INICIALIZACIÓN DE SERVICIOS (CACHED) ---
@st.cache_resource
def initialize_services():
    try:
        yolo_model = YOLO('yolov8m.pt')
        firebase_handler = FirebaseManager()
        gemini_handler = GeminiUtils()
        
        if IS_TWILIO_AVAILABLE and all(k in st.secrets for k in ["TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN"]):
            twilio_client = Client(st.secrets["TWILIO_ACCOUNT_SID"], st.secrets["TWILIO_AUTH_TOKEN"])
        else:
            twilio_client = None
            
        return yolo_model, firebase_handler, gemini_handler, twilio_client
    except Exception as e:
        st.error(f"**Error Crítico de Inicialización:** {e}")
        return None, None, None, None

yolo, firebase, gemini, twilio_client = initialize_services()

if not all([yolo, firebase, gemini]):
    st.stop()

# --- Funciones de Estado de Sesión ---
def init_session_state():
    if 'page' not in st.session_state:
        st.session_state.page = "🏠 Inicio"
    if 'order_ingredients' not in st.session_state:
        st.session_state.order_ingredients = [{'name': '', 'quantity': 1}]

init_session_state()

# --- LÓGICA DE NOTIFICACIONES ---
def send_whatsapp_alert(message):
    if not twilio_client:
        st.toast("Twilio no configurado. Alerta no enviada.", icon="⚠️")
        return
    try:
        from_number = st.secrets["TWILIO_WHATSAPP_FROM_NUMBER"]
        to_number = st.secrets["DESTINATION_WHATSAPP_NUMBER"]
        twilio_client.messages.create(from_=f'whatsapp:{from_number}', body=message, to=f'whatsapp:{to_number}')
        st.toast("¡Alerta de WhatsApp enviada!", icon="📲")
    except Exception as e:
        st.error(f"Error de Twilio: {e}", icon="🚨")

# --- NAVEGACIÓN PRINCIPAL (SIDEBAR) ---
st.sidebar.title("HI-DRIVE 2.0")
PAGES = {
    "🏠 Inicio": "house",
    "📸 Análisis IA": "camera-reels",
    "📦 Inventario": "box-seam",
    "👥 Proveedores": "people",
    "🛒 Pedidos": "cart4",
    "📊 Analítica": "graph-up-arrow",
    " Misión Crítica": "shield-check"
}
for page_name, icon in PAGES.items():
    if st.sidebar.button(f"{page_name}", use_container_width=True, type="primary" if st.session_state.page == page_name else "secondary"):
        st.session_state.page = page_name
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.info("Desarrollado por Joseph Sánchez Acuña.")

# --- RENDERIZADO DE PÁGINAS ---
page_title = st.session_state.page
st.markdown(f'<h1 class="main-header">{page_title}</h1>', unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# ----------------------------------
# PÁGINA: INICIO
# ----------------------------------
if st.session_state.page == "🏠 Inicio":
    st.subheader("Una plataforma robusta que integra IA para reconocimiento, gestión y análisis predictivo de inventarios.")
    
    try:
        items = firebase.get_all_inventory_items()
        orders = firebase.get_orders(status=None)
        suppliers = firebase.get_all_suppliers()
        
        total_inventory_value = sum(item.get('quantity', 0) * item.get('purchase_price', 0) for item in items)
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("📦 Artículos en Inventario", len(items))
        c2.metric("⏳ Pedidos en Proceso", len([o for o in orders if o.get('status') == 'processing']))
        c3.metric("👥 Proveedores Registrados", len(suppliers))
        c4.metric("💰 Valor del Inventario", f"${total_inventory_value:,.2f}")

    except Exception as e:
        st.warning(f"No se pudieron cargar las estadísticas: {e}")
    
    st.markdown("---")
    st.subheader("Módulos del Sistema:")
    st.markdown("""
    - **Análisis IA**: Utiliza IA para identificar, contar y categorizar productos desde una imagen, o escanear códigos de barras/QR para una búsqueda instantánea.
    - **Gestión de Inventario**: Control total sobre artículos, incluyendo umbrales de stock mínimo, costos de compra y un historial completo de movimientos.
    - **Gestión de Proveedores**: Administra tu red de proveedores para optimizar el ciclo de compras.
    - **Gestión de Pedidos**: Crea y procesa pedidos con un sistema que descuenta el stock de forma atómica y segura.
    - **Analítica Avanzada**: Visualiza el rendimiento financiero, la rotación de productos y accede a predicciones de demanda para tomar decisiones proactivas.
    """)

# Resto de las páginas (Análisis IA, Inventario, etc.) se implementarán de forma similar.
# Este es un esqueleto para mostrar la nueva estructura de navegación y la página de inicio.
# En un entorno real, cada bloque 'if st.session_state.page == ...' contendría la lógica completa de esa página.

# ----------------------------------
# PÁGINA: ANÁLISIS IA
# ----------------------------------
elif st.session_state.page == "📸 Análisis IA":
    st.info("Utiliza la cámara para identificar productos con IA, o para escanear códigos de barras y QR.")
    
    source_options = ["🧠 Detección de Objetos", "║█║ Escáner de Código"]
    img_source = st.selectbox("Selecciona el modo de análisis:", source_options)

    img_buffer = st.camera_input("Apunta la cámara al objetivo", key="ia_camera")

    if img_buffer:
        pil_image = Image.open(img_buffer)

        if img_source == "🧠 Detección de Objetos":
            with st.spinner("Detectando objetos con YOLO..."):
                results = yolo(pil_image)
            st.image(results[0].plot(), caption="Objetos detectados.", use_column_width=True)
            
            detections = results[0].boxes
            if detections:
                class_name = results[0].names[detections.cls[0].item()]
                st.write(f"Principal objeto detectado: **{class_name}**")
                if st.button("Analizar con IA de Gemini", type="primary"):
                    with st.spinner("🤖 Gemini está analizando..."):
                        analysis_json_str = gemini.analyze_image(pil_image, class_name)
                        try:
                            analysis_data = json.loads(analysis_json_str)
                            st.session_state.last_analysis = analysis_data
                            st.subheader("Resultados del Análisis:")
                            st.json(analysis_data)
                        except json.JSONDecodeError:
                            st.error("Error al decodificar la respuesta de Gemini.")
                            st.code(analysis_json_str)

        elif img_source == "║█║ Escáner de Código":
            with st.spinner("Buscando códigos..."):
                decoded_objects = decode(pil_image)
                if not decoded_objects:
                    st.warning("No se encontraron códigos de barras o QR en la imagen.")
                for obj in decoded_objects:
                    code_data = obj.data.decode('utf-8')
                    st.success(f"Código detectado: **{code_data}** (Tipo: {obj.type})")
                    item = firebase.get_inventory_item_details(code_data)
                    if item:
                        st.subheader("¡Artículo encontrado en el inventario!")
                        st.json(item)
                    else:
                        st.info("Este código no corresponde a ningún artículo en el inventario.")

# ----------------------------------
# PÁGINA: INVENTARIO
# ----------------------------------
elif st.session_state.page == "📦 Inventario":
    tab1, tab2 = st.tabs(["📋 Lista de Inventario", "➕ Añadir Nuevo Artículo"])

    with tab1:
        st.subheader("Inventario Actual")
        items = firebase.get_all_inventory_items()
        if not items:
            st.info("El inventario está vacío.")
        else:
            for item in items:
                with st.expander(f"**{item.get('name')}** (ID: {item.get('id')}) - Stock: **{item.get('quantity', 0)}**"):
                    c1, c2 = st.columns(2)
                    c1.metric("Costo de Compra", f"${item.get('purchase_price', 0):.2f}")
                    c2.metric("Alerta de Stock Mínimo", item.get('min_stock_alert', 'N/A'))
                    
                    if st.button("Ver Historial de Movimientos", key=f"hist_{item['id']}"):
                        history = firebase.get_inventory_item_history(item['id'])
                        if history:
                            df_hist = pd.DataFrame(history)
                            df_hist['timestamp'] = pd.to_datetime(df_hist['timestamp'])
                            st.dataframe(df_hist[['timestamp', 'type', 'quantity_change', 'details']], use_container_width=True)
                        else:
                            st.write("Sin movimientos registrados.")
    with tab2:
        st.subheader("Añadir o Actualizar Artículo")
        suppliers = firebase.get_all_suppliers()
        supplier_map = {s['name']: s['id'] for s in suppliers}
        
        with st.form("add_item_form"):
            custom_id = st.text_input("ID Personalizado (SKU)")
            name = st.text_input("Nombre del Artículo")
            quantity = st.number_input("Cantidad Actual", min_value=0, step=1)
            purchase_price = st.number_input("Costo de Compra ($)", min_value=0.0)
            min_stock_alert = st.number_input("Umbral de Alerta de Stock Mínimo", min_value=0, step=1)
            
            selected_supplier_name = st.selectbox("Proveedor", list(supplier_map.keys()))

            if st.form_submit_button("Guardar Artículo", type="primary"):
                if custom_id and name:
                    data = {
                        "name": name,
                        "quantity": quantity,
                        "purchase_price": purchase_price,
                        "min_stock_alert": min_stock_alert,
                        "supplier_id": supplier_map.get(selected_supplier_name),
                        "supplier_name": selected_supplier_name,
                        "updated_at": datetime.now().isoformat()
                    }
                    is_new = firebase.get_inventory_item_details(custom_id) is None
                    firebase.save_inventory_item(data, custom_id, is_new)
                    st.success(f"Artículo '{name}' guardado.")
                else:
                    st.warning("ID y Nombre son obligatorios.")

# ----------------------------------
# PÁGINA: PROVEEDORES
# ----------------------------------
elif st.session_state.page == "👥 Proveedores":
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Añadir Proveedor")
        with st.form("add_supplier_form", clear_on_submit=True):
            name = st.text_input("Nombre del Proveedor")
            contact_person = st.text_input("Persona de Contacto")
            email = st.text_input("Email")
            phone = st.text_input("Teléfono")
            if st.form_submit_button("Guardar Proveedor", type="primary"):
                if name:
                    firebase.add_supplier({"name": name, "contact_person": contact_person, "email": email, "phone": phone})
                    st.success(f"Proveedor '{name}' añadido.")
                else:
                    st.warning("El nombre es obligatorio.")
    with col2:
        st.subheader("Lista de Proveedores")
        suppliers = firebase.get_all_suppliers()
        if not suppliers:
            st.info("No hay proveedores registrados.")
        else:
            for s in suppliers:
                with st.container(border=True):
                    st.write(f"**{s['name']}**")
                    st.caption(f"Contacto: {s.get('contact_person', 'N/A')} | Email: {s.get('email', 'N/A')}")
# ----------------------------------
# PÁGINA: PEDIDOS
# ----------------------------------
elif st.session_state.page == "🛒 Pedidos":
    # Lógica de la página de Pedidos similar a la versión anterior,
    # pero utilizando los nuevos componentes y funciones de firebase_utils
    st.info("Funcionalidad de pedidos en desarrollo.")


# ----------------------------------
# PÁGINA: ANALÍTICA
# ----------------------------------
elif st.session_state.page == "📊 Analítica":
    st.info("Módulo de analítica en desarrollo.")
    # Aquí irían los tabs para Rendimiento Financiero, Rotación y Predicción
    
# ----------------------------------
# PÁGINA: MISIÓN CRÍTICA
# ----------------------------------
elif st.session_state.page == " Misión Crítica":
    st.subheader("Acerca del Proyecto")
    st.success("Esta aplicación fue concebida, diseñada y desarrollada como un proyecto de investigación doctoral, con el objetivo de explorar la sinergia entre la inteligencia artificial, la gestión de la cadena de suministro y las interfaces de usuario interactivas para la toma de decisiones en tiempo real en entornos empresariales.")
    
    st.subheader("Arquitecto de la Solución")
    col_img, col_info = st.columns([1, 3])
    with col_img:
        st.image("https://avatars.githubusercontent.com/u/129755299?v=4", width=200)
    with col_info:
        st.title("Joseph Javier Sánchez Acuña")
        st.write("**Investigador en IA | Ingeniero Industrial | Desarrollador de Software**")
        st.markdown(
            """
            - **LinkedIn:** [joseph-javier-sánchez-acuña](https://www.linkedin.com/in/joseph-javier-sánchez-acuña-150410275)
            - **GitHub:** [GIUSEPPESAN21](https://github.com/GIUSEPPESAN21)
            """
        )
