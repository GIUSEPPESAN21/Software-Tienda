# -*- coding: utf-8 -*-
"""
HI-DRIVE: Sistema Avanzado de Gestión de Inventario con IA
Versión 3.17 - Pedidos con Flujo de Escáner Inteligente
"""
import streamlit as st
from PIL import Image
import pandas as pd
import plotly.express as px
import json
from datetime import datetime, timedelta, timezone
import numpy as np
import cv2

# --- Importaciones de utilidades y modelos ---
try:
    from pyzbar.pyzbar import decode
    from skimage.filters import threshold_local
    from firebase_utils import FirebaseManager
    from gemini_utils import GeminiUtils
    # --- NUEVA IMPORTACIÓN ---
    from barcode_manager import BarcodeManager 
    from ultralytics import YOLO
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from twilio.rest import Client
    IS_TWILIO_AVAILABLE = True
except ImportError as e:
    st.error(f"Error de importación: {e}. Asegúrate de que todas las dependencias estén instaladas.")
    st.stop()


# --- CONFIGURACIÓN DE PÁGINA Y ESTILOS ---
st.set_page_config(
    page_title="SAVA | Gestión Avanzada de Inventario",
    page_icon="🧠",
    layout="wide"
)

# --- INYECCIÓN DE CSS ---
@st.cache_data
def load_css():
    try:
        with open("style.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("Archivo style.css no encontrado. Se usarán estilos por defecto.")

load_css()

# --- LECTOR DE CÓDIGOS DE BARRAS POTENCIADO (PARA CÁMARA) ---
def enhanced_barcode_reader(pil_image):
    try:
        image_cv = np.array(pil_image.convert('RGB'))
        gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
        block_size = 41
        local_thresh = threshold_local(gray, block_size, offset=10)
        binary_adaptive = gray > local_thresh
        processed_image = Image.fromarray((binary_adaptive * 255).astype(np.uint8))
        decoded_objects = decode(processed_image)
        if not decoded_objects:
            decoded_objects = decode(pil_image)
        return decoded_objects
    except Exception as e:
        st.error(f"Error procesando imagen para escáner: {e}")
        return []


# --- INICIALIZACIÓN DE SERVICIOS (CACHED) ---
@st.cache_resource
def initialize_services():
    try:
        yolo_model = YOLO('yolov8m.pt')
        firebase_handler = FirebaseManager()
        gemini_handler = GeminiUtils()
        # --- NUEVA INICIALIZACIÓN ---
        barcode_handler = BarcodeManager(firebase_handler)
        
        twilio_client = None
        if IS_TWILIO_AVAILABLE and all(k in st.secrets for k in ["TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN"]):
            twilio_client = Client(st.secrets["TWILIO_ACCOUNT_SID"], st.secrets["TWILIO_AUTH_TOKEN"])
            
        # --- DEVOLVER NUEVO SERVICIO ---
        return yolo_model, firebase_handler, gemini_handler, twilio_client, barcode_handler
    except Exception as e:
        st.error(f"**Error Crítico de Inicialización:** {e}")
        # --- ACTUALIZAR RETORNO EN CASO DE ERROR ---
        return None, None, None, None, None

yolo, firebase, gemini, twilio_client, barcode_manager = initialize_services()

if not all([yolo, firebase, gemini, barcode_manager]):
    st.stop()

# --- Funciones de Estado de Sesión ---
def init_session_state():
    defaults = {
        'page': "🏠 Inicio", 'order_items': [], 'analysis_results': None,
        'editing_item_id': None, 'scanned_item_data': None,
        # --- NUEVOS ESTADOS PARA ESCÁNER USB ---
        'usb_scan_result': None, 'usb_sale_items': []
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

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
st.sidebar.title("OSIRIS")
PAGES = {
    "🏠 Inicio": "house", 
    "📸 Análisis IA": "camera-reels", 
    "🛰️ Escáner USB": "upc-scan", # --- NUEVA PÁGINA ---
    "📦 Inventario": "box-seam",
    "👥 Proveedores": "people", 
    "🛒 Pedidos": "cart4", 
    "📊 Analítica": "graph-up-arrow",
    "👥 Acerca de": "shield-check"
}
for page_name, icon in PAGES.items():
    if st.sidebar.button(f"{page_name}", use_container_width=True, type="primary" if st.session_state.page == page_name else "secondary"):
        st.session_state.page = page_name
        st.session_state.analysis_results = None
        st.session_state.editing_item_id = None
        st.session_state.scanned_item_data = None
        # --- LIMPIAR ESTADOS AL CAMBIAR DE PÁGINA ---
        st.session_state.usb_scan_result = None
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.info("Desarrollado por Joseph Sánchez Acuña.")

# --- RENDERIZADO DE PÁGINAS ---
page_title = st.session_state.page
st.markdown(f'<h1 class="main-header">{page_title}</h1>', unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# --- PÁGINAS ---
if st.session_state.page == "🏠 Inicio":
    st.subheader("Plataforma de gestión inteligente para un control total sobre su inventario.")
    try:
        items = firebase.get_all_inventory_items()
        orders = firebase.get_orders(status=None)
        suppliers = firebase.get_all_suppliers()
        total_inventory_value = sum(item.get('quantity', 0) * item.get('purchase_price', 0) for item in items)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("📦 Artículos Únicos", len(items))
        c2.metric("💰 Valor del Inventario", f"${total_inventory_value:,.2f}")
        c3.metric("⏳ Pedidos en Proceso", len([o for o in orders if o.get('status') == 'processing']))
        c4.metric("👥 Proveedores", len(suppliers))
    except Exception as e:
        st.warning(f"No se pudieron cargar las estadísticas: {e}")
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Acciones Rápidas")
        if st.button("🛰️ Usar Escáner USB", use_container_width=True):
             st.session_state.page = "🛰️ Escáner USB"; st.rerun()
        if st.button("📝 Crear Nuevo Pedido", use_container_width=True):
            st.session_state.page = "🛒 Pedidos"; st.rerun()
    with col2:
        st.subheader("Alertas de Stock Bajo")
        low_stock_items = [item for item in items if item.get('min_stock_alert') and item.get('quantity', 0) <= item.get('min_stock_alert', 0)]
        if not low_stock_items:
            st.success("¡Todo el inventario está por encima del umbral mínimo!")
        else:
            for item in low_stock_items:
                st.warning(f"**{item['name']}**: {item['quantity']} unidades restantes (Umbral: {item['min_stock_alert']})")


# --- [EL CÓDIGO DE LAS PÁGINAS "Análisis IA", "Inventario", "Proveedores", "Pedidos", "Analítica" y "Acerca de"
# --- PERMANECE IGUAL, POR LO QUE SE OMITE AQUÍ PARA BREVEDAD. PEGA EL CÓDIGO ORIGINAL DE ESAS PÁGINAS EN ESTA SECCIÓN]
# --- ... (código sin cambios de las otras páginas)

# --- ###################################################### ---
# --- INICIO DE LA NUEVA PÁGINA: ESCÁNER USB ---
# --- ###################################################### ---
elif st.session_state.page == "🛰️ Escáner USB":
    st.info("Conecta tu lector de códigos de barras USB. Haz clic en el campo de texto y comienza a escanear.")

    mode = st.radio("Selecciona el modo de operación:", 
                    ("Gestión de Inventario", "Punto de Venta (Salida Rápida)"), 
                    horizontal=True, key="usb_scanner_mode")

    st.markdown("---")

    if mode == "Gestión de Inventario":
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Escanear para Gestionar")
            with st.form("usb_inventory_scan_form"):
                barcode_input = st.text_input("Código de Barras", key="usb_barcode_inv", 
                                              help="Haz clic aquí antes de escanear.")
                submitted = st.form_submit_button("Buscar / Registrar", use_container_width=True)
                if submitted and barcode_input:
                    st.session_state.usb_scan_result = barcode_manager.handle_inventory_scan(barcode_input)
                    st.rerun()

        with col2:
            st.subheader("Resultado del Escaneo")
            result = st.session_state.get('usb_scan_result')
            
            if not result:
                st.info("Esperando escaneo...")
            elif result['status'] == 'error':
                st.error(result['message'])
            
            # --- Flujo 1: Producto ENCONTRADO ---
            elif result['status'] == 'found':
                item = result['item']
                st.success(f"✔️ Producto Encontrado: **{item['name']}**")
                
                with st.form("update_item_form"):
                    st.write(f"**Stock Actual:** {item.get('quantity', 0)}")
                    st.write(f"**Precio de Venta:** ${item.get('sale_price', 0):.2f}")
                    
                    new_quantity = st.number_input("Nueva Cantidad Total", min_value=0, value=item.get('quantity', 0), step=1)
                    new_price = st.number_input("Nuevo Precio de Venta ($)", min_value=0.0, value=item.get('sale_price', 0.0), format="%.2f")
                    
                    if st.form_submit_button("Actualizar Producto", type="primary", use_container_width=True):
                        updated_data = item.copy()
                        updated_data.update({'quantity': new_quantity, 'sale_price': new_price, 'updated_at': datetime.now().isoformat()})
                        firebase.save_inventory_item(updated_data, item['id'], is_new=False, details="Actualización vía Escáner USB.")
                        st.success(f"¡'{item['name']}' actualizado con éxito!")
                        st.session_state.usb_scan_result = None # Limpiar para el próximo escaneo
                        st.rerun()

            # --- Flujo 2: Producto NO ENCONTRADO ---
            elif result['status'] == 'not_found':
                barcode = result['barcode']
                st.warning(f"⚠️ El código '{barcode}' no existe. Por favor, regístralo.")

                with st.form("create_from_usb_scan_form"):
                    st.markdown(f"**Código de Barras:** `{barcode}`")
                    name = st.text_input("Nombre del Producto")
                    quantity = st.number_input("Cantidad Inicial", min_value=1, step=1)
                    sale_price = st.number_input("Precio de Venta ($)", min_value=0.0, format="%.2f")
                    purchase_price = st.number_input("Precio de Compra ($)", min_value=0.0, format="%.2f")
                    
                    if st.form_submit_button("Guardar Nuevo Producto", type="primary", use_container_width=True):
                        if name and quantity > 0:
                            data = {"name": name, "quantity": quantity, "sale_price": sale_price, "purchase_price": purchase_price, "updated_at": datetime.now().isoformat()}
                            firebase.save_inventory_item(data, barcode, is_new=True, details="Creado vía Escáner USB.")
                            st.success(f"¡Producto '{name}' guardado!")
                            st.session_state.usb_scan_result = None
                            st.rerun()
                        else:
                            st.warning("El nombre y la cantidad son obligatorios.")

    elif mode == "Punto de Venta (Salida Rápida)":
        col1, col2 = st.columns([2, 3])
        with col1:
            st.subheader("Escanear Productos para Venta")
            with st.form("usb_sale_scan_form"):
                barcode_input = st.text_input("Escanear Código de Producto", key="usb_barcode_sale")
                submitted = st.form_submit_button("Añadir a la Venta", use_container_width=True)
                if submitted and barcode_input:
                    updated_list, status_msg = barcode_manager.add_item_to_sale(barcode_input, st.session_state.usb_sale_items)
                    st.session_state.usb_sale_items = updated_list
                    
                    if status_msg['status'] == 'success': st.toast(status_msg['message'], icon="✅")
                    elif status_msg['status'] == 'warning': st.toast(status_msg['message'], icon="⚠️")
                    else: st.error(status_msg['message'])
                    st.rerun()

        with col2:
            st.subheader("Detalle de la Venta Actual")
            if not st.session_state.usb_sale_items:
                st.info("Escanea un producto para comenzar...")
            else:
                total_sale_price = 0
                df_items = []
                for item in st.session_state.usb_sale_items:
                    total_item_price = item['sale_price'] * item['quantity']
                    total_sale_price += total_item_price
                    df_items.append({
                        "Producto": item['name'],
                        "Cantidad": item['quantity'],
                        "Precio Unit.": f"${item['sale_price']:.2f}",
                        "Subtotal": f"${total_item_price:.2f}"
                    })
                
                st.dataframe(pd.DataFrame(df_items), use_container_width=True, hide_index=True)
                st.markdown(f"### Total Venta: `${total_sale_price:,.2f}`")

                c1, c2 = st.columns(2)
                if c1.button("✅ Finalizar y Descontar Stock", type="primary", use_container_width=True):
                    sale_id = f"VentaDirecta-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
                    success, msg, alerts = firebase.process_direct_sale(st.session_state.usb_sale_items, sale_id)
                    if success:
                        st.success(msg)
                        send_whatsapp_alert(f"💸 Venta Rápida Procesada: {sale_id} por un total de ${total_sale_price:,.2f}")
                        for alert in alerts: send_whatsapp_alert(f"📉 ALERTA DE STOCK: {alert}")
                        st.session_state.usb_sale_items = []
                        st.rerun()
                    else:
                        st.error(msg)

                if c2.button("❌ Cancelar Venta", use_container_width=True):
                    st.session_state.usb_sale_items = []
                    st.toast("Venta cancelada.")
                    st.rerun()

# --- ###################################################### ---
# --- FIN DE LA NUEVA PÁGINA ---
# --- ###################################################### ---

# Resto de las páginas originales
elif st.session_state.page == "📸 Análisis IA":
    # ... (código original de la página Análisis IA)
    pass

elif st.session_state.page == "📦 Inventario":
    # ... (código original de la página Inventario)
    pass

elif st.session_state.page == "👥 Proveedores":
    # ... (código original de la página Proveedores)
    pass

elif st.session_state.page == "🛒 Pedidos":
    # ... (código original de la página Pedidos)
    pass

elif st.session_state.page == "📊 Analítica":
    # ... (código original de la página Analítica)
    pass

elif st.session_state.page == "👥 Acerca de":
    # ... (código original de la página Acerca de)
    pass
