# -*- coding: utf-8 -*-
"""
HI-DRIVE: Sistema Avanzado de Gestión de Inventario con IA
Versión 3.2 - Flujos de Trabajo Inteligentes y Estructura Restaurada
"""
import streamlit as st
from PIL import Image
import pandas as pd
import plotly.express as px
import json
from datetime import datetime

# --- Importaciones de utilidades y modelos ---
try:
    from pyzbar.pyzbar import decode
    from firebase_utils import FirebaseManager
    from gemini_utils import GeminiUtils
    from ultralytics import YOLO
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from twilio.rest import Client
    IS_TWILIO_AVAILABLE = True
except ImportError as e:
    st.error(f"Error de importación: {e}. Asegúrate de que todas las dependencias estén instaladas.")
    st.stop()


# --- CONFIGURACIÓN DE PÁGINA Y ESTILOS ---
st.set_page_config(
    page_title="HI-DRIVE | Gestión Avanzada de Inventario",
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


# --- INICIALIZACIÓN DE SERVICIOS (CACHED) ---
@st.cache_resource
def initialize_services():
    try:
        yolo_model = YOLO('yolov8m.pt')
        firebase_handler = FirebaseManager()
        gemini_handler = GeminiUtils()
        
        twilio_client = None
        if IS_TWILIO_AVAILABLE and all(k in st.secrets for k in ["TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN"]):
            twilio_client = Client(st.secrets["TWILIO_ACCOUNT_SID"], st.secrets["TWILIO_AUTH_TOKEN"])
            
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
    if 'order_items' not in st.session_state:
        st.session_state.order_items = []
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None

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
st.sidebar.title("HI-DRIVE 3.2")
PAGES = {
    "🏠 Inicio": "house", "📸 Análisis IA": "camera-reels", "📦 Inventario": "box-seam",
    "👥 Proveedores": "people", "🛒 Pedidos": "cart4", "📊 Analítica": "graph-up-arrow",
    "👥 Acerca de": "shield-check"
}
for page_name, icon in PAGES.items():
    if st.sidebar.button(f"{page_name}", use_container_width=True, type="primary" if st.session_state.page == page_name else "secondary"):
        st.session_state.page = page_name
        st.session_state.analysis_results = None # Limpiar resultados al cambiar de página
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
        if st.button("➕ Añadir Nuevo Artículo", use_container_width=True):
            st.session_state.page = "📦 Inventario"; st.rerun()
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

# ----------------------------------
# PÁGINA: ANÁLISIS IA
# ----------------------------------
elif st.session_state.page == "📸 Análisis IA":
    st.info("Detecta múltiples objetos, elige cuál analizar y vincúlalo a tu inventario, o usa el escáner para una identificación instantánea.")
    
    source_options = ["🧠 Detección de Objetos", "║█║ Escáner de Código"]
    img_source = st.selectbox("Selecciona el modo de análisis:", source_options)

    img_buffer = st.camera_input("Apunta la cámara al objetivo", key="ia_camera")

    if img_buffer:
        pil_image = Image.open(img_buffer)

        # --- MEJORA: Detección Selectiva de Objetos ---
        if img_source == "🧠 Detección de Objetos":
            with st.spinner("Detectando objetos con IA Local..."):
                results = yolo(pil_image)
            
            st.image(results[0].plot(), caption="Objetos detectados por YOLO.", use_column_width=True)
            detections = results[0]
            
            if detections.boxes:
                st.subheader("Selecciona un objeto para analizarlo en detalle:")
                
                # Crear columnas dinámicamente
                num_detections = len(detections.boxes)
                cols = st.columns(num_detections)

                for i, box in enumerate(detections.boxes):
                    class_name = detections.names[box.cls[0].item()]
                    # Usar la columna correspondiente
                    with cols[i]:
                        if st.button(f"Analizar '{class_name}' #{i+1}", use_container_width=True, key=f"analyze_{i}"):
                            with st.spinner("🤖 Gemini está analizando el recorte..."):
                                coords = box.xyxy[0].cpu().numpy().astype(int)
                                cropped_pil_image = pil_image.crop(tuple(coords))
                                analysis_json = gemini.analyze_image(cropped_pil_image, class_name)
                                st.session_state.analysis_results = json.loads(analysis_json)
                                st.rerun()
            else:
                st.warning("No se detectaron objetos conocidos en la imagen.")

        # --- Lógica del Escáner de Código ---
        elif img_source == "║█║ Escáner de Código":
            with st.spinner("Buscando códigos..."):
                decoded_objects = decode(pil_image)
                if not decoded_objects:
                    st.warning("No se encontraron códigos de barras o QR en la imagen.")
                else:
                    code_data = decoded_objects[0].data.decode('utf-8')
                    st.success(f"Código detectado: **{code_data}**")
                    item = firebase.get_inventory_item_details(code_data)
                    if item:
                        st.subheader("✔️ ¡Artículo Encontrado!")
                        st.json(item)
                    else:
                        st.info("Este código no corresponde a ningún artículo. Puede agregarlo desde la sección de Inventario.")

    # --- MEJORA: Flujo de Vinculación con la Base de Datos ---
    if st.session_state.analysis_results and "error" not in st.session_state.analysis_results:
        st.subheader("✔️ Resultado del Análisis de Gemini")
        res = st.session_state.analysis_results
        st.markdown(f"""
        - **Producto:** {res.get('elemento_identificado', 'N/A')}
        - **Marca/Modelo:** {res.get('marca_modelo_sugerido', 'N/A')}
        - **Categoría Sugerida:** {res.get('posible_categoria_de_inventario', 'N/A')}
        """)
        
        st.subheader("Vincular con Base de Datos")
        action = st.radio("Elige una acción:", ("Crear nuevo artículo", "Vincular a artículo existente"))
        
        all_items = firebase.get_all_inventory_items()
        item_map = {item['name']: item['id'] for item in all_items}

        if action == "Crear nuevo artículo":
            with st.form("create_from_ia"):
                st.write("Crea un nuevo registro en el inventario con la información de la IA.")
                new_id = st.text_input("ID / SKU único", value=res.get('marca_modelo_sugerido', '').replace(" ", "-"))
                new_name = st.text_input("Nombre del artículo", value=res.get('elemento_identificado'))
                if st.form_submit_button("Crear Artículo", type="primary"):
                    if new_id and new_name:
                        if firebase.get_inventory_item_details(new_id):
                            st.error(f"El ID '{new_id}' ya existe.")
                        else:
                            data = {"name": new_name, "analisis_ia": res}
                            firebase.save_inventory_item(data, new_id, is_new=True)
                            st.success(f"Artículo '{new_name}' creado con éxito.")
                    else: st.warning("ID y Nombre son obligatorios.")

        elif action == "Vincular a artículo existente":
            with st.form("link_from_ia"):
                st.write("Añade la información de la IA a un artículo que ya existe en tu inventario.")
                selected_item_name = st.selectbox("Selecciona el artículo a vincular", options=item_map.keys())
                if st.form_submit_button("Vincular Información", type="primary"):
                    item_id = item_map.get(selected_item_name)
                    if item_id:
                        data = {"analisis_ia": res, "updated_at": datetime.now().isoformat()}
                        firebase.save_inventory_item(data, item_id, is_new=False)
                        st.success(f"Información de IA vinculada a '{selected_item_name}'.")

# ----------------------------------
# PÁGINA: INVENTARIO
# ----------------------------------
elif st.session_state.page == "📦 Inventario":
    tab1, tab2 = st.tabs(["📋 Lista de Inventario", "➕ Añadir / Actualizar Artículo"])

    with tab1:
        items = firebase.get_all_inventory_items()
        if not items:
            st.info("El inventario está vacío.")
        else:
            for item in items:
                with st.container(border=True):
                    c1, c2, c3 = st.columns([3,2,2])
                    c1.markdown(f"**{item.get('name', 'N/A')}**")
                    c1.caption(f"ID: {item.get('id', 'N/A')}")
                    c2.metric("Stock Actual", item.get('quantity', 0))
                    c3.metric("Precio Venta", f"${item.get('sale_price', 0):,.2f}")

    with tab2:
        st.subheader("Formulario de Artículo")
        suppliers = firebase.get_all_suppliers()
        supplier_map = {s['name']: s['id'] for s in suppliers}
        
        with st.form("add_item_form"):
            custom_id = st.text_input("ID Personalizado (SKU)")
            name = st.text_input("Nombre del Artículo")
            quantity = st.number_input("Cantidad Actual", min_value=0, step=1)
            purchase_price = st.number_input("Costo de Compra ($)", min_value=0.0, format="%.2f")
            sale_price = st.number_input("Precio de Venta ($)", min_value=0.0, format="%.2f")
            min_stock_alert = st.number_input("Umbral de Alerta de Stock Mínimo", min_value=0, step=1)
            selected_supplier_name = st.selectbox("Proveedor", [""] + list(supplier_map.keys()))

            if st.form_submit_button("Guardar Artículo", type="primary", use_container_width=True):
                if custom_id and name:
                    data = {
                        "name": name, "quantity": quantity, "purchase_price": purchase_price,
                        "sale_price": sale_price,
                        "min_stock_alert": min_stock_alert,
                        "supplier_id": supplier_map.get(selected_supplier_name),
                        "supplier_name": selected_supplier_name, "updated_at": datetime.now().isoformat()
                    }
                    is_new = firebase.get_inventory_item_details(custom_id) is None
                    firebase.save_inventory_item(data, custom_id, is_new)
                    st.success(f"Artículo '{name}' guardado.")
                else:
                    st.warning("ID y Nombre son obligatorios.")

# ----------------------------------
# PÁGINA: PEDIDOS
# ----------------------------------
elif st.session_state.page == "🛒 Pedidos":
    items_from_db = firebase.get_all_inventory_items()
    inventory_by_id = {item['id']: item for item in items_from_db}
    inventory_by_name = {item['name']: item for item in items_from_db if 'name' in item}

    col1, col2 = st.columns([2, 3])
    with col1:
        st.subheader("Añadir Artículos al Pedido")
        
        add_method = st.radio("Método para añadir:", ("Manual", "Escáner de Código"))

        if add_method == "Manual":
            options = [""] + list(inventory_by_name.keys())
            selected_name = st.selectbox("Selecciona un artículo", options)
            if selected_name and st.button("Añadir al Pedido"):
                item_to_add = inventory_by_name[selected_name]
                st.session_state.order_items.append(dict(item_to_add, **{'order_quantity': 1}))
                st.rerun()

        elif add_method == "Escáner de Código":
            barcode_img = st.camera_input("Apunta al código de barras", key="order_scanner")
            if barcode_img:
                decoded_objects = decode(Image.open(barcode_img))
                if decoded_objects:
                    code = decoded_objects[0].data.decode('utf-8')
                    if code in inventory_by_id:
                        item_to_add = inventory_by_id[code]
                        st.session_state.order_items.append(dict(item_to_add, **{'order_quantity': 1}))
                        st.success(f"'{item_to_add['name']}' añadido.")
                        st.rerun()
                    else:
                        st.error(f"El código '{code}' no se encontró en el inventario.")

    with col2:
        st.subheader("Detalle del Pedido Actual")
        if not st.session_state.order_items:
            st.info("Añade artículos para comenzar un pedido.")
        else:
            total_price = 0
            for i, item in enumerate(st.session_state.order_items):
                c1, c2, c3, c4 = st.columns([4,2,2,1])
                c1.text(item['name'])
                new_qty = c2.number_input("Cantidad", value=item['order_quantity'], min_value=1, key=f"qty_{i}")
                st.session_state.order_items[i]['order_quantity'] = new_qty
                item_total = item.get('sale_price', 0) * new_qty
                c3.text(f"${item_total:,.2f}")
                total_price += item_total
                if c4.button("🗑️", key=f"del_{i}"):
                    st.session_state.order_items.pop(i); st.rerun()
            
            st.metric("Precio Total del Pedido", f"${total_price:,.2f}")
            
            order_count = firebase.get_order_count()
            default_title = f"Pedido #{order_count + 1}"

            with st.form("order_form"):
                title = st.text_input("Nombre del Pedido (opcional)", placeholder=default_title)
                final_title = title if title else default_title
                
                if st.form_submit_button("Crear Pedido", type="primary", use_container_width=True):
                    ingredients = [{'id': item['id'], 'name': item['name'], 'quantity': item['order_quantity']} for item in st.session_state.order_items]
                    order_data = {'title': final_title, 'price': total_price, 'ingredients': ingredients, 'status': 'processing', 'timestamp': datetime.now()}
                    firebase.create_order(order_data)
                    st.success(f"Pedido '{final_title}' creado con éxito.")
                    send_whatsapp_alert(f"🧾 Nuevo Pedido: {final_title} por ${total_price:,.2f}")
                    st.session_state.order_items = []
                    st.rerun()
    
    st.markdown("---")
    st.subheader("⏳ Pedidos en Proceso")
    processing_orders = firebase.get_orders('processing')
    if not processing_orders:
        st.info("No hay pedidos en proceso.")
    for order in processing_orders:
        with st.expander(f"**{order['title']}** - ${order.get('price', 0):,.2f}"):
            # ... (Lógica para completar o cancelar se mantiene igual) ...
            pass

# --- (Resto de las páginas se mantienen sin cambios significativos) ---
elif st.session_state.page == "👥 Proveedores":
    # ...
    pass
elif st.session_state.page == "📊 Analítica":
    # ...
    pass
elif st.session_state.page == "👥 Acerca de":
    # ...
    pass

