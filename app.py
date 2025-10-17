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

# --- LECTOR DE CÓDIGOS DE BARRAS POTENCIADO ---
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
    defaults = {
        'page': "🏠 Inicio", 'order_items': [], 'analysis_results': None,
        'editing_item_id': None, 'scanned_item_data': None
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
    "🏠 Inicio": "house", "📸 Análisis IA": "camera-reels", "📦 Inventario": "box-seam",
    "👥 Proveedores": "people", "🛒 Pedidos": "cart4", "📊 Analítica": "graph-up-arrow",
    "👥 Acerca de": "shield-check"
}
for page_name, icon in PAGES.items():
    if st.sidebar.button(f"{page_name}", use_container_width=True, type="primary" if st.session_state.page == page_name else "secondary"):
        st.session_state.page = page_name
        st.session_state.analysis_results = None
        st.session_state.editing_item_id = None
        st.session_state.scanned_item_data = None
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.info("Desarrollado por Joseph Sánchez Acuña.")

# --- RENDERIZADO DE PÁGINAS ---
page_title = st.session_state.page
st.markdown(f'<h1 class="main-header">{page_title}</h1>', unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# --- PÁGINAS ---
if st.session_state.page == "🏠 Inicio":
    # ... (código sin cambios)
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

elif st.session_state.page == "📸 Análisis IA":
    # ... (código sin cambios)
    st.info("Usa la detección de objetos para un análisis detallado o el escáner de códigos para una gestión rápida de inventario.")
    source_options = ["🧠 Detección de Objetos", "║█║ Escáner de Código"]
    img_source = st.selectbox("Selecciona el modo de análisis:", source_options)
    
    if img_source == "║█║ Escáner de Código":
        st.subheader("Gestión de Inventario por Código")
        img_buffer = st.camera_input("Apunta la cámara al código de barras", key="scanner_ia_page")

        if img_buffer:
            with st.spinner("Buscando códigos..."):
                pil_image = Image.open(img_buffer)
                decoded_objects = enhanced_barcode_reader(pil_image)
                if decoded_objects:
                    code_data = decoded_objects[0].data.decode('utf-8')
                    item = firebase.get_inventory_item_details(code_data)
                    st.session_state.scanned_item_data = {'code': code_data, 'item': item}
                else:
                    st.warning("No se encontraron códigos de barras o QR.")
                    st.session_state.scanned_item_data = None
        
        if st.session_state.scanned_item_data:
            scan_data = st.session_state.scanned_item_data
            item, code = scan_data['item'], scan_data['code']
            st.success(f"Último código escaneado: **{code}**")
            if item:
                st.subheader("✔️ Artículo Encontrado: ¿Qué deseas hacer?")
                st.markdown(f"**Nombre:** {item.get('name', 'N/A')} | **Stock Actual:** {item.get('quantity', 0)}")
                if st.button("✏️ Editar Detalles Completos", help="Ir a la página de inventario para editar todos los campos de este producto."):
                    st.session_state.editing_item_id = item['id']
                    st.session_state.page = "📦 Inventario"
                    st.rerun()
            else:
                st.subheader("➕ Artículo Nuevo: Registrar en Inventario")
                st.info(f"El código **{code}** no está en la base de datos. Por favor, completa los detalles.")
                with st.form("create_from_scan_form"):
                    suppliers = firebase.get_all_suppliers()
                    supplier_map = {s['name']: s['id'] for s in suppliers}
                    name = st.text_input("Nombre del Artículo")
                    quantity = st.number_input("Cantidad Inicial", min_value=1, step=1)
                    sale_price = st.number_input("Precio de Venta ($)", min_value=0.0, format="%.2f")
                    purchase_price = st.number_input("Precio de Compra ($)", min_value=0.0, format="%.2f")
                    min_stock_alert = st.number_input("Umbral de Alerta", min_value=0, step=1)
                    selected_supplier_name = st.selectbox("Proveedor", [""] + list(supplier_map.keys()))
                    if st.form_submit_button("Guardar Nuevo Artículo", type="primary"):
                        if name and quantity > 0:
                            data = {"name": name, "quantity": quantity, "sale_price": sale_price,"purchase_price": purchase_price, "min_stock_alert": min_stock_alert,"supplier_id": supplier_map.get(selected_supplier_name),"supplier_name": selected_supplier_name, "updated_at": datetime.now().isoformat()}
                            firebase.save_inventory_item(data, code, is_new=True)
                            st.success(f"Nuevo artículo '{name}' guardado con éxito.")
                            st.session_state.scanned_item_data = None
                            st.rerun()
                        else:
                            st.warning("El nombre y la cantidad son obligatorios.")
    elif img_source == "🧠 Detección de Objetos":
        img_buffer = st.camera_input("Apunta la cámara al objetivo", key="detector_ia_page")
        if img_buffer:
            pil_image = Image.open(img_buffer)
            with st.spinner("Detectando objetos con IA Local..."):
                results = yolo(pil_image)
            st.image(results[0].plot(), caption="Objetos detectados por YOLO.", use_column_width=True)
            detections = results[0]
            if detections.boxes:
                st.subheader("Selecciona un objeto para analizarlo en detalle:")
                cols = st.columns(min(len(detections.boxes), 4))
                for i, box in enumerate(detections.boxes):
                    class_name = detections.names[box.cls[0].item()]
                    if cols[i % 4].button(f"Analizar '{class_name}' #{i+1}", use_container_width=True, key=f"analyze_{i}"):
                        with st.spinner("🤖 Gemini está analizando el recorte..."):
                            coords = box.xyxy[0].cpu().numpy().astype(int)
                            analysis_json = gemini.analyze_image(pil_image.crop(tuple(coords)), class_name)
                            st.session_state.analysis_results = json.loads(analysis_json)
                            st.session_state.scanned_item_data = None
                            st.rerun()
            else:
                st.warning("No se detectaron objetos conocidos en la imagen.")
    if st.session_state.analysis_results and "error" not in st.session_state.analysis_results:
        res = st.session_state.analysis_results
        st.subheader("✔️ Resultado del Análisis de Gemini")
        st.markdown(f"""- **Producto:** {res.get('elemento_identificado', 'N/A')}\n- **Marca/Modelo:** {res.get('marca_modelo_sugerido', 'N/A')}""")
        st.subheader("Vincular con Base de Datos")
        action = st.radio("Elige una acción:", ("Crear nuevo artículo", "Vincular a artículo existente"))
        all_items = firebase.get_all_inventory_items()
        item_map = {item['name']: item['id'] for item in all_items}
        if action == "Crear nuevo artículo":
            with st.form("create_from_ia"):
                new_id = st.text_input("ID / SKU único", value=res.get('marca_modelo_sugerido', '').replace(" ", "-"))
                new_name = st.text_input("Nombre del artículo", value=res.get('elemento_identificado'))
                if st.form_submit_button("Crear Artículo", type="primary"):
                    if new_id and new_name and not firebase.get_inventory_item_details(new_id):
                        data = {"name": new_name, "analisis_ia": res}
                        firebase.save_inventory_item(data, new_id, is_new=True)
                        st.success(f"Artículo '{new_name}' creado.")
                    else:
                        st.error("ID no válido o ya existente.")
        elif action == "Vincular a artículo existente":
            with st.form("link_from_ia"):
                selected_item_name = st.selectbox("Selecciona el artículo a vincular", options=item_map.keys())
                if st.form_submit_button("Vincular Información", type="primary"):
                    item_id = item_map.get(selected_item_name)
                    if item_id:
                        data = {"analisis_ia": res, "updated_at": datetime.now().isoformat()}
                        firebase.save_inventory_item(data, item_id, is_new=False, details="Vinculación de datos de IA.")
                        st.success(f"Información vinculada a '{selected_item_name}'.")

elif st.session_state.page == "📦 Inventario":
    # ... (código sin cambios)
    if st.session_state.editing_item_id:
        item_to_edit = firebase.get_inventory_item_details(st.session_state.editing_item_id)
        st.subheader(f"✏️ Editando: {item_to_edit.get('name')}")
        with st.form("edit_item_form"):
            suppliers = firebase.get_all_suppliers()
            supplier_map = {s['name']: s['id'] for s in suppliers}
            supplier_names = [""] + list(supplier_map.keys())
            current_supplier = item_to_edit.get('supplier_name')
            current_supplier_index = supplier_names.index(current_supplier) if current_supplier in supplier_names else 0
            name = st.text_input("Nombre del Artículo", value=item_to_edit.get('name'))
            quantity = st.number_input("Cantidad Actual", value=item_to_edit.get('quantity', 0), min_value=0, step=1)
            purchase_price = st.number_input("Costo de Compra ($)", value=item_to_edit.get('purchase_price', 0.0), format="%.2f")
            sale_price = st.number_input("Precio de Venta ($)", value=item_to_edit.get('sale_price', 0.0), format="%.2f")
            min_stock_alert = st.number_input("Umbral de Alerta", value=item_to_edit.get('min_stock_alert', 0), min_value=0, step=1)
            selected_supplier_name = st.selectbox("Proveedor", supplier_names, index=current_supplier_index)
            c1, c2 = st.columns(2)
            if c1.form_submit_button("Guardar Cambios", type="primary", use_container_width=True):
                if name:
                    data = {"name": name, "quantity": quantity, "purchase_price": purchase_price, "sale_price": sale_price,
                            "min_stock_alert": min_stock_alert, "supplier_id": supplier_map.get(selected_supplier_name),
                            "supplier_name": selected_supplier_name, "updated_at": datetime.now().isoformat()}
                    firebase.save_inventory_item(data, st.session_state.editing_item_id, is_new=False, details=f"Edición manual de datos.")
                    st.success(f"Artículo '{name}' actualizado.")
                    st.session_state.editing_item_id = None; st.rerun()
            if c2.form_submit_button("Cancelar", use_container_width=True):
                st.session_state.editing_item_id = None; st.rerun()
    else:
        tab1, tab2 = st.tabs(["📋 Inventario Actual", "➕ Añadir Artículo"])
        with tab1:
            st.info("Haz clic en '✏️' en cualquier artículo para editar sus detalles.")
            items = firebase.get_all_inventory_items()
            if not items:
                st.info("El inventario está vacío.")
            else:
                for item in items:
                    with st.container(border=True):
                        c1, c2, c3, c4 = st.columns([4, 2, 2, 1])
                        c1.markdown(f"**{item.get('name', 'N/A')}**"); c1.caption(f"ID: {item.get('id', 'N/A')}")
                        c2.metric("Stock", item.get('quantity', 0))
                        c3.metric("Precio Venta", f"${item.get('sale_price', 0):,.2f}")
                        if c4.button("✏️", key=f"edit_{item['id']}", help="Editar este artículo"):
                            st.session_state.editing_item_id = item['id']; st.rerun()
        with tab2:
            st.subheader("Añadir Nuevo Artículo al Inventario")
            suppliers = firebase.get_all_suppliers()
            supplier_map = {s['name']: s['id'] for s in suppliers}
            with st.form("add_item_form_new"):
                custom_id = st.text_input("ID Personalizado (SKU)")
                name = st.text_input("Nombre del Artículo")
                quantity = st.number_input("Cantidad Inicial", min_value=0, step=1)
                purchase_price = st.number_input("Costo de Compra ($)", min_value=0.0, format="%.2f")
                sale_price = st.number_input("Precio de Venta ($)", min_value=0.0, format="%.2f")
                min_stock_alert = st.number_input("Umbral de Alerta", min_value=0, step=1)
                selected_supplier_name = st.selectbox("Proveedor", [""] + list(supplier_map.keys()))
                if st.form_submit_button("Guardar Nuevo Artículo", type="primary", use_container_width=True):
                    if custom_id and name and not firebase.get_inventory_item_details(custom_id):
                        data = {"name": name, "quantity": quantity, "purchase_price": purchase_price, "sale_price": sale_price,
                                "min_stock_alert": min_stock_alert, "supplier_id": supplier_map.get(selected_supplier_name),
                                "supplier_name": selected_supplier_name, "updated_at": datetime.now().isoformat()}
                        firebase.save_inventory_item(data, custom_id, is_new=True)
                        st.success(f"Artículo '{name}' guardado.")
                    else:
                        st.error("ID no válido, vacío o ya existente.")

elif st.session_state.page == "👥 Proveedores":
    # ... (código sin cambios)
    col1, col2 = st.columns([1, 2])
    with col1:
        with st.form("add_supplier_form", clear_on_submit=True):
            st.subheader("Añadir Proveedor")
            name = st.text_input("Nombre del Proveedor")
            contact = st.text_input("Persona de Contacto")
            email = st.text_input("Email")
            phone = st.text_input("Teléfono")
            if st.form_submit_button("Guardar", type="primary", use_container_width=True):
                if name:
                    firebase.add_supplier({"name": name, "contact_person": contact, "email": email, "phone": phone})
                    st.success(f"Proveedor '{name}' añadido.")
                    st.rerun()
    with col2:
        st.subheader("Lista de Proveedores")
        suppliers = firebase.get_all_suppliers()
        for s in suppliers:
            with st.expander(f"**{s['name']}**"):
                st.write(f"**Contacto:** {s.get('contact_person', 'N/A')}")
                st.write(f"**Email:** {s.get('email', 'N/A')}")
                st.write(f"**Teléfono:** {s.get('phone', 'N/A')}")

elif st.session_state.page == "🛒 Pedidos":
    items_from_db = firebase.get_all_inventory_items()
    inventory_by_id = {item['id']: item for item in items_from_db}
    inventory_by_name = {item['name']: item for item in items_from_db if 'name' in item}
    
    col1, col2 = st.columns([2, 3])
    with col1:
        st.subheader("Añadir Artículos al Pedido")
        
        add_method = st.radio("Método para añadir:", ("Manual", "Escáner de Código", "Detección de Objetos"))
        
        if add_method == "Manual":
            options = [""] + list(inventory_by_name.keys())
            selected_name = st.selectbox("Selecciona un artículo", options)
            if selected_name and st.button("Añadir 1 unidad al Pedido"):
                item_to_add = inventory_by_name[selected_name]
                existing_item_index = next((i for i, item in enumerate(st.session_state.order_items) if item['id'] == item_to_add['id']), None)
                if existing_item_index is not None:
                    st.session_state.order_items[existing_item_index]['order_quantity'] += 1
                else:
                    st.session_state.order_items.append(dict(item_to_add, **{'order_quantity': 1}))
                st.rerun()
        else:
            img_buffer = st.camera_input("Apunta la cámara al objetivo", key="order_camera")
            if img_buffer:
                pil_image = Image.open(img_buffer)

                if add_method == "Escáner de Código":
                    with st.spinner("Buscando códigos..."):
                        decoded_objects = enhanced_barcode_reader(pil_image)
                        if decoded_objects:
                            code = decoded_objects[0].data.decode('utf-8')
                            if code in inventory_by_id:
                                item_to_add = inventory_by_id[code]
                                existing_item_index = next((i for i, item in enumerate(st.session_state.order_items) if item['id'] == code), None)
                                if existing_item_index is not None:
                                    st.session_state.order_items[existing_item_index]['order_quantity'] += 1
                                else:
                                    st.session_state.order_items.append(dict(item_to_add, **{'order_quantity': 1}))
                                st.success(f"'{item_to_add['name']}' añadido/actualizado en el pedido.")
                                st.rerun()
                            else:
                                st.error(f"El código '{code}' no se encontró en el inventario. Por favor, créalo primero.")
                        else:
                            st.warning("No se detectaron códigos.")
                
                elif add_method == "Detección de Objetos":
                    with st.spinner("Detectando objetos..."):
                         results = yolo(pil_image)
                    st.image(results[0].plot(), caption="Objetos detectados.", use_column_width=True)
                    detections = results[0]
                    if detections.boxes:
                        st.subheader("Selecciona un objeto para añadir al pedido:")
                        cols = st.columns(min(len(detections.boxes), 4))
                        for i, box in enumerate(detections.boxes):
                            class_name = detections.names[box.cls[0].item()]
                            if cols[i % 4].button(f"Buscar '{class_name}'", key=f"add_obj_{i}"):
                                found_item = next((item for item in items_from_db if class_name.lower() in item.get('name', '').lower()), None)
                                if found_item:
                                    existing_item_index = next((i for i, item in enumerate(st.session_state.order_items) if item['id'] == found_item['id']), None)
                                    if existing_item_index is not None:
                                        st.session_state.order_items[existing_item_index]['order_quantity'] += 1
                                    else:
                                        st.session_state.order_items.append(dict(found_item, **{'order_quantity': 1}))
                                    st.success(f"'{found_item['name']}' añadido/actualizado.")
                                    st.rerun()
                                else:
                                    st.error(f"No se encontró un artículo llamado '{class_name}' en el inventario.")
                    else:
                        st.warning("No se detectaron objetos.")
    with col2:
        st.subheader("Detalle del Pedido Actual")
        if not st.session_state.order_items:
            st.info("Añade artículos para comenzar un pedido.")
        else:
            total_price = 0
            items_to_remove_indices = []
            for i, item in enumerate(st.session_state.order_items):
                c1, c2, c3, c4 = st.columns([4,2,2,1])
                c1.text(item['name'])
                item_id = item.get('id', f'item_{i}')
                new_qty = c2.number_input("Cantidad", value=item['order_quantity'], min_value=1, key=f"qty_{item_id}_{i}")
                st.session_state.order_items[i]['order_quantity'] = new_qty
                item_total = item.get('sale_price', 0) * new_qty
                c3.text(f"${item_total:,.2f}")
                total_price += item_total
                if c4.button("🗑️", key=f"del_{item_id}_{i}"):
                    items_to_remove_indices.append(i)
            if items_to_remove_indices:
                for index in sorted(items_to_remove_indices, reverse=True):
                    st.session_state.order_items.pop(index)
                st.rerun()

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
                    st.session_state.order_items = []; st.rerun()
    st.markdown("---")
    st.subheader("⏳ Pedidos en Proceso")
    processing_orders = firebase.get_orders('processing')
    if not processing_orders:
        st.info("No hay pedidos en proceso.")
    else:
        for order in processing_orders:
            with st.expander(f"**{order['title']}** - ${order.get('price', 0):,.2f}"):
                st.write("Artículos:")
                for item in order.get('ingredients', []):
                    st.write(f"- {item.get('name')} (x{item.get('quantity')})")
                c1, c2 = st.columns(2)
                if c1.button("✅ Completar Pedido", key=f"comp_{order['id']}", type="primary", use_container_width=True):
                    success, msg, alerts = firebase.complete_order(order['id'])
                    if success:
                        st.success(msg); send_whatsapp_alert(f"✅ Pedido Completado: {order['title']}")
                        for alert in alerts: send_whatsapp_alert(f"📉 ALERTA DE STOCK: {alert}")
                        st.rerun()
                    else: st.error(msg)
                if c2.button("❌ Cancelar Pedido", key=f"canc_{order['id']}", use_container_width=True):
                    firebase.cancel_order(order['id']); st.rerun()

elif st.session_state.page == "📊 Analítica":
    try:
        completed_orders = firebase.get_orders('completed')
        all_inventory_items = firebase.get_all_inventory_items()
    except Exception as e:
        st.error(f"No se pudieron cargar los datos para el análisis: {e}"); st.stop()
    if not completed_orders:
        st.info("No hay pedidos completados para generar analíticas.")
    else:
        tab1, tab2, tab3 = st.tabs(["💰 Rendimiento Financiero", "🔄 Rotación de Inventario", "📈 Predicción de Demanda"])
        with tab1:
            st.subheader("Indicadores Clave de Rendimiento (KPIs)")
            total_revenue = sum(o.get('price', 0) for o in completed_orders)
            total_cogs = sum(ing.get('purchase_price', 0) * ing.get('quantity', 0) for o in completed_orders for ing in o.get('ingredients', []))
            gross_profit = total_revenue - total_cogs
            num_orders = len(completed_orders)
            avg_order_value = total_revenue / num_orders if num_orders > 0 else 0
            profit_margin = (gross_profit / total_revenue) * 100 if total_revenue > 0 else 0
            kpi_cols = st.columns(5)
            kpi_cols[0].metric("Ingresos Totales", f"${total_revenue:,.2f}")
            kpi_cols[1].metric("Beneficio Bruto", f"${gross_profit:,.2f}")
            kpi_cols[2].metric("Margen de Beneficio", f"{profit_margin:.2f}%")
            kpi_cols[3].metric("Pedidos Completados", num_orders)
            kpi_cols[4].metric("Valor Promedio/Pedido", f"${avg_order_value:,.2f}")
            st.markdown("---")
            st.subheader("Tendencia de Ingresos y Beneficios Diarios")
            sales_data = []
            for order in completed_orders:
                if 'timestamp_obj' in order and order['timestamp_obj'] is not None:
                    order_profit = order.get('price', 0) - sum(ing.get('purchase_price', 0) * ing.get('quantity', 0) for ing in order.get('ingredients', []))
                    sales_data.append({'Fecha': order['timestamp_obj'].date(), 'Ingresos': order.get('price', 0), 'Beneficios': order_profit})
            if sales_data:
                df_trends = pd.DataFrame(sales_data).groupby('Fecha').sum()
                st.line_chart(df_trends)
            else:
                st.warning("No hay suficientes datos de fecha para generar un gráfico de tendencias.")
        with tab2:
            all_items_sold = [ing for o in completed_orders for ing in o.get('ingredients', [])]
            item_sales, item_profits = {}, {}
            for item in all_items_sold:
                if 'name' in item:
                    item_sales[item['name']] = item_sales.get(item['name'], 0) + item.get('quantity', 0)
                    profit = (item.get('sale_price', item.get('purchase_price', 0)) - item.get('purchase_price', 0)) * item.get('quantity', 0)
                    item_profits[item['name']] = item_profits.get(item['name'], 0) + profit
            df_sales = pd.DataFrame(list(item_sales.items()), columns=['Artículo', 'Unidades Vendidas']).sort_values('Unidades Vendidas', ascending=False)
            df_profits = pd.DataFrame(list(item_profits.items()), columns=['Artículo', 'Beneficio Generado']).sort_values('Beneficio Generado', ascending=False)
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Top 5 - Artículos Más Vendidos")
                st.dataframe(df_sales.head(5), hide_index=True)
            with col2:
                st.subheader("Top 5 - Artículos Más Rentables")
                st.dataframe(df_profits.head(5), hide_index=True)
            st.markdown("---")
            st.subheader("Inventario de Lenta Rotación (no vendido en los últimos 30 días)")
            thirty_days_ago = datetime.now(timezone.utc) - timedelta(days=30)
            sold_item_ids = {ing['id'] for o in completed_orders if o.get('timestamp_obj') and o['timestamp_obj'].replace(tzinfo=timezone.utc) > thirty_days_ago for ing in o.get('ingredients', [])}
            slow_moving_items = [item for item in all_inventory_items if item['id'] not in sold_item_ids]
            if not slow_moving_items:
                st.success("¡Todos los artículos han tenido movimiento en los últimos 30 días!")
            else:
                for item in slow_moving_items:
                    st.warning(f"- **{item.get('name')}** (Stock actual: {item.get('quantity')})")
        with tab3:
            st.subheader("Predecir Demanda Futura de un Artículo")
            item_names = [item['name'] for item in all_inventory_items if 'name' in item]
            item_to_predict = st.selectbox("Selecciona un artículo:", item_names)
            if item_to_predict:
                sales_history = []
                for order in completed_orders:
                    for item in order.get('ingredients', []):
                        if item.get('name') == item_to_predict and order.get('timestamp_obj'):
                            sales_history.append({'date': order['timestamp_obj'], 'quantity': item['quantity']})
                
                df_hist = pd.DataFrame(sales_history)
                
                # --- INICIO DE LA CORRECCIÓN: LÓGICA DE PREDICCIÓN MEJORADA ---
                if df_hist.empty:
                    st.warning("No hay historial de ventas para este artículo.")
                else:
                    df_hist['date'] = pd.to_datetime(df_hist['date'])
                    df_hist = df_hist.set_index('date').resample('D').sum().fillna(0)

                    MIN_DAYS_FOR_SEASONAL = 14  # 2 ciclos de 7 días
                    MIN_DAYS_FOR_SIMPLE = 5

                    if len(df_hist) < MIN_DAYS_FOR_SIMPLE:
                        st.warning(f"No hay suficientes datos para una predicción fiable. Se necesitan al menos {MIN_DAYS_FOR_SIMPLE} días de ventas.")
                    else:
                        try:
                            model = None
                            if len(df_hist) >= MIN_DAYS_FOR_SEASONAL:
                                st.info("Datos suficientes. Usando modelo de predicción estacional.")
                                model = ExponentialSmoothing(df_hist['quantity'], seasonal='add', seasonal_periods=7, trend='add').fit()
                            else:
                                st.info("Datos insuficientes para estacionalidad. Usando modelo de tendencia simple.")
                                model = ExponentialSmoothing(df_hist['quantity'], trend='add').fit()
                            
                            prediction = model.forecast(30)
                            prediction[prediction < 0] = 0 # Evitar predicciones negativas
                            
                            st.success(f"Se estima una demanda de **{int(round(prediction.sum()))} unidades** para los próximos 30 días.")
                            st.line_chart(prediction)
                        except Exception as e:
                            st.error(f"No se pudo generar la predicción: {e}")
                # --- FIN DE LA CORRECCIÓN ---


elif st.session_state.page == "👥 Acerca de":
    st.header("Sobre el Proyecto y sus Creadores")
    with st.container(border=True):
        col_img_est, col_info_est = st.columns([1, 3])
        with col_img_est:
            st.image("https://avatars.githubusercontent.com/u/129755299?v=4", width=200, caption="Joseph Javier Sánchez Acuña")
        with col_info_est:
            st.title("Joseph Javier Sánchez Acuña")
            st.subheader("Estudiante de Ingeniería Industrial")
            st.subheader("Experto en Inteligencia Artificial y Desarrollo de Software.")
            st.markdown(
                """
                - **LinkedIn:** [joseph-javier-sánchez-acuña](https://www.linkedin.com/in/joseph-javier-sánchez-acuña-150410275)
                - **GitHub:** [GIUSEPPESAN21](https://github.com/GIUSEPPESAN21)
                - **Email:** [joseph.sanchez@uniminuto.edu.co](mailto:joseph.sanchez@uniminuto.edu.co)
                """
            )







