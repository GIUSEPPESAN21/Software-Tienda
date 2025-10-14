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
from datetime import datetime, timedelta
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
        st.session_state.order_ingredients = []

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
    "👥 Acerca de": "shield-check"
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
            st.session_state.page = "📦 Inventario"
            st.rerun()
        if st.button("📝 Crear Nuevo Pedido", use_container_width=True):
            st.session_state.page = "🛒 Pedidos"
            st.rerun()
    with col2:
        st.subheader("Alertas de Stock Bajo")
        low_stock_items = [item for item in items if item.get('quantity', 0) <= item.get('min_stock_alert', 0) and item.get('min_stock_alert', 0) > 0]
        if not low_stock_items:
            st.success("¡Todo el inventario está por encima del umbral mínimo!")
        else:
            for item in low_stock_items:
                st.warning(f"**{item['name']}**: {item['quantity']} unidades restantes (Umbral: {item['min_stock_alert']})")

# ----------------------------------
# PÁGINA: ANÁLISIS IA
# ----------------------------------
elif st.session_state.page == "📸 Análisis IA":
    source_options = ["🧠 Detección de Objetos", "║█║ Escáner de Código"]
    img_source = st.selectbox("Selecciona el modo de análisis:", source_options)

    img_buffer = st.camera_input("Apunta la cámara al objetivo", key="ia_camera")

    if img_buffer:
        pil_image = Image.open(img_buffer)
        
        # --- LÓGICA DE DETECCIÓN DE OBJETOS ---
        if img_source == "🧠 Detección de Objetos":
            with st.spinner("Detectando objetos con IA Local..."):
                results = yolo(pil_image)
            st.image(results[0].plot(), caption="Objetos detectados.", use_column_width=True)
            
            if results[0].boxes:
                class_name = results[0].names[results[0].boxes.cls[0].item()]
                st.info(f"Principal objeto detectado: **{class_name}**. Analizando con IA Avanzada...")
                with st.spinner("🤖 Gemini está analizando..."):
                    analysis_json_str = gemini.analyze_image(pil_image, class_name)
                    analysis_data = json.loads(analysis_json_str)
                    
                    if "error" in analysis_data:
                        st.error(f"Error de Gemini: {analysis_data['error']}")
                    else:
                        st.subheader("✔️ Resultado del Análisis")
                        st.markdown(f"""
                        - **Producto:** {analysis_data.get('elemento_identificado', 'N/A')}
                        - **Marca/Modelo:** {analysis_data.get('marca_modelo_sugerido', 'N/A')}
                        - **Cantidad Detectada:** {analysis_data.get('cantidad_aproximada', 'N/A')}
                        - **Categoría Sugerida:** {analysis_data.get('posible_categoria_de_inventario', 'N/A')}
                        - **Estado:** {analysis_data.get('estado_condicion', 'N/A')}
                        - **Características:** {analysis_data.get('caracteristicas_distintivas', 'N/A')}
                        """)
                        
                        with st.form("add_from_ia_form"):
                            st.subheader("➕ Agregar al Inventario")
                            custom_id = st.text_input("ID / SKU (usar código de barras si existe)", value=analysis_data.get('marca_modelo_sugerido', '').replace(" ", "-"))
                            name = st.text_input("Nombre del Artículo", value=analysis_data.get('elemento_identificado'))
                            quantity = st.number_input("Cantidad a registrar", value=analysis_data.get('cantidad_aproximada', 1), min_value=1)
                            
                            if st.form_submit_button("Guardar Artículo", type="primary"):
                                if not custom_id or not name:
                                    st.warning("El ID y el Nombre son obligatorios.")
                                else:
                                    item = firebase.get_inventory_item_details(custom_id)
                                    if item:
                                        st.error(f"¡Error! Ya existe un artículo con el ID '{custom_id}'. Use la página de Inventario para actualizarlo.")
                                    else:
                                        data = {"name": name, "quantity": quantity}
                                        firebase.save_inventory_item(data, custom_id, is_new=True)
                                        st.success(f"¡Nuevo artículo '{name}' guardado con éxito!")


        # --- LÓGICA DE ESCÁNER DE CÓDIGO ---
        elif img_source == "║█║ Escáner de Código":
            with st.spinner("Buscando códigos..."):
                decoded_objects = decode(pil_image)
                if not decoded_objects:
                    st.warning("No se encontraron códigos de barras o QR en la imagen.")
                for obj in decoded_objects:
                    code_data = obj.data.decode('utf-8')
                    st.success(f"Código detectado: **{code_data}**")
                    
                    item = firebase.get_inventory_item_details(code_data)
                    if item:
                        st.subheader("✔️ ¡Artículo Encontrado!")
                        st.markdown(f"""
                        - **Nombre:** {item.get('name', 'N/A')}
                        - **Stock Actual:** {item.get('quantity', 0)}
                        - **Precio de Compra:** ${item.get('purchase_price', 0):.2f}
                        - **Proveedor:** {item.get('supplier_name', 'N/A')}
                        """)
                    else:
                        st.info("Este código no corresponde a ningún artículo. Puede agregarlo a continuación.")
                        with st.form("add_from_scan_form"):
                             st.subheader("➕ Agregar Nuevo Artículo")
                             name = st.text_input("Nombre del Artículo")
                             quantity = st.number_input("Cantidad a registrar", min_value=1)
                             if st.form_submit_button("Guardar Artículo", type="primary"):
                                if name:
                                    data = {"name": name, "quantity": quantity}
                                    firebase.save_inventory_item(data, code_data, is_new=True)
                                    st.success(f"¡Nuevo artículo '{name}' guardado con el ID '{code_data}'!")
                                else:
                                    st.warning("El nombre es obligatorio.")

# ----------------------------------
# PÁGINA: INVENTARIO
# ----------------------------------
elif st.session_state.page == "📦 Inventario":
    tab1, tab2 = st.tabs(["📋 Lista de Inventario", "➕ Añadir / Actualizar Artículo"])

    with tab1:
        items = firebase.get_all_inventory_items()
        if not items:
            st.info("El inventario está vacío. Añade tu primer artículo en la pestaña de al lado.")
        else:
            for item in items:
                col1, col2, col3, col4 = st.columns([3, 2, 2, 2])
                with col1:
                    st.markdown(f"**{item.get('name')}**")
                    st.caption(f"ID: {item.get('id')}")
                with col2:
                    st.metric("Stock Actual", item.get('quantity', 0))
                with col3:
                    st.metric("Costo Compra", f"${item.get('purchase_price', 0):.2f}")
                with col4:
                    if st.button("Ver Detalles", key=f"details_{item['id']}", use_container_width=True):
                         st.session_state.selected_item_id = item['id']

                st.markdown("---")

            if 'selected_item_id' in st.session_state:
                item_details = firebase.get_inventory_item_details(st.session_state.selected_item_id)
                st.subheader(f"Detalles de: {item_details['name']}")
                st.write(f"**Proveedor:** {item_details.get('supplier_name', 'No asignado')}")
                st.write(f"**Alerta de Stock Mínimo:** {item_details.get('min_stock_alert', 'No definido')}")
                
                st.subheader("Historial de Movimientos")
                history = firebase.get_inventory_item_history(item_details['id'])
                if history:
                    df_hist = pd.DataFrame(history)
                    df_hist['timestamp'] = pd.to_datetime(df_hist['timestamp'])
                    st.dataframe(df_hist[['timestamp', 'type', 'quantity_change', 'details']], hide_index=True)
                else:
                    st.write("Sin movimientos registrados.")


    with tab2:
        st.subheader("Formulario de Artículo")
        suppliers = firebase.get_all_suppliers()
        supplier_map = {s['name']: s['id'] for s in suppliers}
        
        with st.form("add_item_form"):
            custom_id = st.text_input("ID Personalizado (SKU)")
            name = st.text_input("Nombre del Artículo")
            quantity = st.number_input("Cantidad Actual", min_value=0, step=1)
            purchase_price = st.number_input("Costo de Compra ($)", min_value=0.0, format="%.2f")
            min_stock_alert = st.number_input("Umbral de Alerta de Stock Mínimo", min_value=0, step=1)
            
            selected_supplier_name = st.selectbox("Proveedor", [""] + list(supplier_map.keys()))

            if st.form_submit_button("Guardar Artículo", type="primary", use_container_width=True):
                if custom_id and name:
                    data = {
                        "name": name, "quantity": quantity, "purchase_price": purchase_price,
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
# PÁGINA: PROVEEDORES
# ----------------------------------
elif st.session_state.page == "👥 Proveedores":
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

# ----------------------------------
# PÁGINA: PEDIDOS
# ----------------------------------
elif st.session_state.page == "🛒 Pedidos":
    items_from_db = firebase.get_all_inventory_items()
    inventory = {
        item['name']: {
            'id': item.get('id'), 
            'quantity': item.get('quantity', 0)
        } 
        for item in items_from_db if 'name' in item and item.get('name')
    }
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📝 Crear Nuevo Pedido")
        with st.form("order_form"):
            title = st.text_input("Nombre o Título del Pedido")
            price = st.number_input("Precio de Venta ($)", min_value=0.01, format="%.2f")
            
            st.markdown("**Artículos del Pedido:**")
            
            # Lógica para añadir ingredientes dinámicamente
            for i in range(len(st.session_state.order_ingredients)):
                c1,c2,c3 = st.columns([4,2,1])
                st.session_state.order_ingredients[i]['name'] = c1.selectbox(f"Artículo {i+1}", list(inventory.keys()), key=f"ing_name_{i}")
                st.session_state.order_ingredients[i]['quantity'] = c2.number_input("Cantidad", min_value=1, key=f"ing_qty_{i}")
                if c3.button("🗑️", key=f"del_ing_{i}"):
                    st.session_state.order_ingredients.pop(i)
                    st.rerun()

            if st.button("➕ Añadir Artículo al Pedido"):
                st.session_state.order_ingredients.append({'name': '', 'quantity': 1})
                st.rerun()

            if st.form_submit_button("Crear Pedido", type="primary", use_container_width=True):
                valid_ings = []
                for ing in st.session_state.order_ingredients:
                    if ing['name'] and inventory[ing['name']]['quantity'] >= ing['quantity']:
                        valid_ings.append({'id': inventory[ing['name']]['id'], 'name': ing['name'], 'quantity': ing['quantity']})
                    else:
                        st.error(f"Stock insuficiente para {ing['name']}.")
                
                if title and price > 0 and valid_ings:
                    order_data = {'title': title, 'price': price, 'ingredients': valid_ings, 'status': 'processing', 'timestamp': datetime.now()}
                    firebase.create_order(order_data)
                    st.success("Pedido creado.")
                    send_whatsapp_alert(f"🧾 Nuevo Pedido: {title} por ${price:.2f}")
                    st.session_state.order_ingredients = []
                    st.rerun()

    with col2:
        st.subheader("⏳ Pedidos en Proceso")
        for order in firebase.get_orders('processing'):
            with st.container(border=True):
                st.markdown(f"**{order['title']}** - ${order['price']:.2f}")
                items_str = ", ".join([f"{i['name']} (x{i['quantity']})" for i in order['ingredients']])
                st.caption(items_str)
                
                c1, c2 = st.columns(2)
                if c1.button("✅ Completar", key=f"comp_{order['id']}", type="primary", use_container_width=True):
                    success, msg, alerts = firebase.complete_order(order['id'])
                    if success:
                        st.success(msg)
                        send_whatsapp_alert(f"✅ Pedido Completado: {order['title']}")
                        for alert in alerts: send_whatsapp_alert(f"📉 ALERTA DE STOCK: {alert}")
                        st.rerun()
                    else: st.error(msg)
                if c2.button("❌ Cancelar", key=f"canc_{order['id']}", use_container_width=True):
                    firebase.cancel_order(order['id']); st.rerun()

# ----------------------------------
# PÁGINA: ANALÍTICA
# ----------------------------------
elif st.session_state.page == "📊 Analítica":
    tab1, tab2, tab3 = st.tabs(["💰 Rendimiento Financiero", "🔄 Rotación de Inventario", "📈 Predicción de Demanda"])
    
    completed_orders = firebase.get_orders('completed')
    if not completed_orders:
        st.info("No hay pedidos completados para generar analíticas.")
    else:
        with tab1:
            total_revenue = sum(o.get('price', 0) for o in completed_orders)
            total_cogs = sum(ing.get('purchase_price', 0) * ing.get('quantity', 0) for o in completed_orders for ing in o.get('ingredients', []))
            gross_profit = total_revenue - total_cogs
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Ingresos Totales", f"${total_revenue:,.2f}")
            c2.metric("Costo de Ventas (COGS)", f"${total_cogs:,.2f}")
            c3.metric("Beneficio Bruto", f"${gross_profit:,.2f}")

        with tab2:
            all_items_sold = [ing for o in completed_orders for ing in o.get('ingredients', [])]
            item_sales = {}
            for item in all_items_sold:
                item_sales[item['name']] = item_sales.get(item['name'], 0) + item['quantity']
            
            df_sales = pd.DataFrame(list(item_sales.items()), columns=['Artículo', 'Unidades Vendidas']).sort_values('Unidades Vendidas', ascending=False)
            
            st.subheader("Top 10 - Artículos Más Vendidos")
            st.dataframe(df_sales.head(10), hide_index=True)

            fig = px.bar(df_sales.head(10), x='Artículo', y='Unidades Vendidas', title="Rendimiento de Artículos")
            st.plotly_chart(fig, use_container_width=True)

        with tab3:
            st.subheader("Predecir Demanda Futura de un Artículo")
            inventory_items = firebase.get_all_inventory_items()
            item_to_predict = st.selectbox("Selecciona un artículo:", [item['name'] for item in inventory_items])

            if item_to_predict:
                sales_history = []
                for order in completed_orders:
                    for item in order.get('ingredients', []):
                        if item['name'] == item_to_predict:
                            sales_history.append({'date': order['timestamp'], 'quantity': item['quantity']})
                
                if len(sales_history) < 3:
                    st.warning("No hay suficientes datos de ventas para este artículo para hacer una predicción.")
                else:
                    df_hist = pd.DataFrame(sales_history)
                    df_hist['date'] = pd.to_datetime(df_hist['date'])
                    df_hist = df_hist.set_index('date').resample('D').sum()

                    model = ExponentialSmoothing(df_hist['quantity'], seasonal='add', seasonal_periods=7).fit()
                    prediction = model.forecast(30)
                    st.success(f"Se estima una demanda de **{int(prediction.sum())} unidades** para los próximos 30 días.")
                    
                    st.line_chart(prediction)

# ----------------------------------
# PÁGINA: ACERCA DE
# ----------------------------------
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


