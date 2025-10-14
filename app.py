# -*- coding: utf-8 -*-
"""
Aplicación Streamlit Unificada para Gestión de Inventario con IA.
Combina el reconocimiento de objetos con la gestión de pedidos,
utilizando Firebase como base de datos central.
"""
import streamlit as st
from PIL import Image
import numpy as np
import cv2
import pandas as pd
import plotly.express as px
import json
from collections import Counter
import random
import base64
from datetime import datetime

# --- Importaciones de utilidades y dependencias ---
from firebase_utils import FirebaseManager
from gemini_utils import GeminiUtils
from ultralytics import YOLO

try:
    from twilio.rest import Client
    from twilio.base.exceptions import TwilioRestException
    IS_TWILIO_AVAILABLE = True
except ImportError:
    IS_TWILIO_AVAILABLE = False
    Client, TwilioRestException = None, None

# --- CONFIGURACIÓN DE PÁGINA Y ESTILOS ---
st.set_page_config(
    page_title="HI-DRIVE | Gestión de Inventario IA",
    page_icon="🤖",
    layout="wide"
)

# --- FUNCIÓN PARA CARGAR Y CODIFICAR EL LOGO ---
@st.cache_data
def get_logo_base_64(file_path):
    try:
        with open(file_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    except FileNotFoundError:
        return None

# --- INYECCIÓN DE CSS PARA UNA INTERFAZ MEJORADA ---
logo_base64 = get_logo_base_64("logo.jpg")
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');

    /* --- Estilos Generales --- */
    body, .stApp {{
        font-family: 'Poppins', sans-serif;
        background-color: #f8f9fa;
    }}

    /* --- Logo en la parte superior --- */
    .main-header-container {{
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 20px;
        margin-bottom: 0rem;
    }}
    .main-header-container img {{
        height: 80px;
    }}
    .main-header {{
        font-size: 3rem;
        font-weight: 700;
        color: #264653; /* Azul oscuro */
        text-align: center;
    }}
    
    /* --- Estilos de Navegación y Botones --- */
    h2, h3 {{ color: #2a9d8f; font-weight: 600; }}
    div[role="radiogroup"] > div {{ display: flex; flex-wrap: wrap; justify-content: center; gap: 1rem; background-color: #ffffff; padding: 1rem; border-radius: 1rem; margin-bottom: 2rem; box-shadow: 0 4px 12px rgba(0,0,0,0.08); }}
    div[role="radiogroup"] label {{ background-color: #f8f9fa; padding: 0.7rem 1.5rem; border-radius: 0.75rem; border: 1px solid #e0e0e0; cursor: pointer; transition: all 0.3s ease; font-weight: 500; }}
    div[role="radiogroup"] [aria-checked="true"] {{ background-color: #2a9d8f; color: white; border-color: #2a9d8f; box-shadow: 0 4px 14px rgba(42, 157, 143, 0.4); transform: translateY(-2px); }}
    .stButton > button {{ border-radius: 0.75rem; padding: 10px 22px; font-weight: 600; transition: all 0.2s ease; border: none; }}
    .stButton > button[kind="primary"] {{ background-color: #e76f51; color: white; box-shadow: 0 4px 14px rgba(231, 111, 81, 0.3); }}
    .stButton > button[kind="primary"]:hover {{ transform: translateY(-2px); box-shadow: 0 6px 20px rgba(231, 111, 81, 0.4); }}
    .stButton > button:not([kind="primary"]) {{ background-color: #ffffff; color: #555; border: 1px solid #ddd; }}
    .stButton > button:not([kind="primary"]):hover {{ transform: translateY(-2px); box-shadow: 0 4px 12px rgba(0,0,0,0.1); border-color: #2a9d8f; color: #2a9d8f; }}
    
    /* --- Estilos de Contenedores y Métricas --- */
    .stMetric {{ background-color: #ffffff; padding: 2rem; border-radius: 1rem; box-shadow: 0 4px 12px rgba(0,0,0,0.08); border-left: 5px solid #2a9d8f; }}
    .stExpander {{ background-color: #ffffff; border-radius: 1rem !important; border: none !important; box-shadow: 0 4px 12px rgba(0,0,0,0.08); }}
    .report-box {{ background-color: #e9f5f4; padding: 1.5rem; border-radius: 1rem; border-left: 5px solid #2a9d8f; margin-bottom: 1rem; }}

    /* --- Estilos para la lista de inventario --- */
    .inventory-item {{
        background-color: #ffffff;
        padding: 1rem 1.5rem;
        border-radius: 0.75rem;
        margin-bottom: 0.75rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
        border-left: 5px solid #e9c46a;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }}
    .inventory-item .name {{
        font-weight: 600;
        color: #264653;
        font-size: 1.1rem;
    }}
    .inventory-item .details {{
        font-size: 0.9rem;
        color: #555;
    }}
    .inventory-item .quantity {{
        font-weight: 700;
        font-size: 1.2rem;
        color: #2a9d8f;
        background-color: #e9f5f4;
        padding: 0.3rem 0.8rem;
        border-radius: 0.5rem;
    }}
</style>
""", unsafe_allow_html=True)

# --- INICIALIZACIÓN DE SERVICIOS ---
@st.cache_resource
def initialize_services():
    try:
        yolo_model = YOLO('yolov8m.pt')
        firebase_handler = FirebaseManager()
        gemini_handler = GeminiUtils()
        return yolo_model, firebase_handler, gemini_handler
    except Exception as e:
        st.error(f"**Error Crítico de Inicialización.** No se pudo cargar un modelo o conectar a un servicio.")
        st.code(f"Detalle: {e}", language="bash")
        return None, None, None

yolo_model, firebase, gemini = initialize_services()

if not all([yolo_model, firebase, gemini]):
    st.stop()
    
# --- LÓGICA DE TWILIO ---
@st.cache_resource
def inicializar_twilio_client():
    if not IS_TWILIO_AVAILABLE: return None
    try:
        if hasattr(st, 'secrets') and all(k in st.secrets for k in ["TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN"]):
            return Client(st.secrets["TWILIO_ACCOUNT_SID"], st.secrets["TWILIO_AUTH_TOKEN"])
    except Exception: return None
    return None

twilio_client = inicializar_twilio_client()

def enviar_alerta_whatsapp(mensaje):
    if not twilio_client:
        st.warning("Cliente de Twilio no inicializado. No se pueden enviar alertas.")
        return
    try:
        from_number = st.secrets["TWILIO_WHATSAPP_FROM_NUMBER"]
        to_number = st.secrets["DESTINATION_WHATSAPP_NUMBER"]
        mensaje_final = f"Your Twilio code is {random.randint(1000,9999)}\n\n{mensaje}"
        twilio_client.messages.create(from_=f'whatsapp:{from_number}', body=mensaje_final, to=f'whatsapp:{to_number}')
        st.toast("¡Alerta de WhatsApp enviada!", icon="📲")
    except Exception as e:
        st.error(f"Error de Twilio: {e}", icon="🚨")

# --- NAVEGACIÓN PRINCIPAL ---
if logo_base64:
    st.markdown(
        f'<div class="main-header-container">'
        f'<img src="data:image/jpeg;base64,{logo_base64}" alt="HI-DRIVE Logo">'
        f'<h1 class="main-header">HI-DRIVE</h1>'
        f'</div>',
        unsafe_allow_html=True
    )
else:
    st.markdown('<h1 class="main-header">🌟 HI-DRIVE | Sistema de Inventario</h1>', unsafe_allow_html=True)

page = st.radio(
    "Navegación del Sistema",
    ["🏠 Inicio", "📸 Análisis de Imagen", "📦 Inventario", "🛒 Pedidos", "📊 Dashboard", "👥 Acerca de"],
    horizontal=True,
    label_visibility="collapsed"
)

st.markdown("---")

# --- LÓGICA DE LAS PÁGINAS ---

if page == "🏠 Inicio":
    st.subheader("Una solución unificada que integra IA para reconocimiento y gestión completa de inventario y pedidos.")
    st.markdown("---")
    
    try:
        items = firebase.get_all_inventory_items()
        orders = firebase.get_orders(status=None)
        item_count = len(items)
        processing_orders = len([o for o in orders if o.get('status') == 'processing'])
        
        col1, col2, col3 = st.columns(3, gap="large")
        col1.metric("📦 Artículos en Inventario", item_count)
        col2.metric("⏳ Pedidos en Proceso", processing_orders)
        col3.metric("✅ Pedidos Completados", len([o for o in orders if o.get('status') == 'completed']))
    except Exception as e:
        st.warning(f"No se pudieron cargar las estadísticas: {e}")
    
    st.markdown("---")
    st.subheader("Funcionalidades Principales:")
    st.markdown("""
    - **Análisis de Imagen**: Usa la IA de Gemini para identificar, contar y categorizar productos automáticamente.
    - **Gestión de Inventario**: Añade, busca y elimina artículos de tu inventario en tiempo real.
    - **Gestión de Pedidos**: Crea nuevos pedidos, procésalos descontando el stock y mantén un historial.
    - **Dashboard**: Visualiza la composición y actividad de tu inventario con gráficos interactivos.
    - **Alertas**: Recibe notificaciones por WhatsApp cuando se crean o completan pedidos.
    """)

elif page == "📸 Análisis de Imagen":
    st.header("📸 Detección y Análisis de Objetos por Imagen")

    if 'analysis_in_progress' in st.session_state and st.session_state.analysis_in_progress:
        st.subheader("✔️ Resultado del Análisis de Gemini")
        analysis_text = st.session_state.last_analysis
        
        try:
            clean_json_str = analysis_text.strip().replace("```json", "").replace("```", "")
            analysis_data = json.loads(clean_json_str)
            
            if "error" not in analysis_data:
                # Muestra los resultados del análisis
                st.markdown('<div class="report-box">', unsafe_allow_html=True)
                st.write(f"<span class='report-header'>Elemento Identificado:</span> <span class='report-data'>{analysis_data.get('elemento_identificado', 'N/A')}</span>", unsafe_allow_html=True)
                st.write(f"<span class='report-header'>Cantidad Aproximada:</span> <span class='report-data'>{analysis_data.get('cantidad_aproximada', 'N/A')}</span>", unsafe_allow_html=True)
                st.write(f"<span class='report-header'>Condición/Estado:</span> <span class='report-data'>{analysis_data.get('estado_condicion', 'N/A')}</span>", unsafe_allow_html=True)
                st.write(f"<span class='report-header'>Categoría Sugerida:</span> <span class='report-data'>{analysis_data.get('posible_categoria_de_inventario', 'N/A')}</span>", unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

                # Formulario para guardar en la base de datos
                with st.form("save_to_db_form"):
                    st.subheader("💾 Registrar en Inventario")
                    custom_id = st.text_input("ID Personalizado (SKU):", key="custom_id")
                    description = st.text_input("Descripción:", value=analysis_data.get('elemento_identificado', ''))
                    quantity = st.number_input("Unidades:", min_value=1, value=analysis_data.get('cantidad_aproximada', 1), step=1)
                    
                    if st.form_submit_button("Añadir a la Base de Datos", type="primary", use_container_width=True):
                        if not custom_id or not description:
                            st.warning("El ID y la Descripción son obligatorios.")
                        else:
                            with st.spinner("Guardando..."):
                                data_to_save = {
                                    "name": description,
                                    "quantity": quantity,
                                    "tipo": "imagen",
                                    "analisis_ia": analysis_data,
                                    "timestamp": datetime.now().isoformat()
                                }
                                firebase.save_inventory_item(data_to_save, custom_id)
                                st.success(f"¡Artículo '{description}' guardado con éxito!")
                                st.session_state.analysis_in_progress = False
                                st.rerun()
            else:
                 st.error(f"Error de Gemini: {analysis_data['error']}")
        except json.JSONDecodeError:
            st.error("La IA devolvió un formato inesperado. Por favor, intenta de nuevo.")
            st.code(analysis_text, language='text')

        if st.button("↩️ Analizar otra imagen"):
            st.session_state.analysis_in_progress = False
            st.rerun()
    else:
        # Interfaz para capturar o subir la imagen
        img_source = st.radio("Elige la fuente de la imagen:", ["Cámara en vivo", "Subir un archivo"], horizontal=True)
        img_buffer = None
        if img_source == "Cámara en vivo":
            img_buffer = st.camera_input("Apunta la cámara a los objetos", key="camera_input")
        else:
            img_buffer = st.file_uploader("Sube un archivo de imagen", type=['png', 'jpg', 'jpeg'], key="file_uploader")

        if img_buffer:
            pil_image = Image.open(img_buffer)
            
            with st.spinner("🧠 Detectando objetos con IA local (YOLO)..."):
                results = yolo_model(pil_image)

            st.subheader("🔍 Objetos Detectados")
            annotated_image = results[0].plot()
            st.image(annotated_image, caption="Imagen con objetos detectados por YOLO.", use_column_width=True)

            detections = results[0]
            
            if detections.boxes:
                st.info(f"Se detectaron {len(detections.boxes)} objetos. Selecciona uno para un análisis detallado con Gemini.")
                # Crear columnas para los botones de análisis
                cols = st.columns(min(len(detections.boxes), 4))
                for i, box in enumerate(detections.boxes):
                    class_name = detections.names[box.cls[0].item()]
                    col = cols[i % 4]
                    if col.button(f"Analizar '{class_name}' #{i+1}", key=f"classify_{i}", use_container_width=True):
                        coords = box.xyxy[0].cpu().numpy().astype(int)
                        cropped_pil_image = pil_image.crop(tuple(coords))
                        
                        st.image(cropped_pil_image, caption=f"Recorte de '{class_name}' enviado a Gemini...")

                        with st.spinner("🤖 Gemini está analizando el recorte..."):
                            analysis_text = gemini.analyze_image(cropped_pil_image, f"Objeto detectado como {class_name}")
                            st.session_state.last_analysis = analysis_text
                            st.session_state.analysis_in_progress = True
                            st.rerun()
            else:
                st.warning("No se detectaron objetos conocidos en la imagen.")

elif page == "📦 Inventario":
    st.header("📦 Gestión de Inventario")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Inventario Actual en Firebase")
        if st.button("🔄 Refrescar Datos"): st.rerun()
        
        try:
            with st.spinner("Cargando inventario..."):
                items = firebase.get_all_inventory_items()
            
            if items:
                st.markdown('<div class="inventory-list">', unsafe_allow_html=True)
                for item in items:
                    name = item.get('name', 'N/A')
                    quantity = item.get('quantity', 0)
                    item_id = item.get('id', 'N/A')
                    tipo = item.get('tipo', 'N/A')
                    
                    st.markdown(
                        f"""
                        <div class="inventory-item">
                            <div>
                                <div class="name">{name}</div>
                                <div class="details">ID: {item_id} | Tipo: {tipo}</div>
                            </div>
                            <div class="quantity">{quantity if quantity is not None else '0'}</div>
                        </div>
                        """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("El inventario está vacío. ¡Añade tu primer artículo!")
                
        except Exception as e:
            st.error(f"No se pudo conectar con la base de datos: {e}")
    
    with col2:
        with st.container(border=True):
            st.subheader("➕ Añadir Artículo")
            with st.form("manual_add_form", clear_on_submit=True):
                custom_id = st.text_input("ID Personalizado (SKU)")
                name = st.text_input("Nombre o Descripción")
                quantity = st.number_input("Cantidad", min_value=1, step=1)
                
                if st.form_submit_button("Guardar Artículo", type="primary", use_container_width=True):
                    if not custom_id or not name:
                        st.warning("El ID y el Nombre son obligatorios.")
                    else:
                        data = {"name": name, "quantity": quantity, "tipo": "manual", "timestamp": datetime.now().isoformat()}
                        firebase.save_inventory_item(data, custom_id)
                        st.success(f"Artículo '{name}' guardado.")
                        st.rerun()
        
        if 'items' in locals() and items:
            with st.container(border=True):
                st.subheader("🗑️ Eliminar Artículo")
                item_to_delete_name = st.selectbox("Selecciona un artículo", [""] + [f"{item['name']} ({item['id']})" for item in items])
                if item_to_delete_name:
                    item_id_to_delete = item_to_delete_name.split('(')[-1].replace(')','')
                    if st.button(f"Eliminar '{item_to_delete_name.split('(')[0].strip()}'", type="primary", use_container_width=True):
                        firebase.delete_inventory_item(item_id_to_delete)
                        st.success(f"Artículo eliminado.")
                        st.rerun()

elif page == "🛒 Pedidos":
    st.header("🛒 Gestión de Pedidos")
    
    inventory_items = firebase.get_all_inventory_items()
    if inventory_items:
        inventory_map = {item['name']: item['id'] for item in inventory_items}
        inventory_names = [""] + sorted(inventory_map.keys())
    else:
        inventory_map = {}
        inventory_names = [""]
        st.warning("No hay artículos en el inventario para crear pedidos.", icon="📦")

    col1, col2 = st.columns(2, gap="large")

    with col1:
        with st.container(border=True):
            st.subheader("📝 Crear Nuevo Pedido")
            if 'order_ingredients' not in st.session_state: 
                st.session_state.order_ingredients = [{'name': '', 'quantity': 1, 'id': None}]

            for i, ing in enumerate(st.session_state.order_ingredients):
                c1, c2, c3 = st.columns([3, 1, 1])
                selected_name = c1.selectbox(f"Ingrediente {i+1}", inventory_names, key=f"ing_name_{i}", index=inventory_names.index(ing['name']) if ing['name'] in inventory_names else 0)
                ing['name'] = selected_name
                ing['id'] = inventory_map.get(selected_name)
                ing['quantity'] = c2.number_input("Cant.", min_value=1, step=1, key=f"ing_qty_{i}", value=ing['quantity'])
                if c3.button("➖", key=f"del_ing_{i}"):
                    st.session_state.order_ingredients.pop(i); st.rerun()
            
            if st.button("Añadir Ingrediente"):
                st.session_state.order_ingredients.append({'name': '', 'quantity': 1, 'id': None}); st.rerun()

            with st.form("order_form", clear_on_submit=True):
                title = st.text_input("Título del Pedido")
                price = st.number_input("Precio de Venta ($)", min_value=0.01, format="%.2f")
                if st.form_submit_button("Crear Pedido", type="primary", use_container_width=True):
                    valid_ings = [ing for ing in st.session_state.order_ingredients if ing['id']]
                    if not title or price <= 0 or not valid_ings:
                        st.error("El pedido debe tener título, precio e ingredientes válidos.")
                    else:
                        order_data = {'title': title, 'price': price, 'ingredients': valid_ings, 'status': 'processing', "timestamp": datetime.now().isoformat()}
                        firebase.create_order(order_data)
                        st.success(f"Pedido '{title}' creado.")
                        enviar_alerta_whatsapp(f"🧾 Nuevo Pedido: {title} por ${price:.2f}")
                        st.session_state.order_ingredients = [{'name': '', 'quantity': 1, 'id': None}]; st.rerun()
    with col2:
        st.subheader("⏳ Pedidos en Proceso")
        processing_orders = firebase.get_orders(status='processing')
        if not processing_orders:
            st.info("No hay pedidos en proceso.")
        for order in processing_orders:
            with st.container(border=True):
                st.subheader(f"{order['title']} - ${order.get('price', 0):.2f}")
                st.caption(f"Ingredientes: {', '.join([f'{i['name']} (x{i['quantity']})' for i in order['ingredients']])}")
                b1, b2 = st.columns(2)
                if b1.button("✅ Completar", key=f"comp_{order['id']}", type="primary", use_container_width=True):
                    with st.spinner("Procesando..."):
                        success, message = firebase.complete_order(order['id'])
                    if success:
                        st.success(message)
                        enviar_alerta_whatsapp(f"✅ Pedido Completado: {order['title']}")
                        st.rerun()
                    else:
                        st.warning(message)
                if b2.button("❌ Cancelar", key=f"canc_{order['id']}", use_container_width=True):
                    firebase.cancel_order(order['id']); st.rerun()

    st.markdown("---")
    st.subheader("✅ Historial de Pedidos Completados")
    completed_orders = firebase.get_orders(status='completed')
    if completed_orders:
        df_completed = pd.DataFrame(completed_orders)
        st.dataframe(df_completed[['id', 'title', 'price']], hide_index=True, use_container_width=True)
    else:
        st.info("No hay pedidos en el historial.")

elif page == "📊 Dashboard":
    st.header("📊 Dashboard del Inventario")
    try:
        with st.spinner("Generando estadísticas..."):
            items = firebase.get_all_inventory_items()
        
        if items:
            df = pd.DataFrame(items)
            
            col1, col2 = st.columns(2, gap="large")
            with col1:
                st.subheader("Distribución por Tipo de Registro")
                type_counts = df['tipo'].value_counts()
                fig_pie = px.pie(
                    type_counts, 
                    values=type_counts.values, 
                    names=type_counts.index,
                    color_discrete_sequence=["#2a9d8f", "#e9c46a", "#f4a261"]
                )
                st.plotly_chart(fig_pie, use_container_width=True)

            with col2:
                st.subheader("Top 5 - Artículos con Mayor Stock")
                # Asegurarse que la columna 'quantity' sea numérica
                df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce').fillna(0)
                df_quant = df.sort_values('quantity', ascending=False).head(5)
                fig_bar = px.bar(
                    df_quant,
                    x='name',
                    y='quantity',
                    color='name',
                    labels={'name':'Artículo', 'quantity':'Cantidad'}
                )
                fig_bar.update_layout(showlegend=False)
                st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.warning("No hay datos en el inventario para generar un dashboard.")
    except Exception as e:
        st.error(f"Error al crear el dashboard: {e}")

elif page == "👥 Acerca de":
    st.header("👥 Sobre el Proyecto y sus Creadores")
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
