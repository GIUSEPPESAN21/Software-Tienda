import google.generativeai as genai
import logging
from PIL import Image
import streamlit as st
import json

# Configurar logging para ver el proceso en la consola
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeminiUtils:
    def __init__(self):
        # Obtener API key desde Streamlit secrets de forma segura
        self.api_key = st.secrets.get('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY no encontrada en los secrets de Streamlit")
        
        genai.configure(api_key=self.api_key)
        
        # Inicializa el modelo usando el método robusto de selección con tu lista de modelos
        self.model = self._get_available_model()
    
    def _get_available_model(self):
        """
        Intenta inicializar el mejor modelo de Gemini disponible de la lista proporcionada.
        """
        # Lista de modelos priorizada, AHORA INCLUYE el modelo experimental.
        model_candidates = [
            "gemini-2.0-flash-exp",       # Modelo experimental más reciente (prioridad 1)
            "gemini-1.5-flash-latest",    # Versión más reciente y rápida de 1.5
            "gemini-1.5-pro-latest",      # Versión Pro más reciente de 1.5
            "gemini-1.5-flash",           # Modelo Flash básico
            "gemini-1.5-pro",             # Modelo Pro básico
        ]
        
        for model_name in model_candidates:
            try:
                model = genai.GenerativeModel(model_name)
                # Se realiza una pequeña prueba para asegurar que el modelo es compatible con imágenes
                test_image = Image.new('RGB', (1, 1)) # Crea una imagen de 1x1 pixel
                model.generate_content(["test", test_image])
                logger.info(f"✅ Modelo de visión '{model_name}' inicializado y verificado con éxito.")
                return model # Retorna el primer modelo que funcione
            except Exception as e:
                logger.warning(f"⚠️ Modelo '{model_name}' no disponible o no compatible: {e}")
                continue # Si falla, intenta con el siguiente de la lista
        
        # Si ningún modelo de la lista funciona, se lanza un error crítico.
        raise Exception("No se pudo inicializar ningún modelo de visión de Gemini compatible. Verifica tu API Key y los modelos disponibles en tu cuenta.")
    
    def analyze_image(self, image_pil: Image, description: str = ""):
        """
        Analiza una imagen (en formato PIL) y devuelve una respuesta JSON estructurada y limpia.
        """
        try:
            # El prompt está optimizado para forzar una salida JSON limpia
            prompt = f"""
            Analiza esta imagen de un objeto de inventario.
            Descripción adicional del sistema de detección: "{description}"
            
            Tu tarea es actuar como un experto catalogador. Responde ÚNICAMENTE con un objeto JSON válido con las siguientes claves:
            - "elemento_identificado": (string) El nombre específico y descriptivo del objeto.
            - "cantidad_aproximada": (integer) El número de unidades que ves.
            - "estado_condicion": (string) La condición aparente (ej: "Nuevo en caja", "Usado").
            - "caracteristicas_distintivas": (string) Una lista de características visuales en una sola cadena de texto.
            - "posible_categoria_de_inventario": (string) La categoría más lógica (ej: "Electrónica").

            IMPORTANTE: Tu respuesta debe ser solo el objeto JSON, sin texto adicional, explicaciones, ni las marcas ```json.
            """
            
            response = self.model.generate_content([prompt, image_pil])
            
            if response and response.text:
                return response.text.strip()
            else:
                return json.dumps({"error": "La IA no devolvió una respuesta válida."})
                
        except Exception as e:
            logger.error(f"Error crítico durante el análisis de imagen con Gemini: {e}")
            return json.dumps({"error": f"No se pudo contactar al servicio de IA: {str(e)}"})

