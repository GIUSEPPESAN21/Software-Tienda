import logging
from firebase_utils import FirebaseManager

# Configuración del logger para este módulo
logger = logging.getLogger(__name__)

class BarcodeManager:
    """
    Gestiona toda la lógica de negocio relacionada con el escaneo de códigos de barras,
    actuando como intermediario entre la interfaz de usuario y la base de datos.
    """
    def __init__(self, firebase_manager: FirebaseManager):
        """
        Inicializa el gestor con una instancia del manejador de Firebase.

        Args:
            firebase_manager (FirebaseManager): Instancia para interactuar con Firestore.
        """
        self.db = firebase_manager

    def handle_inventory_scan(self, barcode: str):
        """
        Procesa un código de barras escaneado en el modo de gestión de inventario.
        Verifica si el producto existe y devuelve su estado.

        Args:
            barcode (str): El código de barras escaneado.

        Returns:
            dict: Un diccionario con el estado ('found' o 'not_found') y los datos del producto
                  o el código de barras si no se encuentra.
        """
        if not barcode:
            return {'status': 'error', 'message': 'El código de barras no puede estar vacío.'}
        
        try:
            item = self.db.get_inventory_item_details(barcode)
            if item:
                logger.info(f"Producto encontrado para el código '{barcode}': {item['name']}")
                return {'status': 'found', 'item': item}
            else:
                logger.info(f"Producto no encontrado para el código '{barcode}'. Se solicitará creación.")
                return {'status': 'not_found', 'barcode': barcode}
        except Exception as e:
            logger.error(f"Error al procesar el escaneo de inventario para '{barcode}': {e}")
            return {'status': 'error', 'message': str(e)}

    def add_item_to_sale(self, barcode: str, current_sale_items: list):
        """
        Añade un artículo a una venta en curso o incrementa su cantidad si ya existe.
        Realiza una comprobación de stock antes de añadirlo.

        Args:
            barcode (str): El código de barras del producto a añadir.
            current_sale_items (list): La lista de artículos en la venta actual.

        Returns:
            tuple: (list, dict) La lista de venta actualizada y un mensaje de estado
                   (éxito, advertencia o error).
        """
        if not barcode:
            return current_sale_items, {'status': 'error', 'message': 'El código de barras no puede estar vacío.'}

        try:
            item_data = self.db.get_inventory_item_details(barcode)
            if not item_data:
                return current_sale_items, {'status': 'error', 'message': f"Producto con código '{barcode}' no encontrado."}

            # Comprobar si hay suficiente stock
            if item_data.get('quantity', 0) <= 0:
                 return current_sale_items, {'status': 'warning', 'message': f"¡Stock agotado para '{item_data['name']}'!"}

            # Buscar si el artículo ya está en la venta
            existing_item = next((item for item in current_sale_items if item['id'] == barcode), None)
            
            if existing_item:
                # Verificar stock antes de incrementar
                if item_data.get('quantity', 0) > existing_item['quantity']:
                    existing_item['quantity'] += 1
                    msg = {'status': 'success', 'message': f"'{item_data['name']}' (+1). Total: {existing_item['quantity']}"}
                else:
                    msg = {'status': 'warning', 'message': f"No hay más stock disponible para '{item_data['name']}'."}
            else:
                # Añadir nuevo artículo a la venta
                new_item = {
                    'id': item_data['id'],
                    'name': item_data['name'],
                    'sale_price': item_data.get('sale_price', 0),
                    'purchase_price': item_data.get('purchase_price', 0),
                    'quantity': 1 # Cantidad para esta venta
                }
                current_sale_items.append(new_item)
                msg = {'status': 'success', 'message': f"'{item_data['name']}' añadido a la venta."}
            
            return current_sale_items, msg

        except Exception as e:
            logger.error(f"Error al añadir artículo a la venta '{barcode}': {e}")
            return current_sale_items, {'status': 'error', 'message': str(e)}

