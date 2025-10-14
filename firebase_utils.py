import firebase_admin
from firebase_admin import credentials, firestore
import json
import base64
import logging
from datetime import datetime
import streamlit as st

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- LÓGICA DE TRANSACCIÓN ATÓMICA ---
@firestore.transactional
def _complete_order_atomic(transaction, db, order_id):
    """
    Operación transaccional para completar un pedido. Garantiza que la actualización
    del stock y el estado del pedido ocurran de forma atómica (todo o nada).
    """
    order_ref = db.collection('orders').document(order_id)
    order_snapshot = order_ref.get(transaction=transaction)
    if not order_snapshot.exists:
        raise ValueError("El pedido no existe.")
    
    order_data = order_snapshot.to_dict()
    low_stock_alerts = []

    for ing in order_data.get('ingredients', []):
        if 'id' not in ing:
            raise ValueError(f"Dato inconsistente: al ingrediente '{ing.get('name')}' le falta su ID.")

        item_ref = db.collection('inventory').document(ing['id'])
        item_snapshot = item_ref.get(transaction=transaction)
        
        if not item_snapshot.exists:
            raise ValueError(f"Ingrediente '{ing.get('name')}' no encontrado en el inventario.")
        
        item_data = item_snapshot.to_dict()
        current_quantity = item_data.get('quantity', 0)
        
        if current_quantity < ing['quantity']:
            raise ValueError(f"Stock insuficiente para '{ing.get('name', ing['id'])}'. Se necesitan {ing['quantity']}, hay {current_quantity}.")
        
        new_quantity = current_quantity - ing['quantity']
        transaction.update(item_ref, {'quantity': new_quantity})

        # --- NUEVO: Registrar en el historial de movimientos ---
        history_ref = item_ref.collection('history').document()
        history_data = {
            "timestamp": datetime.now(),
            "type": "Venta (Pedido)",
            "quantity_change": -ing['quantity'],
            "details": f"Pedido ID: {order_id} - {order_data.get('title', 'N/A')}"
        }
        transaction.set(history_ref, history_data)

        # --- NUEVO: Comprobar alerta de stock mínimo ---
        min_stock_alert = item_data.get('min_stock_alert', 0)
        if 0 < new_quantity <= min_stock_alert:
            low_stock_alerts.append(f"'{item_data.get('name')}' ha alcanzado el umbral de stock mínimo ({new_quantity}/{min_stock_alert}).")

    transaction.update(order_ref, {'status': 'completed', 'completed_at': datetime.now()})
    return True, f"Pedido '{order_data['title']}' completado.", low_stock_alerts

class FirebaseManager:
    def __init__(self):
        self.db = None
        self.project_id = "reconocimiento-inventario"
        self._initialize_firebase()
    
    def _initialize_firebase(self):
        try:
            if not firebase_admin._apps:
                creds_base64 = st.secrets.get('FIREBASE_SERVICE_ACCOUNT_BASE64')
                if not creds_base64:
                    raise ValueError("El secret 'FIREBASE_SERVICE_ACCOUNT_BASE64' no fue encontrado.")
                
                creds_json_str = base64.b64decode(creds_base64).decode('utf-8')
                creds_dict = json.loads(creds_json_str)
                
                cred = credentials.Certificate(creds_dict)
                firebase_admin.initialize_app(cred, {'projectId': self.project_id})
                logger.info("Firebase inicializado correctamente.")
            
            self.db = firestore.client()
        except Exception as e:
            logger.error(f"Error fatal al inicializar Firebase: {e}")
            raise

    # --- Métodos de Inventario Mejorados ---
    def save_inventory_item(self, data, custom_id, is_new=False):
        try:
            doc_ref = self.db.collection('inventory').document(custom_id)
            doc_ref.set(data, merge=True)
            
            # Registrar en el historial
            history_type = "Stock Inicial" if is_new else "Ajuste Manual"
            history_data = {
                "timestamp": datetime.now(),
                "type": history_type,
                "quantity_change": data.get('quantity', 0),
                "details": "Artículo creado o actualizado manualmente."
            }
            doc_ref.collection('history').add(history_data)

            logger.info(f"Elemento de inventario guardado/actualizado: {custom_id}")
        except Exception as e:
            logger.error(f"Error al guardar en 'inventory': {e}")
            raise

    def get_inventory_item_details(self, doc_id):
        try:
            doc = self.db.collection('inventory').document(doc_id).get()
            if doc.exists:
                item = doc.to_dict()
                item['id'] = doc.id
                return item
            return None
        except Exception as e:
            logger.error(f"Error al obtener detalle de '{doc_id}': {e}")
            return None

    def get_all_inventory_items(self):
        try:
            docs = self.db.collection('inventory').stream()
            items = [dict(item.to_dict(), **{'id': item.id}) for item in docs]
            return sorted(items, key=lambda x: x.get('name', '').lower())
        except Exception as e:
            logger.error(f"Error al obtener de 'inventory': {e}")
            return []
            
    def get_inventory_item_history(self, doc_id):
        try:
            history_ref = self.db.collection('inventory').document(doc_id).collection('history').order_by("timestamp", direction=firestore.Query.DESCENDING).stream()
            return [record.to_dict() for record in history_ref]
        except Exception as e:
            logger.error(f"Error al obtener historial de '{doc_id}': {e}")
            return []

    def delete_inventory_item(self, doc_id):
        try:
            self.db.collection('inventory').document(doc_id).delete()
            # Opcional: Implementar borrado de sub-colecciones si es necesario
            logger.info(f"Elemento de inventario {doc_id} eliminado.")
        except Exception as e:
            logger.error(f"Error al eliminar de 'inventory': {e}")
            raise

    # --- Métodos de Pedidos Mejorados ---
    def create_order(self, order_data):
        try:
            # --- NUEVO: Enriquecer ingredientes con costo de compra ---
            enriched_ingredients = []
            for ing in order_data['ingredients']:
                item_details = self.get_inventory_item_details(ing['id'])
                if item_details:
                    ing['purchase_price'] = item_details.get('purchase_price', 0)
                enriched_ingredients.append(ing)
            
            order_data['ingredients'] = enriched_ingredients
            self.db.collection('orders').add(order_data)
            logger.info("Nuevo pedido creado con datos enriquecidos.")
        except Exception as e:
            logger.error(f"Error al crear pedido: {e}")
            raise

    def get_orders(self, status=None):
        try:
            query = self.db.collection('orders')
            if status:
                query = query.where(filter=firestore.FieldFilter('status', '==', status))
            
            docs = query.stream()
            orders = [dict(order.to_dict(), **{'id': order.id}) for order in docs]
            
            # Ordenar en Python para evitar la necesidad de un índice compuesto inmediato
            orders.sort(key=lambda x: x.get('timestamp'), reverse=True)
            
            return orders
        except Exception as e:
            logger.error(f"Error al obtener pedidos: {e}")
            return []

    def cancel_order(self, order_id):
        try:
            self.db.collection('orders').document(order_id).delete()
            logger.info(f"Pedido {order_id} cancelado.")
        except Exception as e:
            logger.error(f"Error al cancelar pedido: {e}")
            raise

    def complete_order(self, order_id):
        try:
            transaction = self.db.transaction()
            return _complete_order_atomic(transaction, self.db, order_id)
        except Exception as e:
            logger.error(f"Fallo la transacción para el pedido {order_id}: {e}")
            return False, f"Error en la transacción: {str(e)}", []
            
    # --- NUEVO: Métodos CRUD para Proveedores ---
    def add_supplier(self, supplier_data):
        try:
            self.db.collection('suppliers').add(supplier_data)
            logger.info("Nuevo proveedor añadido.")
        except Exception as e:
            logger.error(f"Error al añadir proveedor: {e}")
            raise

    def get_all_suppliers(self):
        try:
            docs = self.db.collection('suppliers').stream()
            suppliers = [dict(s.to_dict(), **{'id': s.id}) for s in docs]
            return sorted(suppliers, key=lambda x: x.get('name', '').lower())
        except Exception as e:
            logger.error(f"Error al obtener proveedores: {e}")
            return []

    def update_supplier(self, supplier_id, data):
        try:
            self.db.collection('suppliers').document(supplier_id).set(data, merge=True)
            logger.info(f"Proveedor {supplier_id} actualizado.")
        except Exception as e:
            logger.error(f"Error al actualizar proveedor: {e}")
            raise

    def delete_supplier(self, supplier_id):
        try:
            self.db.collection('suppliers').document(supplier_id).delete()
            logger.info(f"Proveedor {supplier_id} eliminado.")
        except Exception as e:
            logger.error(f"Error al eliminar proveedor: {e}")
            raise

