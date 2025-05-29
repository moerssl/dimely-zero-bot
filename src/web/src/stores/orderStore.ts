import { defineStore } from 'pinia';
import { ref } from 'vue';

export const useOrderStore = defineStore('orderStore', () => {
  const orderContracts = ref<string | null>(null);
  const orders = ref<string | null>(null);
  const loading = ref<boolean>(false);
  const error = ref<string | null>(null);

  const fetchOrdersContracts = async () => {
    loading.value = true;
    error.value = null;

    try {
      const response = await fetch('http://localhost:8050/api/orders/contracts', {
        method: 'GET',
        headers: {
          'Accept': 'application/json', // Ensures the server returns JSON data
        },
      });
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      orderContracts.value = data; // Adjust the key based on your API response structure
    } catch (err: any) {
      error.value = err.message;
    } finally {
      loading.value = false;
    }
  };

  const fetchOrders = async () => {
    loading.value = true;
    error.value = null;

    try {
      const response = await fetch('http://localhost:8050/api/orders', {
        method: 'GET',
        headers: {
          'Accept': 'application/json', // Ensures the server returns JSON data
        },
      });
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      orders.value = data; // Adjust the key based on your API response structure
    } catch (err: any) {
      error.value = err.message;
    } finally {
      loading.value = false;
    }
  };

  return {
    orders,
    orderContracts,
    loading,
    error,
    fetchOrdersContracts,
    fetchOrders
  };
});
