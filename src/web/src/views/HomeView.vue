<script setup lang="ts">
import OrderTable from '@/components/OrderTable.vue';
import { useOrderStore } from '@/stores/orderStore';
import { onMounted } from 'vue';

const orderStore = useOrderStore();

onMounted(() => {
  orderStore.fetchOrders();
  orderStore.fetchOrdersContracts();
});
</script>

<template>
  <main>
    <h1>Dings und neue daten</h1>
    <p v-if="orderStore.loading">Lade Daten...</p>
    <p v-else-if="orderStore.error">Fehler: {{ orderStore.error }}</p>
    <div>
      <OrderTable :orders="orderStore.orderContracts" />
    </div>
    <div>
      <OrderTable :orders="orderStore.orders" />
    </div>
    <pre>
      {{ orderStore.orders }}
    </pre>
  </main>
</template>
