<template>
  <main>
    <h1>Candle Data</h1>
    <div style="min-height: 25px">

      <p v-if="orderStore.loading">Loading candles...</p>
      <p v-else-if="orderStore.error">Error: {{ orderStore.error }}</p>
    </div>
    <button @click="fetchCandles">Fetch Candles</button>
    <CandleStickChart :chartData="orderStore.candles" v-if="orderStore.candles"/>
    <pre>
      {{ orderStore.candles }}
    </pre>
  </main>

</template>
<script setup lang="ts">
import CandleStickChart from '@/components/CandleStickChart.vue';
import { useOrderStore } from '@/stores/orderStore';

const orderStore = useOrderStore();

const fetchCandles = async () => {
  try {
    await orderStore.fetchCandles();
  } catch (error) {
    console.error('Error fetching candles:', error);
  }
};

// fetch every 10 seconds
setInterval(fetchCandles, 5000);
fetchCandles();
</script>
<style scoped>
.chart {
  width: 100%;
  height: 700px;
  outline: 1px solid #ccc;
  display: flex;
}
main {
  padding: 20px;
  width: 100vw;

}
</style>
