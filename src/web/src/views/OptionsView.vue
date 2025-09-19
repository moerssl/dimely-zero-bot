<template>
  <div>
    <button @click="orderStore.fetchOptions">Reload</button>
    <p>Filter</p>
    <OrderTable :orders="resultOptions"></OrderTable>
    <hr />
    <p>All</p>
    <OrderTable :orders="orderStore.options"></OrderTable>
  </div>
</template>
<script setup>
import OrderTable from '@/components/OrderTable.vue';
import { useOrderStore } from '@/stores/orderStore';
import { onMounted, computed } from 'vue';

const orderStore = useOrderStore();

onMounted(() => {
  orderStore.fetchOptions();
});

setInterval(orderStore.fetchOptions,10000)

const resultOptions = computed(() => {
  const all = orderStore.options;
  if (all == undefined || all?.length == 0)
    return all;

  const boundaries = [-0.4, 0.4, -0.16, 0.16];
  const picked = [];
  const used = new Set();

  for (const boundary of boundaries) {
    let best = null;
    let bestDiff = Infinity;

    for (const op of all) {
      // Ensure numeric comparisons
      const delta = parseFloat(op.delta);
      const strike = parseFloat(op.Strike);
      const undPrice = parseFloat(op.undPrice);
      const type = op.Type;

      if (used.has(op)) continue;

      const isCall = type == 'C' && strike >= undPrice;
      const isPut = type == 'P' && strike <= undPrice;
      if (!isCall && !isPut) continue;

      const diff = Math.abs(delta - boundary);
      if (diff < bestDiff) {
        console.log("fpinmd")
        best = op;
        bestDiff = diff;
      }
    }

    if (best) {
      picked.push(best);
      used.add(best);
    }
  }

  return picked.sort((a, b) => b.Strike - a.Strike);
});

</script>
<style>

</style>
