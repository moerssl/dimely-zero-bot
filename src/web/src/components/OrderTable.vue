<template>
  <div class="orders-table-container">
    <table class="orders-table">
      <thead>
        <tr>
          <th v-for="col in columns" :key="col">{{ col }}</th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="(orderItem, rowIndex) in orders" :key="rowIndex">
          <td v-for="col in columns" :key="col">
            <template v-if="col == 'time'">
                {{ formatDate(orderItem[col]) }}
            </template>
            <template v-else>
              <span v-if="!isObject(orderItem[col])">{{ orderItem[col] }}</span>
              <span v-else>{{ stringify(orderItem[col]) }}</span>
            </template>

          </td>
        </tr>
      </tbody>
    </table>
  </div>
</template>

<script setup>
import { computed } from 'vue';
import { toRaw } from 'vue';


const props = defineProps(['orders']);

// Compute table columns based on first order object keys
const columns = computed(() => {
  if (!props.orders || props.orders.length === 0) return [];
  return Object.keys(props.orders[0]);
});

function formatDate(val) {
  const now = new Date(val)

  // Get date string in YYYY-MM-DD format
  const dateStr = now.toISOString().split('T')[0];

  // Get time string in HH:MM:SS format
  const timeStr = now.toTimeString().split(' ')[0];
  return timeStr
}

// Helper to check for objects
function isObject(val) {
  return val !== null && typeof val === 'object';
}

// Stringify objects for display
function stringify(val) {
  try {
    return JSON.stringify(toRaw(val));
  } catch (e) {
    return String(val);
  }
}
</script>

<style scoped>
.orders-table-container {
  overflow-x: auto;
  margin: 1rem 0;
}

.orders-table {
  width: 100%;
  border-collapse: collapse;
}

.orders-table th,
.orders-table td {
  padding: 0.5rem;
  border: 1px solid #ccc;
  text-align: left;
  vertical-align: top;
  font-size: 0.875rem;
}

.orders-table th {

  font-weight: bold;
}
</style>
