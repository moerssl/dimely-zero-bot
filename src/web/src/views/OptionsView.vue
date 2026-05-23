<template>
  <div class="w-100">
    <button @click="orderStore.fetchOptions">Reload</button>
    <p>Filter</p>
    <v-data-table :items="selectedOptionsRows"></v-data-table>
    <hr />
    <p>All</p>
    <!-- OrderTable :orders="orderStore.options"></OrderTable -->
    <v-data-table :items="orderStore.options"></v-data-table>
  </div>
</template>
<script setup>
import OrderTable from '@/components/OrderTable.vue';
import { useOrderStore } from '@/stores/orderStore';
import { onMounted, computed } from 'vue';


const orderStore = useOrderStore()

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


// ─── helpers ────────────────────────────────────────────────────────────────

function parseExpiry(raw) {
  if (!raw) return null
  const s = String(raw).trim()
  // Strip time suffix: "20250321 15:00:00" or "20250321-15:00:00"
  const dateStr = s.length > 8 && /^\d{8}/.test(s) ? s.slice(0, 8) : s
  // YYYYMM → last day of month
  if (/^\d{6}$/.test(dateStr)) {
    const d = new Date(dateStr.slice(0, 4), parseInt(dateStr.slice(4, 6)), 0)
    return isNaN(d) ? null : d
  }
  // YYYYMMDD
  if (/^\d{8}$/.test(dateStr)) {
    const d = new Date(dateStr.slice(0, 4), parseInt(dateStr.slice(4, 6)) - 1, dateStr.slice(6, 8))
    return isNaN(d) ? null : d
  }
  return null
}

function dte(expiryDate) {
  if (!expiryDate) return null
  const today = new Date()
  today.setHours(0, 0, 0, 0)
  return Math.floor((expiryDate - today) / 86_400_000)
}

// ─── core selection logic (mirrors Python) ───────────────────────────────────

function pickOption(options, { targetDelta, minPrice, type, targetUnderlyingPrice = null }) {
  let rows = options
    .filter(o => o.Type === type)
    .map(o => ({
      ...o,
      _delta:  parseFloat(o.delta),
      _ask:    parseFloat(o.ask),
      _strike: parseFloat(o.Strike),
      _expiry: parseExpiry(o.Expiry),
    }))
    .filter(o =>
      !isNaN(o._delta) &&
      !isNaN(o._ask)   &&
      o._expiry        !== null  &&
      Math.abs(o._delta) <= Math.abs(targetDelta) &&
      o._ask             >= minPrice
    )

  // Optional strike constraint
  if (targetUnderlyingPrice !== null) {
    if (type === 'C') rows = rows.filter(o => o._strike >= targetUnderlyingPrice)
    if (type === 'P') rows = rows.filter(o => o._strike <= targetUnderlyingPrice)
  }

  // Keep only non-expired
  rows = rows.filter(o => dte(o._expiry) >= 0)

  if (!rows.length) return null

  // Sort: DTE asc, ask desc  — mirrors Python
  rows.sort((a, b) => {
    const dteDiff = dte(a._expiry) - dte(b._expiry)
    return dteDiff !== 0 ? dteDiff : b._ask - a._ask
  })

  const winner = rows[0]
  return { ...winner, _dte: dte(winner._expiry) }
}

function getValidOptions(options, { targetDelta, minPrice, type, targetUnderlyingPrice = null }) {
  let rows = options
    .filter(o => o.Type === type)
    .map(o => ({
      ...o,
      _delta:  parseFloat(o.delta),
      _ask:    parseFloat(o.ask),
      _strike: parseFloat(o.Strike),
      _expiry: parseExpiry(o.Expiry),
    }))
    .filter(o =>
      !isNaN(o._delta) &&
      !isNaN(o._ask)   &&
      o._expiry        !== null  &&
      Math.abs(o._delta) <= Math.abs(targetDelta) &&
      o._ask             >= minPrice
    )

  if (targetUnderlyingPrice !== null) {
    if (type === 'C') rows = rows.filter(o => o._strike >= targetUnderlyingPrice)
    if (type === 'P') rows = rows.filter(o => o._strike <= targetUnderlyingPrice)
  }

  rows = rows.filter(o => dte(o._expiry) >= 0)

  // Same sort as Python: DTE asc, ask desc
  rows.sort((a, b) => {
    const dteDiff = dte(a._expiry) - dte(b._expiry)
    return dteDiff !== 0 ? dteDiff : b._ask - a._ask
  })

  return rows.map((o, index) => ({
    symbol: o.Symbol,
    type:   type === 'C' ? 'Call' : 'Put',
    strike: o._strike,
    expiry: o.Expiry,
    dte:    dte(o._expiry),
    delta:  o._delta.toFixed(2),
    ask:    o._ask.toFixed(2),
    conId:  o.ConId,
    rank:   index + 1,          // 1 = what .iloc[0] would have picked
    urgent: dte(o._expiry) <= 3,
  }))
}



// ─── computed table rows ──────────────────────────────────────────────────────

const TARGET_DELTA = 0.30   // ← your params here
const MIN_PRICE    = 0.20

const selectedOptionsRows = computed(() => {
  const allOptions = orderStore.options
  const symbols = [...new Set(allOptions.map(o => o.Symbol))]
  const rows = []

  for (const symbol of symbols) {
    const symbolOptions = allOptions.filter(o => o.Symbol === symbol)
    for (const type of ['C', 'P']) {
      rows.push(...getValidOptions(symbolOptions, {
        targetDelta: TARGET_DELTA,
        minPrice:    MIN_PRICE,
        type,
      }))
    }
  }

  return rows
})

</script>
<style>

</style>
