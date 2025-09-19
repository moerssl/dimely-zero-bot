<template>
  <!-- Using v-chart (from vue-echarts) with a fixed overall height to accommodate multiple grids -->
  <button @click="toggleStrategyCol">Toggle Strategy Column</button>
  Current strategy column: {{ strategyCol }}
  <div>
    <p v-for="(item, index) in strategyColorMap" :key="index">
      {{ index }}: <span :style="'background-color: '+item+'; width: 20px; height: 1rem; display: inline-block'"></span>

    </p>
  </div>
  <div style="display: flex">

    <v-chart
    :option="chartOption"
    @datazoom="onZoomChanged"
    style="width: 100vw; height: 100vh;"
    theme="shine" />
  </div>


</template>

<script setup>
import { computed, ref } from 'vue'
import VChart from 'vue-echarts'
import { use } from 'echarts/core'
import { CandlestickChart, LineChart, BarChart, ScatterChart } from 'echarts/charts'
import { CanvasRenderer } from 'echarts/renderers'

import { GridComponent, MarkAreaComponent, MarkLineComponent, VisualMapComponent, VisualMapPiecewiseComponent } from 'echarts/components';
use([GridComponent, MarkAreaComponent, MarkLineComponent, VisualMapComponent, VisualMapPiecewiseComponent]);

import { TooltipComponent } from 'echarts/components';
use([TooltipComponent]);

import { LegendComponent,DataZoomComponent } from 'echarts/components';
use([LegendComponent]);


use([DataZoomComponent])
// Register the required ECharts components
use([CandlestickChart, LineChart, BarChart, CanvasRenderer, ScatterChart])



// The prop "chartData" is expected to be an array of objects with the following keys:
// date, open, high, low, close, ema8, ema13, rsi, MACD, MACDHist, strategy.
const props = defineProps({
  chartData: {
    type: Array,
    required: true
  }
})

const sliderStart = ref(0)
const sliderEnd = ref(100)
const onZoomChanged = (zoomEvent) => {
  sliderStart.value = zoomEvent.start
  sliderEnd.value = zoomEvent.end

}

const toggleStrategyCol = () => {
  strategyCol.value = strategyCol.value === 'final_signal' ? 'tech_signal' : 'final_signal'
}

// Quick check: if chartData is empty, we'll bypass any computation.
const hasData = computed(() => props.chartData.length > 0)

// For the main x-axis we use the index of each candle:
const xAxisData = computed(() => props.chartData.map((o, i) => i))

// Prepare the candlestick data (ECharts expects an array of [open, close, low, high])
const candlestickData = computed(() =>
  props.chartData.map(d => [d.open, d.close, d.low, d.high])
)

// Prepare indicator arrays:
const ema8Data = computed(() => props.chartData.map(d => d.EMA8))
const ema13Data = computed(() => props.chartData.map(d => d.EMA13))
const rsiData = computed(() => props.chartData.map(d => d.RSI))
const macdData = computed(() => props.chartData.map(d => d.MACD))
const macdSignalData = computed(() => props.chartData.map(d => d.MACDSignal))
const macdHistData = computed(() => props.chartData.map(d => d.MACDHist))
const adxData = computed(() => props.chartData.map(d => d.adx))
const atrPercentData = computed(() => props.chartData.map(d => d.ATR_percent))

// For the candlestick's y-axis, compute the minimum and maximum price
const yMin = computed(() => Math.min(...props.chartData.map(d => d.low)))
const yMax = computed(() => Math.max(...props.chartData.map(d => d.high)))

// For the MACD panel, compute its overall min and max from the MACD and MACDHist values
const macdAllValues = computed(() =>
  props.chartData.reduce((acc, d) => {
    acc.push( d.MACDHist)
    return acc
  }, [])
)
const macdMin = computed(() => hasData.value ? Math.min(...macdAllValues.value) - 1 : 0)
const macdMax = computed(() => hasData.value ? Math.max(...macdAllValues.value) + 1 : 0)

// create dots for SellCall Strategy 1% below the close price
const sellCallData = computed(() => {
  let prevVal = null
  return props.chartData.map((d, i) => {
    if (d[strategyCol.value].name === 'SellCall') {
      prevVal = d.close + 2
      return [i, prevVal] // 1% below the close price
    } else if (d[strategyCol.value].name === 'SellPut') {
      prevVal = null
      return [i, prevVal] // 1% above the close price
    }
    return prevVal !== null ? [i, prevVal] : null
  }).filter(d => d !== null)
})
// create dots for SellPut Strategy 1% above the close price
const sellPutData = computed(() => {
  let prevVal = null
  return props.chartData.map((d, i) => {
    if (d[strategyCol.value].name === 'SellPut') {
      prevVal = d.close - 2
      return [i, prevVal] // 1% above the close price
    } else if (d[strategyCol.value].name === 'SellCall') {
      prevVal = null
      return [i, prevVal] // 1% below the close price
    }
    return prevVal !== null ? [i, prevVal] : null
  }).filter(d => d !== null)
})

// ----------------------------------------------------------------------------
// For background highlighting: use splitArea on the main xAxis.
// Determine unique strategies and map each one to a predefined semi-transparent color.
const uniqueStrategies = computed(() =>
  Array.from(new Set(props.chartData.map(d => d.tech_signal.name)))
)
const uniqueSentiments = computed(() =>
  Array.from(new Set(props.chartData.map(d => d.sentiment)))
)
const sentimentColorMap = computed(() => {
  return {
  "_bullish_exhaustion": "rgba(255, 206, 86, 0.2)", // Yellowish
  "_bearish_exhaustion": "rgba(153, 102, 255, 0.2)", // Purple
  "bearish": "red", // Red
  "bullish": "green" // Green

  }
})


const strategyColorMap = computed(() => {
  return {
    "SellCall": "#00ff00aa", // Green

    "SellPut": "#ff0000aa"
  }

})
const splitAreaColors = computed(() => {
  if (props.chartData?.length === 0) return null

  return props.chartData?.map(d => {
    return {
      "areaStyle": {
        "color": sentimentColorMap.value[d.sentiment] || 'transparent'
      }
    }

  })
})
const splitAreaColors2 = computed(() =>
  props.chartData.map(d => d.sentiment +": "+sentimentColorMap.value[d.sentiment] || 'transparent')
)

const strategyCol = ref('final_signal') // Default to 'tech_signal', can toggle to 'strategy'
// Compute background highlighting using markArea
const markAreas = computed(() => {
  const areas = []
  const col = strategyCol.value
  props.chartData.forEach((d, idx) => {
    const signal = d[col].name
    areas.push([

      { xAxis: idx , itemStyle: { color: strategyColorMap.value[signal] || 'transparent' } },
      { xAxis: idx-1 , itemStyle: { color: strategyColorMap.value[signal] || 'transparent' } },

    ])

  })
  return {
    silent: false,
    itemStyle: {
      opacity: 0.5
    },
    data: areas
  }
})
// ----------------------------------------------------------------------------
// Build the Complete ECharts Option:
// Three grids are defined:
//   - Grid 0: Main panel for Candlestick and EMA overlays.
//   - Grid 1: RSI indicator panel.
//   - Grid 2: MACD indicator panel (with MACD and MACDHist).
const chartOption = computed(() => {
  if (!hasData.value) return {}

  return {
    // Define three distinct grid areas:
    grid: [
      {
        // Main chart area for candlestick and EMA lines.
        left: '10%',
        right: '10%',
        top: '0%',
        height: '50%'
      },
      {
        // Second grid for the RSI indicator.
        left: '10%',
        right: '10%',
        top: '52%',
        height: '10%'
      },
      {
        // Third grid for the MACD indicator.
        left: '10%',
        right: '10%',
        top: '65%',
        height: '10%'
      },
      {
        // Fourth grid for the strategy column.
        left: '10%',
        right: '10%',
        top: '77%',
        height: '10%'
      },
      {
        // Fourth grid for the ADX indicator.
        left: '10%',
        right: '10%',
        top: '89%',
        height: '10%'
      }
    ],
        dataZoom: [
      {
        type: 'inside',
        start: sliderStart.value,
        end: sliderEnd.value,
        xAxisIndex: 'all',
      },
      {
        type: 'slider',
        start: sliderStart.value,
        end: sliderEnd.value,
        xAxisIndex: 'all',
        orient: 'vertical',
        left: "10px"
      }
    ],
    // Define an xAxis for each grid.
    xAxis: [
      {
        type: 'category',
        gridIndex: 0,
        data: xAxisData.value,

        axisLabel: {
          formatter: (value, index) => props.chartData[index]?.date || ''
        }
      },
      {
        type: 'category',
        gridIndex: 1,
        data: xAxisData.value,
        axisLabel: { show: false },
        boundaryGap: false
      },
      {
        type: 'category',
        gridIndex: 2,
        data: xAxisData.value,
        axisLabel: { show: false },
        boundaryGap: false
      },
      {
        type: 'category',
        gridIndex: 3,
        data: xAxisData.value,
        axisLabel: { show: false },
        boundaryGap: false
      },
      {
        type: 'category',
        gridIndex: 4,
        data: xAxisData.value,
        axisLabel: { show: false },
        boundaryGap: false
      },
    ],
    // Define corresponding yAxis for each grid.
    yAxis: [
      {
        // yAxis for the main chart (price)
        gridIndex: 0,
        scale: true,

        maxInterval: 1,
      },
      {
        // yAxis for RSI (conventionally 0 to 100)
        gridIndex: 1,
        min: 0,
        max: 100,
        name: 'RSI',
        nameLocation: 'middle',
        nameGap: 30,
      },
      {
        // yAxis for MACD indicators
        gridIndex: 2,
        min: macdMin.value,
        max: macdMax.value,
        name: 'MACD',
        nameLocation: 'middle',
        nameGap: 30,
      },
      {

        gridIndex: 3,
        min: 0,
        max: 100,
        name: 'ADX',
        nameLocation: 'middle',
        nameGap: 30,

      },
      {
        // yAxis for RSI (conventionally 0 to 100)
        gridIndex: 4,
        min: 0,
        name: 'ATR %',
        nameLocation: 'middle',
        nameGap: 40,

      },
    ],
    tooltip: {
      trigger: 'axis'
    },
    legend: {
      data: ['Candlestick', 'EMA8', 'EMA13', 'RSI', 'MACD', 'MACDHist'],
      top: '0%'
    },
    series: [
      // ----- Grid 0 (Main Chart: Candlestick + EMA) -----
      {
        type: 'candlestick',
        name: 'Candlestick',
        markArea: markAreas.value,
        data: candlestickData.value,
        xAxisIndex: 0,
        yAxisIndex: 0,
        itemStyle: {
          color: '#06B800',    // rising candle color
          color0: '#FA0000',   // falling candle color
          borderColor: '#06B800',
          borderColor0: '#FA0000'
        }
      },
      {
        type: 'line',
        name: 'EMA8',
        data: ema8Data.value,
        xAxisIndex: 0,
        yAxisIndex: 0,
        smooth: true,
        showSymbol: false,
        itemStyle: {
          width: 1.5,
          color: 'blue'
        }
      },
      {
        type: 'line',
        name: 'EMA13',
        data: ema13Data.value,
        xAxisIndex: 0,
        yAxisIndex: 0,
        smooth: true,
        showSymbol: false,
        itemStyle: {
          width: 1.5,
          color: 'orange'
        }
      },
      // ----- Grid 1 (RSI Indicator) -----
      {
        type: 'line',
        name: 'RSI',
        data: rsiData.value,
        xAxisIndex: 1,
        yAxisIndex: 1,
        smooth: true,
        showSymbol: false,
        lineStyle_: {
          width: 1.5,
          color: 'white'
        }
      },
      // ----- Grid 2 (MACD Indicator) -----
      {
        type: 'line',
        name: 'MACD',
        data: macdData.value,
        xAxisIndex: 2,
        yAxisIndex: 2,
        smooth: true,
        showSymbol: false,
        lineStyle: {
          width: 1.5,
          color: 'green'
        }
      },
      {
        type: 'bar',
        name: 'MACDHist',
        data: macdHistData.value,
        xAxisIndex: 2,
        yAxisIndex: 2,
        itemStyle: {
          color: 'gray'
        },
        barWidth: '60%'
      },
      {
        type: 'line',
        name: 'adx',
        data: adxData.value,
        xAxisIndex: 3,
        yAxisIndex: 3,
        smooth: true,
        showSymbol: false,


      },
      {
        type: 'scatter',
        name: 'SellCall',
        data: sellCallData.value,
        xAxisIndex: 0,
        yAxisIndex: 0,
        itemStyle: {
          color: 'green'
        },
        symbolSize: 5
      },
      {
        type: 'scatter',
        name: 'SellPut',
        data: sellPutData.value,
        xAxisIndex: 0,
        yAxisIndex: 0,
        itemStyle: {
          color: 'red'
        },
        symbolSize: 5
      },
      {
        type: 'line',
        name: 'MACDSignal',
        data: macdSignalData.value,
        xAxisIndex: 2,
        yAxisIndex: 2,
        itemStyle: {
          color: '#ffffffaa'
        }
      },
      {
        type: 'line',
        name: 'ATR_percent',
        data: atrPercentData.value,
        xAxisIndex: 4,
        yAxisIndex: 4,
        itemStyle: {
          color: '#ffffffaa'
        }
      },
    ],
    visualMap: [
      {
        seriesIndex: 6,
        pieces: [
          {
            gt: 0,
            lte: 20,
            color: 'white'
          },
          {
            gt: 20,
            lte: 25,
            color: 'orange'
          },
          {
            gt: 25,
            lte: 30,
            color: 'green'
          },
          {
            gt: 30,
            color: 'red'
          }

        ],
        outOfRange: {
          color: '#999'
        }
      },
      {
        seriesIndex: 3,
        pieces: [
          {
            gt: 0,
            lte: 40,
            color: 'red'
          },
          {
            gt: 40,
            lte: 65,
            color: 'white'
          },
          {
            gt: 65,
            lte: 100,
            color: 'green'
          },

        ],
        outOfRange: {
          color: '#999'
        }
      },
      {
        seriesIndex: 5,
        pieces: [
          {
            gt: -100,
            lte: 0,
            color: 'red'
          },
          {
            gt: 0,
            lte: 100,
            color: 'green'
          },

        ],
        outOfRange: {
          color: '#999'
        }
      },
      {
        seriesIndex: 10,
        pieces: [
          {
            gt: -100,
            lte: 0.125,
            color: 'green'
          },
          {
            gt: 0.125,
            lte: 100,
            color: 'red'
          },

        ],
        outOfRange: {
          color: '#999'
        }
      },

    ]
  }
})
</script>
