<script setup lang="ts">
import { RouterLink, RouterView } from 'vue-router'
import HelloWorld from './components/HelloWorld.vue'
import { useOrderStore } from './stores/orderStore';
import  router  from './router'

const orderStore = useOrderStore()
const routes = router.getRoutes()
</script>

<template>
  <header>
    <!-- Generate menu from router-->
    <img alt="Vue logo" class="logo" src="@/assets/logo.svg" width="125" height="125" />
    <nav>
      <RouterLink :to="route.path" v-for="route in routes">{{ route.name }}</RouterLink>
    </nav>
    <div class="sub">
      <select v-model="orderStore.port">
        <option value="8051">IWM</option>
        <option value="8052">QQQ</option>
        <option value="8053">SPY</option>
        <option value="8054">SPX</option>
        <option value="8055">XSP</option>
        <option value="8050">Other</option>
      </select>
      <span :class='{"show": orderStore.loading }'>Lade&nbsp;Daten...</span>

    </div>

      <!-- pre>

        {{ routes }}
      </pre -->


  </header>
  <RouterView />
</template>

<style scoped>
header {
  line-height: 1.5;
}

.logo {
  display: block;
  margin: 0 auto 2rem;
}

nav {
  width: 100%;
  font-size: 12px;
  text-align: center;
  margin-top: 2rem;

  display: inline-block;

  text-transform: capitalize;
}

nav a.router-link-exact-active {
  color: var(--color-text);
}

nav a.router-link-exact-active:hover {
  background-color: transparent;
}

nav a {
  display: inline-block;
  padding: 0 1rem;
  border-left: 1px solid var(--color-border);
}

nav a:first-of-type {
  border: 0;
}



.sub span {
  background: red;
  text-align: right;
  visibility: hidden;

}

.sub span.show{
  visibility: visible;
}

@media (min-width: 1024px) {
  header {
    display: flex;
    place-items: center;
    padding-right: calc(var(--section-gap) / 2);
  }

  .logo {
    margin: 0 2rem 0 0;
  }

  header .wrapper {
    display: flex;
    place-items: flex-start;
    flex-wrap: wrap;
  }

  nav {
    text-align: left;
    margin-left: -1rem;
    font-size: 1rem;

    padding: 1rem 0;
    margin-top: 1rem;
  }
}
</style>
