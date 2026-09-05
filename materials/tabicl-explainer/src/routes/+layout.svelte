<script lang="ts">
	import '~/styles/app.css';
	import '~/styles/global.scss';
</script>

<div id="app">
	<slot />
</div>

<style>
	#app {
		height: 100vh;
		width: 100%;
		overflow: hidden;
	}
</style>
