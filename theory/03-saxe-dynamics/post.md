# Widgets test

<style>
.sw-controls, .mw-controls { display: flex; flex-wrap: wrap; gap: .4rem 1.2rem; align-items: center; margin-bottom: .6rem; }
.sw-row, .mw-row { display: flex; flex-wrap: wrap; gap: 18px; align-items: flex-start; }
.sw-hint { color: #6b6b70; font-size: .8rem; margin-top: .3rem; }
.sw-table table { font-size: .8rem; margin: .5rem 0 0; }
.sw-table td, .sw-table th { padding: .15rem .45rem; }
.sw-sw, .mw-key { display: inline-block; width: 14px; height: 4px; border-radius: 2px; vertical-align: middle; }
.widget button { font: inherit; font-size: .82rem; padding: .25rem .7rem; border: 1px solid #d8d3ca; border-radius: 6px; background: #fcfbf8; cursor: pointer; }
.widget button:hover { background: #f3f0ea; }
.widget input[type=range] { width: 110px; accent-color: #b5452b; }
.widget select { font: inherit; font-size: .82rem; }
.mw-group { display: flex; flex-wrap: wrap; gap: .3rem 1rem; }
.mw-matwrap { display: flex; gap: 12px; align-items: center; margin-top: 12px; }
.mw-matcap { font-size: .78rem; color: #6b6b70; max-width: 260px; line-height: 1.35; }
.mw-note { font-size: .78rem; color: #6b6b70; margin-top: .4rem; }
</style>

<div class="widget wide" id="saddle-widget"></div>
<script src="widgets/common.js"></script>
<script src="widgets/saddle.js"></script>

<div class="widget wide" id="modes-widget"></div>
<script src="widgets/modes.js"></script>
