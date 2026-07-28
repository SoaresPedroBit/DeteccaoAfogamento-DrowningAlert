/* Dashboard — Detecção de Afogamentos
   SPA única (D-08, opção B): três telas, Chart.js, SSE, sem build. */

"use strict";

/* ================================================================ tema */
function cor(nome) {
  return getComputedStyle(document.documentElement).getPropertyValue(nome).trim();
}

function aplicarTemaCharts() {
  Chart.defaults.font.family = 'system-ui, -apple-system, "Segoe UI", sans-serif';
  Chart.defaults.font.size = 11;
  Chart.defaults.color = cor("--text-muted");
  Chart.defaults.borderColor = cor("--grid");
  Chart.defaults.plugins.legend.labels.boxWidth = 10;
  Chart.defaults.plugins.legend.labels.boxHeight = 10;
  Chart.defaults.plugins.legend.labels.usePointStyle = true;
  Chart.defaults.plugins.legend.labels.color = cor("--text-secondary");
  Chart.defaults.plugins.tooltip.backgroundColor = cor("--text-primary");
  Chart.defaults.plugins.tooltip.titleColor = cor("--page");
  Chart.defaults.plugins.tooltip.bodyColor = cor("--page");
  Chart.defaults.animation = false;
}

matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
  aplicarTemaCharts();
  if (ultimasMetricas) desenharGraficos(ultimasMetricas);
});

/* ============================================== plugins Chart.js locais */
// Linha vertical da média (G1) — valor em unidade do eixo x linear
const pluginLinhaMedia = {
  id: "linhaMedia",
  afterDatasetsDraw(chart, _args, opts) {
    if (opts.valor == null) return;
    const x = chart.scales.x.getPixelForValue(opts.valor);
    const { top, bottom } = chart.chartArea;
    const ctx = chart.ctx;
    ctx.save();
    ctx.strokeStyle = cor("--text-secondary");
    ctx.setLineDash([4, 3]);
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(x, top);
    ctx.lineTo(x, bottom);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = cor("--text-secondary");
    ctx.font = "600 10.5px system-ui";
    ctx.textAlign = x > chart.chartArea.right - 90 ? "right" : "left";
    ctx.fillText(` média ${opts.texto} `, x + (ctx.textAlign === "left" ? 4 : -4), top + 10);
    ctx.restore();
  },
};

// Haste de desvio padrão sobre barras (G4)
const pluginDesvio = {
  id: "desvio",
  afterDatasetsDraw(chart) {
    const ds = chart.data.datasets[0];
    if (!ds || !ds.desvios) return;
    const meta = chart.getDatasetMeta(0);
    const ctx = chart.ctx;
    ctx.save();
    ctx.strokeStyle = cor("--text-secondary");
    ctx.lineWidth = 1.5;
    meta.data.forEach((barra, i) => {
      const desvio = ds.desvios[i];
      if (!desvio) return;
      const yScale = chart.scales.y;
      const yTopo = yScale.getPixelForValue(ds.data[i] + desvio);
      const yBase = yScale.getPixelForValue(Math.max(0, ds.data[i] - desvio));
      const x = barra.x;
      ctx.beginPath();
      ctx.moveTo(x, yTopo); ctx.lineTo(x, yBase);
      ctx.moveTo(x - 5, yTopo); ctx.lineTo(x + 5, yTopo);
      ctx.moveTo(x - 5, yBase); ctx.lineTo(x + 5, yBase);
      ctx.stroke();
    });
    ctx.restore();
  },
};

Chart.register(pluginLinhaMedia, pluginDesvio);

/* =============================================================== estado */
const charts = {};            // id -> instância Chart
const dadosBrutos = {};       // id -> linhas p/ exportação CSV
let ultimasMetricas = null;
let filtroDias = 7;
let paginaAtual = 1;
let totalPaginas = 1;
let alarmeAtivo = null;

/* ============================================================ navegação */
document.querySelectorAll("nav button").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll("nav button").forEach((b) => b.classList.remove("ativa"));
    document.querySelectorAll(".tela").forEach((t) => t.classList.remove("ativa"));
    btn.classList.add("ativa");
    document.getElementById("tela-" + btn.dataset.tela).classList.add("ativa");
    if (btn.dataset.tela === "analise") carregarMetricas();
    if (btn.dataset.tela === "historico") carregarHistorico(1);
  });
});

/* =============================================================== status */
function fmtUptime(s) {
  if (!s) return "—";
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60);
  return h ? `${h}h${String(m).padStart(2, "0")}` : `${m}min`;
}

async function atualizarStatus() {
  try {
    const r = await fetch("/api/status");
    const s = await r.json();
    document.getElementById("led-camera").classList.toggle("on", s.camera_conectada);
    document.getElementById("st-camera").textContent =
      s.camera_conectada ? `${s.camera_id} conectada` : "câmera desconectada";
    document.getElementById("st-fps").textContent =
      s.camera_conectada ? Number(s.fps).toFixed(1) : "—";
    document.getElementById("st-modelo").textContent = s.modelo;
    document.getElementById("st-inferencia").textContent =
      s.camera_conectada && s.inferencia_ms_media ? `${Number(s.inferencia_ms_media).toFixed(1)} ms` : "—";
    document.getElementById("st-uptime").textContent = fmtUptime(s.uptime_s);
    document.getElementById("stream-offline").style.display = s.camera_conectada ? "none" : "flex";
  } catch {
    document.getElementById("st-camera").textContent = "API indisponível";
    document.getElementById("led-camera").classList.remove("on");
  }
}

/* ============================================================ SSE (D-07) */
function conectarSSE() {
  const es = new EventSource("/api/eventos/stream");
  es.addEventListener("evento", (msg) => {
    const ev = JSON.parse(msg.data);
    inserirMiniEvento(ev, true);
    if (ev.classe === "afogamento") dispararAlerta(ev);
    if (document.getElementById("tela-historico").classList.contains("ativa")) {
      carregarHistorico(paginaAtual);
    }
  });
  es.onerror = () => { /* EventSource reconecta sozinho */ };
}

function urlSnapshot(ev) {
  if (!ev.caminho_snapshot || ev.caminho_snapshot === "expirado") return null;
  return "/" + ev.caminho_snapshot.replace(/\\/g, "/");
}

function inserirMiniEvento(ev, noTopo) {
  const lista = document.getElementById("ultimos-eventos");
  const vazio = lista.querySelector(".vazio");
  if (vazio) vazio.remove();
  const el = document.createElement("div");
  el.className = "mini-evento";
  const foto = urlSnapshot(ev);
  el.innerHTML = `
    ${foto ? `<img src="${foto}" alt="">` : `<img alt="" style="visibility:hidden">`}
    <div class="info">
      <b>${ev.classe === "afogamento" ? "🚨 " : ""}${ev.classe}</b>
      ${new Date(ev.timestamp_alerta).toLocaleString("pt-BR")}<br>
      conf. ${(ev.confianca_media * 100).toFixed(0)}% · ${ev.modelo}
    </div>`;
  el.addEventListener("click", () => abrirPainel(ev.id));
  noTopo ? lista.prepend(el) : lista.append(el);
  while (lista.children.length > 5) lista.lastChild.remove();
}

async function carregarUltimosEventos() {
  const r = await fetch("/api/eventos?por_pagina=5");
  const dados = await r.json();
  const lista = document.getElementById("ultimos-eventos");
  lista.innerHTML = dados.eventos.length ? "" : '<div class="vazio">Nenhum evento registrado.</div>';
  dados.eventos.forEach((ev) => inserirMiniEvento(ev, false));
}

/* ========================================================= alerta sonoro */
function dispararAlerta(ev) {
  const banner = document.getElementById("banner-alerta");
  document.getElementById("banner-texto").textContent =
    `POSSÍVEL AFOGAMENTO — ${new Date(ev.timestamp_alerta).toLocaleTimeString("pt-BR")}` +
    ` · confiança ${(ev.confianca_media * 100).toFixed(0)}%`;
  banner.classList.add("visivel");
  tocarSirene();
  clearTimeout(dispararAlerta._t);
  dispararAlerta._t = setTimeout(silenciarAlerta, 30000);
}

function tocarSirene() {
  try {
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const ganho = ctx.createGain();
    ganho.gain.value = 0.25;
    ganho.connect(ctx.destination);
    const osc = ctx.createOscillator();
    osc.type = "square";
    osc.connect(ganho);
    const t0 = ctx.currentTime;
    for (let i = 0; i < 6; i++) {
      osc.frequency.setValueAtTime(880, t0 + i * 0.5);
      osc.frequency.setValueAtTime(660, t0 + i * 0.5 + 0.25);
    }
    osc.start(t0);
    osc.stop(t0 + 3);
    alarmeAtivo = { ctx, osc };
    osc.onended = () => ctx.close();
  } catch { /* sem áudio disponível */ }
}

function silenciarAlerta() {
  document.getElementById("banner-alerta").classList.remove("visivel");
  if (alarmeAtivo) { try { alarmeAtivo.osc.stop(); } catch {} alarmeAtivo = null; }
}

/* ======================================================= filtros análise */
document.querySelectorAll("#filtro-periodo button").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll("#filtro-periodo button").forEach((b) => b.classList.remove("ativa"));
    btn.classList.add("ativa");
    const custom = btn.dataset.dias === "custom";
    document.getElementById("periodo-custom").classList.toggle("visivel", custom);
    if (!custom) { filtroDias = Number(btn.dataset.dias); carregarMetricas(); }
    else filtroDias = "custom";
  });
});
document.getElementById("filtro-modelo").addEventListener("change", carregarMetricas);

function intervaloAtual() {
  if (filtroDias === "custom") {
    return { de: document.getElementById("filtro-de").value || null,
             ate: document.getElementById("filtro-ate").value || null };
  }
  const ate = new Date();
  const de = new Date(ate.getTime() - (filtroDias - 1) * 86400000);
  return { de: de.toISOString().slice(0, 10), ate: ate.toISOString().slice(0, 10) };
}

/* ============================================================== métricas */
async function carregarMetricas() {
  const { de, ate } = intervaloAtual();
  const modelo = document.getElementById("filtro-modelo").value;
  const qs = new URLSearchParams();
  if (de) qs.set("de", de);
  if (ate) qs.set("ate", ate);
  if (modelo) qs.set("modelo", modelo);
  const r = await fetch("/api/metricas?" + qs);
  ultimasMetricas = await r.json();
  desenharGraficos(ultimasMetricas);
}

function preencherTiles(m) {
  const r = m.resumo;
  document.getElementById("tile-total").textContent = r.total_eventos;
  document.getElementById("tile-pendentes").textContent =
    r.pendentes ? `${r.pendentes} pendente(s) de revisão` : "";
  document.getElementById("tile-confirmados").textContent = r.confirmados;
  document.getElementById("tile-fp").textContent = r.falsos_positivos;
  document.getElementById("tile-precisao").textContent =
    r.precisao == null ? "—" : (r.precisao * 100).toFixed(1) + "%";
  document.getElementById("tile-latencia").textContent =
    r.latencia_media_ms == null ? "—" : (r.latencia_media_ms / 1000).toFixed(2) + "s";
}

function novoChart(id, config) {
  if (charts[id]) charts[id].destroy();
  charts[id] = new Chart(document.getElementById("chart-" + id), config);
  return charts[id];
}

function marcarVazio(id, vazio) {
  const el = document.getElementById("vazio-" + id);
  if (el) el.hidden = !vazio;
}

function binar(valores, largura, minimo, maximo) {
  const bins = [];
  for (let x = minimo; x < maximo; x += largura) bins.push({ de: x, ate: x + largura, n: 0 });
  valores.forEach((v) => {
    const i = Math.min(bins.length - 1, Math.max(0, Math.floor((v - minimo) / largura)));
    bins[i].n++;
  });
  return bins;
}

const escalasBase = () => ({
  x: { grid: { display: false }, border: { color: cor("--axis") } },
  y: { beginAtZero: true, grid: { color: cor("--grid") }, border: { display: false },
       ticks: { precision: 0 } },
});

function desenharGraficos(m) {
  aplicarTemaCharts();
  preencherTiles(m);

  const azul = cor("--series-1"), laranja = cor("--series-2"), aqua = cor("--series-3");
  const surface = cor("--surface-1");

  // preenche o seletor de modelo sem perder a seleção
  const sel = document.getElementById("filtro-modelo");
  const selecionado = sel.value;
  const modelos = new Set(m.g4_inferencia_por_modelo.map((x) => x.modelo));
  m.g3_fps_por_periodo.forEach((x) => modelos.add(x.modelo));
  sel.innerHTML = '<option value="">todos</option>' +
    [...modelos].sort().map((x) => `<option ${x === selecionado ? "selected" : ""}>${x}</option>`).join("");

  /* ---- G1: histograma de latência + média (série única: sem legenda) */
  const lat = m.g1_latencias_ms.map((v) => v / 1000);
  marcarVazio("g1", !lat.length);
  if (lat.length) {
    const maxLat = Math.max(5, Math.ceil(Math.max(...lat)));
    const bins = binar(lat, 0.5, 0, maxLat);
    const media = lat.reduce((a, b) => a + b, 0) / lat.length;
    dadosBrutos.g1 = [["latencia_s"], ...lat.map((v) => [v.toFixed(3)])];
    novoChart("g1", {
      type: "bar",
      data: {
        labels: bins.map((b) => b.de.toFixed(1)),
        datasets: [{ data: bins.map((b) => b.n), backgroundColor: azul,
                     maxBarThickness: 24, borderRadius: 4, borderSkipped: "bottom",
                     categoryPercentage: 0.92, barPercentage: 1 }],
      },
      options: {
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          linhaMedia: null,
          tooltip: { callbacks: {
            title: (it) => `${bins[it[0].dataIndex].de.toFixed(1)}–${bins[it[0].dataIndex].ate.toFixed(1)} s`,
            label: (it) => `${it.parsed.y} evento(s)`,
          } },
        },
        scales: { ...escalasBase(),
          x: { ...escalasBase().x, title: { display: true, text: "latência (s)", color: cor("--text-muted") } } },
      },
    });
    // linha da média em coordenada de categoria (índice fracionário do bin)
    charts.g1.options.plugins.linhaMedia = { valor: media / 0.5 - 0.5, texto: media.toFixed(2) + "s" };
    charts.g1.update();
    document.getElementById("g1-sub").textContent =
      `${lat.length} evento(s) · média ${media.toFixed(2)}s`;
  } else if (charts.g1) { charts.g1.destroy(); delete charts.g1; }

  /* ---- G2: histograma de confiança, VP vs FP (2 séries: legenda presente) */
  const vp = m.g2_confianca_por_status.confirmado || [];
  const fp = m.g2_confianca_por_status.falso_positivo || [];
  marcarVazio("g2", !vp.length && !fp.length);
  if (vp.length || fp.length) {
    const binsVP = binar(vp, 0.05, 0.3, 1.0);
    const binsFP = binar(fp, 0.05, 0.3, 1.0);
    dadosBrutos.g2 = [["faixa_confianca", "ocorrencias_reais", "falsos_positivos"],
      ...binsVP.map((b, i) => [`${b.de.toFixed(2)}–${b.ate.toFixed(2)}`, b.n, binsFP[i].n])];
    novoChart("g2", {
      type: "bar",
      data: {
        labels: binsVP.map((b) => b.de.toFixed(2)),
        datasets: [
          { label: "Ocorrência real", data: binsVP.map((b) => b.n), backgroundColor: azul,
            maxBarThickness: 24, borderRadius: 4, borderSkipped: "bottom" },
          { label: "Falso positivo", data: binsFP.map((b) => b.n), backgroundColor: laranja,
            maxBarThickness: 24, borderRadius: 4, borderSkipped: "bottom" },
        ],
      },
      options: {
        maintainAspectRatio: false,
        plugins: { legend: { position: "top", align: "end" } },
        scales: { ...escalasBase(),
          x: { ...escalasBase().x, title: { display: true, text: "confiança média do evento", color: cor("--text-muted") } } },
      },
    });
  } else if (charts.g2) { charts.g2.destroy(); delete charts.g2; }

  /* ---- G3: FPS no tempo por modelo (linhas) */
  const porModelo = {};
  m.g3_fps_por_periodo.forEach((r) => {
    (porModelo[r.modelo] = porModelo[r.modelo] || {})[r.periodo] = r.fps;
  });
  const periodos = [...new Set(m.g3_fps_por_periodo.map((r) => r.periodo))].sort();
  const nomesModelos = Object.keys(porModelo).sort();
  marcarVazio("g3", !periodos.length);
  if (periodos.length) {
    const cores = [azul, laranja, aqua];
    dadosBrutos.g3 = [["periodo", ...nomesModelos],
      ...periodos.map((p) => [p, ...nomesModelos.map((mo) => porModelo[mo][p]?.toFixed(2) ?? "")])];
    novoChart("g3", {
      type: "line",
      data: {
        labels: periodos.map(fmtPeriodo),
        datasets: nomesModelos.map((mo, i) => ({
          label: mo,
          data: periodos.map((p) => porModelo[mo][p] ?? null),
          borderColor: cores[i % 3], backgroundColor: cores[i % 3],
          borderWidth: 2, tension: 0.25, spanGaps: true,
          pointRadius: periodos.length > 40 ? 0 : 4,
          pointBorderColor: surface, pointBorderWidth: 2,
        })),
      },
      options: {
        maintainAspectRatio: false,
        interaction: { mode: "index", intersect: false },
        plugins: { legend: { display: nomesModelos.length > 1, position: "top", align: "end" } },
        scales: { ...escalasBase(),
          y: { ...escalasBase().y, ticks: {}, title: { display: true, text: "FPS", color: cor("--text-muted") } },
          x: { ...escalasBase().x, ticks: { maxTicksLimit: 8, maxRotation: 0 } } },
      },
    });
  } else if (charts.g3) { charts.g3.destroy(); delete charts.g3; }

  /* ---- G4: tempo de inferência por modelo (barras + desvio padrão) */
  const g4 = m.g4_inferencia_por_modelo;
  marcarVazio("g4", !g4.length);
  if (g4.length) {
    dadosBrutos.g4 = [["modelo", "media_ms", "desvio_ms", "min_ms", "max_ms", "amostras"],
      ...g4.map((x) => [x.modelo, x.media, x.desvio, x.minimo, x.maximo, x.amostras])];
    novoChart("g4", {
      type: "bar",
      data: {
        labels: g4.map((x) => x.modelo),
        datasets: [{ data: g4.map((x) => x.media), desvios: g4.map((x) => x.desvio),
                     backgroundColor: azul, maxBarThickness: 24,
                     borderRadius: 4, borderSkipped: "bottom" }],
      },
      options: {
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: { callbacks: { label: (it) =>
            ` média ${it.parsed.y.toFixed(1)} ms · desvio ±${g4[it.dataIndex].desvio.toFixed(1)} ms` } },
        },
        scales: { ...escalasBase(),
          y: { ...escalasBase().y, ticks: {}, title: { display: true, text: "ms por frame", color: cor("--text-muted") } } },
      },
    });
  } else if (charts.g4) { charts.g4.destroy(); delete charts.g4; }

  /* ---- G5: eventos por hora do dia */
  const horas = [...Array(24).keys()];
  const contagens = horas.map((h) => m.g5_eventos_por_hora[h] || 0);
  const temG5 = contagens.some((c) => c > 0);
  marcarVazio("g5", !temG5);
  if (temG5) {
    dadosBrutos.g5 = [["hora", "eventos"], ...horas.map((h) => [h, contagens[h]])];
    novoChart("g5", {
      type: "bar",
      data: {
        labels: horas.map((h) => String(h).padStart(2, "0") + "h"),
        datasets: [{ data: contagens, backgroundColor: azul, maxBarThickness: 24,
                     borderRadius: 4, borderSkipped: "bottom",
                     categoryPercentage: 0.92, barPercentage: 1 }],
      },
      options: {
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: { ...escalasBase(),
          x: { ...escalasBase().x, ticks: { maxTicksLimit: 12, maxRotation: 0 } } },
      },
    });
  } else if (charts.g5) { charts.g5.destroy(); delete charts.g5; }

  /* ---- G6: precisão (%) e FP/hora em painéis separados (um eixo por painel) */
  const g6 = m.g6_precisao_por_dia.filter((d) => d.precisao != null || d.fp_por_hora != null);
  marcarVazio("g6", !g6.length);
  if (g6.length) {
    dadosBrutos.g6 = [["dia", "precisao", "fp_por_hora", "confirmados", "falsos_positivos"],
      ...g6.map((d) => [d.dia, d.precisao ?? "", d.fp_por_hora ?? "", d.confirmados, d.falsos_positivos])];
    const labels = g6.map((d) => d.dia.slice(5));
    const linhaBase = (dados, corSerie, rotulo) => ({
      type: "line",
      data: { labels, datasets: [{ label: rotulo, data: dados,
        borderColor: corSerie, backgroundColor: corSerie, borderWidth: 2,
        tension: 0.25, spanGaps: true, pointRadius: labels.length > 40 ? 0 : 4,
        pointBorderColor: surface, pointBorderWidth: 2 }] },
      options: {
        maintainAspectRatio: false,
        plugins: { legend: { display: false },
          title: { display: true, text: rotulo, align: "start",
                   color: cor("--text-secondary"), font: { size: 11, weight: "600" }, padding: { bottom: 2 } } },
        scales: { ...escalasBase(),
          x: { ...escalasBase().x, ticks: { maxTicksLimit: 8, maxRotation: 0 } },
          y: { ...escalasBase().y, ticks: {} } },
      },
    });
    const cfgA = linhaBase(g6.map((d) => d.precisao == null ? null : d.precisao * 100), azul, "Precisão (%)");
    cfgA.options.scales.y.max = 100;
    if (charts.g6a) charts.g6a.destroy();
    charts.g6a = new Chart(document.getElementById("chart-g6a"), cfgA);
    if (charts.g6b) charts.g6b.destroy();
    charts.g6b = new Chart(document.getElementById("chart-g6b"),
      linhaBase(g6.map((d) => d.fp_por_hora), laranja, "Falsos positivos por hora de operação"));
  } else {
    ["g6a", "g6b"].forEach((id) => { if (charts[id]) { charts[id].destroy(); delete charts[id]; } });
  }
}

function fmtPeriodo(p) {
  // "2026-07-22" | "2026-07-22T14" | "2026-07-22T14:35"
  if (p.length === 10) return p.slice(8, 10) + "/" + p.slice(5, 7);
  return p.slice(8, 10) + "/" + p.slice(5, 7) + " " + p.slice(11) + (p.length === 13 ? "h" : "");
}

/* ============================================================ exportação */
function exportarPNG(id) {
  const chart = charts[id] || charts[id + "a"];
  if (!chart) return toast("Sem dados para exportar");
  if (id === "g6") {
    // junta os dois painéis num único PNG
    const a = charts.g6a.canvas, b = charts.g6b.canvas;
    const c = document.createElement("canvas");
    c.width = Math.max(a.width, b.width);
    c.height = a.height + b.height;
    const ctx = c.getContext("2d");
    ctx.fillStyle = cor("--surface-1");
    ctx.fillRect(0, 0, c.width, c.height);
    ctx.drawImage(a, 0, 0);
    ctx.drawImage(b, 0, a.height);
    baixar(c.toDataURL("image/png"), "grafico_g6.png");
    return;
  }
  // fundo opaco para a figura ir direto à monografia
  const c = document.createElement("canvas");
  c.width = chart.canvas.width;
  c.height = chart.canvas.height;
  const ctx = c.getContext("2d");
  ctx.fillStyle = cor("--surface-1");
  ctx.fillRect(0, 0, c.width, c.height);
  ctx.drawImage(chart.canvas, 0, 0);
  baixar(c.toDataURL("image/png"), `grafico_${id}.png`);
}

function exportarCSV(id) {
  const linhas = dadosBrutos[id];
  if (!linhas) return toast("Sem dados para exportar");
  const csv = "﻿" + linhas.map((l) => l.join(";")).join("\r\n");
  baixar(URL.createObjectURL(new Blob([csv], { type: "text/csv" })), `dados_${id}.csv`);
}

function baixar(href, nome) {
  const a = document.createElement("a");
  a.href = href;
  a.download = nome;
  a.click();
}

/* ============================================================= histórico */
function filtrosHistorico() {
  const qs = new URLSearchParams();
  const de = document.getElementById("hist-de").value;
  const ate = document.getElementById("hist-ate").value;
  const classe = document.getElementById("hist-classe").value;
  const status = document.getElementById("hist-status").value;
  const modelo = document.getElementById("hist-modelo").value;
  if (de) qs.set("de", de);
  if (ate) qs.set("ate", ate);
  if (classe) qs.set("classe", classe);
  if (status) qs.set("status", status);
  if (modelo) qs.set("modelo", modelo);
  return qs;
}

const rotuloStatus = { pendente: "pendente", confirmado: "ocorrência real", falso_positivo: "falso positivo" };

async function carregarHistorico(pagina) {
  paginaAtual = pagina;
  const qs = filtrosHistorico();
  qs.set("pagina", pagina);
  qs.set("por_pagina", 20);
  const r = await fetch("/api/eventos?" + qs);
  const dados = await r.json();
  totalPaginas = Math.max(1, Math.ceil(dados.total / dados.por_pagina));

  // popula filtro de modelos do histórico a partir dos dados visíveis
  const selModelo = document.getElementById("hist-modelo");
  const atuais = new Set([...selModelo.options].map((o) => o.value));
  dados.eventos.forEach((ev) => {
    if (!atuais.has(ev.modelo)) {
      selModelo.add(new Option(ev.modelo, ev.modelo));
      atuais.add(ev.modelo);
    }
  });

  const corpo = document.getElementById("hist-corpo");
  if (!dados.eventos.length) {
    corpo.innerHTML = '<tr><td colspan="8" class="vazio">Nenhum evento com esses filtros.</td></tr>';
  } else {
    corpo.innerHTML = dados.eventos.map((ev) => {
      const foto = urlSnapshot(ev);
      return `<tr onclick="abrirPainel(${ev.id})">
        <td>${foto ? `<img class="thumb" loading="lazy" src="${foto}" alt="">`
                   : `<span style="font-size:11px;color:var(--text-muted)">${ev.caminho_snapshot === "expirado" ? "expirado" : "—"}</span>`}</td>
        <td class="num">${new Date(ev.timestamp_alerta).toLocaleString("pt-BR")}</td>
        <td><span class="badge ${ev.classe === "afogamento" ? "afogamento" : ""}">${ev.classe}</span></td>
        <td class="num">${(ev.confianca_media * 100).toFixed(1)}%</td>
        <td class="num">${(ev.latencia_ms / 1000).toFixed(2)}s</td>
        <td>${ev.modelo}</td>
        <td>${ev.camera_id}</td>
        <td><span class="badge ${ev.status}">${rotuloStatus[ev.status] || ev.status}</span></td>
      </tr>`;
    }).join("");
  }
  document.getElementById("hist-info").textContent =
    `${dados.total} evento(s) · página ${pagina} de ${totalPaginas}`;
  document.getElementById("hist-ant").disabled = pagina <= 1;
  document.getElementById("hist-prox").disabled = pagina >= totalPaginas;
}

function mudarPagina(delta) {
  const nova = paginaAtual + delta;
  if (nova >= 1 && nova <= totalPaginas) carregarHistorico(nova);
}

function exportarHistoricoCSV() {
  window.location = "/api/exportar?" + filtrosHistorico();
}

/* ======================================================== painel detalhe */
async function abrirPainel(id) {
  const r = await fetch("/api/eventos/" + id);
  if (!r.ok) return toast("Evento não encontrado");
  const ev = await r.json();
  document.getElementById("painel-titulo").textContent =
    `Evento #${ev.id} — ${ev.classe}`;
  const foto = urlSnapshot(ev);
  document.getElementById("painel-conteudo").innerHTML = `
    ${foto ? `<img class="snapshot" src="${foto}" alt="Snapshot do evento ${ev.id}">`
           : `<div class="snapshot-expirado">${ev.caminho_snapshot === "expirado"
                ? "Snapshot removido pela rotina de retenção (D-09). Metadados preservados."
                : "Sem snapshot disponível."}</div>`}
    <div class="metadados">
      <div><span>Início do evento</span><b>${new Date(ev.timestamp_inicio).toLocaleString("pt-BR")}</b></div>
      <div><span>Alerta emitido</span><b>${new Date(ev.timestamp_alerta).toLocaleString("pt-BR")}</b></div>
      <div><span>Latência</span><b>${(ev.latencia_ms / 1000).toFixed(2)} s</b></div>
      <div><span>Câmera</span><b>${ev.camera_id}</b></div>
      <div><span>Confiança média</span><b>${(ev.confianca_media * 100).toFixed(1)}%</b></div>
      <div><span>Confiança máxima</span><b>${(ev.confianca_maxima * 100).toFixed(1)}%</b></div>
      <div><span>Frames positivos</span><b>${ev.frames_positivos} / ${ev.frames_janela}</b></div>
      <div><span>Modelo</span><b>${ev.modelo}</b></div>
    </div>
    <div class="revisao">
      <h3>Revisão do operador (D-06)</h3>
      <textarea id="painel-obs" placeholder="Observação (opcional)">${ev.observacao || ""}</textarea>
      <div class="acoes-revisao">
        <button class="botao sucesso" onclick="classificar(${ev.id}, 'confirmado')">✓ Ocorrência real</button>
        <button class="botao perigo" onclick="classificar(${ev.id}, 'falso_positivo')">✗ Falso positivo</button>
        <button class="botao" onclick="classificar(${ev.id}, 'pendente')">Voltar a pendente</button>
      </div>
      <div class="aviso-revisado">${ev.revisado_em
        ? `Revisado em ${new Date(ev.revisado_em).toLocaleString("pt-BR")} — ${rotuloStatus[ev.status]}.`
        : "Ainda não revisado."}</div>
    </div>`;
  document.getElementById("painel-fundo").classList.add("visivel");
  document.getElementById("painel-detalhe").classList.add("visivel");
}

function fecharPainel() {
  document.getElementById("painel-fundo").classList.remove("visivel");
  document.getElementById("painel-detalhe").classList.remove("visivel");
}

async function classificar(id, status) {
  const observacao = document.getElementById("painel-obs").value.trim() || null;
  const r = await fetch(`/api/eventos/${id}/status`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ status, observacao }),
  });
  if (r.ok) {
    toast(status === "pendente" ? "Evento voltou a pendente" : `Evento marcado: ${rotuloStatus[status]}`);
    fecharPainel();
    if (document.getElementById("tela-historico").classList.contains("ativa")) carregarHistorico(paginaAtual);
    carregarUltimosEventos();
  } else {
    toast("Falha ao gravar a revisão");
  }
}

/* ================================================================ toast */
function toast(msg) {
  const el = document.getElementById("toast");
  el.textContent = msg;
  el.classList.add("visivel");
  clearTimeout(toast._t);
  toast._t = setTimeout(() => el.classList.remove("visivel"), 2600);
}

document.addEventListener("keydown", (e) => { if (e.key === "Escape") fecharPainel(); });

/* ================================================================ início */
aplicarTemaCharts();
atualizarStatus();
setInterval(atualizarStatus, 2000);
carregarUltimosEventos();
conectarSSE();
