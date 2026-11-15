/**
 * Driver Behaviour Telemetry & Safety Monitoring Console (app.js)
 * Clean, utilitarian client implementation. No AI slop.
 * Sequential HMM Telemetry, Donut Behaviour Distribution & Replay Timeline.
 */

// ---------------------------------------------------------------------------
// Global Application State
// ---------------------------------------------------------------------------
const state = {
  allTrips: [],
  selectedTripId: null,
  telemetry: null,
  charts: {
    ax: null,
    ay: null,
    jerk: null,
    speed: null,
    donut: null,
  },
  replay: {
    isPlaying: false,
    timer: null,
    currentTime: 0,
    speedMultiplier: 2,
    fps: 25,
  }
};

const STATE_CLASSES = {
  NORMAL: 'state-normal',
  DROWSY: 'state-drowsy',
  AGGRESSIVE: 'state-aggressive',
  UNKNOWN: 'state-neutral'
};

const STATE_COLORS = {
  NORMAL: '#10b981',
  DROWSY: '#f59e0b',
  AGGRESSIVE: '#ef4444',
  UNKNOWN: '#94a3b8'
};

// ---------------------------------------------------------------------------
// DOM Elements
// ---------------------------------------------------------------------------
const els = {
  driverFilter: document.getElementById('driverFilter'),
  tripSelect: document.getElementById('tripSelect'),
  btnLoadTrip: document.getElementById('btnLoadTrip'),
  kpiPredictedStyle: document.getElementById('kpiPredictedStyle'),
  kpiGroundTruth: document.getElementById('kpiGroundTruth'),
  kpiDuration: document.getElementById('kpiDuration'),
  kpiDistance: document.getElementById('kpiDistance'),
  kpiEventsCount: document.getElementById('kpiEventsCount'),
  kpiHighSeverity: document.getElementById('kpiHighSeverity'),
  kpiRiskScore: document.getElementById('kpiRiskScore'),
  kpiLogProb: document.getElementById('kpiLogProb'),
  eventsTableBody: document.getElementById('eventsTableBody'),
  eventsCountBadge: document.getElementById('eventsCountBadge'),
  btnPlayPause: document.getElementById('btnPlayPause'),
  playIcon: document.getElementById('playIcon'),
  btnRestart: document.getElementById('btnRestart'),
  timeScrubber: document.getElementById('timeScrubber'),
  currentTime: document.getElementById('currentTime'),
  totalTime: document.getElementById('totalTime'),
  replaySpeed: document.getElementById('replaySpeed'),
  hudSpeed: document.getElementById('hudSpeed'),
  hudAx: document.getElementById('hudAx'),
  hudAy: document.getElementById('hudAy'),
  hudState: document.getElementById('hudState'),
  timelineDurationLabel: document.getElementById('timelineDurationLabel'),
  stateRibbonWrapper: document.getElementById('stateRibbonWrapper'),
  stateRibbonCanvas: document.getElementById('stateRibbonCanvas'),
  statePlayhead: document.getElementById('statePlayhead'),
  timelineEventsTrack: document.getElementById('timelineEventsTrack'),
  viterbiTable: document.getElementById('viterbiTable').querySelector('tbody'),
  statRecall: document.getElementById('statRecall'),
  statAccuracy: document.getElementById('statAccuracy'),
  statMacroF1: document.getElementById('statMacroF1'),
};

// ---------------------------------------------------------------------------
// App Startup
// ---------------------------------------------------------------------------
document.addEventListener('DOMContentLoaded', () => {
  initEventListeners();
  renderDonutChart(null); // Render illustrative donut chart as in user screenshot
  loadTripsList();
  loadAcademicBenchmarks();
});

function initEventListeners() {
  els.driverFilter.addEventListener('change', filterTripsByDriver);
  els.btnLoadTrip.addEventListener('click', onRunInferenceClicked);
  
  // Replay controls
  els.btnPlayPause.addEventListener('click', togglePlayPause);
  els.btnRestart.addEventListener('click', restartReplay);
  els.timeScrubber.addEventListener('input', onScrubberInput);
  els.replaySpeed.addEventListener('change', (e) => {
    state.replay.speedMultiplier = parseFloat(e.target.value);
  });

  // State ribbon click-to-seek
  els.stateRibbonWrapper.addEventListener('click', (e) => {
    if (!state.telemetry || !state.telemetry.duration_s) return;
    const rect = els.stateRibbonWrapper.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const fraction = Math.max(0, Math.min(1, clickX / rect.width));
    seekToTime(fraction * state.telemetry.duration_s);
  });

  // Filter Buttons
  document.querySelectorAll('.filter-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      filterChartsView(btn.dataset.chart);
    });
  });
}

// ---------------------------------------------------------------------------
// Fetch Trips & Academic Benchmarks
// ---------------------------------------------------------------------------
async function loadTripsList() {
  try {
    const res = await fetch('/trips');
    const data = await res.json();
    state.allTrips = data.trips || [];
    populateTripDropdown(state.allTrips);
  } catch (err) {
    console.error('Failed to load trips:', err);
    els.tripSelect.innerHTML = '<option value="">ERROR CONNECTING TO API</option>';
  }
}

function populateTripDropdown(trips) {
  els.tripSelect.innerHTML = '';
  if (trips.length === 0) {
    els.tripSelect.innerHTML = '<option value="">NO RECORDINGS FOUND</option>';
    return;
  }

  trips.forEach(t => {
    const opt = document.createElement('option');
    opt.value = t.trip_id;
    opt.dataset.driver = t.driver;
    opt.textContent = `${t.driver} // ${t.label} [${t.road}] — ${t.trip_id.substring(0, 14)}`;
    els.tripSelect.appendChild(opt);
  });
}

function filterTripsByDriver() {
  const selectedDriver = els.driverFilter.value;
  if (selectedDriver === 'ALL') {
    populateTripDropdown(state.allTrips);
  } else {
    const filtered = state.allTrips.filter(t => t.driver === selectedDriver);
    populateTripDropdown(filtered);
  }
}

async function loadAcademicBenchmarks() {
  try {
    const res = await fetch('/metrics/all');
    const data = await res.json();

    // Populate Viterbi Benchmark Table
    if (data.viterbi_comparison && data.viterbi_comparison.length > 0) {
      els.viterbiTable.innerHTML = '';
      data.viterbi_comparison.forEach(r => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
          <td class="mono">SEQ ${r.sequence_idx}</td>
          <td class="mono">${r.length}</td>
          <td class="mono">${r.custom_time_ms.toFixed(1)} ms</td>
          <td class="mono">${r.ref_time_ms.toFixed(2)} ms</td>
          <td class="mono" style="color: #047857; font-weight: 700;">${(r.sequence_agreement * 100).toFixed(2)}%</td>
          <td class="mono">${r.log_prob_diff.toFixed(4)}</td>
        `;
        els.viterbiTable.appendChild(tr);
      });
    }

    // Populate Metrics
    if (data.event_metrics) {
      els.statRecall.textContent = `${(data.event_metrics.recall * 100).toFixed(2)}%`;
    }
    if (data.baseline_comparison) {
      els.statAccuracy.textContent = `${(data.baseline_comparison.hmm_accuracy * 100).toFixed(1)}%`;
      els.statMacroF1.textContent = data.baseline_comparison.hmm_macro_f1.toFixed(3);
    }
  } catch (err) {
    console.error('Failed to load benchmarks:', err);
  }
}

// ---------------------------------------------------------------------------
// Run HMM Inference & Ingest Telemetry
// ---------------------------------------------------------------------------
async function onRunInferenceClicked() {
  const tripId = els.tripSelect.value;
  if (!tripId) return;

  pauseReplay();
  els.btnLoadTrip.disabled = true;
  els.btnLoadTrip.textContent = 'INFERRING...';

  try {
    // 1. Run HMM inference
    await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ trip_id: tripId })
    });

    // 2. Fetch full synchronized telemetry
    const res = await fetch(`/telemetry/${tripId}`);
    state.telemetry = await res.json();

    // 3. Render all components (Donut Chart, Timeline, Waveforms, Table)
    renderKPICards(state.telemetry);
    renderDonutChart(state.telemetry);
    renderStateTimeline(state.telemetry);
    renderEventsTable(state.telemetry.events);
    renderCharts(state.telemetry);
    setupReplay(state.telemetry);

  } catch (err) {
    console.error('Inference error:', err);
    alert('Failed to run inference: ' + err.message);
  } finally {
    els.btnLoadTrip.disabled = false;
    els.btnLoadTrip.textContent = 'RUN INFERENCE';
  }
}

// ---------------------------------------------------------------------------
// Render KPI Metrics
// ---------------------------------------------------------------------------
function renderKPICards(data) {
  const style = data.dominant_label;
  const pillClass = STATE_CLASSES[style] || 'state-neutral';
  els.kpiPredictedStyle.innerHTML = `<span class="state-pill ${pillClass}">${style}</span>`;
  els.kpiGroundTruth.textContent = `REFERENCE LABEL: ${data.ground_truth_label}`;

  const minutes = Math.floor(data.duration_s / 60);
  const seconds = (data.duration_s % 60).toFixed(1);
  els.kpiDuration.textContent = `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(4, '0')}`;
  els.kpiDistance.textContent = `DRIVER: ${data.driver}`;

  const events = data.events || [];
  els.kpiEventsCount.textContent = events.length;
  const highSev = events.filter(e => e.severity === 'high').length;
  els.kpiHighSeverity.textContent = `${highSev} HIGH-SEVERITY EVENTS`;

  // Risk Score
  let riskText = 'LOW';
  let riskClass = 'state-low';
  if (highSev >= 3 || style === 'AGGRESSIVE') {
    riskText = 'HIGH';
    riskClass = 'state-aggressive';
  } else if (highSev >= 1 || style === 'DROWSY') {
    riskText = 'MODERATE';
    riskClass = 'state-drowsy';
  }
  els.kpiRiskScore.innerHTML = `<span class="state-pill ${riskClass}">${riskText}</span>`;
  els.kpiLogProb.textContent = `SEQ LOG-LIKELIHOOD: ${data.log_prob.toFixed(1)}`;
}

// ---------------------------------------------------------------------------
// Render Donut Chart: Driving Behaviour Analysis (Matches Uploaded Screenshot)
// ---------------------------------------------------------------------------
function renderDonutChart(data) {
  // Default illustrative distribution matching user reference image
  let aggressivePct = 16;
  let brakingPct = 14;
  let normalPct = 58;
  let turningPct = 12;

  if (data && data.timestamps && data.timestamps.length > 0) {
    const total = data.timestamps.length;
    let normalCount = 0;
    let aggressiveCount = 0;
    let brakingCount = 0;
    let turningCount = 0;

    const ax = data.ax_filt || [];
    const ay = data.ay_filt || [];

    for (let i = 0; i < total; i++) {
      const a_x = ax[i] || 0;
      const a_y = Math.abs(ay[i] || 0);

      if (a_x <= -0.07) {
        brakingCount++;
      } else if (a_y >= 0.055) {
        turningCount++;
      } else if (a_x >= 0.045) {
        aggressiveCount++;
      } else {
        normalCount++;
      }
    }

    normalPct = Math.round((normalCount / total) * 100);
    brakingPct = Math.round((brakingCount / total) * 100);
    turningPct = Math.round((turningCount / total) * 100);
    aggressivePct = Math.max(1, 100 - (normalPct + brakingPct + turningPct));
  }

  const canvasEl = document.getElementById('chartDonut');
  if (!canvasEl) return;

  if (state.charts.donut) {
    state.charts.donut.destroy();
    state.charts.donut = null;
  }

  const ctx = canvasEl.getContext('2d');
  state.charts.donut = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: ['Aggressive', 'Braking', 'Normal', 'Turning'],
      datasets: [{
        data: [aggressivePct, brakingPct, normalPct, turningPct],
        backgroundColor: ['#52c474', '#f17a4a', '#3b82f6', '#f4cc4d'],
        borderWidth: 2,
        borderColor: '#ffffff',
        hoverOffset: 6
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: true,
      aspectRatio: 1,
      cutout: '62%',
      plugins: {
        legend: {
          display: false
        },
        tooltip: {
          bodyFont: { family: 'Plus Jakarta Sans', size: 12 },
          titleFont: { family: 'Plus Jakarta Sans', size: 12 },
          callbacks: {
            label: function(context) {
              return ` ${context.label}: ${context.raw}%`;
            }
          }
        }
      }
    }
  });
}

// ---------------------------------------------------------------------------
// Render Decoded State Timeline Ribbon (Pure Sequential Visualization)
// ---------------------------------------------------------------------------
function renderStateTimeline(data) {
  const duration = data.duration_s || 1.0;
  els.timelineDurationLabel.textContent = `${duration.toFixed(1)}s`;

  const canvas = els.stateRibbonCanvas;
  const rect = els.stateRibbonWrapper.getBoundingClientRect();
  canvas.width = rect.width * (window.devicePixelRatio || 1);
  canvas.height = 56 * (window.devicePixelRatio || 1);

  const ctx = canvas.getContext('2d');
  ctx.scale(window.devicePixelRatio || 1, window.devicePixelRatio || 1);
  ctx.clearRect(0, 0, rect.width, 56);

  // Background
  ctx.fillStyle = '#f1f5f9';
  ctx.fillRect(0, 0, rect.width, 56);

  // Draw decoded state segments
  const intervals = data.state_intervals || [];
  if (intervals.length > 0) {
    intervals.forEach(iv => {
      const xStart = (iv.t_start / duration) * rect.width;
      const xEnd = (iv.t_end / duration) * rect.width;
      const width = Math.max(2, xEnd - xStart);

      ctx.fillStyle = STATE_COLORS[iv.label] || '#94a3b8';
      ctx.globalAlpha = 0.85;
      ctx.fillRect(xStart, 0, width, 56);
    });
    ctx.globalAlpha = 1.0;
  } else {
    ctx.fillStyle = '#e2e8f0';
    ctx.fillRect(0, 0, rect.width, 56);
  }

  // Render event markers along track
  els.timelineEventsTrack.innerHTML = '';
  const events = data.events || [];
  events.forEach(ev => {
    const fraction = Math.max(0, Math.min(1, ev.t_start / duration));
    const percent = fraction * 100;

    const tick = document.createElement('div');
    const sevClass = `event-tick-${ev.severity}`;
    tick.className = `event-timeline-tick ${sevClass}`;
    tick.style.left = `${percent}%`;
    tick.title = `${formatEventName(ev.event_type)} (${ev.t_start.toFixed(1)}s)`;
    tick.textContent = ev.severity === 'high' ? '!' : '•';

    tick.addEventListener('click', (e) => {
      e.stopPropagation();
      seekToTime(ev.t_start);
    });

    els.timelineEventsTrack.appendChild(tick);
  });
}

// ---------------------------------------------------------------------------
// Render Safety Event Table
// ---------------------------------------------------------------------------
function renderEventsTable(events) {
  els.eventsTableBody.innerHTML = '';
  if (!events || events.length === 0) {
    els.eventsTableBody.innerHTML = '<tr><td colspan="5" class="empty-notice">No safety-critical events detected.</td></tr>';
    els.eventsCountBadge.textContent = '0 DETECTED';
    return;
  }

  els.eventsCountBadge.textContent = `${events.length} DETECTED`;

  events.forEach(ev => {
    const tr = document.createElement('tr');
    const sevClass = `sev-${ev.severity}`;

    tr.innerHTML = `
      <td class="mono">${ev.t_start.toFixed(1)}s</td>
      <td><strong>${formatEventName(ev.event_type)}</strong></td>
      <td><span class="sev-tag ${sevClass}">${ev.severity.toUpperCase()}</span></td>
      <td class="mono">${ev.ax_peak.toFixed(3)} g</td>
      <td class="mono">${ev.ay_peak.toFixed(3)} g</td>
    `;

    tr.addEventListener('click', () => {
      seekToTime(ev.t_start);
    });

    els.eventsTableBody.appendChild(tr);
  });
}

// ---------------------------------------------------------------------------
// Render Clean Kinematic Waveforms (Chart.js)
// ---------------------------------------------------------------------------
function renderCharts(data) {
  const ts = data.timestamps;
  destroyCharts();

  const baseConfig = {
    responsive: true,
    maintainAspectRatio: false,
    animation: false,
    elements: {
      point: { radius: 0 },
      line: { borderWidth: 1.4 }
    },
    scales: {
      x: {
        grid: { color: '#e2e8f0' },
        ticks: { color: '#64748b', font: { family: 'Plus Jakarta Sans', size: 11, weight: '600' }, maxTicksLimit: 12 }
      },
      y: {
        grid: { color: '#e2e8f0' },
        ticks: { color: '#64748b', font: { family: 'Plus Jakarta Sans', size: 11, weight: '600' } }
      }
    },
    plugins: {
      legend: { display: false }
    }
  };

  // Longitudinal Accel ax
  state.charts.ax = new Chart(document.getElementById('chartAx'), {
    type: 'line',
    data: {
      labels: ts,
      datasets: [{
        data: data.ax_filt,
        borderColor: '#0284c7',
        borderWidth: 1.4
      }]
    },
    options: baseConfig
  });

  // Lateral Accel ay
  state.charts.ay = new Chart(document.getElementById('chartAy'), {
    type: 'line',
    data: {
      labels: ts,
      datasets: [{
        data: data.ay_filt,
        borderColor: '#ea580c',
        borderWidth: 1.4
      }]
    },
    options: baseConfig
  });

  // Longitudinal Jerk
  state.charts.jerk = new Chart(document.getElementById('chartJerk'), {
    type: 'line',
    data: {
      labels: ts,
      datasets: [{
        data: data.jerk_long,
        borderColor: '#dc2626',
        borderWidth: 1.4
      }]
    },
    options: baseConfig
  });

  // Speed
  state.charts.speed = new Chart(document.getElementById('chartSpeed'), {
    type: 'line',
    data: {
      labels: ts,
      datasets: [{
        data: data.speed_kmh,
        borderColor: '#059669',
        borderWidth: 1.4
      }]
    },
    options: baseConfig
  });
}

function destroyCharts() {
  ['ax', 'ay', 'jerk', 'speed'].forEach(k => {
    if (state.charts[k]) {
      state.charts[k].destroy();
      state.charts[k] = null;
    }
  });
}

function filterChartsView(type) {
  const boxes = document.querySelectorAll('.waveform-box');
  if (type === 'all') {
    boxes.forEach(b => b.style.display = 'block');
  } else if (type === 'accel') {
    boxes[0].style.display = 'block';
    boxes[1].style.display = 'block';
    boxes[2].style.display = 'none';
    boxes[3].style.display = 'none';
  } else if (type === 'jerk') {
    boxes[0].style.display = 'none';
    boxes[1].style.display = 'none';
    boxes[2].style.display = 'block';
    boxes[3].style.display = 'none';
  } else if (type === 'speed') {
    boxes[0].style.display = 'none';
    boxes[1].style.display = 'none';
    boxes[2].style.display = 'none';
    boxes[3].style.display = 'block';
  }
}

// ---------------------------------------------------------------------------
// Replay Engine Implementation
// ---------------------------------------------------------------------------
function setupReplay(data) {
  state.replay.currentTime = 0;
  els.timeScrubber.min = 0;
  els.timeScrubber.max = data.duration_s;
  els.timeScrubber.value = 0;
  els.currentTime.textContent = '00:00.0';
  els.totalTime.textContent = formatTime(data.duration_s);
  updateReplayHUD(0);
}

function togglePlayPause() {
  if (!state.telemetry) return;
  if (state.replay.isPlaying) {
    pauseReplay();
  } else {
    startReplay();
  }
}

function startReplay() {
  if (!state.telemetry) return;
  state.replay.isPlaying = true;
  els.playIcon.textContent = 'PAUSE';

  const intervalMs = 1000 / state.replay.fps;
  const timeStep = (intervalMs / 1000) * state.replay.speedMultiplier;

  state.replay.timer = setInterval(() => {
    state.replay.currentTime += timeStep;
    if (state.replay.currentTime >= state.telemetry.duration_s) {
      pauseReplay();
      state.replay.currentTime = state.telemetry.duration_s;
    }
    els.timeScrubber.value = state.replay.currentTime;
    updateReplayHUD(state.replay.currentTime);
  }, intervalMs);
}

function pauseReplay() {
  state.replay.isPlaying = false;
  els.playIcon.textContent = 'PLAY';
  if (state.replay.timer) {
    clearInterval(state.replay.timer);
    state.replay.timer = null;
  }
}

function restartReplay() {
  pauseReplay();
  seekToTime(0);
}

function onScrubberInput(e) {
  const t = parseFloat(e.target.value);
  state.replay.currentTime = t;
  updateReplayHUD(t);
}

function seekToTime(t) {
  state.replay.currentTime = t;
  els.timeScrubber.value = t;
  updateReplayHUD(t);
}

function updateReplayHUD(t) {
  els.currentTime.textContent = formatTime(t);

  const d = state.telemetry;
  if (!d || !d.timestamps || d.timestamps.length === 0) return;

  // Playhead position along timeline ribbon
  const duration = d.duration_s || 1.0;
  const fraction = Math.max(0, Math.min(1, t / duration));
  els.statePlayhead.style.left = `${(fraction * 100).toFixed(2)}%`;

  // Search nearest sample index
  const ts = d.timestamps;
  let idx = 0;
  for (let i = 0; i < ts.length; i++) {
    if (ts[i] >= t) {
      idx = i;
      break;
    }
  }

  // Live HUD
  const spd = d.speed_kmh[idx] !== undefined ? d.speed_kmh[idx] : (d.speed_ms[idx] * 3.6);
  els.hudSpeed.innerHTML = `${spd.toFixed(1)} <span class="unit">km/h</span>`;
  els.hudAx.innerHTML = `${d.ax_filt[idx].toFixed(3)} <span class="unit">g</span>`;
  els.hudAy.innerHTML = `${d.ay_filt[idx].toFixed(3)} <span class="unit">g</span>`;

  // Active HMM State
  let currentState = 'NORMAL';
  if (d.state_intervals) {
    for (let interval of d.state_intervals) {
      if (t >= interval.t_start && t <= interval.t_end) {
        currentState = interval.label;
        break;
      }
    }
  }
  els.hudState.textContent = currentState;
  els.hudState.className = `readout-value mono state-text-${currentState.toLowerCase()}`;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
function formatTime(seconds) {
  const m = Math.floor(seconds / 60);
  const s = (seconds % 60).toFixed(1);
  return `${String(m).padStart(2, '0')}:${String(s).padStart(4, '0')}`;
}

function formatEventName(type) {
  return type
    .replace(/_/g, ' ')
    .toUpperCase();
}