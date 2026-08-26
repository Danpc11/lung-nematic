'use client';

import { ChangeEvent, useEffect, useMemo, useRef, useState } from 'react';

type Visit = { patientId: string; month: number; fvc: number; dlco?: number; spo2?: number; treated: boolean };
type Fit = { patientId: string; visits: Visit[]; fvc0: number; dlco0?: number; untreatedRate: number; observedRate: number; projectedRate: number; confidence: number };

const DEMO: Visit[] = [
  { patientId: 'IPF-017', month: 0, fvc: 74, dlco: 57, spo2: 95, treated: false },
  { patientId: 'IPF-017', month: 6, fvc: 71.2, dlco: 54.8, spo2: 94, treated: true },
  { patientId: 'IPF-017', month: 12, fvc: 69.1, dlco: 52.3, spo2: 94, treated: true },
  { patientId: 'IPF-017', month: 18, fvc: 66.8, dlco: 49.7, spo2: 93, treated: true },
  { patientId: 'IPF-042', month: 0, fvc: 81, dlco: 63, spo2: 96, treated: true },
  { patientId: 'IPF-042', month: 6, fvc: 79.8, dlco: 62.1, spo2: 96, treated: true },
  { patientId: 'IPF-042', month: 12, fvc: 78.7, dlco: 61, spo2: 95, treated: true },
  { patientId: 'IPF-042', month: 18, fvc: 77.9, dlco: 60.2, spo2: 95, treated: true },
];

function linearFit(visits: Visit[], field: 'fvc' | 'dlco') {
  const points = visits.filter((v) => Number.isFinite(v[field]));
  if (points.length < 2) return { intercept: Number(points[0]?.[field] ?? NaN), slope: 0, se: 8 };
  const xs = points.map((v) => v.month / 12), ys = points.map((v) => Number(v[field]));
  const mx = xs.reduce((a, b) => a + b, 0) / xs.length, my = ys.reduce((a, b) => a + b, 0) / ys.length;
  const denom = xs.reduce((s, x) => s + (x - mx) ** 2, 0);
  const slope = denom ? xs.reduce((s, x, i) => s + (x - mx) * (ys[i] - my), 0) / denom : 0;
  const intercept = my - slope * mx;
  const residual = Math.sqrt(ys.reduce((s, y, i) => s + (y - intercept - slope * xs[i]) ** 2, 0) / Math.max(1, ys.length - 2));
  return { intercept, slope, se: Math.max(field === 'fvc' ? 3 : 5, residual) };
}

function fitPatients(visits: Visit[], treatmentEffect: number): Fit[] {
  return [...new Set(visits.map((v) => v.patientId))].map((patientId) => {
    const patientVisits = visits.filter((v) => v.patientId === patientId).sort((a, b) => a.month - b.month);
    const fvc = linearFit(patientVisits, 'fvc'), dlco = linearFit(patientVisits, 'dlco');
    const treatedFraction = patientVisits.filter((v) => v.treated).length / patientVisits.length;
    const observedRate = Math.max(0, -fvc.slope);
    return { patientId, visits: patientVisits, fvc0: fvc.intercept, dlco0: Number.isFinite(dlco.intercept) ? dlco.intercept : undefined, observedRate, untreatedRate: observedRate + treatmentEffect * treatedFraction, projectedRate: observedRate, confidence: fvc.se };
  });
}

function parseCsv(text: string): Visit[] {
  const lines = text.trim().split(/\r?\n/).filter(Boolean);
  if (lines.length < 2) throw new Error('El archivo no contiene visitas.');
  const headers = lines[0].split(',').map((h) => h.trim().toLowerCase());
  const index = (...names: string[]) => names.map((n) => headers.indexOf(n)).find((i) => i >= 0) ?? -1;
  const p = index('patient_id', 'patient', 'paciente'), m = index('month', 'months', 'mes', 'meses', 'weeks');
  const f = index('fvc_pct', 'percent', 'fvc_percent'), d = index('dlco_pct', 'dlco'), s = index('spo2'), t = index('antifibrotic', 'treated', 'tratamiento');
  if (p < 0 || m < 0 || f < 0) throw new Error('Se requieren patient_id, month (o Weeks) y fvc_pct (o Percent).');
  const weeks = headers[m] === 'weeks';
  return lines.slice(1).map((line, row) => {
    const cells = line.split(',').map((c) => c.trim());
    const visit: Visit = { patientId: cells[p], month: Number(cells[m]) * (weeks ? 12 / 52 : 1), fvc: Number(cells[f]), treated: t >= 0 ? ['1', 'true', 'yes', 'sí', 'si'].includes(cells[t].toLowerCase()) : false };
    if (d >= 0 && cells[d] !== '') visit.dlco = Number(cells[d]);
    if (s >= 0 && cells[s] !== '') visit.spo2 = Number(cells[s]);
    if (!visit.patientId || !Number.isFinite(visit.month) || !Number.isFinite(visit.fvc)) throw new Error(`Fila ${row + 2}: datos inválidos.`);
    return visit;
  });
}

function TwinChart({ fit, horizon }: { fit: Fit; horizon: number }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = ref.current; if (!canvas) return;
    const rect = canvas.getBoundingClientRect(), ratio = window.devicePixelRatio || 1;
    canvas.width = rect.width * ratio; canvas.height = rect.height * ratio;
    const ctx = canvas.getContext('2d'); if (!ctx) return; ctx.scale(ratio, ratio);
    const w = rect.width, h = rect.height, pad = { l: 48, r: 20, t: 24, b: 38 };
    const maxMonth = Math.max(horizon, ...fit.visits.map((v) => v.month));
    const projected = fit.fvc0 - fit.projectedRate * maxMonth / 12;
    const values = [...fit.visits.map((v) => v.fvc), projected - fit.confidence, projected + fit.confidence];
    const ymin = Math.floor((Math.min(...values) - 4) / 5) * 5, ymax = Math.ceil((Math.max(...values) + 4) / 5) * 5;
    const x = (month: number) => pad.l + month / maxMonth * (w - pad.l - pad.r);
    const y = (value: number) => pad.t + (ymax - value) / (ymax - ymin) * (h - pad.t - pad.b);
    ctx.font = '12px Arial'; ctx.fillStyle = '#64706d'; ctx.strokeStyle = '#dfe7e4'; ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) { const value = ymin + (ymax - ymin) * i / 4; ctx.beginPath(); ctx.moveTo(pad.l, y(value)); ctx.lineTo(w - pad.r, y(value)); ctx.stroke(); ctx.fillText(value.toFixed(0), 12, y(value) + 4); }
    ctx.fillText('meses', w - 52, h - 10); ctx.fillStyle = 'rgba(19,126,105,.13)'; ctx.beginPath();
    for (let month = 0; month <= maxMonth; month++) ctx.lineTo(x(month), y(fit.fvc0 - fit.projectedRate * month / 12 + fit.confidence * Math.sqrt(1 + month / 12)));
    for (let month = maxMonth; month >= 0; month--) ctx.lineTo(x(month), y(fit.fvc0 - fit.projectedRate * month / 12 - fit.confidence * Math.sqrt(1 + month / 12)));
    ctx.closePath(); ctx.fill(); ctx.strokeStyle = '#137e69'; ctx.lineWidth = 3; ctx.beginPath(); ctx.moveTo(x(0), y(fit.fvc0)); ctx.lineTo(x(maxMonth), y(projected)); ctx.stroke();
    fit.visits.forEach((v) => { ctx.fillStyle = v.treated ? '#a94e72' : '#172c28'; ctx.beginPath(); ctx.arc(x(v.month), y(v.fvc), 5, 0, Math.PI * 2); ctx.fill(); });
  }, [fit, horizon]);
  return <canvas ref={ref} className="twin-chart" aria-label="Trayectoria longitudinal de FVC" />;
}

export default function Home() {
  const [visits, setVisits] = useState(DEMO), [selected, setSelected] = useState('IPF-017');
  const [horizon, setHorizon] = useState(36), [effect, setEffect] = useState(2.64), [error, setError] = useState('');
  const fits = useMemo(() => fitPatients(visits, effect), [visits, effect]);
  const fit = fits.find((item) => item.patientId === selected) ?? fits[0];
  async function loadFile(event: ChangeEvent<HTMLInputElement>) { const file = event.target.files?.[0]; if (!file) return; try { const parsed = parseCsv(await file.text()); setVisits(parsed); setSelected(parsed[0].patientId); setError(''); } catch (e) { setError(e instanceof Error ? e.message : 'No fue posible leer el archivo.'); } }

  return <main>
    <header className="topbar"><div className="brand"><span className="brandmark">LT</span><div><strong>LungTwin IPF</strong><small>Gemelo digital de investigación</small></div></div><span className="research-badge">Uso experimental · no clínico</span></header>
    <section className="hero"><div><p className="eyebrow">Seguimiento longitudinal personalizado</p><h1>De visitas clínicas a una trayectoria interpretable.</h1><p className="hero-copy">Carga FVC, DLCO y tratamiento. El gemelo estima la velocidad individual de progresión, proyecta escenarios y muestra lo que los datos todavía no permiten concluir.</p></div><label className="upload-button">Cargar CSV<input type="file" accept=".csv,text/csv" onChange={loadFile} /></label></section>
    {error && <div className="error" role="alert">{error}</div>}
    <section className="workspace">
      <aside className="patient-panel"><div className="panel-heading"><div><span>PACIENTES</span><strong>{fits.length}</strong></div><button onClick={() => { setVisits(DEMO); setSelected('IPF-017'); }}>Usar ejemplo</button></div><div className="patient-list">{fits.map((item) => <button key={item.patientId} className={item.patientId === fit?.patientId ? 'active' : ''} onClick={() => setSelected(item.patientId)}><span className="avatar">{item.patientId.slice(-2)}</span><span><strong>{item.patientId}</strong><small>{item.visits.length} visitas · {item.visits.at(-1)?.month.toFixed(0)} meses</small></span><i>›</i></button>)}</div><div className="format-help"><strong>Formato mínimo</strong><code>patient_id, month, fvc_pct</code><small>También acepta Weeks + Percent de OSIC.</small></div></aside>
      {fit && <section className="dashboard">
        <div className="patient-title"><div><p className="eyebrow">GEMELO ACTIVO</p><h2>{fit.patientId}</h2></div><div className="controls"><label>Proyección<select value={horizon} onChange={(e) => setHorizon(Number(e.target.value))}><option value="24">24 meses</option><option value="36">36 meses</option><option value="60">60 meses</option></select></label><label>Efecto antifibrótico<input value={effect} step="0.1" min="0" max="8" type="number" onChange={(e) => setEffect(Number(e.target.value))} /></label></div></div>
        <div className="metrics"><article><span>FVC basal</span><strong>{fit.fvc0.toFixed(1)}<small>% pred.</small></strong><p>ancla del paciente</p></article><article><span>Declive observado</span><strong>{fit.observedRate.toFixed(1)}<small>pp/año</small></strong><p>{fit.observedRate > 5 ? 'trayectoria rápida' : fit.observedRate > 2 ? 'trayectoria intermedia' : 'trayectoria lenta'}</p></article><article><span>Carga sin tratamiento</span><strong>{fit.untreatedRate.toFixed(1)}<small>pp/año</small></strong><p>escenario contrafactual</p></article><article><span>Incertidumbre</span><strong>±{fit.confidence.toFixed(1)}<small>pp</small></strong><p>{fit.visits.length < 4 ? 'datos insuficientes' : 'estimación preliminar'}</p></article></div>
        <article className="chart-card"><div className="card-head"><div><h3>Trayectoria de FVC</h3><p>Observaciones, ajuste individual e intervalo exploratorio</p></div><div className="legend"><span className="observed">sin tratamiento</span><span className="treated">con antifibrótico</span><span className="model">modelo</span></div></div><TwinChart fit={fit} horizon={horizon} /></article>
        <div className="lower-grid"><article className="state-card"><h3>Estado latente estimado</h3><div className="state-row"><span>Carga fibrótica relativa</span><div><i style={{ width: `${Math.min(100, fit.observedRate * 10)}%` }} /></div><strong>{Math.min(100, fit.observedRate * 10).toFixed(0)}%</strong></div><div className="state-row"><span>Reserva funcional</span><div><i className="reserve" style={{ width: `${Math.max(8, 100 - fit.observedRate * 7)}%` }} /></div><strong>{Math.max(8, 100 - fit.observedRate * 7).toFixed(0)}%</strong></div><p>Estados normalizados para explorar la trayectoria; no son biomarcadores validados.</p></article><article className="guardrail"><span className="shield">!</span><div><h3>Lectura responsable</h3><p>El modelo caracteriza tendencia y precisión. No estima supervivencia ni un “punto de no retorno” con este número de visitas.</p><strong>{fit.visits.length >= 4 ? 'Apto para exploración longitudinal' : 'Se recomiendan ≥4 visitas'}</strong></div></article></div>
      </section>}
    </section>
  </main>;
}
