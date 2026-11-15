import type { FeatureSample, SpeechEvent } from "../types";

const EVENT_COLOR: Record<SpeechEvent["type"], string> = {
  repetition: "rgba(226, 177, 90, 0.28)",
  prolongation: "rgba(195, 178, 214, 0.32)",
  block: "rgba(142, 176, 201, 0.32)",
  restart: "rgba(134, 196, 190, 0.3)",
  pause: "rgba(138, 129, 120, 0.22)",
};

export function drawCharts(
  canvas: HTMLCanvasElement,
  samples: FeatureSample[],
  events: SpeechEvent[],
  cursor: number,
  showEvents: boolean,
  windowSec = 12,
): void {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  const width = canvas.clientWidth || 640;
  const height = canvas.clientHeight || 280;
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, width, height);

  const start = cursor - windowSec;
  const visible = samples.filter((sample) => sample.t >= start && sample.t <= cursor + 0.01);
  const rows = [
    { label: "Loudness", pick: (sample: FeatureSample) => sample.rms, scale: 0.12 },
    { label: "Lip opening", pick: (sample: FeatureSample) => sample.lip, scale: 0.7 },
    { label: "Jaw opening", pick: (sample: FeatureSample) => sample.jaw, scale: 1 },
    { label: "Lip shape", pick: (sample: FeatureSample) => sample.compression, scale: 1 },
  ];
  const rowH = height / rows.length;
  ctx.font = "12px Outfit, sans-serif";

  rows.forEach((row, index) => {
    const top = index * rowH;
    ctx.fillStyle = index % 2 === 0 ? "rgba(255,248,236,0.03)" : "rgba(255,248,236,0.015)";
    ctx.fillRect(0, top, width, rowH);
    if (showEvents) {
      for (const event of events) {
        if (event.end < start || event.start > cursor) continue;
        const x1 = xAt(Math.max(event.start, start), start, windowSec, width);
        const x2 = xAt(Math.min(event.end, cursor), start, windowSec, width);
        ctx.fillStyle = EVENT_COLOR[event.type];
        ctx.fillRect(x1, top + 4, Math.max(2, x2 - x1), rowH - 8);
      }
    }
    ctx.strokeStyle = "rgba(243,236,223,0.8)";
    ctx.lineWidth = 1.4;
    ctx.beginPath();
    visible.forEach((sample, point) => {
      const x = xAt(sample.t, start, windowSec, width);
      const y = top + rowH - 16 - Math.min(1, row.pick(sample) / row.scale) * (rowH - 28);
      if (point === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.fillStyle = "rgba(181,170,155,0.95)";
    ctx.fillText(row.label, 10, top + 16);
  });
}

function xAt(time: number, start: number, windowSec: number, width: number): number {
  return ((time - start) / windowSec) * width;
}
