import { FaceLandmarker } from "@mediapipe/tasks-vision";
import type { Pt } from "./faceFeatures";

const GROUPS: { links: { start: number; end: number }[]; color: string; width: number }[] = [
  { links: FaceLandmarker.FACE_LANDMARKS_TESSELATION, color: "rgba(226, 177, 90, 0.28)", width: 0.7 },
  { links: FaceLandmarker.FACE_LANDMARKS_FACE_OVAL, color: "rgba(156, 190, 166, 0.9)", width: 1.3 },
  { links: FaceLandmarker.FACE_LANDMARKS_LEFT_EYE, color: "rgba(156, 190, 166, 0.95)", width: 1.1 },
  { links: FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE, color: "rgba(156, 190, 166, 0.95)", width: 1.1 },
  { links: FaceLandmarker.FACE_LANDMARKS_LEFT_EYEBROW, color: "rgba(226, 177, 90, 0.7)", width: 1 },
  { links: FaceLandmarker.FACE_LANDMARKS_RIGHT_EYEBROW, color: "rgba(226, 177, 90, 0.7)", width: 1 },
  { links: FaceLandmarker.FACE_LANDMARKS_LIPS, color: "rgba(243, 236, 223, 0.95)", width: 1.5 },
];

export function drawMesh(
  canvas: HTMLCanvasElement,
  video: HTMLVideoElement,
  landmarks: Pt[] | null,
  visible: boolean,
): void {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  const width = canvas.clientWidth || 640;
  const height = canvas.clientHeight || 480;
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  ctx.clearRect(0, 0, width, height);
  if (!visible || !landmarks || !video.videoWidth || !video.videoHeight) return;

  const videoAspect = video.videoWidth / video.videoHeight;
  const boxAspect = width / height;
  let drawW = width;
  let drawH = height;
  let offsetX = 0;
  let offsetY = 0;
  if (videoAspect > boxAspect) {
    drawH = width / videoAspect;
    offsetY = (height - drawH) / 2;
  } else {
    drawW = height * videoAspect;
    offsetX = (width - drawW) / 2;
  }

  const point = (index: number) => {
    const landmark = landmarks[index];
    if (!landmark) return null;
    return { x: offsetX + landmark.x * drawW, y: offsetY + landmark.y * drawH };
  };

  for (const group of GROUPS) {
    ctx.beginPath();
    ctx.strokeStyle = group.color;
    ctx.lineWidth = group.width;
    for (const link of group.links) {
      const a = point(link.start);
      const b = point(link.end);
      if (!a || !b) continue;
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
    }
    ctx.stroke();
  }
}
