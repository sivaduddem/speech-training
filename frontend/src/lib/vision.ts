import { FaceLandmarker, FilesetResolver, type FaceLandmarkerResult } from "@mediapipe/tasks-vision";

const WASM = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21/wasm";
const MODEL =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";

export interface VisionFace {
  landmarks: number[][];
  blendshapes: Record<string, number>;
  matrix: number[] | null;
}

let landmarker: FaceLandmarker | null = null;
let loading: Promise<FaceLandmarker> | null = null;

/**
 * MediaPipe loads its runtime with importScripts, or with a script tag when that is missing.
 * A worker has neither, so the landmarker is created on the page.
 */
export function loadFaceLandmarker(): Promise<FaceLandmarker> {
  if (landmarker) return Promise.resolve(landmarker);
  if (!loading) {
    loading = createLandmarker().catch((error: unknown) => {
      loading = null;
      throw error;
    });
  }
  return loading;
}

async function createLandmarker(): Promise<FaceLandmarker> {
  const vision = await FilesetResolver.forVisionTasks(WASM);
  const options = {
    baseOptions: { modelAssetPath: MODEL },
    runningMode: "VIDEO" as const,
    numFaces: 1,
    outputFaceBlendshapes: true,
    outputFacialTransformationMatrixes: true,
  };
  try {
    landmarker = await FaceLandmarker.createFromOptions(vision, {
      ...options,
      baseOptions: { ...options.baseOptions, delegate: "GPU" },
    });
  } catch {
    landmarker = await FaceLandmarker.createFromOptions(vision, {
      ...options,
      baseOptions: { ...options.baseOptions, delegate: "CPU" },
    });
  }
  return landmarker;
}

export function detectFace(video: HTMLVideoElement, timestamp: number): VisionFace | null {
  if (!landmarker || video.readyState < 2 || !video.videoWidth) return null;
  const result = landmarker.detectForVideo(video, timestamp);
  return faceFromResult(result);
}

function faceFromResult(result: FaceLandmarkerResult): VisionFace | null {
  const landmarks = result.faceLandmarks?.[0];
  if (!landmarks) return null;
  const categories = result.faceBlendshapes?.[0]?.categories ?? [];
  const matrix = result.facialTransformationMatrixes?.[0]?.data;
  return {
    landmarks: landmarks.map((point) => [point.x, point.y, point.z ?? 0]),
    blendshapes: Object.fromEntries(categories.map((item) => [item.categoryName ?? "", item.score ?? 0])),
    matrix: matrix ? Array.from(matrix) : null,
  };
}
