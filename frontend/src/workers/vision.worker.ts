import { FaceLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";

const WASM = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21/wasm";
const MODEL =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";

const scope = self as unknown as {
  onmessage: ((event: MessageEvent) => void) | null;
  postMessage: (message: unknown) => void;
};

let landmarker: FaceLandmarker | null = null;

async function createLandmarker(): Promise<FaceLandmarker> {
  if (landmarker) return landmarker;
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

scope.onmessage = async (event: MessageEvent) => {
  const data = event.data as { type: string; id?: number; bitmap?: ImageBitmap; timestamp?: number };
  try {
    if (data.type === "init") {
      await createLandmarker();
      scope.postMessage({ type: "ready" });
      return;
    }
    if (data.type === "frame" && data.bitmap && data.id !== undefined && data.timestamp !== undefined) {
      const marker = await createLandmarker();
      const result = marker.detectForVideo(data.bitmap, data.timestamp);
      data.bitmap.close();
      const landmarks = result.faceLandmarks?.[0];
      if (!landmarks) {
        scope.postMessage({ type: "result", id: data.id, face: null });
        return;
      }
      const categories = result.faceBlendshapes?.[0]?.categories ?? [];
      const matrix = result.facialTransformationMatrixes?.[0]?.data;
      scope.postMessage({
        type: "result",
        id: data.id,
        face: {
          landmarks: landmarks.map((point) => [point.x, point.y, point.z ?? 0]),
          blendshapes: Object.fromEntries(categories.map((item) => [item.categoryName ?? "", item.score ?? 0])),
          matrix: matrix ? Array.from(matrix) : null,
        },
      });
    }
  } catch (error) {
    scope.postMessage({
      type: "error",
      message: error instanceof Error ? error.message : "Facial tracking stopped.",
    });
  }
};
