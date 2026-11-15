class CaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.chunks = [];
    this.samples = 0;
    this.target = Math.round(sampleRate * 0.05);
  }

  process(inputs) {
    const channel = inputs[0] && inputs[0][0];
    if (!channel) return true;
    this.chunks.push(new Float32Array(channel));
    this.samples += channel.length;
    if (this.samples >= this.target) {
      const merged = new Float32Array(this.samples);
      let offset = 0;
      for (const chunk of this.chunks) {
        merged.set(chunk, offset);
        offset += chunk.length;
      }
      this.port.postMessage({ samples: merged, sampleRate }, [merged.buffer]);
      this.chunks = [];
      this.samples = 0;
    }
    return true;
  }
}

registerProcessor("capture-processor", CaptureProcessor);
