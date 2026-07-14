//load the candle Whisper decoder wasm module
console.log("Worker script loading...");

let init, Decoder;
let moduleLoadError = null;
let moduleLoaded = false;
let pendingMessages = [];

// Load the module asynchronously
(async () => {
  try {
    const module = await import("./build/m.js");
    init = module.default;
    console.log("Module imported, initializing WASM...");
    
    // Initialize WASM first to make Decoder available
    await init("./build/m_bg.wasm");
    console.log("WASM initialized");
    
    // Now Decoder should be available
    Decoder = module.Decoder;
    console.log("Decoder available:", !!Decoder);
    
    moduleLoaded = true;
    console.log("Successfully imported and initialized module");
    
    // Process any pending messages
    pendingMessages.forEach(event => handleMessage(event));
    pendingMessages = [];
  } catch (error) {
    console.error("Failed to import module:", error);
    moduleLoadError = error;
    self.postMessage({ 
      error: "Failed to load WASM module: " + error.message 
    });
  }
})();

async function fetchArrayBuffer(url) {
  const cacheName = "whisper-candle-cache";
  const cache = await caches.open(cacheName);
  const cachedResponse = await cache.match(url);
  if (cachedResponse) {
    const data = await cachedResponse.arrayBuffer();
    return new Uint8Array(data);
  }
  const res = await fetch(url, { cache: "force-cache" });
  cache.put(url, res.clone());
  return new Uint8Array(await res.arrayBuffer());
}
class Whisper {
  static instance = {};
  // Retrieve the Whisper model. When called for the first time,
  // this will load the model and save it for future use.
  static async getInstance(params) {
    const {
      weightsURL,
      modelID,
      tokenizerURL,
      mel_filtersURL,
      configURL,
      quantized,
      is_multilingual,
      timestamps,
      task,
      language,
    } = params;
    // load individual modelID only once
    if (!this.instance[modelID]) {
      // WASM is already initialized at the top level, no need to init again
      self.postMessage({ status: "loading", message: "Loading Model" });
      
      console.log("Loading weights from:", weightsURL);
      const weightsArrayU8 = await fetchArrayBuffer(weightsURL);
      console.log("Weights loaded, size:", weightsArrayU8.length);
      
      console.log("Loading tokenizer from:", tokenizerURL);
      const tokenizerArrayU8 = await fetchArrayBuffer(tokenizerURL);
      console.log("Tokenizer loaded, size:", tokenizerArrayU8.length);
      
      console.log("Loading mel_filters from:", mel_filtersURL);
      const mel_filtersArrayU8 = await fetchArrayBuffer(mel_filtersURL);
      console.log("Mel filters loaded, size:", mel_filtersArrayU8.length);
      
      console.log("Loading config from:", configURL);
      const configArrayU8 = await fetchArrayBuffer(configURL);
      console.log("Config loaded, size:", configArrayU8.length);

      console.log("Creating Decoder with params:", {
        weightsSize: weightsArrayU8.length,
        tokenizerSize: tokenizerArrayU8.length,
        mel_filtersSize: mel_filtersArrayU8.length,
        configSize: configArrayU8.length,
        quantized,
        is_multilingual,
        timestamps
      });
      
      try {
        this.instance[modelID] = new Decoder(
          weightsArrayU8,
          tokenizerArrayU8,
          mel_filtersArrayU8,
          configArrayU8,
          quantized,
          is_multilingual,
          timestamps,
          task,
          language
        );
        console.log("Decoder created successfully");
      } catch (error) {
        console.error("Failed to create Decoder:", error);
        throw error;
      }
    } else {
      self.postMessage({ status: "loading", message: "Model Already Loaded" });
    }
    return this.instance[modelID];
  }
}

async function handleMessage(event) {
  const {
    weightsURL,
    modelID,
    tokenizerURL,
    configURL,
    mel_filtersURL,
    audioURL,
  } = event.data;
  
  console.log("Worker received message:", event.data);
  
  if (moduleLoadError) {
    console.error("Module failed to load previously:", moduleLoadError);
    self.postMessage({ 
      error: "WASM module failed to load: " + moduleLoadError.message 
    });
    return;
  }
  
  if (!moduleLoaded || !init || !Decoder) {
    console.log("Module not yet loaded, queueing message. Status:", {
      moduleLoaded,
      initDefined: !!init,
      DecoderDefined: !!Decoder
    });
    pendingMessages.push(event);
    return;
  }
  
  try {
    self.postMessage({ status: "decoding", message: "Starting Decoder" });
    let quantized = false;
    if (modelID.includes("quantized")) {
      quantized = true;
    }
    let is_multilingual = false;
    if (modelID.includes("multilingual")) {
      is_multilingual = true;
    }

    let timestamps = true;
    const decoder = await Whisper.getInstance({
      weightsURL,
      modelID,
      tokenizerURL,
      mel_filtersURL,
      configURL,
      quantized,
      is_multilingual,
      timestamps,
      task: null,
      language: null,
    });


    self.postMessage({ status: "decoding", message: "Loading Audio" });
    console.log("Fetching audio from:", audioURL);
    const audioArrayU8 = await fetchArrayBuffer(audioURL);
    console.log("Audio loaded, size:", audioArrayU8.length);

    self.postMessage({ status: "decoding", message: "Running Decoder..." });
    console.log("Starting decode...");
    const segments = decoder.decode(audioArrayU8);
    console.log("Decode complete, segments:", segments);

    // Send the segment back to the main thread as JSON
    self.postMessage({
      status: "complete",
      message: "complete",
      output: JSON.parse(segments),
    });
  } catch (e) {
    console.error("Worker error:", e);
    self.postMessage({ error: e.message || e.toString() });
  }
}

self.addEventListener("message", (event) => {
  handleMessage(event);
});
